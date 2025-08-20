"""
Personal Calendar → TAF API (FastAPI)

Turns the *Personal WEEK TAF — Smart Classifier Widget* into a Python web API.
It pulls events from:
  • One or more ICS calendar feeds (public or private tokenized URLs)  OR
  • A JSON payload you POST with events
Then classifies each event with offline regex rules (fast) and optional
HuggingFace zero-shot fallback, and returns a TAF-like plaintext plus JSON.

Quick start
-----------
1) Create a venv and install deps:

   pip install fastapi uvicorn httpx icalendar recurring-ical-events python-dateutil pydantic-settings

2) Save this file as main.py and run:

   uvicorn main:app --reload --port 8080

3) Open interactive docs:

   http://127.0.0.1:8080/docs

Environment (optional)
----------------------
HUGGINGFACE_TOKEN  = "hf_xxx"  # enables cloud fallback
DEFAULT_STATION_ID = "CMDY"     # station in the header (e.g., your callsign or initials)

Endpoints
---------
GET  /health
GET  /weektaf                 -> Build from ICS URL(s) (query: cal_url, days, tz, etc.)
POST /weektaf                 -> Build from JSON events you provide
POST /classify                -> Classify a single event title

Notes
-----
• Recurrence: expanded via "recurring_ical_events" within the requested window.
• Timezones: internally UTC; display uses the tz you pass (default America/New_York).
• Output lines look like: "WRK F 1300-2100 03717" or "MEET T 0900-0930 Project Sync @HQ".
• If no events are found: lines = ["NOSIG"].
"""
from __future__ import annotations

import os
import re
import math
import json
import typing as t
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dateutil import tz as dateutil_tz
from dateutil.parser import parse as dtparse
from icalendar import Calendar
import recurring_ical_events

# -----------------------------
# Config & constants
# -----------------------------

DOW = ["Su", "M", "T", "W", "Th", "F", "Sa"]

class Settings(BaseSettings):
    HUGGINGFACE_TOKEN: str | None = None
    DEFAULT_STATION_ID: str = os.getenv("DEFAULT_STATION_ID", "CMDY")
    DEFAULT_CAL_URL: str | None = os.getenv("DEFAULT_CAL_URL")
    class Config:
        env_file = ".env"

settings = Settings()

# ---- Style flags to mimic widget behavior ----
USE_24H_DEFAULT = True
SHOW_LOC_DEFAULT = False

# ---- Rule set (regex offline classifier) ----
@dataclass
class Rule:
    code: str
    rx: re.Pattern
    tag: str | None = None
    tag_from: t.Callable[[str], str] | None = None

RULES: list[Rule] = [
    Rule("WRK", re.compile(r"(CFA\s+Thomson\s+FSU\s+\(03717\)|\bshift\b|\bwork\b|\bclock in\b)", re.I), tag="03717"),
    Rule("VG",  re.compile(r"^(VG\s+.+)|\bvolleyball\b|\bscrimmage\b|\btournament\b", re.I), tag_from=lambda t: re.sub(r"^VG\s+", "", t, flags=re.I).strip()),
    Rule("FLT", re.compile(r"\bFLT\b|flight lesson|discovery flight|CFI|pattern|sim", re.I)),
    Rule("GRP", re.compile(r"\bgroup\b|hang( out)?|friends|youth( group)?", re.I)),

    Rule("DCTR", re.compile(r"\bdoctor\b|\bdr\.?\b|primary care|pediatrician|urgent care|clinic|physical|check-?up|checkup", re.I)),
    Rule("DENT", re.compile(r"dentist|orthodontist|ortho|braces|cleaning|whitening", re.I)),
    Rule("SCHL", re.compile(r"\bclass\b|lecture|lab\b|midterm|final|exam|quiz|homework|assignment", re.I)),
    Rule("MEET", re.compile(r"meeting|sync|standup|1[: ]?1|interview|call|zoom|teams|webex|google meet", re.I)),
    Rule("GYM",  re.compile(r"\bgym\b|workout|lift|run\b|cardio|crossfit|practice(?!.*volleyball)", re.I)),
    Rule("CHUR", re.compile(r"church|service|bible|small group|worship", re.I)),
    Rule("TRVL", re.compile(r"trip|travel|hotel|drive to|roadtrip|airport\b|\bflight\b", re.I)),
    Rule("SHOP", re.compile(r"shopping|grocer(y|ies)|target|costco|walmart|pickup order|store run", re.I)),
    Rule("CAR",  re.compile(r"oil change|tire|alignment|inspection|dmv|car wash|detail", re.I)),
    Rule("HOME", re.compile(r"clean|laundry|dishes|vacuum|yard|mow|errand(s)?", re.I)),
    Rule("APPT", re.compile(r"appointment|appt|haircut|barber|salon|eye exam|optometrist", re.I)),
    Rule("BANK", re.compile(r"bank|wells fargo|deposit|withdrawal|transfer|notary", re.I)),
    Rule("BDAY", re.compile(r"birthday|b[-\s]?day", re.I)),
]

HF_LABELS = {
    "work shift / job / restaurant": "WRK",
    "volleyball / sports match / scrimmage": "VG",
    "flight training / flying lesson": "FLT",
    "group hangout / social gathering": "GRP",
    "doctor / medical appointment": "DCTR",
    "dentist / orthodontist": "DENT",
    "school class / lecture / exam": "SCHL",
    "meeting / interview / call": "MEET",
    "gym / workout / practice": "GYM",
    "church / service / bible study": "CHUR",
    "travel / trip / flight": "TRVL",
    "shopping / groceries / store run": "SHOP",
    "car service / dmv / maintenance": "CAR",
    "home chores / errands": "HOME",
    "appointment / haircut / salon": "APPT",
    "bank / finance errand": "BANK",
    "birthday or celebration": "BDAY",
    "other / miscellaneous": "OTH",
}

# -----------------------------
# Models
# -----------------------------

class EventIn(BaseModel):
    title: str
    start: datetime
    end: datetime
    location: str | None = None

class BuildParams(BaseModel):
    station_id: str = Field(default_factory=lambda: settings.DEFAULT_STATION_ID)
    days_ahead: int = Field(7, ge=1, le=30)
    tz: str = Field("America/New_York")
    use_24h: bool = Field(USE_24H_DEFAULT)
    show_loc: bool = Field(SHOW_LOC_DEFAULT)

class WeekTafOut(BaseModel):
    header: str
    lines: list[str]
    count: int
    start: datetime
    end: datetime
    plaintext: str

class ClassifyOut(BaseModel):
    code: str
    tag: str | None = None
    used_hf: bool = False

# -----------------------------
# Helpers
# -----------------------------

def yyyymmdd(d: datetime) -> str:
    return d.strftime("%Y%m%d")

def fmt_time(d: datetime, tz_name: str, use_24h: bool) -> str:
    try:
        tz = dateutil_tz.gettz(tz_name) or timezone.utc
    except Exception:
        tz = timezone.utc
    d_local = d.astimezone(tz)
    h = d_local.hour
    m = d_local.minute
    if use_24h:
        return f"{h:02d}{m:02d}"
    ampm = "p" if h >= 12 else "a"
    h12 = h % 12
    if h12 == 0:
        h12 = 12
    return f"{h12}{':' + str(m).zfill(2) if m else ''}{ampm}"

def taf_header(station_id: str, start: datetime, end: datetime) -> str:
    now = datetime.now(timezone.utc)
    iss = now.strftime("%H%MZ")
    span = f"{yyyymmdd(start)[2:]}/{yyyymmdd(end)[2:]}"
    return f"WEEKTAF {station_id} {iss} {span}"

# Classification: rules first, optional HF fallback
async def classify_title(title: str) -> ClassifyOut:
    tclean = (title or "").strip()
    # Rules pass
    for r in RULES:
        if r.rx.search(tclean):
            tag = r.tag_from(tclean) if r.tag_from else (r.tag or "")
            return ClassifyOut(code=r.code, tag=(tag or None), used_hf=False)
    # HF fallback
    token = settings.HUGGINGFACE_TOKEN
    if not token:
        return ClassifyOut(code="OTH", tag=tclean or None, used_hf=False)
    try:
        labels = list(HF_LABELS.keys())
        body = {"inputs": tclean, "parameters": {"candidate_labels": labels, "multi_label": False}}
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                "https://api-inference.huggingface.co/models/facebook/bart-large-mnli",
                headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
                json=body,
            )
            r.raise_for_status()
            data = r.json()
        best = (data.get("labels") or [None])[0]
        mapped = HF_LABELS.get(best, "OTH") if best else "OTH"
        return ClassifyOut(code=mapped, tag=tclean or None, used_hf=True)
    except Exception:
        return ClassifyOut(code="OTH", tag=tclean or None, used_hf=False)

# Build a single line for an event
async def line_for_event(ev: EventIn, cls: ClassifyOut, tz: str, use_24h: bool, show_loc: bool) -> str:
    local_tz = dateutil_tz.gettz(tz) or timezone.utc
    dow = DOW[ ev.start.astimezone(local_tz).weekday() ]
    span = f"{fmt_time(ev.start, tz, use_24h)}-{fmt_time(ev.end, tz, use_24h)}"
    bits: list[str] = []
    if cls.code == "WRK" and (cls.tag or "").strip():
        bits.append(cls.tag.strip())
    elif cls.code == "VG" and (cls.tag or "").strip():
        id_ = re.sub(r"^VG\s+", "", cls.tag or "", flags=re.I).strip()
        if id_:
            bits.append(f"ID:{id_}")
    else:
        # for other codes, include original title as context
        if (cls.tag or "").strip():
            bits.append((cls.tag or "").strip())
    if show_loc and ev.location:
        bits.append(f"@{ev.location}")
    note = (" " + " ".join(bits)) if bits else ""
    return f"{cls.code} {dow} {span}{note}"

# -----------------------------
# ICS ingestion
# -----------------------------

async def fetch_ics(url: str) -> Calendar:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url)
            r.raise_for_status()
            return Calendar.from_ical(r.content)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Unable to fetch ICS: {e}")

async def events_from_ics(urls: list[str], start: datetime, end: datetime) -> list[EventIn]:
    out: list[EventIn] = []
    for u in urls:
        cal = await fetch_ics(u)
        # Recurrence expansion
        # recurring_ical_events expects a naive datetime in local tz; we use UTC and then localize back
        # To be safe, convert to naive UTC for the lib, then back to aware UTC.
        start_naive = datetime.utcfromtimestamp(start.timestamp())
        end_naive = datetime.utcfromtimestamp(end.timestamp())
        try:
            vevents = recurring_ical_events.of(cal).between(start_naive, end_naive)
        except Exception:
            # Fallback: simple VEVENT scan
            vevents = [comp for comp in cal.walk() if comp.name == 'VEVENT']
        for ev in vevents:
            try:
                dtstart = ev.decoded('DTSTART')
                dtend = ev.decoded('DTEND') if 'DTEND' in ev else (dtstart + timedelta(minutes=int(ev.get('DURATION').totalseconds()/60)) if 'DURATION' in ev else dtstart + timedelta(hours=1))
                if dtstart.tzinfo is None:
                    dtstart = dtstart.replace(tzinfo=timezone.utc)
                if dtend.tzinfo is None:
                    dtend = dtend.replace(tzinfo=timezone.utc)
                title = str(ev.get('SUMMARY') or '')
                loc = str(ev.get('LOCATION') or '') or None
                # filter to window (inclusive start, exclusive end)
                if dtend <= start or dtstart >= end:
                    continue
                out.append(EventIn(title=title, start=dtstart.astimezone(timezone.utc), end=dtend.astimezone(timezone.utc), location=loc))
            except Exception:
                continue
    # sort
    out.sort(key=lambda e: (e.start, e.end))
    return out

# -----------------------------
# Builder
# -----------------------------

async def build_report(events: list[EventIn], params: BuildParams) -> WeekTafOut:
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=params.days_ahead)
    header = taf_header(params.station_id, start, end)
    if not events:
        lines = ["NOSIG"]
        plaintext = "\n" + "\n".join([header] + lines) + "\n"
        return WeekTafOut(header=header, lines=lines, count=1, start=start, end=end, plaintext=plaintext)

    lines: list[str] = []
    for ev in events:
        cls = await classify_title(ev.title)
        line = await line_for_event(ev, cls, params.tz, params.use_24h, params.show_loc)
        lines.append(line)

    plaintext = "\n" + "\n".join([header] + lines) + "\n"
    return WeekTafOut(header=header, lines=lines, count=len(lines), start=start, end=end, plaintext=plaintext)

# -----------------------------
# FastAPI app
# -----------------------------

app = FastAPI(title="Personal Calendar TAF API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/classify", response_model=ClassifyOut)
async def classify(title: str = Query(..., description="Event title to classify")):
    return await classify_title(title)

@app.get("/weektaf", response_model=WeekTafOut)
async def weektaf_from_ics(
    cal_url: str | None = Query(None, description="Comma-separated ICS feed URLs"),
    station_id: str = Query(settings.DEFAULT_STATION_ID),
    days: int = Query(7, ge=1, le=30),
    tz: str = Query("America/New_York"),
    use_24h: bool = Query(USE_24H_DEFAULT),
    show_loc: bool = Query(SHOW_LOC_DEFAULT),
):
    # If user didn't pass ?cal_url=, fall back to DEFAULT_CAL_URL from .env
    if not cal_url:
        if settings.DEFAULT_CAL_URL:
            cal_url = settings.DEFAULT_CAL_URL
        else:
            raise HTTPException(status_code=400, detail="Provide ?cal_url=... or set DEFAULT_CAL_URL in .env")

    urls = [u.strip() for u in cal_url.split(",") if u.strip()]
    start = datetime.now(timezone.utc)
    end = start + timedelta(days=days)
    events = await events_from_ics(urls, start, end)
    params = BuildParams(station_id=station_id, days_ahead=days, tz=tz, use_24h=use_24h, show_loc=show_loc)
    return await build_report(events, params)


class WeekTafIn(BaseModel):
    events: list[EventIn] = Field(default_factory=list)
    params: BuildParams = Field(default_factory=BuildParams)

@app.post("/weektaf", response_model=WeekTafOut)
async def weektaf_from_json(body: WeekTafIn):
    return await build_report(body.events, body.params)

@app.get("/")
async def root():
    # If DEFAULT_CAL_URL is configured, return the current TAF immediately for convenience
    if settings.DEFAULT_CAL_URL:
        urls = [u.strip() for u in settings.DEFAULT_CAL_URL.split(",") if u.strip()]
        start = datetime.now(timezone.utc)
        end = start + timedelta(days=7)
        events = await events_from_ics(urls, start, end)
        params = BuildParams()
        out = await build_report(events, params)
        return {"hint": "Pass ?cal_url= to override or POST /weektaf.", **out.model_dump()}
    return {"message": "See /docs for the Personal Calendar TAF API. Set DEFAULT_CAL_URL in .env or pass ?cal_url=..."}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
