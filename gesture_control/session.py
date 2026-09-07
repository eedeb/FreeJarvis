"""Recording what actually happened, so clicking can be debugged from evidence.

Every guess about why a click misfires is a guess about a signal nobody has
looked at.  This keeps the last few minutes of that signal -- the gap between
the fingertips, what the filter made of it, what the click state machine did
about it -- and writes it out on exit, along with a summary that names the
failure modes rather than leaving them to be inferred.
"""

from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class Frame:
    t: float
    hand: bool
    index_raw: float      # thumb-to-index gap as measured, cm
    index_used: float     # ...after filtering, which is what the threshold sees
    middle_raw: float
    mode: str
    down: bool            # left button held
    dragging: bool
    x: float
    y: float


class SessionLog:
    """A ring buffer of recent frames, plus the analysis of them."""

    def __init__(self, seconds: float = 180.0, fps: float = 30.0):
        self.frames: deque[Frame] = deque(maxlen=int(seconds * fps))
        self.clicks = 0
        self.right_clicks = 0
        self.drags = 0

    def add(self, frame: Frame) -> None:
        self.frames.append(frame)

    # -- writing out -------------------------------------------------------

    def save(self, path) -> int:
        rows = list(self.frames)
        if not rows:
            return 0
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["t", "hand", "index_raw_cm", "index_used_cm",
                             "middle_raw_cm", "mode", "down", "dragging", "x", "y"])
            start = rows[0].t
            for f in rows:
                writer.writerow([f"{f.t - start:.3f}", int(f.hand),
                                 f"{f.index_raw:.2f}", f"{f.index_used:.2f}",
                                 f"{f.middle_raw:.2f}", f.mode, int(f.down),
                                 int(f.dragging), f"{f.x:.0f}", f"{f.y:.0f}"])
        return len(rows)

    # -- analysis ----------------------------------------------------------

    def summary(self, settings) -> list[str]:
        """Plain statements about what went on, and what looks wrong."""
        rows = [f for f in self.frames]
        if len(rows) < 30:
            return ["Not enough was recorded to say anything useful."]

        out: list[str] = []
        span = rows[-1].t - rows[0].t
        seen = sum(1 for f in rows if f.hand)
        out.append(f"{span:.0f}s of use, hand visible in {seen / len(rows):.0%} "
                   f"of {len(rows)} frames")
        out.append(f"{self.clicks} left clicks, {self.right_clicks} right clicks, "
                   f"{self.drags} drags")

        gaps = np.array([f.index_raw for f in rows if f.hand and f.index_raw > 0])
        if gaps.size > 20:
            out.append(f"thumb-to-index gap: median {np.median(gaps):.1f}cm, "
                       f"10th pct {np.percentile(gaps, 10):.1f}cm, "
                       f"90th {np.percentile(gaps, 90):.1f}cm "
                       f"(clicks under {settings.pinch_on_cm}cm)")

        # Press and release runs, which is where the trouble usually shows.
        presses, run = [], 0
        for f in rows:
            if f.down:
                run += 1
            elif run:
                presses.append(run)
                run = 0
        if run:
            presses.append(run)
        if presses:
            held = np.array(presses) / 30.0 * 1000.0
            out.append(f"button held for: shortest {held.min():.0f}ms, "
                       f"median {np.median(held):.0f}ms, longest {held.max():.0f}ms")
            brief = int((held < 120).sum())
            if brief:
                out.append(f"  ! {brief} press(es) under 120ms -- too short to be "
                           f"meant, so something is firing on its own")

        # Clicks bunched together are the signature of one pinch stuttering.
        downs = [rows[i].t for i in range(1, len(rows))
                 if rows[i].down and not rows[i - 1].down]
        if len(downs) > 1:
            gaps_ms = np.diff(downs) * 1000.0
            bunched = int((gaps_ms < 250).sum())
            out.append(f"gap between clicks: shortest {gaps_ms.min():.0f}ms, "
                       f"median {np.median(gaps_ms):.0f}ms")
            if bunched:
                out.append(f"  ! {bunched} click(s) within 250ms of the one before "
                           f"-- either double clicks, or one pinch counted twice")

        # Drags that nobody asked for.
        drag_runs = []
        run = 0
        for f in rows:
            if f.dragging:
                run += 1
            elif run:
                drag_runs.append(run)
                run = 0
        if run:
            drag_runs.append(run)
        if drag_runs:
            short = sum(1 for r in drag_runs if r < 6)
            out.append(f"{len(drag_runs)} drag(s), shortest "
                       f"{min(drag_runs) / 30 * 1000:.0f}ms")
            if short:
                out.append(f"  ! {short} drag(s) lasted under 200ms -- probably "
                           f"clicks that turned into drags by accident")

        # How jumpy the measurement is *while a click is being held*, which is
        # what decides whether the filter is wide enough. Selecting frames by
        # value instead would throw away the spikes this is looking for.
        held_frames = [f for f in rows if f.hand and f.down]
        if len(held_frames) > 10:
            vals = np.array([f.index_raw for f in held_frames])
            jumps = np.abs(np.diff(vals))
            out.append(f"while pinched the gap jumps {np.median(jumps):.1f}cm per "
                       f"frame typically, worst {jumps.max():.1f}cm")
            if jumps.max() > settings.pinch_off_cm:
                out.append(f"  ! a single frame moved further than the whole "
                           f"threshold band -- the filter window needs widening")
        return out
