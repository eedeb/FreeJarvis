"""The look: one palette, one frame, one set of small drawings.

Every see-through surface in this app -- the session log, the audio widget,
the cards Jarvis puts on the screen -- has to read as the same projector.
That is easy to say and easy to lose: three modules each with their own
tuple of amber-ish numbers drift apart the first time one of them is adjusted
in isolation, and the result stops looking designed and starts looking
assembled.

So the colours and the chrome live here and nowhere else.  What each surface
puts *inside* the frame is its own business; the frame, the glass, the
scanlines and the corner brackets are not.

The palette reads blue-dominant because everything downstream is OpenCV and
these are BGR triples.  It is projected cyan light, hot and near-white at its
brightest -- the same light the orb and the gauntlet hologram are drawn in.
"""

from __future__ import annotations

import cv2
import numpy as np

WARM = (246, 170, 44)            # the body of the light
HOT = (255, 240, 198)            # its hottest parts, and titles
FAINT = (158, 124, 92)           # labels, rules, anything secondary
COOL = (210, 150, 60)            # values, one step down from HOT
ALERT = (120, 120, 255)          # something wrong -- red, in BGR
GOOD = (130, 220, 150)           # something fine -- green
GLASS = (20, 14, 9)              # the panel's own tint, before its alpha

# How solid a surface is. The glass is see-through on purpose -- the whole
# point is to look projected onto the desk -- but the frame and the text are
# not: something you have to squint past your wallpaper to read has failed at
# the only job it has. 182 is where a pale wallpaper stops swallowing the dim
# lines; much below and the faint text goes illegible over bright ground,
# much above and it stops looking projected and starts looking like a box.
GLASS_ALPHA, EDGE_ALPHA = 182, 240

# The corner brackets: how long each arm is, and how thick.
BRACKET, BRACKET_W = 14, 2


def chrome(width: int, height: int, head_h: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """The parts of a panel that never change: (colour, alpha).

    Drawn once by whoever owns the panel and copied per frame, because none of
    it animates and blurring or re-stroking it every frame costs more than
    everything drawn on top of it put together.
    """
    base = np.empty((height, width, 3), np.uint8)
    base[:] = GLASS
    # Scanlines: every third row a shade darker. The cue that says "projected"
    # more than any amount of transparency does.
    base[::3] = (base[::3].astype(np.uint16) * 7 // 10).astype(np.uint8)

    alpha = np.full((height, width), GLASS_ALPHA, np.uint8)
    right, bottom = width - 1, height - 1
    cv2.rectangle(base, (0, 0), (right, bottom),
                  tuple(int(c * 0.82) for c in WARM), 1)
    cv2.rectangle(alpha, (0, 0), (right, bottom), EDGE_ALPHA, 1)

    if head_h:
        rule = tuple(int(c * 0.5) for c in WARM)
        cv2.line(base, (1, head_h), (right - 1, head_h), rule, 1)
        cv2.line(alpha, (1, head_h), (right - 1, head_h), EDGE_ALPHA, 1)

    for cx, cy, dx, dy in ((0, 0, 1, 1), (right, 0, -1, 1),
                           (0, bottom, 1, -1), (right, bottom, -1, -1)):
        for canvas, value in ((base, HOT), (alpha, 255)):
            cv2.line(canvas, (cx, cy), (cx + dx * BRACKET, cy), value, BRACKET_W)
            cv2.line(canvas, (cx, cy), (cx, cy + dy * BRACKET), value, BRACKET_W)
    return base, alpha


def level_colour(fraction: float, inverted: bool = False) -> tuple[int, int, int]:
    """Green, amber or red for a 0..1 reading.

    `inverted` for readings where *low* is the bad news -- free disk space,
    battery charge -- so one function covers both without every caller
    remembering to flip its own number first.
    """
    value = 1.0 - fraction if inverted else fraction
    if value >= 0.9:
        return ALERT
    if value >= 0.7:
        return WARM
    return GOOD


def bar(colour: np.ndarray, alpha: np.ndarray, box: tuple[int, int, int, int],
        fraction: float, tint=None) -> None:
    """A horizontal meter: a dim track with a bright part filled in."""
    x0, y0, x1, y1 = box
    fraction = float(min(max(fraction, 0.0), 1.0))
    cv2.rectangle(colour, (x0, y0), (x1, y1), (38, 30, 22), -1)
    cv2.rectangle(alpha, (x0, y0), (x1, y1), GLASS_ALPHA, -1)
    filled = x0 + int((x1 - x0) * fraction)
    if filled > x0:
        cv2.rectangle(colour, (x0, y0), (filled, y1), tint or WARM, -1)
        cv2.rectangle(alpha, (x0, y0), (filled, y1), 255, -1)
    # A tick at the end of the fill, so a nearly-empty bar is still visibly a
    # bar rather than an empty rectangle someone forgot to draw.
    cv2.line(colour, (filled, y0), (filled, y1), HOT, 1)
    cv2.line(alpha, (filled, y0), (filled, y1), 255, 1)


def sparkline(colour: np.ndarray, alpha: np.ndarray,
              box: tuple[int, int, int, int], points, top: float | None = None,
              tint=None) -> None:
    """A filled area graph of `points` across `box`, oldest at the left.

    Scaled to `top` when given and to the series' own peak otherwise. The
    difference matters: a CPU graph should be read against 100%, so a quiet
    machine looks quiet, while a network graph has no natural ceiling and
    would be a flat line at the bottom of the panel for ever if it had one.
    """
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    if width < 4 or height < 4:
        return
    data = [float(p) for p in points][-width:]
    if len(data) < 2:
        return
    ceiling = top if top else max(max(data), 1e-6)
    ceiling = max(ceiling, 1e-6)

    # One column per pixel, so the graph never interpolates between samples it
    # does not have -- a graph that draws more detail than it measured is a
    # lie told with a spline.
    step = width / len(data)
    curve = [(int(x0 + i * step),
              int(y1 - min(v / ceiling, 1.0) * (height - 1)))
             for i, v in enumerate(data)]
    shape = np.array([(x0, y1)] + curve + [(curve[-1][0], y1)], np.int32)
    shade = tuple(int(c * 0.34) for c in (tint or WARM))
    cv2.fillPoly(colour, [shape], shade)
    cv2.fillPoly(alpha, [shape], 225)
    cv2.polylines(colour, [np.array(curve, np.int32)], False, tint or WARM, 1,
                  cv2.LINE_AA)
    cv2.polylines(alpha, [np.array(curve, np.int32)], False, 255, 1)
