"""Direct mapping: the camera frame is the screen, with nothing fitted."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from dataclasses import replace

import numpy as np

from gesture_control import landmarks as lm
from gesture_control.config import Settings
from gesture_control.controller import GestureController
from gesture_control.mapping import AimSample, DirectMap
from synthetic_hand import OPEN_PALM, FakeTracker, make_hand

ok = True
SCREEN = (1920, 1080)


def check(label, cond):
    global ok
    ok = ok and bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")


def at(m, u, v):
    return m.predict(AimSample(tip=(u, v), direction=(0.0, 0.0), scale=0.09))


print("The frame maps onto the screen:")
m = DirectMap(SCREEN, mirror=True)
cx, cy = at(m, 0.5, 0.5)
check(f"the middle of the frame is the middle of the screen ({cx:.0f}, {cy:.0f})",
      abs(cx - SCREEN[0] / 2) < 2 and abs(cy - SCREEN[1] / 2) < 2)

# A quarter of the way across the frame is a quarter of the way across the
# screen -- that is the whole idea.
for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
    x, _ = at(m, 1.0 - frac, 0.5)      # 1-frac because the view is mirrored
    want = frac * (SCREEN[0] - 1)
    ok = ok and abs(x - want) < 2
print("  [PASS] every fraction across the frame is the same fraction across the screen"
      if ok else "  [FAIL] proportions do not line up")

print("\nMirroring:")
# The camera faces you, so your right hand appears on the left of the raw
# image. Moving right must still move the cursor right.
left_of_image, _ = at(m, 0.15, 0.5)
right_of_image, _ = at(m, 0.85, 0.5)
check(f"your hand's right maps to the screen's right "
      f"({left_of_image:.0f} > {right_of_image:.0f})",
      left_of_image > right_of_image)
un = DirectMap(SCREEN, mirror=False)
check("and mirroring can be turned off", at(un, 0.15, 0.5)[0] < at(un, 0.85, 0.5)[0])

print("\nVertical is not mirrored:")
check("a hand high in the frame puts the cursor high on the screen",
      at(m, 0.5, 0.2)[1] < at(m, 0.5, 0.8)[1])

print("\nCorners reach the corners:")
tl = at(m, 1.0, 0.0)
br = at(m, 0.0, 1.0)
check(f"top-left ({tl[0]:.0f}, {tl[1]:.0f}) and bottom-right "
      f"({br[0]:.0f}, {br[1]:.0f})",
      tl[0] < 2 and tl[1] < 2
      and br[0] > SCREEN[0] - 3 and br[1] > SCREEN[1] - 3)

print("\nOverscan: the outer tenth of the frame is already off the screen.")
over = DirectMap(SCREEN, mirror=True, margin=0.10)
top = at(over, 0.5, 0.10)[1]
bottom = at(over, 0.5, 0.90)[1]
print(f"      10% down the frame -> y {top:.0f};  90% down -> y {bottom:.0f}")
check("a hand a tenth from the top puts the cursor at the very top", top < 2)
check("a hand a tenth from the bottom puts it at the very bottom",
      bottom > SCREEN[1] - 3)
check("anything further out just stays pinned there",
      at(over, 0.5, 0.02)[1] < 2 and at(over, 0.5, 0.98)[1] > SCREEN[1] - 3)
check("the corners are reachable from a tenth in",
      at(over, 0.9, 0.1) == (0.0, 0.0)
      and at(over, 0.1, 0.9) == (SCREEN[0] - 1.0, SCREEN[1] - 1.0))
# The middle still behaves: half way is half way.
mid = at(over, 0.5, 0.5)
check(f"the centre is unmoved ({mid[0]:.0f}, {mid[1]:.0f})",
      abs(mid[0] - SCREEN[0] / 2) < 2 and abs(mid[1] - SCREEN[1] / 2) < 2)

print("\n      what it costs: the middle 80% is stretched over 100%,")
span_no, span_yes = [], []
for u in (0.4, 0.6):
    span_no.append(at(m, u, 0.5)[0])
    span_yes.append(at(over, u, 0.5)[0])
gain = abs(span_yes[1] - span_yes[0]) / abs(span_no[1] - span_no[0])
print(f"      so hand movement and jitter are both magnified {gain:.2f}x")
check("the magnification is 1/(1 - 2*margin), i.e. 1.25", abs(gain - 1.25) < 0.02)

print("\nOverscan is on by default:")
# The exact figure is a matter of taste -- how far you want to reach against
# how much magnification you will accept -- so it is left free to tune. What
# is worth asserting is that it is switched on and stays sane.
configured = Settings().direct_margin
print(f"      direct_margin = {configured} "
      f"({1 / (1 - 2 * configured):.2f}x magnification)")
check("it is enabled", configured > 0.0)
check("and within a sensible range", 0.0 < configured <= 0.30)

print("\nWhat this buys: no magnification of tracking noise.")
# Every fitted mapping stretches part of the frame over the whole screen and
# stretches the jitter with it. Here the factor is one for matched resolutions.
rng = np.random.default_rng(5)
jitter_px = 3.0
frame_w = 1920
noisy = rng.normal(0.5, jitter_px / frame_w, 4000)
mapped = np.array([at(m, u, 0.5)[0] for u in noisy])
print(f"      {jitter_px:.0f}px of landmark jitter on a {frame_w}px camera "
      f"-> {mapped.std():.1f}px of cursor jitter")
check("jitter passes through at roughly 1:1", 0.8 < mapped.std() / jitter_px < 1.25)
# For contrast: a calibrated map whose hand only swept a quarter of the frame.
print(f"      a fitted map over a 25% sweep would have made that "
      f"{jitter_px * 4:.0f}px")

print("\nDriving the real controller, with no calibration anywhere:")
settings = replace(Settings(), aim_mode="direct")
model = DirectMap(SCREEN, settings.mirror, settings.direct_margin)


class FakeMouse:
    def __init__(self): self.events = []; self._left = False
    def move_to(self, x, y): self.events.append(("move", x, y))
    def left_down(self):
        if not self._left: self._left = True; self.events.append(("left_down",))
    def left_up(self):
        if self._left: self._left = False; self.events.append(("left_up",))
    @property
    def left_is_down(self): return self._left
    def right_click(self): self.events.append(("right_click",))
    def scroll(self, n): self.events.append(("scroll", n))
    def release_all(self): self.left_up()


def controller():
    c = GestureController(FakeTracker(), model, settings)
    c.mouse = FakeMouse()
    return c


def hand_at(u, v, pose=OPEN_PALM, stamp=0.0):
    """A hand whose palm centre sits at (u, v) of the camera frame."""
    h = make_hand((960, 540), pose, anchor="palm", noise=0.0, stamp=stamp)
    shift = np.array([u, v]) - lm.palm_center(h.norm)
    h.norm[:] = h.norm + shift
    h.metric[:] = h.norm * np.array([16 / 9, 1.0])
    return h


def expect(u, v):
    """Where the settings say a hand at (u, v) of the frame should put it."""
    return model.predict(AimSample(tip=(u, v), direction=(0.0, 0.0), scale=0.09))


c = controller()
t = 0.0
for u, v in [(0.5, 0.5)] * 25 + [(0.2, 0.3)] * 25:
    t += 1 / 30
    c._update(hand_at(u, v, stamp=t))
last = [e for e in c.mouse.events if e[0] == "move"][-1]
want = expect(0.2, 0.3)
check(f"the cursor follows the hand to ({last[1]:.0f}, {last[2]:.0f}), "
      f"expecting ({want[0]:.0f}, {want[1]:.0f})",
      abs(last[1] - want[0]) < 40 and abs(last[2] - want[1]) < 40)

print("\nPinching thumb and index clicks where the hand was:")


def pinched_at(u, v, stamp=0.0):
    h = hand_at(u, v, OPEN_PALM, stamp)
    # Thumb tip brought onto the index tip, in both the image and in 3D.
    thumb = [lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]
    h.metric[thumb] += (h.metric[lm.INDEX_TIP] + np.array([0.004, 0.010])
                        - h.metric[lm.THUMB_TIP])
    h.norm[thumb] = h.metric[thumb] / np.array([16 / 9, 1.0])
    h.world[thumb] += (h.world[lm.INDEX_TIP] + np.array([0.004, 0.008, 0.002])
                       - h.world[lm.THUMB_TIP])
    return h


open_ratio = hand_at(0.35, 0.6).pinch(lm.INDEX_TIP)
shut_ratio = pinched_at(0.35, 0.6).pinch(lm.INDEX_TIP)
print(f"      open hand reads {open_ratio:.2f}, pinched reads {shut_ratio:.2f} "
      f"(clicks below {settings.pinch_on})")
check("the two states sit either side of the threshold",
      shut_ratio < settings.pinch_on < open_ratio)

c = controller()
t = 0.0
for maker, n in [(hand_at, 22), (pinched_at, 5), (hand_at, 8)]:
    for _ in range(n):
        t += 1 / 30
        c._update(maker(0.35, 0.6, stamp=t))
names = [e[0] for e in c.mouse.events]
check("exactly one click", names.count("left_down") == 1 and names.count("left_up") == 1)
down = c.mouse.events[names.index("left_down") - 1]
want = expect(0.35, 0.6)
check(f"and it lands at ({down[1]:.0f}, {down[2]:.0f}), "
      f"expecting ({want[0]:.0f}, {want[1]:.0f})",
      abs(down[1] - want[0]) < 50 and abs(down[2] - want[1]) < 50)
check("nothing is left held down", not c.mouse.left_is_down)

print("\nPinching does not drag the aim, because the palm does not move:")
before = lm.palm_center(hand_at(0.35, 0.6).norm)
after = lm.palm_center(pinched_at(0.35, 0.6).norm)
shift = float(np.linalg.norm(after - before)) * SCREEN[0]
print(f"      the palm centre moves {shift:.1f}px on screen as the pinch closes")
check("the aim point is unmoved by the click", shift < 3.0)

print("\nHolding the pinch drags:")
c = controller()
t = 0.0
for maker, u, v, n in [(hand_at, 0.4, 0.5, 14), (pinched_at, 0.4, 0.5, 18),
                       (pinched_at, 0.7, 0.8, 22), (hand_at, 0.7, 0.8, 14)]:
    for _ in range(n):
        t += 1 / 30
        c._update(maker(u, v, stamp=t))
names = [e[0] for e in c.mouse.events]
moves = [e for e in c.mouse.events[names.index("left_down"):names.index("left_up")]
         if e[0] == "move"]
check(f"the cursor follows while pinched ({len(moves)} moves)", len(moves) > 8)
check("and releases when the fingers part", names.count("left_up") == 1)

print("\nA slow open-pinch-open cycle must give exactly one click:")
c = controller()
t = 0.0
for cycle in range(3):
    for frac in list(np.linspace(0, 1, 12)) + list(np.linspace(1, 0, 12)):
        t += 1 / 30
        h = hand_at(0.5, 0.5, stamp=t)
        shut = pinched_at(0.5, 0.5, stamp=t)
        h.metric[lm.THUMB_TIP] = (h.metric[lm.THUMB_TIP] * (1 - frac)
                                  + shut.metric[lm.THUMB_TIP] * frac)
        h.norm[lm.THUMB_TIP] = h.metric[lm.THUMB_TIP] / np.array([16 / 9, 1.0])
        h.world[lm.THUMB_TIP] = (h.world[lm.THUMB_TIP] * (1 - frac)
                                 + shut.world[lm.THUMB_TIP] * frac)
        c._update(h)
clicks = [e[0] for e in c.mouse.events].count("left_down")
print(f"      three slow cycles produced {clicks} clicks")
check("no extra clicks from the closing motion", clicks == 3)

# ------------------------------------------------- how wide a click actually is
print("\nHow far apart the fingertips may be and still click:")


def gap_at(cm):
    """A hand whose thumb sits `cm` centimetres from the index tip."""
    h = hand_at(0.5, 0.5)
    palm_m = lm.hand_scale(h.world)
    offset = cm / 100.0
    h.world[lm.THUMB_TIP] = h.world[lm.INDEX_TIP] + np.array([offset, 0.0, 0.0])
    # The image must agree, or the flat measure would contradict the 3D one.
    frac = offset / palm_m
    palm_img = np.linalg.norm(h.metric[lm.MIDDLE_MCP] - h.metric[lm.WRIST])
    h.metric[lm.THUMB_TIP] = h.metric[lm.INDEX_TIP] + np.array([frac * palm_img, 0.0])
    h.norm[lm.THUMB_TIP] = h.metric[lm.THUMB_TIP] / np.array([16 / 9, 1.0])
    return h


print(f"      threshold: click under {settings.pinch_on_cm}cm, "
      f"release over {settings.pinch_off_cm}cm")
for cm in (0.5, 1.0, 1.5, 2.0, 3.0, 5.0):
    reported = gap_at(cm).pinch_cm(lm.INDEX_TIP)
    verdict = "CLICK" if reported < settings.pinch_on_cm else "no click"
    print(f"      fingertips {cm:4.1f}cm apart -> measured {reported:4.1f}cm  {verdict}")
check("a 1cm gap clicks", gap_at(1.0).pinch_cm(lm.INDEX_TIP) < settings.pinch_on_cm)
# Deliberately generous. A tight threshold reads better than it works: on a
# real session 2.1cm missed 167 frames of a genuinely pinched hand, where 4cm
# missed 38. Fingers a few centimetres apart counting as a click is the price
# of a click registering at all.
check("a 3cm gap also clicks, by design",
      gap_at(3.0).pinch_cm(lm.INDEX_TIP) < settings.pinch_on_cm)
check("a 6cm gap does not",
      gap_at(6.0).pinch_cm(lm.INDEX_TIP) > settings.pinch_on_cm)
check("the measurement is accurate to a couple of millimetres",
      all(abs(gap_at(cm).pinch_cm(lm.INDEX_TIP) - cm) < 0.2
          for cm in (1.0, 2.0, 3.0, 5.0)))

# ------------------------------------------------------------- double clicking
print("\nDouble clicking:")
from gesture_control.mouse import double_click_slop, double_click_time_ms

slop_x, _ = double_click_slop()
print(f"      Windows allows {double_click_time_ms():.0f}ms and {slop_x}px "
      f"between the two clicks")


def two_clicks(assist, gap_frames=4, jitter=0.004):
    """Pinch twice in quick succession; return the two click positions."""
    c = controller()
    c.settings = replace(settings, double_click_assist=assist)
    c._double_click_ms = double_click_time_ms()
    rng = np.random.default_rng(7)
    t = 0.0
    positions = []
    for phase in ("hold", "click", "release", "click", "release"):
        n = {"hold": 20, "click": 3, "release": gap_frames}[phase]
        for _ in range(n):
            t += 1 / 30
            u = 0.5 + rng.normal(0, jitter)
            v = 0.5 + rng.normal(0, jitter)
            h = pinched_at(u, v, stamp=t) if phase == "click" else hand_at(u, v, stamp=t)
            c._update(h)
    names = [e[0] for e in c.mouse.events]
    for i, name in enumerate(names):
        if name == "left_down":
            positions.append(np.array(c.mouse.events[i - 1][1:]))
    return positions


without = two_clicks(assist=False)
with_assist = two_clicks(assist=True)
check("two clicks are produced either way",
      len(without) == 2 and len(with_assist) == 2)
if len(without) == 2 and len(with_assist) == 2:
    d_off = float(np.linalg.norm(without[1] - without[0]))
    d_on = float(np.linalg.norm(with_assist[1] - with_assist[0]))
    print(f"      without the assist the two clicks land {d_off:.1f}px apart"
          f"  -> Windows sees {'a double click' if d_off <= slop_x else 'two singles'}")
    print(f"      with it            they land {d_on:.1f}px apart"
          f"  -> Windows sees {'a double click' if d_on <= slop_x else 'two singles'}")
    check("the assist puts the second click exactly on the first", d_on == 0.0)
    check("which is what Windows needs", d_on <= slop_x)

# It must not drag a genuinely separate click back to the old spot.
c = controller()
c._double_click_ms = double_click_time_ms()
t = 0.0
for maker, u, v, n in [(hand_at, 0.4, 0.4, 20), (pinched_at, 0.4, 0.4, 4),
                       (hand_at, 0.4, 0.4, 6), (hand_at, 0.8, 0.8, 10),
                       (pinched_at, 0.8, 0.8, 4), (hand_at, 0.8, 0.8, 6)]:
    for _ in range(n):
        t += 1 / 30
        c._update(maker(u, v, stamp=t))
names = [e[0] for e in c.mouse.events]
spots = [np.array(c.mouse.events[i - 1][1:])
         for i, n_ in enumerate(names) if n_ == "left_down"]
check(f"a click somewhere else is left where it belongs "
      f"({float(np.linalg.norm(spots[1] - spots[0])):.0f}px away)",
      len(spots) == 2 and float(np.linalg.norm(spots[1] - spots[0])) > 200)

# A drag must not seed a double click either.
c = controller()
c._double_click_ms = double_click_time_ms()
t = 0.0
for maker, u, v, n in [(hand_at, 0.4, 0.5, 14), (pinched_at, 0.4, 0.5, 20),
                       (pinched_at, 0.7, 0.5, 20), (hand_at, 0.7, 0.5, 8),
                       (pinched_at, 0.7, 0.5, 4), (hand_at, 0.7, 0.5, 8)]:
    for _ in range(n):
        t += 1 / 30
        c._update(maker(u, v, stamp=t))
names = [e[0] for e in c.mouse.events]
spots = [np.array(c.mouse.events[i - 1][1:])
         for i, n_ in enumerate(names) if n_ == "left_down"]
check("a drag does not anchor the click that follows it",
      len(spots) == 2 and float(np.linalg.norm(spots[1] - spots[0])) > 100)

# ------------------------------------------------------------------- dragging
print("\nDragging in detail:")


def run_frames(c, steps):
    """steps: (maker, u, v, count). Returns the recorded mouse events."""
    t = [getattr(c, "_t", 0.0)]
    for maker, u, v, n in steps:
        for _ in range(n):
            t[0] += 1 / 30
            c._update(maker(u, v, stamp=t[0]))
    c._t = t[0]
    return c.mouse.events


# Holding still must never become a drag: that is how someone clicks
# carefully, and how the second click of a double click looks.
c = controller()
run_frames(c, [(hand_at, 0.5, 0.5, 20), (pinched_at, 0.5, 0.5, 40),
               (hand_at, 0.5, 0.5, 4)])
check(f"a pinch held still for {40 / 30:.1f}s stays a click, never a drag",
      not any(e[0] == "move"
              for e in c.mouse.events[
                  [x[0] for x in c.mouse.events].index("left_down"):
                  [x[0] for x in c.mouse.events].index("left_up")]))

# Moving while pinched starts a drag.
c = controller()
run_frames(c, [(hand_at, 0.45, 0.45, 18), (pinched_at, 0.45, 0.45, 4),
               (pinched_at, 0.62, 0.62, 20), (hand_at, 0.62, 0.62, 14)])
names = [e[0] for e in c.mouse.events]
held = c.mouse.events[names.index("left_down"):names.index("left_up")]
drag_moves = [e for e in held if e[0] == "move"]
check(f"moving while pinched drags ({len(drag_moves)} moves)", len(drag_moves) > 6)

# And the drag must not jerk when it begins.
if len(drag_moves) >= 2:
    press_at = np.array(c.mouse.events[names.index("left_down") - 1][1:])
    first = np.array(drag_moves[0][1:])
    steps = [float(np.linalg.norm(np.array(b[1:]) - np.array(a[1:])))
             for a, b in zip(drag_moves, drag_moves[1:])]
    jump = float(np.linalg.norm(first - press_at))
    print(f"      the drag opens {jump:.0f}px from the press point, "
          f"largest later step {max(steps):.0f}px")
    check("the drag continues from the cursor rather than snapping to the hand",
          jump < 40)
    check("and moves smoothly from there", max(steps) < 60)

print("\n      A drag takes a firmer opening to end than a click does:")
print(f"      click releases over {settings.pinch_off_cm}cm, "
      f"drag over {settings.drag_release_cm}cm")


def half_open(u, v, stamp=0.0):
    """Fingertips ~3.6cm apart: enough to end a click, not enough for a drag."""
    h = hand_at(u, v, OPEN_PALM, stamp)
    palm_m = lm.hand_scale(h.world)
    offset = 0.036
    thumb = [lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]
    h.world[thumb] += (h.world[lm.INDEX_TIP] + np.array([offset, 0.0, 0.0])
                       - h.world[lm.THUMB_TIP])
    palm_img = np.linalg.norm(h.metric[lm.MIDDLE_MCP] - h.metric[lm.WRIST])
    h.metric[thumb] += (h.metric[lm.INDEX_TIP]
                        + np.array([offset / palm_m * palm_img, 0.0])
                        - h.metric[lm.THUMB_TIP])
    h.norm[thumb] = h.metric[thumb] / np.array([16 / 9, 1.0])
    return h


print(f"      the wobble used below measures "
      f"{half_open(0.5, 0.5).pinch_cm(lm.INDEX_TIP):.1f}cm")
c = controller()
run_frames(c, [(hand_at, 0.4, 0.4, 18), (pinched_at, 0.4, 0.4, 4),
               (pinched_at, 0.6, 0.6, 12), (half_open, 0.6, 0.6, 2),
               (pinched_at, 0.6, 0.6, 12), (hand_at, 0.6, 0.6, 14)])
names = [e[0] for e in c.mouse.events]
check(f"a brief wobble mid-drag does not drop it "
      f"({names.count('left_up')} release, not 2)",
      names.count("left_down") == 1 and names.count("left_up") == 1)
check("and the drag still ends when the hand really opens",
      not c.mouse.left_is_down)

# The same wobble during a plain click is allowed to end it, since nothing
# is being carried.
c = controller()
run_frames(c, [(hand_at, 0.5, 0.5, 18), (pinched_at, 0.5, 0.5, 4),
               (half_open, 0.5, 0.5, 5), (hand_at, 0.5, 0.5, 6)])
check("a click still releases promptly at the normal threshold",
      [e[0] for e in c.mouse.events].count("left_up") == 1)

print("\n      Short drags need a real nudge now, not a twitch:")
# The thresholds come from a measured session: the cursor wanders about 42px
# in 200ms from tracking noise and a hand held in the air alone, so anything
# below that cannot be told from a click that stayed put. Losing sub-50px
# drags is the price of clicks not turning into drags by themselves.
c = controller()
run_frames(c, [(hand_at, 0.5, 0.5, 18), (pinched_at, 0.5, 0.5, 16),
               (pinched_at, 0.54, 0.5, 14), (hand_at, 0.54, 0.5, 14)])
names = [e[0] for e in c.mouse.events]
held = c.mouse.events[names.index("left_down"):names.index("left_up")]
check(f"a held pinch nudged a few pixels drags too "
      f"({len([e for e in held if e[0] == 'move'])} moves)",
      any(e[0] == "move" for e in held))

# --------------------------------------------------- one pinch, one click
print("\nOne pinch must produce exactly one click:")
# Fingertips in contact occlude each other and MediaPipe's estimate of where
# they are gets jumpy, so the measured gap spikes wide for the odd frame in
# the middle of a perfectly steady pinch. That is what turns one click into
# several.
def gap_frames(pattern, u=0.5, v=0.5, start=0.0):
    """Feed a sequence of fingertip gaps in cm."""
    c = controller()
    t = start
    for cm in pattern:
        t += 1 / 30
        c._update(gap_at_uv(cm, u, v, stamp=t))
    return c


def gap_at_uv(cm, u, v, stamp=0.0):
    h = hand_at(u, v, OPEN_PALM, stamp)
    palm_m = lm.hand_scale(h.world)
    offset = cm / 100.0
    thumb = [lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]
    h.world[thumb] += (h.world[lm.INDEX_TIP] + np.array([offset, 0.0, 0.0])
                       - h.world[lm.THUMB_TIP])
    palm_img = np.linalg.norm(h.metric[lm.MIDDLE_MCP] - h.metric[lm.WRIST])
    h.metric[thumb] += (h.metric[lm.INDEX_TIP]
                        + np.array([offset / palm_m * palm_img, 0.0])
                        - h.metric[lm.THUMB_TIP])
    h.norm[thumb] = h.metric[thumb] / np.array([16 / 9, 1.0])
    return h


steady = [6.0] * 20                    # hand open
pinch = [0.8] * 18                     # a single, steady, deliberate pinch
after = [6.0] * 8
clean = gap_frames(steady + pinch + after)
clicks = [e[0] for e in clean.mouse.events].count("left_down")
print(f"      a clean 18-frame pinch -> {clicks} click(s)")
check("a clean pinch gives one click", clicks == 1)

# The same pinch, with the tracker briefly losing the fingertips twice.
spiky = list(pinch)
spiky[5] = 7.0                          # one frame reads wide open
spiky[6] = 5.0
spiky[11] = 8.0                         # and again later
noisy = gap_frames(steady + spiky + after)
clicks = [e[0] for e in noisy.mouse.events].count("left_down")
print(f"      the same pinch with 3 spiked frames -> {clicks} click(s)")
check("spikes in the middle of a pinch do not restart it", clicks == 1)

# Rapid alternation should not machine-gun clicks either.
chatter = []
for _ in range(10):
    chatter += [0.8, 0.8, 7.0]
noisy2 = gap_frames(steady + chatter + after)
clicks = [e[0] for e in noisy2.mouse.events].count("left_down")
print(f"      10 cycles of 2-frames-shut/1-frame-open -> {clicks} click(s)")
check("chatter cannot machine-gun clicks", clicks <= 3)

# But a real double click still has to get through.
deliberate = ([6.0] * 20 + [0.8] * 4 + [6.0] * 7 + [0.8] * 4 + [6.0] * 6)
dbl = gap_frames(deliberate)
clicks = [e[0] for e in dbl.mouse.events].count("left_down")
gap_ms = 7 / 30 * 1000
print(f"      two deliberate pinches {gap_ms:.0f}ms apart -> {clicks} click(s)")
check("a deliberate double click still gets through", clicks == 2)

print("\nThe on-screen keyboard pose:")
from gesture_control import keyboard as kb


def thumbs_up(u, v, stamp=0.0):
    """Thumb out, every finger shut."""
    from synthetic_hand import FIST
    h = hand_at(u, v, FIST, stamp)
    # Push the thumb out straight so it reads as extended.
    h.world[lm.THUMB_TIP] = h.world[lm.THUMB_MCP] + (
        h.world[lm.THUMB_MCP] - h.world[lm.THUMB_CMC]) * 2.2
    return h


c = controller()
check(f"a thumbs-up is recognised ('{c._classify(thumbs_up(0.5, 0.5))}')",
      c._classify(thumbs_up(0.5, 0.5)) == "keyboard")
check("an open hand is not", c._classify(hand_at(0.5, 0.5)) == "open")
check("nor is a pinch", c._classify(pinched_at(0.5, 0.5)) != "keyboard")
check("the keyboard module can see whether osk is up",
      isinstance(kb.is_showing(), bool))

# -------------------------------------------------------------- right click
print("\nRight click, and why it used to fire a left click:")


def touch(u, v, tip, cm=0.7, stamp=0.0, pose=OPEN_PALM):
    """Thumb brought `cm` from the given fingertip, in image and in 3D."""
    h = hand_at(u, v, pose, stamp)
    palm_m = lm.hand_scale(h.world)
    off = cm / 100.0
    thumb = [lm.THUMB_MCP, lm.THUMB_IP, lm.THUMB_TIP]
    h.world[thumb] += h.world[tip] + np.array([off, 0.0, 0.0]) - h.world[lm.THUMB_TIP]
    palm_img = np.linalg.norm(h.metric[lm.MIDDLE_MCP] - h.metric[lm.WRIST])
    h.metric[thumb] += (h.metric[tip] + np.array([off / palm_m * palm_img, 0.0])
                        - h.metric[lm.THUMB_TIP])
    h.norm[thumb] = h.metric[thumb] / np.array([16 / 9, 1.0])
    return h


# The shape people actually make for a "finger gun": thumb resting against
# the side of the index finger. That is a centimetre or two from the index
# tip -- which is a pinch, as far as any distance test can tell.
gun = touch(0.5, 0.5, lm.INDEX_TIP, cm=1.2)
print(f"      a finger-gun's thumb sits {gun.pinch_cm(lm.INDEX_TIP):.1f}cm from the "
      f"index tip, under the {settings.pinch_on_cm}cm click threshold")
check("which is why that pose could never be told from a left click",
      gun.pinch_cm(lm.INDEX_TIP) < settings.pinch_on_cm)

right = touch(0.5, 0.5, lm.MIDDLE_TIP, cm=0.7)
print(f"      the middle-finger pinch: thumb {right.pinch_cm(lm.MIDDLE_TIP):.1f}cm "
      f"from the middle tip, {right.pinch_cm(lm.INDEX_TIP):.1f}cm from the index")
check("those two distances are far enough apart to tell apart",
      right.pinch_cm(lm.INDEX_TIP)
      > right.pinch_cm(lm.MIDDLE_TIP) + settings.right_click_clearance_cm)


def hold(maker, n, c=None, start=0.0, **kw):
    c = c or controller()
    t = getattr(c, "_t", start)
    for _ in range(n):
        t += 1 / 30
        c._update(maker(stamp=t, **kw))
    c._t = t
    return c


c = controller()
hold(lambda stamp: hand_at(0.5, 0.5, stamp=stamp), 20, c)
hold(lambda stamp: touch(0.5, 0.5, lm.MIDDLE_TIP, 0.7, stamp), 10, c)
hold(lambda stamp: hand_at(0.5, 0.5, stamp=stamp), 8, c)
names = [e[0] for e in c.mouse.events]
print(f"      pinching the middle finger -> {names.count('right_click')} right, "
      f"{names.count('left_down')} left")
check("pinching the middle finger right clicks", names.count("right_click") == 1)
check("and does not left click", names.count("left_down") == 0)

# The ordinary left click must be untouched by all this.
c = controller()
hold(lambda stamp: hand_at(0.4, 0.4, stamp=stamp), 20, c)
hold(lambda stamp: touch(0.4, 0.4, lm.INDEX_TIP, 0.7, stamp), 5, c)
hold(lambda stamp: hand_at(0.4, 0.4, stamp=stamp), 8, c)
names = [e[0] for e in c.mouse.events]
print(f"      pinching the index finger  -> {names.count('right_click')} right, "
      f"{names.count('left_down')} left")
check("pinching the index still left clicks", names.count("left_down") == 1)
check("and does not right click", names.count("right_click") == 0)

# Holding the right-click pose must fire once, not repeatedly.
c = controller()
hold(lambda stamp: hand_at(0.5, 0.5, stamp=stamp), 20, c)
hold(lambda stamp: touch(0.5, 0.5, lm.MIDDLE_TIP, 0.7, stamp), 40, c)
hold(lambda stamp: hand_at(0.5, 0.5, stamp=stamp), 8, c)
fired = [e[0] for e in c.mouse.events].count("right_click")
print(f"      holding it for {40 / 30:.1f}s -> {fired} right click(s)")
check("holding the pose fires one right click, not a stream", fired == 1)

print("\nA single cursor jump must not start a drag:")
# Measured on a real session the cursor jumped 270px in one frame at the 99th
# percentile and 889px at worst, which is past any sane travel threshold. Only
# travel that stays travelled counts as a drag.
c = controller()
t = 0.0
for _ in range(18):
    t += 1 / 30
    c._update(hand_at(0.5, 0.5, stamp=t))
for _ in range(4):
    t += 1 / 30
    c._update(pinched_at(0.5, 0.5, stamp=t))
t += 1 / 30
c._update(pinched_at(0.62, 0.5, stamp=t))       # one frame far away
for _ in range(4):
    t += 1 / 30
    c._update(pinched_at(0.5, 0.5, stamp=t))    # and straight back
check("one stray frame does not begin a drag", not c._dragging)
for _ in range(10):                             # now really move
    t += 1 / 30
    c._update(pinched_at(0.70, 0.5, stamp=t))
check("sustained travel still does", c._dragging)

print("\nRESULT:", "all checks passed" if ok else "SOME CHECKS FAILED")
raise SystemExit(0 if ok else 1)
