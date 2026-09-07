# Gesture Control

Point at your screen and the mouse goes there. Pinch to click.

A webcam watches your hand, MediaPipe finds 21 landmarks on it, and a mapping
learned during calibration turns your fingertip into a screen coordinate. Real
Windows mouse input comes out the other end, so it works in every application,
not just this one.

## Start it

Double-click **`run.bat`**.

The first run takes about a minute: it builds a Python environment, installs the
dependencies, and downloads the hand tracking model (7.8 MB, from Google's
official MediaPipe bucket). After that it starts in a couple of seconds.

You need Python 3.10 or newer installed. If it is missing, `run.bat` says so and
links to the installer — tick *Add python.exe to PATH* during setup.

## How it works

The camera's view is stretched to fill your screen, and then hidden. Wherever
your hand is on that invisible picture is where the cursor goes: a hand a third
of the way across the frame puts the cursor a third of the way across the
screen. Nothing is calibrated, because there is nothing to fit.

It is mirrored, so moving your hand to the right moves the cursor right.

The whole screen therefore costs a whole armspan of movement, and that is the
point rather than a drawback. Every fitted alternative squeezes some smaller
region of the camera's view over the full screen and magnifies the tracking
noise by the same factor — a calibration where the hand only swept a quarter of
the frame multiplied every wobble by four. Here the factor is the screen width
over the camera width: exactly 1 for a 1080p camera on a 1080p monitor. Three
pixels of landmark jitter stay three pixels of cursor jitter.

**Overscan.** The outer tenth of the frame counts as already off the screen: a
hand a tenth from the top puts the cursor at the very top, and anything beyond
stays pinned there. So the middle 80% of the view covers the whole monitor, the
corners are reachable without stretching to the limits of the camera's view,
and less of your reach happens out at the frame edges where landmark tracking
is worst.

It costs a 1.25× magnification — of your hand's movement, and of the tracking
jitter with it. `direct_margin` in `settings.json` sets it: `0` maps the frame
one to one and gives back the exact 1:1 jitter, `0.15` shortens the reach
further at 1.4×.

## Controlling

| gesture | does |
| --- | --- |
| move your hand | move the cursor |
| pinch thumb + index | left click |
| keep pinching and move | drag |
| index + middle up, move up/down | scroll |
| pinch thumb + middle finger | right click |
| thumbs-up (thumb out, hand shut) | show/hide the on-screen keyboard |

`P` in the preview window pauses. There is no pause *gesture*: your hand is how
you aim, so no ordinary pose can also mean stop.

Your palm is what aims, and a pinch barely moves it, so clicking does not tug
the cursor off target the way a whole-hand gesture would. Set
`click_gesture: "fist"` in `settings.json` to close your whole hand instead.

**How wide a click is.** The threshold is set in centimetres —
`pinch_on_cm` (1.6) and `pinch_off_cm` (3.2) in `settings.json` — because that
is the unit a hand is actually imagined in. MediaPipe's world landmarks give
your palm's real size, so the gap between fingertips converts to real
centimetres without any calibration, and the measurement is accurate to about
two millimetres.

The gap itself is measured in the flat image, which is precise: a centimetre is
about seventeen pixels of a 1080p frame. The 3D reading is consulted only when
it claims a gap over ~4 cm in depth, which catches the one thing the image
cannot see — a thumb held clear of the finger but hidden *behind* it. Using the
larger of the two readings unconditionally, as an earlier version did, is a
trap: for fingertips genuinely in contact the 3D distance is almost entirely
depth error, so it sets the threshold from the noise floor rather than from
your hand. That is what put the click 4.5 cm wide.

**Right click** is a thumb-to-*middle*-finger pinch, and it is deliberately
measured relatively: it only counts when the index finger is clearly further
from the thumb than the middle finger is. The gesture it replaced was a
"finger gun" — index out, thumb out, hand shut — which almost nobody makes
with the thumb clear of the index. Resting against it puts the thumb about
1.2 cm from the index tip, inside the 1.6 cm click threshold, so forming the
pose correctly fired a *left* click. No amount of tuning fixes that; the two
gestures had to become distinguishable by measurement rather than by hoping a
hand shape reads cleanly.

**Typing.** A thumbs-up — thumb out, everything else shut — brings up
Windows' own on-screen keyboard, `osk.exe`. Its keys are ordinary clickable
controls, so the cursor, the pinch-click and the snapping all work on them
already. `K` in the preview window does the same.

**One pinch, one click.** Fingertips in contact occlude each other, and
MediaPipe's estimate of where they are gets jumpy exactly then: the measured
gap spikes wide for a frame or two in the middle of a perfectly steady pinch,
which chops one click into several. The filtering is deliberately lopsided —
starting a click asks for the median of recent frames, so one stray close
reading cannot fire one, while *ending* a click asks for the smallest of them,
so the pinch is only over once every recent frame agrees it is. A short
refractory period after each click stops one pinch stuttering into a burst,
kept well under the double-click time so deliberate double clicks get through.

**Dragging.** A pinch becomes a drag by *moving*, not by being held: hold
perfectly still and it stays a click however long you keep pinching, which is
what makes careful clicks and double clicks possible. Move past
`drag_start_px` (26) and it drags; or hold past `drag_after_ms` having drifted
a few pixels, so small drags are still available.

When the drag begins the cursor carries on from where it is rather than
snapping to your hand. The gap between the two is the distance your hand
travelled while the cursor was held still for the click, and jumping it would
fling whatever you just picked up.

Ending a drag takes a more deliberate opening than ending a click —
`drag_release_cm` (4.6) against `pinch_off_cm` (3.2), and three agreeing frames
instead of one. Dropping a file mid-drag because of one badly tracked frame is
far worse than a click that lingers a frame longer.

**Double clicking.** Windows only treats two clicks as a double click if they
land within a few pixels of each other — four, by default — and cursor jitter
here is around thirteen, so a double click would reliably arrive as two
unrelated single clicks. A second click that follows quickly is therefore
placed at exactly the first one's position, which removes the distance test
from the equation and leaves the timing to you. It only applies within
`double_click_slack_px` (90), so clicking somewhere genuinely different is left
alone, and a drag never seeds one.

### Why the hand and not a finger

Aiming with a pointing finger is the obvious design and the worse one. Pointing
at a screen means pointing roughly at the camera, which is the single hardest
view to read a hand from: the finger is foreshortened to a stub and half hidden
behind itself, and a thumb held clear of it can project to the same spot and
read as a pinch that never happened.

An open hand shows the camera all of itself. A fist is four fingers
independently agreeing they are curled, against a pinch's one small distance
between two landmarks. And the aim point — the wrist and four knuckles,
averaged — barely moves as the fingers close: measured on the same hand,
closing it shifts the palm centre by 0.03 palm-lengths against the fingertip's
1.16, so clicking does not drag your aim off target.

Open and closed are separated by a wide deadband rather than one threshold, and
the hand is always somewhere inside it on the way between them. Nothing fires
there. Three slow open-close-open cycles produce exactly three clicks.

The old scheme is still there if you prefer it: set `aim_mode` to `"finger"` in
`settings.json` and recalibrate.

### Locking on to controls

Near a button, link, menu item or other control, the cursor jumps to its
centre and **stays there** while your hand wobbles. Not a gentle pull: a latch.
A control small enough to be worth helping with is smaller than the hand's own
jitter, so a nudge in the right direction is not enough — the cursor has to
stop moving altogether. Measured on a 28x28 button with 12px of hand jitter,
clicks land on target **15% of the time unaided and 100% locked on**.

A control that is much longer than it is wide is a track rather than a point —
a window title bar, a scrollbar, a menu row — so only the short axis locks. A
title bar holds the cursor at its vertical centre while letting it run freely
left and right, which is what makes a 30-pixel-tall bar easy to grab; a
scrollbar does the reverse.

Letting go takes more than latching on: the *hand*, not the pinned cursor, has
to leave a wider radius (`snap_radius_px` 85 to acquire, `snap_unlock_px` 150
to release). Without that gap the cursor would sit on the boundary and rattle
in and out of the lock.

It reads the on-screen controls through Windows UI Automation, the same
accessibility interface a screen reader uses, so it works with whatever
application is in front. Native Windows apps expose the most; browsers expose
their page content only once accessibility is active, so it is partial there.
Nothing is read except each control's rectangle and its kind.

Locking switches off while you drag or scroll — a stale rectangle would drag
the window sideways as it moved — and `S` in the preview window toggles it.

A small preview window shows what the camera sees and what mode you are in.
**`Q` or `Esc` in that window quits**, `P` pauses. `Ctrl+C` in the console works
too, and quitting always releases the mouse button, so a drag can never get
stuck down.

## When clicks misfire

Run **`tune.bat`**. It watches you point for seven seconds, then pinch for
seven, and sets the click threshold from what your hand actually does rather
than from a guess about an average one. It writes the raw measurements to
`tune_report.json`, and it refuses to save a threshold pair it cannot make
work — in particular one whose release point sits inside the range your hand
shows while merely pointing, which would let a pinch begin and never end. It also reports what it found: how far
apart your pinched and unpinched states are, how often tracking drops out, how
often your pointing finger reads as curled, and whether the pause gesture is at
risk of firing by itself. If the two pinch states overlap badly it says so —
that is a lighting or distance problem, and no threshold will paper over it.

Pose is read from MediaPipe's 3D world landmarks rather than from the flat
image, which matters more than it sounds. Pointing at a screen means pointing
roughly at the camera, and in that view a thumb held 5 cm clear of the index
finger can project to almost the same spot — a phantom pinch, measured at a
ratio of 0.03 in 2D against 0.56 in 3D for the same hand. A click also has to
survive two consecutive frames before it counts, so one badly-tracked frame
cannot fire one.

## Working out what is wrong with clicking

Quitting prints a summary of what just happened and writes
`session_log.csv`, a frame-by-frame record of the last few minutes: the gap
between your fingertips as measured, what the filter made of it, and what the
click state machine did about it. The summary names the failure modes rather
than leaving them to be guessed at:

```
14s of use, hand visible in 98% of 421 frames
6 left clicks, 1 right clicks, 2 drags
thumb-to-index gap: median 5.8cm, 10th pct 0.9cm, 90th 7.1cm (clicks under 1.6cm)
button held for: shortest 100ms, median 280ms, longest 620ms
  ! 4 press(es) under 120ms -- too short to be meant, so something is firing on its own
gap between clicks: shortest 167ms, median 890ms
  ! 3 click(s) within 250ms of the one before -- either double clicks, or one pinch counted twice
```

Each `!` line points at a different cause and a different fix, which is the
point: "clicking feels wrong" is not something a threshold can be tuned
against, but "the button is being held for 100ms at a time" is.

## When something is wrong

Run **`run.bat check`**. It lists the cameras it can actually open, confirms the
model is present, then watches your hand for five seconds and reports the frame
rate and what it detected.

- **"Could not open camera 0"** — the usual cause on a fresh machine is not a
  broken webcam but Python's camera *permission*. Python installed from the
  Microsoft Store is a packaged app, so Windows gives it its own entry under
  *Settings › Privacy & security › Camera* rather than covering it under "let
  desktop apps access your camera" — and it defaults to off. A blocked app can
  still list cameras by name but never open one, which looks exactly like a
  missing webcam. `run.bat check` detects this case and says so. Either switch
  Python on in that Settings list, or install Python from python.org (an
  ordinary desktop app, not gated separately), delete `.venv`, and rerun.
- **No camera found at all** — close anything else using the webcam (Teams,
  Zoom, the Camera app). If you have more than one camera, pick another with
  `run.bat --camera 1`.
- **It keeps losing my hand / the cursor stalls** — check the frame rate first
  with `run.bat check`, which measures what each resolution actually delivers.
  A smaller frame is *not* reliably faster: webcams offer compressed formats at
  some resolutions and not others, and on the camera this was built against
  1920×1080 runs at 30 fps while 1280×720 manages 10. `tune.bat` reports how
  often tracking actually drops out and how often your pointing finger reads as
  curled.
- **It pauses by itself** — the open-palm gesture now needs all five fingers
  straight *and* splayed apart *and* the hand held still for most of a second.
  If it still fires, raise `palm_spread_min` in `settings.json`, or set
  `palm_pause_ms` very high to retire the gesture and use `P` in the preview
  window instead.
- **Cursor is jumpy** — more light on your hand helps most; watch the `jitter`
  figure in the preview while you adjust. Otherwise lower `filter_min_cutoff`
  in `settings.json` for more smoothing (at the cost of a little lag).
- **Cursor drifts off target near the edges** — recalibrate with more dots:
  `calibrate.bat --grid 5x4`.
- **Clicks fire when you did not mean them** — run `tune.bat`, which measures
  your hand and sets the threshold from it. Nudging `pinch_on` down by hand
  works too, but the measurement tells you *why* it was wrong.
- **Second monitor** — calibration covers the monitor Windows calls primary, and
  the cursor stays on it. Multi-monitor pointing would need a separate
  calibration per screen.
- **The cursor will not move at all in one particular app** — Windows blocks
  synthetic input aimed at a window running as administrator, and many
  fullscreen games clip the cursor to themselves. Everything still works
  everywhere else; alt-tab out, or start `run.bat` as administrator too if you
  need to control an elevated window.

## Making it more accurate

Adding calibration dots is the obvious lever and very nearly the useless one.
Measured against a simulated camera (1920×1080, RMS error in screen pixels):

| calibration dots | mapping alone | error you actually get |
| --- | --- | --- |
| 4 | 6.1 px | 13.9 px |
| 12 | 6.9 px | 14.7 px |
| 48 | 2.9 px | 13.2 px |

Quadrupling the dots halves the mapping error and moves the real error by about
5%, because the mapping is already far below the noise floor. Per-frame landmark
jitter is what you are actually fighting: at typical tracking quality the fit
contributes ~4 px and jitter ~13 px. Calibration noise averages away (18 frames
per dot); runtime jitter does not.

Four of these are already applied by default; the first two are up to you.

1. **Point across more of the camera frame.** If your fingertip only sweeps a
   quarter of the frame, every pixel of jitter is magnified onto the screen:
   25% sweep → 29 px error, 40% → 18 px, 55% → 14 px. Sit closer, or make
   larger pointing movements. Calibration measures your sweep and says so if it
   was small.
2. **Light your hand**, not the wall behind it. Good light versus poor is worth
   roughly 3× (8.6 px vs 24.6 px). The preview window shows a live `jitter`
   figure in pixels — green under 8, red over 18 — so you can aim a lamp and
   watch it improve.
3. **Camera resolution** now defaults to 1920×1080 rather than 720p, for more
   pixels on the hand. If your machine cannot keep up, the app says so on
   startup; drop `camera_width` / `camera_height` to 1280 / 720.
4. **Snapping to controls** (above) gives away the last dozen pixels where it
   counts.
5. **Click anchoring** places a click where your hand was just before the pinch,
   averaged over however long you held still — a pinch tugs the finger off
   target, and the averaging window shortens automatically if you clicked the
   instant you arrived.
6. **More smoothing** — lower `filter_min_cutoff`. Buys steadiness with lag.
7. **More dots** — last, and only for edge accuracy on a distorting lens.

The ceiling is structural: a single camera sees where your fingertip *is*, not
where you are *aiming*. Shift your body without changing your aim and the
cursor moves anyway. Only head tracking, to cast a real eye-through-fingertip
ray, would fix that — so calibrate sitting the way you actually sit.

## How the mapping works

The interesting question is: given a hand in a camera image, where on the screen
is it pointing? That is `gesture_control/mapping.py`, and it happens in two
stages.

**A homography.** Your fingertip moves through space; the screen is a plane. A
homography is the exact projective relationship between two planes, and it has
eight degrees of freedom, so four dots are enough to solve for one. Fitting it
absorbs everything about how the camera happens to be positioned — off to one
side, tilted, above or below the screen — without needing to know any of it.
It is fitted with the normalised DLT, and on clean synthetic data it recovers a
known mapping to within 5e-13 px.

**A residual correction.** A homography assumes your fingertip moves in a flat
plane, which it does not. Lens distortion bends the edges, where your finger
*aims* matters as well as where it *is*, and leaning towards or away from the
camera shifts everything. So a second stage does ridge regression on whatever
error the homography leaves, over features built from exactly those effects:
screen position, its quadratic terms, the direction your index finger is
pointing, and a depth term from the apparent size of your palm.

That second stage has enough parameters to memorise a dozen dots and learn
nothing, so it is not trusted by default. The feature set and the ridge strength
are both chosen by **leave-one-out cross-validation over the whole pipeline** —
the homography is refit inside every fold — and the correction is only kept if
it beats the plain homography on dots it never saw. On a well-behaved camera it
usually loses, and gets dropped. The cross-validated error is also the accuracy
figure reported after calibration, so that number is an honest out-of-sample
estimate rather than a fit residual.

Two smaller things that matter more than they sound:

- **Click anchoring.** Pinching physically tugs your index finger off target, so
  a click would land a few pixels from where you aimed. The controller keeps a
  short history of cursor positions and presses the button at the position from
  ~110 ms *before* the pinch started. The cursor then freezes until you release,
  unless you hold long enough for it to become a drag.
- **1-Euro filtering.** A fixed low-pass filter forces a choice between a
  jittery cursor and a laggy one. This one raises its cutoff with hand speed:
  slow movement is filtered hard (where jitter shows and lag does not), fast
  movement passes through nearly untouched.

## Layout

```
run.bat                  start here
calibrate.bat            recalibrate
tune.bat                 fix clicking by measuring your hand
gesture_control/
  mapping.py             the calibration fit and aim -> screen mapping
  calibrate.py           the fullscreen dot-clicking window
  controller.py          gestures -> mouse events
  targets.py             finding clickable controls, and snapping to them
  tune.py                measuring your hand to set the click threshold
  tracker.py             MediaPipe hand landmarks, on a worker thread
  camera.py              webcam capture, on another one
  mouse.py               Windows SendInput
  landmarks.py           finger geometry (extended? pinched? pointing where?)
  filters.py             1-Euro pointer smoothing
  config.py, cli.py      settings and command line
tests/
calibration.json         written by calibration
settings.json            written when calibration tunes your pinch thresholds
```

## Tests

```bash
.venv\Scripts\python tests\test_mapping.py
```

Fits the mapping against a simulated camera with known distortion and noise, and
checks the error at hundreds of points the fit never saw.

```bash
.venv\Scripts\python tests\test_end_to_end.py
```

Drives the whole calibration window and the gesture state machine with a
synthetic hand — no camera, no real mouse, nothing appears on screen. Checks
that clicks land on target, that a held pinch becomes a drag, that scrolling
does not move the cursor, and that the button cannot get stuck down.

```bash
.venv\Scripts\python tests\test_snapping.py
```

The snapping geometry: the attraction curve, that it is continuous and
monotonic so the cursor never jumps or backtracks, that it roughly triples the
odds of landing on a small button, and that it never amplifies jitter by more
than 1.4×.

The two below touch the real machine.

```bash
.venv\Scripts\python tests\test_mouse.py
```

Moves the real cursor a few pixels and puts it back, to verify the SendInput
coordinate maths against the actual desktop.

```bash
.venv\Scripts\python tests\test_targets_live.py
```

Asks Windows UI Automation for the controls in whatever windows are open and
checks the snapping works against real geometry. Reads only; clicks nothing.
