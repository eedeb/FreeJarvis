# FreeJarvis

Point at your screen and the mouse goes there. Pinch to click. Say "hey Jarvis"
and an assistant with 44 tools answers, on a translucent workbench it can put
live gauges and cards on.

A webcam watches your hand, MediaPipe finds 21 landmarks on it, and a mapping
learned during calibration turns your fingertip into a screen coordinate. Real
Windows mouse input comes out the other end, so it works in every application,
not just this one.

## Install

In PowerShell:

```powershell
irm https://raw.githubusercontent.com/eedeb/FreeJarvis/main/install.ps1 | iex
```

No administrator rights, and nothing is hosted anywhere but that one script —
the source comes from this repo, the interpreter from python.org and the
hand-tracking model from Google, so a release never has to be cut for an
install to work.

It lands in `%LOCALAPPDATA%\FreeJarvis` with a Start Menu and desktop
shortcut, a `freejarvis` command on your PATH, and a private Python that
touches nothing else on the machine. Budget a few minutes and about 700 MB:
MediaPipe, ONNX Runtime and the speech models are not small.

Re-run the same line to update. Your settings, calibration and FreeClaw link
are never touched by it.

```powershell
# start with Windows
$env:FREEJARVIS_AUTOSTART = 1
irm https://raw.githubusercontent.com/eedeb/FreeJarvis/main/install.ps1 | iex

# somewhere else, no shortcuts, do not launch it afterwards
$env:FREEJARVIS_DIR = "D:\Apps\FreeJarvis"
$env:FREEJARVIS_NO_SHORTCUT = 1
$env:FREEJARVIS_NO_START = 1
irm https://raw.githubusercontent.com/eedeb/FreeJarvis/main/install.ps1 | iex
```

To remove it:

```powershell
& "$env:LOCALAPPDATA\FreeJarvis\uninstall.ps1"
```

It offers to copy your settings, calibration and FreeClaw link to the desktop
first, and it refuses to run in a directory that has no install marker — so
pointing it at a source checkout deletes nothing.

**You need a webcam.** Everything else is optional: without a microphone it is
silent, and without a FreeClaw it is still a hand-tracking mouse.

### What the installer actually does

| | |
|---|---|
| **source** | `git clone --depth 1`, or a zip download when git isn't installed |
| **python** | uses a 3.11–3.13 you already have; otherwise fetches 3.12 and installs it *privately*, per-user, not on PATH |
| **deps** | a venv under the install directory, `pip install -r requirements.txt` |
| **model** | fetches `hand_landmarker.task` up front, so the first launch is instant |
| **shortcuts** | Start Menu and desktop, running `pythonw` so there is no console window |

The private Python is the full python.org installer rather than the embeddable
build that would be the obvious choice, for one reason: the embeddable
distribution ships without tkinter, and tkinter is what draws the "Add
FreeClaw" window and the calibration grid. An install built on it would look
perfectly fine until the moment somebody pressed `Ctrl+Alt+J`.

## Running from a checkout

Double-click **`run.bat`**. It builds a `.venv` beside the source, installs the
dependencies and downloads the model on first run. You need Python 3.11 or
newer on PATH — 3.11 because ONNX Runtime does not publish wheels below it.

## The overlay

Your hand gets an armoured glove projected onto it, floating over your desktop.
The webcam picture itself is hidden — there is no video on your screen, just the
glove tracking your hand, the orb, and whatever Jarvis is saying. You can see through it and click through it, and it sits below
every application window — open a browser and the browser is in front of it; it
covers only the wallpaper and the desktop icons.

Because it never takes focus, its keys are all `Ctrl`+`Alt`+letter, so a plain
letter still reaches whatever you are typing in:

| | |
|---|---|
| `Ctrl+Alt+Q` | quit (or `Ctrl+C` in the terminal) |
| `Ctrl+Alt+P` | pause and resume control |
| `Ctrl+Alt+K` | on-screen keyboard |
| `Ctrl+Alt+S` | snapping on and off |
| `Ctrl+Alt+J` | add or re-link a FreeClaw install (see [Jarvis](#jarvis)) |
| `Ctrl+Alt+M` | hand the mouse back, and take it again |

`--overlay-dim` brings the webcam picture back, from `0` (hidden, the default)
to `1` (fully visible). The glove and the readouts stay solid whatever it is,
so raising it fades the room in behind them rather than fading everything.
Mostly useful when the tracking is behaving oddly and you want to see what the
camera sees.

```bash
.venv\Scripts\python -m gesture_control run --overlay-dim 0.3
```

`--window` puts the preview back in an ordinary floating window, which is the
one to use when working on the gauntlet itself: it can be screenshotted and
dragged around, and a layered window cannot be.

## Jarvis

An arc reactor sits in the top right of the overlay. Say **"hey Jarvis"**, ask
for something, and it answers out loud.

The orb is a real 3D object, not a picture of one: two nested spheres of
glowing filaments, counter-rotating, projected through a perspective divide and
shaded by depth every frame. It turns slowly all the time, so you can watch the
near side sweep across and the far side dim behind it — that parallax is the
whole reason it reads as a sphere rather than as a drawn circle.

It is also the status light. It drifts when idle; **flashes and swells the
instant it hears the wake word**; brightens while the microphone is open; spins
fast while the answer is being worked out; and **pulses in time with the actual
speech** while it reads the reply back, driven by the playback waveform rather
than a fixed "talking" brightness.

The orb is drawn over a backdrop that takes the wallpaper down to near-black
first, so it looks the same over a bright sky as over a dark desktop. That
backdrop states its own opacity rather than relying on the overlay's
differs-from-the-camera rule — a detail that matters, because darkening a dim
room barely differs from it at all.

### The session log

Under the orb sits a small terminal, and everything a turn does scrolls
through it as it happens: what you said, which provider took it, the model's
reasoning, every tool call with its arguments and what it returned, and the
reply building a word at a time. A turn that spends thirty seconds running
tools looks like it is working, instead of looking broken.

```
 JARVIS // SESSION LOG                            THINKING   LIVE
 > open chrome and tell me how much memory this thing has
 :: asking Groq
 .. Two things: launch the browser, then report memory.
 $ open_app(name=chrome)
 $ -> Opened Google Chrome.
 $ system_info()
 $ -> CPU: 16 logical cores. Memory: 31.7 GB total, 12.4 GB free.
 < Chrome is up, sir. 31.7 gigabytes, with 12.4 free.
```

It is a real character grid — Consolas rendered once into a fixed cell at
startup, then the whole body drawn as a single array lookup — because tool
calls only read as tool calls when the columns line up.

It has a **fixed size and stays well clear of the bottom of the screen**. Older
output is scrollback rather than gone: it follows the newest line while you
leave it alone, stops the moment you scroll up (the header turns from `LIVE` to
`HOLD -12`, so you can see how far back you are), and follows again when you
return to the bottom. New output never moves the text out from under you
mid-sentence.

Scroll it with the **wheel**, or by **dragging it** with the mouse — or with the
hand-tracked cursor, which has no wheel. Both exist for a reason: the window
never takes focus, so the wheel only reaches it while Windows is routing wheel
events by hover, which is the default but can be turned off.

Like the audio widget, it is its own see-through window rather than something
painted on the overlay. That is what lets it redraw on its own clock — a reply
streaming in while the camera is stalled still arrives — and what makes its
text sharp, since the overlay stretches the camera frame to fill the screen and
anything painted into it is stretched with it.

### Talking to it

Three things make it a conversation rather than a command line with a
microphone on it.

**You only summon it once.** For nine seconds after Jarvis finishes speaking,
plain speech is the next turn — no second "hey Jarvis". Outside that window
the wake word is required again, because a microphone that reacts to any
noise in the room is worse than one that occasionally needs asking twice.

**You can interrupt it.** The wake word keeps being scored *while Jarvis is
talking*, so saying "hey Jarvis" over a long answer cuts it off and starts
listening. Previously the microphone was deafened during playback so Jarvis
could not wake itself, which also meant you had to sit through whatever it
had decided to say.

**It starts talking sooner.** Replies are spoken a sentence at a time as they
stream in, with the synthesiser running one sentence ahead of playback. Most
of what makes an assistant feel slow is the silence between asking and the
first sound, and that silence was as long as the whole answer took to write.

Speech recognition defaults to `base.en`, which is noticeably better than
`tiny.en` on short commands. If it is not downloaded yet Jarvis listens on
`tiny.en` immediately and swaps over in the background, so a better default
never costs a slow first run.

The brain is [FreeClaw](https://freeclaw.eedeb.dev), running either on this
machine or another one on your network.

### Connecting it

Press `Ctrl+Alt+J`, enter the address and password of your FreeClaw install,
and press Connect. The address is just the IP — `192.168.1.40`, or
`localhost` if it is on this machine — and port 6767 is assumed. The password
is the one FreeClaw's own web UI asks for.

That one button does five things:

1. creates a FreeClaw user called **Jarvis**
2. writes the Jarvis persona into that user's `context.md`
3. starts an MCP server *here* and registers it with FreeClaw
4. switches FreeClaw's OpenAI-compatible API on
5. checks it has an LLM provider enabled, and says so if it does not

It is safe to run again — that is how you push an edited persona, move to a
different FreeClaw, or repair the registration after a reinstall.

**Run it again after updating this app, too.** FreeClaw remembers what a tool
server offered the first time it asked, keyed on the server's address, and
nothing about re-registering the same address changes that. So a version of
this app with new tools in it stays invisible until you press `Ctrl+Alt+J`,
which gives FreeClaw an address it has not seen before and makes it look
again.

### What it can do to this computer

The MCP server gives the agent 44 tools that act on **this** machine, not on
FreeClaw's:

| | |
|---|---|
| **files** | `create_file` `create_folder` `read_file` `list_folder` `move_path` `delete_path` `open_path` |
| **apps** | `open_app` `list_apps` `open_url` `run_program` |
| **windows** | `list_windows` `focus_window` `window_state` `close_window` |
| **sound** | `get_volume` `set_volume` `set_muted` `media_key` |
| **the machine** | `system_info` `battery_status` `disk_usage` `list_processes` `end_process` `take_screenshot` `clipboard_get` `clipboard_set` `lock_screen` |
| **over time** | `sensor_history` |
| **the network** | `network_status` `ping` `public_network` `fetch_url` |
| **finding things** | `search_files` |
| **the screen** | `show_monitor` `show_card` `show_file` `close_card` `list_cards` |
| **standing orders** | `watch_for` `list_watches` `stop_watching` |
| **keyboard** | `type_text` `press_keys` |

`open_app` finds things by the name they have on your Start Menu or desktop,
so "open Chrome" works without knowing where Chrome is installed. `run_program`
runs a command and gives back what it printed.

**The file tools are confined** to your Desktop, Documents, Downloads and temp
folder. That boundary is deliberately narrow: an agent that browses the web can
be told what to do by a web page, and this is what stops that reaching anything
but your own documents. Widening `ROOTS` in
[tools.py](gesture_control/jarvis/tools.py) widens that too.

**The two keyboard tools are not confined by anything.** `type_text` and
`press_keys` can do whatever you could do at the keyboard, including typing
into a terminal — and they are reachable by the same agent that reads web
pages. They are on because an assistant that cannot type is much less useful,
and the persona tells Jarvis to ask before typing anywhere that runs what it is
given. If you would rather not have them at all, set `"tools_keyboard": false`
in `jarvis.json`; everything else keeps working.

The server listens on port 8788 with a bearer token minted at setup. If
FreeClaw is on another machine, that port has to be open through this one's
firewall — FreeClaw dials *in* to run a tool.

### Handing the mouse back

A small **GESTURES ON / MOUSE ONLY** button sits in the top-right corner, with
the orb below it. Click it to stop the hand driving the cursor; click it again
to take it back. Your hand is still tracked and the gauntlet still drawn either
way — only the mouse events stop.

It has to be its own little window rather than part of the overlay, because the
overlay is click-through: that is what stops it swallowing the clicks this app
makes, and the flag is per-window, so there is no way to carve one clickable
rectangle out of it. It sits in the same layer as the overlay, under every
application window, so a browser covers it like it covers everything else —
which does mean it is only clickable when nothing is on top of it.

`Ctrl+Alt+M` is the way to reach it the rest of the time, and is why the
keyboard route exists at all.

Two more buttons sit beside it: **RESET** clears the conversation on both
sides — the session log here and FreeClaw's own history, because clearing
only one is a lie — and **QUIT** closes the app, the same exit `Ctrl+Alt+Q`
takes. QUIT is placed at the far end of the strip, not next to the toggle,
since it is the one you least want to hit while reaching for the one you
actually use. So does `Ctrl+Alt+P`, and the held-open-palm pose
in finger-aiming mode; the button shows whichever of them last changed it.

### The workbench

The stacking, top to bottom:

```
your applications        untouched, they cover everything below
the widgets              toggle button, session log, cards, audio panel
the camera overlay       see-through camera, glove, orb
your wallpaper           and the desktop icons
```

**The audio widget**, bottom right. The top half is a 24-band spectrum of what
your speakers are *actually* playing, mirrored about a centre line so it reads
as a waveform. The bottom half is the system volume — drag it and the real
master volume moves, the same control the tray icon drives.

It reads the output two ways, in order of preference: WASAPI loopback, which
gives the real samples and so a real spectrum; or failing that the audio
endpoint's own peak meter, which is one level rather than a spectrum, so the
bars become a scrolling history instead. When it falls back it says `level` in
the corner rather than letting a history pass for a spectrum.

Like the session log, this panel answers the mouse — a slider has to be
draggable, and the overlay is click-through by design. So it is its own
translucent window, which is also why it redraws on its own 30 Hz clock rather
than with the camera. Both sit *under* your applications, so the only clicks
they take are ones that would otherwise have landed on the wallpaper.

### Cards, and monitoring the situation

Jarvis can put things on the screen and leave them there. Two kinds, and the
difference between them is the whole point:

**Monitors** watch something and keep being true on their own — `cpu`,
`memory`, `network`, `battery`, `disk`, `processes`. "Keep an eye on the
network" opens one and Jarvis stops thinking about it; it redraws itself from
the sensors ten times a second whether or not anyone is talking. Each shows
the current reading, a bar, the numbers behind it, and a two-minute graph, so
a glance answers *has it been like this for long* as well as *what is it now*.

**Cards** are Jarvis's own text, held still until it has reason to change
them: a checklist, a countdown, the four steps you asked for. Writing to the
same title replaces the contents, which is how one gets kept up to date.

**File cards** show a file — text, markdown, or a picture — and *follow* it.
`show_file` once, then rewrite the file, and the screen changes on its own
with no second tool call. That is the difference between a widget and a
snapshot: a card that has to be re-issued is silently stale the moment
anything else happens. Point one at `ping.md` and Jarvis has a panel it can
keep updating by writing a file.

They stack down the left edge — the right is the orb, the log and the audio
widget — and they are placed when they open and do not move afterwards, so
the thing you were reading stays where you were reading it.

```
hey jarvis, keep an eye on the cpu and tell me if it spikes
hey jarvis, put the deployment steps on my screen
hey jarvis, clear the screen
```

### Standing orders

A monitor is something *you* look at. `watch_for` is the other half — "tell me
if the network drops" — and it is what makes it worth leaving running rather
than something you go and consult. Jarvis speaks up on its own when a
condition holds.

Three things keep that from being a nuisance:

- **It has to hold.** CPU touches 100% every time anything launches. A watch
  has to be continuously true for eight seconds before it says anything.
- **It fires once.** Having said the disk is nearly full, saying it again
  thirty seconds later carries no information and interrupts whatever you did
  about the first one. A tripped watch stays quiet until the reading has
  recovered past a margin — hysteresis, not a cooldown, so a value sitting on
  the threshold cannot chatter across it.
- **It waits its turn.** Announcements are held while a turn is running or
  while Jarvis is speaking, rather than cutting across your own conversation.

```
hey jarvis, tell me if the wifi drops
hey jarvis, warn me when the disk gets under twenty gigs
hey jarvis, what are you watching?
```

### What it can find out

The sensors behind those monitors are also tools, so Jarvis answers from
measurements rather than from guesses. Everything here reads the same
one-second sampler, so the number it says out loud is the number on the card.

| | |
|---|---|
| **the machine** | `system_info` `battery_status` `disk_usage` `list_processes` `end_process` |
| **over time** | `sensor_history` — *has* it been like this for long, not just *is* it |
| **the network** | `network_status` `ping` `public_network` `fetch_url` |
| **finding things** | `search_files` |

`network_status` reads this machine only: Wi-Fi name, signal, band and link
rate from `netsh`, addresses and the default route, whether the internet
answers and how fast traffic is flowing. `public_network` is kept separate
because it is the one that asks an outside service — different question,
different consequences, and a tool list is the wrong place to be vague about
which is which.

`fetch_url` fetches from *this* machine, which is the point: a router's
status page, a printer, a NAS, something on localhost — none of them
reachable from wherever FreeClaw is running.

### Turning bits of it off

```jsonc
// jarvis.json, beside the app
{ "voice": false,        // stop listening for "hey Jarvis"
  "speech": false,       // reply on the overlay only, no speaking
  "reactor_size": 0.26,  // orb size, as a fraction of the shorter screen edge
  "speech_model": "base.en",   // faster-whisper model for what you say
  "voice_name": "en-GB-RyanNeural",
  "followup_seconds": 9,       // 0 requires "hey Jarvis" every turn
  "audio_widget": true,  // the spectrum and volume slider
  "terminal": true,      // the session log under the orb
  "terminal_width": 0.30,  // its size, as a fraction of the screen
  "terminal_height": 0.30,
  "tools_keyboard": true // let the agent type and press keys
}
```

Everything here is optional. No FreeClaw, no microphone or no speakers each
turns off just the part that needs it, says so once at startup, and leaves the
rest of the overlay working.

## The glove

Your hand gets a gauntlet — not painted onto it, projected onto it. It is
translucent, lit along its edges and faint across its faces, drawn in one
colour of cyan with fine scanlines and a colour fringe, so it reads as light
rather than as a glove you are wearing. Whatever is on your desktop shows
through it.

With the camera hidden (the default) there is nothing behind the glove for it
to be measured against, so its own brightness becomes its opacity — bright
edges are solid, faint faces are see-through. Turn the camera back on and it is
your hand showing through instead.

That is one property doing most of the work: a surface turned away from the
camera is bright, a surface facing it is faint. It is what you see of a
projected solid — its edges, not its faces — and it comes out of the fragment
shader in a line.

`--gauntlet-style solid` paints it as red and gold armour instead.

### What the overlay costs

Worth knowing, because it shares a machine with hand tracking and the two are
competing for the same cores. Measured on the machine this was built on, at
30fps with six monitors open:

| | per frame | per second |
|---|---|---|
| mirroring the camera frame | 0.22 ms | 7 ms |
| the arc reactor | 3.7 ms | 111 ms |
| packing and scaling for Windows | 3.5 ms | 105 ms |
| six monitors (asked once a second) | 0.28 ms | 0.3 ms |
| **the overlay, all in** | | **223 ms — 22% of a core** |

Hand tracking is about 10 ms a frame on its own thread and is not on that
list: MediaPipe's cost is the neural net, which runs at a fixed input size, so
feeding it a smaller frame does not make it cheaper. That was measured against
the real captures in `captures/` rather than assumed, and a smaller input was
also measurably *less* accurate.

Four things carry most of that budget, and each is documented where it lives:

- **The orb's wide bloom** is blurred at quarter size and scaled back up.
  A Gaussian of sigma *s* on an image reduced by *k* is a Gaussian of *s·k* on
  the original, at 1/k² the pixels with a 1/k kernel — the same glow for a
  sixteenth of the arithmetic, with the worst pixel differing by 0.8 of 255.
- **Depth shading is a 256-entry table**, not arithmetic over eight thousand
  points. The colour of a point depends on one number, so there are only 256
  distinct answers.
- **The frame is cropped before it is mirrored.** Two thirds of a mirrored
  1080p frame were being thrown away immediately afterwards. With the camera
  hidden it is not mirrored at all — the result was known before the camera
  was read.
- **The pixels are packed at the frame's size and scaled once**, rather than
  scaling the colour and the mask separately and then doing three passes at
  screen size. That is also the more correct order: premultiplied alpha is
  exactly the representation in which interpolating a translucent image is
  valid, so scaling afterwards is what *stops* edges haloing.

The monitors are asked what they say once a second rather than ten times,
because the sampler behind them only moves once a second. One of them had been
opening a TCP connection to a DNS resolver on every redraw to check whether
the internet was up — ten sockets a second for a number that cannot change
that fast.

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
- **It feels laggy with a lot on screen** — the overlay's own drawing costs
  about 22% of one core at 30fps with every monitor open, and hand tracking
  costs roughly another 30% on its own thread. If that is too much on your
  machine, in rough order of what buys the most back:
  `close_card all` (each card is a layered window the compositor has to
  blend), `"audio_widget": false`, a smaller `"reactor_size"` — the orb is the
  single most expensive thing drawn, and its cost goes with the square of its
  size — and finally `"terminal": false`.
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
  keyboard.py            Windows key events, for the scroll and menu poses
  landmarks.py           finger geometry (extended? pinched? pointing where?)
  filters.py             1-Euro pointer smoothing
  session.py             the CSV log of what the controller did each frame
  overlay.py             the see-through, click-through full-screen window
  jarvis/
    agent.py             the state machine: heard -> asked -> spoken
    client.py            the FreeClaw HTTP surfaces
    link.py              "Add FreeClaw": one call that wires the two together
    tools.py             the MCP server FreeClaw calls back into
    voice.py             "hey Jarvis", speech to text, and the voice it answers in
    reactor.py           the arc reactor in the corner
    terminal.py          the session log under it, and its scrollback
    sensors.py           one sampler: cpu, memory, disks, battery, network
    status.py            the tools that answer rather than change
    screen.py            the tools that put cards on the screen
    cards.py             those cards, and the board that arranges them
    holo.py              one palette and one frame for every see-through panel
    watch.py             standing orders it speaks up about on its own
    toggle.py            the clickable GESTURES ON / MOUSE ONLY button
    widget.py            a see-through window that answers the mouse
    audio.py             the output spectrum, and the system volume
    audio_panel.py       the audio widget drawn on top of those
    actions.py           apps, windows, sound, clipboard, screenshots
    dialog.py            the address-and-password window
    persona.md           who Jarvis is, written into FreeClaw's context.md
  glove.py               the gauntlet overlay: skinning the rig to the hand
  gltf.py                reading models/gauntlet.glb
  render.py              drawing it, on the GPU or on the CPU
  gauntlet.py            the flat drawn fallback, when neither is available
  config.py, cli.py      settings and command line
tools/
  bl_gauntlet.py         generates models/gauntlet.glb, runs inside Blender
  blender_run.py         runs a bl_*.py script inside headless Blender
tests/
models/gauntlet.glb      the generated glove (committed; rebuild with bl_gauntlet)
jarvis.json              written by Add FreeClaw (holds the password)
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
