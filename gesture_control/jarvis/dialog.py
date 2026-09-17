"""The "Add FreeClaw" window: an address, a password, and a running log.

Tkinter, because it is in the standard library and this window appears at most
once.  Anything richer would be a second UI toolkit in the dependency list for
two text fields.

It runs in its own process rather than on the overlay's thread.  Two reasons,
both practical: Tk insists on owning the thread it was created on and will
deadlock or crash if driven from another, and the overlay's own loop must keep
running while the dialog is open -- the reactor should not freeze mid-spin
because someone is typing a password.  `open_dialog()` launches it with
`python -m gesture_control.jarvis.dialog` and returns immediately; the wiring
it performs is persisted to jarvis.json, which the app re-reads afterwards.
"""

from __future__ import annotations

import queue
import subprocess
import sys
import threading

from . import link, settings as store

PAPER = "#12151b"
INK = "#c8d2e0"
GLOW = "#ffa020"
FIELD = "#1c212b"


def open_dialog() -> subprocess.Popen | None:
    """Launch the dialog in its own process. Returns it, or None if it failed."""
    try:
        return subprocess.Popen(
            [sys.executable, "-m", "gesture_control.jarvis.dialog"],
            cwd=str(store.ROOT),
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0))
    except OSError:
        return None


def run() -> int:
    import tkinter as tk
    from tkinter import ttk

    saved = store.load()
    root = tk.Tk()
    root.title("Add FreeClaw")
    root.configure(bg=PAPER)
    root.resizable(False, False)
    # Above the overlay and anything else, since it was asked for explicitly.
    root.attributes("-topmost", True)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    style.configure("J.TLabel", background=PAPER, foreground=INK)
    style.configure("J.TEntry", fieldbackground=FIELD, foreground=INK,
                    insertcolor=GLOW, borderwidth=0)
    style.configure("J.TButton", background=FIELD, foreground=INK, borderwidth=0)
    style.map("J.TButton", background=[("active", "#2a313f")])

    frame = tk.Frame(root, bg=PAPER, padx=22, pady=18)
    frame.pack(fill="both", expand=True)

    tk.Label(frame, text="J A R V I S", bg=PAPER, fg=GLOW,
             font=("Consolas", 15, "bold")).grid(row=0, column=0, columnspan=2,
                                                 sticky="w")
    tk.Label(frame, text="Point Jarvis at a FreeClaw install.", bg=PAPER, fg=INK,
             font=("Segoe UI", 9)).grid(row=1, column=0, columnspan=2,
                                        sticky="w", pady=(2, 14))

    tk.Label(frame, text="Address", bg=PAPER, fg=INK,
             font=("Segoe UI", 9)).grid(row=2, column=0, sticky="w")
    address = ttk.Entry(frame, width=34, style="J.TEntry", font=("Consolas", 10))
    address.insert(0, saved.url or "127.0.0.1:6767")
    address.grid(row=2, column=1, sticky="we", pady=3)

    tk.Label(frame, text="Password", bg=PAPER, fg=INK,
             font=("Segoe UI", 9)).grid(row=3, column=0, sticky="w")
    password = ttk.Entry(frame, width=34, show="•", style="J.TEntry",
                         font=("Consolas", 10))
    password.insert(0, saved.password or "")
    password.grid(row=3, column=1, sticky="we", pady=3)

    tk.Label(frame, text="The IP of the machine FreeClaw runs on, and the "
                         "password its\nweb UI asks for. Port 6767 is assumed.",
             bg=PAPER, fg="#6b7789", font=("Segoe UI", 8),
             justify="left").grid(row=4, column=0, columnspan=2, sticky="w",
                                  pady=(6, 10))

    log = tk.Text(frame, height=10, width=54, bg="#0c0f14", fg=INK,
                  font=("Consolas", 9), relief="flat", wrap="word",
                  insertbackground=PAPER, highlightthickness=0)
    log.grid(row=5, column=0, columnspan=2, sticky="we")
    log.configure(state="disabled")
    log.tag_configure("bad", foreground="#ff6b6b")
    log.tag_configure("good", foreground="#7ddc8a")

    buttons = tk.Frame(frame, bg=PAPER)
    buttons.grid(row=6, column=0, columnspan=2, sticky="e", pady=(12, 0))
    connect = ttk.Button(buttons, text="Connect", style="J.TButton", width=12)
    connect.pack(side="right", padx=(8, 0))
    ttk.Button(buttons, text="Close", style="J.TButton", width=10,
               command=root.destroy).pack(side="right")

    # The worker cannot touch a Tk widget, so it posts lines here and the UI
    # thread drains the queue on a timer. Driving Tk from another thread is
    # the classic way to make it hang with no error at all.
    lines: queue.Queue = queue.Queue()

    def write(text: str, tag: str = "") -> None:
        log.configure(state="normal")
        log.insert("end", text + "\n", tag)
        log.see("end")
        log.configure(state="disabled")

    def drain() -> None:
        while True:
            try:
                text, tag = lines.get_nowait()
            except queue.Empty:
                break
            if text is None:
                connect.configure(state="normal", text="Connect")
            else:
                write(text, tag)
        root.after(80, drain)

    def work(url: str, secret: str) -> None:
        try:
            for line in link.link(url, secret, progress=None):
                lines.put((line, "good" if line.startswith("Ready") else ""))
        except link.LinkError as exc:
            lines.put((str(exc), "bad"))
        except Exception as exc:                                   # noqa: BLE001
            lines.put((f"Unexpected failure: {exc}", "bad"))
        finally:
            lines.put((None, ""))

    def go() -> None:
        log.configure(state="normal")
        log.delete("1.0", "end")
        log.configure(state="disabled")
        connect.configure(state="disabled", text="Working...")
        threading.Thread(target=work, args=(address.get().strip(),
                                            password.get()),
                         daemon=True).start()

    connect.configure(command=go)
    root.bind("<Return>", lambda _event: go())
    root.bind("<Escape>", lambda _event: root.destroy())
    password.focus_set() if saved.url else address.focus_set()

    root.after(80, drain)
    root.eval("tk::PlaceWindow . center")
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
