"""Talking to a running FreeClaw install.

Two authentication schemes, because FreeClaw uses two:

  * **the admin API** (`/login`, `/api/users`, `/api/mcp`, `/api/providers`,
    `/api/users/<name>/context`) wants a session cookie, so `admin()` logs in
    and hands back a `requests.Session` carrying it.  Used once, by link.py.

  * **`/v1/chat/completions`** wants the same password as a bearer token, and
    runs one whole agent turn per call.  That is `ask()` -- the whole reply or
    nothing, no visibility into the middle.

  * **the `/chat` SSE stream**, the same endpoint the web UI runs on, also
    session-authenticated.  That is `Chat`, and it is what "hey jarvis" ends
    up in: it narrates the turn as it happens -- which provider answered, the
    model's reasoning, every tool call and its result -- so the overlay can
    show a thirty-second turn working rather than sitting blank.

`ask()` is kept for the one job it is better at: a single question with no
session to establish, used to check the connection.  Everything the user
actually says goes through `Chat`.

Nothing here modifies FreeClaw.  Every call is an endpoint it already serves.
"""

from __future__ import annotations

import json

import requests

# A turn can search the web, scrape pages and run several tools before it says
# anything.  Tight on connect so a wrong IP fails while the dialog is still
# open; generous on read so a working one is allowed to think.
CONNECT_TIMEOUT = 6
READ_TIMEOUT = 300


class FreeClawError(Exception):
    """Anything that stopped us getting an answer, phrased for the user."""


def _base(url: str) -> str:
    return (url or "").rstrip("/")


def normalise(url: str) -> str:
    """Turn what someone typed into a URL.

    People type an IP.  Assume http, and assume FreeClaw's port rather than
    80 -- `192.168.1.40` means the FreeClaw over there, not a web server.
    """
    url = (url or "").strip().rstrip("/")
    if not url:
        return ""
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    host = url.split("://", 1)[1]
    if ":" not in host.split("/", 1)[0]:
        url = url + ":6767"
    return url


def _why(exc: Exception, url: str) -> FreeClawError:
    if isinstance(exc, requests.exceptions.ConnectTimeout):
        return FreeClawError(f"FreeClaw did not answer at {url}. Is it running?")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return FreeClawError(f"Could not reach FreeClaw at {url}. Is it running, "
                             f"and is that address right?")
    if isinstance(exc, requests.exceptions.ReadTimeout):
        return FreeClawError("FreeClaw took too long to answer.")
    return FreeClawError(f"Could not reach FreeClaw: {exc}")


def _body(resp, limit: int = 300) -> str:
    """Best-effort readable text out of an error response."""
    try:
        data = resp.json()
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:limit]
            if err:
                return str(err)[:limit]
        return str(data)[:limit]
    except ValueError:
        return (resp.text or "")[:limit]


# -- admin -----------------------------------------------------------------

def admin(url: str, password: str) -> requests.Session:
    """Log in and return a session carrying the cookie."""
    url = _base(url)
    session = requests.Session()
    try:
        session.post(f"{url}/login", data={"password": password},
                     timeout=(CONNECT_TIMEOUT, 30), allow_redirects=False)
    except requests.RequestException as exc:
        raise _why(exc, url) from exc
    # A good password redirects and sets a cookie; a bad one re-renders the
    # login form with a 200 and no cookie, so the status code says nothing.
    if not session.cookies.get("session"):
        raise FreeClawError("FreeClaw rejected that password.")
    return session


def users(session, url: str) -> list[str]:
    resp = session.get(f"{_base(url)}/api/users", timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code != 200:
        raise FreeClawError(f"Could not list FreeClaw users: {_body(resp)}")
    return [u.get("name") for u in resp.json().get("users", [])]


def create_user(session, url: str, name: str) -> bool:
    """Create a user. False if it was already there."""
    resp = session.post(f"{_base(url)}/api/users", json={"name": name},
                        timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code == 409:
        return False
    if resp.status_code != 200:
        raise FreeClawError(f"Could not create the '{name}' user: {_body(resp)}")
    return True


def set_context(session, url: str, name: str, context: str) -> None:
    """Replace a user's context.md -- its persona and long-term memory.

    FreeClaw applies this to the conversation already running, so the persona
    takes effect on the next turn rather than the next reset. Doing it over
    HTTP rather than by writing the file is what allows a FreeClaw that is not
    on this machine.
    """
    resp = session.put(f"{_base(url)}/api/users/{name}/context",
                       json={"context": context}, timeout=(CONNECT_TIMEOUT, 60))
    if resp.status_code == 404:
        raise FreeClawError(f"FreeClaw has no user called '{name}'.")
    if resp.status_code != 200:
        raise FreeClawError(f"Could not write the persona: {_body(resp)}")


def get_context(session, url: str, name: str) -> str:
    resp = session.get(f"{_base(url)}/api/users/{name}/context",
                       timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code != 200:
        raise FreeClawError(f"Could not read the persona: {_body(resp)}")
    return resp.json().get("context", "")


def mcp_servers(session, url: str) -> list[dict]:
    resp = session.get(f"{_base(url)}/api/mcp", timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code != 200:
        raise FreeClawError(f"Could not list MCP servers: {_body(resp)}")
    return resp.json().get("servers", [])


def add_mcp(session, url: str, name: str, tool_url: str, token: str) -> tuple[int, str | None]:
    """Register an HTTP MCP server. Returns (tool count, warning or None).

    FreeClaw dials the server straight away to count its tools, so a wrong
    address surfaces here rather than in the middle of a conversation.
    """
    resp = session.post(
        f"{_base(url)}/api/mcp",
        json={"name": name, "transport": "http", "url": tool_url, "token": token},
        timeout=(CONNECT_TIMEOUT, 120))
    if resp.status_code == 409:
        raise FreeClawError(f"An MCP server named '{name}' already exists.")
    if resp.status_code != 200:
        raise FreeClawError(f"Could not add the MCP server: {_body(resp)}")
    data = resp.json()
    return data.get("tool_count", 0), data.get("warning")


def remove_mcp(session, url: str, name: str) -> bool:
    resp = session.delete(f"{_base(url)}/api/mcp/{name}",
                          timeout=(CONNECT_TIMEOUT, 60))
    return resp.status_code == 200


def providers(session, url: str) -> list[dict]:
    """The LLM providers FreeClaw can call. Empty means it cannot answer."""
    resp = session.get(f"{_base(url)}/api/providers", timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code != 200:
        return []
    return resp.json().get("providers", [])


def api_enabled(session, url: str) -> bool:
    resp = session.get(f"{_base(url)}/api/api-status", timeout=(CONNECT_TIMEOUT, 30))
    return resp.status_code == 200 and bool(resp.json().get("enabled"))


def enable_api(session, url: str) -> None:
    """Switch on the OpenAI-compatible API. Without it, /v1 is closed."""
    resp = session.post(f"{_base(url)}/api/api-status", json={"enabled": True},
                        timeout=(CONNECT_TIMEOUT, 30))
    if resp.status_code != 200:
        raise FreeClawError(f"Could not switch FreeClaw's API on: {_body(resp)}")


# -- asking it something ---------------------------------------------------

def ask(url: str, password: str, user: str, message: str,
        timeout: float = READ_TIMEOUT) -> str:
    """Run one agent turn and return what it said."""
    url = _base(url)
    try:
        resp = requests.post(
            f"{url}/v1/chat/completions",
            headers={"Authorization": f"Bearer {password}",
                     "Content-Type": "application/json"},
            json={"model": user, "messages": [{"role": "user", "content": message}]},
            timeout=(CONNECT_TIMEOUT, timeout))
    except requests.RequestException as exc:
        raise _why(exc, url) from exc

    if resp.status_code == 401:
        raise FreeClawError("FreeClaw rejected the password.")
    if resp.status_code == 503:
        raise FreeClawError("FreeClaw's API is switched off. Re-run Add FreeClaw, "
                            "or type /startapi in its chat.")
    if resp.status_code == 404:
        raise FreeClawError(f"FreeClaw has no user called '{user}'. "
                            f"Re-run Add FreeClaw to create it.")
    if resp.status_code >= 400:
        raise FreeClawError(f"FreeClaw returned {resp.status_code}: {_body(resp)}")
    try:
        return resp.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise FreeClawError("FreeClaw sent back a reply I could not read.") from exc


# -- watching it think -----------------------------------------------------

# The event vocabulary `/chat` streams, copied from FreeClaw's src/agent.py as
# the contract this depends on. Nothing enforces it stays in step, so a
# FreeClaw update that renames one of these is the first place to look if the
# panel goes quiet.
#
#   intent    {tag}                  which kind of request it decided this is
#   provider  {name}                 which model answered
#   reasoning {text}                 a chunk of the model thinking out loud
#   token     {text}                 a chunk of the reply
#   tool_call {name, arguments}      a tool is starting
#   tool_result {name, result}       it finished
#   tool_throttled {name, limit}     held back, because it was looping
#   approval_request {id, command}   see below
#   stopped                          cancelled
#   done  {conversation, updated_at} terminal, success
#   error {error}                    terminal, failure


class Chat:
    """One FreeClaw user's conversation, narrated as it happens.

    `/v1/chat/completions` returns the finished reply and nothing else, which
    is fine for a caption but leaves a turn that spends thirty seconds running
    tools looking identical to one that has crashed.  `/chat` is the endpoint
    the web UI itself uses, and it streams every step -- which is what lets
    the overlay show tool calls and reasoning as they arrive.

    The catch is that `/chat` assumes a browser is listening and will answer
    approval prompts, and offers no way to say otherwise.  Nothing here can
    render one, and FreeClaw's own timeout is five minutes, so a turn that hit
    one would hang for exactly as long as the visibility this class exists to
    provide is meant to prevent.  Every `approval_request` is therefore denied
    the moment it arrives -- which is precisely what `/v1` does on its own for
    a caller with nobody to ask.  A command with a saved always-allow rule
    never reaches this, so it is only ever a fresh, unrecognised one that gets
    refused.

    One instance per user, reused across turns: logging in and selecting the
    conversation only has to happen once.  Not safe for concurrent turns --
    the caller serialises.
    """

    def __init__(self, url: str, password: str, user: str) -> None:
        self.url = _base(url)
        self.password = password
        self.user = user
        self._session = None

    def _login(self):
        session = admin(self.url, self.password)
        # Selecting the user is what makes /chat talk to the right
        # conversation; it is a session property, not a request parameter.
        resp = session.get(f"{self.url}/chat", params={"user": self.user},
                           timeout=(CONNECT_TIMEOUT, 30))
        if resp.status_code != 200:
            raise FreeClawError(
                f"Could not select the FreeClaw user '{self.user}' "
                f"(HTTP {resp.status_code}). Re-run Add FreeClaw.")
        self._session = session
        return session

    def _deny(self, session, event) -> None:
        request_id = event.get("id")
        if not request_id:
            return
        try:
            session.post(f"{self.url}/api/approval",
                         json={"id": request_id, "decision": "deny"},
                         timeout=(CONNECT_TIMEOUT, 15))
        except requests.RequestException:
            pass

    def reset(self) -> None:
        """Throw away the conversation and start a fresh one.

        FreeClaw keeps the history per user rather than per request, so this
        has to happen on its side -- clearing the overlay's own transcript
        only hides it, and the next turn would still arrive with everything
        that came before it in context.

        `/reset` acts on whichever user the session has selected, which
        `_login` already set to this one.
        """
        session = self._session or self._login()
        try:
            resp = session.post(f"{self.url}/reset",
                                timeout=(CONNECT_TIMEOUT, 30))
            if resp.status_code == 401:
                resp = self._login().post(f"{self.url}/reset",
                                          timeout=(CONNECT_TIMEOUT, 30))
        except requests.RequestException as exc:
            raise FreeClawError(f"Could not reach FreeClaw to reset ({exc}).") from None
        if resp.status_code >= 400:
            raise FreeClawError(
                f"FreeClaw would not reset the conversation "
                f"(HTTP {resp.status_code}).")

    def turn(self, message: str, timeout: float = READ_TIMEOUT):
        """Run one turn, yielding events. The last is always done or error."""
        session = self._session or self._login()
        resp = self._post(session, message, timeout)
        if resp.status_code == 401:
            # The cookie expired, or FreeClaw restarted with a new secret key.
            session = self._login()
            resp = self._post(session, message, timeout)
        if resp.status_code != 200:
            raise FreeClawError(f"FreeClaw returned {resp.status_code} "
                                f"starting the turn.")
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if not payload:
                continue
            try:
                event = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "approval_request":
                self._deny(session, event)
            yield event

    def _post(self, session, message: str, timeout: float):
        try:
            return session.post(f"{self.url}/chat", json={"message": message},
                                stream=True, timeout=(CONNECT_TIMEOUT, timeout))
        except requests.RequestException as exc:
            raise _why(exc, self.url) from exc
