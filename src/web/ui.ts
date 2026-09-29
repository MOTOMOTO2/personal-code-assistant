/**
 * The chat client, as one self-contained page you can open on a phone.
 *
 * It is a string rather than a `.html` file on disk because the build is plain
 * `tsc` - it emits JavaScript and copies nothing else, so a separate asset would
 * be missing from `dist/` and the daemon would 404 in exactly the situation where
 * it matters. No framework, no bundler, no build step.
 *
 * What makes it a messaging client rather than a web page with a text box:
 *
 *   - Conversations are server-side and listed in a drawer, so the thread is
 *     still there tomorrow, and the one you started on your phone is the one you
 *     continue at your desk.
 *   - The server owns the turn. This page attaches to a conversation's event
 *     stream and reattaches when the tab wakes up; an answer that finished while
 *     the screen was off is simply there in the next snapshot.
 *   - Nothing is rendered optimistically. Every message shown came back from the
 *     server, which is why two devices in the same conversation agree.
 *
 * Deliberately no backticks or `${}` inside the embedded script, so the template
 * literals below need no escaping and stay readable.
 */

const STYLE = `
:root {
  color-scheme: light dark;
  --bg: #ffffff; --fg: #17181a; --muted: #6b7076; --line: #e4e6e9;
  --panel: #f7f8f9; --panel2: #eef0f2; --accent: #4a5bd4; --accent-fg: #ffffff;
  --local: #1a7f52; --claude: #8b4bc4; --code: #f2f3f5; --err: #c8372b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #17181a; --fg: #e8e9ea; --muted: #9a9ea3; --line: #2b2d30;
    --panel: #1e1f22; --panel2: #26282b; --accent: #5b6ae0; --accent-fg: #ffffff;
    --local: #4ec48a; --claude: #c08ae8; --code: #232528; --err: #ef6a5c;
  }
}
* { box-sizing: border-box; -webkit-tap-highlight-color: transparent; }
html, body { height: 100%; margin: 0; }
body {
  background: var(--bg); color: var(--fg); display: flex; overflow: hidden;
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}

/* Drawer: off-canvas on a phone, a fixed sidebar from tablet width up. */
#scrim {
  position: fixed; inset: 0; background: rgba(0,0,0,.45); z-index: 15;
  opacity: 0; pointer-events: none; transition: opacity .18s ease;
}
body.drawer #scrim { opacity: 1; pointer-events: auto; }
aside {
  position: fixed; top: 0; bottom: 0; left: 0; width: 284px; z-index: 16;
  background: var(--panel); border-right: 1px solid var(--line);
  display: flex; flex-direction: column;
  transform: translateX(-102%); transition: transform .2s ease;
  padding-bottom: env(safe-area-inset-bottom);
}
body.drawer aside { transform: none; }
aside header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 14px 14px 10px; gap: 8px;
}
aside h1 { font-size: 15px; margin: 0; letter-spacing: -.01em; }
aside .sub { font-size: 11px; color: var(--muted); }
#threads { flex: 1; overflow-y: auto; padding: 4px 8px 8px; }
.thread {
  display: block; width: 100%; text-align: left; border: 0; background: none;
  color: inherit; font: inherit; padding: 9px 10px; border-radius: 9px;
  cursor: pointer; margin-bottom: 2px;
}
.thread:hover { background: var(--panel2); }
.thread.on { background: var(--panel2); }
.thread .t { font-size: 13.5px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.thread .p {
  font-size: 11.5px; color: var(--muted); white-space: nowrap; overflow: hidden;
  text-overflow: ellipsis; margin-top: 1px;
}
.thread .row { display: flex; gap: 6px; align-items: baseline; justify-content: space-between; }
.thread .when { font-size: 10.5px; color: var(--muted); flex: none; }
.thread .acts { display: none; gap: 6px; margin-top: 6px; }
.thread.on .acts { display: flex; }
.thread .acts button {
  font: inherit; font-size: 11px; padding: 2px 8px; border-radius: 6px;
  border: 1px solid var(--line); background: var(--bg); color: var(--muted); cursor: pointer;
}
#facts { border-top: 1px solid var(--line); padding: 10px 14px 14px; font-size: 12px; }
.stat { display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; }
.stat span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
button.wide {
  font: inherit; font-size: 12.5px; padding: 7px 12px; width: 100%; margin-top: 8px;
  border: 1px solid var(--line); border-radius: 8px; background: var(--bg);
  color: var(--fg); cursor: pointer;
}
button.wide:hover { border-color: var(--accent); }
button.wide:disabled { opacity: .55; cursor: default; }

main { flex: 1; min-width: 0; display: flex; flex-direction: column; height: 100dvh; }
main > header {
  display: flex; align-items: center; gap: 10px; padding: 10px 12px;
  padding-top: calc(10px + env(safe-area-inset-top));
  border-bottom: 1px solid var(--line); background: var(--bg);
}
#menu {
  font: inherit; font-size: 18px; line-height: 1; padding: 6px 9px; cursor: pointer;
  background: none; border: 1px solid var(--line); border-radius: 8px; color: var(--fg);
}
#title { font-size: 14px; font-weight: 600; flex: 1; min-width: 0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
#conn { font-size: 11px; color: var(--muted); display: flex; align-items: center; gap: 5px; flex: none; }
#dot { width: 7px; height: 7px; border-radius: 50%; background: var(--muted); }
#conn.live #dot { background: var(--local); }
#conn.off #dot { background: var(--err); }

#log { flex: 1; overflow-y: auto; padding: 16px 12px 6px; -webkit-overflow-scrolling: touch; }
.msg { max-width: 780px; margin: 0 auto 14px; display: flex; flex-direction: column; }
.msg.you { align-items: flex-end; }
.bubble {
  max-width: 88%; padding: 9px 13px; border-radius: 17px;
  white-space: pre-wrap; word-wrap: break-word; overflow-wrap: anywhere;
}
.msg.you .bubble { background: var(--accent); color: var(--accent-fg); border-bottom-right-radius: 5px; }
.msg.ai .bubble { background: var(--panel); border: 1px solid var(--line); border-bottom-left-radius: 5px; }
.msg.pend .bubble { opacity: .75; }
.bubble code {
  background: var(--code); padding: 1px 5px; border-radius: 4px;
  font: 13px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
}
.msg.you .bubble code { background: rgba(255,255,255,.18); }
.bubble pre {
  background: var(--code); padding: 10px 12px; border-radius: 9px; overflow-x: auto;
  margin: 8px 0; white-space: pre;
}
.bubble pre code { background: none; padding: 0; }
.bubble a { color: inherit; }
.cite {
  font: 12.5px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  background: var(--code); border-radius: 4px; padding: 0 4px;
}
.meta {
  display: flex; flex-wrap: wrap; align-items: center; gap: 7px;
  font-size: 11px; color: var(--muted); margin-top: 5px; padding: 0 4px;
}
.badge {
  font-size: 10px; padding: 1px 6px; border-radius: 20px; border: 1px solid currentColor;
}
.badge.local { color: var(--local); }
.badge.claude { color: var(--claude); }
.meta button {
  font: inherit; font-size: 11px; padding: 2px 8px; border-radius: 20px; cursor: pointer;
  border: 1px solid var(--line); background: var(--bg); color: var(--fg);
}
.thinking {
  font-size: 12.5px; color: var(--muted); border-left: 2px solid var(--line);
  padding-left: 10px; margin: 0 0 7px; white-space: pre-wrap;
  max-height: 8.5em; overflow-y: auto;
}
.typing { display: inline-flex; gap: 3px; align-items: center; height: 1.2em; }
.typing i {
  width: 5px; height: 5px; border-radius: 50%; background: var(--muted);
  animation: blink 1.2s infinite ease-in-out;
}
.typing i:nth-child(2) { animation-delay: .18s; }
.typing i:nth-child(3) { animation-delay: .36s; }
@keyframes blink { 0%, 60%, 100% { opacity: .25 } 30% { opacity: 1 } }
.empty { color: var(--muted); text-align: center; max-width: 420px; margin: 18vh auto 0; font-size: 14px; }
.empty b { color: var(--fg); font-weight: 600; }
.err { color: var(--err); }
#banner {
  display: none; font-size: 12px; padding: 7px 12px; background: var(--panel);
  border-bottom: 1px solid var(--line); color: var(--muted);
}
#banner.on { display: block; }

form {
  display: flex; gap: 8px; align-items: flex-end; border-top: 1px solid var(--line);
  padding: 9px 12px; padding-bottom: calc(9px + env(safe-area-inset-bottom));
  background: var(--bg); max-width: 806px; margin: 0 auto; width: 100%;
}
textarea {
  flex: 1; resize: none; font: inherit; font-size: 16px; line-height: 1.4;
  padding: 9px 14px; border-radius: 19px; border: 1px solid var(--line);
  background: var(--bg); color: var(--fg); min-height: 40px; max-height: 38vh;
}
textarea:focus { outline: none; border-color: var(--accent); }
#send {
  flex: none; width: 40px; height: 40px; border-radius: 50%; border: 0; cursor: pointer;
  background: var(--accent); color: var(--accent-fg); font-size: 17px; line-height: 1;
}
#send:disabled { opacity: .45; cursor: default; }
.hint { font-size: 11.5px; color: var(--muted); text-align: center; padding: 0 12px 8px; }

@media (min-width: 860px) {
  aside { position: static; transform: none; }
  #scrim, #menu { display: none; }
  #log { padding: 22px 24px 8px; }
}
`;

const SCRIPT = `
(function () {
  var params = new URLSearchParams(location.search);
  var token = params.get("token") || localStorage.getItem("pca.token") || "";
  if (params.get("token")) {
    // Keep it for the next visit, so an installed home-screen shortcut that
    // starts at "/" still authenticates. The server also sets a cookie.
    localStorage.setItem("pca.token", params.get("token"));
  }

  var state = {
    threads: [],
    id: null,
    messages: [],
    pending: null,
    queued: [],
    claude: false,
    connection: "connecting",
    sending: false
  };
  var es = null;
  var nodes = {};
  var pendingNode = null;

  var log = document.getElementById("log");
  var input = document.getElementById("input");
  var form = document.getElementById("form");
  var sendBtn = document.getElementById("send");

  function api(path) {
    return path + (token ? (path.indexOf("?") < 0 ? "?" : "&") + "token=" + encodeURIComponent(token) : "");
  }
  function headers(extra) {
    var h = extra || {};
    if (token) h["authorization"] = "Bearer " + token;
    return h;
  }
  async function getJson(path) {
    var res = await fetch(api(path), { headers: headers() });
    if (!res.ok) throw new Error("HTTP " + res.status);
    return res.json();
  }
  async function postJson(path, body, method) {
    var res = await fetch(api(path), {
      method: method || "POST",
      headers: headers({ "content-type": "application/json" }),
      body: JSON.stringify(body || {})
    });
    if (!res.ok) {
      var detail = "";
      try { detail = (await res.json()).error || ""; } catch (e) {}
      throw new Error(detail || "HTTP " + res.status);
    }
    return res.json();
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }
  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  }

  // Just enough markdown for what the two engines actually emit, plus the
  // path:line citations both of them produce.
  function render(md) {
    var html = escapeHtml(md == null ? "" : md);
    html = html.replace(/\`\`\`([a-z]*)\\n([\\s\\S]*?)\`\`\`/g, function (_m, _lang, body) {
      return "<pre><code>" + body + "</code></pre>";
    });
    html = html.replace(/\`([^\`\\n]+)\`/g, "<code>$1</code>");
    html = html.replace(/\\*\\*([^*\\n]+)\\*\\*/g, "<strong>$1</strong>");
    html = html.replace(/(^|[\\s(])((?:[\\w./-]+)\\.[a-z]{1,8}:\\d+)/g, function (m, pre, cite) {
      return pre + "<span class=\\"cite\\">" + cite + "</span>";
    });
    html = html.replace(/(https?:\\/\\/[^\\s<)]+)/g, "<a href=\\"$1\\" rel=\\"noreferrer\\">$1</a>");
    return html;
  }

  function when(iso) {
    if (!iso) return "";
    var then = new Date(iso);
    var mins = Math.round((Date.now() - then.getTime()) / 60000);
    if (mins < 1) return "now";
    if (mins < 60) return mins + "m";
    if (mins < 24 * 60) return Math.round(mins / 60) + "h";
    var days = Math.round(mins / 1440);
    if (days < 7) return days + "d";
    return then.toISOString().slice(5, 10);
  }

  function setConnection(value) {
    state.connection = value;
    var box = document.getElementById("conn");
    box.className = value === "live" ? "live" : value === "offline" ? "off" : "";
    document.getElementById("connText").textContent =
      value === "live" ? "live" : value === "offline" ? "reconnecting" : "connecting";
  }

  function banner(text) {
    var box = document.getElementById("banner");
    box.textContent = text || "";
    box.className = text ? "on" : "";
  }

  /* ---------- conversations ---------- */

  function renderThreads() {
    var box = document.getElementById("threads");
    box.innerHTML = "";
    state.threads.forEach(function (thread) {
      var row = el("button", "thread" + (thread.id === state.id ? " on" : ""));
      var head = el("div", "row");
      head.appendChild(el("div", "t", thread.title || "New conversation"));
      head.appendChild(el("div", "when", when(thread.updatedAt)));
      row.appendChild(head);
      row.appendChild(el("div", "p", thread.preview || "no messages yet"));
      var acts = el("div", "acts");
      var rename = el("button", null, "Rename");
      rename.addEventListener("click", function (e) {
        e.stopPropagation();
        var title = prompt("Name this conversation", thread.title || "");
        if (title) {
          postJson("/api/threads/" + thread.id, { title: title }, "PATCH").then(refreshThreads);
        }
      });
      var remove = el("button", null, "Delete");
      remove.addEventListener("click", function (e) {
        e.stopPropagation();
        if (!confirm("Delete this conversation?")) return;
        postJson("/api/threads/" + thread.id, {}, "DELETE")
          .catch(function () {})
          .then(function () {
            if (thread.id === state.id) state.id = null;
            refreshThreads().then(function () {
              if (!state.id && state.threads.length > 0) open(state.threads[0].id);
              else if (!state.id) newThread();
            });
          });
      });
      acts.appendChild(rename);
      acts.appendChild(remove);
      row.appendChild(acts);
      row.addEventListener("click", function () {
        if (thread.id !== state.id) open(thread.id);
        closeDrawer();
      });
      box.appendChild(row);
    });
  }

  async function refreshThreads() {
    try {
      var data = await getJson("/api/threads");
      state.threads = data.threads || [];
      renderThreads();
    } catch (e) { /* the event stream will heal this */ }
  }

  async function newThread() {
    var data = await postJson("/api/threads", {});
    state.threads.unshift(data.thread);
    open(data.thread.id);
    closeDrawer();
    input.focus();
  }

  function open(id) {
    state.id = id;
    state.messages = [];
    state.pending = null;
    state.queued = [];
    localStorage.setItem("pca.thread", id);
    renderThreads();
    renderAll();
    connect();
  }

  /* ---------- the event stream ---------- */

  function connect() {
    if (es) { es.close(); es = null; }
    if (!state.id) return;
    setConnection("connecting");
    es = new EventSource(api("/api/threads/" + encodeURIComponent(state.id) + "/events"));
    es.onopen = function () { setConnection("live"); };
    es.onerror = function () { setConnection("offline"); };
    es.onmessage = function (e) {
      setConnection("live");
      var ev;
      try { ev = JSON.parse(e.data); } catch (err) { return; }
      handle(ev);
    };
  }

  function handle(ev) {
    if (ev.type === "snapshot") {
      var snap = ev.snapshot;
      state.messages = snap.messages || [];
      state.pending = snap.pending;
      state.queued = snap.queued || [];
      document.getElementById("title").textContent = snap.thread.title || "New conversation";
      upsertThread(snap.thread);
      renderAll();
      return;
    }
    if (ev.type === "user") {
      state.messages[ev.message.seq] = ev.message;
      renderAll();
      return;
    }
    if (ev.type === "queued") {
      state.queued.push({ id: ev.id, question: ev.question });
      renderAll();
      return;
    }
    if (ev.type === "route") {
      state.queued = state.queued.filter(function (q) { return q.id !== ev.id; });
      state.pending = state.pending || { id: ev.id, question: "", text: "", thinking: "" };
      state.pending.route = ev.route;
      state.pending.why = ev.why;
      state.pending.status = ev.route === "claude" ? "thinking" : "streaming";
      renderAll();
      return;
    }
    if (ev.type === "thinking" || ev.type === "text") {
      if (!state.pending) state.pending = { id: ev.id, question: "", text: "", thinking: "" };
      if (ev.type === "text") state.pending.text = (state.pending.text || "") + ev.text;
      else state.pending.thinking = (state.pending.thinking || "") + ev.text;
      state.pending.status = "streaming";
      paintPending();
      return;
    }
    if (ev.type === "done") {
      state.messages[ev.message.seq] = ev.message;
      state.pending = null;
      upsertThread(ev.thread);
      renderAll();
      if (ev.lessonId) refreshFacts();
      return;
    }
    if (ev.type === "error") {
      state.pending = null;
      state.messages.push({
        seq: state.messages.length, role: "assistant", content: ev.message,
        at: new Date().toISOString(), kind: "error"
      });
      renderAll();
      return;
    }
    if (ev.type === "thread") {
      upsertThread(ev.thread);
      document.getElementById("title").textContent = ev.thread.title || "New conversation";
    }
  }

  function upsertThread(thread) {
    if (!thread) return;
    var found = false;
    state.threads = state.threads.map(function (t) {
      if (t.id !== thread.id) return t;
      found = true;
      return thread;
    });
    if (!found) state.threads.unshift(thread);
    state.threads.sort(function (a, b) { return a.updatedAt < b.updatedAt ? 1 : -1; });
    renderThreads();
  }

  /* ---------- messages ---------- */

  function atBottom() {
    return log.scrollHeight - log.scrollTop - log.clientHeight < 90;
  }
  function toBottom(force) {
    if (force || atBottom()) log.scrollTop = log.scrollHeight;
  }

  function messageNode(msg, index) {
    var mine = msg.role === "user";
    var wrap = el("div", "msg " + (mine ? "you" : "ai"));
    var bubble = el("div", "bubble");
    if (msg.kind === "error") {
      bubble.className = "bubble err";
      bubble.textContent = msg.content;
    } else if (mine) {
      bubble.textContent = msg.content;
    } else {
      bubble.innerHTML = render(msg.content);
    }
    wrap.appendChild(bubble);

    if (!mine && msg.kind !== "error") {
      var meta = el("div", "meta");
      if (msg.route) meta.appendChild(el("span", "badge " + msg.route, msg.route));
      if (msg.why) meta.appendChild(el("span", null, msg.why));
      if (msg.costUsd) meta.appendChild(el("span", null, "$" + Number(msg.costUsd).toFixed(4)));
      if (msg.sources && msg.sources.length > 0) {
        meta.appendChild(el("span", "cite", msg.sources.slice(0, 3).join("  ")));
      }
      meta.appendChild(el("span", null, when(msg.at)));
      // Escalating the newest local answer is the one-tap version of /claude,
      // and it is also how the router learns it guessed wrong.
      var isLast = index === state.messages.length - 1;
      if (isLast && msg.route === "local" && state.claude && !state.pending) {
        var ask = el("button", null, "Ask Claude instead");
        ask.addEventListener("click", function () {
          ask.disabled = true;
          postJson("/api/threads/" + state.id + "/escalate", {}).catch(function (e) {
            banner("Could not ask Claude: " + e.message);
            ask.disabled = false;
          });
        });
        meta.appendChild(ask);
      }
      wrap.appendChild(meta);
    }
    return wrap;
  }

  function pendingBubble() {
    var wrap = el("div", "msg ai pend");
    if (state.pending && state.pending.thinking) {
      wrap.appendChild(el("div", "thinking", state.pending.thinking));
    }
    var bubble = el("div", "bubble");
    if (state.pending && state.pending.text) bubble.innerHTML = render(state.pending.text);
    else {
      var dots = el("span", "typing");
      dots.appendChild(el("i"));
      dots.appendChild(el("i"));
      dots.appendChild(el("i"));
      bubble.appendChild(dots);
    }
    wrap.appendChild(bubble);
    var meta = el("div", "meta");
    if (state.pending && state.pending.route) {
      meta.appendChild(el("span", "badge " + state.pending.route, state.pending.route));
    }
    if (state.pending && state.pending.why) meta.appendChild(el("span", null, state.pending.why));
    else meta.appendChild(el("span", null, "thinking"));
    wrap.appendChild(meta);
    return wrap;
  }

  function paintPending() {
    if (!pendingNode) { renderAll(); return; }
    var bubble = pendingNode.querySelector(".bubble");
    if (state.pending.text) bubble.innerHTML = render(state.pending.text);
    var think = pendingNode.querySelector(".thinking");
    if (state.pending.thinking) {
      if (!think) {
        think = el("div", "thinking");
        pendingNode.insertBefore(think, pendingNode.firstChild);
      }
      think.textContent = state.pending.thinking;
      think.scrollTop = think.scrollHeight;
    }
    toBottom(false);
  }

  function renderAll() {
    var stick = atBottom();
    log.innerHTML = "";
    pendingNode = null;
    if (state.messages.length === 0 && !state.pending && state.queued.length === 0) {
      var empty = el("div", "empty");
      empty.innerHTML = "<b>Message your own assistant.</b><br>" +
        "It answers from the models trained on this codebase, and hands the hard " +
        "questions to Claude - then remembers those answers so it can do them itself " +
        "next time.<br><br>Prefix a message with /local or /claude to choose for it.";
      log.appendChild(empty);
      return;
    }
    state.messages.forEach(function (msg, i) {
      if (msg) log.appendChild(messageNode(msg, i));
    });
    state.queued.forEach(function (q) {
      var wrap = el("div", "msg you pend");
      wrap.appendChild(el("div", "bubble", q.question));
      wrap.appendChild(el("div", "meta", "queued"));
      log.appendChild(wrap);
    });
    if (state.pending) {
      pendingNode = pendingBubble();
      log.appendChild(pendingNode);
    }
    toBottom(stick || state.pending !== null);
  }

  /* ---------- sending ---------- */

  async function send(text) {
    var force = null;
    if (text.indexOf("/local ") === 0) { force = "local"; text = text.slice(7); }
    else if (text.indexOf("/claude ") === 0) { force = "claude"; text = text.slice(8); }
    if (!text.trim()) return;
    if (!state.id) {
      await newThread();
    }
    state.sending = true;
    sendBtn.disabled = true;
    banner("");
    try {
      await postJson("/api/threads/" + state.id + "/messages", { text: text, force: force });
    } catch (e) {
      banner("Not sent: " + e.message + " - your message is back in the box.");
      input.value = text;
      resize();
    } finally {
      state.sending = false;
      sendBtn.disabled = false;
    }
  }

  function resize() {
    input.style.height = "auto";
    input.style.height = Math.min(window.innerHeight * 0.38, input.scrollHeight) + "px";
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value;
    if (!text.trim() || state.sending) return;
    input.value = "";
    resize();
    send(text);
  });

  // A physical keyboard sends on Enter; a touch keyboard must not, because
  // Enter is how you write a second line on a phone.
  var coarse = window.matchMedia("(pointer: coarse)").matches;
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey && !coarse) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });
  input.addEventListener("input", resize);

  /* ---------- drawer, facts, lifecycle ---------- */

  function openDrawer() { document.body.classList.add("drawer"); }
  function closeDrawer() { document.body.classList.remove("drawer"); }
  document.getElementById("menu").addEventListener("click", openDrawer);
  document.getElementById("scrim").addEventListener("click", closeDrawer);
  document.getElementById("new").addEventListener("click", function () { newThread(); });

  async function refreshFacts() {
    try {
      var s = await getJson("/api/state");
      state.claude = !!s.claudeAvailable;
      state.threads = s.threads || state.threads;
      var m = s.manifest;
      var rows = [
        ["corpus", m ? m.corpusTokens.toLocaleString() + " tok" : "not trained"],
        ["files", m ? String(m.corpusFiles) : "0"],
        ["chunks", s.index ? s.index.chunks.toLocaleString() : "0"],
        ["memory", String(s.memory ? s.memory.lessons : 0) + " answers"],
        ["router", String(s.router ? s.router.updates : 0) + " updates"],
        ["claude", s.claudeAvailable ? "available" : "no API key"]
      ];
      var box = document.getElementById("facts-rows");
      box.innerHTML = "";
      rows.forEach(function (r) {
        var d = el("div", "stat");
        d.appendChild(el("span", null, r[0]));
        d.appendChild(el("span", null, r[1]));
        box.appendChild(d);
      });
      document.getElementById("hint").textContent = s.claudeAvailable
        ? (coarse ? "Local models first, Claude when it matters." : "Enter sends, Shift+Enter for a newline.")
        : "No ANTHROPIC_API_KEY on the machine running pca - local models only.";
      renderThreads();
    } catch (e) {
      banner("Cannot reach pca: " + e.message);
    }
  }

  document.getElementById("train").addEventListener("click", async function () {
    var btn = this;
    btn.disabled = true;
    btn.textContent = "Training...";
    try {
      var r = await postJson("/train", {});
      btn.textContent = r.error ? "Failed" : "Trained in " + r.seconds.toFixed(1) + "s";
    } catch (e) {
      btn.textContent = "Failed";
    }
    await refreshFacts();
    setTimeout(function () {
      btn.disabled = false;
      btn.textContent = "Retrain on code + memory";
    }, 2500);
  });

  // Waking up is the normal case on a phone: the tab was frozen, the stream is
  // dead, and an answer probably finished while the screen was off. Pull a fresh
  // snapshot and reattach.
  function wake() {
    if (document.visibilityState !== "visible") return;
    if (state.id) {
      getJson("/api/threads/" + encodeURIComponent(state.id))
        .then(function (snap) { handle({ type: "snapshot", snapshot: snap }); })
        .catch(function () {});
    }
    if (!es || es.readyState === 2 || state.connection !== "live") connect();
    refreshThreads();
  }
  document.addEventListener("visibilitychange", wake);
  window.addEventListener("online", wake);
  window.addEventListener("focus", wake);
  if (window.visualViewport) {
    // The on-screen keyboard resizes the viewport rather than the window.
    window.visualViewport.addEventListener("resize", function () { toBottom(false); });
  }

  (async function start() {
    await refreshFacts();
    var wanted = localStorage.getItem("pca.thread");
    var exists = state.threads.some(function (t) { return t.id === wanted; });
    if (exists) open(wanted);
    else if (state.threads.length > 0) open(state.threads[0].id);
    else await newThread();
    if (!coarse) input.focus();
  })();
})();
`;

export const CHAT_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="theme-color" content="#17181a">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="pca">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="mobile-web-app-capable" content="yes">
<link rel="manifest" href="/manifest.webmanifest">
<link rel="icon" href="/icon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="/icon.svg">
<title>pca chat</title>
<style>${STYLE}</style>
</head>
<body>
<div id="scrim"></div>
<aside>
  <header>
    <div>
      <h1>pca</h1>
      <div class="sub">local models first, Claude when it matters</div>
    </div>
  </header>
  <div style="padding: 0 14px 8px"><button class="wide" id="new">New conversation</button></div>
  <div id="threads"></div>
  <div id="facts">
    <div id="facts-rows"></div>
    <button class="wide" id="train">Retrain on code + memory</button>
  </div>
</aside>
<main>
  <header>
    <button id="menu" aria-label="Conversations">&#9776;</button>
    <div id="title">pca chat</div>
    <div id="conn"><span id="dot"></span><span id="connText">connecting</span></div>
  </header>
  <div id="banner"></div>
  <div id="log"></div>
  <div class="hint" id="hint"></div>
  <form id="form">
    <textarea id="input" rows="1" placeholder="Message pca" autocapitalize="sentences"></textarea>
    <button type="submit" id="send" aria-label="Send">&#8593;</button>
  </form>
</main>
<script>${SCRIPT}</script>
</body>
</html>
`;

/** Icon for the installed app: the two halves of the assistant, joined. */
export const ICON_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 192 192">
  <rect width="192" height="192" rx="42" fill="#17181a"/>
  <path d="M40 132V60h16l20 44 20-44h16v72h-15V88l-17 38h-8L55 88v44z" fill="#e8e9ea"/>
  <circle cx="140" cy="68" r="12" fill="#4ec48a"/>
  <circle cx="140" cy="118" r="12" fill="#c08ae8"/>
</svg>
`;

/**
 * Enough for "Add to Home Screen" to install this as its own app. `start_url` has
 * no token in it on purpose: the page stores the token, and the daemon also sets
 * a cookie on the first authenticated load, so a bare launch still works.
 */
export const WEB_MANIFEST = {
  name: "pca chat",
  short_name: "pca",
  description: "Your own code assistant: local models first, Claude when it matters.",
  start_url: "/",
  scope: "/",
  display: "standalone",
  background_color: "#17181a",
  theme_color: "#17181a",
  icons: [
    { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
    { src: "/icon.svg", sizes: "any", type: "image/svg+xml", purpose: "maskable" },
  ],
};
