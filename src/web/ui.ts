/**
 * The browser chat client, as a single self-contained page.
 *
 * It is a string rather than a `.html` file on disk because the build is plain
 * `tsc` - it emits JavaScript and copies nothing else, so a separate asset
 * would be missing from `dist/` and the daemon would 404 in exactly the
 * situation where it matters. No framework, no bundler, no build step: the
 * whole client is below, and it talks to the same daemon the editor does.
 *
 * Deliberately no backticks or `${}` in the embedded script, so this template
 * literal needs no escaping and stays readable.
 */

export const CHAT_HTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>pca chat</title>
<style>
  :root {
    color-scheme: light dark;
    --bg: #ffffff; --fg: #17181a; --muted: #6b7076; --line: #e4e6e9;
    --panel: #f7f8f9; --accent: #4a5bd4; --local: #1a7f52; --claude: #8b4bc4;
    --code: #f2f3f5;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #17181a; --fg: #e8e9ea; --muted: #9a9ea3; --line: #2b2d30;
      --panel: #1e1f22; --accent: #8b97ee; --local: #4ec48a; --claude: #c08ae8;
      --code: #232528;
    }
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--fg);
    font: 15px/1.6 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    display: grid; grid-template-columns: 250px 1fr; height: 100vh;
  }
  aside {
    border-right: 1px solid var(--line); background: var(--panel);
    padding: 20px 18px; overflow-y: auto; font-size: 13px;
  }
  aside h1 { font-size: 15px; margin: 0 0 2px; letter-spacing: -0.01em; }
  aside .sub { color: var(--muted); font-size: 12px; margin-bottom: 20px; }
  aside h2 {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin: 22px 0 8px; font-weight: 600;
  }
  .stat { display: flex; justify-content: space-between; gap: 10px; padding: 3px 0; }
  .stat span:last-child { color: var(--muted); font-variant-numeric: tabular-nums; }
  .wbar { height: 3px; background: var(--accent); border-radius: 2px; opacity: .55; }
  button {
    font: inherit; font-size: 13px; padding: 7px 12px; width: 100%;
    border: 1px solid var(--line); border-radius: 7px; background: var(--bg);
    color: var(--fg); cursor: pointer; margin-top: 8px;
  }
  button:hover { border-color: var(--accent); }
  button:disabled { opacity: .5; cursor: default; }
  main { display: flex; flex-direction: column; min-width: 0; }
  #log { flex: 1; overflow-y: auto; padding: 28px 32px 8px; }
  .turn { max-width: 780px; margin: 0 auto 26px; }
  .who {
    font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--muted); margin-bottom: 6px; display: flex; gap: 8px; align-items: center;
  }
  .badge {
    font-size: 10px; padding: 1px 6px; border-radius: 20px;
    border: 1px solid currentColor; text-transform: none; letter-spacing: 0;
  }
  .badge.local { color: var(--local); }
  .badge.claude { color: var(--claude); }
  .body { white-space: pre-wrap; word-wrap: break-word; }
  .body code {
    background: var(--code); padding: 1px 5px; border-radius: 4px;
    font: 13px ui-monospace, "SF Mono", Menlo, Consolas, monospace;
  }
  .body pre {
    background: var(--code); padding: 12px 14px; border-radius: 8px;
    overflow-x: auto; margin: 10px 0;
  }
  .body pre code { background: none; padding: 0; }
  .body strong { font-weight: 600; }
  .body em { color: var(--muted); font-style: normal; font-size: 13px; }
  .thinking {
    color: var(--muted); font-size: 13px; border-left: 2px solid var(--line);
    padding-left: 12px; margin-bottom: 10px; white-space: pre-wrap;
  }
  form {
    border-top: 1px solid var(--line); padding: 16px 32px 22px;
    display: flex; gap: 10px; max-width: 844px; margin: 0 auto; width: 100%;
  }
  textarea {
    flex: 1; resize: none; font: inherit; padding: 11px 14px; border-radius: 10px;
    border: 1px solid var(--line); background: var(--bg); color: var(--fg);
    min-height: 46px; max-height: 200px;
  }
  textarea:focus { outline: none; border-color: var(--accent); }
  form button { width: auto; align-self: flex-end; padding: 11px 18px; }
  .hint { color: var(--muted); font-size: 12px; text-align: center; padding-bottom: 14px; }
  .err { color: #d4483b; }
</style>
</head>
<body>
<aside>
  <h1>pca chat</h1>
  <div class="sub">local models first, Claude when it matters</div>

  <h2>Trained</h2>
  <div id="trained"></div>
  <button id="train">Retrain on code + memory</button>

  <h2>Router</h2>
  <div id="router"></div>

  <h2>Session</h2>
  <div id="session"></div>
  <button id="reset">New conversation</button>
</aside>
<main>
  <div id="log"></div>
  <div class="hint" id="hint"></div>
  <form id="form">
    <textarea id="input" placeholder="Ask about this codebase. Prefix with /local or /claude to force a route." rows="1"></textarea>
    <button type="submit" id="send">Send</button>
  </form>
</main>
<script>
(function () {
  var token = new URLSearchParams(location.search).get("token") || "";
  var sessionId = "web-" + Date.now().toString(36);
  var log = document.getElementById("log");
  var input = document.getElementById("input");
  var form = document.getElementById("form");
  var sendBtn = document.getElementById("send");
  var busy = false;

  function headers(extra) {
    var h = extra || {};
    if (token) h["authorization"] = "Bearer " + token;
    return h;
  }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined) n.textContent = text;
    return n;
  }

  function escapeHtml(s) {
    return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // Just enough markdown for what the two engines actually emit.
  function render(md) {
    var html = escapeHtml(md);
    html = html.replace(/\`\`\`([a-z]*)\\n([\\s\\S]*?)\`\`\`/g, function (_m, _lang, body) {
      return "<pre><code>" + body + "</code></pre>";
    });
    html = html.replace(/\`([^\`\\n]+)\`/g, "<code>$1</code>");
    html = html.replace(/\\*\\*([^*\\n]+)\\*\\*/g, "<strong>$1</strong>");
    html = html.replace(/_([^_\\n]{3,})_/g, "<em>$1</em>");
    return html;
  }

  function addTurn(who, route) {
    var turn = el("div", "turn");
    var head = el("div", "who");
    head.appendChild(el("span", null, who));
    if (route) head.appendChild(el("span", "badge " + route, route));
    turn.appendChild(head);
    var body = el("div", "body");
    turn.appendChild(body);
    log.appendChild(turn);
    log.scrollTop = log.scrollHeight;
    return { turn: turn, head: head, body: body, raw: "" };
  }

  function setBusy(v) {
    busy = v;
    sendBtn.disabled = v;
    sendBtn.textContent = v ? "..." : "Send";
  }

  async function refresh() {
    try {
      var res = await fetch("/stats", { headers: headers() });
      var s = await res.json();
      var m = s.manifest;
      var trained = document.getElementById("trained");
      trained.innerHTML = "";
      var rows = [
        ["corpus", m ? m.corpusTokens.toLocaleString() + " tok" : "not trained"],
        ["files", m ? String(m.corpusFiles) : "0"],
        ["vocab", m ? String(m.vocabSize) : "0"],
        ["chunks", s.index ? s.index.chunks.toLocaleString() : "0"],
        ["memory", String(s.memory ? s.memory.lessons : 0) + " answers"],
        ["neural", m && m.neural ? "ppl " + m.neural.valPerplexity.toFixed(1) : "off"]
      ];
      rows.forEach(function (r) {
        var d = el("div", "stat");
        d.appendChild(el("span", null, r[0]));
        d.appendChild(el("span", null, r[1]));
        trained.appendChild(d);
      });

      var router = document.getElementById("router");
      router.innerHTML = "";
      var rs = s.router || { updates: 0, weights: [] };
      var head = el("div", "stat");
      head.appendChild(el("span", null, "updates"));
      head.appendChild(el("span", null, String(rs.updates)));
      router.appendChild(head);
      var max = 0;
      rs.weights.forEach(function (w) { max = Math.max(max, Math.abs(w.weight)); });
      rs.weights.slice(0, 5).forEach(function (w) {
        var d = el("div", "stat");
        d.appendChild(el("span", null, w.feature));
        d.appendChild(el("span", null, w.weight.toFixed(2)));
        router.appendChild(d);
        var bar = el("div", "wbar");
        bar.style.width = max > 0 ? Math.round((Math.abs(w.weight) / max) * 100) + "%" : "0";
        router.appendChild(bar);
      });
      document.getElementById("hint").textContent = s.claudeAvailable
        ? "Enter to send, Shift+Enter for a newline."
        : "No ANTHROPIC_API_KEY - local models only.";
    } catch (e) {
      document.getElementById("hint").textContent = "Daemon unreachable: " + e.message;
    }
  }

  function updateSession(cost, turns) {
    var s = document.getElementById("session");
    s.innerHTML = "";
    var rows = [["turns", String(turns)], ["cost", "$" + cost.toFixed(4)]];
    rows.forEach(function (r) {
      var d = el("div", "stat");
      d.appendChild(el("span", null, r[0]));
      d.appendChild(el("span", null, r[1]));
      s.appendChild(d);
    });
  }

  var turnCount = 0;
  var totalCost = 0;

  async function send(text) {
    var force = null;
    if (text.indexOf("/local ") === 0) { force = "local"; text = text.slice(7); }
    else if (text.indexOf("/claude ") === 0) { force = "claude"; text = text.slice(8); }

    addTurn("you").body.textContent = text;
    setBusy(true);
    var out = addTurn("assistant");
    var thinkingEl = null;

    try {
      var res = await fetch("/chat", {
        method: "POST",
        headers: headers({ "content-type": "application/json" }),
        body: JSON.stringify({ message: text, force: force, sessionId: sessionId })
      });
      if (!res.ok || !res.body) {
        var msg = await res.text();
        out.body.className = "body err";
        out.body.textContent = "Error " + res.status + ": " + msg;
        return;
      }
      var reader = res.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      for (;;) {
        var chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        var parts = buffer.split("\\n\\n");
        buffer = parts.pop();
        for (var i = 0; i < parts.length; i++) {
          var line = parts[i];
          if (line.indexOf("data: ") !== 0) continue;
          var ev = JSON.parse(line.slice(6));
          if (ev.type === "route") {
            out.head.appendChild(el("span", "badge " + ev.route, ev.route));
            out.head.appendChild(el("span", null, ev.why));
          } else if (ev.type === "thinking") {
            if (!thinkingEl) {
              thinkingEl = el("div", "thinking");
              out.turn.insertBefore(thinkingEl, out.body);
            }
            thinkingEl.textContent += ev.text;
          } else if (ev.type === "text") {
            out.raw += ev.text;
            out.body.innerHTML = render(out.raw);
          } else if (ev.type === "done") {
            if (ev.text && !out.raw) {
              out.raw = ev.text;
              out.body.innerHTML = render(out.raw);
            }
            totalCost += ev.costUsd || 0;
            turnCount += 1;
            updateSession(totalCost, turnCount);
            if (ev.lessonId) refresh();
          } else if (ev.type === "error") {
            out.body.className = "body err";
            out.body.textContent = ev.message;
          }
          log.scrollTop = log.scrollHeight;
        }
      }
    } catch (e) {
      out.body.className = "body err";
      out.body.textContent = String(e);
    } finally {
      setBusy(false);
      input.focus();
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    var text = input.value.trim();
    if (!text || busy) return;
    input.value = "";
    input.style.height = "auto";
    send(text);
  });

  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.dispatchEvent(new Event("submit"));
    }
  });
  input.addEventListener("input", function () {
    input.style.height = "auto";
    input.style.height = Math.min(200, input.scrollHeight) + "px";
  });

  document.getElementById("train").addEventListener("click", async function () {
    var btn = this;
    btn.disabled = true;
    btn.textContent = "Training...";
    try {
      var res = await fetch("/train", { method: "POST", headers: headers({ "content-type": "application/json" }), body: "{}" });
      var r = await res.json();
      btn.textContent = r.error ? "Failed" : "Trained in " + r.seconds.toFixed(1) + "s";
    } catch (e) {
      btn.textContent = "Failed";
    }
    await refresh();
    setTimeout(function () {
      btn.disabled = false;
      btn.textContent = "Retrain on code + memory";
    }, 2500);
  });

  document.getElementById("reset").addEventListener("click", function () {
    sessionId = "web-" + Date.now().toString(36);
    log.innerHTML = "";
    turnCount = 0;
    totalCost = 0;
    updateSession(0, 0);
    input.focus();
  });

  updateSession(0, 0);
  refresh();
  input.focus();
})();
</script>
</body>
</html>
`;
