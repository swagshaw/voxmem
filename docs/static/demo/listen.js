// "Hear the difference": real VoxMem items, played with full audio or reduced to their transcripts.
// Data: window.VOXMEM_DEMO from demo.js (built by promo/tools/build_demo.py).
(function () {
  var DATA = window.VOXMEM_DEMO;
  var root = document.getElementById("listen-demo");
  if (!DATA || !root) return;

  var EV = {
    speaker: {cls: "c-spk", label: "Speaker identity", tab: "Who said it", c: "var(--spk)"},
    paralinguistic: {cls: "c-par", label: "Paralinguistic cues", tab: "How it was said", c: "var(--par)"},
    environmental: {cls: "c-env", label: "Environmental sound", tab: "What was audible", c: "var(--env)"},
    semantic: {cls: "c-sem", label: "Speech semantics", tab: "What was said", c: "var(--sem)"}
  };
  var OPS = {IE: "Information extraction", MSR: "Multi-session reasoning", TET: "Temporal tracking", AR: "Answer refusal"};
  var VOICE_COLORS = ["#4A8FE7", "#E0679A", "#2BA59A", "#C08A1E"];

  var state = {ex: 0, mode: "audio", revealed: false};
  var audio = new Audio();
  audio.preload = "none";
  var playing = null;   // {clip, btn, canvas, words}
  var raf = 0;

  var tabs = root.querySelector(".ls-tabs");
  var modeSeg = root.querySelector(".ls-mode");
  var card = root.querySelector(".ls-card");

  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]; }); }
  function fmtTime(s) { return (s < 10 ? "0:0" : "0:") + Math.round(s); }

  // ---- tabs ----
  DATA.forEach(function (ex, i) {
    var ev = EV[ex.evidence_type];
    var b = document.createElement("button");
    b.type = "button"; b.setAttribute("role", "tab");
    b.className = "ls-tab " + ev.cls;
    b.innerHTML = '<span class="chip ' + ev.cls + '">' + ev.tab + "</span>" + (ex.evidence_type === "semantic" ? '<small>control</small>' : "");
    b.addEventListener("click", function () { select(i); });
    tabs.appendChild(b);
  });
  modeSeg.querySelectorAll("button").forEach(function (b) {
    b.addEventListener("click", function () { setMode(b.getAttribute("data-mode")); });
  });

  function select(i) {
    stop();
    state.ex = i; state.revealed = false;
    tabs.querySelectorAll(".ls-tab").forEach(function (b, k) { b.setAttribute("aria-selected", k === i ? "true" : "false"); });
    render();
  }
  function setMode(m) {
    if (m === "text") stop();
    state.mode = m;
    modeSeg.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", b.getAttribute("data-mode") === m ? "true" : "false"); });
    card.classList.toggle("text-mode", m === "text");
    markModel();
  }

  // ---- one clip row (evidence turn or question) ----
  function clipRow(clip, opts) {
    var words = clip.words && clip.words.length ? clip.words : clip.transcript.split(/\s+/).map(function (w) { return {w: w}; });
    var text = words.map(function (w, k) { return '<span data-k="' + k + '">' + esc(w.w) + "</span>"; }).join(" ");
    var spk = opts.voice ?
      '<span class="ls-spk" style="--v:' + opts.voice.color + '"><i></i><span class="a">' + esc(opts.voice.name) + '</span><span class="t">User</span></span>' : "";
    var cue = clip.cue ? '<span class="ls-cue">' + esc(clip.cue) + "</span>" : "";
    return '<div class="ls-row" data-clip="' + opts.key + '">' +
      '<button type="button" class="ls-play" aria-label="Play this turn"><svg viewBox="0 0 24 24"><path class="p" d="M8 5v14l11-7z"/><path class="s" d="M7 5h4v14H7zM13 5h4v14h-4z"/></svg></button>' +
      '<div class="ls-wave"><canvas></canvas><span class="ls-dur">' + fmtTime(clip.duration) + '</span>' +
      '<div class="ls-removed">audio removed: transcript only</div></div>' + cue +
      '</div><p class="ls-text">' + spk + text + "</p>";
  }

  function render() {
    var ex = DATA[state.ex], ev = EV[ex.evidence_type];
    card.style.setProperty("--c", ev.c);
    // distinct voices get distinct names/colours in audio mode, and all collapse to "User" in transcript mode
    var voices = {}, vn = 0;
    function voiceOf(name) {
      if (!name) return null;
      if (!voices[name]) { voices[name] = {name: "Voice " + String.fromCharCode(65 + vn), color: VOICE_COLORS[vn % VOICE_COLORS.length]}; vn++; }
      return voices[name];
    }
    var clips = {};
    var ticks = "";
    var evid = {}; ex.turns.forEach(function (t) { evid[t.session_index] = t.role; });
    var roles = ex.history.session_roles || [];
    for (var s = 0; s < ex.history.n_sessions; s++) {
      var role = roles[s] === "evidence" ? "evidence" : /haystack/.test(roles[s] || "") ? "distractor" : (evid[s] || "");
      ticks += '<span class="ls-tick ' + role + '"' + (evid[s] ? ' data-s="' + s + '"' : "") + ' title="Session ' + (s + 1) + (role ? " · " + (role === "evidence" ? "evidence" : "same-topic distractor") : "") + '"></span>';
    }
    ticks += '<span class="ls-tick q" title="Question"></span>';

    var turns = ex.turns.map(function (t, k) {
      var key = "t" + k; clips[key] = t;
      return '<div class="ls-turn ' + t.role + '" id="ls-s' + t.session_index + '">' +
        '<div class="ls-when"><b>Session ' + (t.session_index + 1) + "</b> · " + esc(t.timestamp || "") +
        (t.role === "distractor" ? ' <span class="ls-role">same-topic distractor</span>' : "") + "</div>" +
        clipRow(t, {key: key, voice: voiceOf(t.speaker)}) +
        (t.assistant_reply ? '<p class="ls-reply"><span>Assistant</span>' + esc(t.assistant_reply) + "</p>" : "") +
        "</div>";
    }).join("");
    clips.q = ex.question;

    var m = ex.model;
    function ans(cond, label, cls) {
      if (!cond) return "";
      return '<div class="ls-model ' + cls + '"><div class="ls-mh">' + esc(m.name) + " · " + label +
        '<span class="ls-verdict ' + (cond.correct ? "ok" : "no") + '">' + (cond.correct ? "correct" : "wrong") + "</span></div>" +
        "<p>" + esc(cond.answer) + (cond.truncated ? ' <span class="ls-cut">[cut off at the output-token limit]</span>' : "") + "</p></div>";
    }
    card.innerHTML =
      '<div class="ls-head"><span class="chip ' + ev.cls + '">' + ev.label + '</span><span class="ls-op">' + OPS[ex.operation] + "</span>" +
      '<span class="ls-meta">8K history · ' + ex.history.n_sessions + " sessions</span></div>" +
      '<div class="ls-track" aria-hidden="true">' + ticks + "</div>" +
      '<div class="ls-turns">' + turns + "</div>" +
      '<div class="ls-q"><div class="ls-when"><b>The question</b> · asked after the whole history</div>' +
      clipRow(ex.question, {key: "q", voice: voiceOf(ex.question.speaker)}) +
      '<div class="ls-reveal-row"><button type="button" class="btn ls-reveal">Reveal the answer</button>' +
      '<span class="ls-hint">Try answering it first, in both modes.</span></div>' +
      '<div class="ls-answers" hidden><div class="ls-gold"><span>Gold answer</span><b>' + esc(ex.gold_display) + "</b></div>" +
      '<div class="ls-models">' + ans(m.condition_full_audio, "full audio", "a") + ans(m.condition_transcript, "transcript only", "t") + "</div></div></div>";

    card.querySelectorAll(".ls-row").forEach(function (row) {
      var clip = clips[row.getAttribute("data-clip")];
      var cv = row.querySelector("canvas");
      draw(cv, clip.peaks, 0);
      row.querySelector(".ls-play").addEventListener("click", function () { toggle(clip, row); });
      row.querySelector(".ls-wave").addEventListener("click", function (e) {
        if (state.mode !== "audio") return;
        var r = cv.getBoundingClientRect(), f = (e.clientX - r.left) / r.width;
        toggle(clip, row, Math.max(0, Math.min(1, f)) * clip.duration);
      });
    });
    card.querySelectorAll(".ls-tick[data-s]").forEach(function (tk) {
      tk.addEventListener("click", function () {
        var el = document.getElementById("ls-s" + tk.getAttribute("data-s"));
        if (!el) return;
        el.scrollIntoView({behavior: "smooth", block: "center"});
        el.classList.remove("flash"); void el.offsetWidth; el.classList.add("flash");
      });
    });
    card.querySelector(".ls-reveal").addEventListener("click", function () {
      state.revealed = true;
      card.querySelector(".ls-answers").hidden = false;
      this.parentNode.hidden = true;
      markModel();
    });
    markModel();
  }

  function markModel() {
    card.querySelectorAll(".ls-model").forEach(function (el) {
      el.classList.toggle("current", (state.mode === "audio") === el.classList.contains("a"));
    });
  }

  // ---- waveform ----
  function draw(cv, peaks, prog) {
    var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (!w) return;
    if (cv.width !== Math.round(w * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    var g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, w, h);
    var n = peaks.length, bw = w / n, col = getComputedStyle(card).getPropertyValue("--c").trim() || "#2D4A8A";
    var on = resolve(col);
    for (var i = 0; i < n; i++) {
      var bh = Math.max(2, peaks[i] * (h - 4));
      g.fillStyle = (i + 0.5) / n <= prog ? on : "#C9D0DE";
      roundRect(g, i * bw + bw * 0.18, (h - bh) / 2, Math.max(1, bw * 0.64), bh, Math.min(2, bw * 0.3));
    }
  }
  var resolved = {};
  function resolve(c) {   // canvas cannot read var(--x); resolve through a probe element
    if (c.indexOf("var(") !== 0) return c;
    if (resolved[c]) return resolved[c];
    var p = document.createElement("span"); p.style.color = c; document.body.appendChild(p);
    var v = getComputedStyle(p).color; p.remove(); return (resolved[c] = v);
  }
  function roundRect(g, x, y, w, h, r) {
    g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
    g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath(); g.fill();
  }

  // ---- playback ----
  function toggle(clip, row, at) {
    if (state.mode !== "audio") return;
    if (playing && playing.row === row && at === undefined) { stop(); return; }
    stop();
    audio.src = clip.audio;
    audio.currentTime = at || 0;
    playing = {clip: clip, row: row, cv: row.querySelector("canvas"), words: row.nextElementSibling.querySelectorAll("[data-k]")};
    row.classList.add("playing");
    audio.play().catch(function () { stop(); });
    tick();
  }
  function stop() {
    if (!playing) return;
    audio.pause();
    cancelAnimationFrame(raf);
    playing.row.classList.remove("playing");
    draw(playing.cv, playing.clip.peaks, 0);
    playing.words.forEach(function (w) { w.classList.remove("on"); });
    playing = null;
  }
  function tick() {
    if (!playing) return;
    var c = playing.clip, t = audio.currentTime;
    draw(playing.cv, c.peaks, t / c.duration);
    if (c.words) playing.words.forEach(function (w, k) { var wd = c.words[k]; w.classList.toggle("on", wd && wd.t0 !== undefined && t >= wd.t0); });
    raf = requestAnimationFrame(tick);
  }
  audio.addEventListener("ended", stop);
  window.addEventListener("resize", function () {
    card.querySelectorAll(".ls-row").forEach(function (row) { /* redraw at new width */ row.querySelector("canvas").width = 0; });
    var ex = DATA[state.ex];
    card.querySelectorAll(".ls-row").forEach(function (row) {
      var k = row.getAttribute("data-clip"), clip = k === "q" ? ex.question : ex.turns[+k.slice(1)];
      draw(row.querySelector("canvas"), clip.peaks, 0);
    });
  });

  select(0);
  setMode("audio");
})();
