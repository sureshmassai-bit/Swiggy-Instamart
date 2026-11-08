/* Help chat: the bot answers first; "Not solved" turns the conversation into a ticket for a person. */
(function () {
  const panel = document.getElementById("chat-panel"); if (!panel) return;
  const fab = document.getElementById("chat-fab"), log = document.getElementById("chat-log"), actions = document.getElementById("chat-actions");
  const form = document.getElementById("chat-form"), input = document.getElementById("chat-input");
  const csrf = panel.dataset.csrf; let messages = [], turns = 0, busy = false, finished = false;
  const add = (who, text) => { messages.push({who, text}); const d = document.createElement("div"); d.className = "cm " + who; d.textContent = text; log.appendChild(d); log.scrollTop = log.scrollHeight; };
  const post = (url, body) => fetch(url, {method: "POST", headers: {"Content-Type": "application/json", "X-CSRF": csrf}, body: JSON.stringify(body)}).then(r => r.json().then(j => ({ok: r.ok, j})));
  const btn = (label, cls, fn) => { const b = document.createElement("button"); b.type = "button"; b.textContent = label; b.className = cls || ""; b.onclick = fn; actions.appendChild(b); };
  function open() { panel.hidden = false; fab.setAttribute("aria-expanded", "true"); fab.hidden = true; if (!messages.length) add("bot", "Hi! I'm the Instamart Copilot assistant. Ask me about orders, refunds, delivery, payments or privacy and I'll try to sort it out. If I can't, I'll raise a ticket for a support agent."); input.focus(); }
  function close() { panel.hidden = true; fab.hidden = false; fab.setAttribute("aria-expanded", "false"); }
  function reset() { messages = []; turns = 0; finished = false; log.textContent = ""; actions.textContent = ""; }
  function offer(handled) {
    actions.textContent = "";
    if (handled) { btn("✔ Solved, thank you", "good", solved); btn("✖ Not solved – create a ticket", "ghost", ticket); }
    else btn("Create a ticket for a support agent", "", ticket);
  }
  async function solved() { actions.textContent = ""; finished = true; add("bot", "Great, glad that helped! Tap “New chat” if you need anything else."); await post(panel.dataset.solvedUrl, {turns}); btn("New chat", "ghost", () => { reset(); open(); }); }
  async function ticket() {
    actions.textContent = ""; busy = true; const r = await post(panel.dataset.ticketUrl, {messages}); busy = false;
    if (!r.ok) { add("bot", r.j.error || "Sorry, the ticket could not be created. Please try again."); offer(false); return; }
    finished = true; add("bot", "Done. Ticket " + r.j.ref + " has been created with this whole conversation, so you won't need to repeat anything. A support agent will reply on the ticket.");
    const a = document.createElement("a"); a.href = r.j.url; a.className = "btn"; a.textContent = "View " + r.j.ref; actions.appendChild(a); btn("New chat", "ghost", () => { reset(); open(); });
  }
  form.addEventListener("submit", async e => {
    e.preventDefault(); const text = input.value.trim(); if (busy || finished || text.length < 2) return;
    input.value = ""; actions.textContent = ""; add("user", text); busy = true; turns += 1;
    try { const prior = messages.filter(m => m.who === "user").map(m => m.text).slice(0, -1).slice(-20);
      const r = await post(panel.dataset.chatUrl, {message: text, turns: turns - 1, history: prior}); busy = false;
      if (!r.ok) { add("bot", r.j.error || "Sorry, something went wrong."); return; }
      add("bot", r.j.reply); if (r.j.ask) actions.textContent = ""; else offer(r.j.handled);
    } catch (err) { busy = false; add("bot", "I could not reach the server. Please try again."); }
  });
  document.addEventListener("click", e => { if (e.target.closest("[data-chat-open]")) open(); });
  document.getElementById("chat-close").addEventListener("click", close);
  document.addEventListener("keydown", e => { if (e.key === "Escape" && !panel.hidden) close(); });
})();
