const messagesEl = document.getElementById("messages");
const form = document.getElementById("chat-form");
const input = document.getElementById("input");

let history = []; // {role, content}

function addMessage(role, content, sources = []) {
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  el.textContent = content;
  if (role === "assistant" && sources && sources.length) {
    const src = document.createElement("div");
    src.className = "sources";
    src.textContent = "Sources: " + sources.map(s => s.tag).join(", ");
    el.appendChild(src);
  }
  messagesEl.appendChild(el);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

async function sendMessage(text) {
  addMessage("user", text);
  input.value = "";
  const btn = form.querySelector("button");
  btn.disabled = true;

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, history }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || "Server error");
    }
    const data = await res.json();
    addMessage("assistant", data.reply, data.sources);

    history.push({ role: "user", content: text });
    history.push({ role: "assistant", content: data.reply });
    if (history.length > 12) history = history.slice(-12);
  } catch (e) {
    addMessage("assistant", `⚠️ ${e.message}`);
  } finally {
    btn.disabled = false;
  }
}

form.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text) return;
  sendMessage(text);
});

fetch("/api/health").catch(() => {});