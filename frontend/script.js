// Configuration
const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const chatMessages = document.getElementById("chat-messages");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
// const micButton = document.getElementById("mic-button");
const loading = document.getElementById("loading");

// State
let isProcessing = false;

function setupChatInput() {
  if (!sendButton || !messageInput) return;

  sendButton.addEventListener("click", (e) => {
    e.preventDefault();
    sendMessage();
  });

  messageInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  console.log("Financial Bot Frontend loaded");
  setupChatInput();
  messageInput.focus();
});

/*
 * Voice / mic (ChatVoiceInput) — paused; re-enable when Streamlit + static mic UX is fixed.
 *
 * document.addEventListener("DOMContentLoaded", () => {
 *   ChatVoiceInput.init({
 *     messageInput,
 *     sendButton,
 *     micButton,
 *     voiceHint,
 *     toastRegion,
 *     getIsProcessing: () => isProcessing,
 *     onSubmit: () => sendMessage(),
 *   });
 * });
 */

function clearPreviousTurn() {
  const previousUser = document.getElementById("active-user-message");
  const previousBot = document.getElementById("active-bot-response");
  if (previousUser) previousUser.remove();
  if (previousBot) previousBot.remove();
}

async function consumeChatStream(response, botContentDiv) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let answer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      for (const line of block.split("\n")) {
        if (!line.startsWith("data: ")) continue;
        let payload;
        try {
          payload = JSON.parse(line.slice(6));
        } catch {
          continue;
        }
        if (payload.type === "status" && loading) {
          loading.textContent = payload.message || "Working…";
        } else if (payload.type === "token") {
          answer += payload.text || "";
          botContentDiv.textContent = answer;
          chatMessages.scrollTop = chatMessages.scrollHeight;
        } else if (payload.type === "done") {
          answer = payload.response?.answer || answer;
          botContentDiv.textContent = answer;
        } else if (payload.type === "error") {
          throw new Error(payload.message || "Stream error");
        }
      }
    }
  }
  return answer;
}

async function sendMessage() {
  if (isProcessing) {
    return;
  }

  const message = messageInput.value.trim();
  if (!message) {
    return;
  }

  isProcessing = true;
  setProcessingState(true);
  clearPreviousTurn();

  const userMessageDiv = addMessage(message, "user");
  userMessageDiv.id = "active-user-message";
  const botMessageDiv = addMessage("", "bot");
  botMessageDiv.id = "active-bot-response";
  const botContentDiv = botMessageDiv.querySelector(".message-content");

  try {
    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "text/event-stream",
      },
      body: JSON.stringify({ query: message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    await consumeChatStream(response, botContentDiv);
  } catch (error) {
    console.error("Error:", error);
    botContentDiv.textContent =
      "Sorry, I encountered an error while processing your question. Please try again.";
  } finally {
    setProcessingState(false);
  }
}

function addMessage(text, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}-message`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = text;

  messageDiv.appendChild(contentDiv);
  chatMessages.appendChild(messageDiv);

  chatMessages.scrollTop = chatMessages.scrollHeight;
  return messageDiv;
}

function setProcessingState(processing) {
  isProcessing = processing;

  if (processing) {
    loading.classList.remove("hidden");
    loading.textContent = "Working…";
  } else {
    loading.classList.add("hidden");
    messageInput.focus();
  }
}

function formatMessage(text) {
  return text.trim();
}

function sanitizeInput(text) {
  return text.replace(/[<>]/g, "");
}

function handleApiError(error) {
  console.error("API Error:", error);
}
