// Configuration
const API_BASE_URL = "http://localhost:8000";

// DOM Elements
const chatMessages = document.getElementById("chat-messages");
const messageInput = document.getElementById("message-input");
const sendButton = document.getElementById("send-button");
const loading = document.getElementById("loading");

// State
let isProcessing = false;

// Initialize the application
document.addEventListener("DOMContentLoaded", () => {
  console.log("Financial Bot Frontend loaded");
  setupEventListeners();
  messageInput.focus();
});

// Event Listeners
function setupEventListeners() {
  sendButton.addEventListener("click", sendMessage);
  messageInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// Message Handling
async function sendMessage() {
  const message = messageInput.value.trim();

  if (!message || isProcessing) {
    return;
  }

  // Add user message to chat
  addMessage(message, "user");

  // Clear input and disable controls
  messageInput.value = "";
  setProcessingState(true);

  try {
    // Send request to backend
    const response = await fetch(`${API_BASE_URL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query: message }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Add bot response to chat
    addMessage(data.answer, "bot");
  } catch (error) {
    console.error("Error:", error);
    addMessage(
      "Sorry, I encountered an error while processing your question. Please try again.",
      "bot"
    );
  } finally {
    setProcessingState(false);
  }
}

// UI Functions
function addMessage(text, sender) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}-message`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.textContent = text;

  messageDiv.appendChild(contentDiv);
  chatMessages.appendChild(messageDiv);

  // Scroll to bottom
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setProcessingState(processing) {
  isProcessing = processing;

  if (processing) {
    loading.classList.remove("hidden");
    sendButton.disabled = true;
    messageInput.disabled = true;
  } else {
    loading.classList.add("hidden");
    sendButton.disabled = false;
    messageInput.disabled = false;
    messageInput.focus();
  }
}

// Utility Functions
function formatMessage(text) {
  // Basic formatting for better display
  return text.trim();
}

function sanitizeInput(text) {
  // Basic sanitization
  return text.replace(/[<>]/g, "");
}

// Error Handling
function handleApiError(error) {
  console.error("API Error:", error);

  let errorMessage = "An error occurred. ";

  if (error.message.includes("Failed to fetch")) {
    errorMessage += "Please check your internet connection.";
  } else if (error.message.includes("503")) {
    errorMessage += "The service is temporarily unavailable.";
  } else {
    errorMessage += "Please try again later.";
  }

  addMessage(errorMessage, "bot");
}

// Keyboard Shortcuts
document.addEventListener("keydown", (e) => {
  // Ctrl/Cmd + Enter to send message
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    sendMessage();
  }

  // Escape to clear input
  if (e.key === "Escape") {
    messageInput.value = "";
    messageInput.focus();
  }
});

// Auto-resize input
messageInput.addEventListener("input", () => {
  messageInput.style.height = "auto";
  messageInput.style.height = Math.min(messageInput.scrollHeight, 100) + "px";
});

// Prevent form submission on Enter
messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
  }
});

// Initialize
console.log("Financial Bot Frontend initialized successfully");
