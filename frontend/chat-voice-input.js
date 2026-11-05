/**
 * Shared mic + textarea wiring for static chat and Streamlit iframe input.
 */
(function (global) {
  "use strict";

  /**
   * @param {object} opts
   * @param {HTMLTextAreaElement} opts.messageInput
   * @param {HTMLButtonElement} opts.sendButton
   * @param {HTMLButtonElement} opts.micButton
   * @param {HTMLElement} opts.voiceHint
   * @param {HTMLElement} opts.toastRegion
   * @param {() => boolean} opts.getIsProcessing
   * @param {(text: string) => void} opts.onSubmit
   * @param {() => void} [opts.onResize]
   */
  function init(opts) {
    const {
      messageInput,
      sendButton,
      micButton,
      voiceHint,
      toastRegion,
      getIsProcessing,
      onSubmit,
      onResize,
    } = opts;

    if (!messageInput || !sendButton) {
      if (sendButton) {
        sendButton.addEventListener("click", (e) => {
          e.preventDefault();
          if (messageInput && messageInput.value.trim()) onSubmit(messageInput.value.trim());
        });
      }
      return { showToast: () => {}, setProcessingDisabled: () => {}, destroy: () => {} };
    }

    let toastTimer = null;
    /** @type {ReturnType<typeof VoiceDictation.create> | null} */
    let voiceDictation = null;

    function notifyResize() {
      if (onResize) onResize();
    }

    function showToast(message, durationMs = 5000) {
      if (!toastRegion || !message) return;
      if (toastTimer) {
        clearTimeout(toastTimer);
        toastTimer = null;
      }
      toastRegion.innerHTML = "";
      const el = document.createElement("div");
      el.className = "toast";
      el.textContent = message;
      toastRegion.appendChild(el);
      notifyResize();
      toastTimer = setTimeout(() => {
        toastRegion.innerHTML = "";
        toastTimer = null;
        notifyResize();
      }, durationMs);
    }

    function setRecordingUi(active) {
      if (voiceHint) {
        voiceHint.classList.toggle("hidden", !active);
      }
      if (micButton) {
        micButton.classList.toggle("recording", active);
        micButton.setAttribute("aria-pressed", active ? "true" : "false");
        micButton.setAttribute(
          "aria-label",
          active ? "Stop voice input" : "Start voice input"
        );
      }
      messageInput.classList.toggle("recording-active", active);
      notifyResize();
    }

    function submitCurrent() {
      const processing = getIsProcessing();
      const text = messageInput.value.trim();
      if (processing) return;
      if (voiceDictation && voiceDictation.isRecording()) {
        voiceDictation.requestEnterSendWhileRecording();
        return;
      }
      if (!text) {
        showToast("Please enter a question.");
        return;
      }
      onSubmit(text);
    }

    if (!global.VoiceDictation) {
      if (micButton) {
        micButton.disabled = true;
        micButton.title = "Voice input is not available";
      }
    } else if (!VoiceDictation.isSupported()) {
      if (micButton) {
        micButton.disabled = true;
        micButton.title = "Voice input is not supported in this browser";
      }
    } else {
      voiceDictation = VoiceDictation.create({
        getInputValue: () => messageInput.value,
        setInputValue: (value) => {
          messageInput.value = value;
        },
        onRecordingChange: setRecordingUi,
        onError: (message) => {
          setRecordingUi(false);
          showToast(message);
        },
        onEnterSendAfterStop: submitCurrent,
      });
    }

    sendButton.addEventListener("click", (e) => {
      e.preventDefault();
      submitCurrent();
    });

    messageInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitCurrent();
      }
    });

    messageInput.addEventListener("input", () => {
      if (!voiceDictation || !voiceDictation.isRecording()) {
        return;
      }
      voiceDictation.handleManualInputEdit();
    });

    if (micButton) {
      micButton.addEventListener("click", (e) => {
        e.preventDefault();
        if (!voiceDictation || getIsProcessing()) return;
        if (!VoiceDictation.isSupported()) {
          showToast(
            "Voice input is not supported in this browser. Try Chrome or Edge on desktop."
          );
          return;
        }
        if (voiceDictation.isRecording()) {
          voiceDictation.stop();
        } else {
          voiceDictation.start();
        }
      });
    }

    global.addEventListener("beforeunload", () => {
      if (voiceDictation) voiceDictation.destroy();
    });

    return {
      showToast,
      setProcessingDisabled(disabled) {
        if (micButton) {
          micButton.disabled =
            disabled || !global.VoiceDictation || !VoiceDictation.isSupported();
        }
      },
      destroy() {
        if (voiceDictation) voiceDictation.destroy();
      },
    };
  }

  global.ChatVoiceInput = { init };
})(typeof window !== "undefined" ? window : globalThis);
