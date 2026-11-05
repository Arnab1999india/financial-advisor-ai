/**
 * Browser Web Speech API wrapper for chat voice dictation (English, en-US).
 * Keeps recognition logic separate from UI wiring in script.js.
 */
(function (global) {
  "use strict";

  const SpeechRecognition =
    global.SpeechRecognition || global.webkitSpeechRecognition;

  function isSupported() {
    return Boolean(SpeechRecognition);
  }

  function joinSpeechParts(base, committed, interim) {
    const prefix = base || "";
    const middle = committed || "";
    const tail = interim || "";
    let combined = prefix;
    if (middle) {
      combined = combined
        ? combined.replace(/\s+$/, "") + (/\s$/.test(combined) ? "" : " ") + middle.trimStart()
        : middle;
    }
    if (tail) {
      combined = combined
        ? combined.replace(/\s+$/, "") + (/\s$/.test(combined) ? "" : " ") + tail.trimStart()
        : tail;
    }
    return combined;
  }

  /**
   * @param {object} options
   * @param {(text: string) => void} options.onTranscriptChange
   * @param {(recording: boolean) => void} [options.onRecordingChange]
   * @param {(message: string) => void} [options.onError]
   * @param {() => string} options.getInputValue
   * @param {(value: string) => void} options.setInputValue
   */
  function createVoiceDictation(options) {
    const {
      onTranscriptChange,
      onRecordingChange,
      onError,
      getInputValue,
      setInputValue,
    } = options;

    let recognition = null;
    let recording = false;
    let dictationBase = "";
    let applyingProgrammatically = false;
    let pendingEnterSend = false;
    let restartAfterStop = false;
    let finalizeResolve = null;
    let finalizePromise = null;

    function notifyRecording(active) {
      recording = active;
      if (onRecordingChange) onRecordingChange(active);
    }

    function buildFromResults(event) {
      let committed = "";
      let interim = "";
      for (let i = 0; i < event.results.length; i += 1) {
        const piece = event.results[i][0].transcript;
        if (event.results[i].isFinal) {
          committed += piece;
        }
      }
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        if (!event.results[i].isFinal) {
          interim += event.results[i][0].transcript;
        }
      }
      return joinSpeechParts(dictationBase, committed, interim);
    }

    function applyTranscript(text) {
      applyingProgrammatically = true;
      setInputValue(text);
      applyingProgrammatically = false;
      if (onTranscriptChange) onTranscriptChange(text);
    }

    function attachRecognitionHandlers() {
      recognition.onstart = () => {
        notifyRecording(true);
      };

      recognition.onresult = (event) => {
        const next = buildFromResults(event);
        applyTranscript(next);
      };

      recognition.onerror = (event) => {
        const code = event.error || "unknown";
        let message = "Voice input failed. Please try again.";
        if (code === "not-allowed" || code === "service-not-allowed") {
          message = "Microphone access was denied. Allow the mic in your browser settings.";
        } else if (code === "no-speech") {
          message = "No speech was detected. Try speaking again.";
        } else if (code === "network") {
          message = "Voice recognition needs a network connection.";
        } else if (code === "aborted") {
          return;
        }
        if (onError) onError(message);
        cleanupSession(false);
      };

      recognition.onend = () => {
        notifyRecording(false);
        if (finalizeResolve) {
          finalizeResolve();
          finalizeResolve = null;
          finalizePromise = null;
        }
        if (restartAfterStop) {
          restartAfterStop = false;
          startRecognitionInstance(false);
          return;
        }
        if (pendingEnterSend) {
          pendingEnterSend = false;
          if (options.onEnterSendAfterStop) {
            options.onEnterSendAfterStop();
          }
        }
      };
    }

    function cleanupSession(clearHandlers) {
      notifyRecording(false);
      restartAfterStop = false;
      dictationBase = "";
      if (clearHandlers && recognition) {
        recognition.onstart = null;
        recognition.onresult = null;
        recognition.onerror = null;
        recognition.onend = null;
      }
    }

    function startRecognitionInstance(resetBase) {
      if (!isSupported()) {
        return { ok: false, error: "unsupported" };
      }
      if (resetBase) {
        dictationBase = getInputValue();
      }

      recognition = new SpeechRecognition();
      recognition.lang = "en-US";
      recognition.continuous = true;
      recognition.interimResults = true;
      recognition.maxAlternatives = 1;
      attachRecognitionHandlers();

      try {
        recognition.start();
      } catch {
        cleanupSession(true);
        if (onError) onError("Could not start voice input. Try again.");
        return { ok: false, error: "start_failed" };
      }
      return { ok: true };
    }

    function start() {
      if (recording) {
        return { ok: false, error: "already_recording" };
      }
      if (!isSupported()) {
        const msg =
          "Voice input is not supported in this browser. Try Chrome or Edge on desktop.";
        if (onError) onError(msg);
        return { ok: false, error: "unsupported" };
      }
      return startRecognitionInstance(true);
    }

    function stop() {
      if (!recognition || !recording) {
        return Promise.resolve();
      }
      if (!finalizePromise) {
        finalizePromise = new Promise((resolve) => {
          finalizeResolve = resolve;
        });
      }
      try {
        recognition.stop();
      } catch {
        cleanupSession(false);
        if (finalizeResolve) {
          finalizeResolve();
          finalizeResolve = null;
          finalizePromise = null;
        }
      }
      return finalizePromise;
    }

    function toggle() {
      if (recording) {
        return stop();
      }
      start();
      return Promise.resolve();
    }

    function handleManualInputEdit() {
      if (!recording || applyingProgrammatically || !recognition) {
        return;
      }
      dictationBase = getInputValue();
      restartAfterStop = true;
      try {
        recognition.stop();
      } catch {
        restartAfterStop = false;
      }
    }

    function requestEnterSendWhileRecording() {
      if (!recording) {
        return false;
      }
      pendingEnterSend = true;
      stop();
      return true;
    }

    function destroy() {
      pendingEnterSend = false;
      if (recognition && recording) {
        try {
          recognition.stop();
        } catch {
          /* ignore */
        }
      }
      cleanupSession(true);
      recognition = null;
    }

    return {
      isSupported,
      start,
      stop,
      toggle,
      isRecording: () => recording,
      handleManualInputEdit,
      requestEnterSendWhileRecording,
      destroy,
    };
  }

  global.VoiceDictation = {
    isSupported,
    create: createVoiceDictation,
    joinSpeechParts,
  };
})(typeof window !== "undefined" ? window : globalThis);
