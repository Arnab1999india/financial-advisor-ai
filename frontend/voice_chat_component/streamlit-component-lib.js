/**
 * Minimal Streamlit custom component bridge (v1 message protocol).
 * Based on streamlit-component-lib behaviour for declare_component iframes.
 */
(function () {
  "use strict";

  function send(type, data) {
    window.parent.postMessage(
      Object.assign({ isStreamlitMessage: true, type: type }, data),
      "*"
    );
  }

  const Streamlit = {
    RENDER_EVENT: "streamlit:render",
    events: {
      addEventListener: function (eventName, callback) {
        window.addEventListener("message", function (event) {
          const data = event.data;
          if (!data || data.type !== eventName) return;
          callback({
            detail: {
              args: data.args || {},
              disabled: data.disabled,
              theme: data.theme,
            },
          });
        });
      },
    },
    setComponentReady: function () {
      send("streamlit:componentReady", { apiVersion: 1 });
    },
    setComponentValue: function (value) {
      send("streamlit:setComponentValue", { value: value, dataType: "json" });
    },
    setFrameHeight: function (height) {
      send("streamlit:setFrameHeight", { height: height });
    },
  };

  window.Streamlit = Streamlit;
})();
