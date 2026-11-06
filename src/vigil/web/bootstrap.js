// Keep startup failures visible even if a module download or WebGL initialization
// prevents app.js from reaching its WebSocket connection and procedure selector.
(() => {
  let receivedStatus = false;
  let moduleReady = false;
  let failed = false;
  const el = (id) => document.getElementById(id);
  const show = (title, message, error = false) => {
    const boot = el("boot");
    boot.hidden = false;
    boot.classList.remove("leaving");
    boot.classList.toggle("failed", error);
    el("boot-sub").textContent = title;
    el("boot-choices").hidden = true;
    el("boot-progress").hidden = false;
    el("boot-err").hidden = false;
    el("boot-err").textContent = message;
    el("boot-eta").textContent = "";
  };
  const report = (error) => {
    if (receivedStatus) return;
    failed = true;
    clearTimeout(watchdog);
    const detail = error?.stack || error?.message || String(error);
    show("DASHBOARD COULD NOT START", detail, true);
  };
  const watchdog = setTimeout(() => {
    if (receivedStatus || failed) return;
    show("WAITING FOR DASHBOARD", moduleReady
      ? "The dashboard loaded but has not received the server's startup status. Check the server address and connection."
      : "The dashboard scripts are still loading. A module download or graphics initialization may be stalled. Check the browser Console and Network tabs.");
  }, 15000);
  const onError = (event) => { if (event.error) report(event.error); };
  const onRejection = (event) => report(event.reason);
  window.addEventListener("error", onError);
  window.addEventListener("unhandledrejection", onRejection);
  window.addEventListener("vigil:startup-status", () => {
    receivedStatus = true;
    clearTimeout(watchdog);
    window.removeEventListener("error", onError);
    window.removeEventListener("unhandledrejection", onRejection);
    el("boot-err").hidden = true;
  }, { once: true });
  import("./app.js").then(() => {
    moduleReady = true;
    if (!receivedStatus && !failed) el("boot-sub").textContent = "CONNECTING TO SERVER…";
  }).catch(report);
})();
