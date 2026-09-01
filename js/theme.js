// theme system — dark default to match andrewzhou.org, persisted per-browser.
// loaded synchronously in <head> so the light class lands before first paint.
(function () {
  try {
    if (localStorage.getItem("theme") === "light") {
      document.documentElement.classList.add("theme-light");
    }
  } catch (e) {
    /* private mode etc. — stay dark */
  }

  window.toggleTheme = function () {
    var light = document.documentElement.classList.toggle("theme-light");
    try {
      localStorage.setItem("theme", light ? "light" : "dark");
    } catch (e) {}
    var btn = document.querySelector(".theme-toggle");
    if (btn) btn.textContent = light ? "dark mode" : "light mode";
  };

  document.addEventListener("DOMContentLoaded", function () {
    var btn = document.querySelector(".theme-toggle");
    if (btn && document.documentElement.classList.contains("theme-light")) {
      btn.textContent = "dark mode";
    }
  });
})();
