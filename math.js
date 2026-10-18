// Render the math spans emitted by build.py (markdown-it dollarmath) with KaTeX.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".math").forEach((el) => {
    const display = el.classList.contains("block");
    try {
      katex.render(el.textContent, el, { displayMode: display, throwOnError: false });
    } catch (e) {
      console.warn("KaTeX failed:", el.textContent, e);
    }
  });
});
