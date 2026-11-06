// Renders the ```mermaid blocks of every page with the colours defined in the diagrams themselves.
// document$ is Material's page-load signal (also fires on instant navigation).
document$.subscribe(async () => {
  const blocks = document.querySelectorAll("div.mermaid-diagram:not([data-rendered])");
  if (!blocks.length) return;
  mermaid.initialize({ startOnLoad: false, theme: "default", securityLevel: "loose",
                       flowchart: { htmlLabels: true, useMaxWidth: true } });
  let n = 0;
  for (const block of blocks) {
    const source = block.textContent;
    block.dataset.rendered = "1";
    try {
      const { svg } = await mermaid.render(`mermaid-${Date.now()}-${n++}`, source);
      block.innerHTML = svg;
    } catch (err) {
      block.innerHTML = `<pre>${source}</pre>`;
      console.error("Mermaid:", err);
    }
  }
});
