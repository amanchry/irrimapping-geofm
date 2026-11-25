// Renders the ```mermaid blocks of every page with ONE consistent, neutral style (the diagrams themselves carry no
// colours). Diagrams keep their natural size (readable text) and scroll sideways when wider than the page.
// document$ is Material's page-load signal (also fires on instant navigation).
const MERMAID_CONFIG = {
  startOnLoad: false,
  securityLevel: "loose",
  theme: "base",
  themeVariables: {
    fontFamily: "Roboto, 'Helvetica Neue', Arial, sans-serif",   // the site font (loaded by Material)
    fontSize: "15px",
    primaryColor: "#FFFFFF",          // boxes
    primaryBorderColor: "#455A64",
    primaryTextColor: "#1F2933",
    secondaryColor: "#F2F4F5",
    tertiaryColor: "#F7F8F9",
    lineColor: "#546E7A",             // arrows
    clusterBkg: "#F5F7F8",            // subgraphs (groups)
    clusterBorder: "#B0BEC5",
    edgeLabelBackground: "#FFFFFF",
    // Gantt
    taskBkgColor: "#CFD8DC", taskBorderColor: "#546E7A", taskTextColor: "#1F2933",
    taskTextOutsideColor: "#1F2933", taskTextLightColor: "#1F2933",
    activeTaskBkgColor: "#90A4AE", activeTaskBorderColor: "#37474F",
    doneTaskBkgColor: "#ECEFF1", doneTaskBorderColor: "#90A4AE",
    critBkgColor: "#B0BEC5", gridColor: "#E0E4E7", todayLineColor: "#00897B",
    sectionBkgColor: "#F7F8F9", altSectionBkgColor: "#FFFFFF", sectionBkgColor2: "#F7F8F9",
  },
  flowchart: { htmlLabels: true, useMaxWidth: false, nodeSpacing: 40, rankSpacing: 50, padding: 12,
               wrappingWidth: 260,
               curve: "basis" },
  gantt: { useMaxWidth: true, fontSize: 14, sectionFontSize: 14, barHeight: 24, barGap: 6,
           leftPadding: 200, topPadding: 40, gridLineStartPadding: 30 },
};

const MIN_SCALE = 0.8;   // never shrink text below 80 % (12 px); wider diagrams are redrawn top-down or scroll

async function draw(block, source, id) {
  const { svg } = await mermaid.render(id, source, block);
  block.innerHTML = svg;
  const el = block.querySelector("svg");
  const width = el.getBoundingClientRect().width, room = block.clientWidth - 24;
  return { el, width, room };
}

document$.subscribe(async () => {
  const blocks = document.querySelectorAll("div.mermaid-diagram:not([data-rendered])");
  if (!blocks.length) return;
  if (document.fonts) await document.fonts.ready;   // measure labels with the final web font, not a fallback
  mermaid.initialize(MERMAID_CONFIG);
  let n = 0;
  for (const block of blocks) {
    let source = block.textContent;
    block.dataset.rendered = "1";
    block.textContent = "";                       // render inside the block: diagrams (Gantt) get the column width
    try {
      let { el, width, room } = await draw(block, source, `mermaid-${Date.now()}-${n++}`);
      if (width > room && room / width < MIN_SCALE && /^\s*(flowchart|graph)\s+LR/m.test(source)) {
        source = source.replace(/^(\s*(?:flowchart|graph))\s+LR/m, "$1 TD");      // long chain: draw top-down
        ({ el, width, room } = await draw(block, source, `mermaid-${Date.now()}-${n++}`));
      }
      if (width > room && room / width >= MIN_SCALE) {                            // slightly too wide: fit
        el.style.width = `${room}px`;
      }
    } catch (err) {
      block.innerHTML = `<pre>${source}</pre>`;
      console.error("Mermaid:", err);
    }
  }
});
