"""
a5_parcel_explorer.py  —  interactive map of the labelled parcels with their monthly time series
===============================================================================================

What it does
------------
Writes ONE self-contained HTML page to inspect the training parcels (not the chips):
  * all labelled parcels as polygons on a satellite (Esri) or street (Google) basemap, coloured by class (or fold);
  * click a parcel -> its monthly NDVI (or NDWI, Sentinel-1 VH / VV, number of clear Sentinel-2
    observations) over its agricultural year, drawn against the median and interquartile range of all
    irrigated and all rainfed parcels of the same agricultural year;
  * pin several parcels to compare their curves; filter by class, agricultural year, zone and fold;
    search by parcel ID.
The values are the parcel features of ml_dataset_prep/04_make_views.py (median over the inner pixels per
month), i.e. exactly what the baseline classifier sees.

Open the page in any browser (internet is needed only for the basemap tiles).

Inputs
------
outputs/dataset/02_folds/parcels_folds.geojson                       polygons + class, zone, fold
outputs/dataset/03_tiles/agyear/views/parcel_features_<FEATURES>.csv  monthly values per parcel

Output
------
outputs/analysis/parcel_explorer.html

Run (from the project folder; standard Python only)
---------------------------------------------------
    python tools/a5_parcel_explorer.py
"""
import csv
import json
from calendar import month_abbr
from pathlib import Path
from statistics import quantiles

# ================================ SETTINGS ================================
ROOT = Path(__file__).resolve().parents[1]                         # project folder
WINDOW = "agyear"
FEATURES = "inner"                                                  # "inner" | "all" (parcel_features_<FEATURES>.csv)
PARCELS_FILE = ROOT / "outputs" / "dataset" / "02_folds" / "parcels_folds.geojson"
FEATURES_FILE = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "views" / f"parcel_features_{FEATURES}.csv"
TILES_INFO = ROOT / "outputs" / "dataset" / "03_tiles" / WINDOW / "tiles_info.json"
OUT_FILE = ROOT / "outputs" / "analysis" / "parcel_explorer.html"

VARIABLES = {                    # key in the page -> (column prefix, label, decimals)
    "NDVI": ("NDVI", "NDVI", 3),
    "NDWI": ("NDWI", "NDWI (NIR vs SWIR1)", 3),
    "VH": ("VH_ASC", "Sentinel-1 VH, ascending (dB)", 2),
    "VV": ("VV_ASC", "Sentinel-1 VV, ascending (dB)", 2),
    "NCLEAR": ("S2_NCLEAR", "Clear Sentinel-2 observations", 1),
}
COORD_DECIMALS = 5               # ~1 m; keeps the page small
# ==========================================================================


def rounded(coords):
    if isinstance(coords[0], (int, float)):
        return [round(coords[0], COORD_DECIMALS), round(coords[1], COORD_DECIMALS)]
    return [rounded(c) for c in coords]


def to_float(v):
    try:
        x = float(v)
        return None if x != x else x
    except ValueError:
        return None


def main():
    info = json.loads(TILES_INFO.read_text())
    n_months, start = info["n_months"], info["start_month"]
    months = [month_abbr[(start - 1 + k) % 12 + 1] for k in range(n_months)]

    # monthly values per parcel
    series = {}
    with open(FEATURES_FILE, newline="") as fh:
        for row in csv.DictReader(fh):
            series[row["parcel_id"]] = {
                key: [None if (v := to_float(row.get(f"{prefix}_m{k + 1:02d}", ""))) is None else round(v, dec)
                      for k in range(n_months)]
                for key, (prefix, _, dec) in VARIABLES.items()} | {"npx": int(float(row["n_px_used"]))}

    # parcels
    gj = json.loads(PARCELS_FILE.read_text())
    parcels, missing = [], 0
    for f in gj["features"]:
        p = f["properties"]
        s = series.get(p["parcel_id"])
        if s is None:
            missing += 1
        parcels.append({"id": p["parcel_id"], "cls": int(p["label"]), "season": p["ag_season"], "zone": p["zone"],
                        "fold": int(p["fold"]), "area": round(float(p["area_ha"]), 2), "date": p.get("date", ""),
                        "geom": {"type": f["geometry"]["type"], "coordinates": rounded(f["geometry"]["coordinates"])},
                        "ts": s})

    # class reference curves: median and IQR per agricultural year, class, variable and month
    ref = {}
    for season in sorted({p["season"] for p in parcels}):
        for cls in (1, 2):
            group = [p["ts"] for p in parcels if p["season"] == season and p["cls"] == cls and p["ts"]]
            if len(group) < 5:
                continue
            for key in VARIABLES:
                rows = []
                for k in range(n_months):
                    vals = sorted(g[key][k] for g in group if g[key][k] is not None)
                    q = quantiles(vals, n=4) if len(vals) >= 4 else [None, None, None]
                    rows.append([round(x, 4) if x is not None else None for x in q])
                ref.setdefault(season, {}).setdefault(str(cls), {})[key] = {"n": len(group), "q": rows}

    data = {"months": months, "start_month": start, "parcels": parcels, "ref": ref,
            "vars": {k: {"label": v[1]} for k, v in VARIABLES.items()}, "features": FEATURES}
    html = TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(html, encoding="utf-8")
    print(f"{len(parcels):,} parcels ({missing} without features), {OUT_FILE.stat().st_size / 1e6:.1f} MB -> {OUT_FILE}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parcel Explorer</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#FAF9F5;--panel:#FFFFFF;--ink:#1C2733;--muted:#6B7480;--line:#E2DED3;--irr:#1F5F8B;--rf:#B4541A;--accent:#C8A21A}
*{box-sizing:border-box}
html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
#app{display:grid;grid-template-columns:minmax(0,1fr) 440px;height:100vh}
#map{height:100%}
aside{background:var(--panel);border-left:1px solid var(--line);overflow-y:auto;padding:18px 20px 28px}
h1{font-size:18px;margin:0 0 2px}
.sub{color:var(--muted);font-size:12.5px;margin:0 0 14px}
.box{border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin-bottom:14px;background:#fff}
.box h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:0 0 8px}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:6px 0}
label{font-size:13px;display:inline-flex;gap:5px;align-items:center;cursor:pointer}
select,input[type=text]{font:inherit;font-size:13px;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink)}
input[type=text]{flex:1;min-width:0}
button{font:inherit;font-size:12.5px;padding:5px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;color:var(--ink);cursor:pointer}
button:hover{border-color:var(--ink)}
.seg button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.sw{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-1px}
.count{color:var(--muted);font-size:12.5px}
#info dl{display:grid;grid-template-columns:auto 1fr;gap:3px 12px;margin:0;font-size:13px}
#info dt{color:var(--muted)} #info dd{margin:0;word-break:break-all}
.badge{display:inline-block;padding:1px 8px;border-radius:99px;color:#fff;font-size:12px;font-weight:600}
#chart svg{width:100%;height:auto;display:block}
#pins{display:flex;flex-direction:column;gap:4px;margin-top:6px}
.pin{display:flex;align-items:center;gap:8px;font-size:12.5px}
.pin span.id{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.pin button{padding:1px 7px}
.empty{color:var(--muted);font-size:13px}
.legend{background:rgba(255,255,255,.94);padding:8px 10px;border-radius:8px;font-size:12.5px;line-height:1.7;box-shadow:0 1px 4px rgba(0,0,0,.2)}
@media (max-width:860px){#app{grid-template-columns:1fr;grid-template-rows:55vh auto;height:auto}#map{height:55vh}aside{border-left:0;border-top:1px solid var(--line);padding:14px 16px}}
</style>
</head>
<body>
<div id="app">
  <div id="map"></div>
  <aside>
    <h1>Parcel Explorer</h1>
    <p class="sub">Labelled training parcels · click a polygon to see its monthly profile</p>

    <div class="box">
      <h2>Filter</h2>
      <div class="row">
        <label><input type="checkbox" id="fIrr" checked><span class="sw" style="background:var(--irr)"></span>Irrigated</label>
        <label><input type="checkbox" id="fRf" checked><span class="sw" style="background:var(--rf)"></span>Rainfed</label>
      </div>
      <div class="row">
        <select id="fSeason"></select><select id="fZone"></select><select id="fFold"></select>
      </div>
      <div class="row">
        <span class="count">Colour by</span>
        <select id="colorBy"><option value="class">class</option><option value="fold">fold</option></select>
        <span class="count" id="nShown"></span>
      </div>
      <div class="row">
        <input type="text" id="search" placeholder="Search parcel ID…" list="ids"><datalist id="ids"></datalist>
        <button id="go">Find</button>
      </div>
    </div>

    <div class="box" id="info"><h2>Parcel</h2><div class="empty">Click a parcel on the map.</div></div>

    <div class="box">
      <h2>Monthly profile</h2>
      <div class="row seg" id="vars"></div>
      <div id="chart"><div class="empty">No parcel selected.</div></div>
      <div class="row">
        <button id="pinBtn" disabled>Pin for comparison</button>
        <button id="clearPins">Clear pins</button>
      </div>
      <div id="pins"></div>
    </div>
    <p class="sub" id="foot"></p>
  </aside>
</div>
<script>
const D = __DATA__;
const IRR = getComputedStyle(document.documentElement).getPropertyValue('--irr').trim();
const RF = getComputedStyle(document.documentElement).getPropertyValue('--rf').trim();
const CLS = {1: {name: 'Irrigated', color: IRR}, 2: {name: 'Rainfed', color: RF}};
const FOLDC = ['#4E79A7', '#F28E2B', '#59A14F', '#B07AA1', '#EDC948'];
const PINC = ['#111111', '#7B3FA0', '#2A9D8F', '#D62828', '#6A994E', '#E76F51'];
let curVar = 'NDVI', selected = null, pins = [];

document.getElementById('foot').textContent =
  `Values: median of the parcel's ${D.features} pixels per month (ml_dataset_prep/04_make_views.py). ` +
  `Bands: median and interquartile range of all parcels of the same class and agricultural year.`;

// ---------- map
const map = L.map('map', {preferCanvas: true, zoomControl: true});
const sat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: 'Imagery © Esri, Maxar, Earthstar Geographics'});
const satLabels = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}',
  {maxZoom: 19, attribution: '© Esri'});
const osm = L.tileLayer('https://mt{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}',
  {maxZoom: 20, subdomains: ['0', '1', '2', '3'], attribution: 'Map data © Google'});
sat.addTo(map);
L.control.layers({'Satellite': sat, 'Street map': osm}, {'Place names': satLabels}, {collapsed: false}).addTo(map);
L.control.scale({imperial: false}).addTo(map);

const legend = L.control({position: 'bottomleft'});
legend.onAdd = () => { const d = L.DomUtil.create('div', 'legend'); d.id = 'legend'; return d; };
legend.addTo(map);

function styleFor(p) {
  const by = document.getElementById('colorBy').value;
  const c = by === 'fold' ? FOLDC[p.fold % 5] : CLS[p.cls].color;
  const isSel = selected && selected.id === p.id;
  return {color: isSel ? '#FFD23F' : c, weight: isSel ? 3 : 1.4, fillColor: c, fillOpacity: isSel ? 0.15 : 0.35, opacity: 1};
}
const layers = [];
D.parcels.forEach(p => {
  const lyr = L.geoJSON(p.geom, {style: () => styleFor(p)});
  lyr.on('click', () => select(p));
  lyr.bindTooltip(() => `${CLS[p.cls].name} · ${p.zone} · ${p.season}`, {sticky: true});
  p.layer = lyr; layers.push(lyr);
});

// ---------- filters
function fillSelect(id, label, values) {
  const s = document.getElementById(id);
  s.innerHTML = `<option value="">All ${label}</option>` + values.map(v => `<option value="${v}">${label === 'folds' ? 'Fold ' + v : v}</option>`).join('');
}
fillSelect('fSeason', 'years', [...new Set(D.parcels.map(p => p.season))].sort());
fillSelect('fZone', 'zones', [...new Set(D.parcels.map(p => p.zone))].sort());
fillSelect('fFold', 'folds', [...new Set(D.parcels.map(p => p.fold))].sort());
document.getElementById('ids').innerHTML = D.parcels.map(p => `<option value="${p.id}">`).join('');

function visible(p) {
  const fs = document.getElementById('fSeason').value, fz = document.getElementById('fZone').value, ff = document.getElementById('fFold').value;
  if (p.cls === 1 && !document.getElementById('fIrr').checked) return false;
  if (p.cls === 2 && !document.getElementById('fRf').checked) return false;
  if (fs && p.season !== fs) return false;
  if (fz && p.zone !== fz) return false;
  if (ff !== '' && String(p.fold) !== ff) return false;
  return true;
}
function refresh(fit) {
  let n = 0; const shown = [];
  D.parcels.forEach(p => {
    const v = visible(p);
    if (v) { n++; shown.push(p.layer); if (!map.hasLayer(p.layer)) p.layer.addTo(map); p.layer.setStyle(styleFor(p)); }
    else if (map.hasLayer(p.layer)) map.removeLayer(p.layer);
  });
  document.getElementById('nShown').textContent = `${n.toLocaleString()} of ${D.parcels.length.toLocaleString()} parcels`;
  if (fit && shown.length) map.fitBounds(L.featureGroup(shown).getBounds(), {padding: [20, 20]});
  const by = document.getElementById('colorBy').value;
  document.getElementById('legend').innerHTML = by === 'fold'
    ? FOLDC.map((c, i) => `<span class="sw" style="background:${c}"></span> Fold ${i}`).join('<br>')
    : `<span class="sw" style="background:${IRR}"></span> Irrigated<br><span class="sw" style="background:${RF}"></span> Rainfed`;
}
['fIrr', 'fRf', 'fSeason', 'fZone', 'fFold'].forEach(id => document.getElementById(id).addEventListener('change', () => refresh(true)));
document.getElementById('colorBy').addEventListener('change', () => refresh(false));
function find() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const p = D.parcels.find(x => x.id.toLowerCase() === q) || D.parcels.find(x => x.id.toLowerCase().includes(q));
  if (!p) return;
  if (!visible(p)) { ['fSeason', 'fZone', 'fFold'].forEach(id => document.getElementById(id).value = ''); document.getElementById(p.cls === 1 ? 'fIrr' : 'fRf').checked = true; refresh(false); }
  map.fitBounds(p.layer.getBounds(), {maxZoom: 17}); select(p);
}
document.getElementById('go').addEventListener('click', find);
document.getElementById('search').addEventListener('keydown', e => { if (e.key === 'Enter') find(); });

// ---------- selection and info
function select(p) {
  const prev = selected; selected = p;
  if (prev) prev.layer.setStyle(styleFor(prev));
  p.layer.setStyle(styleFor(p)); p.layer.bringToFront();
  const c = CLS[p.cls];
  const y0 = parseInt(p.season.slice(0, 4));
  document.getElementById('info').innerHTML = `<h2>Parcel</h2><dl>
    <dt>Class</dt><dd><span class="badge" style="background:${c.color}">${c.name}</span></dd>
    <dt>ID</dt><dd>${p.id}</dd>
    <dt>Zone</dt><dd>${p.zone}</dd>
    <dt>Agricultural year</dt><dd>${p.season} (Sep ${y0} – Aug ${y0 + 1})</dd>
    <dt>Survey date</dt><dd>${p.date || '–'}</dd>
    <dt>Area</dt><dd>${p.area} ha${p.ts ? ` · ${p.ts.npx} pixels used` : ''}</dd>
    <dt>Fold</dt><dd>${p.fold}</dd></dl>`;
  document.getElementById('pinBtn').disabled = !p.ts;
  draw();
}

// ---------- variable buttons
document.getElementById('vars').innerHTML = Object.keys(D.vars).map(k => `<button data-v="${k}" class="${k === curVar ? 'on' : ''}">${k === 'NCLEAR' ? 'Clear obs.' : k}</button>`).join('');
document.querySelectorAll('#vars button').forEach(b => b.addEventListener('click', () => {
  curVar = b.dataset.v; document.querySelectorAll('#vars button').forEach(x => x.classList.toggle('on', x === b)); draw();
}));

// ---------- pins
document.getElementById('pinBtn').addEventListener('click', () => {
  if (selected && selected.ts && !pins.includes(selected) && pins.length < PINC.length) { pins.push(selected); draw(); }
});
document.getElementById('clearPins').addEventListener('click', () => { pins = []; draw(); });

// ---------- chart (plain SVG)
function draw() {
  const el = document.getElementById('chart');
  if (!selected) { el.innerHTML = '<div class="empty">No parcel selected.</div>'; renderPins(); return; }
  if (!selected.ts) { el.innerHTML = '<div class="empty">This parcel has no pixel of its own (fully covered by an overlapping parcel), so it has no time series.</div>'; renderPins(); return; }
  const W = 400, H = 250, m = {l: 44, r: 10, t: 14, b: 30};
  const iw = W - m.l - m.r, ih = H - m.t - m.b, n = D.months.length;
  const season = selected.season, ref = D.ref[season] || {};
  const series = [selected, ...pins.filter(p => p !== selected)];
  const vals = [];
  series.forEach(p => p.ts[curVar].forEach(v => v != null && vals.push(v)));
  [1, 2].forEach(c => ref[c] && ref[c][curVar].q.forEach(q => q.forEach(v => v != null && vals.push(v))));
  let lo = Math.min(...vals), hi = Math.max(...vals); const pad = (hi - lo) * 0.08 || 0.05; lo -= pad; hi += pad;
  const x = i => m.l + (i + 0.5) * iw / n, y = v => m.t + ih - (v - lo) / (hi - lo) * ih;
  const path = arr => arr.map((v, i) => v == null ? null : `${x(i).toFixed(1)},${y(v).toFixed(1)}`).filter(Boolean).join(' L');
  let s = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Monthly ${curVar} profile">`;
  const ticks = 5;
  for (let k = 0; k <= ticks; k++) {
    const v = lo + (hi - lo) * k / ticks, yy = y(v);
    s += `<line x1="${m.l}" x2="${W - m.r}" y1="${yy}" y2="${yy}" stroke="#EEEAE0"/><text x="${m.l - 6}" y="${yy + 4}" font-size="10.5" text-anchor="end" fill="#6B7480">${Math.abs(hi - lo) < 2 ? v.toFixed(2) : v.toFixed(1)}</text>`;
  }
  D.months.forEach((mo, i) => s += `<text x="${x(i)}" y="${H - 10}" font-size="10.5" text-anchor="middle" fill="#6B7480">${mo}</text>`);
  [1, 2].forEach(c => {
    if (!ref[c]) return; const q = ref[c][curVar].q, col = CLS[c].color;
    const up = q.map((r, i) => r[2] == null ? null : `${x(i).toFixed(1)},${y(r[2]).toFixed(1)}`).filter(Boolean);
    const dn = q.map((r, i) => r[0] == null ? null : `${x(i).toFixed(1)},${y(r[0]).toFixed(1)}`).filter(Boolean).reverse();
    s += `<path d="M${up.join(' L')} L${dn.join(' L')} Z" fill="${col}" opacity="0.13"/>`;
    s += `<path d="M${path(q.map(r => r[1]))}" fill="none" stroke="${col}" stroke-width="1.6" stroke-dasharray="5 4" opacity="0.9"/>`;
  });
  series.forEach((p, k) => {
    const col = k === 0 ? CLS[p.cls].color : PINC[pins.indexOf(p) % PINC.length];
    const arr = p.ts[curVar];
    s += `<path d="M${path(arr)}" fill="none" stroke="${k === 0 ? '#1C2733' : col}" stroke-width="${k === 0 ? 4.2 : 2.2}" stroke-linejoin="round"/>`;
    if (k === 0) s += `<path d="M${path(arr)}" fill="none" stroke="${col}" stroke-width="2.6" stroke-linejoin="round"/>`;
    arr.forEach((v, i) => { if (v != null) s += `<circle cx="${x(i)}" cy="${y(v)}" r="${k === 0 ? 3.4 : 2.4}" fill="${k === 0 ? col : col}" stroke="#fff" stroke-width="1"><title>${D.months[i]}: ${v}</title></circle>`; });
  });
  s += `</svg>`;
  const nI = ref[1] ? ref[1][curVar].n : 0, nR = ref[2] ? ref[2][curVar].n : 0;
  s += `<div class="count" style="margin-top:4px">${D.vars[curVar].label}. Thick line = selected parcel. Dashed = class median, band = interquartile range, ${season} (irrigated n=${nI.toLocaleString()}, rainfed n=${nR.toLocaleString()}).</div>`;
  el.innerHTML = s; renderPins();
}
function renderPins() {
  document.getElementById('pins').innerHTML = pins.map((p, i) =>
    `<div class="pin"><span class="sw" style="background:${PINC[i % PINC.length]}"></span><span class="id" title="${p.id}">${CLS[p.cls].name} · ${p.id}</span><button data-i="${i}">×</button></div>`).join('');
  document.querySelectorAll('#pins button').forEach(b => b.addEventListener('click', () => { pins.splice(+b.dataset.i, 1); draw(); }));
}
refresh(true);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
