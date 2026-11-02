// @ts-nocheck
import workerSource from 'inline-worker:./detect/worker';
import { parseFormation, formatFormation, applyNorth, FBlock } from './formation';
import { createSolver } from '../solver';

const $ = (s: string) => document.querySelector(s) as any;
const view = $('#view') as HTMLCanvasElement, ctx = view.getContext('2d')!;
let img: HTMLImageElement | ImageBitmap | null = null, blocks: FBlock[] = [], sel = -1, quad: number[][] = [], drawing = false, drag = -1, solver: any = null, abort: AbortController | null = null, worker: Worker | null = null, ready = false, ref: number[][] | null = null;

function setStatus(t: string, k = '') { $('#statusText').textContent = t; $('#dot').className = 'dot ' + k; }
function log(t: string) { const l = $('#detLog'); l.classList.add('on'); l.textContent += t + '\n'; l.scrollTop = 1e9; }

/* texture swatches + reference */
const texImg = new Image(); texImg.src = 'grass_block_top.png';
const texReady = new Promise<void>(r => { texImg.onload = () => r(); });
texReady.then(() => {
  const c = document.createElement('canvas'); c.width = c.height = 16; const x = c.getContext('2d')!; x.drawImage(texImg, 0, 0, 16, 16);
  const d = x.getImageData(0, 0, 16, 16).data; ref = []; for (let y = 0; y < 16; y++) { ref.push([]); for (let xx = 0; xx < 16; xx++) ref[y].push(d[4 * (y * 16 + xx) + 1]); }
  document.querySelectorAll('.swatches canvas').forEach((cv: any) => { const g = cv.getContext('2d'); g.imageSmoothingEnabled = false; g.translate(8, 8); g.rotate(Number(cv.dataset.rot) * Math.PI / 2); g.drawImage(texImg, -8, -8, 16, 16); });
});

/* worker */
function startWorker() {
  worker = new Worker(URL.createObjectURL(new Blob([workerSource], { type: 'text/javascript' })));
  const base = new URL('.', location.href).href;
  worker.onmessage = e => {
    const m = e.data;
    if (m.type === 'ready') { ready = true; $('#detBackend').textContent = m.backend; setStatus('ready', 'ok'); upd(); }
    else if (m.type === 'log' || m.type === 'progress') { log(m.text); if (m.fraction != null) { $('#detBar').hidden = false; $('#detBar i').style.width = (m.fraction * 100) + '%'; } }
    else if (m.type === 'error') { log('error: ' + m.text); setStatus('detector error', 'err'); $('#detBar').hidden = true; busyDet(false); }
    else if (m.type === 'result') onResult(m.result);
  };
  worker.postMessage({ type: 'init', opencvUrl: base + 'vendor/opencv.js', wasmUrl: base + 'detect-wasm.wasm' });
  setStatus('loading detector', 'busy');
}
function busyDet(b: boolean) { $('#btnAuto').disabled = b || !ready || !img; $('#btnGrow').disabled = b || !ready || !img || quad.length < 4; if (!b) $('#detBar').hidden = true; }
function onResult(r: any) {
  busyDet(false); setStatus('ready', 'ok');
  if (!r.blocks || !r.blocks.length) { log((r.reason || 'nothing found') + ' (' + (r.seconds || 0).toFixed(1) + 's, ' + r.backend + ')'); return; }
  blocks = r.blocks.map((b: any) => ({ x: b.x, y: b.y, z: b.z, rotation: b.rotation, is_side: 0, include: true, score: b.score, quad: b.quad }));
  log('found ' + blocks.length + ' faces in ' + r.seconds.toFixed(1) + 's (' + r.backend + ')'); refresh();
}
function runDetect(seeds?: any) {
  if (!img || !ref || !ready) return;
  const c = document.createElement('canvas'); c.width = img.width; c.height = img.height; const x = c.getContext('2d')!; x.drawImage(img as any, 0, 0);
  const id = x.getImageData(0, 0, c.width, c.height); $('#detLog').textContent = ''; busyDet(true); setStatus('detecting', 'busy'); $('#detBar').hidden = false;
  worker!.postMessage({ type: 'detect', width: c.width, height: c.height, buffer: id.data.buffer, ref, seeds, opts: { accept: Number($('#thr').value) } }, [id.data.buffer]);
}
$('#btnAuto').onclick = () => runDetect();
$('#btnGrow').onclick = () => runDetect([{ quad: quad.map(p => p.slice()), manual: true }]);

/* image loading */
async function load(src: Blob | string) {
  const url = typeof src === 'string' ? src : URL.createObjectURL(src), im = new Image();
  await new Promise((res, rej) => { im.onload = res; im.onerror = rej; im.src = url; });
  img = im; blocks = []; quad = []; sel = -1; view.width = im.width; view.height = im.height; $('#wrap').classList.add('has'); $('#hint').textContent = im.width + 'x' + im.height; refresh(); busyDet(false);
}
$('#file').onchange = (e: any) => e.target.files[0] && load(e.target.files[0]);
const wrap = $('#wrap');
addEventListener('dragover', e => { e.preventDefault(); wrap.classList.add('over'); });
addEventListener('dragleave', () => wrap.classList.remove('over'));
addEventListener('drop', e => { e.preventDefault(); wrap.classList.remove('over'); const f = e.dataTransfer?.files[0]; if (f) load(f); });
addEventListener('paste', e => { const f = [...(e.clipboardData?.files || [])][0]; if (f) load(f); });
$('#btnHill').onclick = async () => {
  await load('examples/hill.png'); const j = await (await fetch('examples/hill.json')).json();
  blocks = j.blocks.map((b: any) => ({ x: b.x, y: b.y, z: b.z, rotation: b.rotation, is_side: 0, include: true, score: b.score, quad: b.quad }));
  $('#detLog').classList.add('on'); $('#detLog').textContent = 'FIXTURE: ' + j.note + '\n'; refresh();
};

/* canvas drawing + quad editing */
function draw() {
  ctx.clearRect(0, 0, view.width, view.height); if (!img) return; ctx.drawImage(img as any, 0, 0);
  const lw = Math.max(1.5, view.width / 500);
  blocks.forEach((b, i) => {
    if (!b.quad) return; ctx.beginPath(); b.quad.forEach((p, k) => k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); ctx.closePath();
    ctx.lineWidth = lw; ctx.strokeStyle = !b.include ? '#888' : i === sel ? '#ffd43b' : '#3ddc63'; if (i === sel) { ctx.fillStyle = '#ffd43b55'; ctx.fill(); } ctx.stroke();
    const c = b.quad.reduce((s, p) => [s[0] + p[0] / 4, s[1] + p[1] / 4], [0, 0]);
    const q = b.quad, up = [(q[0][0] + q[1][0]) / 2, (q[0][1] + q[1][1]) / 2]; // texture up edge midpoint at rotation 0
    const e = (r => [q[r][0], q[r][1], q[(r + 1) % 4][0], q[(r + 1) % 4][1]])((4 - b.rotation) % 4), m = [(e[0] + e[2]) / 2, (e[1] + e[3]) / 2];
    ctx.beginPath(); ctx.moveTo(c[0], c[1]); ctx.lineTo(c[0] + (m[0] - c[0]) * .7, c[1] + (m[1] - c[1]) * .7); ctx.strokeStyle = '#fff'; ctx.lineWidth = lw; ctx.stroke();
    ctx.fillStyle = '#000a'; const fs = Math.max(9, view.width / 90); ctx.font = '600 ' + fs + 'px ui-monospace,monospace'; const t = b.y + ' r' + b.rotation; ctx.fillRect(c[0] - fs * 1.5, c[1] - fs * .6, fs * 3, fs * 1.2); ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(t, c[0], c[1]);
  });
  if (quad.length) { ctx.beginPath(); quad.forEach((p, k) => k ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1])); if (quad.length === 4) ctx.closePath(); ctx.strokeStyle = '#d9531e'; ctx.lineWidth = lw * 1.5; ctx.stroke(); quad.forEach((p, k) => { ctx.beginPath(); ctx.arc(p[0], p[1], lw * 3, 0, 7); ctx.fillStyle = '#d9531e'; ctx.fill(); ctx.fillStyle = '#fff'; ctx.font = '700 ' + lw * 4 + 'px sans-serif'; ctx.fillText(String(k + 1), p[0] + lw * 5, p[1] - lw * 5); }); }
}
function pt(e: PointerEvent) { const r = view.getBoundingClientRect(); return [(e.clientX - r.left) * view.width / r.width, (e.clientY - r.top) * view.height / r.height]; }
function inside(q: number[][], p: number[]) { let c = false; for (let i = 0, j = 3; i < 4; j = i++) if ((q[i][1] > p[1]) !== (q[j][1] > p[1]) && p[0] < (q[j][0] - q[i][0]) * (p[1] - q[i][1]) / (q[j][1] - q[i][1]) + q[i][0]) c = !c; return c; }
view.onpointerdown = e => {
  if (!img) return; const p = pt(e), tol = view.width / 60;
  if (drawing && quad.length < 4) { quad.push(p); if (quad.length === 4) { drawing = false; $('#btnQuad').setAttribute('aria-pressed', 'false'); } busyDet(false); draw(); return; }
  drag = quad.findIndex(q => Math.hypot(q[0] - p[0], q[1] - p[1]) < tol);
  if (drag >= 0) { view.setPointerCapture(e.pointerId); return; }
  const i = blocks.findIndex(b => b.quad && inside(b.quad, p)); sel = i; refresh(); if (i >= 0) document.querySelector('#tbl tr[data-i="' + i + '"]')?.scrollIntoView({ block: 'nearest' });
};
view.onpointermove = e => { if (drag >= 0) { quad[drag] = pt(e); draw(); } };
view.onpointerup = () => { drag = -1; };
$('#btnQuad').onclick = () => { drawing = true; quad = []; $('#btnQuad').setAttribute('aria-pressed', 'true'); $('#hint').textContent = 'click 4 corners clockwise from top-left of the tile'; busyDet(false); draw(); };
$('#btnClearQuad').onclick = () => { quad = []; drawing = false; $('#btnQuad').setAttribute('aria-pressed', 'false'); busyDet(false); draw(); };

/* table */
function table() {
  const tb = $('#tbl tbody'); tb.innerHTML = '';
  blocks.forEach((b, i) => {
    const tr = document.createElement('tr'); tr.dataset.i = String(i); if (i === sel) tr.className = 'sel'; if (!b.include) tr.classList.add('off');
    const num = (k: string) => '<td><input type="number" data-k="' + k + '" value="' + b[k] + '"></td>';
    tr.innerHTML = '<td><input type="checkbox" data-k="include"' + (b.include ? ' checked' : '') + '></td>' + num('x') + num('y') + num('z') + '<td><input type="number" min="0" max="3" data-k="rotation" value="' + b.rotation + '"></td><td><input type="checkbox" data-k="is_side"' + (b.is_side ? ' checked' : '') + '></td><td>' + (b.score ? b.score.toFixed(2) : '') + '</td><td><button class="x" aria-label="delete row" data-del>×</button></td>';
    tb.appendChild(tr);
  });
  $('#count').textContent = blocks.filter(b => b.include).length + ' blocks';
}
$('#tbl').addEventListener('input', (e: any) => { const tr = e.target.closest('tr'); if (!tr) return; const b = blocks[+tr.dataset.i], k = e.target.dataset.k; b[k] = e.target.type === 'checkbox' ? (k === 'include' ? e.target.checked : +e.target.checked) : +e.target.value; $('#count').textContent = blocks.filter(b => b.include).length + ' blocks'; upd(); draw(); });
$('#tbl').addEventListener('click', (e: any) => { const tr = e.target.closest('tr'); if (!tr || e.target.tagName === 'INPUT') return; if (e.target.dataset.del !== undefined) { blocks.splice(+tr.dataset.i, 1); sel = -1; } else sel = +tr.dataset.i; refresh(); });
$('#btnAdd').onclick = () => { blocks.push({ x: 0, y: 0, z: 0, rotation: 0, is_side: 0, include: true }); refresh(); };
$('#btnImport').onclick = () => $('#importFile').click();
$('#importFile').onchange = async (e: any) => { const f = e.target.files[0]; if (!f) return; try { blocks = parseFormation(await f.text()); sel = -1; refresh(); } catch (er: any) { alert('Could not read formation: ' + er.message); } };
function download(name: string, text: string) { const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], { type: 'text/plain' })); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
$('#btnExport').onclick = () => download('formation.txt', formatFormation(blocks, ['exported by grass cracker, north 0 frame']));
$('#btnNative').onclick = () => { const ns = norths(); ns.forEach(n => download('formation-north' + n + '.txt', formatFormation(applyNorth(blocks, n)))); };
function refresh() { table(); draw(); upd(); }

/* search */
function norths() { return [...document.querySelectorAll('.n')].filter((c: any) => c.checked).map((c: any) => +c.value); }
function bounds() { const g = (id: string) => +$('#' + id).value; return { xmin: g('xmin'), xmax: g('xmax'), ymin: g('ymin'), ymax: g('ymax'), zmin: g('zmin'), zmax: g('zmax') }; }
function upd() {
  $('#btnRun').disabled = !blocks.some(b => b.include) || !!abort;
  const b = bounds(), n = norths(); $('#cmd').classList.toggle('on', blocks.length > 0);
  $('#cmd').textContent = 'cuda:  grass_solver.exe ' + [b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax].join(' ') + '\nmetal: ./grass_solver ' + [b.xmin, b.xmax, b.ymin, b.ymax, b.zmin, b.zmax].join(' ') + '\n(run once per formation-north*.txt, copied to formation.txt; norths ' + n.join(',') + ')';
}
document.querySelectorAll('.bounds input,.n').forEach(e => e.addEventListener('input', upd));
document.querySelectorAll('.chip').forEach((c: any) => c.onclick = () => { const r = +c.dataset.r; ['xmin', 'zmin'].forEach(i => $('#' + i).value = -r); ['xmax', 'zmax'].forEach(i => $('#' + i).value = r); upd(); });
$('#btnCancel').onclick = () => abort?.abort();
$('#btnRun').onclick = async () => {
  const be = $('#backend').value; if (be === 'gpu' && !('gpu' in navigator)) { $('#runStats').textContent = 'WebGPU is not available in this browser. Pick CPU or use the native download below.'; return; }
  abort = new AbortController(); $('#btnRun').disabled = true; $('#btnCancel').disabled = false; $('#matches').innerHTML = ''; $('#runBar').hidden = false; setStatus('searching', 'busy');
  try {
    if (!solver || solver.__be !== be) { solver = await createSolver({ backend: be }); solver.__be = be; } const t0 = performance.now();
    const r = await solver.search(blocks.filter(b => b.include).map(b => ({ x: b.x, y: b.y, z: b.z, rotation: b.rotation, is_side: b.is_side })), { ...bounds(), north: norths(), signal: abort.signal,
      onProgress: (p: any) => { $('#runBar i').style.width = (100 * p.done / p.total) + '%'; $('#runStats').textContent = (100 * p.done / p.total).toFixed(1) + '%  ' + (p.rate / 1e9).toFixed(2) + ' G/s  ' + p.seconds.toFixed(1) + 's  ' + (solver.backend + ' ' + (solver.adapterInfo?.description || solver.adapterInfo?.vendor || '')); },
      onMatch: (m: any) => { const li = document.createElement('li'); li.textContent = JSON.stringify(m); $('#matches').appendChild(li); } });
    $('#runStats').textContent = (r.cancelled ? 'cancelled' : 'done') + ': ' + r.matches.length + ' match(es), ' + r.seconds.toFixed(1) + 's';
  } catch (e: any) { $('#runStats').textContent = 'error: ' + (e.message || e); }
  abort = null; $('#btnCancel').disabled = true; $('#runBar').hidden = true; setStatus('ready', 'ok'); upd();
};
texReady.then(startWorker); upd();
