/* ===========================================================================
   ASL classifier UI

   Two paths to the same board: the packed demo images by index, and an
   uploaded photo. Both end in one byte back from the FPGA.
   =========================================================================== */

const $ = id => document.getElementById(id);

// Served by /api/info so it cannot drift from the RTL's cycle count.
let FPGA_MS = 0.0648;

let selected = 0;
let done = 0, right = 0;

/* ---------- helpers ---------- */

// Draw 784 grayscale bytes into a 28x28 canvas. This is the exact byte array
// the board was sent, so the canvas doubles as a check on what went out.
function paint(cv, pixels) {
  const ctx = cv.getContext('2d');
  const img = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) {
    const v = pixels[i];
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function setStatus(text, cls) {
  $('statusText').textContent = text;
  $('dot').className = 'dot' + (cls ? ' ' + cls : '');
}

function showErr(el, msg) { el.hidden = false; el.textContent = msg; }

function chip(text, cls) {
  return `<span class="chip${cls ? ' ' + cls : ''}">${text}</span>`;
}

// Split the round trip into the serial transfer and the actual compute. The
// FPGA segment is clamped to a visible sliver -- at ~0.1% of the bar it would
// otherwise round to nothing, which undersells the point rather than making it.
function drawMeter(totalMs) {
  const fpgaPct = Math.max(0.5, (FPGA_MS / totalMs) * 100);
  $('barWire').style.width = (100 - fpgaPct) + '%';
  $('barFpga').style.width = fpgaPct + '%';
  $('keyWire').textContent = `serial ${(totalMs - FPGA_MS).toFixed(1)} ms`;
  $('meter').hidden = false;
}

/* ---------- test images ---------- */

function resetResult() {
  $('letter').className = 'letter idle';
  $('letter').textContent = '–';
  $('chips').innerHTML = chip('not classified');
  $('meter').hidden = true;
  $('err').hidden = true;
}

async function select(i) {
  selected = i;
  for (const b of $('thumbs').children)
    b.setAttribute('aria-pressed', String(Number(b.dataset.i) === i));

  resetResult();

  const d = await (await fetch(`/api/image/${i}`)).json();
  paint($('canvas'), d.pixels);
  $('caption').textContent = `test image ${i}`;
  $('chips').innerHTML = chip(`truth ${d.ground_truth_letter} · class ${d.ground_truth}`);
}

async function classify() {
  $('go').disabled = true;
  $('err').hidden = true;
  setStatus('classifying', 'busy');

  try {
    const r = await fetch('/predict-test', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ index: selected })
    });
    const d = await r.json();

    if (!r.ok) {
      showErr($('err'), d.error || `HTTP ${r.status}`);
      setStatus('error', 'err');
      return;
    }

    done++;
    if (d.correct) right++;

    $('letter').textContent = d.predicted_letter;
    $('letter').className = 'letter ' + (d.correct ? 'ok' : 'bad');
    // Two independent claims, deliberately kept apart:
    //   matches model -> the ACCELERATOR computed the right thing
    //   correct       -> the MODEL was right about this image
    // The quantized model is ~92% accurate, so it can be wrong while the
    // hardware is perfect. Only a mismatch means the silicon is at fault.
    $('chips').innerHTML =
      chip(d.matches_model ? 'matches model' : 'HARDWARE ≠ MODEL',
           d.matches_model ? 'ok' : 'bad') +
      chip(d.correct ? 'correct' : 'incorrect', d.correct ? 'ok' : 'bad') +
      chip(`predicted ${d.predicted_letter} · class ${d.predicted_class}`) +
      chip(`truth ${d.ground_truth_letter} · class ${d.ground_truth}`) +
      chip(`${d.latency_ms} ms`);
    drawMeter(d.latency_ms);
    $('tally').textContent = `${right} / ${done} correct`;
    setStatus('ready');
  } catch (e) {
    showErr($('err'), String(e));
    setStatus('error', 'err');
  } finally {
    $('go').disabled = false;
  }
}

/* ---------- photo upload ---------- */

async function upload(file) {
  if (!file) return;

  $('uErr').hidden = true;
  $('shots').hidden = false;
  $('orig').src = URL.createObjectURL(file);   // show it before the round trip
  $('uLetter').className = 'letter idle';
  $('uLetter').textContent = '…';
  $('uMatch').hidden = true;
  $('uClass').textContent = 'sending 784 bytes';
  $('uLat').textContent = '—';
  setStatus('classifying', 'busy');

  const body = new FormData();
  body.append('file', file);

  try {
    const r = await fetch('/predict-upload', { method: 'POST', body });
    const d = await r.json();

    if (!r.ok) {
      $('uLetter').textContent = '–';
      $('uClass').textContent = 'failed';
      showErr($('uErr'), d.error || `HTTP ${r.status}`);
      setStatus('error', 'err');
      return;
    }

    $('prep').src = d.preview_png;
    $('uLetter').textContent = d.predicted_letter;
    // There is no ground truth for a photo, so the only verdict available is
    // whether the hardware agrees with the reference model on these bytes.
    // That is the question this panel can actually answer: a wrong letter on a
    // bad photo is the model's business, a mismatch here is the silicon's.
    $('uLetter').className = 'letter ' + (d.matches_model ? 'ok' : 'bad');
    $('uMatch').hidden = false;
    $('uMatch').textContent = d.matches_model
      ? 'hardware matches model'
      : `HARDWARE ≠ MODEL (model says ${d.golden_letter})`;
    $('uMatch').className = 'chip ' + (d.matches_model ? 'ok' : 'bad');
    $('uClass').textContent = `${d.predicted_letter} · class ${d.predicted_class}`;
    $('uLat').textContent = `${d.latency_ms} ms`;
    setStatus('ready');
  } catch (e) {
    $('uLetter').textContent = '–';
    showErr($('uErr'), String(e));
    setStatus('error', 'err');
  }
}

function wireUpload() {
  const drop = $('drop'), input = $('file');

  drop.onclick = () => input.click();
  drop.onkeydown = e => {
    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); input.click(); }
  };
  input.onchange = () => upload(input.files[0]);

  // preventDefault on dragover is what stops the browser from navigating to
  // the dropped file instead of handing it to the page.
  drop.ondragover  = e => { e.preventDefault(); drop.classList.add('over'); };
  drop.ondragleave = () => drop.classList.remove('over');
  drop.ondrop      = e => {
    e.preventDefault();
    drop.classList.remove('over');
    upload(e.dataTransfer.files[0]);
  };
}

/* ---------- boot ---------- */

(async function init() {
  const info = await (await fetch('/api/info')).json();
  if (info.fpga_ms) {
    FPGA_MS = info.fpga_ms;
    // Label the legend from the served value too, so the number in the UI and
    // the number in the bar always come from the same place.
    const k = document.querySelector('.k-fpga');
    if (k) k.textContent = `FPGA ${FPGA_MS} ms`;
  }

  // Thumbnails rather than numbered buttons: you pick the hand you can see.
  const imgs = await Promise.all(
    Array.from({ length: info.n_images },
               (_, i) => fetch(`/api/image/${i}`).then(r => r.json())));

  const frag = document.createDocumentFragment();
  const canvases = [];

  imgs.forEach((d, i) => {
    const b = document.createElement('button');
    b.className = 'thumb';
    b.dataset.i = i;
    b.setAttribute('aria-pressed', 'false');

    const cv = document.createElement('canvas');
    cv.width = cv.height = 28;
    b.appendChild(cv);
    canvases.push([cv, d.pixels]);

    const n = document.createElement('span');
    n.className = 'n';
    n.textContent = i;
    b.appendChild(n);

    const sr = document.createElement('span');
    sr.className = 'sr';
    sr.textContent = `test image ${i}, ground truth ${d.ground_truth_letter}`;
    b.appendChild(sr);

    b.onclick = () => select(i);
    frag.appendChild(b);
  });

  $('thumbs').appendChild(frag);
  for (const [cv, px] of canvases) paint(cv, px);

  $('foot').textContent =
    `${info.port} · ${info.baud} baud 8N1 · 24 classes (J and Z omitted — both need motion)`;
  $('go').onclick = classify;
  wireUpload();

  await select(0);

  // The board can be absent at load, or unplugged later. The server reopens the
  // port by itself on the next request, so this is a starting state, not a
  // terminal one -- classify anyway and it will reconnect if the board is back.
  setStatus(info.connected ? 'ready' : 'board offline',
            info.connected ? '' : 'err');
})();
