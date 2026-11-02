/* =========================================================
   JADI IKAN — MVP
   Kontrol: mangap (jawOpen) = naik, tutup mulut = turun.
   Deteksi wajah: MediaPipe Face Landmarker via CDN.
   ========================================================= */

const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0";
const MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

const TAU = Math.PI * 2;
const clamp = (v, a, b) => (v < a ? a : v > b ? b : v);
const rand = (a, b) => a + Math.random() * (b - a);

/* ---------- DOM ---------- */
const canvas = document.getElementById("game");
const ctx = canvas.getContext("2d");
const video = document.getElementById("cam");
const elHud = document.getElementById("hud");
const elScore = document.getElementById("score");
const elHint = document.getElementById("hint");
const elJaw = document.getElementById("jawfill");
const elCamNote = document.getElementById("camnote");
const screenStart = document.getElementById("screen-start");
const screenOver = document.getElementById("screen-over");
const btnCam = document.getElementById("btn-cam");
const btnTouch = document.getElementById("btn-touch");
const btnRetry = document.getElementById("btn-retry");
const camErr = document.getElementById("cam-err");
const overNote = document.getElementById("over-note");
const finalScore = document.getElementById("final-score");
const bestScore = document.getElementById("best-score");
const btnShare = document.getElementById("btn-share");
const snapsWrap = document.getElementById("snaps");
const snapRow = document.getElementById("snap-row");
const shareNote = document.getElementById("share-note");

/* ---------- Ukuran ---------- */
let W = 0,
  H = 0,
  DPR = 1;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  W = window.innerWidth;
  H = window.innerHeight;
  canvas.width = Math.round(W * DPR);
  canvas.height = Math.round(H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
}
window.addEventListener("resize", resize);
window.addEventListener("orientationchange", () => setTimeout(resize, 120));
resize();

/* ---------- State ---------- */
const START = 0,
  PLAY = 1,
  OVER = 2;
let state = START;

/* ---------- Audio (SFX air halus, tanpa ambient latar) ---------- */
let audioCtx = null;

function ensureAudio() {
  if (!audioCtx) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (AudioContextClass) {
      audioCtx = new AudioContextClass();
    }
  }
  if (audioCtx && audioCtx.state === "suspended") {
    audioCtx.resume().catch(() => {});
  }
}

function blubOnce() {
  try {
    ensureAudio();
    if (!audioCtx) return;

    const t = audioCtx.currentTime;
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();

    osc.type = "sine";

    // Suara beep tetesan air / gelembung halus:
    // Nada menyapu naik lembut dari ~440Hz ke ~920Hz
    const baseFreq = 440 + Math.random() * 30;
    osc.frequency.setValueAtTime(baseFreq, t);
    osc.frequency.exponentialRampToValueAtTime(baseFreq * 2.1, t + 0.08);

    // Envelope lembut: attack 5ms agar bebas klik, decay eksponensial halus ~0.11s
    gain.gain.setValueAtTime(0.0001, t);
    gain.gain.linearRampToValueAtTime(0.2, t + 0.005);
    gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.11);

    osc.connect(gain);
    gain.connect(audioCtx.destination);

    osc.start(t);
    osc.stop(t + 0.12);
  } catch (e) {
    console.warn("audio play failed", e);
  }
}

let mode = "touch"; // "camera" | "touch"
let landmarker = null;
let stream = null;
let lastVideoTime = -1;
let lastTs = -1;
let noFaceFrames = 0;

let mouthRaw = 0; // skor jawOpen 0..1
let mouthOpen = false; // hasil histeresis
let touchHeld = false;
let mouthVisual = 0; // mulut ikan (halus)

let fish = null;
let obstacles = [];
let bubbles = [];
let ambient = [];
let score = 0;
let best = Number(localStorage.getItem("flappyikan.best") || 0);
let speed = 0;
let sinceSpawn = 0;
let spawnSpacing = 0;
let prevGapY = 0;
let shake = 0;
let hintTimer = 0;

// potret lucu
let snaps = []; // canvas 240x240 hasil screenshot wajah
let shownFaces = [];
let lastBox = null; // kotak wajah terakhir (px di ruang video)
let nextSnap = 999;
const SNAP_SIZE = 240;
const MAX_SNAPS = 12;

const FISH_R = 16;

/* =========================================================
   KAMERA + FACE LANDMARKER
   ========================================================= */
async function startCamera() {
  camErr.textContent = "";
  btnCam.disabled = true;
  btnCam.textContent = "Meminta izin kamera…";

  // kamera cuma boleh di origin aman (https / localhost)
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    camErr.textContent =
      "Kamera tidak dapat diakses melalui HTTP (bukan localhost). Silakan gunakan HTTPS atau pilih mode sentuh.";
    btnCam.disabled = false;
    btnCam.textContent = "Nyalakan Kamera";
    return;
  }

  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
  } catch (e) {
    camErr.textContent =
      "Izin kamera ditolak atau tidak tersedia. Silakan gunakan mode sentuh.";
    btnCam.disabled = false;
    btnCam.textContent = "Nyalakan Kamera";
    return;
  }

  video.srcObject = stream;
  try {
    await video.play();
  } catch (_) {}

  btnCam.textContent = "Memuat model wajah…";
  try {
    const mod = await import(CDN);
    const vision = await mod.FilesetResolver.forVisionTasks(`${CDN}/wasm`);
    landmarker = await mod.FaceLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: MODEL, delegate: "GPU" },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: true,
      minFaceDetectionConfidence: 0.4,
      minFacePresenceConfidence: 0.4,
    });
  } catch (e) {
    console.error("MediaPipe gagal:", e);
    landmarker = null;
    mode = "touch";
    startGame("Model wajah gagal dimuat — beralih ke mode sentuh.");
    return;
  }

  mode = "camera";
  startGame("Buka mulut = naik • Tutup mulut = turun");
}

function detectFace(ts) {
  let t = ts;
  if (t <= lastTs) t = lastTs + 1;
  lastTs = t;

  let res;
  try {
    res = landmarker.detectForVideo(video, t);
  } catch (e) {
    return;
  }

  const bs = res.faceBlendshapes && res.faceBlendshapes[0];
  if (!bs || !res.faceLandmarks || !res.faceLandmarks.length) {
    noFaceFrames++;
    if (noFaceFrames > 8) {
      mouthRaw = 0;
      mouthOpen = false;
      elCamNote.classList.add("on");
      elCamNote.textContent = "Wajah belum terdeteksi";
    }
    return;
  }
  noFaceFrames = 0;
  elCamNote.classList.remove("on");

  // lacak posisi wajah buat crop screenshot
  const pts = res.faceLandmarks[0];
  const vw = video.videoWidth;
  const vh = video.videoHeight;
  if (pts.length && vw && vh) {
    let x0 = 1, y0 = 1, x1 = 0, y1 = 0;
    for (const p of pts) {
      if (p.x < x0) x0 = p.x;
      if (p.x > x1) x1 = p.x;
      if (p.y < y0) y0 = p.y;
      if (p.y > y1) y1 = p.y;
    }
    const side = Math.max((x1 - x0) * vw, (y1 - y0) * vh) * 1.7;
    lastBox = {
      cx: ((x0 + x1) / 2) * vw,
      cy: ((y0 + y1) / 2) * vh,
      side: clamp(side, 40, Math.min(vw, vh)),
    };
  }

  const cat = bs.categories.find((c) => c.categoryName === "jawOpen");
  mouthRaw = cat ? cat.score : 0;
  elJaw.style.width = Math.round(clamp(mouthRaw, 0, 1) * 100) + "%";

  // histeresis biar nggak kedip-kedip
  if (mouthRaw > 0.3) {
    if (!mouthOpen && state === PLAY) {
      blubOnce();
    }
    mouthOpen = true;
  } else if (mouthRaw < 0.16) {
    mouthOpen = false;
  }
}

/* =========================================================
   GAME
   ========================================================= */
function reset() {
  fish = {
    x: W * 0.3,
    y: H * 0.46,
    vy: 0,
    r: FISH_R,
    angle: 0,
  };
  obstacles = [];
  bubbles = [];
  score = 0;
  speed = Math.max(95, W * 0.26);
  spawnSpacing = Math.max(330, W * 0.95);
  sinceSpawn = 0;
  prevGapY = H * 0.46;
  shake = 0;
  elScore.textContent = "0";

  // rintangan pertama jauh biar pemula sempat adaptasi
  obstacles.push({ x: W + 330, gapY: prevGapY, gapH: gapSize(), scored: false });
}

function gapSize() {
  return clamp(H * 0.46, 240, 380);
}

function spawn() {
  const gapH = gapSize();
  const min = gapH / 2 + 64;
  const max = H - gapH / 2 - 64;
  let gapY =
    min >= max
      ? H / 2
      : clamp(prevGapY + rand(-1, 1) * H * 0.15, min, max);
  prevGapY = gapY;
  obstacles.push({ x: W + 60, gapY, gapH, scored: false });
}

function showHint(text, secs = 3) {
  elHint.textContent = text;
  elHint.classList.add("on");
  hintTimer = secs;
}

function startGame(hintText) {
  reset();
  state = PLAY;
  screenStart.classList.add("hidden");
  screenOver.classList.add("hidden");
  elHud.classList.remove("hidden");
  nextSnap = rand(1.2, 2.4);
  shownFaces = [];
  shareNote.textContent = "";
  if (hintText) showHint(hintText, 3.2);
  ensureAudio();
}

function die() {
  if (state !== PLAY) return;
  state = OVER;
  shake = 14;
  captureSnap(); // potret dadakan pas kena pancing
  renderSnaps();
  if (score > best) {
    best = score;
    localStorage.setItem("flappyikan.best", String(best));
  }
  finalScore.textContent = String(score);
  bestScore.textContent = String(best);
  overNote.textContent = mode === "camera" ? "" : "Tips: Aktifkan kamera untuk bermain dengan gerakan mulut";
  blubOnce();
  setTimeout(() => screenOver.classList.remove("hidden"), 420);
}

/* =========================================================
   POTRET LUCU + KARTU SHARE
   ========================================================= */
function captureSnap() {
  if (mode !== "camera" || !video.videoWidth || !lastBox || noFaceFrames > 0) return null;

  const vw = video.videoWidth;
  const vh = video.videoHeight;
  const side = clamp(lastBox.side, 40, Math.min(vw, vh));
  const sx = clamp(lastBox.cx - side / 2, 0, Math.max(0, vw - side));
  const sy = clamp(lastBox.cy - side / 2, 0, Math.max(0, vh - side));

  const c = document.createElement("canvas");
  c.width = SNAP_SIZE;
  c.height = SNAP_SIZE;
  const g = c.getContext("2d");
  g.save();
  g.translate(SNAP_SIZE, 0);
  g.scale(-1, 1); // mirror biar sama kayak preview
  g.drawImage(video, sx, sy, side, side, 0, 0, SNAP_SIZE, SNAP_SIZE);
  g.restore();

  snaps.push(c);
  while (snaps.length > MAX_SNAPS) snaps.shift();
  return c;
}

function pickFaces() {
  const pool = snaps.slice();
  const out = [];
  while (out.length < 3 && pool.length) {
    out.push(pool.splice(Math.floor(Math.random() * pool.length), 1)[0]);
  }
  return out;
}

function renderSnaps() {
  shownFaces = pickFaces();
  snapRow.innerHTML = "";
  if (!shownFaces.length) {
    snapsWrap.classList.add("hidden");
    return;
  }
  snapsWrap.classList.remove("hidden");
  for (const f of shownFaces) {
    const box = document.createElement("div");
    box.className = "snap";
    box.appendChild(f);
    snapRow.appendChild(box);
  }
}

function rr(g, x, y, w, h, r) {
  if (g.roundRect) {
    g.beginPath();
    g.roundRect(x, y, w, h, r);
    return;
  }
  g.beginPath();
  g.moveTo(x + r, y);
  g.arcTo(x + w, y, x + w, y + h, r);
  g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r);
  g.arcTo(x, y, x + w, y, r);
  g.closePath();
}

function cardFish(g, cx, cy, r) {
  g.save();
  g.translate(cx, cy);
  g.lineJoin = "round";
  g.lineWidth = 7;
  g.strokeStyle = "#7c2d12";

  g.beginPath();
  g.moveTo(-r * 0.65, 0);
  g.lineTo(-r * 1.5, -r * 0.8);
  g.lineTo(-r * 1.5, r * 0.8);
  g.closePath();
  g.fillStyle = "#f97316";
  g.fill();
  g.stroke();

  g.beginPath();
  g.ellipse(0, 0, r, r * 0.8, 0, 0, TAU);
  g.fillStyle = "#fdba74";
  g.fill();

  g.save();
  g.clip();
  g.fillStyle = "rgba(255,255,255,.95)";
  g.fillRect(-r * 0.35, -r, r * 0.22, r * 2);
  g.fillRect(r * 0.3, -r, r * 0.18, r * 2);
  g.restore();

  g.beginPath();
  g.ellipse(0, 0, r, r * 0.8, 0, 0, TAU);
  g.stroke();

  g.beginPath();
  g.arc(r * 0.45, -r * 0.3, r * 0.2, 0, TAU);
  g.fillStyle = "#fff";
  g.fill();
  g.beginPath();
  g.arc(r * 0.5, -r * 0.3, r * 0.1, 0, TAU);
  g.fillStyle = "#111";
  g.fill();

  g.restore();
}

function buildShareCard(faces) {
  const cw = 1080;
  const ch = 1350;
  const c = document.createElement("canvas");
  c.width = cw;
  c.height = ch;
  const g = c.getContext("2d");
  const font = (px, w) => `${w} ${px}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

  // latar laut
  const bg = g.createLinearGradient(0, 0, 0, ch);
  bg.addColorStop(0, "#38bdf8");
  bg.addColorStop(0.45, "#0e8fc4");
  bg.addColorStop(1, "#053b5c");
  g.fillStyle = bg;
  g.fillRect(0, 0, cw, ch);

  // sinar matahari
  g.save();
  g.globalCompositeOperation = "lighter";
  for (let i = 0; i < 4; i++) {
    const x = 90 + i * 290;
    const ray = g.createLinearGradient(x, 0, x + 140, ch);
    ray.addColorStop(0, "rgba(255,255,255,.14)");
    ray.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = ray;
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x + 80, 0);
    g.lineTo(x + 230, ch);
    g.lineTo(x + 100, ch);
    g.closePath();
    g.fill();
  }
  g.restore();

  // gelembung
  g.strokeStyle = "rgba(255,255,255,.5)";
  g.lineWidth = 4;
  for (let i = 0; i < 24; i++) {
    g.beginPath();
    g.arc((i * 149) % cw, (i * 317) % ch, 7 + (i % 5) * 6, 0, TAU);
    g.stroke();
  }

  // pancingan dekoratif kiri & kanan
  const rig = (x) => {
    g.lineCap = "round";
    g.strokeStyle = "rgba(3,30,46,.5)";
    g.lineWidth = 8;
    g.beginPath();
    g.moveTo(x, 0);
    g.lineTo(x, ch * 0.26);
    g.stroke();
    g.strokeStyle = "rgba(226,244,255,.85)";
    g.lineWidth = 4;
    g.stroke();
    g.beginPath();
    g.moveTo(x, ch * 0.26);
    g.bezierCurveTo(x, ch * 0.26 + 36, x - 6, ch * 0.26 + 54, x - 30, ch * 0.26 + 54);
    g.stroke();
  };
  rig(78);
  rig(cw - 78);

  g.textAlign = "center";
  g.textBaseline = "alphabetic";

  // judul
  const tg = g.createLinearGradient(0, 120, 0, 250);
  tg.addColorStop(0, "#fef3c7");
  tg.addColorStop(1, "#fb923c");
  g.fillStyle = tg;
  g.font = font(112, 900);
  g.shadowColor = "rgba(0,0,0,.35)";
  g.shadowOffsetY = 8;
  g.fillText("JADI IKAN", cw / 2, 232);
  g.shadowOffsetY = 0;
  g.shadowColor = "transparent";

  // panel skor
  const px = 110;
  const py = 300;
  const pw = cw - 220;
  const ph = 430;
  g.fillStyle = "rgba(4,38,58,.72)";
  rr(g, px, py, pw, ph, 44);
  g.fill();
  g.strokeStyle = "rgba(255,255,255,.2)";
  g.lineWidth = 4;
  g.stroke();

  if ("letterSpacing" in g) g.letterSpacing = "14px";
  g.fillStyle = "#8fb6cf";
  g.font = font(46, 800);
  g.fillText("SKOR", cw / 2, py + 96);
  if ("letterSpacing" in g) g.letterSpacing = "0px";

  g.fillStyle = "#fff";
  g.font = font(230, 900);
  g.fillText(String(score), cw / 2, py + 300);

  g.fillStyle = "#ffd7a1";
  g.font = font(52, 800);
  g.fillText("TERBAIK  " + best, cw / 2, py + 384);

  // 3 wajah acak
  const n = faces.length;
  const fy = 930;
  const rad = 130;
  const xs =
    n === 1
      ? [cw / 2]
      : n === 2
      ? [cw / 2 - 175, cw / 2 + 175]
      : [cw / 2 - 195, cw / 2, cw / 2 + 195];

  if (!n) {
    cardFish(g, cw / 2, fy, 140);
  } else {
    for (let i = 0; i < n; i++) {
      const cx = xs[i];
      g.save();
      g.beginPath();
      g.arc(cx, fy, rad, 0, TAU);
      g.closePath();
      g.clip();
      try {
        g.drawImage(faces[i], cx - rad, fy - rad, rad * 2, rad * 2);
      } catch (e) {
        g.fillStyle = "#0b3a52";
        g.fillRect(cx - rad, fy - rad, rad * 2, rad * 2);
      }
      g.restore();

      g.beginPath();
      g.arc(cx, fy, rad, 0, TAU);
      g.strokeStyle = "#fff";
      g.lineWidth = 10;
      g.stroke();
      g.beginPath();
      g.arc(cx, fy, rad + 13, 0, TAU);
      g.strokeStyle = "rgba(251,146,60,.9)";
      g.lineWidth = 6;
      g.stroke();
    }
  }

  // caption
  g.fillStyle = "#eaf6ff";
  g.font = font(54, 800);
  g.fillText("Kontrol menggunakan gerakan mulut", cw / 2, 1135);
  g.fillStyle = "#a5e3ff";
  g.font = font(38, 700);
  g.fillText("Buka mulut = naik  •  Tutup mulut = turun", cw / 2, 1195);

  // watermark
  const wmText = "https://jadiikan.chasan.id";
  g.font = font(34, 700);
  const tm = g.measureText(wmText);
  const ww = tm.width + 56;
  const wh = 54;
  const wx = (cw - ww) / 2;
  const wy = 1242;

  g.fillStyle = "rgba(4,38,58,.72)";
  rr(g, wx, wy, ww, wh, 27);
  g.fill();
  g.strokeStyle = "rgba(255,255,255,.22)";
  g.lineWidth = 3;
  g.stroke();

  g.fillStyle = "#fef3c7";
  g.textBaseline = "middle";
  g.fillText(wmText, cw / 2, wy + wh / 2);
  g.textBaseline = "alphabetic";

  return c;
}

async function doShare() {
  shareNote.textContent = "Menyiapkan gambar…";

  let card;
  try {
    card = buildShareCard(shownFaces);
  } catch (e) {
    console.error("gagal bikin kartu share", e);
    shareNote.textContent = "Gagal membuat gambar.";
    return;
  }

  const blob = await new Promise((res) => card.toBlob(res, "image/png"));
  if (!blob) {
    shareNote.textContent = "Gagal membuat gambar.";
    return;
  }

  const text = `Aku meraih skor ${score} di Jadi Ikan! Main langsung di https://jadiikan.chasan.id 🐟`;

  try {
    const file = new File([blob], `jadi-ikan-${score}.png`, { type: "image/png" });
    const data = { files: [file], title: "Jadi Ikan", text };
    if (navigator.share && navigator.canShare && navigator.canShare(data)) {
      await navigator.share(data);
      shareNote.textContent = "";
      return;
    }
  } catch (e) {
    if (e && e.name === "AbortError") {
      shareNote.textContent = "Berbagi dibatalkan.";
      return;
    }
    console.warn("share gagal, fallback ke unduh", e);
  }

  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `jadi-ikan-${score}.png`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 8000);
  shareNote.textContent = "Gambar berhasil disimpan. Siap dibagikan ke media sosial.";
}

function update(dt, t) {
  const open = (mouthOpen || touchHeld) && state !== OVER;
  mouthVisual += ((open ? 1 : 0) - mouthVisual) * Math.min(1, 16 * dt);

  if (hintTimer > 0) {
    hintTimer -= dt;
    if (hintTimer <= 0) elHint.classList.remove("on");
  }
  if (shake > 0) shake = Math.max(0, shake - dt * 40);

  // gelembung ambient
  if (ambient.length < 26 && Math.random() < 0.08) {
    ambient.push({
      x: rand(0, W),
      y: H + 10,
      r: rand(2, 6),
      v: rand(18, 46),
      dr: rand(-8, 8),
      a: rand(0.15, 0.45),
    });
  }
  for (const b of ambient) {
    b.y -= b.v * dt;
    b.x += b.dr * dt;
  }
  ambient = ambient.filter((b) => b.y > -12);

  for (const b of bubbles) {
    b.y -= b.v * dt;
    b.x -= 12 * dt;
    b.life -= dt;
    b.r += dt * 3;
  }
  bubbles = bubbles.filter((b) => b.life > 0);

  if (state === START) {
    fish.y = H * 0.46 + Math.sin(t * 2) * 10;
    fish.angle = Math.sin(t * 2 + 1) * 0.08;
    return;
  }

  if (state === OVER) {
    fish.vy = Math.min(fish.vy + 1400 * dt, 900);
    fish.y += fish.vy * dt;
    fish.angle = Math.min(fish.angle + dt * 2.2, 1.3);
    return;
  }

  /* --- PLAY --- */
  if (mode === "camera") {
    nextSnap -= dt;
    if (nextSnap <= 0) {
      captureSnap();
      nextSnap = rand(1.6, 3.2);
    }
  }

  const targetVy = open ? -H * 0.42 : H * 0.62;
  const k = 1 - Math.exp(-6.5 * dt);
  fish.vy += (targetVy - fish.vy) * k;
  fish.vy = clamp(fish.vy, -H * 1.4, H * 1.6);
  fish.y += fish.vy * dt;
  fish.angle = clamp(Math.atan2(fish.vy, H * 0.8), -0.55, 1.05);

  if (fish.y - fish.r < 0) {
    fish.y = fish.r;
    fish.vy = Math.max(fish.vy, 60);
  }

  if (open && Math.random() < 0.25) {
    bubbles.push({
      x: fish.x + fish.r * 1.1,
      y: fish.y + rand(-4, 6),
      r: rand(2, 4),
      v: rand(30, 70),
      life: rand(0.5, 1.1),
    });
  }

  const move = speed * dt;
  for (const o of obstacles) o.x -= move;
  sinceSpawn += move;
  if (sinceSpawn >= spawnSpacing) {
    sinceSpawn = 0;
    spawn();
  }
  obstacles = obstacles.filter((o) => o.x > -90);

  for (const o of obstacles) {
    if (!o.scored && o.x < fish.x) {
      o.scored = true;
      score++;
      elScore.textContent = String(score);
      showHint(score % 5 === 0 ? `Keren! ${score}` : "!", 0.8);
    }
  }

  const baseSpeed = Math.max(95, W * 0.26);
  speed = Math.min(baseSpeed * 1.3, baseSpeed + score * 2);

  if (checkHit()) die();
  if (fish.y + fish.r > H) die();
}

function segDist(px, py, x1, y1, x2, y2) {
  const dx = x2 - x1,
    dy = y2 - y1;
  const len2 = dx * dx + dy * dy || 1;
  let u = ((px - x1) * dx + (py - y1) * dy) / len2;
  u = clamp(u, 0, 1);
  const cx = x1 + u * dx,
    cy = y1 + u * dy;
  return Math.hypot(px - cx, py - cy);
}

function checkHit() {
  const f = fish;
  for (const o of obstacles) {
    if (o.x < f.x - 60 || o.x > f.x + 60) continue;
    const gapT = o.gapY - o.gapH / 2;
    const gapB = o.gapY + o.gapH / 2;
    if (segDist(f.x, f.y, o.x, -30, o.x, gapT) < f.r) return true;
    if (segDist(f.x, f.y, o.x, gapB, o.x, H + 30) < f.r) return true;
    // kail (lengkung ke kiri)
    if (Math.hypot(f.x - (o.x - 11), f.y - (gapT + 25)) < f.r + 6) return true;
    if (Math.hypot(f.x - (o.x - 11), f.y - (gapB - 25)) < f.r + 6) return true;
  }
  return false;
}

/* =========================================================
   GAMBAR
   ========================================================= */
function drawBackground(t) {
  const g = ctx.createLinearGradient(0, 0, 0, H);
  g.addColorStop(0, "#38bdf8");
  g.addColorStop(0.45, "#0e8fc4");
  g.addColorStop(1, "#053b5c");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, W, H);

  // sinar matahari
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (let i = 0; i < 3; i++) {
    const sway = Math.sin(t * 0.35 + i * 1.7) * 34;
    const x = W * (0.18 + i * 0.3) + sway;
    const gg = ctx.createLinearGradient(x, 0, x + 70, H * 0.9);
    gg.addColorStop(0, "rgba(255,255,255,0.16)");
    gg.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = gg;
    ctx.beginPath();
    ctx.moveTo(x - 26, -10);
    ctx.lineTo(x + 34, -10);
    ctx.lineTo(x + 120, H);
    ctx.lineTo(x + 20, H);
    ctx.closePath();
    ctx.fill();
  }
  ctx.restore();

  // gelembung ambient
  ctx.strokeStyle = "rgba(255,255,255,0.75)";
  for (const b of ambient) {
    ctx.globalAlpha = b.a;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, TAU);
    ctx.stroke();
  }
  ctx.globalAlpha = 1;
}

function drawHookDown(x, y) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  // mata kail ke bawah, melengkung ke kiri
  ctx.strokeStyle = "#0f2b3d";
  ctx.lineWidth = 7;
  ctx.beginPath();
  ctx.moveTo(x, y);
  ctx.lineTo(x, y + 9);
  ctx.bezierCurveTo(x, y + 24, x - 3, y + 31, x - 12, y + 31);
  ctx.bezierCurveTo(x - 20, y + 31, x - 23, y + 24, x - 21, y + 16);
  ctx.stroke();
  ctx.strokeStyle = "#dbe7ef";
  ctx.lineWidth = 4;
  ctx.stroke();
  // barb
  ctx.beginPath();
  ctx.moveTo(x - 21, y + 15);
  ctx.lineTo(x - 15, y + 21);
  ctx.strokeStyle = "#f1f7fb";
  ctx.lineWidth = 3;
  ctx.stroke();
  ctx.restore();
}

function drawLine(x, y1, y2) {
  ctx.save();
  ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(8,40,60,0.45)";
  ctx.lineWidth = 5;
  ctx.beginPath();
  ctx.moveTo(x, y1);
  ctx.lineTo(x, y2);
  ctx.stroke();
  ctx.strokeStyle = "rgba(226,244,255,0.92)";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawRig(o) {
  const gapT = o.gapY - o.gapH / 2;
  const gapB = o.gapY + o.gapH / 2;
  const x = o.x;

  // pelampung kecil di ujung luar
  drawBobber(x, 8, 1);
  drawBobber(x, H - 8, -1);

  drawLine(x, -20, gapT);
  drawLine(x, gapB, H + 20);

  drawHookDown(x, gapT);

  // kail bawah (dicerminkan)
  ctx.save();
  ctx.translate(x, gapB);
  ctx.scale(1, -1);
  drawHookDown(0, 0);
  ctx.restore();
}

function drawBobber(x, y, dir) {
  ctx.save();
  ctx.translate(x, y);
  ctx.beginPath();
  ctx.ellipse(0, 0, 9, 12, 0, 0, TAU);
  ctx.fillStyle = "#ef4444";
  ctx.fill();
  ctx.beginPath();
  ctx.rect(-9, dir > 0 ? -12 : 0, 18, 12);
  ctx.fillStyle = "#f8fafc";
  ctx.fill();
  ctx.strokeStyle = "rgba(3,30,46,.5)";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.ellipse(0, 0, 9, 12, 0, 0, TAU);
  ctx.stroke();
  ctx.restore();
}

function drawFish(t) {
  const R = fish.r;
  ctx.save();
  ctx.translate(fish.x, fish.y);
  ctx.rotate(fish.angle);

  // ekor
  const wag = Math.sin(t * 9) * 0.3;
  ctx.save();
  ctx.translate(-R * 0.85, 0);
  ctx.rotate(wag);
  ctx.beginPath();
  ctx.moveTo(4, 0);
  ctx.lineTo(-R * 0.95, -R * 0.8);
  ctx.quadraticCurveTo(-R * 0.45, 0, -R * 0.95, R * 0.8);
  ctx.closePath();
  ctx.fillStyle = "#f97316";
  ctx.fill();
  ctx.strokeStyle = "#7c2d12";
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();

  // sirip punggung
  ctx.beginPath();
  ctx.moveTo(-R * 0.35, -R * 0.7);
  ctx.quadraticCurveTo(R * 0.05, -R * 1.35, R * 0.45, -R * 0.62);
  ctx.closePath();
  ctx.fillStyle = "#fb923c";
  ctx.fill();
  ctx.strokeStyle = "#7c2d12";
  ctx.lineWidth = 1.6;
  ctx.stroke();

  // badan
  ctx.save();
  ctx.beginPath();
  ctx.ellipse(0, 0, R * 1.08, R * 0.82, 0, 0, TAU);
  ctx.fillStyle = "#fdba74";
  ctx.fill();
  ctx.clip();

  // belang
  ctx.fillStyle = "rgba(255,255,255,0.95)";
  ctx.fillRect(-R * 0.45, -R, R * 0.3, R * 2);
  ctx.fillRect(R * 0.32, -R, R * 0.24, R * 2);

  // mulut (digambar di clip badan)
  const h = R * (0.05 + 0.62 * mouthVisual);
  ctx.beginPath();
  ctx.moveTo(-R * 0.1, 0);
  ctx.lineTo(R * 1.2, -h);
  ctx.lineTo(R * 1.2, h);
  ctx.closePath();
  ctx.fillStyle = "#4a0f16";
  ctx.fill();
  ctx.restore();

  // garis badan
  ctx.beginPath();
  ctx.ellipse(0, 0, R * 1.08, R * 0.82, 0, 0, TAU);
  ctx.strokeStyle = "#7c2d12";
  ctx.lineWidth = 2;
  ctx.stroke();

  // mata
  ctx.beginPath();
  ctx.arc(R * 0.44, -R * 0.42, R * 0.24, 0, TAU);
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.beginPath();
  ctx.arc(R * 0.5, -R * 0.42, R * 0.12, 0, TAU);
  ctx.fillStyle = "#111";
  ctx.fill();

  // sirip samping
  ctx.beginPath();
  ctx.moveTo(-R * 0.1, R * 0.1);
  ctx.quadraticCurveTo(-R * 0.5, R * 0.75, R * 0.15, R * 0.55);
  ctx.closePath();
  ctx.fillStyle = "#f97316";
  ctx.fill();
  ctx.strokeStyle = "#7c2d12";
  ctx.lineWidth = 1.4;
  ctx.stroke();

  ctx.restore();
}

function drawBubbles() {
  ctx.fillStyle = "rgba(255,255,255,0.75)";
  for (const b of bubbles) {
    ctx.globalAlpha = clamp(b.life, 0, 1);
    ctx.beginPath();
    ctx.arc(b.x, b.y, b.r, 0, TAU);
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

function render(t) {
  ctx.save();
  if (shake > 0) {
    ctx.translate(rand(-shake, shake), rand(-shake, shake));
  }
  drawBackground(t);
  if (state !== START) for (const o of obstacles) drawRig(o);
  drawBubbles();
  drawFish(t);
  ctx.restore();
}

/* =========================================================
   LOOP
   ========================================================= */
let lastFrame = performance.now();

reset(); // inisialisasi state sebelum frame pertama

function frame(now) {
  const dt = Math.min(0.034, (now - lastFrame) / 1000 || 0);
  lastFrame = now;

  if (mode === "camera" && landmarker && video.readyState >= 2 && video.videoWidth) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      detectFace(now);
    }
  }

  update(dt, now / 1000);
  render(now / 1000);
  requestAnimationFrame(frame);
}

/* =========================================================
   INPUT
   ========================================================= */
function pressStart(e) {
  if (state === PLAY) {
    touchHeld = true;
    blubOnce();
  }
}
function pressEnd() {
  touchHeld = false;
}

canvas.addEventListener("pointerdown", (e) => {
  e.preventDefault();
  pressStart(e);
});
window.addEventListener("pointerup", pressEnd);
window.addEventListener("pointercancel", pressEnd);
window.addEventListener("blur", pressEnd);

window.addEventListener("keydown", (e) => {
  if (e.repeat) return;
  if (e.code === "Space" || e.code === "ArrowUp") {
    e.preventDefault();
    if (state === OVER) {
      startGame();
    } else {
      touchHeld = true;
      blubOnce();
    }
  }
});
window.addEventListener("keyup", (e) => {
  if (e.code === "Space" || e.code === "ArrowUp") touchHeld = false;
});

btnCam.addEventListener("click", () => {
  ensureAudio();
  startCamera();
});
btnTouch.addEventListener("click", () => {
  ensureAudio();
  startGame("Tahan layar untuk berenang naik");
});
btnRetry.addEventListener("click", () => {
  ensureAudio();
  startGame(mode === "camera" ? "Buka mulut = naik • Tutup mulut = turun" : "Tahan layar untuk berenang naik");
});

btnShare.addEventListener("click", doShare);

bestScore.textContent = String(best);

requestAnimationFrame(frame);