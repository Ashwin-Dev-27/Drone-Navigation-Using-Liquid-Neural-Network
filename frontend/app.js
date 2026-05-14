/**
 * Adaptive Drone Navigation — Real-time Dashboard
 * ================================================
 * WebSocket client + Canvas renderers for LNN vs PID comparison
 */

// ─────────────────────────────────────────────
// WebSocket Connection
// ─────────────────────────────────────────────
const WS_URL = `ws://${window.location.host}/ws`;
let ws = null;
let paused = false;
let currentDifficulty = 'normal';
let lastState = null;

function connectWS() {
  ws = new WebSocket(WS_URL);

  ws.onopen = () => {
    setStatus('live', 'Live');
    console.log('✅ Connected to simulation server');
  };

  ws.onmessage = (event) => {
    const state = JSON.parse(event.data);
    lastState = state;
    updateDashboard(state);
  };

  ws.onclose = () => {
    setStatus('disconnected', 'Disconnected');
    setTimeout(connectWS, 2000); // auto-reconnect
  };

  ws.onerror = () => {
    setStatus('disconnected', 'Connection Error');
  };
}

function sendCommand(cmd) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(cmd));
  }
}

function setStatus(type, text) {
  const dot = document.querySelector('.status-dot');
  const textEl = document.getElementById('statusText');
  textEl.textContent = text;
  dot.className = 'status-dot' + (type === 'live' ? ' live' : '');
}

// ─────────────────────────────────────────────
// Canvas Setup
// ─────────────────────────────────────────────
const lnnCanvas = document.getElementById('lnnCanvas');
const pidCanvas = document.getElementById('pidCanvas');
const windCanvas = document.getElementById('windCanvas');
const errorCanvas = document.getElementById('errorChart');
const windChartCanvas = document.getElementById('windChart');

const lnnCtx = lnnCanvas.getContext('2d');
const pidCtx = pidCanvas.getContext('2d');
const windCtx = windCanvas.getContext('2d');
const errorCtx = errorCanvas.getContext('2d');
const windChartCtx = windChartCanvas.getContext('2d');

// ─────────────────────────────────────────────
// Drone Canvas Renderer (top-down 2D view)
// ─────────────────────────────────────────────

// World coordinates: -50 to +50 in x/y
const WORLD_SIZE = 100;  // meters total span
const WAYPOINTS = [
  [0, 0], [15, 5], [20, 20], [0, 25], [-15, 15], [0, 0]
];

function worldToCanvas(wx, wy, canvas) {
  const cx = canvas.width / 2;
  const cy = canvas.height / 2;
  const scale = canvas.width / WORLD_SIZE;
  return [cx + wx * scale, cy - wy * scale]; // flip Y (world Y+ = up)
}

function drawDroneView(ctx, canvas, droneData, color, label, obstacles) {
  const W = canvas.width;
  const H = canvas.height;

  // Background
  ctx.fillStyle = '#080c14';
  ctx.fillRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 1;
  const scale = W / WORLD_SIZE;
  for (let x = 0; x <= WORLD_SIZE; x += 10) {
    const cx = x * scale;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, H); ctx.stroke();
  }
  for (let y = 0; y <= WORLD_SIZE; y += 10) {
    const cy = y * scale;
    ctx.beginPath(); ctx.moveTo(0, cy); ctx.lineTo(W, cy); ctx.stroke();
  }

  // Center cross
  ctx.strokeStyle = 'rgba(255,255,255,0.08)';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(W/2, 0); ctx.lineTo(W/2, H); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(0, H/2); ctx.lineTo(W, H/2); ctx.stroke();

  // Obstacles
  if (obstacles) {
    obstacles.forEach(obs => {
      const [ox, oy] = worldToCanvas(obs.x, obs.y, canvas);
      const r = obs.radius * scale;
      ctx.beginPath();
      ctx.arc(ox, oy, r, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(255,71,87,0.15)';
      ctx.fill();
      ctx.strokeStyle = 'rgba(255,71,87,0.5)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    });
  }

  // Waypoints + route line
  ctx.strokeStyle = 'rgba(162,155,254,0.3)';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 6]);
  ctx.beginPath();
  WAYPOINTS.forEach(([wx, wy], i) => {
    const [cx, cy] = worldToCanvas(wx, wy, canvas);
    if (i === 0) ctx.moveTo(cx, cy);
    else ctx.lineTo(cx, cy);
  });
  ctx.stroke();
  ctx.setLineDash([]);

  WAYPOINTS.forEach(([wx, wy], i) => {
    const [cx, cy] = worldToCanvas(wx, wy, canvas);
    const isNext = i === (droneData.waypoint - 1);  // waypoint is 1-based, i is 0-based
    ctx.beginPath();
    ctx.arc(cx, cy, isNext ? 6 : 4, 0, Math.PI * 2);
    ctx.fillStyle = isNext ? '#a29bfe' : 'rgba(162,155,254,0.3)';
    ctx.fill();
    if (isNext) {
      ctx.strokeStyle = 'rgba(162,155,254,0.6)';
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
    // Label
    ctx.fillStyle = 'rgba(162,155,254,0.7)';
    ctx.font = '9px Inter';
    ctx.fillText(`W${i+1}`, cx + 7, cy - 3);
  });

  // Path trail
  if (droneData.path_x && droneData.path_x.length > 1) {
    ctx.beginPath();
    droneData.path_x.forEach((wx, i) => {
      const wy = droneData.path_y[i];
      const [cx, cy] = worldToCanvas(wx, wy, canvas);
      if (i === 0) ctx.moveTo(cx, cy);
      else ctx.lineTo(cx, cy);
    });
    const hex = color === 'lnn' ? '00e5ff' : 'ff9f43';
    ctx.strokeStyle = color === 'lnn'
      ? 'rgba(0,229,255,0.5)' : 'rgba(255,159,67,0.5)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Fade effect
    const last = droneData.path_x.length - 1;
    for (let i = Math.max(0, last - 15); i <= last; i++) {
      const t = (i - (last - 15)) / 15;
      const [cx, cy] = worldToCanvas(droneData.path_x[i], droneData.path_y[i], canvas);
      ctx.beginPath();
      ctx.arc(cx, cy, 1.5, 0, Math.PI * 2);
      ctx.fillStyle = color === 'lnn'
        ? `rgba(0,229,255,${t * 0.7})`
        : `rgba(255,159,67,${t * 0.7})`;
      ctx.fill();
    }
  }

  // Target line (drone → current waypoint)
  const [dx, dy] = worldToCanvas(droneData.pos[0], droneData.pos[1], canvas);
  const target = droneData.target;
  const [tx, ty] = worldToCanvas(target[0], target[1], canvas);
  ctx.beginPath();
  ctx.moveTo(dx, dy);
  ctx.lineTo(tx, ty);
  ctx.strokeStyle = color === 'lnn'
    ? 'rgba(0,229,255,0.25)' : 'rgba(255,159,67,0.25)';
  ctx.lineWidth = 1;
  ctx.setLineDash([3, 5]);
  ctx.stroke();
  ctx.setLineDash([]);

  // Drone icon (with orientation)
  const [px, py] = worldToCanvas(droneData.pos[0], droneData.pos[1], canvas);
  const yaw = droneData.orientation[2];
  const droneColor = color === 'lnn' ? '#00e5ff' : '#ff9f43';
  const glowColor = color === 'lnn' ? 'rgba(0,229,255,0.4)' : 'rgba(255,159,67,0.4)';

  // Glow
  const gradient = ctx.createRadialGradient(px, py, 0, px, py, 18);
  gradient.addColorStop(0, glowColor);
  gradient.addColorStop(1, 'transparent');
  ctx.fillStyle = gradient;
  ctx.beginPath();
  ctx.arc(px, py, 18, 0, Math.PI * 2);
  ctx.fill();

  // Drone body
  ctx.save();
  ctx.translate(px, py);
  ctx.rotate(-yaw);

  // Rotors
  const rotorPositions = [[-9, -9], [9, -9], [-9, 9], [9, 9]];
  rotorPositions.forEach(([rx, ry]) => {
    ctx.beginPath();
    ctx.arc(rx, ry, 5, 0, Math.PI * 2);
    ctx.strokeStyle = droneColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
    ctx.fillStyle = `${droneColor}22`;
    ctx.fill();
  });

  // Arms
  ctx.strokeStyle = droneColor;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(-9, -9); ctx.lineTo(9, 9); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(9, -9); ctx.lineTo(-9, 9); ctx.stroke();

  // Center body
  ctx.beginPath();
  ctx.arc(0, 0, 4, 0, Math.PI * 2);
  ctx.fillStyle = droneColor;
  ctx.fill();

  // Forward indicator
  ctx.beginPath();
  ctx.moveTo(0, 0);
  ctx.lineTo(0, -14);
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.5;
  ctx.stroke();

  ctx.restore();

  // Label
  ctx.fillStyle = 'rgba(255,255,255,0.5)';
  ctx.font = 'bold 10px Inter';
  ctx.fillText(label, px + 14, py - 10);

  // Altitude badge
  ctx.fillStyle = color === 'lnn' ? 'rgba(0,229,255,0.15)' : 'rgba(255,159,67,0.15)';
  ctx.fillRect(px + 14, py + 2, 40, 14);
  ctx.fillStyle = droneColor;
  ctx.font = '9px JetBrains Mono';
  ctx.fillText(`${droneData.pos[2].toFixed(1)}m`, px + 17, py + 12);
}

// ─────────────────────────────────────────────
// Wind Compass
// ─────────────────────────────────────────────
function drawWindCompass(wind) {
  const W = windCanvas.width;
  const H = windCanvas.height;
  const cx = W / 2, cy = H / 2;

  windCtx.fillStyle = '#0d1422';
  windCtx.fillRect(0, 0, W, H);

  // Circles
  [40, 30, 20, 10].forEach((r, i) => {
    windCtx.beginPath();
    windCtx.arc(cx, cy, r, 0, Math.PI * 2);
    windCtx.strokeStyle = `rgba(255,255,255,${0.04 + i * 0.02})`;
    windCtx.lineWidth = 1;
    windCtx.stroke();
  });

  // Axes
  const axes = [0, Math.PI/2, Math.PI, 3*Math.PI/2];
  const labels = ['N', 'E', 'S', 'W'];
  axes.forEach((angle, i) => {
    const x = cx + 44 * Math.sin(angle);
    const y = cy - 44 * Math.cos(angle);
    windCtx.fillStyle = 'rgba(255,255,255,0.3)';
    windCtx.font = '9px Inter';
    windCtx.textAlign = 'center';
    windCtx.textBaseline = 'middle';
    windCtx.fillText(labels[i], x, y);
  });

  // Wind arrow
  const mag = Math.sqrt(wind.x * wind.x + wind.y * wind.y);
  const angle = Math.atan2(wind.y, wind.x);
  const len = Math.min(mag * 4, 38);

  if (len > 1) {
    const ex = cx + len * Math.cos(angle);
    const ey = cy - len * Math.sin(angle);

    windCtx.beginPath();
    windCtx.moveTo(cx, cy);
    windCtx.lineTo(ex, ey);
    windCtx.strokeStyle = '#ffd32a';
    windCtx.lineWidth = 2.5;
    windCtx.lineCap = 'round';
    windCtx.stroke();

    // Arrowhead
    const headLen = 7;
    const headAngle = 0.4;
    windCtx.beginPath();
    windCtx.moveTo(ex, ey);
    windCtx.lineTo(
      ex - headLen * Math.cos(angle - headAngle),
      ey + headLen * Math.sin(angle - headAngle)
    );
    windCtx.moveTo(ex, ey);
    windCtx.lineTo(
      ex - headLen * Math.cos(angle + headAngle),
      ey + headLen * Math.sin(angle + headAngle)
    );
    windCtx.stroke();
  }

  // Center dot
  windCtx.beginPath();
  windCtx.arc(cx, cy, 3, 0, Math.PI * 2);
  windCtx.fillStyle = '#ffd32a';
  windCtx.fill();

  windCtx.textAlign = 'left';
  windCtx.textBaseline = 'alphabetic';
}

// ─────────────────────────────────────────────
// Error Chart
// ─────────────────────────────────────────────
function drawErrorChart(errors) {
  const W = errorCanvas.width;
  const H = errorCanvas.height;
  const pad = { top: 10, right: 10, bottom: 20, left: 30 };
  const chartW = W - pad.left - pad.right;
  const chartH = H - pad.top - pad.bottom;

  errorCtx.fillStyle = 'transparent';
  errorCtx.clearRect(0, 0, W, H);

  const lnn = errors.lnn;
  const pid = errors.pid;
  if (!lnn || lnn.length < 2) return;

  const maxVal = Math.max(...lnn, ...pid, 5) * 1.1;

  // Grid lines
  const gridLines = 4;
  errorCtx.strokeStyle = 'rgba(255,255,255,0.05)';
  errorCtx.lineWidth = 1;
  for (let i = 0; i <= gridLines; i++) {
    const y = pad.top + (chartH * i / gridLines);
    errorCtx.beginPath();
    errorCtx.moveTo(pad.left, y);
    errorCtx.lineTo(pad.left + chartW, y);
    errorCtx.stroke();
    const val = maxVal * (1 - i / gridLines);
    errorCtx.fillStyle = 'rgba(255,255,255,0.25)';
    errorCtx.font = '9px JetBrains Mono';
    errorCtx.fillText(val.toFixed(0), 0, y + 3);
  }

  function drawLine(data, color, fill) {
    if (data.length < 2) return;
    errorCtx.beginPath();
    data.forEach((v, i) => {
      const x = pad.left + (i / (data.length - 1)) * chartW;
      const y = pad.top + chartH * (1 - v / maxVal);
      if (i === 0) errorCtx.moveTo(x, y);
      else errorCtx.lineTo(x, y);
    });
    errorCtx.strokeStyle = color;
    errorCtx.lineWidth = 2;
    errorCtx.lineJoin = 'round';
    errorCtx.stroke();

    // Fill
    errorCtx.lineTo(pad.left + chartW, pad.top + chartH);
    errorCtx.lineTo(pad.left, pad.top + chartH);
    errorCtx.closePath();
    errorCtx.fillStyle = fill;
    errorCtx.fill();
  }

  drawLine(pid, '#ff9f43', 'rgba(255,159,67,0.1)');
  drawLine(lnn, '#00e5ff', 'rgba(0,229,255,0.12)');
}

// ─────────────────────────────────────────────
// Wind History Chart
// ─────────────────────────────────────────────
function drawWindChart(windHistory) {
  const W = windChartCanvas.width;
  const H = windChartCanvas.height;

  windChartCtx.clearRect(0, 0, W, H);
  if (!windHistory || windHistory.length < 2) return;

  const maxW = Math.max(...windHistory, 2);

  windChartCtx.beginPath();
  windHistory.forEach((v, i) => {
    const x = (i / (windHistory.length - 1)) * W;
    const y = H * (1 - v / maxW);
    if (i === 0) windChartCtx.moveTo(x, y);
    else windChartCtx.lineTo(x, y);
  });
  windChartCtx.strokeStyle = '#ffd32a';
  windChartCtx.lineWidth = 1.5;
  windChartCtx.stroke();

  windChartCtx.lineTo(W, H);
  windChartCtx.lineTo(0, H);
  windChartCtx.closePath();
  windChartCtx.fillStyle = 'rgba(255,211,42,0.08)';
  windChartCtx.fill();
}

// ─────────────────────────────────────────────
// Dashboard Update
// ─────────────────────────────────────────────
function updateDashboard(state) {
  const { lnn, pid, environment, errors_history, time, waypoints } = state;

  // Time
  document.getElementById('timeDisplay').textContent = `T+${time.toFixed(1)}s`;

  // LNN stats
  el('lnnError').textContent     = lnn.error.toFixed(2);
  el('lnnAvgError').textContent  = lnn.avg_error.toFixed(2);
  el('lnnAlt').textContent       = lnn.pos[2].toFixed(1);
  el('lnnCollisions').textContent = lnn.collisions;
  el('lnnParams').textContent    = `${lnn.params.toLocaleString()} params`;
  setProgress('lnnProgress', 'lnnProgressPct', lnn.mission_progress);
  setAttitude('lnnRollBar', 'lnnRoll', lnn.orientation[0]);
  setAttitude('lnnPitchBar', 'lnnPitch', lnn.orientation[1]);
  if (el('lnnStatusMsg')) el('lnnStatusMsg').textContent = lnn.status || 'Navigating...';

  // PID stats
  el('pidError').textContent     = pid.error.toFixed(2);
  el('pidAvgError').textContent  = pid.avg_error.toFixed(2);
  el('pidAlt').textContent       = pid.pos[2].toFixed(1);
  el('pidCollisions').textContent = pid.collisions;
  el('pidParams').textContent    = `${pid.params} params`;
  setProgress('pidProgress', 'pidProgressPct', pid.mission_progress);
  setAttitude('pidRollBar', 'pidRoll', pid.orientation[0]);
  setAttitude('pidPitchBar', 'pidPitch', pid.orientation[1]);
  if (el('pidStatusMsg')) el('pidStatusMsg').textContent = pid.status || 'Navigating...';

  // Environment
  const wind = environment.wind;
  el('windMag').textContent = `${wind.magnitude.toFixed(1)} m/s`;
  el('gustCount').textContent = environment.gusts;
  el('turbLevel').textContent = (environment.turbulence * 100).toFixed(0) + '%';
  el('obstCount').textContent = environment.obstacles.length;

  // Score comparison
  if (lnn.avg_error > 0 && pid.avg_error > 0) {
    const advantage = Math.max(0, (pid.avg_error - lnn.avg_error) / pid.avg_error * 100);
    const pct = Math.min(advantage, 100);
    el('scoreBar').style.width = pct + '%';
    el('scorePct').textContent = pct.toFixed(1) + '%';
    if (advantage > 5) {
      el('scoreVerdict').textContent = `LNN error ${advantage.toFixed(1)}% lower than PID under current conditions`;
    } else if (advantage < -5) {
      el('scoreVerdict').textContent = `PID performing similarly — try Stormy difficulty to see LNN shine`;
    } else {
      el('scoreVerdict').textContent = `Performance converging — increase difficulty to stress test`;
    }
  }

  // Canvas draws
  drawDroneView(lnnCtx, lnnCanvas, lnn, 'lnn', 'LNN', environment.obstacles);
  drawDroneView(pidCtx, pidCanvas, pid, 'pid', 'PID', environment.obstacles);
  drawWindCompass(wind);
  drawErrorChart(errors_history);
  drawWindChart(errors_history.wind);
}

function el(id) {
  return document.getElementById(id);
}

function setProgress(barId, pctId, value) {
  el(barId).style.width = value + '%';
  el(pctId).textContent = value.toFixed(0) + '%';
}

function setAttitude(barId, valId, radians) {
  const deg = radians * (180 / Math.PI);
  const bar = el(barId);
  const pct = Math.abs(deg) / 40; // max 40 degrees
  if (deg >= 0) {
    bar.style.left = '50%';
    bar.style.width = (pct * 50) + '%';
  } else {
    bar.style.left = (50 - pct * 50) + '%';
    bar.style.width = (pct * 50) + '%';
  }
  el(valId).textContent = deg.toFixed(1) + '°';
}

// ─────────────────────────────────────────────
// Controls
// ─────────────────────────────────────────────
document.getElementById('btnPause').addEventListener('click', () => {
  paused = !paused;
  sendCommand({ action: 'pause' });
  document.getElementById('btnPause').textContent = paused ? '▶ Resume' : '⏸ Pause';
});

document.getElementById('btnReset').addEventListener('click', () => {
  sendCommand({ action: 'reset', difficulty: currentDifficulty });
});

document.querySelectorAll('.btn-diff').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.btn-diff').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentDifficulty = btn.dataset.level;
    sendCommand({ action: 'difficulty', level: currentDifficulty });
  });
});

// ─────────────────────────────────────────────
// Init
// ─────────────────────────────────────────────
connectWS();

// Draw placeholder canvases while connecting
function drawPlaceholder(ctx, canvas, text) {
  ctx.fillStyle = '#0d1422';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'rgba(255,255,255,0.1)';
  ctx.font = '14px Inter';
  ctx.textAlign = 'center';
  ctx.fillText(text, canvas.width / 2, canvas.height / 2);
  ctx.textAlign = 'left';
}

drawPlaceholder(lnnCtx, lnnCanvas, 'Connecting to simulation…');
drawPlaceholder(pidCtx, pidCanvas, 'Connecting to simulation…');
