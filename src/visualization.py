#3d interactive visualization for pid vs koopman mpc rocket trajectories

import json
import os
import numpy as np


def generate_3d_visualization_html(histories, cfg, outdir, waypoint=None):
    """
    Generate an interactive HTML visualization with:
      - Split, PID-only, MPC-only, and Combined overlay display modes
      - 3D and 2D projection modes
      - Larger zoom-out range
      - Meter tick marks on lateral axes (North/East) in both 3D and 2D views
    """
    os.makedirs(os.path.join(outdir, "plots"), exist_ok=True)

    data_dict = {}
    max_alt = 0.0
    max_north = 0.0
    max_east = 0.0

    for ctrl_type in ["PID", "MPC"]:
        hl = histories.get(ctrl_type, [])
        if not hl:
            continue

        h = hl[0]
        t = np.asarray(h.get("t", []), dtype=float)
        alt = np.asarray(h.get("h", []), dtype=float)
        vz = np.asarray(h.get("vz", []), dtype=float)
        xp = np.asarray(h.get("xp", np.zeros_like(t)), dtype=float)  #north
        y = np.asarray(h.get("y", np.zeros_like(t)), dtype=float)    #east
        vx = np.asarray(h.get("vx", np.zeros_like(t)), dtype=float)
        vy = np.asarray(h.get("vy", np.zeros_like(t)), dtype=float)
        theta = np.asarray(h.get("theta", []), dtype=float)
        pitch = np.asarray(h.get("pitch", np.zeros_like(theta)), dtype=float)
        F = np.asarray(h.get("F", []), dtype=float)
        delta = np.asarray(h.get("delta", []), dtype=float)
        m = np.asarray(h.get("m", []), dtype=float)
        phase = h.get("phase", ["Ascent"] * len(t))
        h_ref = np.asarray(h.get("h_ref", np.zeros_like(t)), dtype=float)
        wind_azimuth = np.asarray(h.get("wind_azimuth", np.zeros_like(t)), dtype=float)
        wind_vx = np.asarray(h.get("wind_vx", np.zeros_like(t)), dtype=float)
        wind_vy = np.asarray(h.get("wind_vy", np.zeros_like(t)), dtype=float)

        n = len(t)
        if n == 0:
            continue

        lat_mag = np.sqrt(xp ** 2 + y ** 2)

        data_dict[ctrl_type] = {
            "t": t.tolist(),
            "h": alt.tolist(),
            "h_ref": h_ref.tolist(),
            "xp": xp.tolist(),
            "y": y.tolist(),
            "lat_mag": lat_mag.tolist(),
            "vz": vz.tolist(),
            "vx": vx.tolist(),
            "vy": vy.tolist(),
            "theta_deg": np.degrees(theta).tolist(),
            "pitch_deg": np.degrees(pitch).tolist(),
            "delta_deg": np.degrees(delta).tolist(),
            "F": F.tolist(),
            "m": m.tolist(),
            "phase": phase,
            "wind_azimuth": wind_azimuth.tolist(),
            "wind_vx": wind_vx.tolist(),
            "wind_vy": wind_vy.tolist(),
        }

        max_alt = max(max_alt, float(np.max(alt)) if len(alt) else 0.0)
        max_north = max(max_north, float(np.max(np.abs(xp))) if len(xp) else 0.0)
        max_east = max(max_east, float(np.max(np.abs(y))) if len(y) else 0.0)

    if not data_dict:
        return None

    max_alt = max(max_alt, 50.0)
    max_north = max(max_north, 10.0)
    max_east = max(max_east, 10.0)
    has_pid = "PID" in data_dict

    wp_json = "null"
    if waypoint is not None:
        if isinstance(waypoint, (list, tuple)) and len(waypoint) >= 3:
            wp_h = float(waypoint[0])
            wp_n = float(waypoint[1])
            wp_e = float(waypoint[2])
        elif isinstance(waypoint, dict):
            wp_h = float(waypoint.get("h", waypoint.get("h_min", 0.0)))
            if waypoint.get("kind", "stationary") == "moving":
                wp_n = float(waypoint.get("N0", 0.0))
                wp_e = float(waypoint.get("E0", 0.0))
            else:
                wp_n = float(waypoint.get("N", 0.0))
                wp_e = float(waypoint.get("E", 0.0))
        else:
            wp_h, wp_n, wp_e = 0.0, 0.0, 0.0
        wp_json = json.dumps({"h": wp_h, "xp": wp_n, "y": wp_e})

    js_data = json.dumps(data_dict)

    html = f"""<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"UTF-8\">
<title>Rocket 3D Trajectory - {'MPC vs PID' if has_pid else 'MPC Only'}</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:\"Segoe UI\",sans-serif;background:#060d1a;color:#e2e8f0;overflow:hidden;height:100vh}}
.wrap{{display:flex;flex-direction:column;height:100vh}}
.hdr{{background:rgba(6,13,26,.98);padding:8px 16px;border-bottom:2px solid #3b82f6;flex-shrink:0;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.hdr h1{{font-size:15px;font-weight:700;white-space:nowrap}}
.hdr p{{font-size:9px;color:#64748b}}
.ctrls{{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.ctrls label{{font-size:10px;color:#93c5fd}}
select,button{{padding:4px 6px;background:#1e293b;border:1px solid #334155;color:#e2e8f0;border-radius:4px;font-size:10px}}
button{{cursor:pointer}}
button:hover,select:hover{{background:#2563eb;border-color:#3b82f6}}
.body{{display:flex;flex:1;gap:6px;padding:6px;min-height:0}}
.cvwrap{{flex:1;display:flex;gap:6px;min-height:0}}
.cbox{{background:rgba(10,18,32,.95);border:1px solid #1e293b;border-radius:8px;padding:4px;display:flex;flex-direction:column;min-height:0;flex:1}}
.ctitle{{font-size:11px;font-weight:700;text-align:center;margin-bottom:3px;padding:3px;border-radius:4px}}
.pid-t{{background:linear-gradient(90deg,rgba(239,68,68,.2),transparent);color:#fca5a5}}
.mpc-t{{background:linear-gradient(90deg,rgba(59,130,246,.2),transparent);color:#93c5fd}}
.cmb-t{{background:linear-gradient(90deg,rgba(34,197,94,.2),transparent);color:#86efac}}
canvas{{flex:1;width:100%;min-height:0;border-radius:4px;display:block;background:#02060f;cursor:grab}}
canvas:active{{cursor:grabbing}}
.hint{{font-size:9px;color:#475569;text-align:center;margin-top:2px}}
.sb{{width:250px;flex-shrink:0;background:rgba(6,13,26,.98);border:1px solid #1e293b;border-radius:8px;padding:8px;display:flex;flex-direction:column;gap:5px;overflow-y:auto}}
.sec{{background:rgba(10,18,32,.9);padding:6px;border-radius:5px;border-left:2px solid #3b82f6}}
.stitle{{font-size:9px;font-weight:700;color:#60a5fa;text-transform:uppercase;letter-spacing:.5px;margin-bottom:4px}}
.btnr{{display:flex;gap:4px}}
.pb{{display:block;padding:2px;border-radius:3px;font-size:9px;font-weight:700;text-align:center;margin-top:3px}}
.pb-prelaunch{{background:#4f46e5}}.pb-ascent{{background:#dc2626}}
.pb-apogee{{background:#d97706}}.pb-descent{{background:#7c3aed}}
input[type=range]{{width:100%;-webkit-appearance:none;height:3px;border-radius:2px;background:#1e293b}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:#0ea5e9;cursor:pointer}}
.mg{{display:grid;grid-template-columns:1fr 1fr;gap:3px}}
.mc{{background:rgba(20,30,50,.8);padding:3px 5px;border-radius:3px}}
.ml{{font-size:8px;color:#64748b;text-transform:uppercase}}
.mv{{font-size:11px;font-weight:700;color:#38bdf8;font-family:monospace}}
</style></head><body>
<div class=\"wrap\">
<div class=\"hdr\">
  <div>
    <h1>{'Koopman MPC vs PID - 3D Trajectory' if has_pid else 'Koopman MPC - 3D Trajectory'}</h1>
    <p>X=North | Y=Altitude | Z=East | Left-drag=orbit | Middle-drag=pan | Scroll=zoom</p>
  </div>
  <div class=\"ctrls\">
    <label>Projection</label>
    <select id=\"projection\" onchange=\"onProjectionChange(this.value)\">
      <option value=\"3d\">3D (North/Alt/East)</option>
      <option value=\"nh\">2D North vs Altitude</option>
      <option value=\"eh\">2D East vs Altitude</option>
      <option value=\"ne\">2D Top-down (North vs East)</option>
    </select>
    <label>Display</label>
    <select id=\"display\" onchange=\"onDisplayChange(this.value)\">
      <option value=\"split\">Split (PID + MPC)</option>
      <option value=\"pid\">PID only</option>
      <option value=\"mpc\">MPC only</option>
      <option value=\"combined\">Combined overlay</option>
    </select>
    <button onclick=\"resetView()\">Reset View</button>
  </div>
</div>
<div class=\"body\">
  <div class=\"cvwrap\">
    <div class=\"cbox\" id=\"box-pid\">
      <div class=\"ctitle pid-t\">PID</div>
      <canvas id=\"cv-pid\"></canvas>
      <div class=\"hint\">Scroll farther out for large-area context</div>
    </div>
    <div class=\"cbox\" id=\"box-mpc\">
      <div class=\"ctitle mpc-t\">Koopman MPC</div>
      <canvas id=\"cv-mpc\"></canvas>
      <div class=\"hint\">Scroll farther out for large-area context</div>
    </div>
    <div class=\"cbox\" id=\"box-both\" style=\"display:none;\">
      <div class=\"ctitle cmb-t\">Combined PID + MPC</div>
      <canvas id=\"cv-both\"></canvas>
      <div class=\"hint\">Overlay: PID (red), MPC (blue)</div>
    </div>
  </div>
  <div class=\"sb\">
    <div class=\"sec\">
      <div class=\"stitle\">Playback</div>
      <div class=\"btnr\">
        <button id=\"pb\" onclick=\"togglePlay()\">Play</button>
        <button onclick=\"pausePlay()\">Pause</button>
        <button onclick=\"resetPlay()\">Reset</button>
      </div>
      <input type=\"range\" id=\"sl\" min=\"0\" max=\"100\" value=\"0\" oninput=\"scrub(this.value)\" style=\"margin-top:5px\">
      <div style=\"font-size:9px;color:#64748b;margin-top:2px\">t = <span id=\"td\">0.00</span> s</div>
    </div>

    <div class=\"sec\">
      <div class=\"stitle\">PID Metrics</div>
      <div class=\"mg\">
        <div class=\"mc\"><div class=\"ml\">Altitude</div><div class=\"mv\"><span id=\"pid-h\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Lateral |r|</div><div class=\"mv\"><span id=\"pid-lat\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">North</div><div class=\"mv\"><span id=\"pid-xp\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">East</div><div class=\"mv\"><span id=\"pid-y\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Velocity</div><div class=\"mv\"><span id=\"pid-vz\">0</span>m/s</div></div>
        <div class=\"mc\"><div class=\"ml\">Thrust</div><div class=\"mv\"><span id=\"pid-F\">0</span>N</div></div>
        <div class=\"mc\"><div class=\"ml\">Gimbal</div><div class=\"mv\"><span id=\"pid-d\">0</span>deg</div></div>
        <div class=\"mc\"><div class=\"ml\">Tilt</div><div class=\"mv\"><span id=\"pid-th\">0</span>deg</div></div>
      </div>
      <span class=\"pb pb-ascent\" id=\"pid-ph\">Ascent</span>
    </div>

    <div class=\"sec\">
      <div class=\"stitle\">MPC Metrics</div>
      <div class=\"mg\">
        <div class=\"mc\"><div class=\"ml\">Altitude</div><div class=\"mv\"><span id=\"mpc-h\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Lateral |r|</div><div class=\"mv\"><span id=\"mpc-lat\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">North</div><div class=\"mv\"><span id=\"mpc-xp\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">East</div><div class=\"mv\"><span id=\"mpc-y\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Velocity</div><div class=\"mv\"><span id=\"mpc-vz\">0</span>m/s</div></div>
        <div class=\"mc\"><div class=\"ml\">Thrust</div><div class=\"mv\"><span id=\"mpc-F\">0</span>N</div></div>
        <div class=\"mc\"><div class=\"ml\">Gimbal</div><div class=\"mv\"><span id=\"mpc-d\">0</span>deg</div></div>
        <div class=\"mc\"><div class=\"ml\">Tilt</div><div class=\"mv\"><span id=\"mpc-th\">0</span>deg</div></div>
      </div>
      <span class=\"pb pb-ascent\" id=\"mpc-ph\">Ascent</span>
    </div>

    <div class=\"sec\" id=\"wp-sec\" style=\"display:none;\">
      <div class=\"stitle\">MPC Waypoint Distance</div>
      <div class=\"mg\">
        <div class=\"mc\"><div class=\"ml\">Current Lateral</div><div class=\"mv\"><span id=\"wp-lat-now\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Closest Lateral</div><div class=\"mv\"><span id=\"wp-lat-best\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Current 3D</div><div class=\"mv\"><span id=\"wp-3d-now\">0</span>m</div></div>
        <div class=\"mc\"><div class=\"ml\">Closest 3D</div><div class=\"mv\"><span id=\"wp-3d-best\">0</span>m</div></div>
      </div>
    </div>

    <div class=\"sec\">
      <div class=\"stitle\">Wind</div>
      <div class=\"mg\">
        <div class=\"mc\"><div class=\"ml\">Direction</div><div class=\"mv\"><span id=\"wind-deg\">0</span>deg</div></div>
        <div class=\"mc\"><div class=\"ml\">Speed</div><div class=\"mv\"><span id=\"wind-spd\">0</span>m/s</div></div>
      </div>
    </div>
  </div>
</div>
</div>

<script>
const DATA = {js_data};
const WAYPOINT = {wp_json};
const LIMITS = {{
  alt: {max_alt:.6f} * 1.15,
  north: {max_north:.6f} * 1.25 + 10,
  east: {max_east:.6f} * 1.25 + 10
}};

const STATE = {{
  idx: 0,
  play: false,
  projection: "3d",
  display: "split",
  zoom: 1.0,
  az: -35,
  el: 24,
  panX: 0,
  panY: 0,
  orbit: false,
  pan: false,
  lx: 0,
  ly: 0
}};

const CVS = {{
  PID: document.getElementById("cv-pid"),
  MPC: document.getElementById("cv-mpc"),
  BOTH: document.getElementById("cv-both")
}};
const CTX = {{
  PID: CVS.PID.getContext("2d"),
  MPC: CVS.MPC.getContext("2d"),
  BOTH: CVS.BOTH.getContext("2d")
}};

function eid(id) {{ return document.getElementById(id); }}
function dataLen(d) {{ return d && d.t ? d.t.length : 0; }}
function maxLen() {{
  return Math.max(dataLen(DATA.PID), dataLen(DATA.MPC));
}}

function niceStep(span, targetTicks) {{
  const raw = Math.max(span / Math.max(targetTicks, 1), 1e-9);
  const pow10 = Math.pow(10, Math.floor(Math.log10(raw)));
  const x = raw / pow10;
  let mult = 1;
  if (x > 5) mult = 10;
  else if (x > 2) mult = 5;
  else if (x > 1) mult = 2;
  return mult * pow10;
}}

function resizeCanvases() {{
  Object.values(CVS).forEach((cv) => {{
    const r = cv.parentElement.getBoundingClientRect();
    cv.width = Math.max(Math.floor(r.width), 200);
    cv.height = Math.max(Math.floor(r.height - 28), 180);
  }});
}}
window.addEventListener("resize", () => {{ resizeCanvases(); draw(); }});

function project3D(north, alt, east, cv) {{
  const az = STATE.az * Math.PI / 180;
  const el = STATE.el * Math.PI / 180;
  const rx = north * Math.cos(az) - east * Math.sin(az);
  const rzY = north * Math.sin(az) + east * Math.cos(az);
  const ry = alt * Math.cos(el) - rzY * Math.sin(el);
  const pxScale = Math.min(cv.width, cv.height) * 0.010 * STATE.zoom;
  return {{
    x: cv.width * 0.50 + rx * pxScale + STATE.panX,
    y: cv.height * 0.54 - ry * pxScale + STATE.panY
  }};
}}

function project2D(mode, a, b, cv) {{
  const nSpan = LIMITS.north;
  const eSpan = LIMITS.east;
  const hSpan = LIMITS.alt;
  let sx = 1;
  let sy = 1;

  if (mode === "nh") {{
    sx = (cv.width - 70) / (2 * nSpan);
    sy = (cv.height - 60) / hSpan;
  }} else if (mode === "eh") {{
    sx = (cv.width - 70) / (2 * eSpan);
    sy = (cv.height - 60) / hSpan;
  }} else {{
    sx = (cv.width - 70) / (2 * eSpan);
    sy = (cv.height - 60) / (2 * nSpan);
  }}

  sx *= STATE.zoom;
  sy *= STATE.zoom;

  if (mode === "nh" || mode === "eh") {{
    return {{
      x: cv.width * 0.50 + a * sx + STATE.panX,
      y: cv.height - 40 - b * sy + STATE.panY
    }};
  }}

  return {{
    x: cv.width * 0.50 + a * sx + STATE.panX,
    y: cv.height * 0.50 - b * sy + STATE.panY
  }};
}}

function drawAxis3DTicks(c, cv) {{
  const nStep = niceStep(LIMITS.north * 2, 8);
  const eStep = niceStep(LIMITS.east * 2, 8);

  c.save();
  c.font = "9px monospace";
  c.textAlign = "center";
  c.textBaseline = "top";

  for (let n = -Math.floor(LIMITS.north / nStep) * nStep; n <= LIMITS.north; n += nStep) {{
    if (Math.abs(n) < 1e-8) continue;
    const p0 = project3D(n, 0, 0, cv);
    const p1 = project3D(n, 3, 0, cv);
    c.strokeStyle = "rgba(239,68,68,0.65)";
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke();
    c.fillStyle = "rgba(248,113,113,0.85)";
    c.fillText(Math.round(n) + "m", p0.x, p0.y + 4);
  }}

  for (let e = -Math.floor(LIMITS.east / eStep) * eStep; e <= LIMITS.east; e += eStep) {{
    if (Math.abs(e) < 1e-8) continue;
    const p0 = project3D(0, 0, e, cv);
    const p1 = project3D(0, 3, e, cv);
    c.strokeStyle = "rgba(96,165,250,0.65)";
    c.lineWidth = 1;
    c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke();
    c.fillStyle = "rgba(147,197,253,0.85)";
    c.fillText(Math.round(e) + "m", p0.x, p0.y + 4);
  }}

  c.restore();
}}

function drawGrid3D(c, cv) {{
  const nStep = niceStep(LIMITS.north * 2, 10);
  const eStep = niceStep(LIMITS.east * 2, 10);

  c.save();
  c.globalAlpha = 0.13;
  c.strokeStyle = "#1e293b";
  c.lineWidth = 0.7;

  for (let n = -Math.floor(LIMITS.north / nStep) * nStep; n <= LIMITS.north; n += nStep) {{
    const a = project3D(n, 0, -LIMITS.east, cv);
    const b = project3D(n, 0, LIMITS.east, cv);
    c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
  }}
  for (let e = -Math.floor(LIMITS.east / eStep) * eStep; e <= LIMITS.east; e += eStep) {{
    const a = project3D(-LIMITS.north, 0, e, cv);
    const b = project3D(LIMITS.north, 0, e, cv);
    c.beginPath(); c.moveTo(a.x, a.y); c.lineTo(b.x, b.y); c.stroke();
  }}

  const o = project3D(0, 0, 0, cv);
  const nx = project3D(LIMITS.north * 0.75, 0, 0, cv);
  const up = project3D(0, LIMITS.alt * 0.60, 0, cv);
  const ez = project3D(0, 0, LIMITS.east * 0.75, cv);

  c.globalAlpha = 0.85;
  c.lineWidth = 1.6;
  c.strokeStyle = "#ef4444";
  c.beginPath(); c.moveTo(o.x, o.y); c.lineTo(nx.x, nx.y); c.stroke();
  c.strokeStyle = "#4ade80";
  c.beginPath(); c.moveTo(o.x, o.y); c.lineTo(up.x, up.y); c.stroke();
  c.strokeStyle = "#60a5fa";
  c.beginPath(); c.moveTo(o.x, o.y); c.lineTo(ez.x, ez.y); c.stroke();

  c.font = "bold 10px sans-serif";
  c.fillStyle = "#ef4444"; c.fillText("N", nx.x + 6, nx.y + 6);
  c.fillStyle = "#4ade80"; c.fillText("Alt", up.x + 6, up.y - 6);
  c.fillStyle = "#60a5fa"; c.fillText("E", ez.x + 6, ez.y - 6);

  const hStep = niceStep(LIMITS.alt, 6);
  c.globalAlpha = 0.5;
  c.fillStyle = "#94a3b8";
  c.font = "9px monospace";
  for (let h = 0; h <= LIMITS.alt + 1e-8; h += hStep) {{
    const p = project3D(0, h, 0, cv);
    c.fillText(Math.round(h) + "m", p.x + 4, p.y + 2);
  }}

  c.restore();
  drawAxis3DTicks(c, cv);
}}

function drawGrid2D(c, cv, mode) {{
  c.save();

  let xSpan = LIMITS.north;
  let ySpan = LIMITS.alt;
  if (mode === "eh") xSpan = LIMITS.east;
  if (mode === "ne") ySpan = LIMITS.north;

  const xStep = niceStep(xSpan * 2, 8);
  const yStep = niceStep((mode === "ne" ? ySpan * 2 : ySpan), 7);

  c.lineWidth = 0.8;

  let xStart = -Math.floor(xSpan / xStep) * xStep;
  let xEnd = xSpan;
  if (mode !== "ne") {{
    xStart = -Math.floor(xSpan / xStep) * xStep;
    xEnd = xSpan;
  }}

  for (let x = xStart; x <= xEnd + 1e-8; x += xStep) {{
    const p0 = project2D(mode, x, mode === "ne" ? -ySpan : 0, cv);
    const p1 = project2D(mode, x, mode === "ne" ? ySpan : ySpan, cv);
    const isZero = Math.abs(x) < 1e-8;
    c.strokeStyle = isZero ? "rgba(34,197,94,0.65)" : "rgba(51,65,85,0.45)";
    c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke();

    c.fillStyle = isZero ? "#86efac" : "#64748b";
    c.font = "9px monospace";
    c.fillText(Math.round(x) + "m", p0.x - 12, cv.height - 20);
  }}

  const yStart = mode === "ne" ? -Math.floor(ySpan / yStep) * yStep : 0;
  const yEnd = ySpan;
  for (let y = yStart; y <= yEnd + 1e-8; y += yStep) {{
    const p0 = project2D(mode, -xSpan, y, cv);
    const p1 = project2D(mode, xSpan, y, cv);
    const isZero = Math.abs(y) < 1e-8;
    c.strokeStyle = isZero ? "rgba(34,197,94,0.65)" : "rgba(51,65,85,0.45)";
    c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke();

    c.fillStyle = isZero ? "#86efac" : "#64748b";
    c.font = "9px monospace";
    c.fillText(Math.round(y) + "m", 6, p0.y + 3);
  }}

  c.font = "bold 10px sans-serif";
  c.fillStyle = "#cbd5e1";
  if (mode === "nh") c.fillText("North [m]", cv.width * 0.5 - 28, cv.height - 4);
  if (mode === "eh") c.fillText("East [m]", cv.width * 0.5 - 24, cv.height - 4);
  if (mode === "ne") c.fillText("East [m]", cv.width * 0.5 - 24, cv.height - 4);

  c.save();
  c.translate(10, cv.height * 0.45);
  c.rotate(-Math.PI / 2);
  if (mode === "ne") c.fillText("North [m]", 0, 0);
  else c.fillText("Altitude [m]", 0, 0);
  c.restore();

  c.restore();
}}

function drawReference(c, cv, d, idx) {{
  if (!d || idx < 1) return;
  c.save();
  c.strokeStyle = "#22c55e";
  c.globalAlpha = 0.42;
  c.lineWidth = 1.2;
  c.setLineDash([5, 4]);
  c.beginPath();
  let started = false;
  for (let i = 0; i <= idx; i++) {{
    let p = null;
    if (STATE.projection === "3d") p = project3D(0, d.h_ref[i], 0, cv);
    else if (STATE.projection === "nh") p = project2D("nh", 0, d.h_ref[i], cv);
    else if (STATE.projection === "eh") p = project2D("eh", 0, d.h_ref[i], cv);
    if (!p) continue;
    if (!started) {{ c.moveTo(p.x, p.y); started = true; }}
    else c.lineTo(p.x, p.y);
  }}
  c.stroke();
  c.setLineDash([]);
  c.restore();
}}

function drawTrajectory(c, cv, d, idx, color) {{
  if (!d || idx < 1) return;
  c.save();
  c.strokeStyle = color;
  c.lineWidth = 2.2;
  c.globalAlpha = 0.92;
  c.beginPath();
  let started = false;
  for (let i = 0; i <= idx; i++) {{
    let p;
    if (STATE.projection === "3d") p = project3D(d.xp[i], d.h[i], d.y[i], cv);
    else if (STATE.projection === "nh") p = project2D("nh", d.xp[i], d.h[i], cv);
    else if (STATE.projection === "eh") p = project2D("eh", d.y[i], d.h[i], cv);
    else p = project2D("ne", d.y[i], d.xp[i], cv);
    if (!started) {{ c.moveTo(p.x, p.y); started = true; }}
    else c.lineTo(p.x, p.y);
  }}
  c.stroke();
  c.restore();
}}

function drawRocket(c, cv, d, idx, color) {{
  let p;
  if (STATE.projection === "3d") p = project3D(d.xp[idx], d.h[idx], d.y[idx], cv);
  else if (STATE.projection === "nh") p = project2D("nh", d.xp[idx], d.h[idx], cv);
  else if (STATE.projection === "eh") p = project2D("eh", d.y[idx], d.h[idx], cv);
  else p = project2D("ne", d.y[idx], d.xp[idx], cv);

  const tilt = (d.theta_deg[idx] || 0) * Math.PI / 180;
  const s = 12;

  c.save();
  c.translate(p.x, p.y);
  c.rotate(tilt);
  c.fillStyle = color;
  c.beginPath();
  c.moveTo(0, -s);
  c.lineTo(s * 0.45, s * 0.65);
  c.lineTo(-s * 0.45, s * 0.65);
  c.closePath();
  c.fill();
  c.strokeStyle = "rgba(255,255,255,0.45)";
  c.lineWidth = 0.8;
  c.stroke();
  c.restore();
}}

function drawWaypoint(c, cv) {{
  if (!WAYPOINT) return;
  let p;
  if (STATE.projection === "3d") p = project3D(WAYPOINT.xp, WAYPOINT.h, WAYPOINT.y, cv);
  else if (STATE.projection === "nh") p = project2D("nh", WAYPOINT.xp, WAYPOINT.h, cv);
  else if (STATE.projection === "eh") p = project2D("eh", WAYPOINT.y, WAYPOINT.h, cv);
  else p = project2D("ne", WAYPOINT.y, WAYPOINT.xp, cv);

  c.save();
  c.globalAlpha = 0.35;
  c.strokeStyle = "#fbbf24";
  c.lineWidth = 1.5;
  c.beginPath(); c.arc(p.x, p.y, 16, 0, Math.PI * 2); c.stroke();
  c.globalAlpha = 1.0;
  c.fillStyle = "#fbbf24";
  c.beginPath();
  c.moveTo(p.x, p.y - 8);
  c.lineTo(p.x + 8, p.y);
  c.lineTo(p.x, p.y + 8);
  c.lineTo(p.x - 8, p.y);
  c.closePath();
  c.fill();
  c.restore();
}}

function drawWind(c, cv, d, idx) {{
  if (!d || !d.wind_vx || !d.wind_vy || idx < 0) return;
  const vx = d.wind_vx[idx] || 0;
  const vy = d.wind_vy[idx] || 0;
  const mag = Math.sqrt(vx * vx + vy * vy);
  if (mag < 1e-6) return;

  let p0, p1;
  const ux = vx / mag;
  const uy = vy / mag;
  const s = 8.0;
  if (STATE.projection === "3d") {{
    p0 = project3D(0, 0, 0, cv);
    p1 = project3D(ux * s, 0, uy * s, cv);
  }} else if (STATE.projection === "nh") {{
    p0 = project2D("nh", 0, 0, cv);
    p1 = project2D("nh", ux * s, 0, cv);
  }} else if (STATE.projection === "eh") {{
    p0 = project2D("eh", 0, 0, cv);
    p1 = project2D("eh", uy * s, 0, cv);
  }} else {{
    p0 = project2D("ne", 0, 0, cv);
    p1 = project2D("ne", uy * s, ux * s, cv);
  }}

  const ang = Math.atan2(p1.y - p0.y, p1.x - p0.x);
  c.save();
  c.strokeStyle = "#f59e0b";
  c.fillStyle = "#f59e0b";
  c.lineWidth = 1.6;
  c.globalAlpha = 0.75;
  c.beginPath(); c.moveTo(p0.x, p0.y); c.lineTo(p1.x, p1.y); c.stroke();
  c.beginPath();
  c.moveTo(p1.x, p1.y);
  c.lineTo(p1.x - 5 * Math.cos(ang - 0.45), p1.y - 5 * Math.sin(ang - 0.45));
  c.lineTo(p1.x - 5 * Math.cos(ang + 0.45), p1.y - 5 * Math.sin(ang + 0.45));
  c.closePath();
  c.fill();
  const deg = (Math.atan2(vy, vx) * 180 / Math.PI + 360) % 360;
  c.font = "bold 10px sans-serif";
  c.fillText(`Wind ${{deg.toFixed(0)}}°`, p1.x + 8, p1.y - 4);
  c.restore();
}}

function updateSidebar(ctrl, d, idx) {{
  const k = ctrl.toLowerCase();
  eid(k + "-h").textContent = (d.h[idx] || 0).toFixed(1);
  eid(k + "-lat").textContent = (d.lat_mag[idx] || 0).toFixed(1);
  eid(k + "-xp").textContent = (d.xp[idx] || 0).toFixed(1);
  eid(k + "-y").textContent = (d.y[idx] || 0).toFixed(1);
  eid(k + "-vz").textContent = (d.vz[idx] || 0).toFixed(1);
  eid(k + "-F").textContent = (d.F[idx] || 0).toFixed(1);
  eid(k + "-d").textContent = (d.delta_deg[idx] || 0).toFixed(1);
  const tilt = Math.sqrt((d.theta_deg[idx] || 0) ** 2 + (d.pitch_deg[idx] || 0) ** 2);
  eid(k + "-th").textContent = tilt.toFixed(1);

  const ph = String(d.phase[idx] || "ascent").toLowerCase().replace(/[^a-z]/g, "");
  const pel = eid(k + "-ph");
  pel.textContent = ph.charAt(0).toUpperCase() + ph.slice(1);
  pel.className = "pb pb-" + ph;
}}

function drawSceneSingle(target, ctrl) {{
  const d = DATA[ctrl];
  if (!d) return;

  const cv = CVS[target];
  const c = CTX[target];
  const idx = Math.max(0, Math.min(STATE.idx, d.t.length - 1));

  c.fillStyle = "#02060f";
  c.fillRect(0, 0, cv.width, cv.height);

  if (STATE.projection === "3d") drawGrid3D(c, cv);
  else drawGrid2D(c, cv, STATE.projection);

  drawReference(c, cv, d, idx);
  drawTrajectory(c, cv, d, idx, ctrl === "PID" ? "#ef4444" : "#3b82f6");
  drawRocket(c, cv, d, idx, ctrl === "PID" ? "#ef4444" : "#3b82f6");
  drawWaypoint(c, cv);
  drawWind(c, cv, d, idx);

  updateSidebar(ctrl, d, idx);
}}

function drawSceneCombined() {{
  const cv = CVS.BOTH;
  const c = CTX.BOTH;
  const pid = DATA.PID;
  const mpc = DATA.MPC;

  c.fillStyle = "#02060f";
  c.fillRect(0, 0, cv.width, cv.height);

  if (STATE.projection === "3d") drawGrid3D(c, cv);
  else drawGrid2D(c, cv, STATE.projection);

  const idxP = pid ? Math.max(0, Math.min(STATE.idx, pid.t.length - 1)) : 0;
  const idxM = mpc ? Math.max(0, Math.min(STATE.idx, mpc.t.length - 1)) : 0;

  if (pid) {{
    drawReference(c, cv, pid, idxP);
    drawTrajectory(c, cv, pid, idxP, "#ef4444");
    drawRocket(c, cv, pid, idxP, "#ef4444");
    drawWind(c, cv, pid, idxP);
    updateSidebar("PID", pid, idxP);
  }}
  if (mpc) {{
    drawTrajectory(c, cv, mpc, idxM, "#3b82f6");
    drawRocket(c, cv, mpc, idxM, "#3b82f6");
    if (!pid) drawWind(c, cv, mpc, idxM);
    updateSidebar("MPC", mpc, idxM);
  }}

  drawWaypoint(c, cv);

  c.save();
  c.font = "bold 11px sans-serif";
  c.fillStyle = "#ef4444";
  c.fillText("PID", 12, 18);
  c.fillStyle = "#3b82f6";
  c.fillText("MPC", 12, 34);
  c.restore();
}}

function applyDisplayMode() {{
  const mode = STATE.display;
  const bp = eid("box-pid");
  const bm = eid("box-mpc");
  const bb = eid("box-both");

  bp.style.display = (mode === "split" || mode === "pid") ? "flex" : "none";
  bm.style.display = (mode === "split" || mode === "mpc") ? "flex" : "none";
  bb.style.display = (mode === "combined") ? "flex" : "none";

  resizeCanvases();
}}

function draw() {{
  if (STATE.display === "split") {{
    if (DATA.PID) drawSceneSingle("PID", "PID");
    if (DATA.MPC) drawSceneSingle("MPC", "MPC");
  }} else if (STATE.display === "pid") {{
    if (DATA.PID) drawSceneSingle("PID", "PID");
  }} else if (STATE.display === "mpc") {{
    if (DATA.MPC) drawSceneSingle("MPC", "MPC");
  }} else {{
    drawSceneCombined();
  }}

  const d = DATA.PID || DATA.MPC;
  if (d) {{
    const idx = Math.max(0, Math.min(STATE.idx, d.t.length - 1));
    eid("td").textContent = (d.t[idx] || 0).toFixed(2);
    const wvx = d.wind_vx ? (d.wind_vx[idx] || 0) : 0;
    const wvy = d.wind_vy ? (d.wind_vy[idx] || 0) : 0;
    const wdeg = (Math.atan2(wvy, wvx) * 180 / Math.PI + 360) % 360;
    const wspd = Math.sqrt(wvx * wvx + wvy * wvy);
    eid("wind-deg").textContent = wdeg.toFixed(0);
    eid("wind-spd").textContent = wspd.toFixed(1);
  }}

  const wpSec = eid("wp-sec");
  if (WAYPOINT) {{
    if (wpSec) wpSec.style.display = "block";
    const m = DATA.MPC;
    if (m && m.t && m.t.length > 0) {{
      const idxM = Math.max(0, Math.min(STATE.idx, m.t.length - 1));
      const curN = (m.xp[idxM] || 0) - WAYPOINT.xp;
      const curE = (m.y[idxM] || 0) - WAYPOINT.y;
      const curH = (m.h[idxM] || 0) - WAYPOINT.h;
      const curDist = Math.sqrt(curN * curN + curE * curE);
      const curDist3 = Math.sqrt(curN * curN + curE * curE + curH * curH);

      let best = Infinity;
      let best3 = Infinity;
      for (let i = 0; i <= idxM; i++) {{
        const dn = (m.xp[i] || 0) - WAYPOINT.xp;
        const de = (m.y[i] || 0) - WAYPOINT.y;
        const dh = (m.h[i] || 0) - WAYPOINT.h;
        const dlat = Math.sqrt(dn * dn + de * de);
        const d3 = Math.sqrt(dn * dn + de * de + dh * dh);
        if (dlat < best) best = dlat;
        if (d3 < best3) best3 = d3;
      }}

      eid("wp-lat-now").textContent = curDist.toFixed(2);
      eid("wp-lat-best").textContent = (best < Infinity ? best : 0).toFixed(2);
      eid("wp-3d-now").textContent = curDist3.toFixed(2);
      eid("wp-3d-best").textContent = (best3 < Infinity ? best3 : 0).toFixed(2);
    }} else {{
      eid("wp-lat-now").textContent = "0.00";
      eid("wp-lat-best").textContent = "0.00";
      eid("wp-3d-now").textContent = "0.00";
      eid("wp-3d-best").textContent = "0.00";
    }}
  }} else {{
    if (wpSec) wpSec.style.display = "none";
  }}

  const ml = Math.max(maxLen() - 1, 1);
  eid("sl").value = Math.round((STATE.idx / ml) * 100);
}}

function onProjectionChange(v) {{
  STATE.projection = v;
  draw();
}}

function onDisplayChange(v) {{
  STATE.display = v;
  applyDisplayMode();
  draw();
}}

function togglePlay() {{
  STATE.play = !STATE.play;
  if (STATE.play) loop();
}}

function pausePlay() {{
  STATE.play = false;
}}

function resetPlay() {{
  STATE.play = false;
  STATE.idx = 0;
  draw();
}}

function scrub(v) {{
  STATE.play = false;
  const ml = Math.max(maxLen() - 1, 1);
  STATE.idx = Math.round((Number(v) / 100) * ml);
  draw();
}}

function loop() {{
  if (!STATE.play) return;
  STATE.idx += 1;
  if (STATE.idx >= maxLen()) {{
    STATE.idx = Math.max(maxLen() - 1, 0);
    STATE.play = false;
  }}
  draw();
  if (STATE.play) setTimeout(loop, 40);
}}

function resetView() {{
  STATE.az = -35;
  STATE.el = 24;
  STATE.panX = 0;
  STATE.panY = 0;
  STATE.zoom = 1.0;
  draw();
}}

[CVS.PID, CVS.MPC, CVS.BOTH].forEach((cv) => {{
  cv.addEventListener("contextmenu", (e) => e.preventDefault());

  cv.addEventListener("mousedown", (e) => {{
    STATE.lx = e.clientX;
    STATE.ly = e.clientY;
    if (e.button === 0) STATE.orbit = true;
    if (e.button === 1) {{ STATE.pan = true; e.preventDefault(); }}
  }});

  cv.addEventListener("mousemove", (e) => {{
    const dx = e.clientX - STATE.lx;
    const dy = e.clientY - STATE.ly;
    STATE.lx = e.clientX;
    STATE.ly = e.clientY;

    if (STATE.orbit) {{
      STATE.az += dx * 0.45;
      STATE.el -= dy * 0.35;
      STATE.el = Math.max(-89, Math.min(89, STATE.el));
      draw();
    }}

    if (STATE.pan) {{
      STATE.panX += dx;
      STATE.panY += dy;
      draw();
    }}
  }});

  cv.addEventListener("wheel", (e) => {{
    e.preventDefault();
    //wider zoom range so users can zoom farther out
    STATE.zoom = Math.max(0.03, Math.min(20.0, STATE.zoom - e.deltaY * 0.002));
    draw();
  }}, {{ passive: false }});
}});

window.addEventListener("mouseup", () => {{
  STATE.orbit = false;
  STATE.pan = false;
}});

window.addEventListener("load", () => {{
  applyDisplayMode();
  resizeCanvases();
  draw();
}});
</script>
</body></html>"""

    out_path = os.path.join(outdir, "plots", "rocket_3d_interactive.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    return out_path
