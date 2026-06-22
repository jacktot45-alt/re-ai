"""Build a self-contained HTML dashboard that replays recorded drives.

The output is a single .html file with the run data embedded, so it opens
directly in any browser (no server, no internet, no dependencies). It shows:

  * a follow-cam top-down view of the car driving the track (with trail),
  * a minimap of the whole route,
  * the 4 live camera feeds (exactly the kind of images the net learns from),
  * telemetry (speed, steering, throttle, lane offset, on/off road),
  * playback controls and a PASS/FAIL verdict.
"""
import json

_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>selfdrive — neural net driving viewer</title>
<style>
  :root { --bg:#0d1117; --panel:#161b22; --line:#30363d; --txt:#c9d1d9;
          --accent:#58a6ff; --good:#3fb950; --bad:#f85149; --warn:#d29922; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--txt);
         font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif; }
  header { padding:12px 18px; border-bottom:1px solid var(--line);
           display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
  h1 { font-size:17px; margin:0; font-weight:600; }
  h1 small { color:#8b949e; font-weight:400; }
  select, button { background:var(--panel); color:var(--txt);
                   border:1px solid var(--line); border-radius:6px;
                   padding:6px 10px; font-size:13px; cursor:pointer; }
  button:hover { border-color:var(--accent); }
  .verdict { margin-left:auto; padding:6px 14px; border-radius:6px;
             font-weight:700; font-size:13px; }
  .pass { background:rgba(63,185,80,.15); color:var(--good);
          border:1px solid var(--good); }
  .fail { background:rgba(248,81,73,.15); color:var(--bad);
          border:1px solid var(--bad); }
  main { display:grid; grid-template-columns:1fr 360px; gap:14px; padding:14px; }
  .card { background:var(--panel); border:1px solid var(--line);
          border-radius:10px; padding:12px; }
  .card h2 { font-size:12px; text-transform:uppercase; letter-spacing:.06em;
             color:#8b949e; margin:0 0 8px; }
  canvas { display:block; width:100%; image-rendering:pixel-art;
           image-rendering:crisp-edges; }
  #map { background:#0a0e14; border-radius:8px; }
  .cams { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .cam h3 { font-size:11px; margin:0 0 4px; color:#8b949e; letter-spacing:.05em; }
  .cam canvas { background:#000; border:1px solid var(--line); border-radius:4px; }
  .telem { display:grid; grid-template-columns:1fr 1fr; gap:10px 14px; }
  .stat { font-variant-numeric:tabular-nums; }
  .stat .k { font-size:11px; color:#8b949e; }
  .stat .v { font-size:20px; font-weight:600; }
  .bar { height:10px; background:#0a0e14; border-radius:5px; overflow:hidden;
         position:relative; margin-top:4px; }
  .bar > div { position:absolute; top:0; bottom:0; }
  .controls { display:flex; align-items:center; gap:10px; margin-top:6px; }
  .controls input[type=range] { flex:1; }
  .minimap { background:#0a0e14; border-radius:8px; margin-top:10px; }
  .pill { font-size:12px; padding:3px 8px; border-radius:10px; }
  .on { background:rgba(63,185,80,.15); color:var(--good); }
  .off { background:rgba(248,81,73,.15); color:var(--bad); }
  footer { color:#6e7681; font-size:12px; padding:0 18px 18px; }
</style>
</head>
<body>
<header>
  <h1>🚗 selfdrive <small>— neural-net driving replay</small></h1>
  <label>Scenario
    <select id="runSel"></select>
  </label>
  <button id="play">⏸ Pause</button>
  <label>Speed
    <select id="rate">
      <option value="0.5">0.5×</option>
      <option value="1" selected>1×</option>
      <option value="2">2×</option>
      <option value="4">4×</option>
    </select>
  </label>
  <label><input type="checkbox" id="loop" checked> loop</label>
  <span id="verdict" class="verdict"></span>
</header>

<main>
  <section class="card">
    <h2>Top-down view <span id="trackName"></span></h2>
    <canvas id="map" width="900" height="520"></canvas>
    <div class="controls">
      <span id="time" class="stat">0.0s</span>
      <input type="range" id="scrub" min="0" max="0" value="0">
      <span id="frameLbl" class="stat">0 / 0</span>
    </div>
    <canvas id="minimap" class="minimap" width="900" height="120"></canvas>
  </section>

  <section class="card">
    <h2>What the 4 cameras see</h2>
    <div class="cams" id="cams"></div>

    <h2 style="margin-top:14px">Telemetry</h2>
    <div class="telem">
      <div class="stat"><div class="k">Speed</div><div class="v"><span id="spd">0</span> <small>m/s</small></div></div>
      <div class="stat"><div class="k">Lane offset</div><div class="v"><span id="lat">0.00</span> <small>m</small></div></div>
      <div class="stat" style="grid-column:1/3">
        <div class="k">Steering</div>
        <div class="bar"><div id="steerBar" style="background:var(--accent)"></div></div>
      </div>
      <div class="stat" style="grid-column:1/3">
        <div class="k">Throttle / Brake</div>
        <div class="bar"><div id="thrBar"></div></div>
      </div>
      <div class="stat"><div class="k">Progress</div><div class="v"><span id="prog">0</span><small>%</small></div></div>
      <div class="stat"><div class="k">Status</div><div class="v"><span id="status" class="pill on">ON ROAD</span></div></div>
    </div>
  </section>
</main>
<footer id="foot"></footer>

<script>
const RUNS = __RUNS__;
const names = Object.keys(RUNS);

let cur = null, frame = 0, playing = true, rafLast = 0, acc = 0;

const $ = id => document.getElementById(id);
const map = $('map'), mctx = map.getContext('2d');
const mini = $('minimap'), nctx = mini.getContext('2d');

function buildRunSelect() {
  names.forEach(n => {
    const o = document.createElement('option'); o.value=n; o.textContent=n; $('runSel').appendChild(o);
  });
}
function buildCams(run) {
  const wrap = $('cams'); wrap.innerHTML='';
  run.cam_layout.forEach(c => {
    const d = document.createElement('div'); d.className='cam';
    d.innerHTML = '<h3>'+c.name.toUpperCase()+'</h3>';
    const cv = document.createElement('canvas'); cv.width=c.w; cv.height=c.h;
    cv.style.height = '92px'; cv.id = 'cam_'+c.name;
    d.appendChild(cv); wrap.appendChild(d);
  });
}

function setRun(name) {
  cur = RUNS[name]; frame = 0;
  $('scrub').max = cur.frames.length - 1;
  $('trackName').textContent = '· ' + cur.track.name + ' · ' + cur.track.length + ' m';
  buildCams(cur);
  const m = cur.metrics;
  const pass = m.finished && m.offroad_steps === 0;
  const v = $('verdict');
  v.className = 'verdict ' + (pass ? 'pass' : 'fail');
  v.textContent = pass ? '✓ PASS — stayed on road, finished'
                       : '✗ check: '+m.completed_pct+'% done, '+m.offroad_steps+' off-road';
  $('foot').textContent = 'Completed ' + m.completed_pct + '% (' + m.completed_m + ' m) · '
     + 'mean lane offset ' + m.mean_dev + ' m · max ' + m.max_dev + ' m · '
     + m.offroad_steps + ' off-road steps · ' + m.steps + ' frames';
  drawMinimapStatic();
}

// ---- world -> screen helpers -------------------------------------------------
function bounds(pts){let a=1e9,b=1e9,c=-1e9,d=-1e9;for(const p of pts){a=Math.min(a,p[0]);b=Math.min(b,p[1]);c=Math.max(c,p[0]);d=Math.max(d,p[1]);}return [a,b,c,d];}

function roadPath(ctx, pts, T){
  ctx.beginPath();
  pts.forEach((p,i)=>{const s=T(p[0],p[1]); if(i===0)ctx.moveTo(s[0],s[1]); else ctx.lineTo(s[0],s[1]);});
}

function drawMap(){
  const f = cur.frames[frame];
  const W=map.width, H=map.height;
  mctx.fillStyle='#0a0e14'; mctx.fillRect(0,0,W,H);
  // follow camera centred on the car, north-up
  const span = 110;                       // metres shown across the width
  const scale = W/span;
  const cx=f.x, cy=f.y;
  const T = (x,y)=>[ (x-cx)*scale + W/2, H/2 - (y-cy)*scale ];

  // road surface (thick stroke along centreline)
  mctx.lineCap='round'; mctx.lineJoin='round';
  roadPath(mctx, cur.track.pts, T);
  mctx.strokeStyle='#21262d'; mctx.lineWidth=cur.track.half_width*2*scale; mctx.stroke();
  // dashed centre line
  roadPath(mctx, cur.track.pts, T);
  mctx.strokeStyle='#e3b341'; mctx.lineWidth=Math.max(1,0.18*scale);
  mctx.setLineDash([14,14]); mctx.stroke(); mctx.setLineDash([]);

  // trail
  mctx.beginPath();
  for(let i=Math.max(0,frame-160);i<=frame;i++){const p=cur.frames[i];const s=T(p.x,p.y);
    if(i===Math.max(0,frame-160))mctx.moveTo(s[0],s[1]);else mctx.lineTo(s[0],s[1]);}
  mctx.strokeStyle='rgba(88,166,255,.7)'; mctx.lineWidth=2; mctx.stroke();

  // the car (rotated rectangle)
  const carL=4.7*scale, carW=2.0*scale;
  mctx.save(); mctx.translate(W/2,H/2); mctx.rotate(-f.th);
  mctx.fillStyle = f.on? '#58a6ff' : '#f85149';
  mctx.fillRect(-carL/2,-carW/2,carL,carW);
  mctx.fillStyle='#0a0e14'; mctx.fillRect(carL/2-carL*0.18,-carW/2,carL*0.18,carW); // windshield
  mctx.restore();
}

function drawMinimapStatic(){
  const W=mini.width,H=mini.height; nctx.fillStyle='#0a0e14'; nctx.fillRect(0,0,W,H);
  const [a,b,c,d]=bounds(cur.track.pts); const pad=20;
  const sx=(W-2*pad)/Math.max(1,(c-a)), sy=(H-2*pad)/Math.max(1,(d-b));
  const sc=Math.min(sx,sy);
  cur._mini = (x,y)=>[pad+(x-a)*sc, H-pad-(y-b)*sc];
  roadPath(nctx, cur.track.pts, cur._mini);
  nctx.strokeStyle='#30363d'; nctx.lineWidth=Math.max(2,cur.track.half_width*2*sc); nctx.stroke();
}
function drawMinimapCar(){
  drawMinimapStatic();
  // driven trail
  nctx.beginPath();
  for(let i=0;i<=frame;i+=2){const p=cur.frames[i];const s=cur._mini(p.x,p.y);
    if(i===0)nctx.moveTo(s[0],s[1]);else nctx.lineTo(s[0],s[1]);}
  nctx.strokeStyle='#58a6ff'; nctx.lineWidth=2; nctx.stroke();
  const f=cur.frames[frame]; const s=cur._mini(f.x,f.y);
  nctx.fillStyle=f.on?'#3fb950':'#f85149'; nctx.beginPath(); nctx.arc(s[0],s[1],4,0,7); nctx.fill();
}

function drawCams(){
  const f=cur.frames[frame];
  cur.cam_layout.forEach(c=>{
    const cv=$('cam_'+c.name); if(!cv)return; const ctx=cv.getContext('2d');
    const img=f.cams[c.name]; const id=ctx.createImageData(c.w,c.h);
    for(let r=0;r<c.h;r++){const row=img[r];
      for(let col=0;col<c.w;col++){const g=Math.round(parseInt(row[col],10)/9*255);
        const o=(r*c.w+col)*4; id.data[o]=g;id.data[o+1]=g;id.data[o+2]=g;id.data[o+3]=255;}}
    ctx.putImageData(id,0,0);
  });
}

function drawTelem(){
  const f=cur.frames[frame];
  $('spd').textContent=f.v.toFixed(1);
  $('lat').textContent=f.lat.toFixed(2);
  $('prog').textContent=Math.round(100*frame/(cur.frames.length-1));
  const st=$('status'); if(f.on){st.className='pill on';st.textContent='ON ROAD';}
                        else {st.className='pill off';st.textContent='OFF ROAD';}
  // steering bar: centre = straight; positive steer = LEFT turn -> fill left
  const sb=$('steerBar'); const s=Math.max(-1,Math.min(1,f.steer));
  const mag=Math.abs(s)*50;
  if(s>=0){ sb.style.left=(50-mag)+'%'; } else { sb.style.left='50%'; }
  sb.style.width=mag+'%';
  // throttle/brake bar
  const tb=$('thrBar'); const t=Math.max(-1,Math.min(1,f.thr));
  if(t>=0){tb.style.left='50%';tb.style.width=t*50+'%';tb.style.background='var(--good)';}
  else{tb.style.left=(50+t*50)+'%';tb.style.width=(-t)*50+'%';tb.style.background='var(--bad)';}
  $('time').textContent=(frame*cur.dt).toFixed(1)+'s';
  $('frameLbl').textContent=frame+' / '+(cur.frames.length-1);
  $('scrub').value=frame;
}

function render(){ drawMap(); drawMinimapCar(); drawCams(); drawTelem(); }

function tick(ts){
  if(playing && cur){
    const rate=parseFloat($('rate').value);
    acc += (ts-rafLast)/1000 * rate;
    while(acc >= cur.dt){ acc-=cur.dt; frame++;
      if(frame>=cur.frames.length){ if($('loop').checked) frame=0; else {frame=cur.frames.length-1; playing=false; $('play').textContent='▶ Play';} }
    }
  }
  rafLast=ts; if(cur) render(); requestAnimationFrame(tick);
}

// ---- wiring ------------------------------------------------------------------
buildRunSelect();
$('runSel').onchange = e => setRun(e.target.value);
$('play').onclick = () => { playing=!playing; $('play').textContent=playing?'⏸ Pause':'▶ Play'; };
$('scrub').oninput = e => { frame=parseInt(e.target.value,10); playing=false; $('play').textContent='▶ Play'; };
setRun(names[0]);
requestAnimationFrame(ts=>{rafLast=ts; tick(ts);});
</script>
</body>
</html>
"""


def build_html(runs):
    """runs: dict name -> run dict (from evaluate.run_and_record)."""
    payload = json.dumps(runs, separators=(",", ":"))
    return _HTML.replace("__RUNS__", payload)


def write_viewer(runs, path="viewer.html"):
    html = build_html(runs)
    with open(path, "w") as f:
        f.write(html)
    return path
