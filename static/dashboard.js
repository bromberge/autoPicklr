function $(id){ return document.getElementById(id); }
const fmtUsd = n => (n==null ? '-' : '$' + Number(n).toLocaleString(undefined,{maximumFractionDigits:2}));
const fmtPct = n => (n==null ? '-' : Number(n).toLocaleString(undefined,{maximumFractionDigits:2}) + '%');
const clsPL  = n => (n>0 ? 'good' : n<0 ? 'bad' : '');

async function getJSON(url){
  const r = await fetch(url,{cache:'no-store'});
  return r.json();
}

/* -------- NAV BAR (LIVE/SIM + exchange) -------- */
function setMode(sim){
  const mode = (sim.mode || 'LIVE').toUpperCase();
  const ex   = (sim.exchange || 'KRAKEN').toUpperCase();
  $('mode-text').textContent = `${mode} / ${ex}`;
  const dot = $('mode-dot');
  dot.classList.remove('live','sim');
  if (mode === 'LIVE') dot.classList.add('live'); else dot.classList.add('sim');
}

/* -------- Helpers -------- */
function asHours(mins){
  if (mins==null) return '-';
  return (Math.floor(Number(mins)/60)).toString();
}

/* -------- KPIs -------- */
async function buildKPIs(sim){
  // Pull live cash/equity/positions from the unified Kraken route
  let live = null;
  try {
    live = await getJSON('/api/account_v2');
  } catch (e) {
    live = null;
  }
  // Win%
  const trades = await getJSON('/api/trades');
  const closed = trades.filter(t => t.exit_ts);
  const wins   = closed.filter(t => Number(t.pnl_usd||0) > 0).length;
  const winPct = closed.length ? (wins/closed.length*100) : 0;
  $('kpi-win').textContent = fmtPct(winPct);

  // Perf (CUM)
  const perf = await getJSON('/api/performance');
  const start = Number(perf.start_equity || 0);
  const cur   = Number(perf.current_equity || sim.wallet_equity || 0);
  const delta = cur - start;
  const pct   = start>0 ? (delta/start*100) : 0;
  $('kpi-cum').textContent = fmtUsd(delta);
  $('kpi-cum-pct').textContent = `(${fmtPct(pct)})`;

  // Open positions
  const openPositions = sim.open_positions || [];
  // If live positions exist, prefer them; fallback to sim
  const livePositionsAll = live?.positions || [];
  const livePositions = livePositionsAll.filter(p => Number(p.value_usd || 0) >= 0.01);
  $('kpi-open').textContent = String(livePositions.length || sim.open_positions_count || (openPositions?.length || 0));

  const liveOpenTotal = livePositions.reduce((a,p)=> a + Number(p.value_usd || 0), 0);

  const simOpenTotal  = (openPositions || []).reduce((a,p)=> a + Number(p.qty||0)*Number(p.price||0), 0);
  $('kpi-open-total').textContent = fmtUsd( livePositions.length ? liveOpenTotal : simOpenTotal );

  // Today’s PNL (unrealized snapshot)
  const todayPnl = openPositions.reduce((a,p)=>a + Number(p.pl_usd||0), 0);
  const todayPct = (cur>0) ? (todayPnl/cur*100) : 0;
  $('kpi-today').textContent = fmtUsd(todayPnl);
  $('kpi-today-pct').textContent = `(${fmtPct(todayPct)})`;

  // Cash & Equity — prefer live Kraken via /api/account_v2
  $('kpi-cash').textContent   = fmtUsd((live && typeof live.cash_usd === 'number') ? live.cash_usd : sim.wallet_balance);
  $('kpi-equity').textContent = fmtUsd((live && typeof live.equity_usd === 'number') ? live.equity_usd : cur);
}

/* -------- Positions Table -------- */
function rowPos(p){
  const plUsd = Number(p.pl_usd || 0);
  const plPct = Number(p.pl_pct || 0);
  const total = Number(p.qty||0) * Number(p.avg||0);
  const conf  = (p.confidence!=null) ? (Number(p.confidence)*100).toFixed(1)+'%' : '—';
  return `
    <tr>
      <td>${p.symbol || ''}</td>
      <td class="right mono">${Number(p.qty).toFixed(6)}</td>
      <td class="right mono">${Number(p.avg).toFixed(6)}</td>
      <td class="right mono">${fmtUsd(total)}</td>
      <td class="right mono">${Number(p.price).toFixed(6)}</td>
      <td class="right mono ${clsPL(plUsd)}">${fmtUsd(plUsd)}</td>
      <td class="right mono ${clsPL(plPct)}">${fmtPct(plPct)}</td>
      <td class="right mono">${conf}</td>
      <td class="right mono">${p.tp1 ? Number(p.tp1).toFixed(6) : '—'}</td>
      <td>${p.be ? '✓' : '—'}</td>
      <td class="right mono">${p.stop ? Number(p.stop).toFixed(6) : '—'}</td>
      <td class="right mono">${p.target ? Number(p.target).toFixed(6) : '—'}</td>
      <td class="right mono">${asHours(p.age_min)}</td>
    </tr>
  `;
}
function renderPositions(sim){
  $('rows-open').innerHTML = (sim.open_positions||[]).map(rowPos).join('');
}

/* -------- Recent Sells -------- */
function rowRecent(t){
  const pnl = Number(t.pnl_usd || 0);
  const pct = (t.entry_px>0 && t.exit_px!=null) ? ((t.exit_px / t.entry_px) - 1)*100 : null;
  const total = (t.exit_px!=null && t.qty!=null) ? Number(t.exit_px)*Number(t.qty) : null;
  return `
    <tr>
      <td>${t.symbol || ''}</td>
      <td class="right mono ${clsPL(pnl)}">${fmtUsd(pnl)}</td>
      <td class="right mono ${clsPL(pct)}">${fmtPct(pct)}</td>
      <td class="right mono">${total==null ? '-' : fmtUsd(total)}</td>
      <td>${t.reason || '-'}</td>
    </tr>
  `;
}
async function renderRecent(){
  const trades = await getJSON('/api/trades');
  const closed = trades.filter(t => t.exit_ts).sort((a,b)=> (b.exit_ts||'').localeCompare(a.exit_ts||''));
  $('rows-recent').innerHTML = closed.slice(0,12).map(rowRecent).join('');
}

/* -------- Equity Curve with axes -------- */
async function renderEquity(){
  const perf = await getJSON('/api/performance');
  const pts = (perf.equity_curve||[]).map(p => ({x:new Date(p.t).getTime(), y:Number(p.equity)}));
  const c = $('equity'); const ctx = c.getContext('2d');
  ctx.clearRect(0,0,c.width,c.height);
  if(!pts.length) return;

  const xs = pts.map(p=>p.x), ys = pts.map(p=>p.y);
  const minX=Math.min(...xs), maxX=Math.max(...xs);
  const minY=Math.min(...ys), maxY=Math.max(...ys);

  const padL=48, padR=12, padT=12, padB=40;
  const W=c.width - padL - padR, H=c.height - padT - padB;

  // Grid + axes
  ctx.strokeStyle = 'rgba(255,255,255,.15)';
  ctx.lineWidth = 1;
  ctx.font = '12px JetBrains Mono, monospace';
  ctx.fillStyle = '#9fb2c5';

  // Y ticks (5)
  for(let i=0;i<=4;i++){
    const v = minY + (i*(maxY-minY)/4);
    const y = padT + H - ((v-minY)/(maxY-minY||1))*H;
    ctx.beginPath(); ctx.moveTo(padL, y); ctx.lineTo(c.width-padR, y); ctx.stroke();
    ctx.fillText('$' + v.toFixed(0), 8, y+4);
  }
  // X ticks (5)
  for(let i=0;i<=4;i++){
    const t = minX + (i*(maxX-minX)/4);
    const x = padL + ((t-minX)/(maxX-minX||1))*W;
    ctx.beginPath(); ctx.moveTo(x, padT); ctx.lineTo(x, padT+H); ctx.stroke();
    const d = new Date(t);
    ctx.fillText(`${d.getMonth()+1}/${d.getDate()}`, x-18, padT+H+18);
  }
  // Axes
  ctx.beginPath(); ctx.moveTo(padL, padT); ctx.lineTo(padL, padT+H); ctx.lineTo(padL+W, padT+H); ctx.stroke();

  // Line
  ctx.strokeStyle = '#06b6d4';
  ctx.lineWidth = 2;
  ctx.beginPath();
  pts.forEach((p,i)=>{
    const x = padL + ((p.x-minX)/(maxX-minX||1))*W;
    const y = padT + H - ((p.y-minY)/(maxY-minY||1))*H;
    if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
  });
  ctx.stroke();
}

/* -------- Refresh -------- */
// ---- boot ----
async function refresh() {
  const sim = await getJSON('/api/sim');
  await buildKPIs(sim);

  // add these two lines:
  await renderRecent();   // #3 below
  await renderEquity();   // equity curve
}

(async () => {
  try {
    await refresh();
    setInterval(refresh, 60_000);
  } catch (e) {
    console.error('boot failed:', e);
  }
})();

