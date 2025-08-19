// /static/app.js  (v4)

// ---------- tiny helpers ----------
const $ = (id) => document.getElementById(id);
const fmtUSD = (n) =>
  (isFinite(n) ? (n < 0 ? "-$" : "$") + Math.abs(n).toLocaleString(undefined, { maximumFractionDigits: 2 }) : "—");
const fmtNum = (n, d = 2) => (isFinite(n) ? n.toFixed(d) : "—");
const badge = (txt) => `<span class="badge badge-conf">${txt}</span>`;

async function getJSON(url) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw new Error(`${url} -> ${r.status}`);
  return r.json();
}

// ---------- charts ----------
let equityChart;
function renderEquityCurve(curve) {
  const ctx = document.getElementById("equityChart");
  if (!ctx) return;
  const labels = curve.map(p => p.ts || p[0] || "");
  const data = curve.map(p => Number(p.equity ?? p[1] ?? NaN));

  if (equityChart) { equityChart.destroy(); }
  equityChart = new Chart(ctx, {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: "Equity",
        data,
        borderWidth: 2,
        fill: false,
        tension: 0.25
      }]
    },
    options: {
      responsive: true,
      scales: {
        x: { display: labels.length > 0 },
        y: { beginAtZero: false }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// ---------- render sections ----------
function renderMetrics(sim, perf) {
  const eq = Number(sim.wallet_equity ?? perf.current_equity ?? 0);
  const pnl = Number(sim.total_pnl ?? 0);
  const win = Number(sim.win_rate ?? 0);
  const open = Number(sim.open_positions_count ?? (sim.open_positions?.length ?? 0));

  $("portfolio-value").textContent = fmtUSD(eq);
  $("total-pnl").textContent = fmtUSD(pnl);
  $("win-rate").textContent = `${fmtNum(win,0)}%`;
  $("open-positions").textContent = String(open);
}

function renderPositions(sim) {
  const tbody = $("positions-tbody");
  if (!tbody) return;
  const pos = sim.open_positions || [];
  if (!pos.length) {
    tbody.innerHTML = `<tr><td colspan="12" class="text-center text-muted">No open positions</td></tr>`;
    return;
  }

  const rows = pos.map(p => {
    const qty   = Number(p.qty ?? 0);
    const avg   = Number(p.avg ?? p.entry ?? 0);
    const price = Number(p.price ?? NaN);   // our newer /api/sim returns price; if not, will show "—"
    const plUsd = Number(p.pl_usd ?? (isFinite(price) ? (price - avg) * qty : NaN));
    const plPct = Number(p.pl_pct ?? (isFinite(price) && avg ? ((price/avg)-1)*100 : NaN));
    const conf  = (p.confidence ?? p.score ?? null);
    const tp1   = p.tp1 ?? p.tp1_price ?? null;
    const be    = p.be  ?? p.be_price  ?? (p.be_moved ? avg : null);
    const tsl   = p.tsl ?? p.tsl_price ?? null;

    const clsUsd = isFinite(plUsd) ? (plUsd >= 0 ? "neon-pos" : "neon-neg") : "";
    const clsPct = isFinite(plPct) ? (plPct >= 0 ? "neon-pos" : "neon-neg") : "";

    return `
      <tr class="row-glow">
        <td>${p.symbol}</td>
        <td class="text-end">${fmtNum(qty, 4)}</td>
        <td class="text-end">${fmtNum(avg, 4)}</td>
        <td class="text-end">${isFinite(price) ? fmtNum(price, 4) : "—"}</td>
        <td class="text-end ${clsUsd}">${fmtUSD(plUsd)}</td>
        <td class="text-end ${clsPct}">${isFinite(plPct) ? fmtNum(plPct,2)+"%" : "—"}</td>
        <td class="text-center">${conf == null ? "—" : badge(fmtNum(Number(conf)*100,0)+"%")}</td>
        <td class="text-end">${tp1 == null ? "—" : fmtNum(Number(tp1), 4)}</td>
        <td class="text-end">${be  == null ? "—" : fmtNum(Number(be), 4)}</td>
        <td class="text-end">${tsl == null ? "—" : fmtNum(Number(tsl), 4)}</td>
        <td class="text-end">${fmtNum(Number(p.stop ?? NaN), 4)}</td>
        <td class="text-end">${fmtNum(Number(p.target ?? NaN), 4)}</td>
      </tr>`;
  });

  tbody.innerHTML = rows.join("");
}

function renderRecentTrades(list) {
  const tbody = $("trades-list");
  if (!tbody) return;
  const rows = (list || []).slice(0, 10).map(t => `
    <tr>
      <td>${t.symbol}</td>
      <td class="text-end">${fmtNum(Number(t.entry_px ?? t.entry ?? NaN), 4)}</td>
      <td class="text-end">${fmtNum(Number(t.exit_px  ?? t.exit  ?? NaN), 4)}</td>
      <td class="text-end">${fmtNum(Number(t.qty ?? 0), 6)}</td>
      <td class="text-end ${Number(t.pnl_usd ?? 0) >= 0 ? "neon-pos" : "neon-neg"}">${fmtUSD(Number(t.pnl_usd ?? 0))}</td>
      <td class="text-center">${t.result ?? "—"}</td>
    </tr>
  `);
  tbody.innerHTML = rows.length ? rows.join("") : `<tr><td colspan="6" class="text-center text-muted">No trades yet</td></tr>`;
}

function renderOrders(list) {
  const tbody = $("orders-tbody");
  if (!tbody) return;
  const rows = (list || []).slice(-50).reverse().map(o => `
    <tr>
      <td>${o.ts?.replace("T"," ").slice(0,19) ?? "—"}</td>
      <td>${o.symbol}</td>
      <td>${o.side}</td>
      <td class="text-end">${fmtNum(Number(o.qty ?? 0), 6)}</td>
      <td class="text-end">${fmtNum(Number(o.price_req ?? NaN), 6)}</td>
      <td class="text-end">${fmtNum(Number(o.price_fill ?? NaN), 6)}</td>
      <td>${o.status}</td>
      <td>${o.reason ?? ""}</td>
    </tr>
  `);
  tbody.innerHTML = rows.length ? rows.join("") : `<tr><td colspan="8" class="text-center text-muted">No orders yet</td></tr>`;
}

// tabs
window.showTab = function(tab) {
  document.getElementById("orders-tab").style.display = (tab === "orders" ? "block" : "none");
  document.getElementById("trades-tab").style.display = (tab === "trades" ? "block" : "none");
};

// ---------- main refresh ----------
async function refreshData() {
  try {
    const [sim, perf, orders, trades] = await Promise.all([
      getJSON("/api/sim"),
      getJSON("/api/performance"),
      getJSON("/api/orders"),
      getJSON("/api/trades")
    ]);

    renderMetrics(sim, perf);
    renderEquityCurve(perf.equity_curve || []);
    renderPositions(sim);
    renderOrders(orders || []);
    renderRecentTrades(trades || []);
  } catch (e) {
    console.error("refresh error", e);
  }
}

window.addEventListener("load", () => {
  refreshData();
  // auto-refresh light cadence
  setInterval(refreshData, 10000);
});
