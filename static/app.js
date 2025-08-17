// static/app.js

let equityChart = null;

// Initialize the dashboard
document.addEventListener('DOMContentLoaded', function() {
    initializeEquityChart();
    refreshData();
    
    // Auto-refresh every 30 seconds
    setInterval(refreshData, 30000);
});

function initializeEquityChart() {
    const ctx = document.getElementById('equityChart').getContext('2d');
    equityChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Portfolio Value',
                data: [],
                borderColor: 'rgb(13, 202, 240)',
                backgroundColor: 'rgba(13, 202, 240, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)'
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    },
                    ticks: {
                        color: 'rgba(255, 255, 255, 0.7)',
                        callback: function(value) {
                            return '$' + value.toFixed(2);
                        }
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index'
            }
        }
    });
}

async function refreshData() {
    try {
        // Update portfolio metrics
        const simData = await fetch('/api/sim').then(r => r.json());
        updatePortfolioMetrics(simData);
        updatePositions(simData.open_positions);
        updateRecentTrades(simData.recent_trades);
        
        // Update performance chart
        const performanceData = await fetch('/api/performance').then(r => r.json());
        updateEquityChart(performanceData.equity_curve);
        
        // Update orders and trades tables
        const ordersData = await fetch('/api/orders').then(r => r.json());
        const tradesData = await fetch('/api/trades').then(r => r.json());
        updateOrdersTable(ordersData);
        updateTradesTable(tradesData);
        
    } catch (error) {
        console.error('Error refreshing data:', error);
    }
}

function updatePortfolioMetrics(data) {
    document.getElementById('portfolio-value').textContent = '$' + (data.wallet_equity || 0).toFixed(2);
    
    const pnlElement = document.getElementById('total-pnl');
    const pnl = data.total_pnl || 0;
    pnlElement.textContent = '$' + pnl.toFixed(2);
    pnlElement.className = pnl >= 0 ? 'text-success' : 'text-danger';
    
    document.getElementById('win-rate').textContent = (data.win_rate || 0).toFixed(1) + '%';
    document.getElementById('open-positions').textContent = data.open_positions_count || 0;
}

function updatePositions(positions) {
    const container = document.getElementById('positions-list');
    
    if (!positions || positions.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-chart-line fa-3x mb-3"></i>
                <p>No open positions</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = positions.map(pos => `
        <div class="border-bottom border-secondary pb-3 mb-3">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <h6 class="mb-1">${pos.symbol}</h6>
                    <small class="text-muted">Qty: ${pos.qty.toFixed(4)}</small>
                </div>
                <div class="text-end">
                    <div class="text-info">$${pos.entry.toFixed(2)}</div>
                    <small class="text-muted">Entry</small>
                </div>
            </div>
            <div class="row mt-2">
                <div class="col-6">
                    <small class="text-danger">Stop: $${pos.stop.toFixed(2)}</small>
                </div>
                <div class="col-6 text-end">
                    <small class="text-success">Target: $${pos.target.toFixed(2)}</small>
                </div>
            </div>
        </div>
    `).join('');
}

function updateRecentTrades(trades) {
    const container = document.getElementById('trades-list');
    
    if (!trades || trades.length === 0) {
        container.innerHTML = `
            <div class="text-center text-muted py-4">
                <i class="fas fa-exchange-alt fa-3x mb-3"></i>
                <p>No trades yet</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = trades.slice(0, 5).map(trade => `
        <div class="border-bottom border-secondary pb-2 mb-2">
            <div class="d-flex justify-content-between align-items-center">
                <div>
                    <span class="fw-bold">${trade.symbol}</span>
                    <span class="status-indicator ${trade.result === 'WIN' ? 'status-win' : 'status-loss'}"></span>
                </div>
                <div class="text-end">
                    <div class="${trade.pnl >= 0 ? 'text-success' : 'text-danger'}">
                        $${trade.pnl.toFixed(2)}
                    </div>
                    <small class="text-muted">${formatDateTime(trade.entry_ts)}</small>
                </div>
            </div>
        </div>
    `).join('');
}

function updateEquityChart(equityData) {
    if (!equityData || equityData.length === 0) {
        equityChart.data.labels = [];
        equityChart.data.datasets[0].data = [];
    } else {
        equityChart.data.labels = equityData.map(point => formatDate(point.date));
        equityChart.data.datasets[0].data = equityData.map(point => point.equity);
    }
    equityChart.update('none');
}

function updateOrdersTable(orders) {
    const tbody = document.getElementById('orders-tbody');
    
    if (!orders || orders.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No orders yet</td></tr>';
        return;
    }
    
    tbody.innerHTML = orders.slice(0, 20).map(order => `
        <tr>
            <td>${formatDateTime(order.ts)}</td>
            <td><span class="badge bg-primary">${order.symbol}</span></td>
            <td><span class="badge ${order.side === 'BUY' ? 'bg-success' : 'bg-danger'}">${order.side}</span></td>
            <td>${order.qty.toFixed(4)}</td>
            <td>$${(order.price_fill || order.price_req).toFixed(2)}</td>
            <td><span class="badge ${getStatusColor(order.status)}">${order.status}</span></td>
        </tr>
    `).join('');
}

function updateTradesTable(trades) {
    const tbody = document.getElementById('trades-tbody');
    
    if (!trades || trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No trades yet</td></tr>';
        return;
    }
    
    tbody.innerHTML = trades.slice(0, 50).map(trade => `
        <tr>
            <td><span class="badge bg-primary">${trade.symbol}</span></td>
            <td>$${trade.entry_px.toFixed(2)}</td>
            <td>$${(trade.exit_px || 0).toFixed(2)}</td>
            <td>${trade.qty.toFixed(4)}</td>
            <td class="${(trade.pnl_usd || 0) >= 0 ? 'text-success' : 'text-danger'}">
                $${(trade.pnl_usd || 0).toFixed(2)}
            </td>
            <td>
                <span class="badge ${trade.result === 'WIN' ? 'bg-success' : 'bg-danger'}">
                    ${trade.result || 'OPEN'}
                </span>
            </td>
        </tr>
    `).join('');
}

function showTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.activity-tab').forEach(tab => {
        tab.style.display = 'none';
    });
    
    // Show selected tab
    document.getElementById(tabName + '-tab').style.display = 'block';
    
    // Update button states
    document.querySelectorAll('.btn-group button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
}

function getStatusColor(status) {
    switch(status) {
        case 'FILLED': return 'bg-success';
        case 'NEW': return 'bg-warning';
        case 'CANCELLED': return 'bg-secondary';
        case 'REJECTED': return 'bg-danger';
        default: return 'bg-info';
    }
}

function formatDateTime(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleString('en-US', {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function formatDate(isoString) {
    if (!isoString) return 'N/A';
    const date = new Date(isoString);
    return date.toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric'
    });
}
