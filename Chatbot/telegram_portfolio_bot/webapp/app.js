const config = window.__PORTFOLIO_APP_CONFIG__ || {};
const telegram = window.Telegram ? window.Telegram.WebApp : null;

const state = {
  me: null,
  selectedPortfolioId: null,
  overview: null,
  holdings: null,
  rebalance: null,
  activity: null,
};

const elements = {
  title: document.getElementById("app-title"),
  portfolioSelect: document.getElementById("portfolio-select"),
  identityLine: document.getElementById("identity-line"),
  equityValue: document.getElementById("equity-value"),
  cashValue: document.getElementById("cash-value"),
  holdingsValue: document.getElementById("holdings-value"),
  realizedValue: document.getElementById("realized-value"),
  unrealizedValue: document.getElementById("unrealized-value"),
  portfolioHeadline: document.getElementById("portfolio-headline"),
  rebalanceBadge: document.getElementById("rebalance-badge"),
  topDriftLine: document.getElementById("top-drift-line"),
  overviewMetrics: document.getElementById("overview-metrics"),
  overviewWarnings: document.getElementById("overview-warnings"),
  quickGuidance: document.getElementById("quick-guidance"),
  holdingsList: document.getElementById("holdings-list"),
  rebalanceCaption: document.getElementById("rebalance-caption"),
  rebalanceList: document.getElementById("rebalance-list"),
  activityList: document.getElementById("activity-list"),
  dailyRunCard: document.getElementById("daily-run-card"),
  refreshButton: document.getElementById("refresh-button"),
  jumpStatus: document.getElementById("jump-status"),
  jumpRebalance: document.getElementById("jump-rebalance"),
  jumpFill: document.getElementById("jump-fill"),
  fillForm: document.getElementById("fill-form"),
  fillDate: document.getElementById("trade-date"),
  fillSide: document.getElementById("fill-side"),
  fillSymbol: document.getElementById("fill-symbol"),
  fillQuantity: document.getElementById("fill-quantity"),
  fillPrice: document.getElementById("fill-price"),
  fillCommission: document.getElementById("fill-commission"),
  fillSlippage: document.getElementById("fill-slippage"),
  fillNotes: document.getElementById("fill-notes"),
  fillPanel: document.getElementById("fill-panel"),
  toast: document.getElementById("toast"),
};

function initTelegram() {
  if (!telegram) {
    showToast("Open this page from Telegram to load live portfolio data.");
    return;
  }
  telegram.ready();
  telegram.expand();
  telegram.enableClosingConfirmation();
}

function initDataHeader() {
  return telegram && telegram.initData ? telegram.initData : "";
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      "X-Telegram-Init-Data": initDataHeader(),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  return response.json();
}

function money(value) {
  return new Intl.NumberFormat("en-US", { style: "currency", currency: "USD" }).format(value || 0);
}

function qty(value) {
  return new Intl.NumberFormat("en-US", { maximumFractionDigits: 4 }).format(value || 0);
}

function pnlClass(value) {
  if (value > 0) return "profit";
  if (value < 0) return "loss";
  return "neutral";
}

function showToast(message) {
  elements.toast.hidden = false;
  elements.toast.textContent = message;
  window.clearTimeout(showToast.timer);
  showToast.timer = window.setTimeout(() => {
    elements.toast.hidden = true;
  }, 3200);
}

function renderMe() {
  elements.title.textContent = config.title || "Portfolio App";
  const portfolios = state.me ? state.me.portfolios : [];
  elements.portfolioSelect.innerHTML = portfolios
    .map((portfolio) => `<option value="${portfolio.portfolio_id}">${portfolio.display_name}</option>`)
    .join("");
  if (!state.selectedPortfolioId && portfolios.length > 0) {
    state.selectedPortfolioId = portfolios[0].portfolio_id;
  }
  elements.portfolioSelect.value = state.selectedPortfolioId || "";
  if (state.me) {
    const userBits = [`User ${state.me.user.user_id}`];
    if (state.me.user.username) {
      userBits.push(`@${state.me.user.username}`);
    }
    elements.identityLine.textContent = `${userBits.join(" • ")} • ${portfolios.length} portfolio(s)`;
  }
}

function renderOverview() {
  const data = state.overview;
  if (!data) return;
  elements.equityValue.textContent = money(data.equity);
  elements.cashValue.textContent = money(data.cash);
  elements.holdingsValue.textContent = money(data.holdings_market_value);
  elements.realizedValue.textContent = money(data.realized_pnl);
  elements.realizedValue.className = pnlClass(data.realized_pnl);
  elements.unrealizedValue.textContent = money(data.unrealized_pnl);
  elements.unrealizedValue.className = pnlClass(data.unrealized_pnl);
  elements.portfolioHeadline.textContent = `${data.display_name} • Strategy ${data.strategy_id} • ${data.as_of}`;
  elements.rebalanceBadge.textContent = data.is_rebalance_day ? "Rebalance day" : "Watch only";
  if (data.top_drift) {
    elements.topDriftLine.textContent =
      `Top drift: ${data.top_drift.asset} target ${data.top_drift.delta_quantity > 0 ? "+" : ""}${qty(data.top_drift.delta_quantity)} shares`;
  } else {
    elements.topDriftLine.textContent = "Portfolio is aligned with the current target.";
  }

  elements.overviewMetrics.innerHTML = [
    `<span class="metric-pill">Positions <strong>${data.position_count}</strong></span>`,
    `<span class="metric-pill">Actions <strong>${data.rebalance_action_count}</strong></span>`,
    `<span class="metric-pill">Rebalance Date <strong>${data.rebalance_date}</strong></span>`,
  ].join("");
  const warnings = data.warnings || [];
  elements.overviewWarnings.hidden = warnings.length === 0;
  elements.overviewWarnings.innerHTML = warnings
    .map((warning) => `<div class="warning-item">${warning}</div>`)
    .join("");

  const guidance = [];
  if (data.cash > 0 && data.rebalance_action_count > 0) {
    guidance.push("Use Rebalance to convert drift into concrete fills without overspending your cash view.");
  }
  if (!data.is_rebalance_day) {
    guidance.push("Today is not a rebalance day, so treat target deltas as planning guidance rather than execution instructions.");
  }
  if (data.holdings_market_value === 0) {
    guidance.push("This portfolio is cash-only right now. Equity equals cash until fills are recorded.");
  }
  if (guidance.length === 0) {
    guidance.push("Portfolio is already close to target. Review Activity for the latest ledger events.");
  }
  elements.quickGuidance.innerHTML = guidance
    .map((line) => `<li class="guidance-item">${line}</li>`)
    .join("");
}

function renderHoldings() {
  const data = state.holdings;
  if (!data) return;
  if (data.holdings.length === 0) {
    elements.holdingsList.innerHTML = `<div class="guidance-item">No positions yet. Cash remains ${money(data.cash)}.</div>`;
    return;
  }
  elements.holdingsList.innerHTML = data.holdings
    .map((row) => `
      <div class="holding-row">
        <div>
          <strong>${row.asset}</strong>
          <div class="holding-meta">
            <span>${qty(row.quantity)} shares</span>
            <span>Avg cost ${money(row.avg_cost)}</span>
          </div>
        </div>
        <div class="holding-meta" style="text-align:right">
          <strong>${money(row.market_value)}</strong>
          <span>Mark ${money(row.mark_price)}</span>
          <span class="${pnlClass(row.unrealized_pnl)}">PnL ${money(row.unrealized_pnl)}</span>
        </div>
      </div>
    `)
    .join("");
}

function prefillFill(row) {
  elements.fillSymbol.value = row.asset;
  elements.fillSide.value = row.delta_quantity >= 0 ? "buy" : "sell";
  elements.fillQuantity.value = Math.abs(row.delta_quantity).toFixed(4);
  elements.fillNotes.value = `Prefilled from rebalance for ${row.asset}`;
  elements.fillPanel.scrollIntoView({ behavior: "smooth", block: "start" });
}

function renderRebalance() {
  const data = state.rebalance;
  if (!data) return;
  elements.rebalanceCaption.textContent = data.is_rebalance_day
    ? `Execution day • ${data.instructions.length} action(s)`
    : "Preview only • not a rebalance day";
  const actionable = data.rows.filter((row) => Math.abs(row.delta_quantity) > 1e-9);
  if (actionable.length === 0) {
    elements.rebalanceList.innerHTML = `<div class="guidance-item">No rebalance actions are needed.</div>`;
    return;
  }
  elements.rebalanceList.innerHTML = actionable
    .map((row) => `
      <div class="rebalance-row">
        <div>
          <strong>${row.asset}</strong>
          <div class="rebalance-meta">
            <span>${row.action.toUpperCase()} ${qty(Math.abs(row.delta_quantity))} shares</span>
            <span>Actual ${qty(row.actual_quantity)} -> Target ${qty(row.target_quantity)}</span>
            <span>Value gap ${money(row.target_value - row.actual_value)}</span>
          </div>
        </div>
        <button class="rebal-action secondary-button" type="button" data-asset="${row.asset}">Use</button>
      </div>
    `)
    .join("");

  elements.rebalanceList.querySelectorAll("[data-asset]").forEach((button) => {
    button.addEventListener("click", () => {
      const row = actionable.find((item) => item.asset === button.dataset.asset);
      if (row) {
        prefillFill(row);
      }
    });
  });
}

function renderActivity() {
  const activity = state.activity;
  if (!activity) return;
  if (activity.recent_fills.length === 0) {
    elements.activityList.innerHTML = `<div class="guidance-item">No fills have been recorded yet.</div>`;
  } else {
    elements.activityList.innerHTML = activity.recent_fills
      .map((row) => `
        <div class="activity-row">
          <div>
            <strong>${row.side.toUpperCase()} ${row.symbol}</strong>
            <div class="activity-meta">
              <span>${row.trade_date} • ${qty(row.quantity)} @ ${money(row.fill_price)}</span>
              <span>${row.notes || "No notes"}</span>
            </div>
          </div>
          <div class="activity-meta" style="text-align:right">
            <span>Commission ${money(row.commission)}</span>
            <span>Slippage ${money(row.slippage)}</span>
          </div>
        </div>
      `)
      .join("");
  }

  if (!activity.latest_daily_run) {
    elements.dailyRunCard.innerHTML = `<div class="guidance-item">No daily run output found yet.</div>`;
    return;
  }
  const run = activity.latest_daily_run;
  elements.dailyRunCard.innerHTML = `
    <div class="daily-row">
      <div>
        <strong>${run.as_of}</strong>
        <p>Strategy ${run.strategy_id}</p>
      </div>
      <span class="pill">${run.is_rebalance_day ? "Rebalance day" : "Non-rebalance day"}</span>
    </div>
    <div class="daily-row">
      <span>Equity</span>
      <strong>${money(run.equity)}</strong>
    </div>
    <div class="daily-row">
      <span>Cash</span>
      <strong>${money(run.cash)}</strong>
    </div>
    <div class="daily-row">
      <span>Instructions</span>
      <strong>${run.rebalance_instruction_count}</strong>
    </div>
  `;
}

function activateTab(tabName) {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("is-active", panel.id === `tab-${tabName}`);
  });
}

async function loadPortfolioData() {
  if (!state.selectedPortfolioId) return;
  const base = `${config.apiBasePath}/portfolios/${state.selectedPortfolioId}`;
  const [overview, holdings, rebalance, activity] = await Promise.all([
    api(`${base}/overview`),
    api(`${base}/holdings`),
    api(`${base}/rebalance`),
    api(`${base}/activity`),
  ]);
  state.overview = overview;
  state.holdings = holdings;
  state.rebalance = rebalance;
  state.activity = activity;
  renderOverview();
  renderHoldings();
  renderRebalance();
  renderActivity();
}

async function boot() {
  initTelegram();
  elements.fillDate.value = new Date().toISOString().slice(0, 10);
  elements.title.textContent = config.title || "Portfolio App";

  if (!telegram || !telegram.initData) {
    showToast("Open this page from Telegram using the bot button.");
    return;
  }

  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab));
  });

  elements.refreshButton.addEventListener("click", async () => {
    try {
      await loadPortfolioData();
      showToast("Portfolio refreshed.");
    } catch (error) {
      showToast(error.message);
    }
  });

  elements.jumpStatus.addEventListener("click", async () => {
    try {
      await loadPortfolioData();
      activateTab("rebalance");
      showToast("Portfolio status refreshed.");
    } catch (error) {
      showToast(error.message);
    }
  });
  elements.jumpRebalance.addEventListener("click", () => activateTab("rebalance"));
  elements.jumpFill.addEventListener("click", () => {
    elements.fillPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  });

  elements.portfolioSelect.addEventListener("change", async (event) => {
    state.selectedPortfolioId = event.target.value;
    await loadPortfolioData();
  });

  elements.fillForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!state.selectedPortfolioId) return;
    const payload = {
      trade_date: elements.fillDate.value,
      side: elements.fillSide.value,
      symbol: elements.fillSymbol.value.trim().toUpperCase(),
      quantity: Number(elements.fillQuantity.value),
      fill_price: Number(elements.fillPrice.value),
      commission: Number(elements.fillCommission.value || 0),
      slippage: Number(elements.fillSlippage.value || 0),
      notes: elements.fillNotes.value.trim(),
      client_request_id: window.crypto && crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`,
    };
    try {
      const response = await api(`${config.apiBasePath}/portfolios/${state.selectedPortfolioId}/fills`, {
        method: "POST",
        body: JSON.stringify(payload),
      });
      state.overview = response.overview;
      renderOverview();
      await loadPortfolioData();
      elements.fillNotes.value = "";
      showToast(`Fill recorded: ${response.fill.fill_id}`);
      if (telegram) {
        telegram.HapticFeedback.notificationOccurred("success");
      }
    } catch (error) {
      showToast(error.message);
      if (telegram) {
        telegram.HapticFeedback.notificationOccurred("error");
      }
    }
  });

  try {
    state.me = await api(`${config.apiBasePath}/me`);
    renderMe();
    await loadPortfolioData();
  } catch (error) {
    showToast(error.message);
  }
}

boot();
