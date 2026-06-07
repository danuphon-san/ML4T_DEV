# Telegram Portfolio Bot

This directory contains the Telegram transport for the manual portfolio workflow.
The bot does not own portfolio state. It calls the existing `research/manual_portfolio`
service functions and writes to the same JSON/JSONL state files as the CLI.

## Files

- `telegram_portfolio_bot/`: webhook runtime, access map, message formatting, and notification sender.
- `configs/telegram_bot.yaml`: local bot runtime config.
- `configs/telegram_access_map.yaml`: static Telegram chat/user authorization map.
- `run_telegram_bot.py`: starts the webhook app.
- `send_telegram_notification.py`: sends an operator message to the mapped portfolio chat.

## Configure

Set the Telegram secrets in your shell:

```bash
export TELEGRAM_BOT_TOKEN='123456:your-bot-token'
export TELEGRAM_WEBHOOK_SECRET='random-secret-token'
```

Or keep them in `Chatbot/secret.env` and source it before running:

```bash
source secret.env
```

Edit `Chatbot/configs/telegram_bot.yaml`:

```yaml
public_webhook_url: https://your-host.example.com/telegram/webhook
webhook_path: /telegram/webhook
mini_app_path: /mini-app
mini_app_title: Portfolio App
bind_host: 127.0.0.1
bind_port: 8080
state_root: ../../research/state/manual_portfolios
promotion_registry_path: ../../research/configs/manual_active_strategy.yaml
access_map_path: ./telegram_access_map.yaml
```

`public_webhook_url` must be HTTPS and must route to the local app through your
host reverse proxy or tunnel.

Edit `Chatbot/configs/telegram_access_map.yaml` with your Telegram IDs:

```yaml
portfolios:
  p1:
    chats:
      - 123456789
    users:
      - 123456789
    delivery_chat_id: 123456789
```

Access is deny-by-default. When both `chat_id` and `user_id` are present, both
must match the same portfolio entry.

## Run

Use the `research` environment because it already has the manual portfolio
package and Telegram dependencies synced:

```bash
cd /Users/mit/Project/ML4T/research
uv run python ../Chatbot/run_telegram_bot.py
```

The app exposes:

- `GET /health`
- `POST /telegram/webhook`
- `GET /mini-app`
- `GET /mini-app/api/me`
- `GET /mini-app/api/portfolios/<portfolio_id>/{overview,holdings,rebalance,activity}`
- `POST /mini-app/api/portfolios/<portfolio_id>/fills`

If `register_webhook_on_startup: true`, startup registers the webhook with
Telegram using `public_webhook_url` and `TELEGRAM_WEBHOOK_SECRET`.

## Telegram Commands

Send these commands to the bot from an authorized chat/user:

```text
/start
/app
/portfolios
/status p1
/fill p1 buy AAPL 10 100
/fill p1 sell AAPL 5 110 1.25 0.50 partial take profit
/help
```

`/fill` records through `manual_portfolio.service.record_fill`, using a
deterministic Telegram-derived `fill_id`.

`/start` and `/app` return a Telegram Web App button for the Mini App dashboard.

## Mini App

The Mini App is a mobile-first portfolio operations dashboard that reuses the
same access map and service layer as the Telegram commands. It includes:

- overview cards for equity, cash, holdings market value, and P&L
- holdings list
- top-level Portfolio Status action that refreshes the existing rebalance view
- full rebalance view with target vs actual and prefill into fill entry
- recent fills and latest daily run summary
- fill submission through the same manual ledger path

Open it from Telegram using the `Open Portfolio App` button. API requests are
authenticated with Telegram Mini App `initData`; there are no public portfolio
data endpoints without Telegram auth.

## Send A Test Notification

```bash
cd /Users/mit/Project/ML4T/research
uv run python ../Chatbot/send_telegram_notification.py \
  --portfolio-id p1 \
  --text "test message"
```

## Daily Workflow Hooks

The existing research commands still work. Add `--notify-telegram` when you want
outbound Telegram messages after local artifacts/state are written:

```bash
cd /Users/mit/Project/ML4T/research
uv run daily-workflow --notify-telegram
uv run scheduled-daily-run
uv run python record_fill.py \
  --portfolio-id p1 \
  --trade-date 2026-06-06 \
  --symbol AAPL \
  --side buy \
  --quantity 1 \
  --fill-price 100 \
  --notify-telegram
```

`daily-workflow` is the full operator loop: run the daily data update, refresh
the promoted long-only signal/backtest artifacts, run `daily-run`, verify the
promoted signal/price artifacts plus generated `daily_run.json` and
`rebalance_plan.json`, then emit Telegram summaries from the completed run
payload. The promoted signal and price artifacts must both have latest dates at
or after the workflow `as_of`; stale artifacts stop the workflow before
Telegram. `daily-run` stays computation-only and remains the source of truth for
target vs actual and rebalance guidance.

`scheduled-daily-run` is the cron-safe after-close wrapper around
`daily-workflow`. It uses `America/New_York`, waits until 4:15 PM by default if
invoked early, then runs `daily-workflow --notify-telegram`.

Example crontab entry:

```cron
15 16 * * 1-5 cd /Users/mit/Project/ML4T/research && uv run scheduled-daily-run
```

Notification failures are reported in command output and do not roll back local
portfolio state or daily artifacts.
