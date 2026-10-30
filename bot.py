"""Binance Spot Top-200 live-candle RSI scanner + Telegram subscriber bot."""
from __future__ import annotations

import html
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

ROOT = Path(__file__).resolve().parent
SUBSCRIBERS_FILE = ROOT / "subscribers.json"
STATE_FILE = ROOT / "bot_state.json"

BINANCE_BASE_URL = os.getenv(
    "BINANCE_BASE_URL",
    "https://api.binance.com"
).rstrip("/")

TELEGRAM_BOT_TOKEN = os.getenv(
    "TELEGRAM_BOT_TOKEN", ""
).strip()

RSI_PERIOD = 14
TOP_N = 200
INTERVAL = "1h"

RSI_HIGH = 60.0
RSI_LOW = 40.0

MAX_WORKERS = 8
TELEGRAM_MESSAGE_LIMIT = 3900
KLINE_LIMIT = 100

STABLE_BASE_ASSETS = {
    "USDC", "FDUSD", "TUSD", "USDP", "DAI",
    "BUSD", "USDE", "USDS", "USD1", "USTC",
    "EUR", "AEUR", "TRY", "BRL", "GBP",
    "AUD", "BIDR", "IDRT", "NGN", "RON",
    "UAH", "ZAR", "PLN",
}

LEVERAGED_SUFFIXES = (
    "UP", "DOWN", "BULL", "BEAR"
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

log = logging.getLogger("binance-rsi-scanner")


class APIError(RuntimeError):
    pass


def load_json(path, default):

    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return data if isinstance(data, dict) else default.copy()

    except FileNotFoundError:
        return default.copy()

    except (json.JSONDecodeError, OSError):
        return default.copy()


def save_json(path, data):

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
            sort_keys=True
        )

        file.write("\n")

    temp_path.replace(path)


def http_json(url, params=None, timeout=(8, 25)):

    last_error = None

    for attempt in range(4):

        try:

            response = requests.get(
                url,
                params=params,
                timeout=timeout
            )

            if response.status_code in (418, 429):

                wait = min(
                    float(
                        response.headers.get(
                            "Retry-After",
                            2 ** attempt
                        )
                    ),
                    20.0
                )

                time.sleep(max(wait, 1.0))
                continue

            if response.status_code >= 500:

                time.sleep(2 ** attempt)
                continue

            if not response.ok:

                detail = ""

                try:
                    detail = str(
                        response.json().get("msg", "")
                    )[:160]

                except (ValueError, AttributeError):
                    pass

                raise APIError(
                    f"HTTP {response.status_code}: "
                    f"{detail or 'request rejected'}"
                )

            return response.json()

        except APIError:
            raise

        except (requests.RequestException, ValueError) as exc:

            last_error = exc

            if attempt < 3:
                time.sleep(2 ** attempt)

    raise APIError(
        "Network/API request failed "
        f"({type(last_error).__name__ if last_error else 'unknown'})."
    )


def binance_get(path, params=None):

    return http_json(
        f"{BINANCE_BASE_URL}{path}",
        params=params
    )


# RSI calculation using Wilder smoothing

def calculate_rsi_wilder(
    closes: list[float],
    period: int = RSI_PERIOD
) -> float:

    if len(closes) < period + 1:

        raise ValueError(
            f"Need at least {period + 1} closes"
        )

    changes = [
        float(closes[i]) - float(closes[i - 1])
        for i in range(1, len(closes))
    ]

    gains = [
        max(change, 0.0)
        for change in changes
    ]

    losses = [
        max(-change, 0.0)
        for change in changes
    ]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    for index in range(period, len(changes)):

        avg_gain = (
            avg_gain * (period - 1) + gains[index]
        ) / period

        avg_loss = (
            avg_loss * (period - 1) + losses[index]
        ) / period

    if avg_loss == 0:

        return 50.0 if avg_gain == 0 else 100.0

    relative_strength = avg_gain / avg_loss

    return 100.0 - (
        100.0 / (1.0 + relative_strength)
    )


def is_leveraged_or_stable(base_asset):

    base = base_asset.upper()

    return (
        base in STABLE_BASE_ASSETS
        or base.endswith(LEVERAGED_SUFFIXES)
    )


# Get top 200 Binance Spot USDT pairs

def get_top_usdt_spot_symbols(limit=TOP_N):

    exchange_info = binance_get(
        "/api/v3/exchangeInfo"
    )

    tickers = binance_get(
        "/api/v3/ticker/24hr"
    )

    eligible = set()

    for item in exchange_info.get("symbols", []):

        symbol = item.get("symbol", "")
        base = item.get("baseAsset", "")

        if (
            item.get("status") == "TRADING"
            and item.get("quoteAsset") == "USDT"
            and item.get("isSpotTradingAllowed", False)
            and symbol.endswith("USDT")
            and not is_leveraged_or_stable(base)
        ):

            eligible.add(symbol)

    ranked = []

    for ticker in tickers:

        symbol = ticker.get("symbol")

        if symbol in eligible:

            try:

                quote_volume = float(
                    ticker.get("quoteVolume", 0)
                )

            except (TypeError, ValueError):
                continue

            ranked.append(
                (symbol, quote_volume)
            )

    ranked.sort(
        key=lambda pair: pair[1],
        reverse=True
    )

    symbols = [
        symbol
        for symbol, _ in ranked[:limit]
    ]

    if not symbols:

        raise APIError(
            "Binance returned no eligible USDT Spot symbols."
        )

    return symbols


# Calculate RSI using the currently forming candle

def get_symbol_rsi(
    symbol,
    expected_open_time
):

    klines = binance_get(
        "/api/v3/klines",
        {
            "symbol": symbol,
            "interval": INTERVAL,
            "limit": KLINE_LIMIT
        }
    )

    if (
        not isinstance(klines, list)
        or len(klines) < RSI_PERIOD + 1
    ):

        return None

    # Make sure the last candle is the current open candle

    if int(klines[-1][0]) != expected_open_time:

        return None

    try:

        closes = [
            float(candle[4])
            for candle in klines
        ]

        rsi = calculate_rsi_wilder(
            closes,
            RSI_PERIOD
        )

    except (
        IndexError,
        TypeError,
        ValueError,
        ZeroDivisionError
    ):

        return None

    return symbol, rsi


# Scan all selected symbols

def scan_market():

    server_time = int(
        binance_get("/api/v3/time")["serverTime"]
    )

    hour_ms = 60 * 60 * 1000

    current_candle_open = (
        server_time // hour_ms
    ) * hour_ms

    symbols = get_top_usdt_spot_symbols(TOP_N)

    results = []

    valid_count = 0
    failures = 0

    with ThreadPoolExecutor(
        max_workers=MAX_WORKERS
    ) as pool:

        futures = {
            pool.submit(
                get_symbol_rsi,
                symbol,
                current_candle_open
            ): symbol

            for symbol in symbols
        }

        for future in as_completed(futures):

            symbol = futures[future]

            try:

                result = future.result()

                if result is None:

                    failures += 1
                    continue

                valid_count += 1

                _, rsi = result

                if (
                    rsi > RSI_HIGH
                    or rsi < RSI_LOW
                ):

                    results.append(
                        (symbol, rsi)
                    )

            except Exception as exc:

                failures += 1

                log.warning(
                    "Skipped %s (%s)",
                    symbol,
                    type(exc).__name__
                )

    # Highest RSI first

    results.sort(
        key=lambda item: item[1],
        reverse=True
    )

    log.info(
        "Scan complete: %d symbols, %d valid, %d signals",
        len(symbols),
        valid_count,
        len(results)
    )

    return results, len(symbols), valid_count


# Telegram API

def telegram_api(method, payload):

    if not TELEGRAM_BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{TELEGRAM_BOT_TOKEN}/{method}"
    )

    try:

        response = requests.post(
            url,
            json=payload,
            timeout=(8, 25)
        )

        data = response.json()

    except (requests.RequestException, ValueError) as exc:

        raise RuntimeError(
            f"Telegram API request failed ({type(exc).__name__})."
        ) from None

    if not response.ok or not data.get("ok"):

        description = str(
            data.get("description", "Telegram API error")
        )[:180]

        raise RuntimeError(
            f"Telegram API error: {description}"
        )

    return data.get("result")


# Subscribers

def load_subscribers():

    data = load_json(
        SUBSCRIBERS_FILE,
        {"chat_ids": []}
    )

    chat_ids = set()

    for value in data.get("chat_ids", []):

        try:
            chat_ids.add(int(value))

        except (TypeError, ValueError):
            continue

    return chat_ids


def save_subscribers(chat_ids):

    save_json(
        SUBSCRIBERS_FILE,
        {"chat_ids": sorted(chat_ids)}
    )


# Process /start and /stop commands

def process_telegram_updates(state, subscribers):

    params = {
        "timeout": 0,
        "allowed_updates": ["message"]
    }

    if state.get("update_offset") is not None:

        params["offset"] = int(
            state["update_offset"]
        )

    try:

        updates = telegram_api(
            "getUpdates",
            params
        )

    except Exception as exc:

        log.warning(
            "Could not poll Telegram updates: %s",
            type(exc).__name__
        )

        return

    highest_update_id = None

    for update in updates or []:

        try:

            update_id = int(update["update_id"])

            highest_update_id = (
                update_id
                if highest_update_id is None
                else max(highest_update_id, update_id)
            )

            message = update.get("message") or {}
            chat = message.get("chat") or {}

            # Only private chats

            if chat.get("type") != "private":
                continue

            chat_id = int(chat["id"])

            text = str(
                message.get("text", "")
            ).strip()

            command = (
                text.split(maxsplit=1)[0]
                .split("@", 1)[0]
                .lower()
                if text else ""
            )

            if command == "/start":

                is_new = chat_id not in subscribers

                subscribers.add(chat_id)

                if is_new:

                    telegram_api(
                        "sendMessage",
                        {
                            "chat_id": chat_id,
                            "text": (
                                "You are subscribed to "
                                "Binance RSI Scanner.\n"
                                "Use /stop to unsubscribe."
                            ),
                            "disable_web_page_preview": True
                        }
                    )

                else:

                    telegram_api(
                        "sendMessage",
                        {
                            "chat_id": chat_id,
                            "text": (
                                "You are already subscribed.\n"
                                "Use /stop to unsubscribe."
                            )
                        }
                    )

            elif command == "/stop":

                existed = chat_id in subscribers

                subscribers.discard(chat_id)

                telegram_api(
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": (
                            "You have been unsubscribed."
                            if existed
                            else "You are not subscribed."
                        )
                    }
                )

        except Exception as exc:

            log.warning(
                "Could not process Telegram update (%s)",
                type(exc).__name__
            )

    if highest_update_id is not None:

        state["update_offset"] = (
            highest_update_id + 1
        )


# TradingView link

def tradingview_url(symbol):

    return (
        "https://www.tradingview.com/chart/"
        f"?symbol=BINANCE%3A{quote(symbol, safe='')}"
        "&interval=60"
    )


# Build English-only Telegram report

def build_report(
    results,
    scanned,
    valid,
    scan_time
):

    timestamp = scan_time.astimezone(
        timezone.utc
    ).strftime("%Y-%m-%d %H:%M UTC")

    header = (
        "<b>BINANCE RSI SCANNER</b> · 1H · RSI 14\n"
        f"<code>{timestamp}</code>\n\n"
    )

    footer = (
        f"\n\n<code>{scanned} scanned · "
        f"{valid} valid · "
        f"{len(results)} signals</code>"
    )

    if not results:

        return [
            header + "No RSI signals." + footer
        ]

    messages = []
    current = header

    for symbol, rsi in results:

        safe_symbol = html.escape(symbol)

        line = (
            f'<a href="{tradingview_url(symbol)}">'
            f'{safe_symbol} — RSI {rsi:.2f}</a>\n'
        )

        if (
            len(current) + len(line) + len(footer)
            > TELEGRAM_MESSAGE_LIMIT
            and current != header
        ):

            messages.append(
                current.rstrip() + footer
            )

            current = header

        current += line

    messages.append(
        current.rstrip() + footer
    )

    return messages


# Send reports to subscribers

def send_report(subscribers, messages):

    removed = set()

    for chat_id in sorted(subscribers):

        try:

            for message in messages:

                telegram_api(
                    "sendMessage",
                    {
                        "chat_id": chat_id,
                        "text": message,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True
                    }
                )

        except Exception as exc:

            error_text = str(exc).lower()

            log.warning(
                "Could not deliver report (%s)",
                type(exc).__name__
            )

            if (
                "blocked by the user" in error_text
                or "chat not found" in error_text
            ):

                removed.add(chat_id)

    return removed


# Half-hour scheduling bucket

def current_half_hour_bucket(now):

    now = now.astimezone(timezone.utc)

    minute_bucket = (
        0 if now.minute < 30 else 30
    )

    return now.replace(
        minute=minute_bucket,
        second=0,
        microsecond=0
    ).strftime("%Y-%m-%dT%H:%MZ")


# Main

def main():

    if not TELEGRAM_BOT_TOKEN:

        log.error(
            "Missing TELEGRAM_BOT_TOKEN."
        )

        return 2

    state = load_json(
        STATE_FILE,
        {
            "update_offset": None,
            "last_scan_bucket": None
        }
    )

    subscribers = load_subscribers()

    process_telegram_updates(
        state,
        subscribers
    )

    save_subscribers(subscribers)

    now = datetime.now(timezone.utc)

    bucket = current_half_hour_bucket(now)

    force_scan = (
        os.getenv("FORCE_SCAN", "false").lower()
        in {"1", "true", "yes"}
    )

    if not subscribers:

        log.info(
            "No subscribers yet. Share the bot link."
        )

        state["last_scan_bucket"] = bucket

    elif (
        force_scan
        or state.get("last_scan_bucket") != bucket
    ):

        log.info(
            "Starting RSI scan: %s",
            bucket
        )

        try:

            results, scanned, valid = scan_market()

            report = build_report(
                results,
                scanned,
                valid,
                now
            )

            removed = send_report(
                subscribers,
                report
            )

            subscribers -= removed

            state["last_scan_bucket"] = bucket

            save_subscribers(subscribers)

        except Exception as exc:

            log.exception(
                "Scan failed (%s)",
                type(exc).__name__
            )

    else:

        log.info(
            "This half-hour bucket has already been scanned."
        )

    save_json(
        STATE_FILE,
        state
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
