"""Telegram bot launcher: start, stop, and monitor training remotely.

Run this in a separate terminal alongside (or instead of) train.py:

    uv run python scripts/bot_launcher.py

Commands (send to your bot):
    /train                        start training with current config
    /train epochs=10              start with a single override
    /train epochs=10 batch=32     start with multiple overrides
    /stop                         stop training (checkpoint is saved first)
    /status                       is training running?
    /logs                         last 15 lines of output
    /help                         command reference

Supported override keys:
    epochs, batch_size (alias: batch), learning_rate (alias: lr),
    warmup_steps, early_stopping_patience (alias: patience)

Required environment variables:
    TELEGRAM_BOT_TOKEN   bot token from @BotFather
    TELEGRAM_CHAT_ID     your personal chat ID
"""
from __future__ import annotations

import io
import os
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml

try:
    import requests as _req
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

# Load .env before reading any environment variables
from dotenv import load_dotenv
load_dotenv()

# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = str(os.environ.get("TELEGRAM_CHAT_ID", ""))
BASE_CONFIG = Path("configs/base.yaml")
TRAIN_SCRIPT = Path("scripts/train.py")

_HELP_TEXT = (
    "<b>Bot Launcher Commands</b>\n\n"
    "/train                 – start training with base config\n"
    "/train epochs=10       – start with epoch override\n"
    "/train epochs=5 batch=32  – start with multiple overrides\n"
    "/stop                  – stop training gracefully\n"
    "/status                – show process status\n"
    "/logs                  – last 15 lines of output\n"
    "/help                  – this message\n\n"
    "<i>Supported override keys:</i>\n"
    "epochs, batch (batch_size), lr (learning_rate),\n"
    "warmup_steps, patience (early_stopping_patience)"
)

_FIELD_ALIASES = {
    "batch": "batch_size",
    "lr": "learning_rate",
    "patience": "early_stopping_patience",
}

# ------------------------------------------------------------------ #
#  Shared state                                                        #
# ------------------------------------------------------------------ #

_process: subprocess.Popen | None = None
_process_lock = threading.Lock()
_log_buffer: list[str] = []
_log_lock = threading.Lock()
_last_update_id = 0

# ------------------------------------------------------------------ #
#  Telegram helpers                                                    #
# ------------------------------------------------------------------ #

def _send(text: str) -> None:
    if not (_REQUESTS_OK and BOT_TOKEN and CHAT_ID):
        print(text)
        return
    try:
        _req.post(
            f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
            json={"chat_id": CHAT_ID, "text": text, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception:
        pass


def _poll_updates() -> list[dict]:
    global _last_update_id
    try:
        resp = _req.get(
            f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates",
            params={"offset": _last_update_id + 1, "timeout": 10},
            timeout=15,
        )
        updates = resp.json().get("result", [])
        if updates:
            _last_update_id = updates[-1]["update_id"]
        return updates
    except Exception:
        return []

# ------------------------------------------------------------------ #
#  Process management                                                  #
# ------------------------------------------------------------------ #

def _stream_output(proc: subprocess.Popen) -> None:
    """Pipe subprocess stdout into the shared log buffer (daemon thread)."""
    for line in io.TextIOWrapper(proc.stdout, encoding="utf-8", errors="replace"):
        line = line.rstrip()
        with _log_lock:
            _log_buffer.append(line)
            if len(_log_buffer) > 200:
                _log_buffer.pop(0)


def _parse_overrides(args_str: str) -> dict[str, str]:
    """Parse 'epochs=10 batch=32' into {'epochs': '10', 'batch': '32'}."""
    return {k.lower(): v for k, v in re.findall(r"(\w+)=([^\s]+)", args_str)}


def _write_temp_config(overrides: dict[str, str]) -> str:
    """Apply overrides to base config, write a temp YAML file, return its path."""
    with open(BASE_CONFIG) as f:
        cfg = yaml.safe_load(f)

    for key, raw_val in overrides.items():
        real_key = _FIELD_ALIASES.get(key, key)
        section = cfg.get("training", {})
        if real_key in section:
            try:
                cfg["training"][real_key] = type(section[real_key])(raw_val)
            except (ValueError, TypeError):
                cfg["training"][real_key] = raw_val
        else:
            _send(f"⚠️ Unknown override key: <code>{key}</code> — skipped.")

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir="configs", prefix="tmp_launch_"
    )
    yaml.dump(cfg, tmp, allow_unicode=True, default_flow_style=False)
    tmp.close()
    return tmp.name

# ------------------------------------------------------------------ #
#  Command handlers                                                    #
# ------------------------------------------------------------------ #

def cmd_train(args_str: str) -> None:
    global _process
    with _process_lock:
        if _process and _process.poll() is None:
            _send("⚠️ Training is already running. Send /stop first.")
            return

        overrides = _parse_overrides(args_str)
        if overrides:
            config_path = _write_temp_config(overrides)
            summary = "  ".join(f"{k}={v}" for k, v in overrides.items())
            _send(f"🚀 <b>Training started</b>\n<pre>{summary}</pre>")
        else:
            config_path = str(BASE_CONFIG)
            _send("🚀 <b>Training started</b> (base config)")

        with _log_lock:
            _log_buffer.clear()

        _process = subprocess.Popen(
            [sys.executable, str(TRAIN_SCRIPT), "--config", config_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=Path.cwd(),
        )
        threading.Thread(
            target=_stream_output, args=(_process,), daemon=True
        ).start()


def cmd_stop() -> None:
    with _process_lock:
        if _process is None or _process.poll() is not None:
            _send("ℹ️ No training process is running.")
            return
        _process.send_signal(signal.SIGTERM)
        _send("⏹ <b>SIGTERM sent.</b> Training will stop after the current epoch.")


def cmd_status() -> None:
    with _process_lock:
        if _process is None:
            _send("ℹ️ Training has not been started yet.")
            return
        code = _process.poll()
    if code is None:
        _send(f"⚙️ <b>Training is running</b>\n<pre>PID : {_process.pid}</pre>")
    else:
        _send(f"🏁 <b>Training finished</b>\n<pre>exit code : {code}</pre>")


def cmd_logs() -> None:
    with _log_lock:
        lines = list(_log_buffer[-15:])
    if not lines:
        _send("📋 No output yet.")
        return
    _send(f"<pre>{chr(10).join(lines)[:3500]}</pre>")

# ------------------------------------------------------------------ #
#  Dispatcher                                                          #
# ------------------------------------------------------------------ #

def _dispatch(text: str) -> None:
    text = text.strip()
    if text.lower().startswith("/train"):
        cmd_train(text[6:].strip())
    elif text == "/stop":
        cmd_stop()
    elif text == "/status":
        cmd_status()
    elif text == "/logs":
        cmd_logs()
    elif text == "/help":
        _send(_HELP_TEXT)
    else:
        _send(f"❓ Unknown command: <code>{text}</code>\nSee /help.")


def _handle_update(update: dict) -> None:
    msg = update.get("message", {})
    if str(msg.get("chat", {}).get("id")) != CHAT_ID:
        return  # ignore messages from other chats
    text = msg.get("text", "").strip()
    if text:
        _dispatch(text)

# ------------------------------------------------------------------ #
#  Entry point                                                         #
# ------------------------------------------------------------------ #

def main() -> None:
    if not BOT_TOKEN or not CHAT_ID:
        print("ERROR: TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set.")
        sys.exit(1)
    if not _REQUESTS_OK:
        print("ERROR: 'requests' package not found. Run: uv sync")
        sys.exit(1)

    print(f"Bot launcher ready. Listening for commands (chat_id={CHAT_ID})")
    _send("🤖 <b>Bot Launcher ready.</b>\nSend /help to see available commands.")

    while True:
        for update in _poll_updates():
            _handle_update(update)
        time.sleep(1)


if __name__ == "__main__":
    main()
