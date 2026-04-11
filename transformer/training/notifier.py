"""Telegram notifier + remote command handler for training.

Setup (~5 min):
  1. Open Telegram, message @BotFather  →  /newbot  →  choose a name  →  get TOKEN
  2. Send any message to your new bot to activate it
  3. Retrieve your chat ID:
       curl "https://api.telegram.org/bot<TOKEN>/getUpdates"
       The value at result[0].message.chat.id is your TELEGRAM_CHAT_ID
  4. Add credentials to .env (project root):
       TELEGRAM_BOT_TOKEN=123456:ABC-DEF...
       TELEGRAM_CHAT_ID=123456789

Supported bot commands:
  /status   current epoch, loss, lr
  /stop     stop training after the current epoch finishes
  /help     list all commands
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional


try:
    import requests as _req
    _OK = True
except ImportError:
    _OK = False

_HELP = (
    "<b>Commands</b>\n"
    "/status  – current training state\n"
    "/stop    – stop after current epoch\n"
    "/help    – this message"
)


class TelegramNotifier:
    """Sends per-epoch training summaries and listens for remote commands.

    Silently skips if credentials are missing or a request fails —
    training is never interrupted by notification errors.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.environ.get("TELEGRAM_BOT_TOKEN", "")
        self.chat_id = str(chat_id or os.environ.get("TELEGRAM_CHAT_ID", ""))
        self._enabled = bool(self.bot_token and self.chat_id and _OK)

        self._stop_event = threading.Event()
        self._poll_thread: Optional[threading.Thread] = None
        self._status_cb: Optional[Callable[[], str]] = None
        self._last_update_id = 0

        # Training state — updated by the trainer so /status replies are current
        self._current_epoch = 0
        self._num_epochs = 0
        self._last_train_loss = 0.0
        self._last_val_loss = 0.0
        self._last_lr = 0.0
        self._last_elapsed_str = ""
        self._last_gpu_mem = ""
        self._last_global_step = 0

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stop_requested(self) -> bool:
        return self._stop_event.is_set()

    @property
    def last_val_loss(self) -> float:
        return self._last_val_loss

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start_polling(self) -> None:
        """Start the background command-polling thread."""
        if not self._enabled:
            return
        self._poll_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="tg-poll"
        )
        self._poll_thread.start()

    def update_state(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        val_loss: float,
        lr: float,
    ) -> None:
        self._current_epoch = epoch
        self._num_epochs = num_epochs
        self._last_train_loss = train_loss
        self._last_val_loss = val_loss
        self._last_lr = lr

    def epoch_end(
        self,
        epoch: int,
        num_epochs: int,
        train_loss: float,
        val_loss: float,
        lr: float,
        elapsed_str: str,
        gpu_mem: str,
        global_step: int,
    ) -> None:
        self.update_state(epoch, num_epochs, train_loss, val_loss, lr)
        self._last_elapsed_str = elapsed_str
        self._last_gpu_mem = gpu_mem
        self._last_global_step = global_step
        icon = "✅" if epoch == num_epochs else "📊"
        text = (
            f"{icon} <b>Epoch {epoch}/{num_epochs}</b>\n"
            f"<pre>"
            f"train_loss : {train_loss:.4f}\n"
            f"  val_loss : {val_loss:.4f}\n"
            f"        lr : {lr:.2e}\n"
            f"      time : {elapsed_str}\n"
            f"       gpu : {gpu_mem}\n"
            f"      step : {global_step:,}"
            f"</pre>"
        )
        self._send(text)

    def training_end(self, num_epochs: int, best_val_loss: float, reason: str = "complete") -> None:
        icons = {"complete": "🏁", "early_stop": "🛑", "remote_stop": "⏹"}
        labels = {"complete": "complete", "early_stop": "early stopping", "remote_stop": "remote stop"}
        icon = icons.get(reason, "🏁")
        label = labels.get(reason, "complete")
        text = (
            f"{icon} <b>Training {label}</b> ({num_epochs} epochs)\n"
            f"<pre>best val_loss : {best_val_loss:.4f}</pre>"
        )
        self._send(text)

    def send(self, text: str) -> None:
        self._send(text)

    # ------------------------------------------------------------------ #
    #  Polling                                                             #
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        while True:
            try:
                resp = _req.get(
                    f"https://api.telegram.org/bot{self.bot_token}/getUpdates",
                    params={"offset": self._last_update_id + 1, "timeout": 10},
                    timeout=15,
                )
                for update in resp.json().get("result", []):
                    self._last_update_id = update["update_id"]
                    self._handle_update(update)
            except Exception:
                pass
            time.sleep(1)

    def _handle_update(self, update: dict) -> None:
        msg = update.get("message", {})
        if str(msg.get("chat", {}).get("id")) != self.chat_id:
            return
        text = msg.get("text", "").strip().lower()

        if text == "/stop":
            self._stop_event.set()
            self._send("⏹ <b>Stop requested.</b> Training will finish after the current epoch.")
        elif text == "/status":
            self._send(self._build_status())
        elif text == "/help":
            self._send(_HELP)

    def _build_status(self) -> str:
        if self._current_epoch == 0:
            return "⏳ Training has not started yet."
        lines = [
            f"epoch      : {self._current_epoch}/{self._num_epochs}",
            f"train_loss : {self._last_train_loss:.4f}",
            f"  val_loss : {self._last_val_loss:.4f}",
            f"        lr : {self._last_lr:.2e}",
        ]
        if self._last_elapsed_str:
            lines.append(f"      time : {self._last_elapsed_str}")
        if self._last_gpu_mem:
            lines.append(f"       gpu : {self._last_gpu_mem}")
        if self._last_global_step:
            lines.append(f"      step : {self._last_global_step:,}")
        return "⚙️ <b>Status</b>\n<pre>" + "\n".join(lines) + "</pre>"

    def _send(self, text: str) -> None:
        if not self._enabled:
            return
        try:
            _req.post(
                f"https://api.telegram.org/bot{self.bot_token}/sendMessage",
                json={"chat_id": self.chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception:
            pass
