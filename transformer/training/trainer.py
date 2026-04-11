"""Trainer: AMP, gradient accumulation, checkpointing,
rich terminal display (YOLO-style), TensorBoard logging,
Telegram notifications, early stopping, and remote stop.
"""
from __future__ import annotations

import heapq
import os
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from .loss import LabelSmoothingLoss
from .notifier import TelegramNotifier
from .scheduler import WarmupScheduler

_console = Console()


class Trainer:
    """Full training loop for the Transformer model.

    Args:
        model:                       The Transformer model.
        train_dl:                    Training DataLoader.
        val_dl:                      Validation DataLoader.
        device:                      torch.device to use.
        d_model:                     Model dimensionality (for Noam schedule).
        warmup_steps:                Warmup steps for Noam schedule.
        label_smoothing:             Label smoothing epsilon.
        gradient_clip:               Max gradient norm.
        gradient_accumulation_steps: Accumulate gradients over N batches.
        amp:                         Enable Automatic Mixed Precision.
        save_top_k:                  Keep top-K checkpoints by validation loss.
        checkpoint_dir:              Directory to save checkpoints.
        log_every:                   Log to TensorBoard every N optimizer steps.
        log_dir:                     TensorBoard log directory.
        early_stopping_patience:     Stop if val_loss does not improve for N epochs (0=off).
        early_stopping_min_delta:    Minimum improvement threshold.
        optimizer:                   "adam" or "adamw".
        weight_decay:                L2 regularization coefficient (AdamW decoupled).
        notifier:                    TelegramNotifier instance (None=disabled).
    """

    def __init__(
        self,
        model: nn.Module,
        train_dl: DataLoader,
        val_dl: DataLoader,
        device: torch.device,
        d_model: int = 768,
        warmup_steps: int = 4000,
        label_smoothing: float = 0.1,
        gradient_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        amp: bool = True,
        save_top_k: int = 3,
        checkpoint_dir: str = "checkpoints",
        log_every: int = 100,
        log_dir: str = "runs",
        early_stopping_patience: int = 0,
        early_stopping_min_delta: float = 0.001,
        optimizer: str = "adam",
        weight_decay: float = 0.0,
        notifier: Optional[TelegramNotifier] = None,
        architecture: str = "encoder_decoder",
    ):
        self.model = model.to(device)
        self.architecture = architecture
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.device = device
        self.gradient_clip = gradient_clip
        self.grad_accum = gradient_accumulation_steps
        self.amp = amp and device.type == "cuda"
        self.save_top_k = save_top_k
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_every = log_every
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.notifier = notifier

        self.criterion = LabelSmoothingLoss(
            vocab_size=model.output_proj.out_features,
            pad_id=model.pad_id,
            smoothing=label_smoothing,
        )
        _opt_cls = torch.optim.AdamW if optimizer.lower() == "adamw" else torch.optim.Adam
        self.optimizer = _opt_cls(
            model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9,
            weight_decay=weight_decay,
        )
        self.scheduler = WarmupScheduler(self.optimizer, d_model, warmup_steps)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.amp)
        self.writer = SummaryWriter(log_dir=log_dir)

        self._heap: list = []
        self.global_step: int = 0
        self._num_epochs: int = 0

        # Early stopping state
        self._best_val_loss: float = float("inf")
        self._patience_counter: int = 0

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _gpu_mem(self) -> str:
        if self.device.type != "cuda":
            return "   cpu  "
        return f"{torch.cuda.memory_reserved(self.device) / 1e9:.2f}G"

    def _check_early_stopping(self, val_loss: float) -> bool:
        """Return True when training should stop due to no improvement."""
        if self.early_stopping_patience == 0:
            return False
        if val_loss < self._best_val_loss - self.early_stopping_min_delta:
            self._best_val_loss = val_loss
            self._patience_counter = 0
        else:
            self._patience_counter += 1
        return self._patience_counter >= self.early_stopping_patience

    def _train_progress(self, epoch: int) -> Progress:
        w = len(str(self._num_epochs))
        label = f"{epoch:>{w}}/{self._num_epochs}"
        return Progress(
            TextColumn(f"  [cyan]{label}[/]"),
            TextColumn("[dim]{task.fields[mem]:>7}[/]"),
            BarColumn(bar_width=26, complete_style="cyan", finished_style="bright_cyan"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(
                " [yellow]loss={task.fields[loss]:.4f}[/]"
                "  [dim]lr={task.fields[lr]:.2e}[/]"
            ),
            console=_console,
            transient=True,
            refresh_per_second=10,
        )

    def _val_progress(self, epoch: int) -> Progress:
        w = len(str(self._num_epochs))
        label = f"{epoch:>{w}}/{self._num_epochs} val"
        return Progress(
            TextColumn(f"  [dim]{label}[/]"),
            TextColumn("[dim]{task.fields[mem]:>7}[/]"),
            BarColumn(bar_width=26, complete_style="dim", finished_style="dim"),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=_console,
            transient=True,
            refresh_per_second=5,
        )

    # ------------------------------------------------------------------ #
    #  Epoch Methods                                                       #
    # ------------------------------------------------------------------ #

    def _compute_loss(self, batch):
        """Forward pass + loss for one batch, dispatched by architecture.

        Returns:
            (logits, targets) — both on self.device
        """
        if self.architecture == "encoder_decoder":
            src, tgt_in, tgt_out, src_mask, tgt_mask = batch
            src = src.to(self.device)
            tgt_in = tgt_in.to(self.device)
            tgt_out = tgt_out.to(self.device)
            src_mask = src_mask.to(self.device)
            tgt_mask = tgt_mask.to(self.device)
            logits = self.model(src, tgt_in, src_mask, tgt_mask)
            return logits, tgt_out
        else:
            # encoder_only and decoder_only both use a 3-tuple batch
            input_ids, labels, attn_mask = batch
            input_ids = input_ids.to(self.device)
            labels = labels.to(self.device)
            attn_mask = attn_mask.to(self.device)
            logits = self.model(input_ids, attn_mask)
            return logits, labels

    def _train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        self.optimizer.zero_grad()

        progress = self._train_progress(epoch)
        with progress:
            task = progress.add_task(
                "", total=len(self.train_dl),
                mem=self._gpu_mem(), loss=0.0, lr=self._lr(),
            )
            for i, batch in enumerate(self.train_dl):
                # Check stop flag at every batch for immediate response to /stop
                if self.notifier and self.notifier.stop_requested():
                    break

                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    logits, targets = self._compute_loss(batch)
                    loss = self.criterion(logits, targets) / self.grad_accum

                self.scaler.scale(loss).backward()

                if (i + 1) % self.grad_accum == 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1

                    if self.global_step % self.log_every == 0:
                        running = total_loss / max(n_batches, 1)
                        self.writer.add_scalar("Loss/train_step", running, self.global_step)
                        self.writer.add_scalar("LR/step", self._lr(), self.global_step)
                        # Push live state so /status always returns current info
                        if self.notifier and self.notifier.enabled:
                            self.notifier.update_state(
                                epoch=epoch,
                                num_epochs=self._num_epochs,
                                train_loss=running,
                                val_loss=self.notifier.last_val_loss,
                                lr=self._lr(),
                            )

                total_loss += loss.item() * self.grad_accum
                n_batches += 1

                progress.update(
                    task, advance=1,
                    mem=self._gpu_mem(),
                    loss=total_loss / max(n_batches, 1),
                    lr=self._lr(),
                )

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _val_epoch(self, epoch: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        progress = self._val_progress(epoch)
        with progress:
            task = progress.add_task("", total=len(self.val_dl), mem=self._gpu_mem())
            for batch in self.val_dl:
                with torch.amp.autocast(self.device.type, enabled=self.amp):
                    logits, targets = self._compute_loss(batch)
                    loss = self.criterion(logits, targets)

                total_loss += loss.item()
                n_batches += 1
                progress.update(task, advance=1, mem=self._gpu_mem())

        return total_loss / max(n_batches, 1)

    # ------------------------------------------------------------------ #
    #  Training Entry Point                                                #
    # ------------------------------------------------------------------ #

    def fit(self, num_epochs: int, start_epoch: int = 0):
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._num_epochs = num_epochs

        if self.notifier and self.notifier.enabled:
            self.notifier.start_polling()
            self.notifier.send("🚀 <b>Training started</b>\n"
                               f"<pre>epochs : {start_epoch + 1}–{num_epochs}</pre>")

        _console.print()
        _console.print(
            f"  [bold]{'Epoch':>7}  {'GPU_mem':>7}  "
            f"{'train_loss':>10}  {'val_loss':>10}  {'lr':>10}  {'time':>7}[/bold]"
        )
        _console.rule(style="dim")

        stop_reason = "complete"
        for epoch in range(start_epoch + 1, num_epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(epoch)

            # Detect mid-epoch stop: _train_epoch broke out of its batch loop early.
            # Still run validation so the checkpoint has a valid val_loss.
            mid_epoch_stop = bool(self.notifier and self.notifier.stop_requested())

            val_loss = self._val_epoch(epoch)
            elapsed = time.time() - t0
            elapsed_str = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"

            w = len(str(num_epochs))
            _console.print(
                f"  [cyan]{epoch:>{w}}/{num_epochs:<{w}}[/]  [dim]{self._gpu_mem():>7}[/]"
                f"  [bold]{train_loss:>10.4f}[/]  [green]{val_loss:>10.4f}[/]"
                f"  [blue]{self._lr():>10.2e}[/]  [dim]{elapsed_str:>7}[/]"
            )

            self.writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            self.writer.add_scalar("Loss/val_epoch", val_loss, epoch)
            self.writer.add_scalar("LR/epoch", self._lr(), epoch)

            if self.notifier and self.notifier.enabled:
                self.notifier.epoch_end(
                    epoch=epoch, num_epochs=num_epochs,
                    train_loss=train_loss, val_loss=val_loss,
                    lr=self._lr(), elapsed_str=elapsed_str,
                    gpu_mem=self._gpu_mem(), global_step=self.global_step,
                )

            # Checkpoint is always saved — including on stop and early stopping
            self._save_checkpoint(epoch, val_loss)

            if mid_epoch_stop:
                _console.print("  [yellow]⏹ Stopped mid-epoch. Checkpoint saved.[/]")
                stop_reason = "remote_stop"
                break

            # Early stopping
            if self._check_early_stopping(val_loss):
                _console.print(
                    f"  [yellow]🛑 Early stopping: no improvement for "
                    f"{self.early_stopping_patience} epochs "
                    f"(best val_loss={self._best_val_loss:.4f}).[/]"
                )
                stop_reason = "early_stop"
                break

        _console.print()
        self.writer.close()

        if self.notifier and self.notifier.enabled:
            self.notifier.training_end(
                num_epochs=epoch,
                best_val_loss=self._best_val_loss if self.early_stopping_patience > 0
                              else val_loss,
                reason=stop_reason,
            )

    # ------------------------------------------------------------------ #
    #  Checkpointing                                                       #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, epoch: int, val_loss: float):
        fname = self.checkpoint_dir / f"checkpoint_epoch{epoch:03d}.pt"
        state = {
            "epoch": epoch,
            "val_loss": val_loss,
            "architecture": self.architecture,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
        }
        torch.save(state, fname)

        # Use negated val_loss so heappop removes the worst checkpoint (highest loss),
        # not the best. Python's heapq is a min-heap, so min(-val_loss) = max(val_loss).
        heapq.heappush(self._heap, (-val_loss, str(fname)))
        while len(self._heap) > self.save_top_k:
            _, old_path = heapq.heappop(self._heap)
            if os.path.exists(old_path):
                os.remove(old_path)
                _console.print(f"  [dim]removed: {old_path}[/]")

        _console.print(f"  [dim]✓ {fname.name}  val_loss={val_loss:.4f}[/]")

    def load_checkpoint(self, checkpoint_path: str):
        state = torch.load(checkpoint_path, map_location=self.device)
        saved_arch = state.get("architecture", "encoder_decoder")
        if saved_arch != self.architecture:
            raise ValueError(
                f"Architecture mismatch: checkpoint was trained with {saved_arch!r} "
                f"but base.yaml specifies {self.architecture!r}. "
                "Update model.architecture in base.yaml to match the checkpoint."
            )
        self.model.load_state_dict(state["model_state_dict"])
        self.optimizer.load_state_dict(state["optimizer_state_dict"])
        self.scheduler.load_state_dict(state["scheduler_state_dict"])
        self.global_step = state.get("global_step", 0)
        _console.print(f"[dim]Loaded: {checkpoint_path}  (epoch={state['epoch']})[/]")
        return state["epoch"]
