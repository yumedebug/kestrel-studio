"""Performance Monitor page."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtGui import QHideEvent, QShowEvent
from PySide6.QtWidgets import QGridLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from app.ui.services import AppServices
from app.ui.widgets import Card, muted_label
from app.core.resource_budget import detect_budget


def _bytes(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / (1024 ** 3):.1f} GB"


class PerformancePage(QWidget):
    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(16)
        title = QLabel("Performance Monitor")
        title.setProperty("variant", "title")
        root.addWidget(title)
        root.addWidget(muted_label("このPCとモデルの状態を、必要なときだけ読み取り専用で計測します。"))
        metric_grid = QGridLayout()
        metric_grid.setSpacing(10)
        self.cpu = self._metric(metric_grid, 0, 0, "CPU", "—")
        self.memory = self._metric(metric_grid, 0, 1, "メモリ", "—")
        self.gpu = self._metric(metric_grid, 0, 2, "GPU VRAM", "—")
        self.speed = self._metric(metric_grid, 1, 0, "推論速度", "—")
        self.model_memory = self._metric(metric_grid, 1, 1, "モデル", "—")
        self.processes = self._metric(metric_grid, 1, 2, "プロセス", "—")
        root.addLayout(metric_grid)
        budget_card = Card(soft=True)
        budget_layout = QHBoxLayout(budget_card)
        budget_layout.setContentsMargins(16, 12, 16, 12)
        budget_layout.addWidget(QLabel("メモリプロファイル"))
        self.budget = QLabel("—")
        self.budget.setProperty("variant", "success")
        budget_layout.addWidget(self.budget)
        budget_layout.addStretch(1)
        self.disk = QLabel("ディスク空き —")
        self.disk.setProperty("variant", "muted")
        budget_layout.addWidget(self.disk)
        root.addWidget(budget_card)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("計測履歴"))
        history_row.addStretch(1)
        self.clear_history = QPushButton("履歴を削除")
        self.clear_history.setProperty("variant", "ghost")
        self.clear_history.clicked.connect(self._clear_history)
        history_row.addWidget(self.clear_history)
        root.addLayout(history_row)
        self.history = QListWidget()
        self.history.setMaximumHeight(150)
        root.addWidget(self.history)
        root.addStretch(1)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.setInterval(5000)
        self.budget.setText(detect_budget(self.services.settings).label)

    def _metric(self, grid: QGridLayout, row: int, column: int, label: str, value: str) -> QLabel:
        card = Card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(15, 13, 15, 13)
        caption = QLabel(label)
        caption.setProperty("variant", "muted")
        output = QLabel(value)
        output.setProperty("variant", "heading")
        layout.addWidget(caption)
        layout.addWidget(output)
        grid.addWidget(card, row, column)
        return output

    def showEvent(self, event: QShowEvent) -> None:
        super().showEvent(event)
        self.refresh()
        self._timer.start()

    def hideEvent(self, event: QHideEvent) -> None:
        self._timer.stop()
        super().hideEvent(event)

    def refresh(self) -> None:
        snapshot = self.services.performance.sample(
            store=self.services.agent_store, inference=self.services.inference
        )
        self.cpu.setText(
            "—" if snapshot.cpu_percent is None else f"{snapshot.cpu_percent:.1f}%"
        )
        memory = (
            f"{snapshot.memory_percent:.1f}% "
            f"({_bytes(snapshot.memory_used_bytes)} / {_bytes(snapshot.memory_total_bytes)})"
            if snapshot.memory_percent is not None
            else "—"
        )
        self.memory.setText(memory)
        self.disk.setText(f"ディスク空き  {_bytes(snapshot.disk_free_bytes)} / {_bytes(snapshot.disk_total_bytes)}")
        self.processes.setText(
            "—" if snapshot.process_count is None else str(snapshot.process_count)
        )
        self.gpu.setText(
            f"{_bytes(snapshot.gpu_memory_used_bytes)} / {_bytes(snapshot.gpu_memory_total_bytes)}"
        )
        self.model_memory.setText(_bytes(snapshot.model_memory_bytes))
        self.speed.setText(
            "—" if snapshot.inference_tokens_per_second is None
            else f"{snapshot.inference_tokens_per_second:.2f} tokens/s"
        )
        self.history.clear()
        try:
            for item in self.services.agent_store.list_performance_samples(20):
                self.history.addItem(
                    f"{item.get('created_at', '')[:19]}  CPU={item.get('cpu_percent', '—')} "
                    f"MEM={item.get('memory_percent', '—')}%"
                )
        except Exception:
            pass

    def _clear_history(self) -> None:
        try:
            self.services.agent_store.clear_performance_samples()
            self.history.clear()
        except Exception:
            pass
