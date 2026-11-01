"""Performance Monitor page."""

from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QFormLayout, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget

from app.ui.services import AppServices
from app.ui.widgets import Card, muted_label


def _bytes(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value / (1024 ** 3):.1f} GB"


class PerformancePage(QWidget):
    def __init__(self, services: AppServices, parent: QWidget | None = None):
        super().__init__(parent)
        self.services = services
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 14, 20, 14)
        root.setSpacing(12)
        title = QLabel("Performance Monitor")
        title.setObjectName("sectionTitle")
        root.addWidget(title)
        root.addWidget(muted_label("このPC上の読み取り専用メトリクスを1秒ごとに更新します。"))
        card = Card()
        form = QFormLayout(card)
        self.cpu = QLabel("—")
        self.memory = QLabel("—")
        self.disk = QLabel("—")
        self.processes = QLabel("—")
        self.gpu = QLabel("—")
        self.model_memory = QLabel("—")
        self.speed = QLabel("—")
        form.addRow("CPU", self.cpu)
        form.addRow("メモリ", self.memory)
        form.addRow("ディスク空き", self.disk)
        form.addRow("プロセス数", self.processes)
        form.addRow("GPU VRAM", self.gpu)
        form.addRow("モデルメモリ", self.model_memory)
        form.addRow("推論速度", self.speed)
        root.addWidget(card)
        history_row = QHBoxLayout()
        history_row.addWidget(QLabel("保存済みサンプル"))
        self.clear_history = QPushButton("履歴を削除")
        self.clear_history.clicked.connect(self._clear_history)
        history_row.addWidget(self.clear_history)
        root.addLayout(history_row)
        self.history = QListWidget()
        self.history.setMaximumHeight(150)
        root.addWidget(self.history)
        root.addStretch(1)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.refresh)
        self._timer.start(1000)
        self.refresh()

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
        self.disk.setText(
            f"{_bytes(snapshot.disk_free_bytes)} / {_bytes(snapshot.disk_total_bytes)}"
        )
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
