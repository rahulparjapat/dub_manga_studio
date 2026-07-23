"""Lightweight Prometheus-style metrics for production readiness.

No external dependency is required; this keeps the app runnable in the existing
small environment while exposing a standard /metrics text endpoint.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock
from typing import Any


class MetricsRegistry:
    def __init__(self) -> None:
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = defaultdict(
            list
        )
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._lock = Lock()
        self.started_at = time.time()

    def inc(self, name: str, value: float = 1.0, **labels: Any) -> None:
        with self._lock:
            self._counters[(name, _labels(labels))] += value

    def observe(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._histograms[(name, _labels(labels))].append(value)

    def set(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges[(name, _labels(labels))] = value

    def render_prometheus(self) -> str:
        lines = [
            "# HELP cms_info Chatterbox Manga Studio application info",
            "# TYPE cms_info gauge",
            'cms_info{version="1.0.0"} 1',
        ]
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"# TYPE {name} counter")
                lines.append(f"{name}{_format_labels(labels)} {value}")
            for (name, labels), values in sorted(self._histograms.items()):
                if not values:
                    continue
                lines.append(f"# TYPE {name} summary")
                lines.append(f"{name}_count{_format_labels(labels)} {len(values)}")
                lines.append(f"{name}_sum{_format_labels(labels)} {sum(values)}")
                lines.append(f"{name}_max{_format_labels(labels)} {max(values)}")
            for (name, labels), value in sorted(self._gauges.items()):
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name}{_format_labels(labels)} {value}")
        lines.append(f"cms_uptime_seconds {time.time() - self.started_at:.3f}")
        return "\n".join(lines) + "\n"


def _labels(labels: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(k), str(v)) for k, v in labels.items() if v is not None))


def _format_labels(labels: tuple[tuple[str, str], ...]) -> str:
    if not labels:
        return ""
    escaped = ",".join(f'{k}="{v.replace(chr(34), chr(92)+chr(34))}"' for k, v in labels)
    return "{" + escaped + "}"


metrics = MetricsRegistry()
