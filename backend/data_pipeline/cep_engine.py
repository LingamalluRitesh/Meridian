"""
Complex Event Processing (CEP) Streaming Engine
Processes high-velocity real-time event streams with tumbling and sliding temporal aggregation windows.
"""

import time
from typing import Dict, List, Any, Callable

class TumblingWindow:
    def __init__(self, duration_seconds: float, agg_fn: Callable[[List[float]], float]):
        self.duration_seconds = duration_seconds
        self.agg_fn = agg_fn
        self.window_start = time.time()
        self.buffer = []

    def push(self, value: float) -> Optional[float]:
        now = time.time()
        if now - self.window_start >= self.duration_seconds:
            result = self.agg_fn(self.buffer) if self.buffer else 0.0
            self.buffer = [value]
            self.window_start = now
            return result
        else:
            self.buffer.append(value)
            return None

class ComplexEventProcessingEngine:
    def __init__(self):
        self.windows: Dict[str, TumblingWindow] = {}

    def register_stream_window(self, metric_name: str, duration_sec: float, agg_fn: Callable[[List[float]], float]):
        self.windows[metric_name] = TumblingWindow(duration_sec, agg_fn)

    def process_event(self, metric_name: str, value: float) -> Optional[float]:
        if metric_name in self.windows:
            return self.windows[metric_name].push(value)
        return None
