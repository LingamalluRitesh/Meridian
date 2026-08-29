"""
ModelForge AI Feature Store Stream Processor - Pipeline 41
"""
import time
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

class StreamingCEPProcessor_41:
    def __init__(self, window_size_sec: float = 60.0):
        self.window_size = window_size_sec
        self.events: List[Tuple[float, float]] = []

    def ingest_event(self, timestamp: float, value: float):
        self.events.append((timestamp, value))
        cutoff = timestamp - self.window_size
        self.events = [e for e in self.events if e[0] >= cutoff]

    def compute_window_statistics(self) -> Dict[str, float]:
        if not self.events:
            return {"count": 0, "mean": 0.0, "std": 0.0, "max": 0.0}
        vals = [e[1] for e in self.events]
        return {
            "count": len(vals),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "max": float(np.max(vals))
        }
