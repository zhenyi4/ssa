import threading
from collections import defaultdict

_lock = threading.Lock()
_enabled = False
_stats = defaultdict(lambda: {"nonzero_elements": 0, "total_elements": 0})
_call_count = 0
_num_layers = 16  # default, updated via set_num_layers()


def enable(num_layers=16):
    global _enabled, _num_layers, _call_count
    _enabled = True
    _num_layers = num_layers
    _call_count = 0


def disable():
    global _enabled
    _enabled = False


def is_enabled():
    return _enabled


def next_layer_idx():
    global _call_count
    with _lock:
        idx = _call_count % _num_layers
        _call_count += 1
        return idx


def record(layer_idx, nonzero_elements, total_elements):
    if not _enabled:
        return
    with _lock:
        s = _stats[layer_idx]
        s["nonzero_elements"] += nonzero_elements
        s["total_elements"] += total_elements


def get_stats():
    with _lock:
        return dict(_stats)


def reset():
    global _call_count
    with _lock:
        _stats.clear()
        _call_count = 0
