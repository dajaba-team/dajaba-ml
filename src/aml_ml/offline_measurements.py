"""선택적으로 사용하는 실측 집계이며 물리 I/O나 캐시 상태를 단정하지 않습니다."""

import os
import threading
import time
from collections import Counter
from contextvars import ContextVar

import pyarrow as pa

_CURRENT: ContextVar["Measurement | None"] = ContextVar("offline_measurement", default=None)


def event(name, amount=1):
    measurement = _CURRENT.get()
    if measurement is not None:
        measurement.add(name, amount)


def milestone(name):
    measurement = _CURRENT.get()
    if measurement is not None:
        measurement.mark(name)


def observe(name, value):
    measurement = _CURRENT.get()
    if measurement is not None:
        with measurement.lock:
            counts = measurement.counts
            previous = counts[name + "_count"]
            counts[name + "_count"] += 1
            counts[name + "_sum"] += value
            counts[name + "_min"] = min(counts[name + "_min"], value) if previous else value
            counts[name + "_max"] = max(counts[name + "_max"], value)


def temporary_root(path):
    measurement = _CURRENT.get()
    if measurement is not None:
        with measurement.lock:
            measurement.roots.add(path)


def temporary_closed(path):
    """닫힌 파일당 O(1)이며, 버퍼 미반영·열린 파일의 최고 사용량은 제외합니다."""
    measurement = _CURRENT.get()
    if measurement is not None:
        with measurement.lock:
            if any(path.is_relative_to(root) for root in measurement.roots):
                size = path.stat().st_size
                measurement.disk_current += size - measurement.closed_sizes.get(path, 0)
                measurement.closed_sizes[path] = size
                measurement.peak_disk = max(measurement.peak_disk, measurement.disk_current)


def temporary_retired(path):
    measurement = _CURRENT.get()
    if measurement is not None:
        with measurement.lock:
            # 파일 추적 종료는 상수 시간이 걸립니다. 루트 추적은 단계마다 한 번 종료합니다.
            if path in measurement.closed_sizes:
                measurement.disk_current -= measurement.closed_sizes.pop(path)
            else:
                for name in list(measurement.closed_sizes):
                    if name.is_relative_to(path):
                        measurement.disk_current -= measurement.closed_sizes.pop(name)
            measurement.roots.discard(path)


def _rss():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
                (name, ctypes.c_size_t)
                for name in (
                    "peak_working",
                    "working",
                    "peak_paged",
                    "paged",
                    "peak_nonpaged",
                    "nonpaged",
                    "pagefile",
                    "peak_pagefile",
                )
            ]

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        get = psapi.GetProcessMemoryInfo
        get.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        get.restype = wintypes.BOOL
        values = Counters()
        values.cb = ctypes.sizeof(values)
        if not get(kernel.GetCurrentProcess(), ctypes.byref(values), values.cb):
            return None
        return int(values.working)
    try:
        with open("/proc/self/statm", encoding="ascii") as stream:
            return int(stream.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, IndexError):
        return None


class Measurement:
    """현재 프로세스의 RSS·Arrow를 표본 측정하고 닫힌 임시 파일은 이벤트로 추적합니다.

    RSS·Arrow·디스크 최고치는 표본에서 본 하한이며 정확한 최댓값을 보장하지 않습니다.
    논리 집계는 인코딩 그룹, 디코딩된 Arrow, 해시·파일 바이트를 구분합니다.
    CPU 시간은 모든 스레드를 포함한 프로세스 CPU이며, 경과 시간은 단조 시계로 잽니다.
    """

    def __init__(self, *, interval=1.0):
        if interval <= 0:
            raise ValueError("Positive benchmark sample interval required")
        self.interval = interval
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.roots = set()
        self.closed_sizes = {}
        self.disk_current = 0
        self.counts = Counter()
        self.milestones = {}
        self.peak_rss = None
        self.peak_arrow = self.peak_disk = 0
        self.sample_errors = 0
        self.report = None

    def __enter__(self):
        self.started, self.cpu_started = time.perf_counter(), time.process_time()
        self.token = _CURRENT.set(self)
        self.sample()
        self.thread = threading.Thread(target=self._sample_loop, daemon=True)
        self.thread.start()
        return self

    def add(self, name, amount):
        with self.lock:
            self.counts[name] += amount

    def mark(self, name):
        with self.lock:
            self.milestones.setdefault(name, time.perf_counter() - self.started)

    def sample(self):
        rss = _rss()
        arrow = pa.total_allocated_bytes()
        with self.lock:
            if rss is not None:
                self.peak_rss = rss if self.peak_rss is None else max(self.peak_rss, rss)
            self.peak_arrow = max(self.peak_arrow, arrow)

    def _sample_loop(self):
        while not self.stop.wait(self.interval):
            try:
                self.sample()
            except OSError:
                self.sample_errors += 1

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join(timeout=5)
        try:
            self.sample()
            self.report = {
                "status": "COMPLETED" if kind is None else "FAILED",
                "wall_seconds": time.perf_counter() - self.started,
                "cpu_seconds": time.process_time() - self.cpu_started,
                "logical_io_and_merge_counters": dict(self.counts),
                "milestone_seconds": dict(self.milestones),
                "sampled_peak_rss_bytes": self.peak_rss,
                "sampled_peak_arrow_allocated_bytes": self.peak_arrow,
                "closed_file_temporary_disk_peak_bytes": self.peak_disk,
                "sampling_interval_seconds": self.interval,
                "sampling_errors": self.sample_errors,
                "sampler_stopped": not self.thread.is_alive(),
                "peak_scope": "SAMPLED_RSS_ARROW_AND_EVENT_TRACKED_CLOSED_TEMP_FILES",
                "temporary_disk_limit": "LOWER_BOUND_EXCLUDES_OPEN_PARTIAL_FILES_AND_METADATA",
                "physical_io_bytes": None,
                "cache_cold_or_warm": "UNCONFIRMED",
            }
        finally:
            _CURRENT.reset(self.token)
