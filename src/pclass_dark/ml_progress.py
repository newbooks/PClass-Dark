"""Scoped SIAC terminal logging and throttled counters; no import-time setup."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager

FORMAT = "%(asctime)s [%(processName)s] %(message)s"


def _handler():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(FORMAT, datefmt="%H:%M:%S"))
    return handler


@contextmanager
def terminal_progress():
    """Temporarily add CLI stderr progress without altering a caller's handlers."""
    logger = logging.getLogger("pclass_dark")
    previous = logger.level
    handler = _handler()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()
        logger.setLevel(previous)


def worker_progress(enabled):
    """Spawn initializer: workers need their own handler; parent handlers don't copy."""
    if enabled:
        logger = logging.getLogger("pclass_dark")
        logger.addHandler(_handler())
        logger.setLevel(logging.INFO)


class Progress:
    """Report increasing counts at bounded frequency and at explicit completion."""

    def __init__(self, logger, label, *, total=None, unit="rows", announce=True):
        self.logger, self.label, self.total, self.unit = logger, label, total, unit
        self.started = self.last = time.monotonic()
        if announce:
            self.logger.info("%s: starting", self.label)

    def update(self, count):
        now = time.monotonic()
        if now - self.last >= 5:
            detail = f"{count:,} {self.unit}"
            if self.total:
                detail += f" / {self.total:,} ({100 * count / self.total:.1f}%)"
            self.logger.info(
                "%s: %s; %.1fs elapsed", self.label, detail, now - self.started
            )
            self.last = now

    def finish(self, count):
        self.logger.info(
            "%s: completed %s %s in %.1fs",
            self.label,
            f"{count:,}",
            self.unit,
            time.monotonic() - self.started,
        )
