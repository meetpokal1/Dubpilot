"""Small helpers shared across the pipeline: progress logging and timing."""

import time


def log(step, message):
    """Print a timestamped progress line to the terminal."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] [{step}] {message}")


def format_seconds(seconds):
    """Turn a float number of seconds into mm:ss for progress messages."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


class Timer:
    """Context manager that logs how long a pipeline stage took.

    Used with 'with Timer("transcribe"):' around each stage in main.py so
    the terminal output doubles as a timing report for the submission.
    """

    def __init__(self, label):
        self.label = label
        self.start = None

    def __enter__(self):
        self.start = time.time()
        log(self.label, "started")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        log(self.label, f"finished in {format_seconds(elapsed)}")
