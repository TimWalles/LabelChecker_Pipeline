from enum import Enum
from typing import Optional

from rich.console import Console

console = Console()


class LogLevel(Enum):
    ERROR = "red"
    INFO = "blue"
    PROGRESS = "green"
    WARNING = "yellow"


def log(message: str, level: LogLevel, verbose: bool = False, prefix: Optional[str] = None):
    """
    Generic logging function with rich formatting.

    Args:
        message: The message to log
        level: LogLevel enum indicating the type/color of the message
        verbose: Whether to show the message
        prefix: Optional prefix to show before the message
    """
    if not verbose:
        return

    prefix = prefix or level.name
    console.print(f"\n[bold {level.value}][{prefix}][/bold {level.value}]: {message}")


# Convenience functions
def log_progress(message: str, verbose: bool = True, prefix: Optional[str] = "Progress"):
    log(message, LogLevel.PROGRESS, verbose, prefix)


def log_info(message: str, verbose: bool = False, prefix: Optional[str] = "Info"):
    log(message, LogLevel.INFO, verbose, prefix)


def log_error(message: str, verbose: bool = True, prefix: Optional[str] = "Error"):
    log(message, LogLevel.ERROR, verbose, prefix)


def log_warning(message: str, verbose: bool = True, prefix: Optional[str] = "Warning"):
    log(message, LogLevel.WARNING, verbose, prefix)
