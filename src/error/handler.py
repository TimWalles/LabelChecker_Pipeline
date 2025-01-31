class ErrorTracker:
    name: str
    error: str


class Tracker:
    successful: int = 0
    failed: list[ErrorTracker] = []


def error_handler(
    tracker: Tracker,
    name: str,
    desc: str,
) -> Tracker:
    error_message = ErrorTracker()
    error_message.name = name
    error_message.error = desc
    tracker.failed.append(error_message)
    return tracker
