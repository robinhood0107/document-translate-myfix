class LocalServiceError(Exception):
    """Raised when a required local service is unavailable or returns an invalid response."""
    pass


class LocalServiceConnectionError(LocalServiceError):
    """Raised when a local service cannot be reached."""
    pass


class LocalServiceResponseError(LocalServiceError):
    """Raised when a local service returns an unexpected response."""
    pass
