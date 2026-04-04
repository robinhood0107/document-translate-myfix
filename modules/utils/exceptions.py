class InsufficientCreditsException(Exception):
    """Raised when the user does not have enough credits for an operation."""
    pass


class ContentFlaggedException(Exception):
    """Raised when the content is blocked by safety filters."""
    def __init__(self, message, context="Operation"):
        super().__init__(message)
        self.context = context


class LocalServiceError(Exception):
    """Raised when a required local service is unavailable or returns an invalid response."""
    pass


class LocalServiceConnectionError(LocalServiceError):
    """Raised when a local service cannot be reached."""
    pass


class LocalServiceResponseError(LocalServiceError):
    """Raised when a local service returns an unexpected response."""
    pass
