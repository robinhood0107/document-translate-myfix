class LocalServiceError(Exception):
    """Raised when a required local service is unavailable or returns an invalid response."""

    def __init__(
        self,
        message: str,
        *,
        service_name: str = "PaddleOCR VL",
        settings_page_name: str = "PaddleOCR VL Settings",
    ) -> None:
        super().__init__(message)
        self.service_name = service_name
        self.settings_page_name = settings_page_name


class LocalServiceConnectionError(LocalServiceError):
    """Raised when a local service cannot be reached."""
    pass


class LocalServiceSetupError(LocalServiceError):
    """Raised when a local service cannot be prepared or started."""
    pass


class LocalServiceResponseError(LocalServiceError):
    """Raised when a local service returns an unexpected response."""
    pass


class OperationCancelledError(Exception):
    """Raised when a long-running automatic task is cancelled by the user."""

    pass
