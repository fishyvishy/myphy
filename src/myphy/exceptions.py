class NotFittedError(Exception):
    """Raised when a method requiring a fit is called before fitting the model."""

    pass
