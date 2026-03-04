from typing import Optional


class SDMergerError(Exception):
    """Base exception class for all custom SD-merger exceptions."""

    def __init__(self, message: str, original_error: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.original_error = original_error

    def __str__(self):
        base_msg = self.message
        if self.original_error:
            base_msg += f" (Cause: {self.original_error})"
        return base_msg


class ConfigError(SDMergerError):
    """Raised when there is an issue with the configuration (YAML validation, etc)."""

    pass


class ModelLoadError(SDMergerError):
    """Raised when a model fails to load."""

    pass


class MergeError(SDMergerError):
    """Raised when the merging process encounters an error."""

    pass


class ExtensionError(SDMergerError):
    """Raised when an extension (hook) fails."""

    pass


class GenerationError(SDMergerError):
    """Raised when image generation fails."""

    pass
