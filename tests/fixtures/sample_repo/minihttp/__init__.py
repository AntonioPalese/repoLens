"""minihttp: a tiny HTTP client used as a repoLens test fixture."""

from minihttp.auth import BasicAuth
from minihttp.client import Client, RetryPolicy

__all__ = ["BasicAuth", "Client", "RetryPolicy"]
__version__ = "0.0.1"
