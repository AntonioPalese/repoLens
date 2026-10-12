"""Authentication helpers."""

import base64


class BasicAuth:
    """HTTP Basic authentication."""

    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password

    def header(self) -> str:
        """Value for the ``Authorization`` header."""
        return build_auth_header(self.username, self.password)


def build_auth_header(username: str, password: str) -> str:
    """Encode credentials as an HTTP Basic ``Authorization`` value."""
    token = base64.b64encode(f"{username}:{password}".encode()).decode()
    return f"Basic {token}"


def parseAuthHeader(value: str) -> tuple[str, str]:  # noqa: N802 - camelCase on purpose
    """Inverse of build_auth_header."""
    _, token = value.split(" ", 1)
    username, _, password = base64.b64decode(token).decode().partition(":")
    return username, password
