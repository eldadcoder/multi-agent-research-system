"""Errors tools raise for expected failures. Agents catch these and keep going."""


class ToolError(RuntimeError):
    """A tool failed in an expected way: network, bad page, rate limit, empty content."""