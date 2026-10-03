"""Network workarounds for the Gemini Live WebSocket endpoint.

This machine's network (ISP/firewall) drops TLS traffic to the Google IP
range that ``generativelanguage.googleapis.com`` normally resolves to
(``172.217.112-119.x.4``), so the Gemini Realtime handshake stalls until
the ``open_timeout``: ``Gemini Realtime API error: timed out during opening
handshake``.

The same service is served from other Google anycast ranges that this network
can reach, so we pin the hostname to known-good IPv4 addresses at the socket
layer before any connection is made.
"""

from __future__ import annotations

import logging
import socket
from typing import Any

logger = logging.getLogger("net_fix")

# IPs verified (09-2026) to serve generativelanguage.googleapis.com over TLS
# from this network, via GET /v1beta/models. Google anycast frontends, so the
# SNI stays as the real hostname and routing is handled by the frontend.
_GEMINI_HOST = "generativelanguage.googleapis.com"
_GEMINI_IPS = ("74.125.131.101", "142.251.157.119")

_orig_getaddrinfo = socket.getaddrinfo


def _pinned_getaddrinfo(host: str, port: Any, *args: Any, **kwargs: Any) -> list[Any]:
    if isinstance(host, str) and host == _GEMINI_HOST:
        if len(args) >= 1:
            family = args[0]
            rest = args[1:]
        else:
            family = kwargs.get("family", 0)
            rest = ()
        if family == socket.AF_UNSPEC or family in (0, None):
            for ip in _GEMINI_IPS:
                try:
                    infos = _orig_getaddrinfo(ip, port, socket.AF_INET, *rest, **kwargs)
                except OSError:
                    continue
                if infos:
                    return infos
            logger.warning("no pinned IP for %s reachable", _GEMINI_HOST)
    return _orig_getaddrinfo(host, port, *args, **kwargs)


def patch_gemini_ipv4() -> None:
    if getattr(patch_gemini_ipv4, "_applied", False):
        return
    socket.getaddrinfo = _pinned_getaddrinfo  # type: ignore[assignment]
    patch_gemini_ipv4._applied = True
    logger.warning(
        "pinning %s to IPv4 %s (workaround for blocked targeted Google range)",
        _GEMINI_HOST,
        ", ".join(_GEMINI_IPS),
    )
