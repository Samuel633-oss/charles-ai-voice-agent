import socket

import net_fix


def test_patch_applies_and_is_idempotent():
    net_fix.patch_gemini_ipv4()
    assert socket.getaddrinfo is net_fix._pinned_getaddrinfo
    net_fix.patch_gemini_ipv4()
    assert socket.getaddrinfo is net_fix._pinned_getaddrinfo


def test_gemini_host_resolves_to_pinned_ipv4_only():
    net_fix.patch_gemini_ipv4()
    infos = socket.getaddrinfo(
        "generativelanguage.googleapis.com", 443, socket.AF_UNSPEC, socket.SOCK_STREAM
    )
    assert infos, "pinned resolution should return at least one address"
    for family, _type, _proto, _canon, sockaddr in infos:
        assert family == socket.AF_INET, "pinned addresses must be IPv4"
        ip = sockaddr[0]
        assert ip in net_fix._GEMINI_IPS, f"unexpected pinned IP {ip}"


def test_stream_request_returns_stream_ipv4_pinned():
    net_fix.patch_gemini_ipv4()
    infos = socket.getaddrinfo(
        "generativelanguage.googleapis.com",
        443,
        socket.AF_UNSPEC,
        socket.SOCK_STREAM,
    )
    assert infos, "pinned resolution should return at least one address"
    for family, _type, _proto, _canon, sockaddr in infos:
        assert family == socket.AF_INET, "pinned addresses must be IPv4"
        assert _type == socket.SOCK_STREAM
        assert sockaddr[0] in net_fix._GEMINI_IPS, f"unexpected pinned IP {sockaddr[0]}"


def test_passthrough_unaffected_host():
    net_fix.patch_gemini_ipv4()
    expected = net_fix._orig_getaddrinfo(
        "127.0.0.1", 443, socket.AF_UNSPEC, socket.SOCK_STREAM
    )
    got = socket.getaddrinfo("127.0.0.1", 443, socket.AF_UNSPEC, socket.SOCK_STREAM)
    assert got == expected


def test_no_patch_means_no_pinning():
    socket.getaddrinfo = net_fix._orig_getaddrinfo
    net_fix.patch_gemini_ipv4._applied = False
    assert socket.getaddrinfo is net_fix._orig_getaddrinfo
