"""In-process tests for BrowserController against a real local HTTP server.

These exercise the actual headless Chromium browser (no mocks). They are
independent of any LiveKit session, so they run with a plain ``uv run pytest``.
"""

from __future__ import annotations

import asyncio
import http.server
import threading

import pytest
from livekit.agents.llm import ToolError

from browser import BrowserController

_INDEX_HTML = """<!doctype html>
<html>
<head><title>Test Shop</title></head>
<body>
  <h1>Welcome to the Test Shop</h1>
  <p id="intro">The finest hammers and widgets.</p>
  <input id="query" placeholder="Search">
  <button id="search-btn">Search</button>
  <a id="catalog-link" href="/catalog">Catalog</a>
  <select id="sort">
    <option value="a">Price low to high</option>
    <option value="b">Price high to low</option>
  </select>
  <div id="result"></div>
  <button id="reveal">Reveal Secret</button>
  <div id="secret" style="display:none">The secret is out.</div>
  <script>
    document.getElementById('query').addEventListener('input', (e) => {
      document.getElementById('result').textContent = 'You typed: ' + e.target.value;
    });
    document.getElementById('search-btn').addEventListener('click', () => {
      document.getElementById('result').textContent =
        'Searching for ' + document.getElementById('query').value + '.';
    });
    document.getElementById('reveal').addEventListener('click', () => {
      document.getElementById('secret').style.display = 'block';
    });
  </script>
</body>
</html>
"""

_CATALOG_HTML = """<!doctype html>
<html>
<head><title>Catalog</title></head>
<body><h1>All our wares</h1></body>
</html>
"""


class _Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = _CATALOG_HTML if self.path == "/catalog" else _INDEX_HTML
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def http_server():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_address[1]}"
    yield url
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


@pytest.fixture
async def browser():
    controller = BrowserController(default_timeout_ms=5000, headless=True)
    yield controller
    await controller.close()


def test_default_browser_is_visible_headed():
    controller = BrowserController()
    assert controller.headless is False


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("true", True), ("0", False), ("", False)]
)
def test_headless_env_override(monkeypatch, value, expected):
    if value == "":
        monkeypatch.delenv("CHARLES_BROWSER_HEADLESS", raising=False)
    else:
        monkeypatch.setenv("CHARLES_BROWSER_HEADLESS", value)
    controller = BrowserController()
    assert controller.headless is expected


def test_headless_param_wins():
    controller = BrowserController(headless=True)
    assert controller.headless is True


async def test_open_returns_title_and_url(browser, http_server):
    result = await browser.open(f"{http_server}/")
    assert result["title"] == "Test Shop"
    assert result["url"].rstrip("/") == http_server


async def test_capture_png_returns_none_before_start():
    controller = BrowserController(headless=True)
    assert await controller.capture_png() is None
    await controller.close()


async def test_capture_png_returns_png_bytes_after_open(browser, http_server):
    await browser.open(f"{http_server}/")
    png = await browser.capture_png()
    assert png is not None
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


async def test_fill_then_read_input_feedback(browser, http_server):
    await browser.open(f"{http_server}/")
    await browser.fill("#query", "hammer")
    text = await browser.read_text("#result")
    assert "hammer" in text


async def test_click_updates_page_state(browser, http_server):
    await browser.open(f"{http_server}/")
    await browser.fill("#query", "widget")
    await browser.click("#search-btn")
    text = await browser.read_text("#result")
    assert "Searching for widget" in text


async def test_click_text_reveals_hidden_element(browser, http_server):
    await browser.open(f"{http_server}/")
    await browser.click_text("Reveal Secret")
    text = await browser.read_text("#secret")
    assert "The secret is out" in text


async def test_find_lists_elements(browser, http_server):
    await browser.open(f"{http_server}/")
    found = await browser.find("button")
    assert "search-btn" in found
    assert "Reveal Secret" in found


async def test_navigate_via_link_and_go_back(browser, http_server):
    await browser.open(f"{http_server}/")
    await browser.click("#catalog-link")
    assert await browser.get_title() == "Catalog"
    await browser.go_back()
    assert await browser.get_title() == "Test Shop"


async def test_select_option(browser, http_server):
    await browser.open(f"{http_server}/")
    message = await browser.select_option("#sort", "Price high to low")
    assert "Price high to low" in message


async def test_scroll_rejects_bad_direction(browser, http_server):
    await browser.open(f"{http_server}/")
    assert "down" in await browser.scroll("down")
    assert "up" in await browser.scroll("up")
    with pytest.raises(ToolError):
        await browser.scroll("sideways")


async def test_rejects_non_http_protocol(browser):
    with pytest.raises(ToolError):
        await browser.open("ftp://example.com/file")


async def test_rejects_garbage_url(browser):
    with pytest.raises(ToolError):
        await browser.open("not a web address")


async def test_unknown_host_raises_tool_error(browser):
    with pytest.raises(ToolError):
        await browser.open("http://nonexistent-host.invalid/")


async def test_wait_for_returns_when_present(browser, http_server):
    await browser.open(f"{http_server}/")
    await browser.click("#reveal")
    message = await browser.wait_for("#secret", timeout_ms=3000)
    assert "now visible" in message


async def test_wait_for_missing_element_times_out(browser, http_server):
    await browser.open(f"{http_server}/")
    with pytest.raises(ToolError):
        await browser.wait_for("#never-here", timeout_ms=1500)


async def test_screenshot_writes_file(browser, http_server, tmp_path):
    await browser.open(f"{http_server}/")
    message = await browser.screenshot(str(tmp_path), "home")
    assert (tmp_path / "home.jpeg").exists()
    assert "home.jpeg" in message


async def test_concurrent_operations_are_serialized(browser, http_server):
    await browser.open(f"{http_server}/")
    titles = await asyncio.gather(
        browser.get_title(),
        browser.get_title(),
        browser.get_url(),
        browser.get_url(),
    )
    assert titles[0] == titles[1] == "Test Shop"
    assert titles[2] == titles[3]
