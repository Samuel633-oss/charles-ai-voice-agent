"""Browser control for the Charles voice agent.

A :class:`BrowserController` owns a single Chromium page and serializes access
to it, so browser tools can safely share one live page (session cookies,
navigation state) without racing on it. The browser is launched lazily on first
use and must be closed via :meth:`close`.

The browser opens in *headed* (visible) mode so the person running the agent
can watch what it is doing. Set the environment variable
``CHARLES_BROWSER_HEADLESS=1`` (or pass ``headless=True``) to run it invisibly.
"""

from __future__ import annotations

import asyncio
import logging
import os
import urllib.parse
from collections.abc import AsyncIterator, Awaitable
from contextlib import asynccontextmanager
from typing import Any, Callable

from livekit.agents.llm import ToolError
from playwright.async_api import (
    Browser as PlaywrightBrowser,
)
from playwright.async_api import (
    BrowserContext,
    Page,
    async_playwright,
)
from playwright.async_api import (
    Error as PlaywrightError,
)
from playwright.async_api import (
    TimeoutError as PlaywrightTimeoutError,
)

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = ("http", "https")
_DEFAULT_TIMEOUT_MS = 15_000
_MAX_TEXT_CHARS = 2_000
_MAX_FIND_RESULTS = 20
_MAX_LABEL_CHARS = 100


class BrowserController:
    """A single Chromium page with exclusive access per operation."""

    def __init__(
        self,
        *,
        default_timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        headless: bool | None = None,
    ) -> None:
        self._default_timeout_ms = default_timeout_ms
        if headless is None:
            headless = os.environ.get(
                "CHARLES_BROWSER_HEADLESS", "0"
            ).strip().lower() in ("1", "true", "yes")
        self.headless = headless
        self._lock = asyncio.Lock()
        self._playwright = None
        self._browser: PlaywrightBrowser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def _ensure_started(self) -> None:
        if self._page is not None:
            return
        if self._playwright is not None:
            await self.close()
        try:
            playwright = await async_playwright().start()
            browser = await playwright.chromium.launch(headless=self.headless)
            context = await browser.new_context(viewport={"width": 1280, "height": 800})
            page = await context.new_page()
            page.set_default_timeout(self._default_timeout_ms)
            self._playwright = playwright
            self._browser = browser
            self._context = context
            self._page = page
        except BaseException:
            await self.close()
            raise

    async def close(self) -> None:
        """Shut down the browser and release all resources."""
        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                logger.warning("Error stopping Playwright", exc_info=True)
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    async def capture_png(self) -> bytes | None:
        """Return a PNG screenshot of the current page, or None if the browser is not open."""
        async with self._lock:
            if self._page is None:
                return None
            return await self._page.screenshot(type="png")

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[Page]:
        await self._ensure_started()
        async with self._lock:
            assert self._page is not None
            yield self._page

    async def _run(self, action: str, op: Callable[[Page], Awaitable[Any]]) -> Any:
        try:
            async with self._session() as page:
                return await op(page)
        except (PlaywrightTimeoutError, PlaywrightError) as exc:
            raise ToolError(
                f"The browser could not {action}: {exc}. "
                "The page may have changed, the element may not exist, "
                "or the request timed out."
            ) from exc

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in _ALLOWED_SCHEMES or not parsed.netloc:
            raise ToolError(
                f"Only http and https web addresses are supported ('{url}'). "
                "Please provide a complete address, for example https://example.com/news."
            )

    async def open(self, url: str) -> dict[str, str]:
        """Navigate to a URL and return the final URL and page title."""

        self._validate_url(url)

        async def _go(page: Page) -> dict[str, str]:
            await page.goto(url, wait_until="domcontentloaded")
            return {"url": page.url, "title": (await page.title()).strip()}

        return await self._run("open the page", _go)

    async def click(self, selector: str) -> str:
        async def _click(page: Page) -> str:
            await page.locator(selector).first.click()
            return f"Clicked the first element matching '{selector}'."

        return await self._run("click the element", _click)

    async def click_text(self, text: str) -> str:
        async def _click(page: Page) -> str:
            await page.get_by_text(text, exact=True).first.click()
            return f"Clicked the text '{text}'."

        return await self._run("click the element", _click)

    async def fill(self, selector: str, value: str) -> str:
        async def _fill(page: Page) -> str:
            await page.locator(selector).first.fill(value)
            return f"Filled the input matching '{selector}'."

        return await self._run("fill the input", _fill)

    async def clear(self, selector: str) -> str:
        async def _clear(page: Page) -> str:
            await page.locator(selector).first.clear()
            return f"Cleared the input matching '{selector}'."

        return await self._run("clear the input", _clear)

    async def press(self, key: str) -> str:
        async def _press(page: Page) -> str:
            await page.keyboard.press(key)
            return f"Pressed the '{key}' key."

        return await self._run("press the key", _press)

    async def type_text(self, text: str) -> str:
        async def _type(page: Page) -> str:
            await page.keyboard.type(text)
            return "Typed the text into the focused element."

        return await self._run("type text", _type)

    async def read_text(self, selector: str | None = None) -> str:
        async def _read(page: Page) -> str:
            if selector:
                locator = page.locator(selector).first
                await locator.wait_for(state="visible")
                text = await locator.inner_text()
            else:
                text = await page.locator("body").inner_text()
            return text.strip()[:_MAX_TEXT_CHARS]

        return await self._run("read the text", _read)

    async def find(self, selector: str, max_results: int = _MAX_FIND_RESULTS) -> str:
        async def _find(page: Page) -> str:
            locator = page.locator(selector)
            elements = await locator.evaluate_all(
                """(elements, limit) => elements.slice(0, limit).map((el) => ({
                    tag: el.tagName.toLowerCase(),
                    visible: !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length),
                    text: (el.innerText || '').trim().slice(0, 100),
                    name: el.getAttribute('name') || '',
                    id: el.id || '',
                }))""",
                max_results,
            )
            if not elements:
                return f"No elements matched '{selector}'."
            labels = []
            for el in elements:
                kind = el["tag"]
                if el.get("id"):
                    kind += f"#{el['id']}"
                if el.get("name"):
                    kind += f"[name={el['name']}]"
                status = "visible" if el.get("visible") else "hidden"
                text = (el.get("text") or "").strip()
                label = f"{kind} ({status})"
                if text:
                    label += f' "{text[:_MAX_LABEL_CHARS]}"'
                labels.append(label)
            return "\n".join(labels)

        return await self._run("find the elements", _find)

    async def get_url(self) -> str:
        async def _url(page: Page) -> str:
            return page.url

        return await self._run("read the address", _url)

    async def get_title(self) -> str:
        async def _title(page: Page) -> str:
            return (await page.title()).strip() or "no title"

        return await self._run("read the title", _title)

    async def wait_for(self, selector: str, timeout_ms: int | None = None) -> str:
        async def _wait(page: Page) -> str:
            await page.locator(selector).first.wait_for(
                state="visible", timeout=timeout_ms
            )
            return f"The element matching '{selector}' is now visible."

        return await self._run("wait for the element", _wait)

    async def wait(self, delay_ms: int) -> str:
        async def _wait(page: Page) -> str:
            await page.wait_for_timeout(delay_ms)
            return f"Waited {delay_ms} milliseconds."

        return await self._run("wait", _wait)

    async def scroll(self, direction: str) -> str:
        direction = direction.strip().lower()
        if direction not in ("up", "down"):
            raise ToolError(f"Direction must be 'up' or 'down', got '{direction}'.")

        async def _scroll(page: Page) -> str:
            await page.mouse.wheel(0, -1200 if direction == "up" else 1200)
            return f"Scrolled {direction} the page."

        return await self._run("scroll the page", _scroll)

    async def go_back(self) -> str:
        async def _back(page: Page) -> str:
            await page.go_back(wait_until="domcontentloaded")
            return "Went back to the previous page."

        return await self._run("go back", _back)

    async def go_forward(self) -> str:
        async def _forward(page: Page) -> str:
            await page.go_forward(wait_until="domcontentloaded")
            return "Went forward to the next page."

        return await self._run("go forward", _forward)

    async def reload(self) -> str:
        async def _reload(page: Page) -> str:
            await page.reload(wait_until="domcontentloaded")
            return "Reloaded the page."

        return await self._run("reload the page", _reload)

    async def select_option(self, selector: str, label: str) -> str:
        async def _select(page: Page) -> str:
            await page.locator(selector).first.select_option(label=label)
            return f"Selected the option '{label}' from the dropdown matching '{selector}'."

        return await self._run("select the option", _select)

    async def screenshot(self, directory: str, name: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{name}.jpeg")

        async def _shot(page: Page) -> str:
            await page.screenshot(path=path, type="jpeg", quality=60)
            return f"Screenshot saved to {path}."

        return await self._run("take a screenshot", _shot)
