"""Browser-control tools for the Charles agent.

A :class:`BrowserToolsMixin` exposes a dedicated headless browser through
``@function_tool`` methods. Mix it into the agent class and give the agent
a ``self.browser`` :class:`~browser.BrowserController` instance.
"""

from __future__ import annotations

import os

from livekit.agents import RunContext, function_tool


def _screenshot_dir() -> str:
    return os.environ.get("CHARLES_SCREENSHOT_DIR", "screenshots")


class BrowserToolsMixin:
    """Tools that let the agent control a real web browser."""

    @function_tool
    async def browse_open(self, ctx: RunContext, url: str) -> str:
        """Open a web page in the browser and return the title and final address of the page. Only http and https addresses are allowed.

        Args:
            url: The complete web address to open, for example https://example.com/news.
        """
        await ctx.update(f"Opening {url} in the browser.")
        result = await self.browser.open(url)
        return f"Opened the page titled '{result['title']}' at {result['url']}."

    @function_tool
    async def browse_click(self, ctx: RunContext, selector: str) -> str:
        """Click the first element matching the given CSS selector. Use browse_find first if you do not know which selector fits the element the user mentioned.

        Args:
            selector: A CSS selector identifying the element to click, for example '#add-to-cart' or 'button.submit'.
        """
        return await self.browser.click(selector)

    @function_tool
    async def browse_click_text(self, ctx: RunContext, text: str) -> str:
        """Click the first visible element whose exact text equals the given string, for example a button labelled 'Continue' or a link such as 'Sign in'.

        Args:
            text: The exact text of the element to click.
        """
        return await self.browser.click_text(text)

    @function_tool
    async def browse_fill(self, ctx: RunContext, selector: str, value: str) -> str:
        """Type the given value into the input field matching the CSS selector, replacing any current content (for example a search box or a checkout field).

        Args:
            selector: A CSS selector identifying the input field, for example '#query'.
            value: The text to type into the field.
        """
        return await self.browser.fill(selector, value)

    @function_tool
    async def browse_clear(self, ctx: RunContext, selector: str) -> str:
        """Clear the input field matching the given CSS selector.

        Args:
            selector: A CSS selector identifying the input field to clear.
        """
        return await self.browser.clear(selector)

    @function_tool
    async def browse_press_key(self, ctx: RunContext, key: str) -> str:
        """Press a single keyboard key on the focused element. Use for actions such as submitting a form (Enter) or dismissing a dialog (Escape).

        Args:
            key: The key to press, for example 'Enter', 'Tab', 'Escape', 'Backspace', or 'F5'.
        """
        return await self.browser.press(key)

    @function_tool
    async def browse_type_text(self, ctx: RunContext, text: str) -> str:
        """Type text, character by character, into the currently focused element as if typing on a keyboard.

        Args:
            text: The text to type.
        """
        return await self.browser.type_text(text)

    @function_tool
    async def browse_read_page(
        self, ctx: RunContext, selector: str | None = None
    ) -> str:
        """Read the visible text of the whole page (or of the first element matching the selector) so you can understand its content. Use it after opening a page.

        Args:
            selector: Optional CSS selector; when given, only that element's text is returned.
        """
        return await self.browser.read_text(selector)

    @function_tool
    async def browse_find(self, ctx: RunContext, selector: str) -> str:
        """List the elements matching a CSS selector, including tag, id, visibility, and a short text snippet, so you can choose a reliable selector to act on. The list is limited to the first matches.

        Args:
            selector: A CSS selector to search for, for example 'input', 'button', or 'a'.
        """
        return await self.browser.find(selector)

    @function_tool
    async def browse_get_url(self, ctx: RunContext) -> str:
        """Return the web address currently open in the browser."""
        return await self.browser.get_url()

    @function_tool
    async def browse_get_title(self, ctx: RunContext) -> str:
        """Return the title of the page currently open in the browser."""
        return await self.browser.get_title()

    @function_tool
    async def browse_wait_for(self, ctx: RunContext, selector: str) -> str:
        """Wait until an element matching the given CSS selector becomes visible on the page, up to the navigation timeout. Use it after an action that loads content slowly.

        Args:
            selector: A CSS selector identifying the element to wait for.
        """
        return await self.browser.wait_for(selector)

    @function_tool
    async def browse_wait(self, ctx: RunContext, delay_ms: int) -> str:
        """Pause browser work for the given number of milliseconds. Use sparingly, for example to let a page settle.

        Args:
            delay_ms: How long to pause, in milliseconds (1000 equals one second).
        """
        return await self.browser.wait(delay_ms)

    @function_tool
    async def browse_scroll(self, ctx: RunContext, direction: str) -> str:
        """Scroll the web page down or up.

        Args:
            direction: Either 'down' or 'up'.
        """
        return await self.browser.scroll(direction)

    @function_tool
    async def browse_back(self, ctx: RunContext) -> str:
        """Go back to the previously visited page in browser history."""
        return await self.browser.go_back()

    @function_tool
    async def browse_forward(self, ctx: RunContext) -> str:
        """Go forward to the next page in browser history."""
        return await self.browser.go_forward()

    @function_tool
    async def browse_reload(self, ctx: RunContext) -> str:
        """Reload the page currently open in the browser."""
        return await self.browser.reload()

    @function_tool
    async def browse_select_option(
        self, ctx: RunContext, selector: str, label: str
    ) -> str:
        """Choose an option from a dropdown menu (a select element) by its visible label.

        Args:
            selector: A CSS selector identifying the dropdown element.
            label: The visible label of the option to choose.
        """
        return await self.browser.select_option(selector, label)

    @function_tool
    async def browse_screenshot(self, ctx: RunContext, name: str) -> str:
        """Take a JPEG screenshot of the current page and save it to a local folder, returning the file path. Use it to capture what is on screen.

        Args:
            name: A short lowercase name for the screenshot file, for example 'checkout-review'.
        """
        return await self.browser.screenshot(_screenshot_dir(), name)
