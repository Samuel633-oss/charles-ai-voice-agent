import asyncio
import io

from PIL import Image

import preview


def make_png(size, color):
    img = Image.new("RGBA", size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_png_to_rgba_fixed_dims():
    png = make_png((40, 30), (255, 0, 0, 255))
    width, height, data = preview.png_to_rgba(png)
    assert (width, height) == (preview._PREVIEW_WIDTH, preview._PREVIEW_HEIGHT)
    assert len(data) == width * height * 4
    assert data[0:4] == bytes((255, 0, 0, 255))


def test_png_to_rgba_center_crops_wide_content():
    png = make_png((2000, 1000), (0, 0, 255, 255))
    width, height, data = preview.png_to_rgba(png)
    assert (width, height) == (preview._PREVIEW_WIDTH, preview._PREVIEW_HEIGHT)
    assert len(data) == width * height * 4


def test_png_to_rgba_custom_dims():
    png = make_png((320, 240), (0, 255, 0, 255))
    width, height, _ = preview.png_to_rgba(png, width=128, height=64)
    assert (width, height) == (128, 64)


class _NullController:
    async def capture_png(self) -> None:
        return None


async def test_preview_loop_tolerates_no_page():
    p = preview.BrowserPreview(_NullController())
    assert p.published is False
    await p.start()
    await asyncio.sleep(0.3)
    await p.stop()
    assert p.published is False


async def test_preview_start_stop_is_idempotent():
    p = preview.BrowserPreview(_NullController())
    await p.start()
    await p.start()
    await p.stop()
    await p.stop()
