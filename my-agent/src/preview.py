"""Live browser preview from the agent into the call.

A :class:`BrowserPreview` publishes a low-rate video track (sourced as a
screen-share) into the room so the person talking to Charles can see what the
agent is browsing, even when the agent runs somewhere with no display (LiveKit
Cloud). It ticks screenshots of the browser page and pushes them as frames.
"""

from __future__ import annotations

import asyncio
import contextlib
import io
import logging
from typing import Any

import livekit.rtc as rtc
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

_PREVIEW_TRACK_NAME = "browser-preview"
_PREVIEW_INTERVAL_S = 0.7
_PREVIEW_WIDTH = 853
_PREVIEW_HEIGHT = 480


def png_to_rgba(
    png: bytes, width: int = _PREVIEW_WIDTH, height: int = _PREVIEW_HEIGHT
) -> tuple[int, int, bytes]:
    """Decode a PNG to (width, height, RGBA bytes), center-cropped to fixed dims."""
    with Image.open(io.BytesIO(png)) as img:
        img = ImageOps.fit(img, (width, height), Image.Resampling.LANCZOS).convert(
            "RGBA"
        )
        return width, height, img.tobytes()


class BrowserPreview:
    """Publish the agent's browser as a video track and stream frames into it."""

    def __init__(self, controller: Any) -> None:
        self._controller = controller
        self._room: rtc.Room | None = None
        self._source: rtc.VideoSource | None = None
        self._track: rtc.LocalVideoTrack | None = None
        self._publication: rtc.LocalTrackPublication | None = None
        self._task: asyncio.Task[Any] | None = None

    @property
    def published(self) -> bool:
        return self._publication is not None

    def _ensure_track(self) -> rtc.LocalVideoTrack:
        if self._track is None:
            self._source = rtc.VideoSource(
                _PREVIEW_WIDTH, _PREVIEW_HEIGHT, is_screencast=True
            )
            self._track = rtc.LocalVideoTrack.create_video_track(
                _PREVIEW_TRACK_NAME, self._source
            )
        return self._track

    async def publish(self, room: rtc.Room) -> None:
        self._room = room
        track = self._ensure_track()
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_SCREENSHARE
        options.video_codec = rtc.VideoCodec.VP8
        self._publication = await room.local_participant.publish_track(track, options)
        logger.info("published browser preview track %r", _PREVIEW_TRACK_NAME)

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        if self._source is not None:
            await self._source.aclose()
            self._source = None
        if self._track is not None:
            self._publication = None
            self._track = None

    async def _loop(self) -> None:
        while True:
            try:
                png = await self._controller.capture_png()
                if png is not None and self._source is not None:
                    width, height, data = png_to_rgba(png)
                    frame = rtc.VideoFrame(
                        width, height, int(rtc.VideoBufferType.RGBA), data
                    )
                    await self._source.capture_frame(frame)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("browser preview frame error")
            await asyncio.sleep(_PREVIEW_INTERVAL_S)
