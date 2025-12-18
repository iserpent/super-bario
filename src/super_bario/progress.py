# -*- coding: utf-8 -*-
"""
Super Bario – A progress bar and layout system for Python.
Copyright (c) 2025 Igor Iatsenko
Licensed under the MIT License.
"""

import os
import sys
import signal
import shutil
import time
import itertools
import threading
import weakref
import select
import fcntl
import math
from weakref import ReferenceType, WeakSet, WeakKeyDictionary
from collections import defaultdict
from collections.abc import MutableSequence
from contextlib import contextmanager
from datetime import datetime
from dataclasses import dataclass
from queue import Queue
from typing import (
    TYPE_CHECKING,
    Protocol,
    Optional,
    Tuple,
    Set,
    List,
    Dict,
    Callable,
    Any,
    Iterator,
    Sized,
    Union,
    TextIO,
)
from abc import ABC, abstractmethod
from enum import Enum
import logging

__all__ = [
    "progress",
    "Progress",
    "ProgressContext",
    "Bar",
    "BarItem",
    "View",
    "Theme",
    "Colors",
    "Widget",
    "TitleWidget",
    "BarWidget",
    "PercentageWidget",
    "CounterWidget",
    "SpinnerWidget",
    "RateWidget",
    "ElapsedTimeWidget",
    "EstimatedTimeWidget",
]

logger = logging.getLogger("super-bario")

_sigwinch_pending = False
_terminal_width = 0
_terminal_height = 0
_sigwinch_read_fd = None
_sigwinch_write_fd = None


# ============================================================================
# Terminal utilities
# ============================================================================


class TerminalCapability(Enum):
    """Terminal capability levels"""

    MINIMAL = 1  # No ANSI support
    BASIC = 2  # Basic ANSI colors
    ADVANCED = 3  # Full Unicode and colors


class Colors:
    """ANSI color codes and utilities"""

    # Reset
    RESET = "\033[0m"

    # Basic colors (3/4 bit)
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright colors
    BRIGHT_BLACK = "\033[90m"
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"

    # Background colors
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # Styles
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    @staticmethod
    def rgb(r: int, g: int, b: int) -> str:
        """Create 24-bit RGB color"""
        return f"\033[38;2;{r};{g};{b}m"

    @staticmethod
    def bg_rgb(r: int, g: int, b: int) -> str:
        """Create 24-bit RGB background color"""
        return f"\033[48;2;{r};{g};{b}m"

    @staticmethod
    def gradient(
        progress: float,
        start_color: Tuple[int, int, int],
        end_color: Tuple[int, int, int],
    ) -> str:
        """Generate gradient color based on progress (0.0 to 1.0)"""
        r = int(start_color[0] + (end_color[0] - start_color[0]) * progress)
        g = int(start_color[1] + (end_color[1] - start_color[1]) * progress)
        b = int(start_color[2] + (end_color[2] - start_color[2]) * progress)
        return Colors.rgb(r, g, b)


class Theme:
    """Color theme for progress bars"""

    def __init__(
        self,
        title_color: str = Colors.CYAN,
        bar_complete_color: str = Colors.GREEN,
        bar_incomplete_color: str = Colors.BRIGHT_BLACK,
        percentage_color: str = Colors.BRIGHT_WHITE,
        time_color: str = Colors.YELLOW,
        counter_color: str = Colors.BLUE,
        use_gradient: bool = False,
        gradient_start: Tuple[int, int, int] = (255, 0, 0),  # Red
        gradient_end: Tuple[int, int, int] = (0, 255, 0),
    ):  # Green
        self.title_color = title_color
        self.bar_complete_color = bar_complete_color
        self.bar_incomplete_color = bar_incomplete_color
        self.percentage_color = percentage_color
        self.time_color = time_color
        self.counter_color = counter_color
        self.use_gradient = use_gradient
        self.gradient_start = gradient_start
        self.gradient_end = gradient_end

    @staticmethod
    def default():
        """Default color theme"""
        return Theme()

    @staticmethod
    def minimal():
        """Theme for minimal terminals (no colors)"""
        return Theme(
            title_color="",
            bar_complete_color="",
            bar_incomplete_color="",
            percentage_color="",
            time_color="",
            counter_color="",
            use_gradient=False,
        )

    @staticmethod
    def matrix():
        """Matrix green theme"""
        return Theme(
            title_color=Colors.BRIGHT_GREEN,
            bar_complete_color=Colors.GREEN,
            percentage_color=Colors.BRIGHT_GREEN,
            time_color=Colors.GREEN,
            counter_color=Colors.BRIGHT_GREEN,
        )

    @staticmethod
    def fire():
        """Fire/heat theme with gradient"""
        return Theme(
            title_color=Colors.BRIGHT_YELLOW,
            bar_complete_color=Colors.RED,
            percentage_color=Colors.BRIGHT_RED,
            time_color=Colors.YELLOW,
            counter_color=Colors.BRIGHT_YELLOW,
            use_gradient=True,
            gradient_start=(255, 100, 0),  # Orange
            gradient_end=(255, 50, 50),  # Red
        )

    @staticmethod
    def load():
        """Load theme with gradient"""
        return Theme(
            use_gradient=True,
            gradient_start=(50, 255, 50),  # Green
            gradient_end=(255, 50, 50),  # Red
        )


# ============================================================================
# Widget System
# ============================================================================


class Widget(ABC):
    """Base class for progress bar widgets"""

    _render_priority: int = 100

    @property
    def render_priority(self) -> int:
        """Render priority for widget ordering (lower is rendered first)"""
        return self._render_priority

    def reset(
        self,
        *args,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
        **kwargs,
    ):
        """Reset any internal state"""
        if theme is not None:
            self.theme = theme
        elif not hasattr(self, "theme"):
            self.theme = Theme.default()

        # Auto-detect unicode support if not specified
        if use_unicode is None:
            capability = _detect_terminal_capability()
            use_unicode = capability in [
                TerminalCapability.BASIC,
                TerminalCapability.ADVANCED,
            ]

        self.use_unicode = use_unicode

    @abstractmethod
    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        """Render the widget to a raw and styled string"""
        pass

    def _trim(self, text: str, width: int) -> str:
        """Trim text to fit within the specified width"""
        if width <= 0:
            return ""

        if len(text) <= width:
            return text

        if self.use_unicode:
            text = text[: max(0, width - 1)]
            text += "…"
        else:
            text = text[: max(0, width - 3)]
            text += "." * (width - len(text))

        return text


class TitleWidget(Widget):
    """Widget displaying the title/description"""

    _render_priority = 40

    def __init__(
        self,
        title: Union[str, Callable[[Any], str]] = "Progress",
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        self.title_fn: Optional[Callable[[Any], str]] = None
        self._width = 0
        self.reset(title=title, theme=theme, use_unicode=use_unicode)

    def reset(
        self,
        title: Union[str, Callable[[Any], str], None] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        super().reset(theme=theme, use_unicode=use_unicode)
        if title is not None:
            if callable(title):
                self.title_fn = title
                self._title = ""
                self._width = 0
            else:
                self._title = title
                self._width = len(str(title))

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        prepared = self._title

        if bar.item is not None and self.title_fn:
            prepared = self.title_fn(bar.item)

        prepared = self._trim(prepared, width)
        rendered = f"{self.theme.title_color}{prepared}{Colors.RESET}"

        return (prepared, rendered)


class BarWidget(Widget):
    """Widget displaying the actual progress bar"""

    _render_priority = 50

    def __init__(
        self,
        use_unicode: Optional[bool] = None,
        theme: Optional[Theme] = None,
        char_start_incomplete: Optional[str] = None,
        char_start_progress: Optional[str] = None,
        char_start_complete: Optional[str] = None,
        char_end_incomplete: Optional[str] = None,
        char_end_progress: Optional[str] = None,
        char_end_complete: Optional[str] = None,
        char_incomplete: Optional[str] = None,
        char_complete: Optional[str] = None,
        char_complete_fractions: Optional[List[str]] = None,
    ):
        self.reset(
            use_unicode=use_unicode,
            theme=theme,
            char_start_incomplete=char_start_incomplete,
            char_start_progress=char_start_progress,
            char_start_complete=char_start_complete,
            char_end_incomplete=char_end_incomplete,
            char_end_progress=char_end_progress,
            char_end_complete=char_end_complete,
            char_incomplete=char_incomplete,
            char_complete=char_complete,
            char_complete_fractions=char_complete_fractions,
        )

    def reset(
        self,
        use_unicode: Optional[bool] = None,
        theme: Optional[Theme] = None,
        char_start_incomplete: Optional[str] = None,
        char_start_progress: Optional[str] = None,
        char_start_complete: Optional[str] = None,
        char_end_incomplete: Optional[str] = None,
        char_end_progress: Optional[str] = None,
        char_end_complete: Optional[str] = None,
        char_incomplete: Optional[str] = None,
        char_complete: Optional[str] = None,
        char_complete_fractions: Optional[List[str]] = None,
    ):
        super().reset(theme=theme, use_unicode=use_unicode)

        if self.use_unicode:
            self.char_start_incomplete = "▕"
            self.char_end_incomplete = "▏"
            self.char_incomplete = " "
            self.char_complete = "█"
            self.char_complete_fractions = ["▏", "▎", "▍", "▌", "▋", "▊", "▉", "█"]
        else:
            self.char_start_incomplete = "["
            self.char_end_incomplete = "]"
            self.char_incomplete = " "
            self.char_complete = "#"
            self.char_complete_fractions = ["#"]

        if char_start_incomplete is not None:
            self.char_start_incomplete = char_start_incomplete

        if char_end_incomplete is not None:
            self.char_end_incomplete = char_end_incomplete

        self.char_start_progress = self.char_start_incomplete
        self.char_end_progress = self.char_end_incomplete
        self.char_start_complete = self.char_start_incomplete
        self.char_end_complete = self.char_end_incomplete

        if char_start_progress is not None:
            self.char_start_progress = char_start_progress

        if char_end_progress is not None:
            self.char_end_progress = char_end_progress

        if char_start_complete is not None:
            self.char_start_complete = char_start_complete

        if char_end_complete is not None:
            self.char_end_complete = char_end_complete

        if char_incomplete is not None:
            self.char_incomplete = char_incomplete

        if char_complete is not None:
            self.char_complete = char_complete
            self.char_complete_fractions = []

        if char_complete_fractions is not None:
            self.char_complete_fractions = char_complete_fractions

        if len(self.char_complete_fractions) > 1:
            # Don't show fraction for 0 progress if multiple fractions are defined
            # But if only one fraction is defined, use it for 0 progress as well
            # This allows for single-character progress bars
            self.char_complete_fractions.insert(0, " ")

    def _recalculate_trimmed_parts(
        self,
        width: int,
        filled_width: int,
        trimmed_content: str,
        complete_part: str,
        incomplete_part: str,
    ):
        trimmed_width = (
            len(trimmed_content)
            - len(self.char_start_incomplete)
            - len(self.char_end_incomplete)
        )
        if trimmed_width < width:
            # Adjust parts to fit trimmed width
            if filled_width > trimmed_width:
                complete_part = self.char_complete * trimmed_width
                incomplete_part = ""
            else:
                complete_part = self.char_complete * filled_width
                remaining_width = trimmed_width - filled_width
                incomplete_part = self.char_incomplete * remaining_width
        return complete_part, incomplete_part

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        progress_ratio = min(1.0, bar.progress)

        inner_width_with_chars_progress = (
            width - len(self.char_start_progress) - len(self.char_end_progress)
        )
        filled_width_with_chars_progress = int(
            inner_width_with_chars_progress * progress_ratio
        )

        if progress_ratio >= 1.0:
            char_start = self.char_start_complete
            char_end = self.char_end_complete
        elif filled_width_with_chars_progress > 0:
            char_start = self.char_start_progress
            char_end = self.char_end_progress
        else:
            char_start = self.char_start_incomplete
            char_end = self.char_end_incomplete

        inner_width = width - len(char_start) - len(char_end)

        if bar.total == 0:
            # Indeterminate progress
            content = char_start + "-" * inner_width + char_end
            content = self._trim(content, width)
            return (
                content,
                f"{self.theme.bar_incomplete_color}{content}{Colors.RESET}",
            )

        # Get color for the bar
        if self.theme.use_gradient:
            bar_color = Colors.gradient(
                progress_ratio, self.theme.gradient_start, self.theme.gradient_end
            )
        else:
            bar_color = self.theme.bar_complete_color

        if self.use_unicode:
            # Smooth progress with partial blocks
            filled_blocks = progress_ratio * inner_width
            full_blocks = int(filled_blocks)
            partial_block_index = math.ceil(
                (filled_blocks - full_blocks) * (len(self.char_complete_fractions) - 1)
            )

            has_partial = full_blocks < inner_width
            partial_char = (
                self.char_complete_fractions[partial_block_index]
                if has_partial and self.char_complete_fractions
                else ""
            )
            incomplete_count = inner_width - full_blocks - len(partial_char)

            content = (
                char_start
                + self.char_complete * full_blocks
                + partial_char
                + self.char_incomplete * incomplete_count
                + char_end
            )
            content = self._trim(content, width)

            return (content, f"{bar_color}{content}{Colors.RESET}")

        else:
            # Classic style with complete/incomplete characters
            filled_width = int(inner_width * progress_ratio)

            if progress_ratio >= 1.0:
                content = char_start + self.char_complete * inner_width + char_end
                content = self._trim(content, width)
                return (content, f"{bar_color}{content}{Colors.RESET}")
            else:
                complete_part = self.char_complete * filled_width
                incomplete_part = self.char_incomplete * (inner_width - filled_width)
                content = char_start + complete_part + incomplete_part + char_end
                trimmed_content = self._trim(content, width)

                if not trimmed_content:
                    return ("", "")

                # Recalculate parts after trimming
                complete_part, incomplete_part = self._recalculate_trimmed_parts(
                    inner_width,
                    filled_width,
                    trimmed_content,
                    complete_part,
                    incomplete_part,
                )

                rendered_complete_part = f"{bar_color}{complete_part}{Colors.RESET}"
                rendered_incomplete_part = (
                    f"{self.theme.bar_incomplete_color}{incomplete_part}{Colors.RESET}"
                )
                start_bracket = char_start if len(trimmed_content) > 1 else ""
                end_bracket = char_end if len(trimmed_content) > 1 else ""
                rendered_content = f"{start_bracket}{rendered_complete_part}{rendered_incomplete_part}{end_bracket}"

                return (content, rendered_content)


class PercentageWidget(Widget):
    """Widget displaying percentage"""

    _render_priority = 10

    def __init__(
        self, theme: Optional[Theme] = None, use_unicode: Optional[bool] = None
    ):
        self.reset(theme=theme, use_unicode=use_unicode)

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        prepared = "{:>2.0%}".format(bar.progress)
        prepared = self._trim(prepared, width)
        rendered = f"{self.theme.percentage_color}{prepared}{Colors.RESET}"
        return (prepared, rendered)


class _TimeWidget(Widget):
    """Widget displaying time information"""

    @staticmethod
    def _format_seconds(seconds: float) -> str:
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return "{:02d}:{:02d}:{:02d}".format(hours, minutes, seconds)


class ElapsedTimeWidget(_TimeWidget):
    """Widget displaying elapsed time"""

    _render_priority = 20

    def __init__(
        self,
        label_elapsed: Optional[str] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        self.reset(
            label=label_elapsed,
            theme=theme,
            use_unicode=use_unicode,
        )

    def reset(
        self,
        label: Optional[str] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        super().reset(theme=theme, use_unicode=use_unicode)
        if label is not None:
            self.label = label
        else:
            self.label = "⏱️" if self.use_unicode else ""

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        parts = []
        if self.label:
            parts.append(self.label)
        parts.append(self._format_seconds(bar.elapsed_time()))

        prepared = " ".join(parts)
        prepared = self._trim(prepared, width)

        rendered = f"{self.theme.time_color}{prepared}{Colors.RESET}"

        return (prepared, rendered)


class EstimatedTimeWidget(_TimeWidget):
    """Widget displaying estimated time"""

    _render_priority = 15

    def __init__(
        self,
        label_estimated: Optional[str] = None,
        spinner_style: Optional[str] = None,
        spinner_frames: Optional[List[str]] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        self.spinner_widget = SpinnerWidget()
        self.reset(
            label=label_estimated,
            spinner_style=spinner_style,
            spinner_frames=spinner_frames,
            progress=0.0,
            theme=theme,
            use_unicode=use_unicode,
        )

    def reset(
        self,
        label: Optional[str] = None,
        spinner_style: Optional[str] = None,
        spinner_frames: Optional[List[str]] = None,
        progress: Optional[float] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        super().reset(theme=theme, use_unicode=use_unicode)
        if label is not None:
            self.label = label
        else:
            self.label = "≈⏱️" if self.use_unicode else "ETA"
        if progress is not None:
            self.progress = progress
        self.eta: Optional[float] = None
        self.eta_updated: datetime = datetime.now()
        self.spinner_widget.reset(
            style=spinner_style or "dots",
            frames=spinner_frames,
            theme=self.theme,
            use_unicode=self.use_unicode,
        )

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        parts = []

        if bar.progress < 1.0:
            if self.progress != bar.progress:
                self.progress = bar.progress
                self.eta = bar.estimated_time()
                self.eta_updated = datetime.now()
            remaining = ""
            if self.eta:
                seconds = max(
                    0, self.eta - (datetime.now() - self.eta_updated).total_seconds()
                )
                remaining = self._format_seconds(seconds)
            else:
                remaining = self.spinner_widget.render(bar, width=(width))[0]
            if self.label:
                parts.append(self.label)
            parts.append(remaining)

        prepared = " ".join(parts)
        prepared = self._trim(prepared, width)
        rendered = f"{self.theme.time_color}{prepared}{Colors.RESET}"

        return (prepared, rendered)


class CounterWidget(Widget):
    """Widget displaying current/total count"""

    _render_priority = 30

    def __init__(
        self, theme: Optional[Theme] = None, use_unicode: Optional[bool] = None
    ):
        self.reset(theme=theme, use_unicode=use_unicode)

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        if bar.total > 0:
            prepared = f"{bar.current}/{bar.total}"
        else:
            prepared = f"{bar.current}"
        prepared = self._trim(prepared, width)

        rendered = f"{self.theme.counter_color}{prepared}{Colors.RESET}"
        return (prepared, rendered)


class SpinnerWidget(Widget):
    """Animated spinner for indeterminate progress"""

    _render_priority = 20

    FRAMES_SNAKE = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    FRAMES_DOTS = ["⣷", "⣯", "⣟", "⡿", "⢿", "⣻", "⣽", "⣾"]
    FRAMES_ARROWS = ["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"]
    FRAMES_BOUNCING = ["⠁", "⠈", "⠐", "⠠", "⢀", "⡀", "⠄", "⠂"]
    FRAMES_SPINNER = ["|", "/", "-", "\\"]

    def __init__(
        self,
        style: str = "dots",
        frames: Optional[List[str]] = None,
        theme: Optional[Theme] = None,
        use_unicode: Optional[bool] = None,
    ):
        self._lock = threading.Lock()
        self.reset(style=style, frames=frames, theme=theme, use_unicode=use_unicode)

    def reset(
        self,
        style: str = "dots",
        frames: Optional[List[str]] = None,
        use_unicode: Optional[bool] = None,
        theme: Optional[Theme] = None,
    ):
        super().reset(theme=theme)

        if use_unicode is None:
            capability = _detect_terminal_capability()
            use_unicode = capability in [
                TerminalCapability.BASIC,
                TerminalCapability.ADVANCED,
            ]

        if frames is not None:
            self.frames = frames
        elif style == "spinner" or not use_unicode:
            self.frames = self.FRAMES_SPINNER
        elif style == "dots":
            self.frames = self.FRAMES_DOTS
        elif style == "arrows":
            self.frames = self.FRAMES_ARROWS
        elif style == "bouncing":
            self.frames = self.FRAMES_BOUNCING
        elif style == "snake":
            self.frames = self.FRAMES_SNAKE
        else:
            raise ValueError(f"Unknown spinner style: {style}")

        self.frame_idx = 0

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        with self._lock:
            prepared = self.frames[self.frame_idx % len(self.frames)]
            self.frame_idx += 1

        prepared = self._trim(prepared, width)
        rendered = f"{self.theme.bar_complete_color}{prepared}{Colors.RESET}"
        return (prepared, rendered)


class RateWidget(Widget):
    """Widget displaying processing rate"""

    _render_priority = 40

    def __init__(
        self, theme: Optional[Theme] = None, use_unicode: Optional[bool] = None
    ):
        self._lock = threading.Lock()
        self.reset(theme=theme, use_unicode=use_unicode)

    def reset(self, theme: Optional[Theme] = None, use_unicode: Optional[bool] = None):
        super().reset(theme=theme, use_unicode=use_unicode)
        self.last_time = None
        self.last_count = 0
        self.current_rate = 0.0

    def render(self, bar: "Bar", width: int) -> Tuple[str, str]:
        with self._lock:
            prepared = ""
            if self.last_time is None:
                self.last_time = time.time()
                self.last_count = bar.current
                prepared = "0.0 it/s"
            else:
                current_time = time.time()
                elapsed = current_time - self.last_time

                if elapsed > 0.5:  # Update rate every 0.5 seconds
                    self.current_rate = (bar.current - self.last_count) / elapsed
                    self.last_time = current_time
                    self.last_count = bar.current

                prepared = f"{self.current_rate:.1f} it/s"

            prepared = self._trim(prepared, width)

        rendered = f"{self.theme.counter_color}{prepared}{Colors.RESET}"
        return (prepared, rendered)


# ============================================================================
# Progress Bar Item
# ============================================================================


@dataclass
class BarItem:
    """Data class representing a progress bar item"""

    index: int
    value: Any


# ============================================================================
# Progress Bar
# ============================================================================


class Bar:
    """Individual progress bar with customizable widgets"""

    def __init__(
        self,
        total: int = 0,
        title: Union[str, Callable[[Any], str], None] = None,
        controller: Optional["_ProgressController"] = None,
        remove_on_complete: bool = False,
        indent: int = 0,
        on_update: Optional[Callable[[int, float], None]] = None,
        on_complete: Optional[Callable[[], None]] = None,
    ):
        """
        Create a progress bar.

        Args:
            total: Total number of items (0 for indeterminate)
            title: Title string or callable returning title
            controller: Progress controller
            remove_on_complete: Remove bar when complete
            indent: Indentation level for nested bars
            on_update: Callback on progress update (current, progress)
            on_complete: Callback when progress completes
        """
        # Validation
        if total < 0:
            raise ValueError("total must be non-negative")
        if indent < 0:
            raise ValueError("indent must be non-negative")

        self.total = total
        self.title = title
        self.set_controller(controller)
        self.remove_on_complete = remove_on_complete
        self.indent = indent
        self.on_update = on_update
        self.on_complete = on_complete

        self.current = 0
        self.progress = 0.0
        self.start_time: Optional[datetime] = None
        self._elapsed_time = 0
        self._item: Optional[BarItem] = None
        self._was_complete = False

    def __enter__(self):
        """Enter context manager"""
        if self.start_time is None:
            self.start_time = datetime.now()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager"""
        if not self.is_complete() and self.total > 0:
            self.update(self.total)
        return False

    def start(self):
        """Start the progress bar"""
        if self.start_time is None:
            self.start_time = datetime.now()

    def set_controller(self, controller: Optional["_ProgressController"]):
        """Set the progress controller"""
        self.controller_ref = weakref.ref(controller) if controller else None

    @contextmanager
    def lock(self):
        """Context manager for thread-safe operations"""
        if self.controller_ref is not None:
            controller = self.controller_ref()
            if controller is not None:
                with controller.lock() as lock:
                    yield lock
                return
        yield None

    def update(self, current: int, total: Optional[int] = None):
        """Update current progress"""
        with self.lock():
            self._update_internal(current, total)

    def _update_internal(self, current: int, total: Optional[int] = None):
        """Internal update without locking"""
        self.current = current
        if total is not None:
            self.total = total
        self._calc_progress()

    def increment(self, count: int = 1):
        """Increment progress by count"""
        with self.lock():
            self._increment_internal(count)

    def _increment_internal(self, count: int):
        """Internal increment without locking"""
        self.current += count
        self._calc_progress()

    def _calc_progress(self):
        """Calculate progress ratio"""
        if self.start_time is None:
            self.start_time = datetime.now()

        if self.total > 0:
            self.progress = min(1.0, self.current / float(self.total))
        else:
            self.progress = 0.0

        # Trigger callbacks
        if self.on_update:
            try:
                self.on_update(self.current, self.progress)
            except Exception:
                logger.exception("on_update callback failed")

        # Check for completion
        if not self._was_complete and self.is_complete():
            self._was_complete = True
            if self.on_complete:
                try:
                    self.on_complete()
                except Exception:
                    logger.exception("on_complete callback failed")

    def reset(self):
        """Reset progress bar to initial state"""
        with self.lock():
            self._reset_internal()

    def _reset_internal(self):
        """Internal reset without locking"""
        self.current = 0
        self.progress = 0.0
        self.start_time = None
        self._was_complete = False

    @property
    def item(self) -> Optional[BarItem]:
        """Get current item being processed"""
        return self._item

    def set_item(self, item: BarItem):
        """Set current item being processed"""
        with self.lock():
            self._set_item_internal(item)

    def _set_item_internal(self, item: BarItem):
        """Set current item being processed"""
        self._item = item

    def elapsed_time(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0.0
        if self.progress < 1.0:
            self._elapsed_time = (datetime.now() - self.start_time).total_seconds()
        return self._elapsed_time

    def estimated_time(self) -> Optional[float]:
        """Calculate and format estimated time remaining"""
        if self.progress == 0.0 or self.total == 0:
            return None

        if self.progress >= 1.0:
            return 0.0

        elapsed = self.elapsed_time()
        eta = elapsed / self.progress * (1.0 - self.progress)

        if eta < 0.0:
            return None

        return eta

    def get_indent(self) -> int:
        """Get current indentation level"""
        return self.indent

    def set_indent(self, indent: int):
        """Set indentation level"""
        if indent < 0:
            raise ValueError("indent must be non-negative")
        self.indent = indent

    def is_complete(self) -> bool:
        """Check if progress is complete"""
        return self.total > 0 and self.progress >= 1.0


# ============================================================================
# Progress Bar
# ============================================================================


class View:
    """View for rendering progress bar with widgets"""

    def __init__(
        self,
        bar: Optional[Bar] = None,
        widgets: Optional[List[Widget]] = None,
        theme: Optional[Theme] = None,
        include_widgets: Optional[Set[type]] = None,
        exclude_widgets: Optional[Set[type]] = None,
        use_unicode: Optional[bool] = None,
        min_update_interval: float = 0.1,
        min_update_progress: float = 0.01,
        update_on_item_change: bool = True,
    ):
        """
        Create a progress bar view.

        Args:
            bar: Weak reference to associated progress bar
            widgets: List of widgets to display
            theme: Color theme
            include_widgets: Set of widget types to include
            exclude_widgets: Set of widget types to exclude
            use_unicode: Whether to use Unicode characters
            min_update_interval: Minimum seconds between updates
            min_update_progress: Minimum progress change to trigger update
            update_on_item_change: Whether to update on item change
        """
        self._bar_ref = weakref.ref(bar) if bar else None

        # Set theme based on terminal capability if not specified
        if theme is None:
            capability = _detect_terminal_capability()
            if capability == TerminalCapability.MINIMAL:
                self.theme = Theme.minimal()
            else:
                self.theme = Theme.default()
        else:
            self.theme = theme

        self.include_widgets = include_widgets or set()
        self.exclude_widgets = exclude_widgets or set()

        self.use_unicode = use_unicode

        # Initialize widgets
        if widgets is None:
            self.widgets = self._default_widgets()
        else:
            self.widgets = widgets

        if min_update_interval < 0:
            raise ValueError("min_update_interval must be non-negative")

        self.min_update_interval = min_update_interval

        if min_update_progress < 0 or min_update_progress > 1.0:
            raise ValueError("min_update_progress must be between 0.0 and 1.0")

        self.min_update_progress = min_update_progress
        self.update_on_item_change = update_on_item_change

        self.cache_update_interval = 5  # 5 seconds
        self.last_update_time: float = 0
        self.prev_progress: float = -1.0
        self._last_item: Optional[BarItem] = None

        self.last_update_for_width: Dict[int, Dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )

        self._render_cache = {}

    def _default_widgets(self) -> List[Widget]:
        """Create default widget set"""
        widgets = []
        include_widgets = self.include_widgets or set()
        exclude_widgets = self.exclude_widgets or set()

        def include_widget(widget: Widget) -> bool:
            if type(widget) in exclude_widgets:
                return False
            if include_widgets and type(widget) not in include_widgets:
                return False
            return True

        title = None
        total = 0
        bar = self.get_bar()
        if bar is not None:
            title = bar.title
            total = bar.total

        if title is not None and TitleWidget not in exclude_widgets:
            widgets.extend(
                filter(
                    include_widget,
                    [
                        TitleWidget(
                            title=title, theme=self.theme, use_unicode=self.use_unicode
                        )
                    ],
                )
            )

        if total > 0:
            widgets.extend(
                filter(
                    include_widget,
                    [
                        BarWidget(theme=self.theme, use_unicode=self.use_unicode),
                        PercentageWidget(
                            theme=self.theme, use_unicode=self.use_unicode
                        ),
                        CounterWidget(theme=self.theme, use_unicode=self.use_unicode),
                        ElapsedTimeWidget(
                            theme=self.theme, use_unicode=self.use_unicode
                        ),
                        EstimatedTimeWidget(
                            theme=self.theme, use_unicode=self.use_unicode
                        ),
                    ],
                )
            )
        else:
            widgets.extend(
                filter(
                    include_widget,
                    [
                        SpinnerWidget(theme=self.theme, use_unicode=self.use_unicode),
                        CounterWidget(theme=self.theme, use_unicode=self.use_unicode),
                        ElapsedTimeWidget(
                            theme=self.theme, use_unicode=self.use_unicode
                        ),
                    ],
                )
            )

        return widgets

    def set_bar(self, bar: Bar):
        """Set the associated progress bar"""
        self._bar_ref = weakref.ref(bar) if bar else None

    def get_bar(self) -> Optional[Bar]:
        """Get the associated progress bar"""
        return self._bar_ref() if self._bar_ref else None

    def should_update(self, width: int) -> bool:
        """Check if enough time has passed for an update"""
        bar = self.get_bar()
        if bar is None:
            return False

        with bar.lock():
            return self._should_update_internal(width=width)

    def _validate_last_update_cache(self):
        """Validate and clean up the last update cache"""
        current_time = time.time()
        for width, last_width_info in list(self.last_update_for_width.items()):
            if (
                current_time - last_width_info.get("time", 0)
                >= self.cache_update_interval
            ):
                self._render_cache.pop(width, None)
                self.last_update_for_width.pop(width, None)

    def _should_update_internal(self, width: int) -> bool:
        bar = self.get_bar()
        if bar is None:
            return False

        self._validate_last_update_cache()

        last_width_info = self.last_update_for_width[width]
        last_progress_at_width = last_width_info.get("progress", 0.0)
        last_time_at_width = last_width_info.get("time", 0)

        current_time = time.time()

        item_changed = self.update_on_item_change and (bar._item != self._last_item)
        progress_changed = abs(
            bar.progress - last_progress_at_width
        ) >= self.min_update_progress or (
            bar.progress == 1.0 and bar.progress != last_progress_at_width
        )
        time_passed = (current_time - last_time_at_width) >= self.min_update_interval

        needs_update = item_changed or progress_changed or time_passed

        if needs_update:
            self.last_update_for_width.pop(width, None)
            self._render_cache.pop(width, None)

        return needs_update

    def reset(self):
        """Reset widgets to initial state"""
        bar = self.get_bar()
        if bar is None:
            return

        with bar.lock():
            for widget in self.widgets:
                widget.reset()

    def render(self, width: int) -> List[str]:
        """Render the progress bar to a string"""
        bar = self.get_bar()
        if bar is None:
            return []

        if self._should_update_internal(width) or width not in self._render_cache:
            self._render_cache[width] = self._render_internal(bar, width)
            self.last_update_for_width[width]["progress"] = bar.progress
            self.last_update_for_width[width]["time"] = time.time()

        # Update timestamps
        self.last_update_time = time.time()
        self.prev_progress = bar.progress
        self._last_item = bar._item

        rendered = self._render_cache[width]

        return rendered

    def _render_internal(self, bar: Bar, width: int) -> List[str]:
        """Render the progress bar to a string"""
        rendered_widgets = [("", "")] * len(self.widgets)
        num_spaces = len(self.widgets) - 1
        all_widgets_empty = False
        empty_widget_indexes = set()
        initial_run = True

        widgets_by_priority = sorted(
            enumerate(self.widgets), key=lambda x: x[1].render_priority
        )

        while not all_widgets_empty:
            available_width = max(0, width - bar.indent - num_spaces)
            last_empty_widget_index = None

            for idx, widget in widgets_by_priority:
                if idx in empty_widget_indexes:
                    continue
                rendered = widget.render(bar, available_width)
                rendered_widgets[idx] = rendered
                rendered_width = len(rendered[0])
                if rendered_width == 0:
                    last_empty_widget_index = idx
                available_width = max(0, available_width - rendered_width)

            if last_empty_widget_index is None:
                break

            empty_widget_indexes.add(last_empty_widget_index)
            num_spaces = max(0, len(self.widgets) - len(empty_widget_indexes) - 1)
            all_widgets_empty = len(self.widgets) == len(empty_widget_indexes)

            if initial_run:
                initial_run = False
                # If first pass results in all widgets empty, we still give it another try
                # to render with retained available width from spaces between widgets
                all_widgets_empty = False

        parts = [w[1] for w in rendered_widgets if w and w[0]]
        if not parts:
            return [" " * width]

        indent_str = " " * bar.indent
        rendered = indent_str + " ".join(parts)
        rendered_width = (
            sum(len(w[0]) for w in rendered_widgets if w)
            + len(indent_str)
            + (len(parts) - 1)
        )
        padding = " " * (width - rendered_width) if width > rendered_width else ""

        return [rendered + padding]


# ============================================================================
# Layout - Organizing multiple progress bars
# ============================================================================
#
class _Layout(MutableSequence):
    """Layout configuration for progress bars"""

    class Type(Enum):
        ROW = "row"
        COLUMN = "column"

    def __init__(
        self,
        name: str,
        type: Type = Type.COLUMN,
        components: Optional[List[Union[View, "_Layout"]]] = None,
    ):
        self.name: str = name
        self.parents: Set[str] = set()
        self.type: _Layout.Type = type
        self._components: List[Union[View, _Layout]] = components or []

    def __getitem__(self, index):
        return self._components[index]

    def __setitem__(self, index, value):
        self._components[index] = value

    def __delitem__(self, index):
        del self._components[index]

    def __len__(self):
        return len(self._components)

    def insert(self, index, value):
        self._components.insert(index, value)

    def add(self, component: Union[View, "_Layout"]):
        self._components.append(component)

    def add_parent(self, parent: str):
        self.parents.add(parent)

    def remove_parent(self, parent: str):
        self.parents.discard(parent)

    # Optional: convenience delegation
    def __getattr__(self, name):
        return getattr(self._components, name)

    def __iter__(self):
        return iter(self._components)

    def __contains__(self, item):
        return item in self._components

    def __repr__(self):
        return f"{type(self).__name__}({self._components!r})"

    def render(self, available_width) -> List[str]:
        """Render all progress bars"""
        lines = []
        if self.type == _Layout.Type.COLUMN:
            for component in self:
                rendered_lines = component.render(available_width)
                lines.extend(rendered_lines)
        elif len(self) > 0:  # ROW
            approx_component_available_width = (available_width // len(self)) - 1
            component_widths = [approx_component_available_width] * len(self)
            leftover = (
                available_width
                - (approx_component_available_width * len(self))
                - len(self)
                + 1
            )

            while leftover > 0:
                for i in range(len(component_widths)):
                    component_widths[i] += 1
                    leftover -= 1
                    if leftover == 0:
                        break

            rendered_components = []
            for idx, component in enumerate(self):
                component_available_width = component_widths[idx]
                rendered_lines = component.render(component_available_width)
                if isinstance(component, Bar):
                    rendered_components.append([rendered_lines])
                else:
                    rendered_components.append(rendered_lines)

            max_lines = max(len(r) for r in rendered_components)
            for idx, rendered in enumerate(rendered_components):
                component_available_width = component_widths[idx]
                added_lines_count = max_lines - len(rendered)
                rendered.extend([" " * component_available_width] * added_lines_count)

            for i in range(max_lines):
                line_parts = [rendered[i] for rendered in rendered_components]
                line = " ".join(line_parts)
                lines.append(line)

        return lines


# ============================================================================
# StdProxy - Redirecting stdout/stderr to progress controller
# ============================================================================


class _StdProxy:
    """Proxy to redirect stdout/stderr to progress controller"""

    def __init__(self, controller: "_ProgressController", stream: TextIO):
        self.controller = controller
        self.stream = stream
        self.buffer = []

    def write(self, data):
        if not data:
            return

        with self.controller.lock():
            self.buffer.append(data)

            if "\n" not in data:
                return

            full_data = "".join(self.buffer)
            self.buffer = []

            # Check if there's trailing data after the last newline
            if not full_data.endswith("\n"):
                # Split at last newline
                last_newline = full_data.rfind("\n")
                complete_part = full_data[: last_newline + 1]
                trailing_part = full_data[last_newline + 1 :]

                # Keep trailing part in buffer
                self.buffer.append(trailing_part)
                data = complete_part
            else:
                data = full_data

            if data:
                new_line_count = data.count("\n")
                self.controller._clear_internal(force_top_lines=new_line_count)
                self.stream.write(data)
                self.controller._display_internal()

    def flush(self):
        with self.controller.lock():
            self._flush_internal()

    def _flush_internal(self):
        if self.buffer:
            data = "".join(itertools.chain(self.buffer, ["\n"]))
            self.buffer = []

            new_line_count = data.count("\n")
            self.controller._clear_internal(force_top_lines=new_line_count)
            self.stream.write(data)
            self.stream.flush()
            self.controller._display_internal()
        else:
            self.stream.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


# ============================================================================
# Collection Watcher
# ============================================================================


def _collection_watcher():
    """Watch collections for changes and update watched bars"""
    error_count = 0
    max_errors = 10

    while True:
        try:
            time.sleep(_ProgressController.instance()._watch_interval)

            with _ProgressController.lock():
                controller = _ProgressController._instance
                if controller is None:
                    continue

                for bar in list(controller._watched_bars):
                    collection_ref = controller._watched_collections.get(bar)
                    if collection_ref is None:
                        continue

                    collection = collection_ref()
                    if collection is None:
                        continue

                    if isinstance(collection, Queue):
                        current_size = collection.qsize()
                    elif isinstance(collection, Sized):
                        current_size = len(collection)
                    else:
                        continue

                    if bar.current != current_size:
                        bar._update_internal(current_size)

                controller._refresh_internal()

            error_count = 0  # Reset on success
        except Exception:
            error_count += 1
            if error_count <= max_errors:
                logger.exception(
                    "Collection watcher failed (error %d/%d)", error_count, max_errors
                )
            elif error_count == max_errors + 1:
                logger.error("Collection watcher: suppressing further errors")
            # Continue despite errors, but stop spamming logs
            time.sleep(1)  # Back off on errors


# ===========================================================================
# Terminal Width Watcher
# ===========================================================================


def _terminal_width_watcher():
    error_count = 0
    max_errors = 10

    while True:
        try:
            global _sigwinch_pending

            if _sigwinch_read_fd is not None:
                # Block until data is available on the pipe
                select.select([_sigwinch_read_fd], [], [])

                # Drain the pipe
                try:
                    while True:
                        os.read(_sigwinch_read_fd, 1024)
                except (OSError, BlockingIOError):
                    pass  # Pipe is empty now

            # Check and clear the flag
            if not _sigwinch_pending:
                continue
            _sigwinch_pending = False

            global _terminal_width, _terminal_height
            _terminal_width, _terminal_height = _get_terminal_size()

            with _ProgressController.lock():
                if _ProgressController._instance is None:
                    continue
                _ProgressController._instance._refresh_internal(force_clear=True)

            error_count = 0  # Reset on success
        except Exception:
            error_count += 1
            if error_count <= max_errors:
                logger.exception(
                    "Terminal width watcher failed (error %d/%d)",
                    error_count,
                    max_errors,
                )
            elif error_count == max_errors + 1:
                logger.error("Terminal width watcher: suppressing further errors")
            # Continue despite errors, but stop spamming logs
            time.sleep(1)  # Back off on errors


# ============================================================================
# Progress _ProgressController - Managing multiple progress bars
# ============================================================================


class _ProgressController:
    """Manages multiple progress bars with different layout modes"""

    _instance: Optional["_ProgressController"] = None
    _lock = threading.Lock()
    _initialized = False
    _terminal_width_watcher_thread = None
    _collection_watcher_thread = None
    _DEFAULT_LAYOUT_NAME = "__default__"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls.lock():
                if cls._instance is None:
                    cls._instance = super(_ProgressController, cls).__new__(cls)
        return cls._instance

    def __init__(
        self,
        remove_on_complete: bool = False,
        terminal_padding_right: int = 20,
        watch_interval: float = 0.5,
        min_update_interval: float = 0.1,
        min_update_progress: float = 0.01,
        update_on_item_change: bool = True,
        force_redraw: bool = False,
        stream: TextIO = sys.stderr,
        proxy_stdout: bool = True,
        proxy_stderr: bool = True,
    ):
        """
        Create a progress controller.

        Args:
            remove_on_complete: Clear progress bars when complete
            terminal_padding_right: Padding from right terminal edge for resizing
            watch_interval: Interval in seconds to check terminal size
            min_update_interval: Minimum interval in seconds between updates
            min_update_progress: Minimum progress change to trigger update
            update_on_item_change: Update when current item changes
            force_redraw: Force redraw on each update
            stream: Output stream for progress bars
            proxy_stdout: Redirect stdout to progress controller
            proxy_stderr: Redirect stderr to progress controller
        """
        if _ProgressController._initialized:
            return

        with _ProgressController._lock:
            if _ProgressController._initialized:
                return

            self._quiet = False
            self._remove_on_complete = remove_on_complete

            if terminal_padding_right < 0:
                raise ValueError("right_terminal_padding must be non-negative")

            self._terminal_padding_right = terminal_padding_right

            if watch_interval <= 0:
                raise ValueError("watch_interval must be positive")

            self._watch_interval = watch_interval

            if min_update_interval < 0:
                raise ValueError("min_update_interval must be non-negative")

            self._min_update_interval = min_update_interval

            if min_update_progress < 0 or min_update_progress > 1.0:
                raise ValueError("min_update_progress must be between 0.0 and 1.0")

            self._min_update_progress = min_update_progress
            self._update_on_item_change = update_on_item_change
            self._force_redraw = force_redraw

            self._proxy_stdout = proxy_stdout
            self._proxy_stderr = proxy_stderr

            self._last_lines_drawn_count = 0

            self._bar_usage: WeakKeyDictionary[Bar, int] = WeakKeyDictionary()
            self._view_usage: WeakKeyDictionary[Bar, Dict[View, int]] = (
                WeakKeyDictionary()
            )

            self._active_bars: Set[Bar] = set()

            self._watched_bars: WeakSet[Bar] = WeakSet()
            self._watched_collections: WeakKeyDictionary[Bar, ReferenceType] = (
                WeakKeyDictionary()
            )

            self.layouts = {
                self._DEFAULT_LAYOUT_NAME: _Layout(
                    self._DEFAULT_LAYOUT_NAME, type=_Layout.Type.COLUMN
                )
            }

            self._registered_layouts: Dict[str, _Layout] = {
                self._DEFAULT_LAYOUT_NAME: self.layouts[self._DEFAULT_LAYOUT_NAME]
            }

            self._layout_usage: Dict[str, int] = defaultdict(int)

            self._stream: TextIO = stream
            self._is_in_tty = self._stream.isatty()

            self._original_stdout: TextIO = sys.stdout
            if self._proxy_stdout:
                sys.stdout = _StdProxy(self, self._original_stdout)

            self._original_stderr: TextIO = sys.stderr
            if self._proxy_stderr:
                sys.stderr = _StdProxy(self, self._original_stderr)

            if not _ProgressController._terminal_width_watcher_thread:
                _ProgressController._terminal_width_watcher_thread = threading.Thread(
                    target=_terminal_width_watcher, daemon=True
                )
                _ProgressController._terminal_width_watcher_thread.start()

            if not _ProgressController._collection_watcher_thread:
                _ProgressController._collection_watcher_thread = threading.Thread(
                    target=_collection_watcher, daemon=True
                )
                _ProgressController._collection_watcher_thread.start()

            self._closed = False
            _ProgressController._initialized = True

    # Property setters for configuration
    @property
    def remove_on_complete(self) -> bool:
        """Whether to remove bars when complete"""
        return self._remove_on_complete

    @remove_on_complete.setter
    def remove_on_complete(self, value: bool):
        """Set whether to remove bars when complete"""
        with self.lock():
            self._remove_on_complete = bool(value)

    @property
    def terminal_padding_right(self) -> int:
        """Padding from right terminal edge in characters"""
        return self._terminal_padding_right

    @terminal_padding_right.setter
    def terminal_padding_right(self, value: int):
        """Set terminal padding from right edge"""
        if value < 0:
            raise ValueError("terminal_padding_right must be non-negative")
        with self.lock():
            self._terminal_padding_right = value
            # Force re-render with new padding
            self._refresh_internal(force_clear=True)

    @property
    def watch_interval(self) -> float:
        """Collection watch update interval in seconds"""
        return self._watch_interval

    @watch_interval.setter
    def watch_interval(self, value: float):
        """Set collection watch update interval"""
        if value <= 0:
            raise ValueError("watch_interval must be positive")
        with self.lock():
            self._watch_interval = value

    @property
    def min_update_interval(self) -> float:
        """Minimum update interval in seconds"""
        return self._min_update_interval

    @min_update_interval.setter
    def min_update_interval(self, value: float):
        """Set minimum update interval"""
        if value < 0:
            raise ValueError("min_update_interval must be non-negative")
        with self.lock():
            self._min_update_interval = value

    @property
    def min_update_progress(self) -> float:
        """Minimum update progress change"""
        return self._min_update_progress

    @min_update_progress.setter
    def min_update_progress(self, value: float):
        """Set minimum update progress change"""
        if value < 0 or value > 1.0:
            raise ValueError("min_update_progress must be between 0.0 and 1.0")
        with self.lock():
            self._min_update_progress = value

    @property
    def update_on_item_change(self) -> bool:
        """Whether to update on item change"""
        return self._update_on_item_change

    @update_on_item_change.setter
    def update_on_item_change(self, value: bool):
        """Set whether to update on item change"""
        with self.lock():
            self._update_on_item_change = bool(value)

    @property
    def force_redraw(self) -> bool:
        """Whether to force redraw on each update"""
        return self._force_redraw

    @force_redraw.setter
    def force_redraw(self, value: bool):
        """Set whether to force redraw on each update"""
        with self.lock():
            self._force_redraw = bool(value)

    @property
    def stream(self) -> TextIO:
        """Get the output stream for progress bars"""
        return self._stream

    @stream.setter
    def stream(self, value: TextIO):
        """Set the output stream for progress bars"""
        if isinstance(value, _StdProxy):
            value = value.stream
        with self.lock():
            self._stream = value
            self._is_in_tty = value.isatty()

    @classmethod
    def instance(cls) -> "_ProgressController":
        """Get the singleton instance of _ProgressController"""
        return cls._instance or cls()

    def __del__(self):
        """Destructor to restore original streams"""
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    @classmethod
    @contextmanager
    def lock(cls):
        """Context manager for thread-safe operations"""
        with _ProgressController._lock:
            yield _ProgressController._lock

    def close(self):
        """Manually restore original stdout/stderr"""
        if self._closed:
            return

        with self.lock():
            if self._closed:
                return

            self._refresh_internal(final=True)

            if self._proxy_stdout and self._original_stdout:
                if isinstance(sys.stdout, _StdProxy):
                    sys.stdout._flush_internal()
                sys.stdout = self._original_stdout
            if self._proxy_stderr and self._original_stderr:
                if isinstance(sys.stderr, _StdProxy):
                    sys.stderr._flush_internal()
                sys.stderr = self._original_stderr

            self._closed = True
            _ProgressController._initialized = False
            _ProgressController._instance = None

    def contains_bar(self, bar: Bar) -> bool:
        """Check if a progress bar is active in the controller"""
        with self.lock():
            return bar in self._active_bars

    def add_bar(
        self,
        bar: Bar,
        view: Optional[View] = None,
        layouts: Optional[List[str]] = None,
        **kwargs,
    ):
        """Add a progress bar to the controller"""
        if view is None:
            view = self.create_view(bar, **kwargs)

        if layouts is None:
            layouts = [self._DEFAULT_LAYOUT_NAME]

        with self.lock():
            self._add_bar_internal(bar, view, layouts)

    def _add_bar_internal(self, bar: Bar, view: View, layouts: List[str]):
        bar.set_controller(self)
        view.set_bar(bar)

        self._active_bars.add(bar)

        for layout in layouts:
            _layout = self._registered_layouts[layout]
            _layout.add(view)

            if bar not in self._bar_usage:
                self._bar_usage[bar] = 0
            self._bar_usage[bar] += 1

            if bar not in self._view_usage:
                self._view_usage[bar] = {}
            if view not in self._view_usage[bar]:
                self._view_usage[bar][view] = 0
            self._view_usage[bar][view] += 1

    def create_bar(self, **kwargs) -> Bar:
        """Create and add a new progress bar"""
        bar_args = [
            "total",
            "title",
            "controller",
            "remove_on_complete",
            "indent",
            "on_update",
            "on_complete",
        ]
        bar_config = {key: kwargs[key] for key in bar_args if key in kwargs}
        bar = Bar(**bar_config)

        self.add_bar(bar, **kwargs)

        return bar

    def remove_bar(self, bar: Bar, layouts: Optional[List[str]] = None):
        """Remove a progress bar from the controller"""
        with self.lock():
            self._remove_bar_internal(bar, layouts)

    def _remove_bar_internal(self, bar: Optional[Bar], layouts: Optional[List[str]]):
        if bar is None:
            return

        if bar not in self._active_bars:
            return

        if not layouts:
            layouts = list(self._registered_layouts.keys())

        for layout in layouts:
            _layout = self._registered_layouts.get(layout)
            if _layout is None:
                continue

            views_in_layout = set()
            if bar in self._view_usage:
                for view in self._view_usage[bar]:
                    if view in _layout:
                        views_in_layout.add(view)

            if not views_in_layout or bar not in self._view_usage:
                continue

            for view in views_in_layout:
                if bar not in self._view_usage:
                    continue

                _layout.remove(view)

                bar_view_usage = self._view_usage[bar]
                bar_view_usage[view] -= 1
                if bar_view_usage[view] <= 0:
                    del bar_view_usage[view]

            self._bar_usage[bar] -= 1

            if self._bar_usage[bar] <= 0:
                self._active_bars.discard(bar)
                self._bar_usage.pop(bar, None)
                self._view_usage.pop(bar, None)
                bar.set_controller(None)

    def add_custom_bar(self, **kwargs):
        bar_args = [
            "total",
            "title",
            "controller",
            "remove_on_complete",
            "indent",
            "on_update",
            "on_complete",
        ]
        bar_config = {key: kwargs[key] for key in bar_args if key in kwargs}

        title_widget_args = [
            "title",
            "theme",
            "use_unicode",
        ]
        title_widget_config = {
            key: kwargs[key] for key in title_widget_args if key in kwargs
        }

        bar_widget_args = [
            "use_unicode",
            "theme",
            "use_unicode",
            "char_start_incomplete",
            "char_start_progress",
            "char_start_complete",
            "char_end_incomplete",
            "char_end_progress",
            "char_end_complete",
            "char_incomplete",
            "char_complete",
            "char_complete_fractions",
        ]
        bar_widget_config = {
            key: kwargs[key] for key in bar_widget_args if key in kwargs
        }

        percentage_widget_args = [
            "theme",
            "use_unicode",
        ]
        percentage_widget_config = {
            key: kwargs[key] for key in percentage_widget_args if key in kwargs
        }

        counter_widget_args = [
            "theme",
            "use_unicode",
        ]
        counter_widget_config = {
            key: kwargs[key] for key in counter_widget_args if key in kwargs
        }

        elapsed_time_widget_args = [
            "label_elapsed",
            "theme",
            "use_unicode",
        ]
        elapsed_time_widget_config = {
            key: kwargs[key] for key in elapsed_time_widget_args if key in kwargs
        }

        estimated_time_widget_args = [
            "label_estimated",
            "spinner_style",
            "spinner_frames",
            "theme",
            "use_unicode",
        ]
        estimated_time_widget_config = {
            key: kwargs[key] for key in estimated_time_widget_args if key in kwargs
        }

        view_args = [
            "bar",
            "widgets",
            "theme",
            "include_widgets",
            "exclude_widgets",
            "use_unicode",
            "min_update_interval",
            "min_update_progress",
            "update_on_item_change",
        ]
        view_config = {key: kwargs[key] for key in view_args if key in kwargs}

        bar = view_config.get("bar") or Bar(**bar_config)

        widgets = view_config.get("widgets") or [
            TitleWidget(**title_widget_config),
            BarWidget(**bar_widget_config),
            PercentageWidget(**percentage_widget_config),
            CounterWidget(**counter_widget_config),
            ElapsedTimeWidget(**elapsed_time_widget_config),
            EstimatedTimeWidget(**estimated_time_widget_config),
        ]

        view_config["bar"] = bar
        view_config["widgets"] = widgets

        view = kwargs.get("view") or View(**view_config)
        layouts = kwargs.get("layouts")

        self.add_bar(bar, view, layouts=layouts)

        return bar

    def create_view(self, bar: Optional[Bar] = None, **kwargs) -> View:
        view_args = [
            "widgets",
            "theme",
            "include_widgets",
            "exclude_widgets",
            "use_unicode",
            "min_update_interval",
            "min_update_progress",
            "update_on_item_change",
        ]
        view_config = {key: kwargs[key] for key in view_args if key in kwargs}

        view = View(bar, **view_config)

        return view

    def add_watch(
        self,
        collection: Union[Queue, Sized],
        title: Union[str, Callable[[Any], str], None] = None,
        bar: Optional[Bar] = None,
        max: Optional[int] = None,
        layouts: Optional[List[str]] = None,
    ) -> Bar:
        """Add a bar to watch a collection, like queue or list"""
        if max is None:
            if isinstance(collection, Queue):
                max = collection.maxsize
            elif isinstance(collection, Sized):
                max = len(collection)
        if bar is None:
            bar = self.create_bar(
                title=title,
                total=max,
                layouts=layouts,
                theme=Theme.load(),
                exclude_widgets=set([ElapsedTimeWidget, EstimatedTimeWidget]),
            )

        self._watched_bars.add(bar)
        self._watched_collections[bar] = weakref.ref(collection)

        return bar

    def remove_watch(self, bar: Bar):
        """Remove a watched bar"""
        self._watched_bars.discard(bar)
        self._watched_collections.pop(bar, None)
        if bar in self._active_bars:
            self.remove_bar(bar)

    def _create_layout(
        self,
        name: str,
        type: _Layout.Type = _Layout.Type.COLUMN,
        parents: Optional[List[str]] = None,
    ):
        """Add a new layout configuration"""
        if parents is None:
            parents = [self._DEFAULT_LAYOUT_NAME]

        with self.lock():
            self._create_layout_internal(name, parents, type=type)

    def _create_layout_internal(
        self, name: str, parents: List[str], type: _Layout.Type = _Layout.Type.COLUMN
    ):
        """Internal method to create layout without locking"""
        if name in self._registered_layouts:
            raise ValueError(
                f"Layout '{name}' already exists. Note: layout names must be globally unique."
            )

        _layout = _Layout(name, type=type)
        self._registered_layouts[name] = _layout

        for parent in parents:
            _layout.add_parent(parent)
            self._registered_layouts[parent].append(_layout)
            self._layout_usage[name] += 1

    def add_layout(self, name: str, parents: Optional[List[str]] = None):
        """Add a new layout configuration"""
        if parents is None:
            parents = [self._DEFAULT_LAYOUT_NAME]

        with self.lock():
            self._add_layout_internal(name, parents)

    def _add_layout_internal(self, name: str, parents: List[str]):
        """Internal method to add layout without locking"""
        _layout = self._registered_layouts.get(name)
        if _layout is None:
            raise ValueError(
                f"Layout '{name}' does not exist. It must be created before it can be used."
            )

        for parent in parents:
            _parent = self._registered_layouts.get(parent)
            if _parent is None:
                raise ValueError(f"Parent layout '{parent}' does not exist.")

            layout_parents = {parent} | _parent.parents
            layout_descendants = self._get_layout_descendants(_layout)

            if layout_parents & layout_descendants:
                raise ValueError(
                    f"Cannot add layout '{name}' to parent '{parent}' as it would create a circular reference."
                )

        for parent in parents:
            _layout.add_parent(parent)

            self._registered_layouts[parent].append(_layout)
            self._layout_usage[name] += 1

    def _get_layout_descendants(self, layout: _Layout) -> Set[_Layout]:
        """Recursively get all descendant layouts"""
        descendants = set()
        descendants.add(layout.name)

        for component in layout:
            if isinstance(component, _Layout):
                descendants.update(self._get_layout_descendants(component))

        return descendants

    def create_row(self, name: str, parents: Optional[List[str]] = None):
        """Create a layout that arranges bars side by side"""
        self._create_layout(name, type=_Layout.Type.ROW, parents=parents)

    def create_column(self, name: str, parents: Optional[List[str]] = None):
        """Create a layout that arranges bars stacked vertically"""
        self._create_layout(name, type=_Layout.Type.COLUMN, parents=parents)

    def remove_layout(self, name: str, parents: Optional[List[str]] = None):
        """Remove a layout configuration"""
        with self.lock():
            self._remove_layout_internal(name, parents)

    def _remove_layout_internal(self, name: str, parents: Optional[List[str]]):
        """Remove a layout configuration"""
        if name == self._DEFAULT_LAYOUT_NAME:
            raise ValueError("Cannot remove default layout")

        _layout = self._registered_layouts.get(name)
        if _layout is None:
            raise ValueError(f"Layout '{name}' does not exist.")

        if not parents:
            parents = list(self._registered_layouts.keys())

        for parent in parents:
            _parent = self._registered_layouts.get(parent)
            if _parent is None:
                raise ValueError(f"Parent layout '{parent}' does not exist.")

            _parent.remove(_layout)

            # Layout can appear multiple times in even the same parent
            if _layout not in _parent:
                _layout.remove_parent(parent)

            self._layout_usage[name] -= 1
            if self._layout_usage[name] <= 0:
                self._clear_layout_internal(_layout)
                del self._layout_usage[name]
                del self._registered_layouts[name]

    def _clear_layout_internal(self, layout: _Layout):
        """Recursively clear all bars from a layout"""
        for component in layout:
            if isinstance(component, View):
                self._remove_bar_internal(component.get_bar(), layouts=[layout.name])
            else:
                self._remove_layout_internal(component.name, parents=[layout.name])

    def render(self, available_width) -> List[str]:
        """Render all progress bars according to layout"""
        if not self._is_in_tty or self._quiet:
            return []
        lines = self.layouts[self._DEFAULT_LAYOUT_NAME].render(available_width)
        return lines

    def hide(self):
        with self.lock():
            self._quiet = True
            self.clear(force=True)

    def unhide(self):
        with self.lock():
            self._quiet = False
            self.display(force_update=True)

    def display(
        self,
        item: Optional[BarItem] = None,
        target_bar: Optional[Bar] = None,
        force_update: bool = False,
        force_clear: bool = False,
    ):
        if not self._is_in_tty or self._quiet:
            return

        with self.lock():
            if target_bar and item is not None:
                target_bar._set_item_internal(item)

            # Check if any bar needs updating
            needs_update = force_update
            if not needs_update:
                for bar in list(self._active_bars):
                    views = self._view_usage.get(bar, {})
                    needs_update = any(
                        view._should_update_internal(width=_terminal_width)
                        for view in views
                    )

                    if needs_update:
                        break
            if not needs_update:
                return

            self._refresh_internal(force_clear=force_clear)

    def _refresh_internal(self, force_clear: bool = False, final: bool = False):
        if not self._is_in_tty or self._quiet:
            return

        try:
            # Hide cursor during update
            self._stream.write("\033[?25l")

            lines = self._render_internal()
            lines_to_draw_count = len(lines)

            # Clear previous lines
            self._clear_internal(
                force=force_clear,
                force_bottom_lines=max(
                    0, self._last_lines_drawn_count - lines_to_draw_count
                ),
            )

            self._display_internal(lines=lines, final=final)
        except Exception:
            logger.exception("Display progress failed")
        finally:
            # Show cursor again
            self._stream.write("\033[?25h")

    def _render_internal(self) -> List[str]:
        """Render the progress bars to lines"""

        available_width = max(0, int(_terminal_width - self._terminal_padding_right))
        lines = self.render(available_width)

        max_lines = max(1, _terminal_height - 1)
        lines = lines[: min(len(lines), max_lines)]

        return lines

    def _display_internal(self, lines: Optional[List[str]] = None, final: bool = False):
        """Display the progress bars"""
        if not self._is_in_tty or self._quiet:
            return

        # Handle completion
        all_complete = all(
            bar.is_complete() for bar in self._active_bars if bar.total > 0
        )
        if all_complete:
            if self._remove_on_complete:
                # Clear all lines
                self._clear_internal(force=True)
                return

        # Render new lines
        if lines is None:
            lines = self._render_internal()

        if not lines:
            return

        output = "\r" + "\n".join(lines)
        self._last_lines_drawn_count = len(lines)

        if final:
            output += "\n"
            self._last_lines_drawn_count += 1

        self._stream.write(output)
        self._stream.flush()

    def clear(
        self, force: bool = False, force_top_lines: int = 0, force_bottom_lines: int = 0
    ):
        """Clear the displayed progress bars"""
        if not self._is_in_tty or self._quiet:
            return

        with self.lock():
            self._clear_internal(
                force=force,
                force_top_lines=force_top_lines,
                force_bottom_lines=force_bottom_lines,
            )

    def _clear_internal(
        self, force: bool = False, force_top_lines: int = 0, force_bottom_lines: int = 0
    ):
        """Clear the displayed progress bars"""
        if not self._is_in_tty or self._quiet:
            return

        if self._last_lines_drawn_count == 0:
            return

        lines_to_clear = self._last_lines_drawn_count
        clear_sequence = []

        if (
            self.force_redraw
            or force
            or force_top_lines + force_bottom_lines >= lines_to_clear
        ):
            # Move up and clear each previous line
            clear_sequence.append(
                "\033[F".join(["\r\033[K"] * lines_to_clear)
            )  # Move up one line and clear from cursor to end of line
        else:
            lines_to_skip = lines_to_clear - force_top_lines - force_bottom_lines - 1

            bottom_to_clear = max(0, force_bottom_lines - force_top_lines)
            lines_to_skip += force_bottom_lines - bottom_to_clear

            clear_sequence.append(
                "\r\033[K\033[F" * bottom_to_clear
            )  # Clear bottom lines
            clear_sequence.append(
                "\033[F" * lines_to_skip
            )  # Move cursor up to the first line to clear
            clear_sequence.append("\033[F\r\033[K" * force_top_lines)

        # Move to beginning of first line
        clear_sequence.append("\r")

        self._stream.write("".join(clear_sequence))
        self._stream.flush()

        self._last_lines_drawn_count = 0


class ProgressAPI(Protocol):
    """Protocol for Progress class methods"""

    def __enter__(self) -> "_ProgressController": ...

    def __exit__(self, exc_type, exc_val, exc_tb) -> None: ...

    @classmethod
    def lock(cls): ...

    @classmethod
    def instance(cls) -> "_ProgressController": ...

    @property
    def remove_on_complete(self) -> bool: ...

    @remove_on_complete.setter
    def remove_on_complete(self, value: bool) -> None: ...

    @property
    def terminal_padding_right(self) -> int: ...

    @terminal_padding_right.setter
    def terminal_padding_right(self, value: int) -> None: ...

    @property
    def watch_interval(self) -> float: ...

    @watch_interval.setter
    def watch_interval(self, value: float) -> None: ...

    @property
    def min_update_interval(self) -> float: ...

    @min_update_interval.setter
    def min_update_interval(self, value: float) -> None: ...

    @property
    def min_update_progress(self) -> float: ...

    @min_update_progress.setter
    def min_update_progress(self, value: float) -> None: ...

    @property
    def update_on_item_change(self) -> bool: ...

    @update_on_item_change.setter
    def update_on_item_change(self, value: bool) -> None: ...

    @property
    def force_redraw(self) -> bool: ...

    @force_redraw.setter
    def force_redraw(self, value: bool) -> None: ...

    @property
    def stream(self) -> TextIO: ...

    @stream.setter
    def stream(self, value: TextIO) -> None: ...

    def close(self) -> None: ...

    def contains_bar(self, bar: Bar) -> bool: ...

    def add_bar(
        self,
        bar: Bar,
        view: Optional[View] = None,
        layouts: Optional[List[str]] = None,
        **kwargs,
    ) -> None: ...

    def create_bar(
        self, view: Optional[View] = None, layouts: Optional[List[str]] = None, **kwargs
    ) -> Bar: ...

    def remove_bar(self, bar: Bar, layouts: Optional[List[str]] = None) -> None: ...

    def add_custom_bar(self, **kwargs): ...

    def create_view(self, bar: Optional[Bar] = None, **kwargs) -> View: ...

    def add_watch(
        self,
        collection: Union[Queue, Sized],
        title: Union[str, Callable[[Any], str], None] = None,
        max: Optional[int] = None,
        layouts: Optional[List[str]] = None,
    ) -> Bar: ...

    def remove_watch(self, bar: Bar) -> None: ...

    def create_row(self, name: str, parents: Optional[List[str]] = None) -> None: ...

    def create_column(self, name: str, parents: Optional[List[str]] = None) -> None: ...

    def add_layout(self, name: str, parents: Optional[List[str]] = None) -> None: ...

    def remove_layout(self, name: str, parents: Optional[List[str]] = None) -> None: ...

    def render(self, available_width: int) -> List[str]: ...

    def hide(self) -> None: ...

    def unhide(self) -> None: ...

    def display(
        self,
        item: Optional[BarItem] = None,
        target_bar: Optional[Bar] = None,
        force_update: bool = False,
        force_clear: bool = False,
    ) -> None: ...

    def clear(self, force: bool = False) -> None: ...


class _ProgressMeta(type):
    """Metaclass to delegate class/property access to _ProgressController"""

    def __getattribute__(cls, name):
        # Try normal lookup first (methods like __enter__, etc)
        try:
            return super().__getattribute__(name)
        except AttributeError:
            pass

        controller = _ProgressController
        inst = _ProgressController.instance()

        # Check if name is a data descriptor on the controller class
        obj = controller.__dict__.get(name)
        if obj and hasattr(obj, "__get__"):
            return obj.__get__(inst, cls)

        # Fallback: instance attribute
        return getattr(inst, name)

    def __setattr__(cls, name, value):
        controller_attr = _ProgressController.__dict__.get(name)

        # If controller has a descriptor with a setter → call it
        if controller_attr and hasattr(controller_attr, "__set__"):
            return controller_attr.__set__(_ProgressController.instance(), value)

        # Otherwise assign normally
        return super().__setattr__(name, value)

    def __enter__(cls):
        return _ProgressController.instance().__enter__()

    def __exit__(cls, *args):
        return _ProgressController.instance().__exit__(*args)


class Progress(metaclass=_ProgressMeta):  # pyright: ignore[reportRedeclaration]
    """Public facade that delegates to _ProgressController singleton"""

    def __new__(cls, *args, **kwargs):
        return _ProgressController.instance()

    def __init__(self, *args, **kwargs):
        pass

    def __getattr__(self, name):
        return getattr(_ProgressController.instance(), name)


if TYPE_CHECKING:
    Progress: ProgressAPI  # type: ignore[assignment]


# ============================================================================
# Terminal Width Handling
# ============================================================================


def _get_terminal_size(default: Optional[os.terminal_size] = None) -> Tuple[int, int]:
    """Return the width of the terminal in columns, with a safe fallback."""
    if default is None:
        default = os.terminal_size([80, 24])
    try:
        # Python 3.3+: built-in, cross-platform
        return os.get_terminal_size(Progress.stream.fileno())
    except OSError:
        # Some environments (cron, IDEs, CI, redirected stdout) have no TTY
        pass
    try:
        # shutil is even safer: can take an explicit fallback
        return shutil.get_terminal_size(fallback=default)
    except Exception:
        pass
    return default


def _detect_terminal_capability():
    term = os.environ.get("TERM", "").lower()
    colorterm = os.environ.get("COLORTERM", "").lower()

    # Explicit override
    if os.environ.get("SUPER_BARIO_TRUECOLOR") == "1":
        return TerminalCapability.ADVANCED

    # Truecolor signals
    if "truecolor" in colorterm or "24bit" in colorterm:
        return TerminalCapability.ADVANCED

    # xterm-256color is truecolor-capable in modern terminals
    if term == "xterm-256color":
        return TerminalCapability.ADVANCED

    return TerminalCapability.BASIC


def _init_terminal_width():
    global _terminal_width, _terminal_height, _sigwinch_read_fd, _sigwinch_write_fd
    try:
        _terminal_width, _terminal_height = _get_terminal_size()
    except Exception:
        _terminal_width, _terminal_height = 80, 24

    # Create a pipe for signal wakeup
    _sigwinch_read_fd, _sigwinch_write_fd = os.pipe()
    # Make both ends non-blocking
    flags = fcntl.fcntl(_sigwinch_read_fd, fcntl.F_GETFL)
    fcntl.fcntl(_sigwinch_read_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)
    flags = fcntl.fcntl(_sigwinch_write_fd, fcntl.F_GETFL)
    fcntl.fcntl(_sigwinch_write_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)


_init_terminal_width()


def _update_terminal_width(signum, frame):
    global _sigwinch_pending
    _sigwinch_pending = True

    try:
        if _sigwinch_write_fd is not None:
            os.write(_sigwinch_write_fd, b"\x00")
    except (OSError, BlockingIOError):
        # Pipe might be full, that's ok - we already have pending signal
        pass

    # Chain the previous SIGWINCH handler (if any)
    try:
        prev = _prev_sigwinch_handler
        if prev and prev not in (signal.SIG_DFL, signal.SIG_IGN):
            # If it's a callable, forward the signal
            if callable(prev):
                prev(signum, frame)
    except Exception:
        logger.exception("Chained SIGWINCH handler failed")


_prev_sigwinch_handler = signal.getsignal(signal.SIGWINCH)
signal.signal(signal.SIGWINCH, _update_terminal_width)


# ============================================================================
# Convenience Functions and Context Managers
# ============================================================================


class ProgressContext:
    """Context manager for progress bars"""

    def __init__(
        self,
        total: int = 0,
        title: Union[str, Callable[[Any], str], None] = None,
        bar: Optional[Bar] = None,
        layouts: Optional[List[str]] = None,
        **kwargs,
    ):
        if bar is not None:
            self.bar = bar
            self.bar.total = total
            if not Progress.contains_bar(self.bar):
                Progress.add_bar(self.bar, layouts=layouts, **kwargs)
        else:
            self.bar = Progress.create_bar(
                total=total, title=title, layouts=layouts, **kwargs
            )

    def __enter__(self):
        self.bar.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Ensure final display
        controller = Progress.instance()
        controller.display()

        if self.bar.remove_on_complete:
            controller.remove_bar(self.bar)
            controller.display(force_update=True)

        return False

    def update(self, current: int, item: Optional[BarItem] = None):
        """Update progress and display"""
        self.bar.update(current)
        Progress.instance().display(item, self.bar)

    def increment(self, count: int = 1, item: Optional[BarItem] = None):
        """Increment progress and display"""
        self.bar.increment(count)
        Progress.instance().display(item, self.bar)


def progress(
    iterable,
    total: int = 0,
    title: Union[str, Callable[[Any], str], None] = None,
    bar: Optional[Bar] = None,
    **kwargs,
) -> Iterator:
    """
    Wrap an iterable to display progress automatically.

    Example:
        for item in progress([1, 2, 3, 4, 5], title="Processing"):
            process(item)

    Args:
        iterable: The iterable to wrap
        total: Total items (auto-detected if possible)
        title: Progress bar title
        **kwargs: Additional arguments for Bar
    """
    if not total:
        if bar is not None:
            total = bar.total
        else:
            try:
                total = len(iterable)
            except TypeError:
                pass

    with ProgressContext(total=total, title=title, bar=bar, **kwargs) as ctx:
        for index, item in enumerate(iterable, 1):
            bar_item = BarItem(index, item)
            if index == 1:
                ctx.update(0, item=bar_item)
            yield item
            ctx.update(index, item=bar_item)
