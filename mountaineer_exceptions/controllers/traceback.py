import inspect
import linecache
import time
import traceback
from collections import defaultdict
from logging import Logger
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import PythonLexer, guess_lexer_for_filename
from pygments.util import ClassNotFound

EXCEPTIONS_LOGGER: Logger | None = None


def get_exceptions_logger() -> Logger:
    """
    Shortcut function to get the mountaineer global logger, without causing a circular import.
    """
    global EXCEPTIONS_LOGGER
    if EXCEPTIONS_LOGGER is None:
        from mountaineer.logging import LOGGER

        EXCEPTIONS_LOGGER = LOGGER.getChild(__name__)
    return EXCEPTIONS_LOGGER


class ReprTimingStats:
    """Track timing statistics for __repr__ calls per object class"""

    def __init__(self):
        self.reset()
        self.logger = get_exceptions_logger()

    def reset(self) -> None:
        """Reset all timing statistics"""
        self.call_times: dict[str, list[float]] = defaultdict(list)
        self.total_calls = 0
        self.total_time = 0.0
        self.slow_calls: list[
            tuple[str, float, str]
        ] = []  # (class_name, time, repr_preview)
        self.cached_calls: dict[str, int] = defaultdict(int)  # (class_name, count)
        self.total_cached = 0

    def record_call(
        self, class_name: str, duration: float, repr_preview: str = ""
    ) -> None:
        """Record a __repr__ call timing"""
        if "(cached)" in class_name:
            # Extract original class name
            original_class = class_name.replace(" (cached)", "")
            self.cached_calls[original_class] += 1
            self.total_cached += 1
            return

        self.call_times[class_name].append(duration)
        self.total_calls += 1
        self.total_time += duration

        # Track slow calls (>10ms)
        if duration > 0.010:  # 10ms threshold
            self.slow_calls.append((class_name, duration, repr_preview[:100]))

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of timing statistics"""
        summary = {
            "total_calls": self.total_calls,
            "total_cached": self.total_cached,
            "total_time": self.total_time,
            "avg_time_per_call": self.total_time / self.total_calls
            if self.total_calls > 0
            else 0,
            "slow_call_count": len(self.slow_calls),
            "classes": {},
            "cached_classes": dict(self.cached_calls),
        }

        for class_name, times in self.call_times.items():
            summary["classes"][class_name] = {
                "call_count": len(times),
                "total_time": sum(times),
                "avg_time": sum(times) / len(times),
                "min_time": min(times),
                "max_time": max(times),
            }

        return summary

    def log_summary(self) -> None:
        """Log a detailed summary of __repr__ timing"""
        summary = self.get_summary()

        self.logger.info("=== __repr__ Performance Summary ===")
        self.logger.info(f"Total calls processed: {summary['total_calls']}")
        self.logger.info(f"Total calls cached: {summary['total_cached']}")

        if summary["total_cached"] > 0:
            self.logger.info("=== Cache Performance ===")
            for class_name, count in sorted(
                summary["cached_classes"].items(), key=lambda x: x[1], reverse=True
            ):
                self.logger.info(f"Cached {count} calls: {class_name}")

        if summary["total_calls"] > 0:
            self.logger.info(
                f"__repr__ execution time: {summary['total_time'] * 1000:.2f}ms"
            )
            self.logger.info(
                f"Average per call: {summary['avg_time_per_call'] * 1000:.2f}ms"
            )
            self.logger.info(f"Slow calls (>10ms): {summary['slow_call_count']}")

        if self.slow_calls:
            self.logger.warning("=== Slow __repr__ Calls ===")
            # Sort by duration, slowest first
            self.slow_calls.sort(key=lambda x: x[1], reverse=True)
            for class_name, duration, preview in self.slow_calls[:10]:  # Top 10 slowest
                self.logger.warning(
                    f"{class_name}: {duration * 1000:.2f}ms - {preview}"
                )

        if not summary["classes"] and summary["total_cached"] == 0:
            return

        # Log per-class statistics (sorted by total time)
        if summary["classes"]:
            class_stats = sorted(
                summary["classes"].items(),
                key=lambda x: x[1]["total_time"],
                reverse=True,
            )

            self.logger.info("=== Per-Class __repr__ Stats ===")
            for class_name, stats in class_stats[:15]:  # Top 15 classes by total time
                self.logger.info(
                    f"{class_name}: {stats['call_count']} calls, "
                    f"total: {stats['total_time'] * 1000:.2f}ms, "
                    f"avg: {stats['avg_time'] * 1000:.2f}ms, "
                    f"max: {stats['max_time'] * 1000:.2f}ms"
                )


class ExceptionFrame(BaseModel):
    id: UUID
    file_name: str
    line_number: int
    function_name: str
    local_values: dict[str, str]
    code_context: str
    start_line_number: int
    end_line_number: int


class ParsedException(BaseModel):
    exc_type: str
    exc_value: str
    frames: list[ExceptionFrame]


class ExceptionParser:
    def __init__(self, max_payload_length: int = 1000):
        self.formatter = HtmlFormatter(style="github-dark")
        self.python_lexer = PythonLexer()
        self.max_payload_length = max_payload_length
        self.repr_stats = ReprTimingStats()
        self.logger = get_exceptions_logger()

        # Cache for repr results to avoid reprocessing identical objects
        self._repr_cache: dict[int, str] = {}
        self._cache_hits = 0
        self._cache_misses = 0

    def parse_exception(self, exc: BaseException) -> ParsedException:
        # Reset timing stats and cache for this parsing session
        self.repr_stats.reset()
        self._clear_repr_cache()

        frames = []
        tb = traceback.extract_tb(exc.__traceback__)

        # Log which frame we'll focus on for local variables
        if tb:
            top_frame = tb[-1]
            top_frame_package = self._get_frame_package(top_frame.filename)
            self.logger.info(
                f"Processing local variables for top frame in package: {top_frame_package}"
            )

        for frame_index, frame_summary in enumerate(tb):
            filename = frame_summary.filename
            lineno = frame_summary.lineno
            function = frame_summary.name

            # Determine if we should process local variables for this frame
            should_process_locals = self._should_process_frame(
                frame_index, len(tb), filename
            )

            locals_dict = {}
            if should_process_locals:
                # Get locals from the frame
                frame = None
                tb_frame = exc.__traceback__
                while tb_frame is not None:
                    if (
                        tb_frame.tb_frame.f_code.co_filename == filename
                        and tb_frame.tb_lineno == lineno
                    ):
                        frame = tb_frame.tb_frame
                        break
                    tb_frame = tb_frame.tb_next

                if frame is not None:
                    for key, value in frame.f_locals.items():
                        if not key.startswith("__"):
                            locals_dict[key] = self._format_value(value, key)

            code_context, start_line, end_line = self._get_context(
                filename, lineno or -1
            )

            frames.append(
                ExceptionFrame(
                    id=uuid4(),
                    file_name=self.get_package_path(filename),
                    line_number=lineno or -1,
                    function_name=function,
                    code_context=code_context,
                    local_values=locals_dict,
                    start_line_number=start_line,
                    end_line_number=end_line,
                )
            )

        # Log the timing summary and cache stats for this parse session
        self.repr_stats.log_summary()

        cache_stats = self._get_cache_stats()
        if cache_stats["cache_hits"] > 0:
            self.logger.info("=== Repr Cache Performance ===")
            self.logger.info(
                f"Cache hits: {cache_stats['cache_hits']}, misses: {cache_stats['cache_misses']}"
            )
            self.logger.info(f"Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
            self.logger.info(f"Cache size: {cache_stats['cache_size']} objects")

        return ParsedException(
            exc_type=exc.__class__.__name__, exc_value=str(exc), frames=frames
        )

    def get_style_defs(self) -> str:
        """Get CSS style definitions for syntax highlighting"""
        return self.formatter.get_style_defs()  # type: ignore

    def get_package_path(self, filepath: str) -> str:
        """
        Extract the relevant package path from a full system path.

        Args:
            filepath: Full system path to a Python file

        Returns:
            Shortened path relative to closest parent package
        """
        path = Path(filepath)

        # Find closest parent directory with __init__.py
        current = path.parent
        package_root = None

        while True:
            if (current / "__init__.py").exists():
                package_root = current
                current = current.parent
            else:
                break

        if package_root is None:
            # No package found, use filename only
            return path.name

        # Get relative path from package root
        try:
            rel_path = path.relative_to(package_root.parent)
            return str(rel_path)
        except ValueError:
            # Fallback to filename if relative_to fails
            return path.name

    def _get_cache_key(self, value: object) -> int | None:
        """
        Get a cache key for an object. Returns None if object shouldn't be cached.

        We cache based on object identity (id) for mutable objects like dicts/lists
        since they're often the same object repeated across frames.
        """
        # Only cache objects that are expensive to repr and likely to be repeated
        if isinstance(value, (dict, list, set)):
            return id(value)

        # Don't cache simple types (they're fast anyway) or custom objects (identity may not be meaningful)
        return None

    def _should_process_frame(
        self, frame_index: int, total_frames: int, filename: str
    ) -> bool:
        """
        Determine if we should process local variables for this frame.

        We only process the top frame (most recent/important) to avoid performance issues
        with deep stacks and framework internals.
        """
        # Only process the last frame (top of stack - where the actual error occurred)
        return frame_index == total_frames - 1

    def _get_frame_package(self, filename: str) -> str:
        """Extract the package name that owns this frame"""
        frame_package = self.get_package_path(filename)

        # Extract just the top-level package name
        if "/" in frame_package:
            return frame_package.split("/")[0]
        elif "." in frame_package:
            return frame_package.split(".")[0]
        else:
            return frame_package

    def _format_value(self, value: object, variable_name: str = "") -> str:
        class_name = value.__class__.__qualname__

        # Check cache first
        cache_key = self._get_cache_key(value)
        if cache_key is not None and cache_key in self._repr_cache:
            self._cache_hits += 1
            self.repr_stats.record_call(
                f"{class_name} (cached)", 0.0, "Retrieved from cache"
            )
            return self._repr_cache[cache_key]

        if cache_key is not None:
            self._cache_misses += 1

        try:
            if inspect.isclass(value) or inspect.isfunction(value):
                # For classes and functions, use str() instead of repr()
                start_time = time.perf_counter()
                raw_value = str(value)
                duration = time.perf_counter() - start_time
                self.repr_stats.record_call(
                    f"{class_name} (str)", duration, raw_value[:50]
                )
            else:
                # Time the repr() call
                start_time = time.perf_counter()
                raw_value = repr(value)
                duration = time.perf_counter() - start_time
                self.repr_stats.record_call(class_name, duration, raw_value[:50])

            # Limit payload length to prevent JavaScript parsing issues
            # Truncate before syntax highlighting to avoid breaking HTML tags
            if len(raw_value) > self.max_payload_length:
                truncate_at = (
                    self.max_payload_length - 20
                )  # Leave room for truncation message
                raw_value = raw_value[:truncate_at] + "... [truncated]"

            # Apply syntax highlighting to the (possibly truncated) raw value
            if inspect.isclass(value) or inspect.isfunction(value):
                result = raw_value
            else:
                result = highlight(raw_value, self.python_lexer, self.formatter)

        except Exception as e:
            # Time even the fallback str() call
            start_time = time.perf_counter()
            result = str(value)
            duration = time.perf_counter() - start_time
            self.repr_stats.record_call(
                f"{class_name} (error fallback)", duration, str(e)[:50]
            )

        # Cache the result for future use
        if cache_key is not None:
            self._repr_cache[cache_key] = result

        return result

    def _clear_repr_cache(self) -> None:
        """Clear the representation cache and reset cache statistics"""
        cache_size = len(self._repr_cache)
        self._repr_cache.clear()

        if cache_size > 0 or self._cache_hits > 0:
            self.logger.debug(
                f"Repr cache cleared: {cache_size} entries, {self._cache_hits} hits, {self._cache_misses} misses"
            )

        self._cache_hits = 0
        self._cache_misses = 0

    def _get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = (
            (self._cache_hits / total_requests * 100) if total_requests > 0 else 0
        )

        return {
            "cache_size": len(self._repr_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate_percent": hit_rate,
        }

    def _get_lexer(self, filename: str, code: str):
        try:
            return guess_lexer_for_filename(filename, code)
        except ClassNotFound:
            return self.python_lexer

    def _get_context(
        self, filename: str, lineno: int, context_lines: int = 5
    ) -> tuple[str, int, int]:
        """
        Get the code context and starting line number for the given error location.

        :param filename: Path to the source file
        :param lineno: Line number where the error occurred
        :param context_lines: Number of lines to show before and after the error

        """
        start_line = max(lineno - context_lines, 1)  # Don't go below line 1
        end_line = lineno + context_lines + 1

        lines = []
        for i in range(start_line, end_line):
            line = linecache.getline(filename, i)
            if line:
                lines.append(line)
        code = "".join(lines)

        lexer = self._get_lexer(filename, code)
        highlighted = highlight(code, lexer, self.formatter)

        return highlighted, start_line, end_line
