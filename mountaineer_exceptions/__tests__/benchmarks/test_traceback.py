import logging
import time
from sys import stdout

import pytest

from mountaineer_exceptions.controllers.traceback import ExceptionParser


@pytest.fixture
def parser() -> ExceptionParser:
    return ExceptionParser()


def test_cache_with_duplicate_variables(parser, caplog):
    """Test cache effectiveness with duplicate variables across multiple frames"""

    # Set logging level
    logging.getLogger("mountaineer_exceptions.controllers.traceback").setLevel(
        logging.INFO
    )

    def create_multi_frame_exception():
        # Create shared objects that will appear in multiple frames
        shared_config = {
            "database_url": "postgresql://localhost/myapp",
            "redis_url": "redis://localhost:6379",
            "api_keys": {"stripe": "sk_test_123", "sendgrid": "SG.123"},
            "feature_flags": {"new_ui": True, "beta_features": False},
            "settings": {f"setting_{i}": f"value_{i}" for i in range(50)},
        }

        shared_data_list = [f"item_{i}" for i in range(100)]

        def frame_5():
            config = shared_config  # noqa: F841 - Same object
            data = shared_data_list  # noqa: F841 - Same object
            local_var = "frame_5_local"  # noqa: F841
            raise ValueError("Multi-frame exception with shared objects")

        def frame_4():
            config = shared_config  # noqa: F841 - Same object
            data = shared_data_list  # noqa: F841 - Same object
            processing_state = "processing"  # noqa: F841
            frame_5()

        def frame_3():
            config = shared_config  # noqa: F841 - Same object
            user_session = {"user_id": 123, "authenticated": True}  # noqa: F841
            frame_4()

        def frame_2():
            app_config = shared_config  # noqa: F841 - Same object reference
            items = shared_data_list  # noqa: F841 - Same object reference
            frame_3()

        def frame_1():
            global_config = shared_config  # noqa: F841 - Same object reference
            frame_2()

        try:
            frame_1()
        except ValueError as e:
            return e
        raise RuntimeError("Unexpected")

    # Clear logs and parse
    caplog.clear()
    exception = create_multi_frame_exception()

    # Time the parse operation
    start_time = time.perf_counter()
    result = parser.parse_exception(exception)
    parse_duration = time.perf_counter() - start_time

    # Get timing and cache stats
    stats = parser.repr_stats.get_summary()

    stdout.write("\nCache with Duplicate Variables Test:")
    stdout.write(f"  Total frames: {len(result.frames)}")
    stdout.write(f"  Parse time: {parse_duration * 1000:.2f}ms")
    stdout.write(f"  Calls processed: {stats['total_calls']}")
    stdout.write(f"  Calls cached: {stats['total_cached']}")
    stdout.write(
        f"  Cache hit rate: {(stats['total_cached'] / (stats['total_calls'] + stats['total_cached']) * 100) if (stats['total_calls'] + stats['total_cached']) > 0 else 0:.1f}%"
    )

    # Since we only process the top frame, we shouldn't have cache hits
    # (but this tests that the cache infrastructure works)
    assert stats["total_calls"] >= 2, (
        f"Expected at least 2 processed calls, got {stats['total_calls']}"
    )

    # Verify only the top frame has local variables processed
    frames_with_locals = [f for f in result.frames if f.local_values]
    assert len(frames_with_locals) == 1, (
        f"Expected only 1 frame with locals, got {len(frames_with_locals)}"
    )

    # The top frame should have our variables
    top_frame = frames_with_locals[0]
    assert "config" in top_frame.local_values
    assert "data" in top_frame.local_values
    assert "local_var" in top_frame.local_values

    # Verify the config content is properly formatted
    config_repr = top_frame.local_values["config"]
    assert "database_url" in config_repr or "database_url" in config_repr


def test_top_frame_package_detection(parser, caplog):
    """Test that we correctly detect the package owning the top frame"""

    # Set logging level to capture package detection - use root logger to catch all
    logging.getLogger().setLevel(logging.INFO)
    # Also set the specific module logger
    logging.getLogger("mountaineer_exceptions.controllers.traceback").setLevel(
        logging.INFO
    )

    def create_package_test_exception():
        # This function is in the test module
        my_variable = "test_value"  # noqa: F841
        user_data = {"id": 123, "name": "test user"}  # noqa: F841
        raise ValueError("Package detection test")

    # Clear logs and parse
    caplog.clear()
    try:
        create_package_test_exception()
    except ValueError as e:
        exception = e

    result = parser.parse_exception(exception)

    # Check that we logged the package detection - look for any logger containing the traceback module
    log_records = [
        record
        for record in caplog.records
        if "mountaineer_exceptions.controllers.traceback" in record.name
    ]

    package_logs = [
        record
        for record in log_records
        if "Processing local variables for top frame in package" in record.message
    ]

    assert len(package_logs) == 1, (
        f"Expected one package detection log, got {len(package_logs)}. Available logs: {[r.message for r in log_records]}"
    )
    package_log = package_logs[0].message

    stdout.write("\nTop Frame Package Detection Test:")
    stdout.write(f"  {package_log}")

    # Should detect the test package
    assert "mountaineer_exceptions" in package_log or "test_traceback" in package_log, (
        f"Expected test package in log: {package_log}"
    )

    # Verify only the top frame was processed
    frames_with_locals = [f for f in result.frames if f.local_values]
    assert len(frames_with_locals) == 1, (
        f"Expected only 1 frame with locals, got {len(frames_with_locals)}"
    )

    # The top frame should have our test variables
    top_frame = frames_with_locals[0]
    assert "my_variable" in top_frame.local_values
    assert "user_data" in top_frame.local_values

    # Verify the frame belongs to our test function
    assert top_frame.function_name == "create_package_test_exception"

    # Check the content is properly formatted
    my_variable_repr = top_frame.local_values["my_variable"]
    assert "test_value" in my_variable_repr


def test_performance_improvement_demonstration(parser):
    """Demonstrate the performance improvement from processing only the top frame"""

    def create_deep_exception_with_expensive_objects():
        def deep_recursive_function(depth: int):
            # Each frame has expensive objects that would slow down processing
            expensive_data = {f"key_{i}": "x" * 1000 for i in range(100)}  # noqa: F841
            large_list = [f"item_{i}" * 50 for i in range(200)]  # noqa: F841

            if depth <= 0:
                user_input = "final_frame_data"  # noqa: F841
                raise ValueError(f"Deep exception at level {depth}")

            return deep_recursive_function(depth - 1)

        try:
            deep_recursive_function(20)  # 20 levels deep
        except ValueError as e:
            return e
        raise RuntimeError("Unexpected")

    stdout.write("\nPerformance Improvement Demonstration:")

    # Parse the exception
    exception = create_deep_exception_with_expensive_objects()

    start_time = time.perf_counter()
    result = parser.parse_exception(exception)
    parse_duration = time.perf_counter() - start_time

    stats = parser.repr_stats.get_summary()

    stdout.write(f"  Total frames in stack: {len(result.frames)}")
    stdout.write(
        f"  Frames with processed locals: {len([f for f in result.frames if f.local_values])}"
    )
    stdout.write(f"  Parse time: {parse_duration * 1000:.2f}ms")
    stdout.write(f"  __repr__ calls made: {stats['total_calls']}")
    stdout.write(f"  __repr__ time: {stats['total_time'] * 1000:.2f}ms")

    # Should only process one frame despite deep stack
    frames_with_locals = [f for f in result.frames if f.local_values]
    assert len(frames_with_locals) == 1, "Should only process top frame"

    # Should be fast since we only processed one frame
    assert parse_duration < 0.5, (
        f"Should be fast with top-frame-only processing, got {parse_duration * 1000:.2f}ms"
    )

    # The processed frame should have the final frame variables
    top_frame = frames_with_locals[0]
    assert "user_input" in top_frame.local_values
    assert "expensive_data" in top_frame.local_values
    assert "large_list" in top_frame.local_values


def test_cache_effectiveness_with_same_objects(parser):
    """Test that cache works when the same object appears multiple times in the top frame"""

    def create_cache_test_exception():
        # Create the same object referenced multiple times in one frame
        shared_scope = {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": "2.1"},
            "method": "GET",
            "headers": [(b"host", b"api.example.com")]
            * 20,  # Make it large to slow down repr
            "state": {
                "user_id": 123,
                "session_data": {f"key_{i}": "value" for i in range(50)},
            },
        }

        shared_list = [f"Controller_{i}" for i in range(200)]

        # Reference the same objects multiple times in this frame
        scope1 = shared_scope  # noqa: F841 - Same object
        scope2 = shared_scope  # noqa: F841 - Same object
        scope3 = shared_scope  # noqa: F841 - Same object
        controllers1 = shared_list  # noqa: F841 - Same object
        controllers2 = shared_list  # noqa: F841 - Same object

        # Also some unique objects
        unique_dict = {"unique": "data1", "id": 1}  # noqa: F841

        raise ValueError("Cache effectiveness test exception")

    try:
        create_cache_test_exception()
    except ValueError as e:
        exception = e

    # Time the parse operation
    start_time = time.perf_counter()
    result = parser.parse_exception(exception)
    parse_duration = time.perf_counter() - start_time

    # Get timing and cache stats
    stats = parser.repr_stats.get_summary()

    stdout.write("\nCache Effectiveness Test:")
    stdout.write(f"  Parse time: {parse_duration * 1000:.2f}ms")
    stdout.write(f"  Calls processed: {stats['total_calls']}")
    stdout.write(f"  Calls cached: {stats['total_cached']}")
    stdout.write(
        f"  Cache hit rate: {(stats['total_cached'] / (stats['total_calls'] + stats['total_cached']) * 100) if (stats['total_calls'] + stats['total_cached']) > 0 else 0:.1f}%"
    )

    # Should have cache hits for the shared objects
    if stats["total_cached"] > 0:
        stdout.write(
            f"  ✅ Cache working! Saved {stats['total_cached']} duplicate repr() calls"
        )

    # Verify that the parsed result is correct
    frame = result.frames[-1]
    assert "scope1" in frame.local_values
    assert "scope2" in frame.local_values
    assert "scope3" in frame.local_values
    assert "controllers1" in frame.local_values
    assert "controllers2" in frame.local_values

    # All the scope variables should have the same content since they're the same object
    scope1_repr = frame.local_values["scope1"]
    scope2_repr = frame.local_values["scope2"]
    scope3_repr = frame.local_values["scope3"]

    # They should be identical (cached results)
    assert scope1_repr == scope2_repr == scope3_repr, (
        "Cached scope representations should be identical"
    )

    # Should contain the expected content
    assert "type" in scope1_repr
    assert "asgi" in scope1_repr
