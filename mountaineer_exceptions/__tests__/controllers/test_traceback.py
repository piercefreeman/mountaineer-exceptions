import importlib.util
from pathlib import Path
from textwrap import dedent

import pytest

from mountaineer_exceptions.controllers.traceback import (
    ExceptionParser,
    ParsedException,
)


def nested_function(x: int) -> None:
    local_var = "test string"  # noqa: F841
    nested_dict = {"key": "value"}  # noqa: F841
    raise ValueError(f"Test error with {x}")


def function_with_locals() -> None:
    x = 42
    y = "string"  # noqa: F841
    z = [1, 2, 3]  # noqa: F841
    nested_function(x)


@pytest.fixture
def parser():
    return ExceptionParser()


@pytest.fixture
def complex_exception() -> ValueError:
    try:
        function_with_locals()
    except ValueError as e:
        return e
    raise RuntimeError("Expected ValueError was not raised")


def test_basic_exception_parsing(parser):
    try:
        raise ValueError("Test error")
    except ValueError as e:
        result = parser.parse_exception(e)

    assert result.exc_type == "ValueError"
    assert result.exc_value == "Test error"
    assert len(result.frames) > 0

    frame = result.frames[-1]
    assert frame.file_name.endswith("test_traceback.py")
    assert isinstance(frame.line_number, int)
    assert frame.function_name == "test_basic_exception_parsing"
    assert isinstance(frame.code_context, str)
    assert "<span" in frame.code_context
    assert isinstance(frame.local_values, dict)


def test_nested_exception_with_locals(parser, complex_exception):
    result = parser.parse_exception(complex_exception)

    assert result.exc_type == "ValueError"
    assert "Test error with 42" in result.exc_value
    assert len(result.frames) >= 2

    nested_frame = None
    for frame in result.frames:
        if frame.function_name == "nested_function":
            nested_frame = frame
            break

    assert nested_frame is not None
    assert "local_var" in nested_frame.local_values
    assert "nested_dict" in nested_frame.local_values
    assert isinstance(nested_frame.local_values["local_var"], str)
    assert "<span" in nested_frame.local_values["nested_dict"]


def test_syntax_highlighting(parser):
    try:
        x = {"complex": [1, 2, 3], "dict": {"nested": True}}  # noqa: F841
        raise Exception("Test")
    except Exception as e:
        result = parser.parse_exception(e)

    frame = result.frames[-1]
    assert "class=" in frame.code_context
    assert "span" in frame.code_context
    assert frame.local_values["x"].count("<span") > 1


def test_different_file_types(parser, tmp_path):
    py_file = tmp_path / "test.py"
    py_file.write_text(
        dedent(
            """
        def test_function():
            x = 42
            raise ValueError("Test")
    """
        )
    )

    spec = importlib.util.spec_from_file_location("test_module", py_file)
    assert spec

    module = importlib.util.module_from_spec(spec)
    assert module
    assert spec.loader

    spec.loader.exec_module(module)

    result: ParsedException | None = None
    try:
        module.test_function()
    except ValueError as e:
        result = parser.parse_exception(e)

    assert result

    frame = result.frames[-1]
    assert frame.file_name == "test.py"
    assert "test_function" in frame.code_context
    assert "<span" in frame.code_context


def test_get_style_defs(parser):
    styles = parser.get_style_defs()
    assert isinstance(styles, str)
    assert len(styles) > 0


def test_exception_without_traceback(parser):
    exc = ValueError("Test error")
    result = parser.parse_exception(exc)

    assert result.exc_type == "ValueError"
    assert result.exc_value == "Test error"
    assert isinstance(result.frames, list)


def test_line_numbers_accuracy(parser):
    def error_on_specific_line():
        x = 1  # noqa: F841
        y = 2  # noqa: F841
        raise ValueError("Test")

    result: ParsedException | None = None
    try:
        error_on_specific_line()
    except ValueError as e:
        result = parser.parse_exception(e)

    assert result
    frame = result.frames[-1]
    assert "ValueError" in frame.code_context
    assert frame.line_number > 0


@pytest.fixture
def mock_project(tmp_path):
    """Create a mock project structure with packages"""
    # Main package
    package_root = tmp_path / "myproject"
    package_root.mkdir()
    (package_root / "__init__.py").touch()

    # Subpackage
    subpackage = package_root / "subpackage"
    subpackage.mkdir()
    (subpackage / "__init__.py").touch()

    # Some modules
    (package_root / "module.py").touch()
    (subpackage / "submodule.py").touch()

    # Non-package directory
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "standalone.py").touch()

    return tmp_path


@pytest.mark.parametrize(
    "project_path, expected_path",
    [
        ("myproject/module.py", "myproject/module.py"),
        ("myproject/subpackage/submodule.py", "myproject/subpackage/submodule.py"),
        ("outside/standalone.py", "standalone.py"),
    ],
)
def test_package_module(
    project_path: str, expected_path: str, mock_project: Path, parser: ExceptionParser
):
    path = str(mock_project / project_path)
    assert parser.get_package_path(path) == expected_path


def test_nonexistent_path(parser: ExceptionParser):
    path = "/nonexistent/path/file.py"
    assert parser.get_package_path(path) == "file.py"


def test_nested_package_structure(tmp_path, parser: ExceptionParser):
    # Create deep nested structure
    structure = tmp_path / "root" / "pkg1" / "pkg2" / "pkg3"
    structure.mkdir(parents=True)

    # Create __init__.py files at different levels
    (tmp_path / "root" / "pkg1" / "__init__.py").touch()
    (tmp_path / "root" / "pkg1" / "pkg2" / "__init__.py").touch()
    (tmp_path / "root" / "pkg1" / "pkg2" / "pkg3" / "__init__.py").touch()

    test_file = structure / "module.py"
    test_file.touch()

    assert parser.get_package_path(str(test_file)) == "pkg1/pkg2/pkg3/module.py"


def test_payload_truncation():
    """Test that large variable payloads are truncated to prevent JavaScript parsing issues"""
    parser = ExceptionParser(max_payload_length=1000)

    # Test with a very long string that should be truncated (10k characters)
    long_value = "x" * 10000  # 10,000 character string
    formatted_long = parser._format_value(long_value)

    # Should be significantly shorter than original, but allow for HTML markup overhead
    assert len(formatted_long) < 2000, (
        f"Long payload was not sufficiently truncated: {len(formatted_long)} chars"
    )
    assert len(formatted_long) > 500, (
        f"Long payload was over-truncated: {len(formatted_long)} chars"
    )
    assert "[truncated]" in formatted_long, (
        "Truncation marker not found in truncated payload"
    )

    # Test with a short string that should not be truncated
    short_value = "short string"
    formatted_short = parser._format_value(short_value)

    assert "[truncated]" not in formatted_short, (
        "Short payload was incorrectly truncated"
    )
    assert len(formatted_short) < 1000, "Short payload is unexpectedly long"

    # Test with a complex object that creates a long repr (10k+ characters)
    complex_value = {"key" + str(i): "value" * 200 for i in range(200)}
    formatted_complex = parser._format_value(complex_value)

    # Should be significantly shorter than original, but allow for HTML markup overhead
    assert len(formatted_complex) < 2000, (
        f"Complex payload was not sufficiently truncated: {len(formatted_complex)} chars"
    )
    assert len(formatted_complex) > 500, (
        f"Complex payload was over-truncated: {len(formatted_complex)} chars"
    )
    assert "[truncated]" in formatted_complex, (
        "Truncation marker not found in complex payload"
    )

    # Test with custom truncation limit
    parser_small = ExceptionParser(max_payload_length=100)
    formatted_small = parser_small._format_value("x" * 1000)

    # Should be much smaller but allow for HTML markup overhead
    assert len(formatted_small) < 500, (
        f"Small limit was not respected: {len(formatted_small)} chars"
    )
    assert "[truncated]" in formatted_small, (
        "Truncation marker not found with small limit"
    )


def test_payload_truncation_in_exception_parsing():
    """Test that payload truncation works in actual exception parsing"""
    parser = ExceptionParser(max_payload_length=500)

    def function_with_large_locals():
        # Create much larger data (10k+ characters)
        large_data = {"key" + str(i): "x" * 500 for i in range(100)}  # noqa: F841
        small_data = "normal string"  # noqa: F841
        raise ValueError("Test error with large locals")

    try:
        function_with_large_locals()
    except ValueError as e:
        result = parser.parse_exception(e)

    # Find the frame with our function
    target_frame = None
    for frame in result.frames:
        if frame.function_name == "function_with_large_locals":
            target_frame = frame
            break

    assert target_frame is not None, "Could not find target frame"
    assert "large_data" in target_frame.local_values, "large_data not found in locals"
    assert "small_data" in target_frame.local_values, "small_data not found in locals"

    # Check that large data was truncated (allow for HTML markup overhead)
    large_data_formatted = target_frame.local_values["large_data"]
    assert len(large_data_formatted) < 1500, (
        f"Large data was not sufficiently truncated: {len(large_data_formatted)} chars"
    )
    assert len(large_data_formatted) > 200, (
        f"Large data was over-truncated: {len(large_data_formatted)} chars"
    )
    assert "[truncated]" in large_data_formatted, (
        "Truncation marker not found in large data"
    )

    # Check that small data was not truncated
    small_data_formatted = target_frame.local_values["small_data"]
    assert "[truncated]" not in small_data_formatted, (
        "Small data was incorrectly truncated"
    )
