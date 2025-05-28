from traceback import format_exception

import pytest
from fastapi.testclient import TestClient
from mountaineer import (
    AppController,
    ConfigBase,
    ControllerBase,
    CoreDependencies,
    Depends,
    RenderBase,
)
from mountaineer.cli import handle_build

from mountaineer_exceptions.__tests__.fixtures import get_fixtures_path
from mountaineer_exceptions.controllers.exception_controller import (
    ExceptionController,
)
from mountaineer_exceptions.plugin import plugin as exceptions_plugin


class BrokenRender(RenderBase):
    message: str


class UninheritedConfig(ConfigBase):
    value: str


class BrokenController(ControllerBase):
    """A controller that has an issue in its render function implementation."""

    url = "/broken"
    view_path = "broken/page.tsx"

    def render(
        self,
        bad_config: UninheritedConfig = Depends(
            CoreDependencies.get_config_with_type(UninheritedConfig)
        ),
    ) -> BrokenRender:
        return BrokenRender(message=bad_config.value)


@pytest.fixture
def app_controller() -> AppController:
    return AppController(
        config=ConfigBase(
            ENVIRONMENT="development",
        ),
        view_root=get_fixtures_path("example_view"),
    )


@pytest.fixture
def exception_controller(app_controller: AppController) -> ExceptionController:
    # Make sure we have the properly built plugin before we try to mount it
    handle_build(webcontroller="mountaineer_exceptions.cli:app")

    app_controller.register(exceptions_plugin)

    exception_controller = [
        controller
        for controller in exceptions_plugin.get_controllers()
        if isinstance(controller, ExceptionController)
    ][0]

    return exception_controller


@pytest.fixture
def broken_controller(app_controller: AppController) -> BrokenController:
    broken_controller = BrokenController()
    app_controller.register(broken_controller)
    return broken_controller


@pytest.mark.asyncio
async def test_exception_controller_with_complex_render_error(
    app_controller: AppController,
    exception_controller: ExceptionController,
    broken_controller: BrokenController,
) -> None:
    """
    Test the exception controller by:
    - Creating a controller that has an issue in its render function
    - Catching the stack trace of the complex render error
    - Generating the exception controller HTML using the traceback parser

    """

    # Catch the stack trace of the complex render error
    exc: Exception | None = None
    try:
        # This should trigger the AttributeError
        test_client = TestClient(app_controller.app)
        test_client.get("/broken")
    except Exception as e:
        exc = e

    assert exc is not None
    assert isinstance(exc, TypeError)

    html_payload = await exception_controller._definition.route.view_route(  # type: ignore
        exception=str(exc),
        stack="".join(format_exception(exc)),
        parsed_exception=exception_controller.traceback_parser.parse_exception(exc),
    )
    html = html_payload.body.decode("utf-8")

    assert "Environment: <!-- -->production" in html
    assert "fastapi/dependencies/utils.py" in html
