from mountaineer.plugin import MountaineerPlugin
from mountaineer_exceptions.views import get_core_view_path
from mountaineer_exceptions.controllers.exception_controller import ExceptionController

plugin = MountaineerPlugin(
    name="mountaineer-exceptions",
    controllers=[ExceptionController],
    ssr_root=get_core_view_path("_ssr"),
    static_root=get_core_view_path("_static"),
)
