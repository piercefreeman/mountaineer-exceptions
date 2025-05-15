from mountaineer.client_compiler.postcss import PostCSSBundler
from mountaineer.plugin import BuildConfig, MountaineerPlugin

from mountaineer_exceptions.controllers.exception_controller import ExceptionController
from mountaineer_exceptions.views import get_core_view_path

plugin = MountaineerPlugin(
    name="mountaineer-exceptions",
    controllers=[ExceptionController],
    view_root=get_core_view_path(""),
    build_config=BuildConfig(custom_builders=[PostCSSBundler()]),
)
