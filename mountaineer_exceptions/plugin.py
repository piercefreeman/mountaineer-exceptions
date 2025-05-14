from mountaineer.plugin import MountaineerPlugin, BuildConfig

from mountaineer_exceptions.controllers.exception_controller import ExceptionController
from mountaineer_exceptions.views import get_core_view_path
from mountaineer.client_compiler.postcss import PostCSSBundler

plugin = MountaineerPlugin(
    name="mountaineer-exceptions",
    controllers=[ExceptionController],
    ssr_root=get_core_view_path("_ssr"),
    static_root=get_core_view_path("_static"),
    build_config=BuildConfig(
        view_root=get_core_view_path(""),
        custom_builders=[
            PostCSSBundler()
        ]
    ),
)
