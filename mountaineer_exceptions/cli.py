from mountaineer.cli import handle_build

from mountaineer_exceptions.plugin import plugin
from mountaineer_exceptions.views import get_core_view_path

app = plugin.to_webserver()


def build():
    handle_build(webcontroller="mountaineer_exceptions.cli:app")
