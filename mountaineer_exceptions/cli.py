from mountaineer.cli import handle_build

from mountaineer_exceptions.plugin import plugin

app = plugin.to_webserver()


def build():
    handle_build(webcontroller="mountaineer_exceptions.cli:app")
