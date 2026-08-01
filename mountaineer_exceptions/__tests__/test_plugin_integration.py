import subprocess

from fastapi.testclient import TestClient
from mountaineer.cli import handle_build

from mountaineer_exceptions.plugin import plugin
from mountaineer_exceptions.views import get_core_view_path


def test_plugin_boots_with_mountaineer() -> None:
    component = plugin.to_webserver()

    with TestClient(component.app) as client:
        response = client.get("/openapi.json")

    assert response.status_code == 200
    assert "/_exception" in response.json()["paths"]


def test_plugin_frontend_builds() -> None:
    view_root = get_core_view_path("")
    subprocess.run(["npm", "ci"], cwd=view_root, check=True)

    handle_build(webcontroller="mountaineer_exceptions.cli:app")

    for relative_path in (
        ".mountaineer/static/core_main.css",
        ".mountaineer/static/exception_controller.js",
        ".mountaineer/ssr/exception_controller.js",
    ):
        output = view_root / relative_path
        assert output.is_file() and output.stat().st_size > 0
