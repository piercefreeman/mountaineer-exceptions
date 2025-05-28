from importlib.resources import as_file, files
from pathlib import Path


def get_fixtures_path(fixture_name: str) -> Path:
    with as_file(files(__name__).joinpath(fixture_name)) as path:
        return Path(path)
