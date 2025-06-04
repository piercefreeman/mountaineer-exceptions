# mountaineer-exceptions

![Mountaineer Logo](https://raw.githubusercontent.com/piercefreeman/mountaineer-exceptions/main/media/preview.png)

![Python Version](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Fpiercefreeman%2Fmountaineer-exceptions%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)
[![Test status](https://github.com/piercefreeman/mountaineer-exceptions/actions/workflows/build.yml/badge.svg)](https://github.com/piercefreeman/mountaineer-exceptions/actions)

A small plugin for mountaineer that shows beautiful tracebacks in the browser.

Features:
- Stacktrace exception rendering
- Runtime variable extraction from stack frames
- Python code markup via pygments

This repository also acts as our proof-of-concept for Mountaineer [Plugins](https://github.com/piercefreeman/mountaineer/issues/201).

## Development

Clone a development copy of mountaineer in the same root. Say you store all your projects in `~/projects` that would be `~/projects/mountaineer` and `~/projects/mountaineer-exceptions`. You will need to temporarily redirect mountaineer to point to your locally cloned version instead of the stable remote:

```bash
cd mountaineer
uv add --editable ../mountaineer-exceptions
```

```bash
cd mountaineer-exceptions
uv sync
```

You'll know you have things working when you can run `uv run pytest -vvv` and they pass.
