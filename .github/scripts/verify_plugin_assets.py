import sys
from pathlib import Path
from zipfile import ZipFile


def main(args: list[str]) -> int:
    """Verify a built plugin wheel includes precompiled asset directories."""
    if not args:
        sys.stderr.write(
            "Usage: python verify_plugin_assets.py <asset-prefix> [<asset-prefix> ...]\n"
        )
        return 2

    wheels = sorted(Path("dist").glob("*.whl"))
    if not wheels:
        sys.stderr.write("Expected at least one wheel in dist/, found none\n")
        return 1

    failures: list[tuple[Path, list[str]]] = []
    for wheel_path in wheels:
        missing_prefixes = get_missing_prefixes(wheel_path, args)
        if missing_prefixes:
            failures.append((wheel_path, missing_prefixes))

    if failures:
        for wheel_path, missing_prefixes in failures:
            sys.stderr.write(f"{wheel_path} is missing expected plugin assets:\n")
            for prefix in missing_prefixes:
                sys.stderr.write(f"- {prefix}\n")
        return 1

    sys.stdout.write(
        f"Verified {len(wheels)} wheel(s) include plugin assets: {', '.join(args)}\n"
    )
    return 0


def get_missing_prefixes(wheel_path: Path, asset_prefixes: list[str]) -> list[str]:
    """Return asset prefixes that are absent from a wheel archive."""
    with ZipFile(wheel_path) as wheel:
        names = set(wheel.namelist())

    missing_prefixes: list[str] = []
    for raw_prefix in asset_prefixes:
        prefix = raw_prefix.rstrip("/") + "/"
        if not any(
            name.startswith(prefix) and not name.endswith("/") for name in names
        ):
            missing_prefixes.append(raw_prefix)

    return missing_prefixes


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
