import re
from pathlib import Path

from stable_asr.cli import build_parser


REFERENCE_ROOTS = (
    Path("configs"),
    Path("docs"),
    Path("README.md"),
    Path("ROADMAP.md"),
    Path(".github"),
    Path("scripts"),
    Path("examples"),
)
TEXT_SUFFIXES = {".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml"}
STABLE_ASR_COMMAND_RE = re.compile(r"(?<![\w-])stable-asr\s+([a-z0-9][a-z0-9-]*)")


def test_documented_stable_asr_subcommands_exist() -> None:
    commands = _known_subcommands()
    unknown: list[str] = []

    for path in _reference_files():
        text = path.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in STABLE_ASR_COMMAND_RE.finditer(line):
                subcommand = match.group(1)
                if subcommand not in commands:
                    unknown.append(f"{path}:{line_number}: stable-asr {subcommand}")

    assert not unknown, "Unknown stable-asr subcommands:\n" + "\n".join(unknown)


def _known_subcommands() -> set[str]:
    parser = build_parser()
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if isinstance(choices, dict) and choices:
            return {str(command) for command in choices}
    raise AssertionError("could not find stable-asr subcommands in CLI parser")


def _reference_files() -> list[Path]:
    files: list[Path] = []
    for root in REFERENCE_ROOTS:
        if root.is_file():
            candidates = [root]
        else:
            candidates = [path for path in root.rglob("*") if path.is_file()]
        files.extend(path for path in candidates if path.suffix in TEXT_SUFFIXES)
    return sorted(files)
