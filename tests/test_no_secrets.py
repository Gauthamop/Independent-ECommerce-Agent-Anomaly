"""Fail loudly if a real-looking API key is in any file git tracks (before it can be pushed)."""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
# built from pieces so this file does not match its own patterns
PATTERNS = {
    "Groq key": re.compile("gsk" + r"_[A-Za-z0-9]{20,}"),
    "Anthropic key": re.compile("sk-ant" + r"-[A-Za-z0-9_\-]{20,}"),
}


def tracked_files():
    try:
        out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git is not available")
    return [ROOT / line for line in out.splitlines() if line]


def test_no_real_looking_api_key_is_tracked_by_git():
    found = []
    for path in tracked_files():
        if not path.is_file() or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        found += [f"{path.relative_to(ROOT)} ({name})" for name, pattern in PATTERNS.items() if pattern.search(text)]
    assert not found, f"secret-looking text in tracked files: {found}. Move it to .env (git-ignored) and revoke it."


def test_the_env_file_is_ignored_and_the_template_is_blank():
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT, capture_output=True, text=True)
    assert ignored.stdout.strip() == ".env"
    template = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert re.search(r"^GROQ_API_KEY=\s*$", template, re.MULTILINE)       # empty value: a template, not a secret
