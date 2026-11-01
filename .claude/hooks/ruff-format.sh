#!/usr/bin/env bash
# PostToolUse(Edit|Write): run ruff fix + format on edited Python files.
# Works from WSL directly, or from Windows by dispatching into WSL. Never fails the tool call.
input=$(cat)
file=$(printf '%s' "$input" | tr '\n' ' ' | sed -n 's/.*"file_path" *: *"\([^"]*\)".*/\1/p')

# Normalise JSON-escaped backslashes to forward slashes, then \\wsl.localhost\ubuntu\home\... -> /home/...
file=$(printf '%s' "$file" | sed -E 's#\\#/#g; s#^/+#/#; s#^/wsl(\.localhost|\$)/[^/]+##')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

export PATH="$HOME/.pixi/bin:$PATH"

if command -v pixi >/dev/null 2>&1; then
  cd "${CLAUDE_PROJECT_DIR:-.}" || exit 0
  pixi run ruff check --fix --quiet "$file" >&2
  pixi run ruff format --quiet "$file" >&2
elif command -v wsl.exe >/dev/null 2>&1; then
  # --exec skips the WSL login shell so nothing here is expanded twice
  # MSYS_NO_PATHCONV stops Git Bash from rewriting /home/... into a Windows path
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu --exec bash -c \
    'export PATH="$HOME/.pixi/bin:$PATH"; cd ~/projects/zerog && pixi run ruff check --fix --quiet "$1"; pixi run ruff format --quiet "$1"' \
    _ "$file" >&2
fi
exit 0
