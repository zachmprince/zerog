#!/usr/bin/env bash
# PreToolUse(Bash): refuse git staging that could commit the untracked plan file,
# nuclear data, or everything at once (see CLAUDE.md).
input=$(cat)
cmd=$(printf '%s' "$input" | tr '\n' ' ')

case "$cmd" in
  *"git add"*|*"git commit"*) ;;
  *) exit 0 ;;
esac

block() {
  echo "Blocked: $1. Stage explicit paths instead (see CLAUDE.md)." >&2
  exit 2
}

if printf '%s' "$cmd" | grep -Eq 'git add +(-A|--all|\.)( |"|\\|$)'; then
  block "blanket 'git add' would stage the untracked development plan"
fi
if printf '%s' "$cmd" | grep -Eq 'git (add|commit)[^|;&]*(Pointwise-Energy|Development Plan)'; then
  block "the development plan must never be committed"
fi
if printf '%s' "$cmd" | grep -Eq 'git add[^|;&]*(\.h5|\.hdf5|cross_sections)'; then
  block "nuclear data must never be committed"
fi
exit 0
