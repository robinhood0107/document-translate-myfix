#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

chmod +x .githooks/pre-commit .githooks/commit-msg .githooks/pre-push
git config --local core.hooksPath .githooks

echo "Configured core.hooksPath=.githooks"
echo "Git hooks are now active for this repository."
