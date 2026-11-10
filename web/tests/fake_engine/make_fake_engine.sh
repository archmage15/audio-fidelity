#!/usr/bin/env bash
# Build a test engine: real engine source with the two heavy stages stubbed.
# Usage: make_fake_engine.sh /path/to/audio-fidelity-linux /path/to/target
set -euo pipefail
src="$1"; dst="$2"; here="$(cd "$(dirname "$0")" && pwd)"
rm -rf "$dst"; mkdir -p "$dst"
cp -r "$src/scripts" "$src/evaluate.py" "$src/cache_audio.py" "$src/BUILD_INFO.json" "$dst/"
cp "$here/analyze_reference_stub.py" "$dst/scripts/analyze_reference.py"
cp "$here/run_all_stub.py" "$dst/scripts/run_all.py"
cp "$here/probe_mod.py" "$here/nested_stage.py" "$dst/scripts/"
echo "fake engine ready at $dst"
