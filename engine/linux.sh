#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
VENV="${AUDIO_TEST_VENV:-$ROOT/.venv-linux}"
[[ "$VENV" = /* ]] || VENV="$ROOT/$VENV"
PY="$VENV/bin/python"
trap 'code=$?; printf "[ERROR] Linux launcher failed at line %s, exit %s.\n" "$LINENO" "$code" >&2; exit "$code"' ERR

usage() {
  cat <<'EOF'
Linux audio fidelity toolkit (CPU, Linux x86_64, Python 3.12+)

  bash linux.sh setup [--system-deps] [--skip-models] [--check]
  bash linux.sh cache /path/to/reference-folder [--recursive] [--confirm-text]
  bash linux.sh cache /path/to/reference.wav
  bash linux.sh evaluate --reference /path/ref.wav --imitation /path/mine.wav
  bash linux.sh evaluate /path/ref.wav /path/mine.wav [--output /path/new-output]
  bash linux.sh analyze [--auto]        # legacy audio/reference layout
  bash linux.sh run                     # legacy audio folder evaluation
  bash linux.sh models [--punctuation]  # download/reuse local models
  bash linux.sh doctor [--skip-models] [--skip-config]
  bash linux.sh test

PYTHON_BIN selects the interpreter used for setup.
AUDIO_TEST_VENV selects the Linux environment (default: .venv-linux).
Cache the reference first; evaluate never analyzes an uncached reference.
EOF
}

die() { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

runtime_environment() {
  export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 PYTHONFAULTHANDLER=1
  export MPLBACKEND=Agg
  export TMPDIR="$ROOT/tmp/runtime-linux" TEMP="$ROOT/tmp/runtime-linux" TMP="$ROOT/tmp/runtime-linux"
  export AUDIO_TEST_TEMP_DIR="$TMPDIR"
  export HF_HOME="${HF_HOME:-$ROOT/tmp/huggingface}"
  export HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
  export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
  export NUMBA_CACHE_DIR="${NUMBA_CACHE_DIR:-$ROOT/tmp/numba-cache-linux}"
  export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/tmp/matplotlib-linux}"
  export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/tmp/pip-cache-linux}"
  export NLTK_DATA="${NLTK_DATA:-$ROOT/tmp/nltk-data}"
  mkdir -p -- "$TMPDIR" "$HF_HOME" "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR" "$PIP_CACHE_DIR" "$NLTK_DATA"
}

require_environment() {
  [[ -x "$PY" ]] || die "Linux environment missing: $PY. Run: bash linux.sh setup"
  "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,12) else 1)' || die 'Python 3.12+ is required.'
  runtime_environment
}

select_python() {
  if [[ -n "${PYTHON_BIN:-}" ]]; then
    command -v -- "$PYTHON_BIN" >/dev/null || die "Interpreter not found: $PYTHON_BIN"
    return
  fi
  local candidate
  for candidate in python3.13 python3.12 python3; do
    if command -v "$candidate" >/dev/null && "$candidate" -c 'import sys; sys.exit(sys.version_info < (3,12))'; then
      PYTHON_BIN="$candidate"
      return
    fi
  done
  die 'Python 3.12+ not found. Install it first, or set PYTHON_BIN=/path/to/python3.12. Ubuntu 24.04+ is recommended.'
}

system_dependencies() {
  command -v apt-get >/dev/null || die '--system-deps supports Debian/Ubuntu. Install espeak-ng, libsndfile, libgomp and CJK fonts with your distribution package manager.'
  local privilege=()
  if (( EUID != 0 )); then
    command -v sudo >/dev/null || die 'sudo is required for --system-deps.'
    privilege=(sudo)
  fi
  "${privilege[@]}" apt-get update
  "${privilege[@]}" apt-get install -y --no-install-recommends python3-venv espeak-ng libsndfile1 libgomp1 fonts-noto-cjk ffmpeg
}

setup() {
  local install_system=0 skip_models=0 check_only=0
  while (( $# )); do
    case "$1" in
      --system-deps) install_system=1 ;;
      --skip-models) skip_models=1 ;;
      --check) check_only=1 ;;
      -h|--help) usage; return 0 ;;
      *) die "Unknown setup option: $1" ;;
    esac
    shift
  done
  [[ "$(uname -s)" == Linux && "$(uname -m)" == x86_64 ]] || die 'This dependency profile targets Linux x86_64.'
  select_python
  "$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,12), "Python 3.12+ required"'
  if (( check_only )); then
    "$PYTHON_BIN" -c 'import venv, ensurepip; print("Python/venv preflight passed")'
    command -v espeak-ng >/dev/null || die 'espeak-ng is missing. Install OS dependencies, or use setup --system-deps.'
    espeak-ng --version
    return
  fi
  (( install_system == 0 )) || system_dependencies
  command -v espeak-ng >/dev/null || die 'Install espeak-ng first, or re-run setup --system-deps.'
  runtime_environment
  if [[ -e "$VENV" && ! -x "$PY" ]]; then
    die "The selected environment is not a Linux venv: $VENV. Choose an empty path using AUDIO_TEST_VENV."
  fi
  if [[ ! -x "$PY" ]]; then
    "$PYTHON_BIN" -m venv "$VENV"
  fi
  require_environment
  "$PY" -m pip install --upgrade pip -i https://pypi.org/simple
  "$PY" -m pip install -r "$ROOT/requirements-linux.txt" -i https://pypi.org/simple
  "$PY" "$ROOT/tmp/patch_emphases.py"
  "$PY" -m pip check
  if (( skip_models )); then
    "$PY" "$ROOT/scripts/linux_check.py" --skip-models --skip-config
  else
    "$PY" "$ROOT/scripts/download_models.py"
    "$PY" "$ROOT/scripts/linux_check.py" --skip-config
  fi
  printf '\nLinux setup complete. Configure the project Xunfei JSON, cache references, then evaluate.\n'
}

command_name="${1:-help}"
(( $# == 0 )) || shift
case "$command_name" in
  help|-h|--help) usage ;;
  setup) setup "$@" ;;
  cache|cache-reference)
    require_environment
    exec "$PY" "$ROOT/cache_audio.py" "$@"
    ;;
  evaluate)
    require_environment
    exec "$PY" "$ROOT/evaluate.py" "$@"
    ;;
  analyze)
    require_environment
    if (( $# )) && [[ "$1" != --auto ]]; then
      exec "$PY" "$ROOT/cache_audio.py" "$@"
    fi
    exec "$PY" "$ROOT/scripts/analyze_reference.py" "$@" </dev/null
    ;;
  run)
    require_environment
    if (( $# )); then
      exec "$PY" "$ROOT/evaluate.py" "$@"
    fi
    exec "$PY" "$ROOT/scripts/run_all.py" "$@" </dev/null
    ;;
  models)
    require_environment
    exec "$PY" "$ROOT/scripts/download_models.py" "$@"
    ;;
  doctor)
    require_environment
    exec "$PY" "$ROOT/scripts/linux_check.py" "$@"
    ;;
  test)
    require_environment
    exec "$PY" -m unittest discover -s "$ROOT/tests" -v
    ;;
  *) usage >&2; die "Unknown command: $command_name" ;;
esac
