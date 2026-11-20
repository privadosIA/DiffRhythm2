#!/usr/bin/env bash
# Starts the DiffRhythm2 HTTP service (src/DiffRhythm2.Service), building it first when it is not built.
# Run it from anywhere: it finds the repository from its own location. See docs/http-service.md.
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run-service.sh [options]

Starts diffrhythm2-service in the foreground (Ctrl+C or SIGTERM stops it). Every option has an environment variable;
an option wins over the variable.

  --model DIR        model directory with the six model files        DIFFRHYTHM2_CKPT   (default: ~/models/diffrhythm2)
  --port N           port, on all interfaces                          DIFFRHYTHM2_PORT   (default: 8788)
  --device D         cpu, cuda or cuda:N                              DIFFRHYTHM2_DEVICE (default: cpu)
  --token-file FILE  file whose first line is the bearer token        (or DIFFRHYTHM2_TOKEN; no token leaves the API open)
  --jobs DIR         where finished songs are kept                    DIFFRHYTHM2_JOBS   (default: ~/.local/share/diffrhythm2/jobs)
  --isolation M      process (a worker process per song: memory, video memory included, returns to the system after each
                     song) or inprocess                               DIFFRHYTHM2_ISOLATION (default: process)
  --espeak-lib FILE  the libespeak-ng library (version 1.51 is the reference)   DIFFRHYTHM2_ESPEAK_LIB
  --espeak-data DIR  its espeak-ng-data directory                     DIFFRHYTHM2_ESPEAK_DATA
  --build            build again even if the service is already built
  -h, --help         this text

A cuda device builds the service with the CUDA backend (-p:TorchBackend=cuda-linux), any other with the CPU one; if the
last build used the other backend, it is built again. The first build takes a few minutes.
Exit codes: 0 stopped normally, 2 wrong usage or a setting that cannot be used, anything else: the service's own.
USAGE
}

die() { echo "error: $*" >&2; exit 2; }

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model="${DIFFRHYTHM2_CKPT:-$HOME/models/diffrhythm2}"
port="${DIFFRHYTHM2_PORT:-8788}"
device="${DIFFRHYTHM2_DEVICE:-cpu}"
jobs="${DIFFRHYTHM2_JOBS:-${XDG_DATA_HOME:-$HOME/.local/share}/diffrhythm2/jobs}"
isolation="${DIFFRHYTHM2_ISOLATION:-process}"
token="${DIFFRHYTHM2_TOKEN:-}"
espeak_lib="${DIFFRHYTHM2_ESPEAK_LIB:-}"
espeak_data="${DIFFRHYTHM2_ESPEAK_DATA:-}"
rebuild=0

need_value() { [[ $# -ge 2 ]] || die "option $1 needs a value"; }
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       need_value "$@"; model="$2"; shift 2 ;;
    --port)        need_value "$@"; port="$2"; shift 2 ;;
    --device)      need_value "$@"; device="$2"; shift 2 ;;
    --jobs)        need_value "$@"; jobs="$2"; shift 2 ;;
    --isolation)   need_value "$@"; isolation="$2"; shift 2 ;;
    --espeak-lib)  need_value "$@"; espeak_lib="$2"; shift 2 ;;
    --espeak-data) need_value "$@"; espeak_data="$2"; shift 2 ;;
    --token-file)  need_value "$@"; [[ -r "$2" ]] || die "--token-file: cannot read '$2'"
                   IFS= read -r token < "$2" || true; shift 2 ;;
    --build)       rebuild=1; shift ;;
    -h|--help)     usage; exit 0 ;;
    *)             echo "error: unknown option '$1' (try --help)" >&2; exit 2 ;;
  esac
done

[[ "$port" =~ ^[0-9]+$ ]] && (( port >= 1 && port <= 65535 )) || die "the port must be a number from 1 to 65535, got '$port'"
[[ "$device" =~ ^(cpu|cuda|cuda:[0-9]+)$ ]] || die "the device must be cpu, cuda or cuda:N, got '$device'"
[[ "$isolation" =~ ^(process|inprocess)$ ]] || die "the isolation must be process or inprocess, got '$isolation'"
command -v dotnet >/dev/null || die "dotnet is not installed (the .NET 8 runtime or SDK is needed)"

# The model directory: warn (the service itself reports it too at /health) with the command that fixes it.
missing=()
for f in config.json model.safetensors decoder.json decoder.safetensors mulan_text.safetensors mulan_text_tokenizer.json; do
  [[ -f "$model/$f" ]] || missing+=("$f")
done
if (( ${#missing[@]} > 0 )); then
  echo "warning: $model lacks: ${missing[*]}" >&2
  echo "         the service starts, but /health says it is not ready. To download and convert everything:" >&2
  echo "         dotnet $root/src/DiffRhythm2.Cli/bin/Release/net8.0/diffrhythm2.dll install --model '$model' --accept-licenses" >&2
fi

# Build when needed, with the backend the device asks for.
backend=cpu; [[ "$device" == cuda* ]] && backend=cuda-linux
out="$root/src/DiffRhythm2.Service/bin/Release/net8.0"
dll="${DIFFRHYTHM2_SERVICE_DLL:-$out/diffrhythm2-service.dll}"
marker="$out/.torch-backend"
if [[ -z "${DIFFRHYTHM2_SERVICE_DLL:-}" ]] && { (( rebuild )) || [[ ! -f "$dll" ]] || [[ "$(cat "$marker" 2>/dev/null)" != "$backend" ]]; }; then
  echo "building the service (backend: $backend) ..." >&2
  dotnet build "$root/src/DiffRhythm2.Service" -c Release -p:TorchBackend="$backend" -v quiet -nologo >&2
  echo "$backend" > "$marker"
fi
[[ -f "$dll" ]] || die "the service is not built: $dll does not exist"

export DIFFRHYTHM2_CKPT="$model" DIFFRHYTHM2_PORT="$port" DIFFRHYTHM2_DEVICE="$device" DIFFRHYTHM2_JOBS="$jobs" DIFFRHYTHM2_ISOLATION="$isolation"
[[ -n "$token" ]] && export DIFFRHYTHM2_TOKEN="$token"
if [[ -n "$espeak_lib" ]]; then
  export DIFFRHYTHM2_ESPEAK_LIB="$espeak_lib"
  export LD_LIBRARY_PATH="$(dirname "$espeak_lib")${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"   # its own dependencies sit beside it
fi
[[ -n "$espeak_data" ]] && export DIFFRHYTHM2_ESPEAK_DATA="$espeak_data"

exec dotnet "$dll"
