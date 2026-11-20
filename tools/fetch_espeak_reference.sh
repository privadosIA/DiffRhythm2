#!/bin/bash
# Copies the exact espeak-ng the reference runs (1.51, Debian build) out of the reference image, so the C# port can be checked against the same
# pronunciations without installing anything: the library, its two audio dependencies that a desktop usually lacks (libpcaudio, libsonic) and the
# data directory. Usage: tools/fetch_espeak_reference.sh <output dir>   (outside the repo; espeak-ng is GPL-3, do not commit it)
# Then run the tests with:
#   DIFFRHYTHM2_ESPEAK_LIB=<dir>/libespeak-ng.so.1.1.51 DIFFRHYTHM2_ESPEAK_DATA=<dir>/espeak-ng-data LD_LIBRARY_PATH=<dir> dotnet test -c Release
set -euo pipefail
out="${1:?usage: $0 <output dir>}"
image="${DIFFRHYTHM2_IMAGE:-mediaorquestator/diffrhythm2:1}"
mkdir -p "$out"
cid=$(docker create "$image")
trap 'docker rm "$cid" >/dev/null' EXIT
lib=/usr/lib/x86_64-linux-gnu
docker cp -L "$cid:$lib/libespeak-ng.so.1" "$out/libespeak-ng.so.1.1.51"
docker cp -L "$cid:/lib/x86_64-linux-gnu/libpcaudio.so.0" "$out/libpcaudio.so.0"
docker cp -L "$cid:/lib/x86_64-linux-gnu/libsonic.so.0" "$out/libsonic.so.0"
rm -rf "$out/espeak-ng-data"
docker cp "$cid:$lib/espeak-ng-data" "$out/espeak-ng-data"
chmod +x "$out"/*.so*
echo "espeak-ng from $image copied to $out"
