#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
LIBTORCH_DIR="${LIBTORCH_DIR:-$HOME/opt/libtorch}"
if [[ ! -f "$LIBTORCH_DIR/share/cmake/Torch/TorchConfig.cmake" ]]; then
  echo "ERROR: LibTorch CPU not found at $LIBTORCH_DIR" >&2
  echo 'Set LIBTORCH_DIR=/path/to/libtorch before running.' >&2
  exit 1
fi
for file in checkpoints/lane_unet.pt checkpoints/lane_unet.pt.cfg data/videos/solidYellowLeft.mp4; do
  if [[ ! -f "$file" ]]; then echo "ERROR: missing $file" >&2; exit 1; fi
done
cmake -S . -B build-cpu -DLANE_WITH_TORCH=ON \
  -DCMAKE_PREFIX_PATH="$LIBTORCH_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --target infer_video -j2
mkdir -p reports/videos
./build-cpu/infer_video \
  --video data/videos/solidYellowLeft.mp4 \
  --model checkpoints/lane_unet.pt \
  --out reports/videos/lane_overlay.avi \
  --mask-out reports/videos/lane_mask.avi \
  --geometry-out reports/videos/lane_hough_segments.avi \
  --mode both --threshold 0.5 --roi 1 --min-area 0 \
  --close-vertical 0 --hough 1 --hough-votes 25 \
  --hough-min-length 25 --hough-max-gap 15 \
  --hough-min-angle 25 --hough-min-y 0.55 \
  --hough-max-segments 30 --threads 2 --device cpu --display 0
echo 'Complete. Videos are in reports/videos/.'
