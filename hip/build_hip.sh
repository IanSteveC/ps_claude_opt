#!/bin/bash
#
# Build the optimized period_search app for AMD GPUs (HIP/ROCm), Linux x64.
# Produces one binary with device code objects bundled for every requested
# arch (the AMD analogue of the CUDA multi-arch fatbin):
#   Radeon VII gfx906 | CDNA1 gfx908 | CDNA2 gfx90a | RDNA1 gfx1010 |
#   RDNA2 gfx1030 | RDNA3 gfx1100 | RDNA4 gfx1200
#
# Device code is cross-TU (-fgpu-rdc), matching the CUDA build's -rdc=true, so
# the __constant__/__device__ globals defined in Start.cu resolve from the
# hipMemcpyToSymbol calls in start_CUDA.cu.
#
set -e
cd "$(dirname "$0")"

ROCM="${ROCM:-/opt/rocm}"
HIPCC="$ROCM/bin/hipcc"
BOINC_DIR="${BOINC_DIR:-$HOME/builds/boinc}"
OBJ=obj
mkdir -p "$OBJ"

ARCHS="${ARCHS:-gfx906 gfx908 gfx90a gfx1010 gfx1030 gfx1100 gfx1200}"
OFF=""; for a in $ARCHS; do OFF="$OFF --offload-arch=$a"; done

INC="-I. -I.. -I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib"
HIPFLAGS="-O3 -std=c++17 -fgpu-rdc -ffp-contract=fast $OFF $INC"
APP="period_search_BOINC_rocm_claude"

echo "[hip] device+host TUs for: $ARCHS"
for tu in Start start_CUDA ComputeCapability; do
  echo "  $tu.cu"
  $HIPCC $HIPFLAGS -c "$tu.cu" -o "$OBJ/$tu.o"
done

echo "[cc] host math files (C compiled as C++, matching the CUDA Makefile)"
for f in trifac areanorm sphfunc ellfit ludcmp lubksb covsrt memory dot_product; do
  g++ -O3 -std=c++17 -I.. -c "../$f.c" -o "$OBJ/$f.o"
done
g++ -O3 -std=c++17 -DPS_HIP $INC -c period_search_BOINC.cpp -o "$OBJ/period_search_BOINC.o"

echo "[ld] $APP (device link + host link)"
$HIPCC --hip-link -fgpu-rdc $OFF -o "$APP" \
  "$OBJ/Start.o" "$OBJ/start_CUDA.o" "$OBJ/ComputeCapability.o" \
  "$OBJ/period_search_BOINC.o" \
  "$OBJ/trifac.o" "$OBJ/areanorm.o" "$OBJ/sphfunc.o" "$OBJ/ellfit.o" \
  "$OBJ/ludcmp.o" "$OBJ/lubksb.o" "$OBJ/covsrt.o" "$OBJ/memory.o" "$OBJ/dot_product.o" \
  "$BOINC_DIR/api/libboinc_api.a" "$BOINC_DIR/lib/libboinc.a" -lpthread
echo "built: $APP"

echo "[verify] arches embedded in the binary:"
"$ROCM/llvm/bin/llvm-objdump" --offloading "$APP" 2>/dev/null \
  | grep -oE "gfx[0-9a-z]+" | sort -u | tr '\n' ' '; echo
