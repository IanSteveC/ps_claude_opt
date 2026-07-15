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

ARCHS="${ARCHS:-gfx906 gfx908 gfx90a gfx1010 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}"
OFF=""; for a in $ARCHS; do OFF="$OFF --offload-arch=$a"; done

# FP32=1 builds the FP32-only (df64) experiment: every double in device code
# becomes a two-float emulated real (../mreal.h, shared with the CUDA build).
# CRITICAL: the df64 error-free transforms (two_prod: p=a*b; err=fmaf(a,b,-p))
# require -ffp-contract=on, NOT fast. Under =fast, clang contracts the a*b into
# the fmaf across statements and folds the error term to zero -- verified on
# gfx1030 to collapse mul accuracy from 2^-46 to 2^-24 (and wreck sincos/exp2).
# =on only contracts within a single source expression, so two_prod survives
# while every other multiply-add still fuses.
if [ -n "$FP32" ]; then
  PS_DEFS="-DPS_FP32"
  FPCONTRACT="-ffp-contract=on"
  APP="period_search_BOINC_hip_claude_fp32"
else
  PS_DEFS=""
  FPCONTRACT="-ffp-contract=fast"
  APP="period_search_BOINC_rocm_claude"
fi
# Optional override for contraction-parity experiments (FP64 build only), e.g.
# PS_FPCONTRACT=on to restrict clang to within-expression FMA fusion.
if [ -n "$PS_FPCONTRACT" ]; then FPCONTRACT="-ffp-contract=$PS_FPCONTRACT"; fi

INC="-I. -I.. -I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib"
HIPFLAGS="-O3 -std=c++17 -fgpu-rdc $FPCONTRACT $PS_DEFS ${PS_EXTRA_DEFS:-} $OFF $INC"

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
