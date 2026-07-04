#!/bin/bash
#
# Cross-build the optimized period_search CUDA app for Windows x64 from Linux
# (MinGW-w64), following the pattern proven by primegrid/ap27:
#
#   * device side: the SAME multi-arch fatbin the Linux app carries, built by
#     nvcc on Linux (GPU machine code is host-OS independent) and embedded as
#     a bin2c C array;
#   * host side: start_CUDA.cu compiled as plain C++ with -DPS_DRIVER_API -
#     kernel launches and symbol copies go through the driver API backend in
#     cuda_iface.h (nvcuda.dll); no Windows CUDA toolkit, no MSVC, no cudart;
#   * BOINC libs cross-built via BOINC's own lib/Makefile.mingw (or reused
#     from an existing winlibs dir);
#   * an import library for nvcuda.dll generated from the driver-API symbols
#     the objects actually reference.
#
# Prereqs: x86_64-w64-mingw32-g++, CUDA 12.9 toolkit (nvcc/bin2c), BOINC src.
# Env overrides: CUDA, BOINC_DIR, MINGW, AP27W (existing winlibs to reuse)
#
set -e
cd "$(dirname "$0")"

CUDA="${CUDA:-/usr/local/cuda-12.9}"
BOINC_DIR="${BOINC_DIR:-$HOME/builds/boinc}"
MINGW="${MINGW:-x86_64-w64-mingw32}"
AP27W="${AP27W:-$HOME/builds/primegrid/ap27/cuda/winlibs}"
W=winlibs
APP="period_search_BOINC_cuda1291_claude_win64.exe"

CXX="$MINGW-g++"
CC="$MINGW-gcc"

# 0. BOINC windows libs: reuse existing cross-built ones when available
mkdir -p "$W"
if [ ! -f "$W/libboinc.a" ] || [ ! -f "$W/libboinc_api.a" ]; then
  if [ -f "$AP27W/libboinc.a" ] && [ -f "$AP27W/libboinc_api.a" ]; then
    echo "[boinc] reusing cross-built BOINC libs from $AP27W"
    cp "$AP27W/libboinc.a" "$AP27W/libboinc_api.a" "$W/"
    [ -d "$AP27W/shim" ] && cp -r "$AP27W/shim" "$W/"
  else
    echo "[boinc] cross-building BOINC libs into $W/"
    mkdir -p "$W/shim"
    [ -f "$W/shim/config.h" ] || cat > "$W/shim/config.h" <<'EOF'
#ifndef BOINC_MINGW_CONFIG_H
#define BOINC_MINGW_CONFIG_H
#define HAVE_STRCASECMP 1
#define HAVE__STRICMP 1
#define HAVE_STRDUP 1
#endif
EOF
    [ -f "$W/shim/force.h" ] || printf '#include <stddef.h>\n#include <string.h>\n#include "%s/lib/str_replace.h"\n' "$BOINC_DIR" > "$W/shim/force.h"
    ( cd "$W" && MINGW="$MINGW" BOINC_SRC="$BOINC_DIR" \
        make -f "$BOINC_DIR/lib/Makefile.mingw" \
          INCS="-I$(pwd)/shim -I$BOINC_DIR -I$BOINC_DIR/db -I$BOINC_DIR/lib -I$BOINC_DIR/api -I$BOINC_DIR/zip -I$BOINC_DIR/win_build" \
          OPTFLAGS="-O3 -include $(pwd)/shim/force.h" \
          libboinc.a libboinc_api.a )
  fi
fi

# 1. the fatbin: extracted from the LINUX build's device-link object, so the
#    Windows exe carries byte-identical GPU code to the Linux app (verified:
#    zero SASS diff per arch). Building it via the Linux Makefile also removes
#    any flag-drift risk between the two platforms.
echo "[fatbin] building Linux device-link object and extracting its fatbin"
make pscuda.device-link.o > /dev/null
objcopy -O binary --only-section=.nv_fatbin pscuda.device-link.o Start_win.fatbin

# 2. embed as a C array (COFF-safe)
echo "[bin2c] embedding fatbin"
"$CUDA/bin/bin2c" --const --type char --name ps_fatbin Start_win.fatbin > ps_fatbin_win.c

# 3. compile host objects (plain C++/C - no CUDA toolchain involved)
BINC="-I$W/shim -I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib -I$BOINC_DIR/win_build"
CXXFLAGS="-O3 -m64 -std=gnu++17 -DPS_DRIVER_API -DNDEBUG -I. -I$CUDA/include $BINC"
echo "[cc] host objects (mingw)"
$CXX $CXXFLAGS -x c++ -c -o w_start_CUDA.o start_CUDA.cu
$CXX $CXXFLAGS -x c++ -c -o w_ComputeCapability.o ComputeCapability.cu
$CXX $CXXFLAGS -c -o w_period_search_BOINC.o period_search_BOINC.cpp
$CXX $CXXFLAGS -c -o w_VersionInfo.o VersionInfo.cpp
$CC  -O3 -m64 -DNDEBUG -c -o w_ps_fatbin.o ps_fatbin_win.c
# the Linux Makefile sets CC=g++, i.e. the .c math files are compiled as C++
for f in trifac areanorm sphfunc ellfit ludcmp lubksb covsrt memory dot_product; do
  $CXX -O3 -m64 -std=gnu++17 -DNDEBUG -I. -x c++ -c -o w_$f.o $f.c
done
OBJS="w_start_CUDA.o w_ComputeCapability.o w_period_search_BOINC.o w_VersionInfo.o w_ps_fatbin.o \
 w_trifac.o w_areanorm.o w_sphfunc.o w_ellfit.o w_ludcmp.o w_lubksb.o w_covsrt.o w_memory.o w_dot_product.o"

# 4. import library for nvcuda.dll from the symbols actually referenced
echo "[nvcuda] generating import library"
"$MINGW-nm" -u $OBJS | grep -oE '\bcu[A-Z][A-Za-z_0-9]*' | sort -u > "$W/nvcuda.syms"
{
  echo "LIBRARY nvcuda.dll"
  echo "EXPORTS"
  cat "$W/nvcuda.syms"
} > "$W/nvcuda.def"
"$MINGW-dlltool" -d "$W/nvcuda.def" -l "$W/libnvcuda.a"
echo "  $(wc -l < "$W/nvcuda.syms") driver symbols"

# 5. link (fully static except nvcuda.dll + system DLLs)
echo "[ld] $APP"
$CXX -static -static-libgcc -static-libstdc++ $OBJS \
     "$W/libboinc_api.a" "$W/libboinc.a" "$W/libnvcuda.a" \
     -lpsapi -lws2_32 -lwinmm -lversion -lshlwapi -o "$APP"
echo "built: $APP"

# 6. verify the embedded GPU code matches what we intended to ship
echo "[verify] arches in Start_win.fatbin:"
"$CUDA/bin/cuobjdump" --list-elf Start_win.fatbin | grep -oE "sm_[0-9]+" | sort -u -V | tr '\n' ' '; echo
