#!/bin/bash
#
# Cross-build the optimized period_search HIP app for Windows x64 from Linux
# (MinGW-w64), adapting the milkyway build_hip_win.sh recipe:
#   * device side: hipcc --genco builds ONE multi-arch code object on Linux
#     (the exact GPU ISA the Linux HIP app runs; host-OS neutral), embedded
#     via bin2c;
#   * host side: the module-API path (PS_HIP_MODULE) compiled by MinGW as
#     plain C++ with a hand-declared HIP shim (PS_HIP_WIN, no ROCm headers);
#   * amdhip64 resolved at runtime (LoadLibrary/GetProcAddress) - no import
#     lib, binds whatever the installed AMD driver names its HIP runtime;
#   * BOINC winlibs reused from the CUDA Windows build.
#
# The whole host path is validated on real AMD hardware by the Linux
# -DPS_HIP_WIN build (build_hip.sh's module harness); this only swaps g++->
# MinGW, dlopen->LoadLibrary, and gfx906->all 16 arches.
#
set -e
cd "$(dirname "$0")"

ROCM="${ROCM:-/opt/rocm}"
HIPCC="$ROCM/bin/hipcc"
BOINC_DIR="${BOINC_DIR:-$HOME/builds/boinc}"
MINGW="${MINGW:-x86_64-w64-mingw32}"
W="${W:-../winlibs}"           # reuse BOINC libs cross-built by build_win.sh
OBJ=objwin
mkdir -p "$OBJ"

ARCHS="${ARCHS:-gfx906 gfx908 gfx90a gfx1010 gfx1012 gfx1030 gfx1031 gfx1032 gfx1034 gfx1035 gfx1100 gfx1101 gfx1102 gfx1103 gfx1200 gfx1201}"
OFF=""; for a in $ARCHS; do OFF="$OFF --offload-arch=$a"; done

CXX="$MINGW-g++"
CC="$MINGW-gcc"
APP="period_search_BOINC_hip_claude_win64.exe"

INC="-I. -I.. -I$BOINC_DIR -I$BOINC_DIR/api -I$BOINC_DIR/lib -I$BOINC_DIR/win_build"

# 0. BOINC winlibs must exist (build_win.sh creates ../winlibs)
if [ ! -f "$W/libboinc.a" ] || [ ! -f "$W/libboinc_api.a" ]; then
  echo "ERROR: BOINC winlibs not found in $W (run ../build_win.sh once to cross-build them)"; exit 1
fi

# 1. multi-arch code object (exact device ISA, host-OS neutral)
echo "[genco] hipcc --genco (16 arches: $ARCHS)"
"$HIPCC" --genco $OFF -O3 -std=c++17 -ffp-contract=fast \
  -I. -I.. -I"$BOINC_DIR" -I"$BOINC_DIR/api" -I"$BOINC_DIR/lib" \
  -o "$OBJ/ps_hip.co" Start.cu

# 2. embed the code object as a C array
echo "[bin2c] embedding $(stat -c%s "$OBJ/ps_hip.co") bytes"
python3 modbuild/bin2c.py "$OBJ/ps_hip.co" ps_hip_co > "$OBJ/ps_hip_co.c"

# 3. host objects (MinGW, plain C++, hand-declared HIP shim, no ROCm headers)
CXXFLAGS="-O3 -m64 -std=gnu++17 -fpermissive -DPS_HIP_MODULE -DPS_HIP_WIN -DNDEBUG $INC"
echo "[cc] host objects (mingw)"
$CXX $CXXFLAGS -x c++ -c start_CUDA.cu           -o "$OBJ/start_CUDA.o"
$CXX $CXXFLAGS       -c hip_win_loader.cpp       -o "$OBJ/hip_win_loader.o"
$CXX $CXXFLAGS -DPS_HIP -c period_search_BOINC.cpp -o "$OBJ/period_search_BOINC.o"
$CXX $CXXFLAGS       -c ../VersionInfo.cpp        -o "$OBJ/VersionInfo.o"
$CXX -O3 -m64 -DNDEBUG -x c++ -c "$OBJ/ps_hip_co.c" -o "$OBJ/ps_hip_co.o"
for f in trifac areanorm sphfunc ellfit ludcmp lubksb covsrt memory dot_product; do
  $CXX -O3 -m64 -std=gnu++17 -DNDEBUG -I. -x c++ -c "../$f.c" -o "$OBJ/$f.o"
done
OBJS="$OBJ/start_CUDA.o $OBJ/hip_win_loader.o $OBJ/period_search_BOINC.o $OBJ/VersionInfo.o $OBJ/ps_hip_co.o \
 $OBJ/trifac.o $OBJ/areanorm.o $OBJ/sphfunc.o $OBJ/ellfit.o $OBJ/ludcmp.o $OBJ/lubksb.o $OBJ/covsrt.o $OBJ/memory.o $OBJ/dot_product.o"

# 4. link (fully static except amdhip64.dll [dynamic] + system DLLs)
echo "[ld] $APP"
$CXX -static -static-libgcc -static-libstdc++ $OBJS \
     "$W/libboinc_api.a" "$W/libboinc.a" \
     -lpsapi -lws2_32 -lwinmm -lversion -lshlwapi -o "$APP"
echo "built: $APP"

echo "[verify] arches embedded:"
"$ROCM/llvm/bin/llvm-objdump" --offloading "$OBJ/ps_hip.co" 2>/dev/null | grep -oE "gfx[0-9a-z]+" | sort -u | tr '\n' ' '; echo
file "$APP"
