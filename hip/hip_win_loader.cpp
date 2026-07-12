/* hip_win_loader.cpp
 *
 * PS_HIP_WIN only. Resolves the amdhip64 module/runtime API at runtime instead
 * of hard-linking an import library, so one exe binds whatever the installed
 * AMD stack names its HIP runtime (amdhip64.dll / amdhip64_N.dll on Windows;
 * libamdhip64.so[.N] on Linux) and runs CPU-only if none is present.
 *
 * The pointer set + signatures come from the single PS_HIP_API list in
 * hip_win_shim.h, so this stays in lock-step with the host call sites.
 */
#if defined(PS_HIP_WIN)

#include "hip_win_shim.h"
#include <stdio.h>

/* one definition per API entry, null until resolved */
#define PS_HIP_DEF_PTR(ret, name, params) extern "C" ret (*name) params = 0;
PS_HIP_API(PS_HIP_DEF_PTR)
#undef PS_HIP_DEF_PTR

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
   typedef HMODULE ps_dl_t;
   static ps_dl_t ps_dlopen_any(const char* const* names) {
       for (const char* const* n = names; *n; ++n) {
           ps_dl_t h = LoadLibraryA(*n);
           if (h) { fprintf(stderr, "[hip] loaded runtime: %s\n", *n); return h; }
       }
       return (ps_dl_t)0;
   }
   static void* ps_dlsym(ps_dl_t h, const char* s) { return (void*)GetProcAddress(h, s); }
   static const char* const PS_HIP_LIBS[] = {
       "amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64_5.dll", 0
   };
#else
#  include <dlfcn.h>
   typedef void* ps_dl_t;
   static ps_dl_t ps_dlopen_any(const char* const* names) {
       for (const char* const* n = names; *n; ++n) {
           ps_dl_t h = dlopen(*n, RTLD_NOW | RTLD_GLOBAL);
           if (h) { fprintf(stderr, "[hip] loaded runtime: %s\n", *n); return h; }
       }
       return (ps_dl_t)0;
   }
   static void* ps_dlsym(ps_dl_t h, const char* s) { return dlsym(h, s); }
   static const char* const PS_HIP_LIBS[] = {
       "libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6", 0
   };
#endif

extern "C" int psHipLoadRuntime(void) {
    static int state = -1;   /* -1 untried, 0 ok, 1 failed */
    if (state >= 0) return state ? -1 : 0;

    ps_dl_t h = ps_dlopen_any(PS_HIP_LIBS);
    if (!h) {
        fprintf(stderr, "[hip] no HIP runtime found (amdhip64) - GPU unavailable\n");
        state = 1; return -1;
    }

    int missing = 0;
#define PS_HIP_RESOLVE(ret, name, params) \
    name = (ret (*) params) ps_dlsym(h, #name); \
    if (!name) { fprintf(stderr, "[hip] missing symbol: %s\n", #name); ++missing; }
    PS_HIP_API(PS_HIP_RESOLVE)
#undef PS_HIP_RESOLVE

    if (missing) { fprintf(stderr, "[hip] %d symbols unresolved\n", missing); state = 1; return -1; }
    state = 0; return 0;
}

#endif /* PS_HIP_WIN */
