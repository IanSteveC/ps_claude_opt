/* mreal.h - model-real abstraction for the FP32-only experiment.
 *
 * PS_FP32 undefined (default): mreal == double, mreal4 == double4, and every
 * helper below folds to the identity. The build is the unchanged FP64 app.
 *
 * PS_FP32 defined: mreal == df64, a double-float ("float-float") pair that
 * emulates ~2^-46..2^-48 relative precision using only FP32 hardware ops
 * (Dekker/Knuth error-free transformations, QD-library style algorithms).
 * Every double in DEVICE code becomes a df64; the host keeps computing in
 * double and converts at the copy/launch/readback boundary (df64 and double
 * are both 8 bytes, so all buffer sizes and layouts are unchanged).
 *
 * Precision notes vs true FP64 (53-bit mantissa):
 *  - add/sub/mul: ~48-bit effective mantissa (accurate double-float variants)
 *  - div/rcp/sqrt: ~46-bit
 *  - sincos/exp/exp2/acos: ~44-46-bit absolute accuracy on the ranges this
 *    application uses (trig args are pre-reduced to (-2pi, 2pi) by the same
 *    round()-subtraction the FP64 code does)
 * The Levenberg-Marquardt fit tolerates this: the BOINC validator accepts
 * 0.1/0.1/0.5 absolute on period/rms/chi2 and the CPU<->GPU FP64 builds
 * already differ in the last printed digits.
 *
 * IMPORTANT build constraints for the PS_FP32 device code:
 *  - must be compiled WITHOUT --use_fast_math (approximate division would be
 *    harmless - seeds are Newton-corrected - but there is no reason to risk
 *    contraction/FTZ surprises inside the error-free transformations)
 *  - fmaf() must map to a hardware FMA (any modern GPU arch does)
 */
#pragma once

#ifndef PS_FP32

/* ------------------------------------------------------------------ */
/* FP64 build: plain doubles, zero-cost aliases                        */
/* ------------------------------------------------------------------ */
typedef double mreal;

#if defined(__CUDACC__) || defined(__HIPCC__)
typedef double4 mreal4;
#endif

/* host<->device boundary helpers fold to identity */
static inline double ps_real_to_double(double x) { return x; }
static inline double ps_real_from_double(double x) { return x; }
#define PS_SYMCPY_REAL(sym, src, nelem) \
    PS_SYMCPY(sym, src, (nelem) * sizeof(double))
#define PS_SYMCPY_REAL_ASYNC(sym, src, nelem, stream) \
    PS_SYMCPY_ASYNC(sym, src, (nelem) * sizeof(double), stream)

#else /* PS_FP32 */

/* ------------------------------------------------------------------ */
/* FP32 build: double-float emulation                                  */
/* ------------------------------------------------------------------ */

#include <math.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define DF64_FD __host__ __device__ __forceinline__
#else
#define DF64_FD inline
#endif

struct df64
{
  float hi, lo;   /* value = hi + lo, |lo| <= ulp(hi)/2 (normalized) */

  df64() = default;
  DF64_FD constexpr df64(float h, float l) : hi(h), lo(l) {}
  /* constexpr: double literals in device code are split at COMPILE TIME,
     so no FP64 instruction is emitted for them */
  DF64_FD constexpr df64(double d)
    : hi((float)d), lo((float)(d - (double)(float)d)) {}
  DF64_FD constexpr df64(float f) : hi(f), lo(0.0f) {}
  /* integer ctors avoid the double path: runtime ints must not emit FP64
     conversions (I2F.F64). hi rounds; the residual fits a float exactly. */
  DF64_FD constexpr df64(int i)
    : hi((float)i), lo((float)(int)(i - (long long)(float)i)) {}
  DF64_FD constexpr df64(long long i)
    : hi((float)i), lo((float)(i - (long long)(float)i)) {}
  DF64_FD constexpr df64(unsigned i)
    : hi((float)i), lo((float)(int)((long long)i - (long long)(float)i)) {}

  /* explicit only: keeps overload resolution unambiguous (no silent
     df64 -> double round trips in device code) */
  DF64_FD explicit constexpr operator float() const { return hi; }
  DF64_FD explicit operator double() const { return (double)hi + (double)lo; }
};

/* ---- error-free transformations (all pure FP32) ---- */
DF64_FD float df64_fmaf(float a, float b, float c)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __fmaf_rn(a, b, c);
#else
  return fmaf(a, b, c);
#endif
}

/* s + err == a + b exactly, no magnitude precondition (Knuth) */
DF64_FD float two_sum(float a, float b, float &err)
{
  float s = a + b;
  float bb = s - a;
  err = (a - (s - bb)) + (b - bb);
  return s;
}

/* s + err == a + b exactly, requires |a| >= |b| or a == 0 (Dekker) */
DF64_FD float quick_two_sum(float a, float b, float &err)
{
  float s = a + b;
  err = b - (s - a);
  return s;
}

/* contraction-proof multiply: __fmul_rn is documented to never be merged
   into an FMA. A plain `a * b` here lets nvcc (default -fmad=true) see
   fmaf(a, b, -(a*b)) in two_prod and fold the error term to ZERO, silently
   destroying the transformation (observed on CUDA 12.9 / sm_70). */
DF64_FD float df64_fmul(float a, float b)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __fmul_rn(a, b);
#else
  return a * b;
#endif
}

/* p + err == a * b exactly (FMA form) */
DF64_FD float two_prod(float a, float b, float &err)
{
  float p = df64_fmul(a, b);
  err = df64_fmaf(a, b, -p);
  return p;
}

/* ---- basic arithmetic (QD "accurate" variants) ---- */
DF64_FD df64 operator+(df64 a, df64 b)
{
  float s1, s2, t1, t2;
  s1 = two_sum(a.hi, b.hi, s2);
  /* inf/nan: the error terms would evaluate inf-inf = NaN and poison the
     pair; propagate the IEEE float result instead (as double would) */
  if(!isfinite(s1)) return df64(s1, 0.0f);
  t1 = two_sum(a.lo, b.lo, t2);
  s2 += t1;
  s1 = quick_two_sum(s1, s2, s2);
  s2 += t2;
  s1 = quick_two_sum(s1, s2, s2);
  return df64(s1, s2);
}

DF64_FD df64 operator-(df64 a)
{
  return df64(-a.hi, -a.lo);
}

DF64_FD df64 operator-(df64 a, df64 b)
{
  return a + (-b);
}

DF64_FD df64 operator*(df64 a, df64 b)
{
  float p1, p2;
  p1 = two_prod(a.hi, b.hi, p2);
  if(!isfinite(p1)) return df64(p1, 0.0f);   /* inf/nan propagation */
  p2 = df64_fmaf(a.hi, b.lo, df64_fmaf(a.lo, b.hi, p2));
  p1 = quick_two_sum(p1, p2, p2);
  return df64(p1, p2);
}

/* long division, two corrections (~2^-46) */
DF64_FD df64 operator/(df64 a, df64 b)
{
  float q1 = a.hi / b.hi;
  /* 0/x, x/0, inf and nan operands: keep the IEEE float quotient */
  if(!isfinite(q1) || q1 == 0.0f) return df64(q1, 0.0f);
  df64 r = a - b * df64(q1);
  float q2 = (r.hi + r.lo) / b.hi;
  r = r - b * df64(q2);
  float q3 = (r.hi + r.lo) / b.hi;
  float s1, s2;
  s1 = quick_two_sum(q1, q2, s2);
  df64 q = df64(s1, s2) + df64(q3);
  return q;
}

DF64_FD df64 &operator+=(df64 &a, df64 b) { a = a + b; return a; }
DF64_FD df64 &operator-=(df64 &a, df64 b) { a = a - b; return a; }
DF64_FD df64 &operator*=(df64 &a, df64 b) { a = a * b; return a; }
DF64_FD df64 &operator/=(df64 &a, df64 b) { a = a / b; return a; }

/* ---- comparisons (hi first, then lo; NaN in hi behaves like float NaN) ---- */
DF64_FD bool operator<(df64 a, df64 b)
{ return (a.hi < b.hi) || (a.hi == b.hi && a.lo < b.lo); }
DF64_FD bool operator>(df64 a, df64 b)
{ return (a.hi > b.hi) || (a.hi == b.hi && a.lo > b.lo); }
DF64_FD bool operator<=(df64 a, df64 b)
{ return (a.hi < b.hi) || (a.hi == b.hi && a.lo <= b.lo); }
DF64_FD bool operator>=(df64 a, df64 b)
{ return (a.hi > b.hi) || (a.hi == b.hi && a.lo >= b.lo); }
DF64_FD bool operator==(df64 a, df64 b)
{ return a.hi == b.hi && a.lo == b.lo; }
DF64_FD bool operator!=(df64 a, df64 b)
{ return a.hi != b.hi || a.lo != b.lo; }

/* ---- simple functions ---- */
DF64_FD df64 fabs(df64 a) { return (a.hi < 0.0f || (a.hi == 0.0f && a.lo < 0.0f)) ? -a : a; }
DF64_FD df64 fmin(df64 a, df64 b) { return (b < a) ? b : a; }
DF64_FD df64 fmax(df64 a, df64 b) { return (a < b) ? b : a; }
DF64_FD bool isnan(df64 a) { return ::isnan(a.hi) || ::isnan(a.lo); }

/* round-half-away-from-zero on the pair. An off-by-one at an exact .5
   boundary is tolerable for this app: round() feeds only 2pi/quadrant
   argument reductions where k+-1 shifts by a full period. */
DF64_FD df64 round(df64 a)
{
#if defined(__CUDA_ARCH__)
  float r = roundf(a.hi);
#else
  float r = ::roundf(a.hi);
#endif
  float d = (a.hi - r) + a.lo;  /* |a.hi - r| <= 0.5 -> exact (Sterbenz) */
  if(d > 0.5f)        r += 1.0f;
  else if(d < -0.5f)  r -= 1.0f;
  else if(d == 0.5f  && a.lo > 0.0f) r += 1.0f;
  else if(d == -0.5f && a.lo < 0.0f) r -= 1.0f;
  return df64(r, 0.0f);
}

typedef df64 mreal;

/* host <-> device boundary conversions */
static inline double ps_real_to_double(df64 x) { return (double)x.hi + (double)x.lo; }
static inline df64 ps_real_from_double(double d) { return df64(d); }

#if defined(__CUDACC__) || defined(__HIPCC__)

struct mreal4 { mreal x, y, z, w; };

/* ================================================================== */
/* device-only: math functions and intrinsic overloads                 */
/* ================================================================== */
#if defined(__CUDA_ARCH__) || defined(__CUDACC__) || defined(__HIPCC__)

#define DF64_D __device__ __forceinline__

/* reciprocal: fast seed + one full-precision Newton step in df64.
   e = 1 - b*r0 is computed exactly enough because b*r0 ~= 1 (two_prod). */
DF64_D df64 __drcp_rn(df64 b)
{
  float r0 = __frcp_rn(b.hi);
  /* rcp(+-0) = +-inf (used as an intentional sentinel), rcp(+-inf) = +-0,
     rcp(nan) = nan - all match the FP64 intrinsic; the Newton step would
     turn each of them into NaN via 0*inf */
  if(!isfinite(r0) || r0 == 0.0f) return df64(r0, 0.0f);
  df64 e = df64(1.0f) - b * df64(r0);
  df64 r = df64(r0) + df64(r0) * e;
  return r;
}

/* sqrt: fast seed + one Karp-Markstein style correction */
DF64_D df64 __dsqrt_rn(df64 a)
{
  float s0 = __fsqrt_rn(a.hi);
  /* 0, inf, nan (incl. sqrt of negative): keep the IEEE float result */
  if(!isfinite(s0) || s0 == 0.0f) return df64(s0, 0.0f);
  df64 d = a - df64(s0) * df64(s0);
  float corr = (d.hi + d.lo) * (0.5f * __frcp_rn(s0));
  float r1, r2;
  r1 = quick_two_sum(s0, corr, r2);
  return df64(r1, r2);
}

DF64_D df64 sqrt(df64 a) { return __dsqrt_rn(a); }

DF64_D df64 norm3d(df64 a, df64 b, df64 c)
{
  return __dsqrt_rn(a * a + b * b + c * c);
}

/* ---- sincos ----
   Input contract (same as the FP64 code): |x| < ~2pi + slack; we still
   handle |x| up to ~1e3 correctly via int quadrant arithmetic.
   Reduce by pi/2 (3-float Cody-Waite splitting of pi/2), then degree-17/16
   Taylor on |u| <= pi/4 evaluated in df64. */

#define DF64_PIO2_H  1.57079632679489661923132169163975144
#define DF64_2_OVER_PI 0.63661977236758134307553505349005745

DF64_D df64 df64_sin_taylor(df64 u)
{
  /* sin(u) = u * (1 - u^2/3! * (... Horner ...)), |u| <= pi/4 */
  df64 x2 = u * u;
  df64 p = df64(-1.0 / 355687428096000.0);            /* -1/17! */
  p = p * x2 + df64( 1.0 / 1307674368000.0);          /*  1/15! */
  p = p * x2 + df64(-1.0 / 6227020800.0);             /* -1/13! */
  p = p * x2 + df64( 1.0 / 39916800.0);               /*  1/11! */
  p = p * x2 + df64(-1.0 / 362880.0);                 /* -1/9!  */
  p = p * x2 + df64( 1.0 / 5040.0);                   /*  1/7!  */
  p = p * x2 + df64(-1.0 / 120.0);                    /* -1/5!  */
  p = p * x2 + df64( 1.0 / 6.0);                      /*  1/3!  */
  /* sin = u - u^3 * p_signflip: fold as u + u*x2*(-p) */
  return u + u * (x2 * (-p));
}

DF64_D df64 df64_cos_taylor(df64 u)
{
  df64 x2 = u * u;
  df64 p = df64( 1.0 / 20922789888000.0);             /*  1/16! */
  p = p * x2 + df64(-1.0 / 87178291200.0);            /* -1/14! */
  p = p * x2 + df64( 1.0 / 479001600.0);              /*  1/12! */
  p = p * x2 + df64(-1.0 / 3628800.0);                /* -1/10! */
  p = p * x2 + df64( 1.0 / 40320.0);                  /*  1/8!  */
  p = p * x2 + df64(-1.0 / 720.0);                    /* -1/6!  */
  p = p * x2 + df64( 1.0 / 24.0);                     /*  1/4!  */
  p = p * x2 + df64(-1.0 / 2.0);                      /* -1/2!  */
  return df64(1.0f) + x2 * p;
}

DF64_D void sincos(df64 x, df64 *s, df64 *c)
{
  if(!isfinite(x.hi))   /* (int) of nan/inf is UB; propagate nan */
    {
      df64 q = df64(x.hi - x.hi, 0.0f);
      *s = q; *c = q;
      return;
    }
  /* k = round(x / (pi/2)) */
  df64 kk = round(x * df64(DF64_2_OVER_PI));
  int k = (int)kk.hi;
  float kf = kk.hi;

  /* u = x - k*pi/2 with a 3-float pi/2 (48+ extra bits) */
  const float p1 = 1.57079632679489661923f;              /* (float)pi/2      */
  const float p2 = (float)(DF64_PIO2_H - (double)(float)DF64_PIO2_H);
  const float p3 = (float)(DF64_PIO2_H - (double)(float)DF64_PIO2_H
                           - (double)(float)(DF64_PIO2_H - (double)(float)DF64_PIO2_H));
  float e1, e2;
  float h1 = two_prod(kf, p1, e1);
  float h2 = two_prod(kf, p2, e2);
  df64 u = x - df64(h1, e1);
  u = u - df64(h2, e2);
  u = u - df64(kf * p3);

  df64 st = df64_sin_taylor(u);
  df64 ct = df64_cos_taylor(u);

  int q = k & 3;             /* k may be negative: adjust to 0..3 */
  if(q < 0) q += 4;          /* (k&3 of negative int in C++ is impl-trunc; be safe) */
  switch(q)
    {
    case 0: *s = st;  *c = ct;  break;
    case 1: *s = ct;  *c = -st; break;
    case 2: *s = -st; *c = -ct; break;
    default:*s = -ct; *c = st;  break;
    }
}

/* ---- exp family ----
   exp(x) = 2^k * exp(r), r = x - k*ln2, |r| <= ln2/2, Taylor deg 13.
   Clamped: overflow saturates to a huge FINITE value (keeps LM chisq
   comparisons well-defined without inf-inf NaNs), underflow to 0. */

#define DF64_LN2_H 0.69314718055994530941723212145818

DF64_D df64 df64_exp_taylor(df64 r)
{
  df64 p = df64(1.0 / 6227020800.0);                  /* 1/13! */
  p = p * r + df64(1.0 / 479001600.0);                /* 1/12! */
  p = p * r + df64(1.0 / 39916800.0);                 /* 1/11! */
  p = p * r + df64(1.0 / 3628800.0);                  /* 1/10! */
  p = p * r + df64(1.0 / 362880.0);                   /* 1/9! */
  p = p * r + df64(1.0 / 40320.0);                    /* 1/8! */
  p = p * r + df64(1.0 / 5040.0);                     /* 1/7! */
  p = p * r + df64(1.0 / 720.0);                      /* 1/6! */
  p = p * r + df64(1.0 / 120.0);                      /* 1/5! */
  p = p * r + df64(1.0 / 24.0);                       /* 1/4! */
  p = p * r + df64(1.0 / 6.0);                        /* 1/3! */
  p = p * r + df64(0.5);
  p = p * r + df64(1.0);
  p = p * r + df64(1.0);
  return p;
}

DF64_D df64 df64_ldexp(df64 a, int k)
{
  return df64(ldexpf(a.hi, k), ldexpf(a.lo, k));
}

DF64_D df64 exp(df64 x)
{
  if(isnan(x.hi)) return x;
  /* clamp far above any legitimate value in this app (exp of fitted
     log-coefficients, O(1)) but low enough that downstream products and
     2000-point accumulations cannot overflow df64's float exponent range.
     Oversized trial steps produce a huge-but-finite chisq and are rejected
     by the LM loop exactly as in FP64. */
  if(x.hi >  25.0f) return df64(7.2e10f);
  if(x.hi < -87.0f) return df64(0.0f);
  float kf = roundf(x.hi * 1.4426950408889634f);
  int k = (int)kf;
  const float l1 = 0.693147182464599609375f;             /* (float)ln2 */
  const float l2 = (float)(DF64_LN2_H - (double)0.693147182464599609375f);
  const float l3 = (float)(DF64_LN2_H - (double)0.693147182464599609375f
                           - (double)(float)(DF64_LN2_H - (double)0.693147182464599609375f));
  float e1, e2;
  float h1 = two_prod(kf, l1, e1);
  float h2 = two_prod(kf, l2, e2);
  df64 r = x - df64(h1, e1);
  r = r - df64(h2, e2);
  r = r - df64(kf * l3);
  return df64_ldexp(df64_exp_taylor(r), k);
}

DF64_D df64 exp2(df64 x)
{
  if(isnan(x.hi)) return x;
  if(x.hi >  36.0f) return df64(6.9e10f);  /* see exp() clamp comment */
  if(x.hi < -125.0f) return df64(0.0f);
  float kf = roundf(x.hi);
  int k = (int)kf;
  df64 r = x - df64(kf);            /* exact */
  df64 rl = r * df64(DF64_LN2_H);   /* |rl| <= ln2/2 */
  return df64_ldexp(df64_exp_taylor(rl), k);
}

/* ---- acos: float seed + one Newton step on cos(y) = x in df64 ----
   Callers clamp x to [-1,1] already (solar-phase dot product). */
DF64_D df64 acos(df64 x)
{
  float xs = x.hi;
  if(xs >  1.0f) xs =  1.0f;
  if(xs < -1.0f) xs = -1.0f;
  float y0 = acosf(xs);
  df64 s, c;
  sincos(df64(y0), &s, &c);
  /* near x = +-1, sin(y) ~ 0: acosf's absolute error there is already
     far below anything this app can sense; skip the correction */
  if(fabsf(s.hi) < 1e-6f) return df64(y0);
  df64 y = df64(y0) + (c - x) / s;
  return y;
}

/* ---- load/store/shuffle intrinsic overloads (df64 == float2 layout) ---- */
#ifndef __HIP_PLATFORM_AMD__
/* CUDA: the cache-hint load/store builtins are real functions; provide df64
   overloads that move the pair as one float2 (128-bit vector) transaction. */
DF64_D df64 __ldg(const df64 *p)
{
  float2 v = __ldg((const float2 *)p);
  return df64(v.x, v.y);
}

DF64_D df64 __ldca(const df64 *p)
{
  float2 v = __ldca((const float2 *)p);
  return df64(v.x, v.y);
}

DF64_D df64 __ldcs(const df64 *p)
{
  float2 v = __ldcs((const float2 *)p);
  return df64(v.x, v.y);
}

DF64_D void __stwb(df64 *p, df64 v)
{
  __stwb((float2 *)p, make_float2(v.hi, v.lo));
}

DF64_D void __stcs(df64 *p, df64 v)
{
  __stcs((float2 *)p, make_float2(v.hi, v.lo));
}

DF64_D df64 __shfl_down_sync(unsigned mask, df64 v, unsigned delta)
{
  return df64(__shfl_down_sync(mask, v.hi, delta),
              __shfl_down_sync(mask, v.lo, delta));
}

DF64_D df64 __shfl_xor_sync(unsigned mask, df64 v, int lanemask)
{
  return df64(__shfl_xor_sync(mask, v.hi, lanemask),
              __shfl_xor_sync(mask, v.lo, lanemask));
}

DF64_D df64 __shfl_sync(unsigned mask, df64 v, int lane)
{
  return df64(__shfl_sync(mask, v.hi, lane),
              __shfl_sync(mask, v.lo, lane));
}

#else /* __HIP_PLATFORM_AMD__ */
/* HIP: hip_compat.h already redefined __ldg/__ldca/__stwb/... as the ps_ld/
   ps_st templates (plain loads/stores) - those handle a df64 by value copy,
   so no df64 load/store overload is needed. The shuffles, however, route
   through hip_compat's ps_shfl_* templates which call __shfl_*(v, ..., 32);
   AMD's __shfl_* have no df64 form. Provide non-template ps_shfl_* overloads
   for df64 (a non-template is preferred over the template for df64 args) that
   shuffle the two floats and keep the width pinned to 32 like the scalar
   path. */
DF64_D df64 ps_shfl_down(df64 v, unsigned d)
{
  return df64(__shfl_down(v.hi, d, PS_WARP), __shfl_down(v.lo, d, PS_WARP));
}
DF64_D df64 ps_shfl_xor(df64 v, int m)
{
  return df64(__shfl_xor(v.hi, m, PS_WARP), __shfl_xor(v.lo, m, PS_WARP));
}
DF64_D df64 ps_shfl(df64 v, int s)
{
  return df64(__shfl(v.hi, s, PS_WARP), __shfl(v.lo, s, PS_WARP));
}
#endif /* __HIP_PLATFORM_AMD__ */

#endif /* device parts */
#endif /* __CUDACC__ / __HIPCC__ */

/* host-side conversion helper: convert a double array into a heap df64
   buffer for the SYMCPY shims (freed by the caller). Host-only code, but
   left unguarded so nvcc's device pass can still parse the call sites. */
#include <stdlib.h>
static inline df64 *ps_real_convert_alloc(const double *src, size_t nelem)
{
  df64 *buf = (df64 *)malloc(nelem * sizeof(df64));
  for(size_t i = 0; i < nelem; i++)
    buf[i] = ps_real_from_double(src[i]);
  return buf;
}

/* portable spellings so the same shim serves every host TU:
   - nvcc Linux / MinGW driver-API (PS_DRIVER_API): cudaError_t (cuda_iface.h
     typedefs it to CUresult on the Windows path)
   - hipcc Linux native: __HIP_PLATFORM_AMD__ -> hipError_t
   - MinGW HIP module/win (PS_HIP_MODULE / PS_HIP_WIN): the hand-declared shim
     provides hipError_t + hipStreamSynchronize but does NOT define
     __HIP_PLATFORM_AMD__, so key off the HIP-module macros too. */
#if defined(__HIP_PLATFORM_AMD__) || defined(PS_HIP_MODULE) || defined(PS_HIP_WIN)
#define PS_REAL_ERR_T       hipError_t
#define PS_REAL_STREAM_SYNC hipStreamSynchronize
#else
#define PS_REAL_ERR_T       cudaError_t
#define PS_REAL_STREAM_SYNC cudaStreamSynchronize
#endif

/* symbol-copy shims: convert double -> df64 elementwise, then copy.
   MemcpyToSymbolAsync from pageable memory returns only after the source has
   been staged on CUDA, but NOT on HIP - so both shims synchronize the stream
   before freeing the temp (a handful of copies per workunit; cost is nil). */
#define PS_SYMCPY_REAL(sym, src, nelem)                                  \
    ([&]() {                                                             \
        df64 *ps__tmp = ps_real_convert_alloc((const double *)(src), (nelem)); \
        PS_REAL_ERR_T ps__e = PS_SYMCPY(sym, ps__tmp, (nelem) * sizeof(df64)); \
        free(ps__tmp);                                                   \
        return ps__e;                                                    \
    }())

#define PS_SYMCPY_REAL_ASYNC(sym, src, nelem, stream)                    \
    ([&]() {                                                             \
        df64 *ps__tmp = ps_real_convert_alloc((const double *)(src), (nelem)); \
        PS_REAL_ERR_T ps__e = PS_SYMCPY_ASYNC(sym, ps__tmp, (nelem) * sizeof(df64), stream); \
        PS_REAL_STREAM_SYNC(stream);                                     \
        free(ps__tmp);                                                   \
        return ps__e;                                                    \
    }())

#endif /* PS_FP32 */
