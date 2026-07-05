/* Copyright (c) 2026 Xiph.Org Foundation */
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:

   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if defined(OPUS_USE_PFA_MDCT)

#include "mdct.h"
#include "kiss_fft.h"
#include "_kiss_fft_guts.h"
#include "stack_alloc.h"
#include "mathops.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef kiss_fft_cpx cpx;

static const int celt_tx_pfa_P[15] = {2, 5, 7, 14, 4, 1, 13, 3, 10, 12, 0, 9, 11, 6, 8};

#ifdef FIXED_POINT
#ifdef ENABLE_QEXT
#define SIN_2PI_3  1859775393L /* 0.866025388 * 2147483648 */
#define COS_2PI_5  663608942L  /* 0.309017003 * 2147483648 */
#define SIN_2PI_5  2042378317L /* 0.951056540 * 2147483648 */
#define COS_4PI_5  1737350766L /* 0.809016994 * 2147483648 */
#define SIN_4PI_5  1262259218L /* 0.587785252 * 2147483648 */
#define HALF_VAL   1073741824L /* 0.5 * 2147483648 */
#else
#define SIN_2PI_3  28378  /* 0.866025388 * 32768 */
#define COS_2PI_5  10126  /* 0.309017003 * 32768 */
#define SIN_2PI_5  31164  /* 0.951056540 * 32768 */
#define COS_4PI_5  26509  /* 0.809016994 * 32768 */
#define SIN_4PI_5  19261  /* 0.587785252 * 32768 */
#define HALF_VAL   16384  /* 0.5 * 32768 */
#endif

static void pfa_downshift(cpx *x, int N, int *total, int step) {
   int i;
   int shift = IMIN(step, *total);
   *total -= shift;
   if (shift == 1) {
      for (i = 0; i < N; i++) {
         x[i].r = SHR32(x[i].r, 1);
         x[i].i = SHR32(x[i].i, 1);
      }
   } else if (shift > 0) {
      for (i = 0; i < N; i++) {
         x[i].r = PSHR32(x[i].r, shift);
         x[i].i = PSHR32(x[i].i, shift);
      }
   }
}
#define PFA_DOWNSHIFT(x, N, total, step) pfa_downshift(x, N, total, step)
#else
#define SIN_2PI_3  0.866025388f
#define COS_2PI_5  0.309017003f
#define SIN_2PI_5  0.951056540f
#define COS_4PI_5  0.809016994f
#define SIN_4PI_5  0.587785252f
#define HALF_VAL   0.5f
#define PFA_DOWNSHIFT(x, N, total, step) (void)(x); (void)(N); (void)(total); (void)(step)
#endif

typedef struct OpusTXContext OpusTXContext;
typedef void (*opus_tx_fn)(const OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift));

struct OpusTXContext {
   opus_int32 len;
   opus_int32 inv;
   const opus_int16 *map;
   const void *exp;
   void *tmp;
   const struct OpusTXContext *sub;
   opus_tx_fn fn;
};

#include <stddef.h>
#include "celt_tx_tables.h"
static OPUS_INLINE int split_radix_permutation(int i, int len, int inv)
{
    len >>= 1;
    if (len <= 1)
        return i & 1;
    if (!(i & len))
        return split_radix_permutation(i, len, inv) * 2;
    len >>= 1;
    return split_radix_permutation(i, len, inv) * 4 + 1 - 2*(!(i & len) ^ inv);
}

static OPUS_INLINE const opus_int16 *get_sr_perm_table(int M) {
   static const opus_int16 p4[4]   = { 0, 2, 1, 3 };
   static const opus_int16 p8[8]   = { 0, 4, 2, 6, 1, 5, 7, 3 };
   static const opus_int16 p16[16] = { 0, 8, 4, 12, 2, 10, 14, 6, 1, 9, 5, 13, 15, 7, 3, 11 };
   static const opus_int16 p32[32] = { 0, 16, 8, 24, 4, 20, 28, 12, 2, 18, 10, 26, 30, 14, 6, 22, 1, 17, 9, 25, 5, 21, 29, 13, 31, 15, 7, 23, 3, 19, 27, 11 };
   static const opus_int16 p64[64] = { 0, 32, 16, 48, 8, 40, 56, 24, 4, 36, 20, 52, 60, 28, 12, 44, 2, 34, 18, 50, 10, 42, 58, 26, 62, 30, 14, 46, 6, 38, 54, 22, 1, 33, 17, 49, 9, 41, 57, 25, 5, 37, 21, 53, 61, 29, 13, 45, 63, 31, 15, 47, 7, 39, 55, 23, 3, 35, 19, 51, 59, 27, 11, 43 };
   if (M == 4) return p4;
   if (M == 8) return p8;
   if (M == 16) return p16;
   if (M == 32) return p32;
   if (M == 64) return p64;
   return NULL;
}

static OPUS_INLINE int sr_perm_map(int k, int M) {
   const opus_int16 *perm = get_sr_perm_table(M);
   if (perm != NULL) return perm[k];
   return -split_radix_permutation(k, M, 0) & (M - 1);
}

static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift));
static void get_pfa_crt_params(int M, int *K3, int *K4);


#ifndef FIXED_POINT
static OPUS_INLINE const float *get_p2_twiddle_table(int N, int *tab_N_out) {
   if (N <= 32) {
      *tab_N_out = 32;
      return celt_tx_tab_32_float;
   }
   *tab_N_out = N;
   if (N == 64) return celt_tx_tab_64_float;
#if defined(CUSTOM_MODES) || defined(ENABLE_DRED) || defined(ENABLE_DEEP_PLC)
   if (N == 128) return celt_tx_tab_128_float;
   if (N == 256) return celt_tx_tab_256_float;
   return celt_tx_tab_512_float;
#else
   return NULL;
#endif
}

static OPUS_INLINE cpx lookup_p2_twiddle_float(const float *tab, int tab_N, int N, int j) {
   cpx w;
   int idx = j * (tab_N / N);
   int quarter = tab_N >> 2;
   int half = tab_N >> 1;
   if (idx < quarter) {
      w.r = tab[idx];
      w.i = tab[quarter - idx];
   } else if (idx > quarter) {
      w.r = -tab[half - idx];
      w.i = tab[idx - quarter];
   } else {
      w.r = 0.0f;
      w.i = 1.0f;
   }
   return w;
}
#else /* FIXED_POINT */
# ifdef ENABLE_QEXT
static OPUS_INLINE const opus_int32 *get_p2_twiddle_table(int N, int *tab_N_out) {
   if (N <= 32) {
      *tab_N_out = 32;
      return celt_tx_tab_32_fixed32;
   }
   *tab_N_out = N;
   if (N == 64) return celt_tx_tab_64_fixed32;
#if defined(CUSTOM_MODES) || defined(ENABLE_DRED) || defined(ENABLE_DEEP_PLC)
   if (N == 128) return celt_tx_tab_128_fixed32;
   if (N == 256) return celt_tx_tab_256_fixed32;
   return celt_tx_tab_512_fixed32;
#else
   return NULL;
#endif
}

static OPUS_INLINE cpx lookup_p2_twiddle_fixed32(const opus_int32 *tab, int tab_N, int N, int j) {
   cpx w;
   int idx = j * (tab_N / N);
   int quarter = tab_N >> 2;
   int half = tab_N >> 1;
   if (idx < quarter) {
      w.r = tab[idx];
      w.i = tab[quarter - idx];
   } else if (idx > quarter) {
      w.r = -tab[half - idx];
      w.i = tab[idx - quarter];
   } else {
      w.r = 0;
      w.i = Q31ONE;
   }
   return w;
}
# else /* !ENABLE_QEXT */
static OPUS_INLINE const opus_int16 *get_p2_twiddle_table(int N, int *tab_N_out) {
   if (N <= 32) {
      *tab_N_out = 32;
      return celt_tx_tab_32_fixed16;
   }
   *tab_N_out = N;
   if (N == 64) return celt_tx_tab_64_fixed16;
#if defined(CUSTOM_MODES) || defined(ENABLE_DRED) || defined(ENABLE_DEEP_PLC)
   if (N == 128) return celt_tx_tab_128_fixed16;
   if (N == 256) return celt_tx_tab_256_fixed16;
   return celt_tx_tab_512_fixed16;
#else
   return NULL;
#endif
}

static OPUS_INLINE cpx lookup_p2_twiddle_fixed16(const opus_int16 *tab, int tab_N, int N, int j) {
   cpx w;
   int idx = j * (tab_N / N);
   int quarter = tab_N >> 2;
   int half = tab_N >> 1;
   if (idx < quarter) {
      w.r = tab[idx];
      w.i = tab[quarter - idx];
   } else if (idx > quarter) {
      w.r = -tab[half - idx];
      w.i = tab[idx - quarter];
   } else {
      w.r = 0;
      w.i = Q15ONE;
   }
   return w;
}
# endif
#endif

static OPUS_INLINE cpx get_p2_twiddle(int N, int j) {
   int tab_N;
#ifndef FIXED_POINT
   const float *tab = get_p2_twiddle_table(N, &tab_N);
   return lookup_p2_twiddle_float(tab, tab_N, N, j);
#else
# ifdef ENABLE_QEXT
   const opus_int32 *tab = get_p2_twiddle_table(N, &tab_N);
   return lookup_p2_twiddle_fixed32(tab, tab_N, N, j);
# else
   const opus_int16 *tab = get_p2_twiddle_table(N, &tab_N);
   return lookup_p2_twiddle_fixed16(tab, tab_N, N, j);
# endif
#endif
}

#ifndef FIXED_POINT
#define BF(x, y, a, b)  \
    do {                \
        x = (a) - (b);  \
        y = (a) + (b);  \
    } while (0)

#define CMUL(dre, dim, are, aim, bre, bim)      \
    do {                                        \
        (dre) = (are) * (bre) - (aim) * (bim);  \
        (dim) = (are) * (bim) + (aim) * (bre);  \
    } while (0)

#define SMUL(dre, dim, are, aim, bre, bim)      \
    do {                                        \
        (dre) = (are) * (bre) - (aim) * (bim);  \
        (dim) = (are) * (bim) - (aim) * (bre);  \
    } while (0)

#define BUTTERFLIES(a0, a1, a2, a3)            \
    do {                                       \
        r0=a0.r;                               \
        i0=a0.i;                               \
        r1=a1.r;                               \
        i1=a1.i;                               \
        BF(t3, t5, t5, t1);                    \
        BF(a2.r, a0.r, r0, t5);                \
        BF(a3.i, a1.i, i1, t3);                \
        BF(t4, t6, t2, t6);                    \
        BF(a3.r, a1.r, r1, t4);                \
        BF(a2.i, a0.i, i0, t6);                \
    } while (0)

#define TRANSFORM(a0, a1, a2, a3, wre, wim)    \
    do {                                       \
        CMUL(t1, t2, a2.r, a2.i, wre, -wim);   \
        CMUL(t5, t6, a3.r, a3.i, wre,  wim);   \
        BUTTERFLIES(a0, a1, a2, a3);           \
    } while (0)

static inline void celt_tx_fft_sr_combine_float(cpx *z, const float *cos, int len)
{
    int o1 = 2*len;
    int o2 = 4*len;
    int o3 = 6*len;
    const float *wim = cos + o1 - 7;
    float t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    int i;

    for (i = 0; i < len; i += 4) {
        TRANSFORM(z[0], z[o1 + 0], z[o2 + 0], z[o3 + 0], cos[0], wim[7]);
        TRANSFORM(z[2], z[o1 + 2], z[o2 + 2], z[o3 + 2], cos[2], wim[5]);
        TRANSFORM(z[4], z[o1 + 4], z[o2 + 4], z[o3 + 4], cos[4], wim[3]);
        TRANSFORM(z[6], z[o1 + 6], z[o2 + 6], z[o3 + 6], cos[6], wim[1]);

        TRANSFORM(z[1], z[o1 + 1], z[o2 + 1], z[o3 + 1], cos[1], wim[6]);
        TRANSFORM(z[3], z[o1 + 3], z[o2 + 3], z[o3 + 3], cos[3], wim[4]);
        TRANSFORM(z[5], z[o1 + 5], z[o2 + 5], z[o3 + 5], cos[5], wim[2]);
        TRANSFORM(z[7], z[o1 + 7], z[o2 + 7], z[o3 + 7], cos[7], wim[0]);

        z   += 2*4;
        cos += 2*4;
        wim -= 2*4;
    }
}

static void celt_tx_fft2_float(cpx *dst, const cpx *src)
{
    cpx tmp;
    BF(tmp.r, dst[0].r, src[0].r, src[1].r);
    BF(tmp.i, dst[0].i, src[0].i, src[1].i);
    dst[1] = tmp;
}

static void celt_tx_fft4_float(cpx *dst, const cpx *src)
{
    float t1, t2, t3, t4, t5, t6, t7, t8;

    BF(t3, t1, src[0].r, src[1].r);
    BF(t8, t6, src[3].r, src[2].r);
    BF(dst[2].r, dst[0].r, t1, t6);
    BF(t4, t2, src[0].i, src[1].i);
    BF(t7, t5, src[2].i, src[3].i);
    BF(dst[3].i, dst[1].i, t4, t8);
    BF(dst[3].r, dst[1].r, t3, t7);
    BF(dst[2].i, dst[0].i, t2, t5);
}

static void celt_tx_fft8_float(cpx *dst, const cpx *src)
{
    float t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    const float cos = 0.7071067812f; /* cos(pi/4) */

    celt_tx_fft4_float(dst, src);

    BF(t1, dst[5].r, src[4].r, -src[5].r);
    BF(t2, dst[5].i, src[4].i, -src[5].i);
    BF(t5, dst[7].r, src[6].r, -src[7].r);
    BF(t6, dst[7].i, src[6].i, -src[7].i);

    BUTTERFLIES(dst[0], dst[2], dst[4], dst[6]);
    TRANSFORM(dst[1], dst[3], dst[5], dst[7], cos, cos);
}

static void celt_tx_fft16_float(cpx *dst, const cpx *src)
{
    float t1, t2, t3, t4, t5, t6, r0, i0, r1, i1;
    float cos_16_1 = 0.9238795325f; /* cos(pi/8) */
    float cos_16_2 = 0.7071067812f; /* cos(2*pi/8) */
    float cos_16_3 = 0.3826834324f; /* cos(3*pi/8) */

    celt_tx_fft8_float(dst +  0, src +  0);
    celt_tx_fft4_float(dst +  8, src +  8);
    celt_tx_fft4_float(dst + 12, src + 12);

    t1 = dst[ 8].r;
    t2 = dst[ 8].i;
    t5 = dst[12].r;
    t6 = dst[12].i;
    BUTTERFLIES(dst[0], dst[4], dst[8], dst[12]);

    TRANSFORM(dst[ 2], dst[ 6], dst[10], dst[14], cos_16_2, cos_16_2);
    TRANSFORM(dst[ 1], dst[ 5], dst[ 9], dst[13], cos_16_1, cos_16_3);
    TRANSFORM(dst[ 3], dst[ 7], dst[11], dst[15], cos_16_3, cos_16_1);
}

static void celt_tx_fft32_float(cpx *dst, const cpx *src)
{
    const float *cos = celt_tx_tab_32_float;
    celt_tx_fft16_float(dst, src);
    celt_tx_fft8_float(dst + 16, src + 16);
    celt_tx_fft8_float(dst + 24, src + 24);
    celt_tx_fft_sr_combine_float(dst, cos, 4);
}

static void celt_tx_fft64_float(cpx *dst, const cpx *src)
{
    const float *cos = celt_tx_tab_64_float;
    celt_tx_fft32_float(dst, src);
    celt_tx_fft16_float(dst + 32, src + 32);
    celt_tx_fft16_float(dst + 48, src + 48);
    celt_tx_fft_sr_combine_float(dst, cos, 8);
}

#if defined(CUSTOM_MODES) || defined(ENABLE_DRED) || defined(ENABLE_DEEP_PLC)
static void celt_tx_fft128_float(cpx *dst, const cpx *src)
{
    const float *cos = celt_tx_tab_128_float;
    celt_tx_fft64_float(dst, src);
    celt_tx_fft32_float(dst + 64, src + 64);
    celt_tx_fft32_float(dst + 96, src + 96);
    celt_tx_fft_sr_combine_float(dst, cos, 16);
}

static void celt_tx_fft256_float(cpx *dst, const cpx *src)
{
    const float *cos = celt_tx_tab_256_float;
    celt_tx_fft128_float(dst, src);
    celt_tx_fft64_float(dst + 128, src + 128);
    celt_tx_fft64_float(dst + 192, src + 192);
    celt_tx_fft_sr_combine_float(dst, cos, 32);
}

static void celt_tx_fft512_float(cpx *dst, const cpx *src)
{
    const float *cos = celt_tx_tab_512_float;
    celt_tx_fft256_float(dst, src);
    celt_tx_fft128_float(dst + 256, src + 256);
    celt_tx_fft128_float(dst + 384, src + 384);
    celt_tx_fft_sr_combine_float(dst, cos, 64);
}
#endif

static void celt_tx_fft_sr_c(cpx *dst, const cpx *src, int N)
{
   switch (N) {
      case   2: celt_tx_fft2_float(dst, src); break;
      case   4: celt_tx_fft4_float(dst, src); break;
      case   8: celt_tx_fft8_float(dst, src); break;
      case  16: celt_tx_fft16_float(dst, src); break;
      case  32: celt_tx_fft32_float(dst, src); break;
      case  64: celt_tx_fft64_float(dst, src); break;
#if defined(CUSTOM_MODES) || defined(ENABLE_DRED) || defined(ENABLE_DEEP_PLC)
      case 128: celt_tx_fft128_float(dst, src); break;
      case 256: celt_tx_fft256_float(dst, src); break;
      case 512: celt_tx_fft512_float(dst, src); break;
#endif
      default: celt_assert2(0, "Unsupported Split-Radix FFT size");
   }
}

#undef BF
#undef CMUL
#undef SMUL
#undef BUTTERFLIES
#undef TRANSFORM
#endif /* !FIXED_POINT */

#ifdef FIXED_POINT
static void celt_tx_fft_p2_c(cpx *out, const cpx *in, int N ARG_FIXED(int *downshift_ptr)) {
   int i, j, len, half_len;
   int log2N;
   SAVE_STACK;
   VARDECL(cpx, buf);
   ALLOC(buf, N, cpx);

   /* Bit-reversal permutation */
   log2N = 0;
   for (i = N >> 1; i > 0; i >>= 1) log2N++;

   for (i = 0; i < N; i++) {
      int rev = 0;
      for (j = 0; j < log2N; j++) {
         if ((i >> j) & 1)
            rev |= (1 << (log2N - 1 - j));
      }
      buf[rev] = in[i];
   }

   /* Cooley-Tukey butterfly stages using twiddles from cosine tables */
   /* We use C_MULC to multiply by conjugate twiddles, computing DFT directly */
   {
      int tab_N;
# ifdef ENABLE_QEXT
      const opus_int32 *tab = get_p2_twiddle_table(N, &tab_N);
# else
      const opus_int16 *tab = get_p2_twiddle_table(N, &tab_N);
# endif
      for (len = 2; len <= N; len <<= 1) {
         half_len = len >> 1;
         PFA_DOWNSHIFT(buf, N, downshift_ptr, 1);
         for (i = 0; i < N; i += len) {
            /* j = 0: twiddle (1, 0) -- no multiplication, exact precision */
            {
               cpx u = buf[i];
               cpx v = buf[i + half_len];
               C_ADD(buf[i], u, v);
               C_SUB(buf[i + half_len], u, v);
            }
            for (j = 1; j < half_len; j++) {
# ifdef ENABLE_QEXT
               cpx w = lookup_p2_twiddle_fixed32(tab, tab_N, len, j);
# else
               cpx w = lookup_p2_twiddle_fixed16(tab, tab_N, len, j);
# endif
               cpx u = buf[i + j];
               cpx t;
               C_MULC(t, buf[i + j + half_len], w);
               C_ADD(buf[i + j], u, t);
               C_SUB(buf[i + j + half_len], u, t);
            }
         }
      }
   }
   PFA_DOWNSHIFT(buf, N, downshift_ptr, *downshift_ptr);

   for (i = 0; i < N; i++) out[i] = buf[i];
   RESTORE_STACK;
}
#endif

/*
 * 15-point Good-Thomas Prime Factor Algorithm (PFA) DFT core.
 * Mathematically identical to FFT15_CORE from celt_tx_neon.S.
 */
static void winograd_fft3(const cpx *in, cpx *out) {
   kiss_fft_scalar r_sum12, r_diff12, i_sum12, i_diff12;
   kiss_fft_scalar t1_r, t1_i, t2_r, t2_i;

   r_sum12  = ADD32_ovflw(in[1].r, in[2].r);
   r_diff12 = SUB32_ovflw(in[1].r, in[2].r);
   i_sum12  = ADD32_ovflw(in[1].i, in[2].i);
   i_diff12 = SUB32_ovflw(in[1].i, in[2].i);

   out[0].r = ADD32_ovflw(in[0].r, r_sum12);
   out[0].i = ADD32_ovflw(in[0].i, i_sum12);

   t1_r = S_MUL(i_diff12, SIN_2PI_3);
   t1_i = S_MUL(r_diff12, SIN_2PI_3);
   t2_r = S_MUL(r_sum12, HALF_VAL);
   t2_i = S_MUL(i_sum12, HALF_VAL);

   out[1].r = ADD32_ovflw(SUB32_ovflw(in[0].r, t2_r), t1_r);
   out[1].i = SUB32_ovflw(SUB32_ovflw(in[0].i, t2_i), t1_i);

   out[2].r = SUB32_ovflw(SUB32_ovflw(in[0].r, t2_r), t1_r);
   out[2].i = ADD32_ovflw(SUB32_ovflw(in[0].i, t2_i), t1_i);
}

static void decl_fft5(const cpx *in, int r3, cpx *out, int stride) {
   cpx dc = in[0];
   kiss_fft_scalar r_sum14, r_diff14, i_sum14, i_diff14;
   kiss_fft_scalar r_sum23, r_diff23, i_sum23, i_diff23;
   kiss_fft_scalar r_t4, r_t0, i_t4, i_t0;
   kiss_fft_scalar r_t5, r_t1, i_t5, i_t1;
   int idx0, idx1, idx2, idx3, idx4;

   r_diff14 = SUB32_ovflw(in[1].r, in[4].r);
   r_sum14  = ADD32_ovflw(in[1].r, in[4].r);
   i_diff14 = SUB32_ovflw(in[1].i, in[4].i);
   i_sum14  = ADD32_ovflw(in[1].i, in[4].i);

   r_diff23 = SUB32_ovflw(in[2].r, in[3].r);
   r_sum23  = ADD32_ovflw(in[2].r, in[3].r);
   i_diff23 = SUB32_ovflw(in[2].i, in[3].i);
   i_sum23  = ADD32_ovflw(in[2].i, in[3].i);

   idx0 = (5 * r3) % 15;
   if (idx0 < 0) idx0 += 15;
   out[idx0 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_sum14, r_sum23));
   out[idx0 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_sum14, i_sum23));

   r_t4 = SUB32_ovflw(S_MUL(r_sum14, COS_2PI_5), S_MUL(r_sum23, COS_4PI_5));
   r_t0 = SUB32_ovflw(S_MUL(r_sum23, COS_2PI_5), S_MUL(r_sum14, COS_4PI_5));
   i_t4 = SUB32_ovflw(S_MUL(i_sum14, COS_2PI_5), S_MUL(i_sum23, COS_4PI_5));
   i_t0 = SUB32_ovflw(S_MUL(i_sum23, COS_2PI_5), S_MUL(i_sum14, COS_4PI_5));

   r_t5 = ADD32_ovflw(S_MUL(i_diff14, SIN_2PI_5), S_MUL(i_diff23, SIN_4PI_5));
   r_t1 = SUB32_ovflw(S_MUL(i_diff14, SIN_4PI_5), S_MUL(i_diff23, SIN_2PI_5));
   i_t5 = NEG32_ovflw(ADD32_ovflw(S_MUL(r_diff14, SIN_2PI_5), S_MUL(r_diff23, SIN_4PI_5)));
   i_t1 = SUB32_ovflw(S_MUL(r_diff23, SIN_2PI_5), S_MUL(r_diff14, SIN_4PI_5));

   idx1 = (5 * r3 + 3) % 15; if (idx1 < 0) idx1 += 15;
   idx2 = (5 * r3 + 6) % 15; if (idx2 < 0) idx2 += 15;
   idx3 = (5 * r3 + 9) % 15; if (idx3 < 0) idx3 += 15;
   idx4 = (5 * r3 + 12) % 15; if (idx4 < 0) idx4 += 15;

   out[idx1 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t4, r_t5));
   out[idx1 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t4, i_t5));

   out[idx2 * stride].r = ADD32_ovflw(dc.r, ADD32_ovflw(r_t0, r_t1));
   out[idx2 * stride].i = ADD32_ovflw(dc.i, ADD32_ovflw(i_t0, i_t1));

   out[idx3 * stride].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t0, r_t1));
   out[idx3 * stride].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t0, i_t1));

   out[idx4 * stride].r = ADD32_ovflw(dc.r, SUB32_ovflw(r_t4, r_t5));
   out[idx4 * stride].i = ADD32_ovflw(dc.i, SUB32_ovflw(i_t4, i_t5));
}

static void celt_tx_fft15_c(const cpx *in, cpx *out, int stride) {
   cpx tmp[15];
   int c5, r3;

   for (c5 = 0; c5 < 5; c5++) {
      cpx in_col[3];
      cpx out_col[3];
      for (r3 = 0; r3 < 3; r3++) {
         int r_pfa = celt_tx_pfa_P[(10 * r3 + 6 * c5) % 15];
         in_col[r3] = in[r_pfa];
      }
      winograd_fft3(in_col, out_col);
      tmp[c5] = out_col[0];
      tmp[c5 + 5] = out_col[1];
      tmp[c5 + 10] = out_col[2];
   }

   decl_fft5(tmp, 0, out, stride);
   decl_fft5(tmp + 5, 1, out, stride);
   decl_fft5(tmp + 10, 2, out, stride);
}

static void celt_tx_fft_pfa_15xM_ns_c(const struct OpusTXContext *s, void *out, void *in, ptrdiff_t stride ARG_FIXED(int downshift)) {
   int i, j;
   int len = s->len;
   int M = s->sub->len;
#ifndef FIXED_POINT
   int downshift = 0;
   (void)downshift;
#endif
   cpx *tmp = (cpx *)s->tmp;
   const cpx *in_cpx = (const cpx *)in;
   cpx *out_cpx = (cpx *)out;
   (void)stride;

   PFA_DOWNSHIFT((cpx*)in, len, &downshift, 3);
   for (i = 0; i < M; i++) {
      celt_tx_fft15_c(in_cpx + 15 * i, tmp + i, M);
   }
   PFA_DOWNSHIFT(tmp, len, &downshift, 2);

   for (j = 0; j < 15; j++) {
#ifdef FIXED_POINT
      int sub_shift = downshift;
      celt_tx_fft_p2_c(tmp + j * M, tmp + j * M, M ARG_FIXED(&sub_shift));
      if (j == 14) downshift = sub_shift;
#else
   {
      const opus_int16 *perm;
      cpx row_temp[64];
      const cpx *row;
      int k;

      perm = get_sr_perm_table(M);
      for (j = 0; j < 15; j++) {
         row = tmp + j * M;
         if (perm != NULL) {
            for (k = 0; k < M; k++) {
               row_temp[k] = row[perm[k]];
            }
         } else {
            for (k = 0; k < M; k++) {
               row_temp[k] = row[-split_radix_permutation(k, M, 0) & (M - 1)];
            }
         }
         celt_tx_fft_sr_c(row_temp, row_temp, M);
         for (k = 0; k < M; k++) {
            tmp[j * M + k] = row_temp[k];
         }
      }
   }
#endif
   }

   for (i = 0; i < len; i++) {
      out_cpx[i] = tmp[s->map[i]];
   }
}

static const struct OpusTXContext celt_tx_p2_4_c   = {  4, 1, celt_tx_p2_map_4,  NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_8_c   = {  8, 1, celt_tx_p2_map_8,  NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_16_c  = { 16, 1, celt_tx_p2_map_16, NULL, NULL, NULL, NULL };
static const struct OpusTXContext celt_tx_p2_32_c  = { 32, 1, celt_tx_p2_map_32, NULL, NULL, NULL, NULL };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_p2_64_c  = { 64, 1, celt_tx_p2_map_64, NULL, NULL, NULL, NULL };
#endif

static const struct OpusTXContext celt_tx_pfa_60_c  = {  60, 1, celt_tx_pfa_map_60,  NULL, NULL, &celt_tx_p2_4_c,  NULL };
static const struct OpusTXContext celt_tx_pfa_120_c = { 120, 1, celt_tx_pfa_map_120, NULL, NULL, &celt_tx_p2_8_c,  NULL };
static const struct OpusTXContext celt_tx_pfa_240_c = { 240, 1, celt_tx_pfa_map_240, NULL, NULL, &celt_tx_p2_16_c, NULL };
static const struct OpusTXContext celt_tx_pfa_480_c = { 480, 1, celt_tx_pfa_map_480, NULL, NULL, &celt_tx_p2_32_c, NULL };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_pfa_960_c = { 960, 1, celt_tx_pfa_map_960, NULL, NULL, &celt_tx_p2_64_c, NULL };
#endif

static const struct OpusTXContext celt_tx_mdct_120_c  = {  120, 1, celt_tx_mdct_map_120,  NULL,  NULL, &celt_tx_pfa_60_c,  celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_240_c  = {  240, 1, celt_tx_mdct_map_240,  NULL,  NULL, &celt_tx_pfa_120_c, celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_480_c  = {  480, 1, celt_tx_mdct_map_480,  NULL,  NULL, &celt_tx_pfa_240_c, celt_tx_fft_pfa_15xM_ns_c };
static const struct OpusTXContext celt_tx_mdct_960_c  = {  960, 1, celt_tx_mdct_map_960,  NULL,  NULL, &celt_tx_pfa_480_c, celt_tx_fft_pfa_15xM_ns_c };
#if defined(ENABLE_QEXT)
static const struct OpusTXContext celt_tx_mdct_1920_c = { 1920, 1, celt_tx_mdct_map_1920, NULL, NULL, &celt_tx_pfa_960_c, celt_tx_fft_pfa_15xM_ns_c };
#endif

static const struct OpusTXContext *celt_tx_mdct_kernel_c(int len)
{
   switch (len) {
      case  120: return &celt_tx_mdct_120_c;
      case  240: return &celt_tx_mdct_240_c;
      case  480: return &celt_tx_mdct_480_c;
      case  960: return &celt_tx_mdct_960_c;
#if defined(ENABLE_QEXT)
      case 1920: return &celt_tx_mdct_1920_c;
#endif
      default:   return NULL;
   }
}

#if defined(OPUS_USE_PFA_MDCT)
static void get_pfa_crt_params(int M, int *K3, int *K4)
{
   switch (M) {
      case 4:  *K3 = 3;  *K4 = 4;  break;
      case 8:  *K3 = 7;  *K4 = 2;  break;
      case 16: *K3 = 15; *K4 = 1;  break;
      case 32: *K3 = 15; *K4 = 8;  break;
      case 64: *K3 = 47; *K4 = 4;  break;
      default: *K3 = 0;  *K4 = 0;  break;
   }
}

void opus_fft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int downshift))
{
   int i;
   int nfft = st->nfft;
   int M = nfft / 15;
   int K3, K4;
   const struct OpusTXContext *mdct_tpl = celt_tx_mdct_kernel_c(2 * nfft);
   const struct OpusTXContext *tpl = (mdct_tpl && mdct_tpl->fn == celt_tx_fft_pfa_15xM_ns_c) ? mdct_tpl->sub : NULL;
   struct OpusTXContext pfa;
   VARDECL(cpx, tmp);
   VARDECL(cpx, in_perm);
   SAVE_STACK;

#if !defined(OPUS_USE_PFA_MDCT) || defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API) || defined(ENABLE_DEEP_PLC)
   if (tpl == NULL) {
      if (fin == fout) {
         VARDECL(kiss_fft_cpx, tmp_perm);
         ALLOC(tmp_perm, nfft, kiss_fft_cpx);
         for (i = 0; i < nfft; i++)
            tmp_perm[i] = fin[i];
         for (i = 0; i < nfft; i++)
            fout[st->bitrev[i]] = tmp_perm[i];
      } else {
         for (i = 0; i < nfft; i++)
            fout[st->bitrev[i]] = fin[i];
      }
      opus_fft_impl(st, fout ARG_FIXED(downshift));
      RESTORE_STACK;
      return;
   }
#else
   celt_assert2(tpl != NULL, "PFA FFT called with unsupported size in non-custom mode");
#endif

   ALLOC(tmp, nfft, cpx);
   ALLOC(in_perm, nfft, cpx);

   get_pfa_crt_params(M, &K3, &K4);

   for (i = 0; i < nfft; i++) {
      int r = celt_tx_pfa_P[(i * K4) % 15];
      int c = (i * K3) % M;
      int dest = r + c * 15;
      in_perm[dest] = fin[i];
   }

   pfa = *tpl;
   pfa.tmp = tmp;

   celt_tx_fft_pfa_15xM_ns_c(&pfa, fout, in_perm, 1 ARG_FIXED(downshift));

   RESTORE_STACK;
}

void opus_ifft_pfa_c(const kiss_fft_state *st, const kiss_fft_cpx *fin, kiss_fft_cpx *fout ARG_FIXED(int fft_shift))
{
   int i;
   int nfft = st->nfft;
   int M = nfft / 15;
   int K3, K4;
   const struct OpusTXContext *mdct_tpl = celt_tx_mdct_kernel_c(2 * nfft);
   const struct OpusTXContext *tpl = (mdct_tpl && mdct_tpl->fn == celt_tx_fft_pfa_15xM_ns_c) ? mdct_tpl->sub : NULL;
   struct OpusTXContext pfa;
   VARDECL(cpx, tmp);
   VARDECL(cpx, in_perm);
   SAVE_STACK;

#if !defined(OPUS_USE_PFA_MDCT) || defined(CUSTOM_MODES) || defined(ENABLE_OPUS_CUSTOM_API) || defined(ENABLE_DEEP_PLC)
   if (tpl == NULL) {
      if (fin == fout) {
         VARDECL(kiss_fft_cpx, tmp_perm);
         ALLOC(tmp_perm, nfft, kiss_fft_cpx);
         for (i = 0; i < nfft; i++)
            tmp_perm[i] = fin[i];
         for (i = 0; i < nfft; i++)
            fout[st->bitrev[i]] = tmp_perm[i];
      } else {
         for (i = 0; i < nfft; i++)
            fout[st->bitrev[i]] = fin[i];
      }
      for (i = 0; i < nfft; i++)
         fout[i].i = -fout[i].i;
      opus_fft_impl(st, fout ARG_FIXED(fft_shift));
      for (i = 0; i < nfft; i++)
         fout[i].i = -fout[i].i;
      RESTORE_STACK;
      return;
   }
#else
   celt_assert2(tpl != NULL, "PFA IFFT called with unsupported size in non-custom mode");
#endif

   ALLOC(tmp, nfft, cpx);
   ALLOC(in_perm, nfft, cpx);

   get_pfa_crt_params(M, &K3, &K4);

   /* 1. Permute input to grid order (IDFT input is pre-scaled by caller) */
   for (i = 0; i < nfft; i++) {
      int r = celt_tx_pfa_P[(i * K4) % 15];
      int c = (i * K3) % M;
      int dest = r + c * 15;
      in_perm[dest] = fin[i];
   }

   pfa = *tpl;
   pfa.tmp = tmp;
   celt_tx_fft_pfa_15xM_ns_c(&pfa, fout, in_perm, 1 ARG_FIXED(fft_shift));

   /* 3. Time-reverse the output from index 1 to N-1 to obtain inverse DFT */
   for (i = 1; i < (nfft + 1) / 2; i++) {
      cpx t = fout[i];
      fout[i] = fout[nfft - i];
      fout[nfft - i] = t;
   }

   RESTORE_STACK;
}
#endif

#endif /* OPUS_USE_PFA_MDCT */
