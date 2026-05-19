/*
 * mandelbrot_core.c -- ARM64 NEON-accelerated Mandelbrot pixel engine.
 *
 * Compiled with: -O3 -march=native -ffast-math -funroll-loops
 *
 * SIMD design (ARM64 NEON -- the ARM equivalent of x86 AVX2):
 * -------------------------------------------------------------
 * ARM64 NEON provides 128-bit registers holding 2x float64 (float64x2_t).
 * AVX2 on x86 holds 4x float64 (256-bit); NEON holds 2x (128-bit).
 * The technique is identical -- process multiple pixels in lockstep:
 *
 *   for each iteration:
 *     compute z2 + c for ALL lanes simultaneously (NEON vmul/vadd/vsub)
 *     check escape condition for ALL lanes (vcgtq_f64 -> bitmask)
 *     add 1 to iter count only for lanes that have NOT yet escaped
 *       (masked increment via vandq + vaddq)
 *     continue until ALL lanes escaped or hit max_iter
 *
 * This gives 2x pixel throughput on the iteration loop, and ARM64's
 * out-of-order execution can overlap operations across the two lanes.
 *
 * Key NEON intrinsics used:
 *   vdupq_n_f64(x)       -- broadcast scalar to both lanes
 *   vld1q_f64(ptr)       -- load 2 doubles from memory
 *   vmulq_f64(a,b)       -- a[0]*b[0], a[1]*b[1]
 *   vfmaq_f64(c,a,b)     -- c + a*b (FMA, 1 instruction)
 *   vsubq_f64(a,b)       -- subtraction
 *   vaddq_f64(a,b)       -- addition
 *   vcgtq_f64(a,b)       -- compare >  -> uint64x2 mask (all-1s or all-0s per lane)
 *   vandq_u64(a,b)       -- bitwise AND (used for masked increment)
 *   vaddvq_u64(a)        -- horizontal add (check if any lane still active)
 *   vgetq_lane_f64(v,i)  -- extract single lane
 *
 * Performance expectation on Cortex-A55 (budget Android):  ~1.6x vs scalar
 * Performance expectation on Cortex-A75/A78 (flagship):    ~1.9x vs scalar
 * Combined with x4-row interleaving: effective 3-4x vs original single-row.
 */

#include "mandelbrot_core.h"
#include <math.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

/* Escape radius2 -- used by all perturbation functions.
 * Larger = smoother smooth_color AND earlier exit on escaped pixels.
 * The +2 extra iters in smooth_color compensates for the larger radius. */
#define ESC_R2  256.0

/* ===========================================================================
 *  SINGLE-PIXEL STANDARD PATH  (Mariani-Silver border pixels)
 * =========================================================================== */

double mb_pixel_std(double px, double py, int max_iter) {
    double py2 = py*py, px025 = px-0.25;
    double q = px025*px025 + py2;
    if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625)
        return -1.0;
    double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
    int cp=8, iter=0;
    while (iter < max_iter && rx2+ry2 <= ESC_R2) {
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto esc;
        ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx==old_rx && ry==old_ry) return -1.0;
        if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
    }
    if(rx2+ry2 <= ESC_R2) return -1.0;
esc:;
    double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
    double c=a*a-b*b+px,     d=2.0*a*b+py;
    return smooth_color(iter+2, c, d);
}

/* ===========================================================================
 *  SINGLE-PIXEL JULIA PATH
 * =========================================================================== */

double mb_pixel_julia(double zr0, double zi0, double jcx, double jcy, int max_iter) {
    double rx=zr0, ry=zi0, rx2=rx*rx, ry2=ry*ry;
    double old_rx=0, old_ry=0;
    int cp=8, iter=0;
    while (iter < max_iter && rx2+ry2 <= ESC_R2) {
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx2+ry2>ESC_R2) goto jesc;
        ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
        if(rx==old_rx && ry==old_ry) return -1.0;
        if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
    }
    if(rx2+ry2 <= ESC_R2) return -1.0;
jesc:;
    double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
    double c=a*a-b*b+jcx,     d=2.0*a*b+jcy;
    return smooth_color(iter+2, c, d);
}

/* ===========================================================================
 *  SCALAR SINGLE-ROW  (fallback, used for Julia rows)
 * =========================================================================== */

void mb_row_std(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    double py2 = py*py;
    for (int x = 0; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py2;
        if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= ESC_R2) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>ESC_R2){
            double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
            double c=a*a-b*b+px,     d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

void mb_row_julia(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    double jcx, double jcy,
    int max_iter
) {
    for (int x = 0; x < w; x++) {
        double rx=cx_world+((double)x-half_w)*pixel_size, ry=py;
        double rx2=rx*rx, ry2=ry*ry, old_rx=0, old_ry=0;
        int cp=8, iter=0;
        while (iter < max_iter && rx2+ry2 <= ESC_R2) {
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+jcy; rx=rx2-ry2+jcx; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx && ry==old_ry){ iter=max_iter; break; }
            if(iter>=cp){ old_rx=rx; old_ry=ry; cp*=2; }
        }
        if(rx2+ry2>ESC_R2){
            double a=rx*rx-ry*ry+jcx, b=2.0*rx*ry+jcy;
            double c=a*a-b*b+jcx,     d=2.0*a*b+jcy;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

/* ===========================================================================
 *  NEON 2-WIDE DOUBLE ROW  (primary hot path for standard Mandelbrot)
 *
 *  Processes pixels in pairs.  Both pixels iterate simultaneously in NEON
 *  float64x2 registers.  Escaped pixels are masked out -- their z values
 *  freeze and their iteration counter stops incrementing.  The loop exits
 *  only when both lanes have escaped or max_iter is reached.
 *
 *  Masked iteration counter trick (same as AVX2 Mandelbrot on x86):
 *    active = lanes where mag2 <= 16  (uint64x2, all-1s = active)
 *    iter_vec += (active & 1)         (adds 1 only to active lanes)
 *
 *  After the NEON loop, tail pixel (if w is odd) uses scalar path.
 *  Non-NEON builds fall back to scalar mb_row_std automatically.
 * =========================================================================== */

#ifdef __ARM_NEON

void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    const float64x2_t v_escape  = vdupq_n_f64(ESC_R2);
    const float64x2_t v_two     = vdupq_n_f64(2.0);
    const float64x2_t v_py      = vdupq_n_f64(py);
    const float64x2_t v_neg1    = vdupq_n_f64(-1.0);
    /* For bulb/cardioid check */
    const float64x2_t v_025     = vdupq_n_f64(0.25);
    const float64x2_t v_0625    = vdupq_n_f64(0.0625);
    const float64x2_t v_1       = vdupq_n_f64(1.0);
    const float64x2_t v_py2     = vdupq_n_f64(py * py);

    /* Process pairs of pixels */
    int x = 0;
    for (; x <= w - 2; x += 2) {
        /* Pixel x-coordinates for the pair */
        double px0 = cx_world + ((double)(x)   - half_w) * pixel_size;
        double px1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        float64x2_t v_px = {px0, px1};

        /* -- Bulb / cardioid rejection (scalar check, rare branch) -- */
        /* If both pixels are inside bulb, write -1 and skip. */
        /* We check individually to avoid complicating the NEON path. */
        int in_bulb0, in_bulb1;
        {
            double px025 = px0-0.25, q = px025*px025+py*py;
            in_bulb0 = (q*(q+px025) <= 0.25*py*py) || ((px0+1.0)*(px0+1.0)+py*py <= 0.0625);
        }
        {
            double px025 = px1-0.25, q = px025*px025+py*py;
            in_bulb1 = (q*(q+px025) <= 0.25*py*py) || ((px1+1.0)*(px1+1.0)+py*py <= 0.0625);
        }
        if (in_bulb0 && in_bulb1) { out[x]=-1.0; out[x+1]=-1.0; continue; }

        /* -- NEON Mandelbrot iteration with Brent period detection -- */
        float64x2_t vr  = vdupq_n_f64(0.0);
        float64x2_t vi  = vdupq_n_f64(0.0);
        float64x2_t vr2 = vdupq_n_f64(0.0);
        float64x2_t vi2 = vdupq_n_f64(0.0);
        float64x2_t iter_vec = vdupq_n_f64(0.0);
        uint64x2_t active = vceqq_f64(vdupq_n_f64(0.0), vdupq_n_f64(0.0)); /* all 1s */

        if (in_bulb0) active = vsetq_lane_u64(0, active, 0);
        if (in_bulb1) active = vsetq_lane_u64(0, active, 1);

        /* Brent period detection — catches interior pixels fast */
        float64x2_t old_vr = vr, old_vi = vi;
        int check_period = 32, check_ctr = 0;

        int iter = 0;
        for (; iter < max_iter; iter++) {
            float64x2_t new_r = vaddq_f64(vsubq_f64(vr2, vi2), v_px);
            float64x2_t new_i = vfmaq_f64(v_py, v_two, vmulq_f64(vr, vi));
            vr  = new_r;
            vi  = new_i;
            vr2 = vmulq_f64(vr, vr);
            vi2 = vmulq_f64(vi, vi);
            float64x2_t mag2 = vaddq_f64(vr2, vi2);
            uint64x2_t still_in = vcleq_f64(mag2, v_escape);
            active = vandq_u64(active, still_in);
            float64x2_t one_masked = vreinterpretq_f64_u64(
                vandq_u64(vreinterpretq_u64_f64(v_1), active));
            iter_vec = vaddq_f64(iter_vec, one_masked);
            if (vaddvq_u64(active) == 0) break;
            /* Brent: check for period every check_period iters */
            if (++check_ctr >= check_period) {
                uint64x2_t sameR = vceqq_f64(vr, old_vr);
                uint64x2_t sameI = vceqq_f64(vi, old_vi);
                if (vgetq_lane_u64(vandq_u64(sameR,sameI),0) &&
                    vgetq_lane_u64(vandq_u64(sameR,sameI),1)) {
                    active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(1)); /* all 0 */
                    break;
                }
                old_vr = vr; old_vi = vi;
                check_period *= 2; check_ctr = 0;
            }
        }

        /* Extract results — two scalar stores, no lane loop */
        {
            double it0 = vgetq_lane_f64(iter_vec, 0);
            double it1 = vgetq_lane_f64(iter_vec, 1);
            double r0  = vgetq_lane_f64(vr, 0), i0 = vgetq_lane_f64(vi, 0);
            double r1  = vgetq_lane_f64(vr, 1), i1 = vgetq_lane_f64(vi, 1);
            double m0  = r0*r0 + i0*i0;
            double m1  = r1*r1 + i1*i1;
            if (in_bulb0 || m0 <= ESC_R2) {
                out[x] = -1.0;
            } else {
                double a = r0*r0-i0*i0+px0, b = 2.0*r0*i0+py;
                double c = a*a-b*b+px0,     d = 2.0*a*b+py;
                out[x] = smooth_color((int)it0+2, c, d);
            }
            if (in_bulb1 || m1 <= ESC_R2) {
                out[x+1] = -1.0;
            } else {
                double a = r1*r1-i1*i1+px1, b = 2.0*r1*i1+py;
                double c = a*a-b*b+px1,     d = 2.0*a*b+py;
                out[x+1] = smooth_color((int)it1+2, c, d);
            }
        }
    }

    /* -- Scalar tail for odd width -- */
    for (; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py*py;
        if (q*(q+px025) <= 0.25*py*py || (px+1.0)*(px+1.0)+py*py <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter<max_iter && rx2+ry2<=ESC_R2) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx&&ry==old_ry){iter=max_iter;break;}
            if(iter>=cp){old_rx=rx;old_ry=ry;cp*=2;}
        }
        if(rx2+ry2>ESC_R2){
            double a=rx*rx-ry*ry+px,b=2.0*rx*ry+py;
            double c=a*a-b*b+px,d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

#elif defined(__AVX2__) && defined(__FMA__)

/* ===========================================================================
 *  AVX2 4-WIDE DOUBLE ROW  (x86-64 fast path)
 *
 *  AVX2 __m256d holds 4x float64 in a 256-bit register -- twice the width
 *  of ARM NEON.  The algorithm is identical to the NEON path:
 *    * Compute z2 + c for all 4 lanes simultaneously
 *    * Mask escaped lanes out of the iteration counter
 *    * Exit when all 4 lanes have escaped or hit max_iter
 *
 *  Key AVX2 intrinsics:
 *    _mm256_set1_pd(x)         -- broadcast scalar to all 4 lanes
 *    _mm256_set_pd(d,c,b,a)    -- load 4 distinct values (note: reverse order)
 *    _mm256_mul_pd(a,b)        -- 4-wide multiply
 *    _mm256_fmadd_pd(a,b,c)    -- a*b+c  (FMA, requires __FMA__)
 *    _mm256_sub_pd(a,b)        -- subtraction
 *    _mm256_add_pd(a,b)        -- addition
 *    _mm256_cmp_pd(a,b,_CMP_LE_OQ) -- compare <=, returns all-1s per lane
 *    _mm256_and_pd(a,b)        -- bitwise AND (mask application)
 *    _mm256_movemask_pd(a)     -- 4-bit mask from sign bits (exit test)
 *    _mm256_storeu_pd(ptr,v)   -- store 4 doubles unaligned
 *
 *  Expected speedup vs scalar on modern x86: ~3.0-3.8x
 *  Combined with x4-row interleaving: effective ~12-15x vs original.
 * =========================================================================== */

#include <immintrin.h>

void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    const __m256d v_escape = _mm256_set1_pd(ESC_R2);
    const __m256d v_two    = _mm256_set1_pd(2.0);
    const __m256d v_one    = _mm256_set1_pd(1.0);
    const __m256d v_py     = _mm256_set1_pd(py);
    const __m256d v_neg1   = _mm256_set1_pd(-1.0);
    double py2 = py * py;

    int x = 0;
    /* -- Process 4 pixels at a time -- */
    for (; x <= w - 4; x += 4) {
        double px0 = cx_world + ((double)(x+0) - half_w) * pixel_size;
        double px1 = cx_world + ((double)(x+1) - half_w) * pixel_size;
        double px2 = cx_world + ((double)(x+2) - half_w) * pixel_size;
        double px3 = cx_world + ((double)(x+3) - half_w) * pixel_size;

        /* Bulb/cardioid scalar pre-check -- skip entire group if all inside */
        int b0,b1,b2,b3;
        { double p=px0-0.25,q=p*p+py2; b0=(q*(q+p)<=0.25*py2)||((px0+1)*(px0+1)+py2<=0.0625); }
        { double p=px1-0.25,q=p*p+py2; b1=(q*(q+p)<=0.25*py2)||((px1+1)*(px1+1)+py2<=0.0625); }
        { double p=px2-0.25,q=p*p+py2; b2=(q*(q+p)<=0.25*py2)||((px2+1)*(px2+1)+py2<=0.0625); }
        { double p=px3-0.25,q=p*p+py2; b3=(q*(q+p)<=0.25*py2)||((px3+1)*(px3+1)+py2<=0.0625); }

        if (b0 && b1 && b2 && b3) {
            out[x]=out[x+1]=out[x+2]=out[x+3]=-1.0;
            continue;
        }

        /* Load cx per pixel -- _mm256_set_pd fills lanes 3,2,1,0 (reversed) */
        __m256d v_cx = _mm256_set_pd(px3, px2, px1, px0);

        /* z.r, z.i, z.r2, z.i2 -- all start at 0 */
        __m256d vr  = _mm256_setzero_pd();
        __m256d vi  = _mm256_setzero_pd();
        __m256d vr2 = _mm256_setzero_pd();
        __m256d vi2 = _mm256_setzero_pd();

        /* Iteration counter per lane (as double for masked add) */
        __m256d iter_vec = _mm256_setzero_pd();

        /* Active mask: all lanes active initially.
         * Lanes in the bulb start inactive. */
        __m256d active = _mm256_cmp_pd(_mm256_setzero_pd(),
                                       _mm256_setzero_pd(), _CMP_EQ_OQ); /* all 1s */
        if (b0) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x1);
        if (b1) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x2);
        if (b2) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x4);
        if (b3) active = _mm256_blend_pd(active, _mm256_setzero_pd(), 0x8);

        for (int iter = 0; iter < max_iter; iter++) {
            /* z_new.r = r2 - i2 + cx  */
            __m256d new_r = _mm256_add_pd(_mm256_sub_pd(vr2, vi2), v_cx);
            /* z_new.i = 2*r*i + py  (FMA: py + r*i*2) */
            __m256d new_i = _mm256_fmadd_pd(v_two, _mm256_mul_pd(vr, vi), v_py);

            vr  = new_r;
            vi  = new_i;
            vr2 = _mm256_mul_pd(vr, vr);
            vi2 = _mm256_mul_pd(vi, vi);

            __m256d mag2 = _mm256_add_pd(vr2, vi2);

            /* Lanes still inside: mag2 <= 16 */
            __m256d still_in = _mm256_cmp_pd(mag2, v_escape, _CMP_LE_OQ);

            /* Narrow active: only lanes that were active AND still inside */
            active = _mm256_and_pd(active, still_in);

            /* iter_vec += 1.0 masked to active lanes */
            iter_vec = _mm256_add_pd(iter_vec, _mm256_and_pd(v_one, active));

            /* Exit early if all lanes escaped (_mm256_movemask_pd == 0) */
            if (_mm256_movemask_pd(active) == 0) break;
        }

        /* Extract and write results */
        double iters[4], rs[4], is[4], mag2s[4];
        _mm256_storeu_pd(iters, iter_vec);
        _mm256_storeu_pd(rs,    vr);
        _mm256_storeu_pd(is,    vi);
        __m256d mag2_final = _mm256_add_pd(vr2, vi2);
        _mm256_storeu_pd(mag2s, mag2_final);

        double pxs[4] = {px0, px1, px2, px3};
        int    bulbs[4] = {b0, b1, b2, b3};
        for (int i = 0; i < 4; i++) {
            if (bulbs[i] || mag2s[i] <= ESC_R2) {
                out[x+i] = -1.0;
            } else {
                int it = (int)iters[i];
                double a = rs[i]*rs[i] - is[i]*is[i] + pxs[i];
                double b = 2.0*rs[i]*is[i] + py;
                double c = a*a - b*b + pxs[i];
                double d = 2.0*a*b + py;
                out[x+i] = smooth_color(it + 2, c, d);
            }
        }
    }

    /* -- Scalar tail for remainder (w % 4 != 0) -- */
    for (; x < w; x++) {
        double px = cx_world + ((double)x - half_w) * pixel_size;
        double px025 = px-0.25, q = px025*px025+py2;
        if (q*(q+px025) <= 0.25*py2 || (px+1.0)*(px+1.0)+py2 <= 0.0625) {
            out[x]=-1.0; continue;
        }
        double rx=0,ry=0,rx2=0,ry2=0,old_rx=0,old_ry=0;
        int cp=8, iter=0;
        while (iter<max_iter && rx2+ry2<=ESC_R2) {
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx2+ry2>ESC_R2) break;
            ry=2.0*rx*ry+py; rx=rx2-ry2+px; rx2=rx*rx; ry2=ry*ry; iter++;
            if(rx==old_rx&&ry==old_ry){iter=max_iter;break;}
            if(iter>=cp){old_rx=rx;old_ry=ry;cp*=2;}
        }
        if(rx2+ry2>ESC_R2){
            double a=rx*rx-ry*ry+px, b=2.0*rx*ry+py;
            double c=a*a-b*b+px, d=2.0*a*b+py;
            out[x]=smooth_color(iter+2,c,d);
        } else { out[x]=-1.0; }
    }
}

#else
/* Scalar fallback for non-NEON, non-AVX2 builds */
void mb_row_std_neon(
    double* __restrict__ out, int w,
    double cx_world, double py,
    double pixel_size, double half_w,
    int max_iter
) {
    mb_row_std(out, w, cx_world, py, pixel_size, half_w, max_iter);
}
#endif /* __ARM_NEON / __AVX2__ */

/* ===========================================================================
 *  4-ROW INTERLEAVED BATCH  (calls NEON row function for each row)
 *
 *  Scheduling 4 rows per goroutine dispatch reduces Go scheduler overhead.
 *  Each row uses the NEON 2-wide path internally, so we get 4x2 = 8 pixels
 *  of effective parallelism per goroutine wakeup vs 1 in the original code.
 * =========================================================================== */

void mb_row_std_x4(
    double* __restrict__ out, int w, int n_rows,
    double cx_world, double py_base,
    double pixel_size, double half_w,
    int max_iter
) {
    for (int r = 0; r < n_rows; r++) {
        double py = py_base + (double)r * pixel_size;
        mb_row_std_neon(out + r*w, w, cx_world, py, pixel_size, half_w, max_iter);
    }
}

/* ===========================================================================
 *  PERTURBATION THEORY INNER LOOP  (scalar, 4x unrolled)
 *
 *  Hot path for deep zoom.  Each iteration:
 *    rx = ref_x[i] + dx          (absolute position for escape check)
 *    ry = ref_y[i] + dy
 *    dx_new = 2*ref_x[i]*dx - 2*ref_y[i]*dy + dx2-dy2 + dcx
 *    dy_new = 2*ref_x[i]*dy + 2*ref_y[i]*dx + 2*dx*dy  + dcy
 *
 *  Loop unrolled 4x to expose ILP.  The ref array loads of successive i
 *  can pipeline with the multiply-accumulate of the previous step on OOO
 *  cores (A55 is in-order but A78/X1 are OOO -- both benefit from unroll).
 *
 *  __builtin_expect: escape is rare (most pixels live to ref_len),
 *  so the branch predictor gets a hint and the fast path stays hot.
 * =========================================================================== */

double mb_perturb_pixel(
    const double* __restrict__ ref_x,
    const double* __restrict__ ref_y,
    const double* __restrict__ ref2x,
    const double* __restrict__ ref2y,
    int ref_len,
    double dcx, double dcy,
    double dx0, double dy0,
    int sa_iter,
    double px, double py,
    int max_iter
) {
    double dx=dx0, dy=dy0, rx=0.0, ry=0.0;
    int iter=sa_iter, escaped=0;

    const double *rx_ptr  = ref_x  + iter;
    const double *ry_ptr  = ref_y  + iter;
    const double *r2x_ptr = ref2x  + iter;
    const double *r2y_ptr = ref2y  + iter;

    int remaining = ref_len - iter;
    if (remaining > max_iter - iter) remaining = max_iter - iter;

    /* Period detection (Brent) -- catches interior pixels fast */
    double old_dx=dx, old_dy=dy;
    int check_period=32, check_ctr=0;

    /* -- 4x unrolled body ----------------------------------------------- */
    int i = 0;
    int remaining4 = remaining & ~3;
    for (; i < remaining4; i += 4) {
        __builtin_prefetch(&rx_ptr[i+8],  0, 1);
        __builtin_prefetch(&r2x_ptr[i+8], 0, 1);
        /* Iteration i+0 */
        rx = rx_ptr[i]   + dx; ry = ry_ptr[i]   + dy;
        if (__builtin_expect(rx*rx + ry*ry > 256.0, 0)) { escaped=1; iter+=i+1; goto done; }
        { double dx2=dx*dx, dy2=dy*dy;
          double ndx = r2x_ptr[i]*dx - r2y_ptr[i]*dy + dx2 - dy2 + dcx;
          double ndy = r2x_ptr[i]*dy + r2y_ptr[i]*dx + 2.0*dx*dy  + dcy;
          dx=ndx; dy=ndy; }
        /* Iteration i+1 */
        rx = rx_ptr[i+1] + dx; ry = ry_ptr[i+1] + dy;
        if (__builtin_expect(rx*rx + ry*ry > 256.0, 0)) { escaped=1; iter+=i+2; goto done; }
        { double dx2=dx*dx, dy2=dy*dy;
          double ndx = r2x_ptr[i+1]*dx - r2y_ptr[i+1]*dy + dx2 - dy2 + dcx;
          double ndy = r2x_ptr[i+1]*dy + r2y_ptr[i+1]*dx + 2.0*dx*dy  + dcy;
          dx=ndx; dy=ndy; }
        /* Iteration i+2 */
        rx = rx_ptr[i+2] + dx; ry = ry_ptr[i+2] + dy;
        if (__builtin_expect(rx*rx + ry*ry > 256.0, 0)) { escaped=1; iter+=i+3; goto done; }
        { double dx2=dx*dx, dy2=dy*dy;
          double ndx = r2x_ptr[i+2]*dx - r2y_ptr[i+2]*dy + dx2 - dy2 + dcx;
          double ndy = r2x_ptr[i+2]*dy + r2y_ptr[i+2]*dx + 2.0*dx*dy  + dcy;
          dx=ndx; dy=ndy; }
        /* Iteration i+3 */
        rx = rx_ptr[i+3] + dx; ry = ry_ptr[i+3] + dy;
        if (__builtin_expect(rx*rx + ry*ry > 256.0, 0)) { escaped=1; iter+=i+4; goto done; }
        { double dx2=dx*dx, dy2=dy*dy;
          double ndx = r2x_ptr[i+3]*dx - r2y_ptr[i+3]*dy + dx2 - dy2 + dcx;
          double ndy = r2x_ptr[i+3]*dy + r2y_ptr[i+3]*dx + 2.0*dx*dy  + dcy;
          dx=ndx; dy=ndy; }
        /* Brent period check every check_period iters */
        check_ctr += 4;
        if (__builtin_expect(check_ctr >= check_period, 0)) {
            if (dx == old_dx && dy == old_dy) { iter = max_iter; goto done; }
            old_dx=dx; old_dy=dy; check_period*=2; check_ctr=0;
        }
    }
    /* Scalar tail */
    for (; i < remaining; i++) {
        rx = rx_ptr[i] + dx; ry = ry_ptr[i] + dy;
        if (__builtin_expect(rx*rx + ry*ry > 256.0, 0)) { escaped=1; iter+=i+1; goto done; }
        double dx2=dx*dx, dy2=dy*dy;
        double ndx = r2x_ptr[i]*dx - r2y_ptr[i]*dy + dx2 - dy2 + dcx;
        double ndy = r2x_ptr[i]*dy + r2y_ptr[i]*dx + 2.0*dx*dy  + dcy;
        dx=ndx; dy=ndy;
    }
    iter += remaining;
done:
    if (!escaped) {
        if (iter >= max_iter) return -1.0;
        if (dx*dx+dy*dy > (rx*rx+ry*ry)*1e6) return -2.0;
        return -1.0;
    }
    double rx2=rx*rx-ry*ry+px, ry2=2.0*rx*ry+py;
    double rx3=rx2*rx2-ry2*ry2+px, ry3=2.0*rx2*ry2+py;
    return smooth_color(iter+2, rx3, ry3);
}

/* ===========================================================================
 *  BLA (BILINEAR APPROXIMATION) PERTURBATION ROW
 *
 *  Core idea vs SA:
 *    SA: one global skip N -- every pixel starts at iteration N.
 *    BLA: per-pixel greedy table hop -- a pixel with |dc|= might skip
 *         90% of iterations; a pixel with |dc|=maxDelta skips 10%.
 *
 *  Algorithm per pixel:
 *    1. Start at iter=0, dc=(dcx,dcy), delta=(0,0).
 *    2. While bla[i].ref_iter == iter AND |dc|2 < bla[i].r2:
 *         delta = A * delta + (A-1)*dc + dc   [= A*(delta+dc) - dc + dc]
 *         Actually: delta_new = A*dc  (we store the cumulative A)
 *         iter += bla[i].step
 *         advance i
 *    3. Run normal NEON perturbation loop from (iter, delta).
 *
 *  The table is sorted by ref_iter.  We use a single index `bi` that
 *  advances monotonically -- O(n) total table work across all pixels
 *  on a row (since they all start at bi=0 and advance forward).
 *
 *  Implementation note on delta after BLA hop:
 *    The BLA recurrence for delta (not dc) is:
 *      delta_{n+step}  A * delta_n + (A-1)*dc + ... higher order
 *    For the common case delta_0 = 0 (first hop from iteration 0):
 *      delta_{step} = A * dc
 *    For subsequent hops where delta != 0, full recurrence applies.
 *    We track this correctly via the accumulated A coefficient.
 *
 *  NEON 2-wide: each pixel pair shares the same BLA hop sequence
 *  (same table, same validity check) but has independent deltas.
 *  We process the BLA hop in scalar (it's a few multiplies per entry),
 *  then pass both pixels into the NEON tail loop.
 * =========================================================================== */

/* ===========================================================================
 *  BLA + REBASING PERTURBATION ROW
 *
 *  Two key improvements over the original:
 *
 *  1. REBASING (Zhuoran 2021): After each perturbation step, check if
 *     |Z+z|^2 << |Z|^2 (threshold G=1e-6). If so, the delta has collapsed
 *     near a critical point and floating-point precision is lost. Instead of
 *     letting this produce a glitch pixel, we rebase: reset iter=0, set the
 *     new delta to the current absolute position z = Z[iter]+delta, and
 *     continue from the start of the reference orbit. This avoids glitches
 *     entirely in the inner loop with ~0 extra cost per non-glitch pixel.
 *
 *  2. CORRECT BLA RADIUS (mathr 2022): r = eps * (|Z| - max_dc) / (2|Z|+1)
 *     where eps = 2^-53 (hardware float64 precision), not the old eps2*|Z|^2.
 *     This gives longer valid hops, especially when |Z| >> max_dc.
 *
 * =========================================================================== */

void mb_perturb_row_bla(
    double* __restrict__ out,
    int w,
    const double* __restrict__ ref_x,
    const double* __restrict__ ref_y,
    const double* __restrict__ ref2x,
    const double* __restrict__ ref2y,
    const double* __restrict__ ref_mag2,
    int ref_len,
    double dcx_base, double dcy,
    double pixel_size,
    double cx_world, double row_py,
    const BlaEntry* __restrict__ bla,
    int bla_len,
    int max_iter
) {
    /* Glitch/rebase threshold: rebase when |Z+z|^2 < G*|Z|^2 */
#define REBASE_G 1e-6

    int x;
    for (x = 0; x <= w - 2; x += 2) {
        double dcx0 = dcx_base + (double)x       * pixel_size;
        double dcx1 = dcx_base + (double)(x+1)   * pixel_size;
        double px0  = cx_world + ((double)x       - (double)(w/2)) * pixel_size;
        double px1  = cx_world + ((double)(x+1)   - (double)(w/2)) * pixel_size;

        /* ── BLA greedy hop (scalar — table walk is cheap vs NEON overhead) ── */
        double dx0 = 0.0, dy0 = 0.0;
        double dx1 = 0.0, dy1 = 0.0;
        int iter0 = 0, iter1 = 0;

        {
            double dc2_0 = dcx0*dcx0 + dcy*dcy;
            double dc2_1 = dcx1*dcx1 + dcy*dcy;
            /* Binary search for first BLA entry with ref_iter == 0.
             * The table is sorted by ref_iter ascending; entries for iter=0
             * are at the front, so bi_start=0 is correct here.
             * After a rebase (iter reset to 0) this is also correct.
             * Key insight: entries are grouped by ref_iter; within a group,
             * take hops in order while dc2 < r2. */
            int bi = 0;
            /* Pixel 0 */
            while (bi < bla_len && bla[bi].ref_iter == iter0 && dc2_0 < bla[bi].r2) {
                double ar = bla[bi].ar, ai = bla[bi].ai;
                double ndx = ar*dx0 - ai*dy0 + (ar-1.0)*dcx0 - ai*dcy;
                double ndy = ar*dy0 + ai*dx0 + (ar-1.0)*dcy  + ai*dcx0;
                dx0 = ndx; dy0 = ndy;
                iter0 += bla[bi].step;
                bi++;
            }
            bi = 0;
            /* Pixel 1 */
            while (bi < bla_len && bla[bi].ref_iter == iter1 && dc2_1 < bla[bi].r2) {
                double ar = bla[bi].ar, ai = bla[bi].ai;
                double ndx = ar*dx1 - ai*dy1 + (ar-1.0)*dcx1 - ai*dcy;
                double ndy = ar*dy1 + ai*dx1 + (ar-1.0)*dcy  + ai*dcx1;
                dx1 = ndx; dy1 = ndy;
                iter1 += bla[bi].step;
                bi++;
            }
        }

        /* ── SIMD perturbation tail: NEON on ARM, AVX2 on x86, scalar fallback ── */
        int sa_start = iter0 < iter1 ? iter0 : iter1;
#ifdef __ARM_NEON
        {
        const float64x2_t v_esc  = vdupq_n_f64(ESC_R2);
        const float64x2_t v_one  = vdupq_n_f64(1.0);
        const float64x2_t v_zero = vdupq_n_f64(0.0);

        float64x2_t vdx  = {dx0, dx1};
        float64x2_t vdy  = {dy0, dy1};
        float64x2_t vdcx = {dcx0, dcx1};
        float64x2_t vdcy = vdupq_n_f64(dcy);

        const double *rxp  = ref_x;
        const double *ryp  = ref_y;
        const double *r2xp = ref2x;
        const double *r2yp = ref2y;

        /* Start both at min landing point */
        float64x2_t iter_vec = vdupq_n_f64((double)sa_start);
        uint64x2_t active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(0)); /* all 1s */

        int remaining = ref_len - sa_start;
        if (remaining > max_iter - sa_start) remaining = max_iter - sa_start;

        float64x2_t old_vdx = vdx, old_vdy = vdy;
        int check_period = 32, check_ctr = 0;

        int i = 0;

        #define PSTEP(ii) do { \
            float64x2_t vr2x_ = vdupq_n_f64(r2xp[sa_start+(ii)]); \
            float64x2_t vr2y_ = vdupq_n_f64(r2yp[sa_start+(ii)]); \
            float64x2_t vrx_  = vaddq_f64(vdupq_n_f64(rxp[sa_start+(ii)]), vdx); \
            float64x2_t vry_  = vaddq_f64(vdupq_n_f64(ryp[sa_start+(ii)]), vdy); \
            float64x2_t mag2_ = vaddq_f64(vmulq_f64(vrx_,vrx_), vmulq_f64(vry_,vry_)); \
            uint64x2_t still_in_ = vcleq_f64(mag2_, v_esc); \
            active = vandq_u64(active, still_in_); \
            float64x2_t one_m_ = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v_one), active)); \
            iter_vec = vaddq_f64(iter_vec, one_m_); \
            float64x2_t vdx2_  = vmulq_f64(vdx, vdx); \
            float64x2_t vdy2_  = vmulq_f64(vdy, vdy); \
            float64x2_t v2dxdy_ = vaddq_f64(vmulq_f64(vdx,vdy), vmulq_f64(vdx,vdy)); \
            float64x2_t ndx_ = vaddq_f64(vsubq_f64(vaddq_f64(vmulq_f64(vr2x_,vdx),vsubq_f64(vdx2_,vdy2_)),vmulq_f64(vr2y_,vdy)), vdcx); \
            float64x2_t ndy_ = vaddq_f64(vaddq_f64(vmulq_f64(vr2x_,vdy),vaddq_f64(vmulq_f64(vr2y_,vdx),v2dxdy_)), vdcy); \
            vdx = vreinterpretq_f64_u64(vbslq_u64(active, vreinterpretq_u64_f64(ndx_), vreinterpretq_u64_f64(vdx))); \
            vdy = vreinterpretq_f64_u64(vbslq_u64(active, vreinterpretq_u64_f64(ndy_), vreinterpretq_u64_f64(vdy))); \
            /* Rebasing: if |Z+z|^2 < G*|Z|^2, reset delta to absolute position */ \
            float64x2_t refmag2_ = vdupq_n_f64(ref_mag2[sa_start+(ii)] * REBASE_G); \
            uint64x2_t needs_rebase_ = vandq_u64(active, vcltq_f64(mag2_, refmag2_)); \
            if (__builtin_expect(vaddvq_u64(needs_rebase_) != 0, 0)) { \
                /* Rebase: new delta = absolute position Z+z; restart from iter 0 */ \
                /* Only rebase still-active lanes; escaped lanes keep their result */ \
                if (vgetq_lane_u64(needs_rebase_,0)) { \
                    vdx = vreinterpretq_f64_u64(vbslq_u64((uint64x2_t){~(uint64_t)0,0}, \
                        vreinterpretq_u64_f64(vrx_), vreinterpretq_u64_f64(vdx))); \
                    vdy = vreinterpretq_f64_u64(vbslq_u64((uint64x2_t){~(uint64_t)0,0}, \
                        vreinterpretq_u64_f64(vry_), vreinterpretq_u64_f64(vdy))); \
                } \
                if (vgetq_lane_u64(needs_rebase_,1)) { \
                    vdx = vreinterpretq_f64_u64(vbslq_u64((uint64x2_t){0,~(uint64_t)0}, \
                        vreinterpretq_u64_f64(vrx_), vreinterpretq_u64_f64(vdx))); \
                    vdy = vreinterpretq_f64_u64(vbslq_u64((uint64x2_t){0,~(uint64_t)0}, \
                        vreinterpretq_u64_f64(vry_), vreinterpretq_u64_f64(vdy))); \
                } \
                /* Reset iteration counter and reference pointer to start */ \
                sa_start = 0; i = -1; rem8 = (ref_len < max_iter ? ref_len : max_iter) & ~7; \
                rxp = ref_x; ryp = ref_y; r2xp = ref2x; r2yp = ref2y; \
                remaining = ref_len < max_iter ? ref_len : max_iter; \
                iter_vec = vdupq_n_f64(0.0); \
                old_vdx = vdx; old_vdy = vdy; check_period = 32; check_ctr = 0; \
            } \
        } while(0)

        int rem16 = remaining & ~15;
        int rem8  = remaining & ~7;

        for (; i < rem16; i += 16) {
            __builtin_prefetch(&rxp[sa_start+i+32], 0, 1);
            __builtin_prefetch(&r2xp[sa_start+i+32], 0, 1);
            PSTEP(i+0);  PSTEP(i+1);  PSTEP(i+2);  PSTEP(i+3);
            PSTEP(i+4);  PSTEP(i+5);  PSTEP(i+6);  PSTEP(i+7);
            PSTEP(i+8);  PSTEP(i+9);  PSTEP(i+10); PSTEP(i+11);
            PSTEP(i+12); PSTEP(i+13); PSTEP(i+14); PSTEP(i+15);
            if (vaddvq_u64(active) == 0) goto bla_done_neon;
            check_ctr += 16;
            if (__builtin_expect(check_ctr >= check_period, 0)) {
                uint64x2_t sameX = vceqq_f64(vdx, old_vdx);
                uint64x2_t sameY = vceqq_f64(vdy, old_vdy);
                if (vgetq_lane_u64(vandq_u64(sameX,sameY),0) &&
                    vgetq_lane_u64(vandq_u64(sameX,sameY),1)) {
                    active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(1));
                    goto bla_done_neon;
                }
                old_vdx = vdx; old_vdy = vdy;
                check_period *= 2; check_ctr = 0;
            }
        }
        for (; i < rem8; i += 8) {
            __builtin_prefetch(&rxp[sa_start+i+16], 0, 1);
            __builtin_prefetch(&r2xp[sa_start+i+16], 0, 1);
            PSTEP(i+0); PSTEP(i+1); PSTEP(i+2); PSTEP(i+3);
            PSTEP(i+4); PSTEP(i+5); PSTEP(i+6); PSTEP(i+7);
            if (vaddvq_u64(active) == 0) goto bla_done_neon;
            check_ctr += 8;
            if (__builtin_expect(check_ctr >= check_period, 0)) {
                uint64x2_t sameX = vceqq_f64(vdx, old_vdx);
                uint64x2_t sameY = vceqq_f64(vdy, old_vdy);
                if (vgetq_lane_u64(vandq_u64(sameX,sameY),0) &&
                    vgetq_lane_u64(vandq_u64(sameX,sameY),1)) {
                    active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(1));
                    goto bla_done_neon;
                }
                old_vdx = vdx; old_vdy = vdy;
                check_period *= 2; check_ctr = 0;
            }
        }
        for (; i < remaining; i++) {
            PSTEP(i);
            if (vaddvq_u64(active) == 0) break;
        }
        #undef PSTEP

bla_done_neon:;
        {
            uint64_t a0 = vgetq_lane_u64(active, 0);
            uint64_t a1 = vgetq_lane_u64(active, 1);
            double it0 = vgetq_lane_f64(iter_vec, 0);
            double it1 = vgetq_lane_f64(iter_vec, 1);
            double dx0f = vgetq_lane_f64(vdx,0), dy0f = vgetq_lane_f64(vdy,0);
            double dx1f = vgetq_lane_f64(vdx,1), dy1f = vgetq_lane_f64(vdy,1);
            /* Per-lane last reference index derived from iter_vec, not the shared
             * loop counter i. After a rebase, it* reflects the actual orbit position
             * for that lane. Clamp to valid range. */
            int last_i0 = (int)it0 - sa_start; if (last_i0 < 0) last_i0 = 0; if (last_i0 >= remaining) last_i0 = remaining-1;
            int last_i1 = (int)it1 - sa_start; if (last_i1 < 0) last_i1 = 0; if (last_i1 >= remaining) last_i1 = remaining-1;
            double lrx0 = rxp[sa_start+last_i0]+dx0f, lry0 = ryp[sa_start+last_i0]+dy0f;
            double lrx1 = rxp[sa_start+last_i1]+dx1f, lry1 = ryp[sa_start+last_i1]+dy1f;
            if (a0) {
                out[x] = (dx0f*dx0f+dy0f*dy0f > (lrx0*lrx0+lry0*lry0)*1e6) ? -2.0 : -1.0;
            } else {
                double m0 = lrx0*lrx0+lry0*lry0;
                if (m0 <= ESC_R2*4) { out[x] = smooth_color((int)it0+2, lrx0, lry0); }
                else {
                    double a2=lrx0*lrx0-lry0*lry0+px0, b2=2.0*lrx0*lry0+row_py;
                    double a3=a2*a2-b2*b2+px0,         b3=2.0*a2*b2+row_py;
                    out[x] = smooth_color((int)it0+2, a3, b3);
                }
            }
            if (a1) {
                out[x+1] = (dx1f*dx1f+dy1f*dy1f > (lrx1*lrx1+lry1*lry1)*1e6) ? -2.0 : -1.0;
            } else {
                double m1 = lrx1*lrx1+lry1*lry1;
                if (m1 <= ESC_R2*4) { out[x+1] = smooth_color((int)it1+2, lrx1, lry1); }
                else {
                    double a2=lrx1*lrx1-lry1*lry1+px1, b2=2.0*lrx1*lry1+row_py;
                    double a3=a2*a2-b2*b2+px1,         b3=2.0*a2*b2+row_py;
                    out[x+1] = smooth_color((int)it1+2, a3, b3);
                }
            }
        }
        } /* ARM_NEON */

#elif defined(__AVX2__) && defined(__FMA__)
        /* ── AVX2/FMA 4-wide perturbation tail (x86-64 fast path) ──────────── */
        /* Processes 4 pixels at once using 256-bit AVX2 doubles.
         * Rebase (delta collapse) is handled per-lane via scalar fallback —
         * it's rare (few pixels per frame) so the branch cost is negligible.
         * The common case (no rebase) runs entirely in AVX2 at 4x throughput. */
        {
        /* We process pixels x, x+1, x+2, x+3 in one AVX2 pass.
         * Pre-load BLA hops for all 4 pixels (scalar — cheap), then SIMD tail. */
        double dcx2 = dcx_base + (double)(x+2) * pixel_size;
        double dcx3 = dcx_base + (double)(x+3) * pixel_size;
        double px2   = cx_world + ((double)(x+2) - (double)(w/2)) * pixel_size;
        double px3   = cx_world + ((double)(x+3) - (double)(w/2)) * pixel_size;
        double dx2 = 0.0, dy2 = 0.0, dx3 = 0.0, dy3 = 0.0;
        int iter2 = 0, iter3 = 0;

        /* BLA hops for pixels 2 and 3 */
        {
            double dc2_2 = dcx2*dcx2 + dcy*dcy;
            double dc2_3 = dcx3*dcx3 + dcy*dcy;
            int bi = 0;
            while (bi < bla_len && bla[bi].ref_iter == iter2 && dc2_2 < bla[bi].r2) {
                double ar = bla[bi].ar, ai = bla[bi].ai;
                double ndx = ar*dx2 - ai*dy2 + (ar-1.0)*dcx2 - ai*dcy;
                double ndy = ar*dy2 + ai*dx2 + (ar-1.0)*dcy  + ai*dcx2;
                dx2 = ndx; dy2 = ndy; iter2 += bla[bi].step; bi++;
            }
            bi = 0;
            while (bi < bla_len && bla[bi].ref_iter == iter3 && dc2_3 < bla[bi].r2) {
                double ar = bla[bi].ar, ai = bla[bi].ai;
                double ndx = ar*dx3 - ai*dy3 + (ar-1.0)*dcx3 - ai*dcy;
                double ndy = ar*dy3 + ai*dx3 + (ar-1.0)*dcy  + ai*dcx3;
                dx3 = ndx; dy3 = ndy; iter3 += bla[bi].step; bi++;
            }
        }

        int sa4 = iter0; /* min of all 4 starting iters */
        if (iter1 < sa4) sa4 = iter1;
        if (iter2 < sa4) sa4 = iter2;
        if (iter3 < sa4) sa4 = iter3;
        int remaining4 = (ref_len < max_iter ? ref_len : max_iter) - sa4;

        /* Pack into AVX2 registers: lanes [0,1,2,3] = pixels [x,x+1,x+2,x+3] */
        __m256d vdx  = _mm256_set_pd(dx3,  dx2,  dx1,  dx0);
        __m256d vdy  = _mm256_set_pd(dy3,  dy2,  dy1,  dy0);
        __m256d vdcx = _mm256_set_pd(dcx3, dcx2, dcx1, dcx0);
        __m256d vdcy = _mm256_set1_pd(dcy);
        __m256d v_esc  = _mm256_set1_pd(ESC_R2);
        __m256d v_one  = _mm256_set1_pd(1.0);
        __m256d v_two  = _mm256_set1_pd(2.0);
        __m256d v_zero = _mm256_set1_pd(0.0);
        __m256d v_rebase_g = _mm256_set1_pd(REBASE_G);

        /* active: all-ones if lane still iterating */
        __m256d active = _mm256_castsi256_pd(_mm256_set1_epi64x(-1LL));
        __m256d iter_vec = _mm256_set_pd((double)iter3,(double)iter2,(double)iter1,(double)iter0);
        __m256d v_maxiter = _mm256_set1_pd((double)max_iter);

        int rebase_needed = 0;

        for (int i = 0; i < remaining4; i++) {
            int ii = sa4 + i;
            __builtin_prefetch(&ref_x[ii+16],  0, 1);
            __builtin_prefetch(&ref2x[ii+16], 0, 1);

            __m256d vrx  = _mm256_set1_pd(ref_x[ii]);
            __m256d vry  = _mm256_set1_pd(ref_y[ii]);
            __m256d vr2x = _mm256_set1_pd(ref2x[ii]);
            __m256d vr2y = _mm256_set1_pd(ref2y[ii]);
            __m256d vrmag2 = _mm256_set1_pd(ref_mag2[ii]);

            /* full_rx = ref + delta */
            __m256d full_rx = _mm256_add_pd(vrx, vdx);
            __m256d full_ry = _mm256_add_pd(vry, vdy);
            __m256d mag2    = _mm256_fmadd_pd(full_ry, full_ry, _mm256_mul_pd(full_rx, full_rx));

            /* Escape check */
            __m256d escaped = _mm256_cmp_pd(mag2, v_esc, _CMP_GT_OQ);
            __m256d still   = _mm256_andnot_pd(escaped, active);

            /* Count iters for non-escaped active lanes */
            iter_vec = _mm256_add_pd(iter_vec, _mm256_and_pd(active, v_one));

            /* Rebase check: |delta|^2 < REBASE_G * |ref|^2 → delta collapsed */
            __m256d delta_mag2 = _mm256_fmadd_pd(vdy, vdy, _mm256_mul_pd(vdx, vdx));
            __m256d rebase_thresh = _mm256_mul_pd(vrmag2, v_rebase_g);
            __m256d needs_rebase  = _mm256_and_pd(still, _mm256_cmp_pd(delta_mag2, rebase_thresh, _CMP_LT_OQ));
            if (_mm256_movemask_pd(needs_rebase)) {
                rebase_needed = 1;
                break; /* fall through to scalar for this pixel group */
            }

            /* Update delta: ndx = ref2x*dx - ref2y*dy + dx^2 - dy^2 + dcx */
            __m256d dx2v = _mm256_mul_pd(vdx, vdx);
            __m256d dy2v = _mm256_mul_pd(vdy, vdy);
            __m256d ndx  = _mm256_add_pd(
                _mm256_fmsub_pd(vr2x, vdx, _mm256_mul_pd(vr2y, vdy)),
                _mm256_add_pd(_mm256_sub_pd(dx2v, dy2v), vdcx));
            __m256d ndy  = _mm256_add_pd(
                _mm256_fmadd_pd(vr2x, vdy, _mm256_mul_pd(vr2y, vdx)),
                _mm256_fmadd_pd(v_two, _mm256_mul_pd(vdx, vdy), vdcy));

            /* Zero out escaped lanes' deltas (keep last valid value) */
            vdx = _mm256_blendv_pd(vdx, ndx, still);
            vdy = _mm256_blendv_pd(vdy, ndy, still);
            active = still;

            /* Cap at max_iter */
            __m256d hit_max = _mm256_cmp_pd(iter_vec, v_maxiter, _CMP_GE_OQ);
            active = _mm256_andnot_pd(hit_max, active);

            if (!_mm256_movemask_pd(active)) break;
        }

        if (!rebase_needed) {
            /* Extract results from AVX2 lanes */
            double iters[4], dxf[4], dyf[4];
            _mm256_storeu_pd(iters, iter_vec);
            _mm256_storeu_pd(dxf,   vdx);
            _mm256_storeu_pd(dyf,   vdy);
            double act[4]; _mm256_storeu_pd(act, active);
            double pxv[4] = {px0, px1, px2, px3};
            double dcxv[4] = {dcx0, dcx1, dcx2, dcx3};
            int last_ii = sa4 + ((remaining4>0)?remaining4-1:0);
            if (last_ii >= ref_len) last_ii = ref_len-1;
            for (int k = 0; k < 4; k++) {
                int kx = x + k;
                if (kx >= w) break;
                int itk = (int)iters[k];
                double lrx = ref_x[last_ii]+dxf[k], lry = ref_y[last_ii]+dyf[k];
                if (act[k] != 0.0) { /* still active = not escaped */
                    out[kx] = (dxf[k]*dxf[k]+dyf[k]*dyf[k] > (lrx*lrx+lry*lry)*1e6) ? -2.0 : -1.0;
                } else {
                    double a2=lrx*lrx-lry*lry+pxv[k], b2=2.0*lrx*lry+row_py;
                    double a3=a2*a2-b2*b2+pxv[k],     b3=2.0*a2*b2+row_py;
                    out[kx] = smooth_color(itk+2, a3, b3);
                }
            }
            x += 2; /* outer loop will also add 2, giving x+=4 total */
            continue;
        }
        /* rebase needed: fall through to scalar for all 4 pixels */
        }
        /* scalar fallback (rebase case or non-AVX2) */
        for (int k = 0; k < 4 && (x+k) < w; k++) {
            double dcxk = dcx_base + (double)(x+k) * pixel_size;
            double pxk  = cx_world + ((double)(x+k) - (double)(w/2)) * pixel_size;
            /* re-run BLA for this pixel from scratch */
            double dxk = 0.0, dyk = 0.0; int itk = 0;
            { double dc2k = dcxk*dcxk+dcy*dcy; int bi=0;
              while (bi<bla_len && bla[bi].ref_iter==itk && dc2k<bla[bi].r2) {
                  double ar=bla[bi].ar,ai=bla[bi].ai;
                  double nd=ar*dxk-ai*dyk+(ar-1.0)*dcxk-ai*dcy;
                  double ne=ar*dyk+ai*dxk+(ar-1.0)*dcy+ai*dcxk;
                  dxk=nd; dyk=ne; itk+=bla[bi].step; bi++;
              }
            }
            double rxk=0, ryk=0; int esck=0;
            int remk = (ref_len<max_iter?ref_len:max_iter) - itk;
            for (int i=0; i<remk && !esck; i++) {
                int ii=itk+i;
                rxk=ref_x[ii]+dxk; ryk=ref_y[ii]+dyk;
                double mag2=rxk*rxk+ryk*ryk;
                if (mag2>ESC_R2) { esck=1; itk+=i+1; break; }
                if (mag2<ref_mag2[ii]*REBASE_G) { dxk=rxk; dyk=ryk; itk=0; i=-1; remk=ref_len<max_iter?ref_len:max_iter; continue; }
                double d2=dxk*dxk, e2=dyk*dyk;
                double ndx=ref2x[ii]*dxk-ref2y[ii]*dyk+d2-e2+dcxk;
                double ndy=ref2x[ii]*dyk+ref2y[ii]*dxk+2.0*dxk*dyk+dcy;
                dxk=ndx; dyk=ndy;
            }
            if (esck) {
                double a2=rxk*rxk-ryk*ryk+pxk, b2=2.0*rxk*ryk+row_py;
                double a3=a2*a2-b2*b2+pxk,     b3=2.0*a2*b2+row_py;
                out[x+k] = smooth_color(itk+2, a3, b3);
            } else { out[x+k] = -1.0; }
        }
        x += 2; /* outer loop adds 2 more → net +4 per AVX2 group */
#else
        /* Non-AVX2 scalar tail with rebasing */
        {
        double rx0, ry0; int it0 = sa_start; int esc0 = 0;
        int remaining0 = (ref_len < max_iter ? ref_len : max_iter) - it0;
        for (int i = 0; i < remaining0 && !esc0; i++) {
            int ii = it0+i;
            rx0 = ref_x[ii]+dx0; ry0 = ref_y[ii]+dy0;
            double mag2 = rx0*rx0+ry0*ry0;
            if (mag2 > ESC_R2) { esc0=1; it0+=i+1; break; }
            if (mag2 < ref_mag2[ii]*REBASE_G) { dx0=rx0; dy0=ry0; it0=0; i=-1; remaining0=ref_len<max_iter?ref_len:max_iter; continue; }
            double d2=dx0*dx0, e2=dy0*dy0;
            double ndx = ref2x[ii]*dx0 - ref2y[ii]*dy0 + d2-e2 + dcx0;
            double ndy = ref2x[ii]*dy0 + ref2y[ii]*dx0 + 2.0*dx0*dy0 + dcy;
            dx0=ndx; dy0=ndy;
        }
        if (esc0) {
            double a2=rx0*rx0-ry0*ry0+px0, b2=2.0*rx0*ry0+row_py;
            double a3=a2*a2-b2*b2+px0,     b3=2.0*a2*b2+row_py;
            out[x] = smooth_color(it0+2, a3, b3);
        } else { out[x] = -1.0; }

        double rx1, ry1; int it1 = sa_start; int esc1 = 0;
        int remaining1 = (ref_len < max_iter ? ref_len : max_iter) - it1;
        for (int i = 0; i < remaining1 && !esc1; i++) {
            int ii = it1+i;
            rx1 = ref_x[ii]+dx1; ry1 = ref_y[ii]+dy1;
            double mag2 = rx1*rx1+ry1*ry1;
            if (mag2 > ESC_R2) { esc1=1; it1+=i+1; break; }
            if (mag2 < ref_mag2[ii]*REBASE_G) { dx1=rx1; dy1=ry1; it1=0; i=-1; remaining1=ref_len<max_iter?ref_len:max_iter; continue; }
            double d2=dx1*dx1, e2=dy1*dy1;
            double ndx = ref2x[ii]*dx1 - ref2y[ii]*dy1 + d2-e2 + dcx1;
            double ndy = ref2x[ii]*dy1 + ref2y[ii]*dx1 + 2.0*dx1*dy1 + dcy;
            dx1=ndx; dy1=ndy;
        }
        if (esc1) {
            double a2=rx1*rx1-ry1*ry1+px1, b2=2.0*rx1*ry1+row_py;
            double a3=a2*a2-b2*b2+px1,     b3=2.0*a2*b2+row_py;
            out[x+1] = smooth_color(it1+2, a3, b3);
        } else { out[x+1] = -1.0; }
        } /* non-AVX2 scalar */
#endif

    } /* x += 2 */

    /* Scalar tail for odd width */
    for (; x < w; x++) {
        double dcx = dcx_base + (double)x * pixel_size;
        double px  = cx_world + ((double)x - (double)(w/2)) * pixel_size;
        double dx = 0.0, dy = 0.0;
        int bi = 0, iter = 0;
        double dc2 = dcx*dcx + dcy*dcy;
        while (bi < bla_len && bla[bi].ref_iter == iter && dc2 < bla[bi].r2) {
            double ar=bla[bi].ar, ai=bla[bi].ai;
            double ndx = ar*dx - ai*dy + (ar-1.0)*dcx - ai*dcy;
            double ndy = ar*dy + ai*dx + (ar-1.0)*dcy  + ai*dcx;
            dx=ndx; dy=ndy; iter+=bla[bi].step; bi++;
        }
        double rx=0, ry=0; int escaped=0;
        int lim = ref_len < max_iter ? ref_len : max_iter;
        for (int i=iter; i<lim; i++) {
            rx=ref_x[i]+dx; ry=ref_y[i]+dy;
            double mag2=rx*rx+ry*ry;
            if (mag2 > ESC_R2) { escaped=1; iter=i+1; break; }
            if (mag2 < ref_mag2[i]*REBASE_G) { dx=rx; dy=ry; i=0; iter=0; }
            double d2=dx*dx, e2=dy*dy;
            double ndx=ref2x[i]*dx - ref2y[i]*dy + d2-e2+dcx;
            double ndy=ref2x[i]*dy + ref2y[i]*dx + 2.0*dx*dy+dcy;
            dx=ndx; dy=ndy;
        }
        if (escaped) {
            double a2=rx*rx-ry*ry+px, b2=2.0*rx*ry+row_py;
            double a3=a2*a2-b2*b2+px, b3=2.0*a2*b2+row_py;
            out[x] = smooth_color(iter+2, a3, b3);
        } else { out[x] = -1.0; }
    }
#undef REBASE_G
}


void mb_perturb_row_neon(
    double* __restrict__ out,
    int w,
    const double* __restrict__ ref_x,
    const double* __restrict__ ref_y,
    const double* __restrict__ ref2x,
    const double* __restrict__ ref2y,
    int ref_len,
    double ref_px, double ref_py,
    double dcx_base, double dcy,
    double pixel_size,
    double cx_world, double row_py,
    double dx0_base, double dy0_base,
    double sa_dcx_base, double sa_dcy,
    double sa_ar, double sa_ai,
    double sa_br, double sa_bi,
    double sa_cr, double sa_ci,
    int sa_iter,
    int max_iter
) {
    int use3 = (sa_cr != 0.0 || sa_ci != 0.0);

    /* Helper: compute SA initial delta for a given dcx */
    #define SA_DELTA(dcx_, dcy_, dxout_, dyout_) do { \
        double _dc2r = (dcx_)*(dcx_) - (dcy_)*(dcy_); \
        double _dc2i = 2.0*(dcx_)*(dcy_); \
        (dxout_) = sa_ar*(dcx_) - sa_ai*(dcy_) + sa_br*_dc2r - sa_bi*_dc2i; \
        (dyout_) = sa_ar*(dcy_) + sa_ai*(dcx_) + sa_br*_dc2i + sa_bi*_dc2r; \
        if (use3) { \
            double _dc3r = (dcx_)*_dc2r - (dcy_)*_dc2i; \
            double _dc3i = (dcx_)*_dc2i + (dcy_)*_dc2r; \
            (dxout_) += sa_cr*_dc3r - sa_ci*_dc3i; \
            (dyout_) += sa_cr*_dc3i + sa_ci*_dc3r; \
        } \
    } while(0)

#ifdef __ARM_NEON
    const float64x2_t v_esc  = vdupq_n_f64(ESC_R2);
    const float64x2_t v_one  = vdupq_n_f64(1.0);

    int x = 0;
    for (; x <= w - 2; x += 2) {
        double dcx0 = dcx_base + (double)x       * pixel_size;
        double dcx1 = dcx_base + (double)(x+1)   * pixel_size;
        double px0  = cx_world + ((double)(x)   - (double)(w/2)) * pixel_size;
        double px1  = cx_world + ((double)(x+1) - (double)(w/2)) * pixel_size;

        double dx0v, dy0v, dx1v, dy1v;
        if (sa_iter > 0) {
            SA_DELTA(dcx0, dcy, dx0v, dy0v);
            SA_DELTA(dcx1, dcy, dx1v, dy1v);
        } else {
            dx0v=0; dy0v=0; dx1v=0; dy1v=0;
        }

        float64x2_t vdx  = {dx0v, dx1v};
        float64x2_t vdy  = {dy0v, dy1v};
        float64x2_t vdcx = {dcx0, dcx1};
        float64x2_t vdcy = vdupq_n_f64(dcy);

        const double *rxp  = ref_x  + sa_iter;
        const double *ryp  = ref_y  + sa_iter;
        const double *r2xp = ref2x + sa_iter;
        const double *r2yp = ref2y + sa_iter;

        int remaining = ref_len - sa_iter;
        if (remaining > max_iter - sa_iter) remaining = max_iter - sa_iter;

        /* Masked iteration counter -- float64 trick from shallow path */
        float64x2_t iter_vec = vdupq_n_f64((double)sa_iter);
        /* Active mask: all-1s for lanes not yet escaped */
        uint64x2_t active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(0)); /* all 1s */

        /* Period-detection variables (Brent) */
        float64x2_t old_vdx = vdx, old_vdy = vdy;
        int check_period = 32, check_counter = 0;

        /* -- 8x unrolled NEON loop ---------------------------------------- */
        #define PERTURB_STEP(ii) do { \
            float64x2_t vr2x_ = vdupq_n_f64(r2xp[ii]); \
            float64x2_t vr2y_ = vdupq_n_f64(r2yp[ii]); \
            float64x2_t vrx_  = vaddq_f64(vdupq_n_f64(rxp[ii]),  vdx); \
            float64x2_t vry_  = vaddq_f64(vdupq_n_f64(ryp[ii]),  vdy); \
            float64x2_t mag2_ = vaddq_f64(vmulq_f64(vrx_,vrx_), vmulq_f64(vry_,vry_)); \
            uint64x2_t still_in_ = vcleq_f64(mag2_, v_esc); \
            active = vandq_u64(active, still_in_); \
            float64x2_t one_m_ = vreinterpretq_f64_u64(vandq_u64(vreinterpretq_u64_f64(v_one), active)); \
            iter_vec = vaddq_f64(iter_vec, one_m_); \
            float64x2_t vdx2_  = vmulq_f64(vdx, vdx); \
            float64x2_t vdy2_  = vmulq_f64(vdy, vdy); \
            float64x2_t v2dxdy_ = vaddq_f64(vmulq_f64(vdx,vdy), vmulq_f64(vdx,vdy)); \
            float64x2_t ndx_ = vaddq_f64(vsubq_f64(vaddq_f64(vmulq_f64(vr2x_,vdx),vsubq_f64(vdx2_,vdy2_)),vmulq_f64(vr2y_,vdy)), vdcx); \
            float64x2_t ndy_ = vaddq_f64(vaddq_f64(vmulq_f64(vr2x_,vdy),vaddq_f64(vmulq_f64(vr2y_,vdx),v2dxdy_)), vdcy); \
            vdx = vreinterpretq_f64_u64(vbslq_u64(active, vreinterpretq_u64_f64(ndx_), vreinterpretq_u64_f64(vdx))); \
            vdy = vreinterpretq_f64_u64(vbslq_u64(active, vreinterpretq_u64_f64(ndy_), vreinterpretq_u64_f64(vdy))); \
        } while(0)

        int i = 0;
        int rem8 = remaining & ~7;
        for (; i < rem8; i += 8) {
            PERTURB_STEP(i+0); PERTURB_STEP(i+1); PERTURB_STEP(i+2); PERTURB_STEP(i+3);
            PERTURB_STEP(i+4); PERTURB_STEP(i+5); PERTURB_STEP(i+6); PERTURB_STEP(i+7);
            /* Early exit if both escaped */
            if (vaddvq_u64(active) == 0) goto done_pair;
            /* Brent period check */
            check_counter += 8;
            if (check_counter >= check_period) {
                uint64x2_t sameX = vceqq_f64(vdx, old_vdx);
                uint64x2_t sameY = vceqq_f64(vdy, old_vdy);
                uint64x2_t same  = vandq_u64(sameX, sameY);
                /* If both lanes cycling, mark them interior and exit */
                if (vgetq_lane_u64(same,0) && vgetq_lane_u64(same,1)) {
                    active = vceqq_u64(vdupq_n_u64(0), vdupq_n_u64(1)); /* all 0 */
                    goto done_pair;
                }
                old_vdx = vdx; old_vdy = vdy;
                check_period *= 2; check_counter = 0;
            }
        }
        /* Scalar tail of remaining */
        for (; i < remaining; i++) {
            PERTURB_STEP(i);
            if (vgetq_lane_u64(active, 0) == 0 && vgetq_lane_u64(active, 1) == 0) break;
        }
        #undef PERTURB_STEP

done_pair:;
        /* Extract results */
        uint64_t a0 = vgetq_lane_u64(active, 0);
        uint64_t a1 = vgetq_lane_u64(active, 1);
        double   it0 = vgetq_lane_f64(iter_vec, 0);
        double   it1 = vgetq_lane_f64(iter_vec, 1);
        double   dx0f = vgetq_lane_f64(vdx, 0), dy0f = vgetq_lane_f64(vdy, 0);
        double   dx1f = vgetq_lane_f64(vdx, 1), dy1f = vgetq_lane_f64(vdy, 1);
        /* Final rx/ry from last ref step -- reconstruct from last delta */
        int last_i = (i < remaining ? i : remaining - 1);
        if (last_i < 0) last_i = 0;
        double   lrx0 = rxp[last_i] + dx0f, lry0 = ryp[last_i] + dy0f;
        double   lrx1 = rxp[last_i] + dx1f, lry1 = ryp[last_i] + dy1f;

        /* Lane 0 */
        if (a0) {
            /* Still active = not escaped */
            out[x] = (dx0f*dx0f+dy0f*dy0f > (lrx0*lrx0+lry0*lry0)*1e6) ? -2.0 : -1.0;
        } else {
            int iit = (int)it0;
            /* Recover proper rx,ry at escape: re-read from the iter where escape happened */
            /* We use lrx0 as approximation -- good enough for smooth_color */
            double mag2_esc = lrx0*lrx0 + lry0*lry0;
            if (mag2_esc <= ESC_R2 * 4) {
                /* Escaped via masked approach but final pos inside: use delta */
                out[x] = smooth_color(iit+2, lrx0, lry0);
            } else {
                double a2=lrx0*lrx0-lry0*lry0+px0, b2=2.0*lrx0*lry0+row_py;
                double a3=a2*a2-b2*b2+px0,         b3=2.0*a2*b2+row_py;
                out[x] = smooth_color(iit+2, a3, b3);
            }
        }
        /* Lane 1 */
        if (a1) {
            out[x+1] = (dx1f*dx1f+dy1f*dy1f > (lrx1*lrx1+lry1*lry1)*1e6) ? -2.0 : -1.0;
        } else {
            int iit = (int)it1;
            double mag2_esc = lrx1*lrx1 + lry1*lry1;
            if (mag2_esc <= ESC_R2 * 4) {
                out[x+1] = smooth_color(iit+2, lrx1, lry1);
            } else {
                double a2=lrx1*lrx1-lry1*lry1+px1, b2=2.0*lrx1*lry1+row_py;
                double a3=a2*a2-b2*b2+px1,         b3=2.0*a2*b2+row_py;
                out[x+1] = smooth_color(iit+2, a3, b3);
            }
        }
    }

    /* Scalar tail (odd width) */
    for (; x < w; x++) {
        double px  = cx_world + ((double)x - (double)(w/2)) * pixel_size;
        double dcx = dcx_base + (double)x * pixel_size;
        double dx0s=0, dy0s=0;
        if (sa_iter > 0) { SA_DELTA(dcx, dcy, dx0s, dy0s); }
        out[x] = mb_perturb_pixel(ref_x,ref_y,ref2x,ref2y,ref_len,dcx,dcy,dx0s,dy0s,sa_iter,px,row_py,max_iter);
    }

#else
    /* Non-NEON scalar fallback */
    for (int x = 0; x < w; x++) {
        double px  = cx_world + ((double)x - (double)(w/2)) * pixel_size;
        double dcx = dcx_base + (double)x * pixel_size;
        double dx0s=0, dy0s=0;
        if (sa_iter > 0) { SA_DELTA(dcx, dcy, dx0s, dy0s); }
        out[x] = mb_perturb_pixel(ref_x,ref_y,ref2x,ref2y,ref_len,dcx,dcy,dx0s,dy0s,sa_iter,px,row_py,max_iter);
    }
#endif
    #undef SA_DELTA
}

#undef ESC_R2
