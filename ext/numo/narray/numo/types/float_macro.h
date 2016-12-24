#include "float_def.h"

EXTERN double round(double);
EXTERN double log2(double);
EXTERN double exp2(double);
#ifdef HAVE_EXP10
EXTERN double exp10(double);
#else
EXTERN double pow(double, double);
#endif

#define m_zero 0.0
#define m_one  1.0

#define m_num_to_data(x) NUM2DBL(x)
#define m_data_to_num(x) rb_float_new(x)

#define m_from_double(x) (x)
#define m_from_real(x) (x)

#define m_add(x,y) ((x)+(y))
#define m_sub(x,y) ((x)-(y))
#define m_mul(x,y) ((x)*(y))
#define m_div(x,y) ((x)/(y))
#define m_div_check(x,y) ((y)==0)
#define m_mod(x,y) fmod(x,y)
#define m_divmod(x,y,a,b) {a=(x)/(y); b=m_mod(x,y);}
#define m_pow(x,y) pow(x,y)
#define m_pow_int(x,y) pow_int(x,y)

#define m_abs(x)     fabs(x)
#define m_minus(x)   (-(x))
#define m_reciprocal(x) (1/(x))
#define m_square(x)  ((x)*(x))
#define m_floor(x)   floor(x)
#define m_round(x)   round(x)
#define m_ceil(x)    ceil(x)
#define m_trunc(x)   trunc(x)
#define m_rint(x)    rint(x)
#define m_sign(x)    (((x)==0) ? 0.0:(((x)>0) ? 1.0:(((x)<0) ? -1.0:(x))))

#define m_eq(x,y) ((x)==(y))
#define m_ne(x,y) ((x)!=(y))
#define m_gt(x,y) ((x)>(y))
#define m_ge(x,y) ((x)>=(y))
#define m_lt(x,y) ((x)<(y))
#define m_le(x,y) ((x)<=(y))

#define m_isnan(x) isnan(x)
#define m_isinf(x) isinf(x)
#define m_isfinite(x) isfinite(x)

#define m_mulsum(x,y,z) {z += x*y;}
#define m_mulsum_init INT2FIX(0)

#define m_sprintf(s,x) sprintf(s,"%g",x)

#define cmp(a,b)                                                        \
    (m_isnan(qsort_cast(a)) ? (m_isnan(qsort_cast(b)) ? 0 : 1) :            \
     (m_isnan(qsort_cast(b)) ? -1 :                                       \
      ((qsort_cast(a)==qsort_cast(b)) ? 0 :                             \
       (qsort_cast(a) > qsort_cast(b)) ? 1 : -1)))

#define cmpgt(a,b)                                            \
    ((m_isnan(qsort_cast(a)) && !m_isnan(qsort_cast(b))) ||       \
     (qsort_cast(a) > qsort_cast(b)))

#define m_sqrt(x)    sqrt(x)
#define m_cbrt(x)    cbrt(x)
#define m_log(x)     log(x)
#define m_log2(x)    log2(x)
#define m_log10(x)   log10(x)
#define m_exp(x)     exp(x)
#define m_exp2(x)    exp2(x)
#ifdef HAVE_EXP10
#define m_exp10(x)   exp10(x)
#else
#define m_exp10(x)   pow(10, x)
#endif
#define m_sin(x)     sin(x)
#define m_cos(x)     cos(x)
#define m_tan(x)     tan(x)
#define m_asin(x)    asin(x)
#define m_acos(x)    acos(x)
#define m_atan(x)    atan(x)
#define m_sinh(x)    sinh(x)
#define m_cosh(x)    cosh(x)
#define m_tanh(x)    tanh(x)
#define m_asinh(x)   asinh(x)
#define m_acosh(x)   acosh(x)
#define m_atanh(x)   atanh(x)
#define m_atan2(x,y) atan2(x,y)
#define m_hypot(x,y) hypot(x,y)

#define m_erf(x)     erf(x)
#define m_erfc(x)    erfc(x)
#define m_ldexp(x,y) ldexp(x,y)
#define m_frexp(x,exp) frexp(x,exp)

static inline dtype pow_int(dtype x, int p)
{
    dtype r=1;
    switch(p) {
    case 0: return 1;
    case 1: return x;
    case 2: return x*x;
    case 3: return x*x*x;
    case 4: x=x*x; return x*x;
    }
    if (p<0)  return 1/pow_int(x,-p);
    if (p>64) return pow(x,p);
    while (p) {
        if (p&1) r *= x;
        x *= x;
        p >>= 1;
    }
    return r;
}


static inline dtype f_sum(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    dtype x,y=0;

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            y += x;
        }
        p += stride;
    }
    return y;
}

static inline dtype f_kahan_sum(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    dtype x;
    volatile dtype y=0;
    volatile dtype t,r=0;

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            if (fabs(x) > fabs(y)) {
                dtype z=x; x=y; y=z;
            }
            r += x;
            t = y;
            y += r;
            t = y-t;
            r -= t;
        }
        p += stride;
    }
    return y;
}

#if __SIZEOF_INT__ == 8
#  define BUILTIN_SADD64_OVERFLOW(x, y, sum) (__builtin_sadd_overflow((x), (y), (sum)))
#elif __SIZEOF_LONG__ == 8
#  define BUILTIN_SADD64_OVERFLOW(x, y, sum) (__builtin_saddl_overflow((x), (y), (sum)))
#elif __SIZEOF_LONG_LONG__ == 8
#  define BUILTIN_SADD64_OVERFLOW(x, y, sum) (__builtin_saddll_overflow((x), (y), (sum)))
#endif

static inline int64_t signed_shift_right(int64_t x, unsigned a)
{
    return ( (x < 0) ? (-(-(x+1)>>a)-1) : (x >> a) ) ;
}

static inline dtype f_accurate_sum(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    dtype x;
    int64_t arr[40] = { 0, 0, 0, 0, 0,  0, 0, 0, 0, 0
                      , 0, 0, 0, 0, 0,  0, 0, 0, 0, 0
                      , 0, 0, 0, 0, 0,  0, 0, 0, 0, 0
                      , 0, 0, 0, 0, 0,  0, 0, 0, 0, 0 };
    for (; i--;) {
        x = *(dtype *)p;
        if (x == m_zero) {
            goto loop_cont;
        }
        if (m_isnan(x)) {
            //TODO
        }
        if (m_isinf(x)) {
            //TODO
        }
        {
            int d;
            int64_t g, h, *arr_ptr;
            {
                int e;
                double f = frexp(x, &e);
                if ((d = 1021+e) < 0) {
                    g = (int64_t)(f * exp2((double)(53+d)));
                    d = 0;
                } else {
                    g = (int64_t)(f * exp2(53.0));
                }
            }
            {
                unsigned b = (unsigned)(d / 60);
                arr_ptr = &(arr[b]);
            }
            {
                unsigned c = (unsigned)(d % 60);
                h = signed_shift_right(g, 60U - c);
                g <<= c;
            }
            if (h != (int64_t)-1) {
                if (BUILTIN_SADD64_OVERFLOW(*arr_ptr, g & (int64_t)0xfffffffffffffff, arr_ptr)) {
                    h += (int64_t)16;
                }
                g = h;
                ++arr_ptr;
            }
            if (g) {
                if (g < 0) {
                    if (BUILTIN_SADD64_OVERFLOW(*arr_ptr, g, arr_ptr)) {
                        do {
                            ++arr_ptr;
                        } while (BUILTIN_SADD64_OVERFLOW(*arr_ptr, (int64_t)-16, arr_ptr));
                    }
                } else if (g > 0) {
                    if (BUILTIN_SADD64_OVERFLOW(*arr_ptr, g, arr_ptr)) {
                        do {
                            ++arr_ptr;
                        } while (BUILTIN_SADD64_OVERFLOW(*arr_ptr, (int64_t)16, arr_ptr));
                    }
                }
            }
        }
loop_cont:
        p += stride;
    }
    i = 40;
    {
        int neg, r=0, flg2=0, flg1=0;
        VALUE const denormal_threshold = LL2NUM(0x10000000000000LL);
        VALUE const threshold_53bits   = LL2NUM(0x1fffffffffffffLL);
        VALUE const threshold_54bits   = LL2NUM(0x3fffffffffffffLL);
        VALUE y = INT2FIX(0);
        for (; i--;) {
            y = rb_funcall(y, rb_intern("<<"), 1, INT2FIX(60));
            y = rb_funcall(y, rb_intern("+"), 1, LL2NUM((long long)arr[i]));
        }
        if (RTEST(rb_funcall(y, rb_intern("zero?"), 0))) {
            return 0.0;
        }
        if (RTEST(rb_funcall(y, rb_intern("negative?"), 0))) {
            y = rb_funcall(y, rb_intern("-@"), 0);
            neg = 1;
        } else {
            neg = 0;
        }
        while (RTEST(rb_funcall(y, rb_intern("<"), 1, denormal_threshold))) {
            long long result_ll = NUM2LL(y);
            double result_d = (double)result_ll;
            result_d *= exp2((double)(-1074));
            return ( neg ? (-result_d) : result_d ) ;
        }
        while (RTEST(rb_funcall(y, rb_intern("even?"), 0))) {
            y = rb_funcall(y, rb_intern(">>"), 1, INT2FIX(1));
            r += 1;
        }
        if (RTEST(rb_funcall(y, rb_intern(">"), 1, threshold_53bits))) {
            while (RTEST(rb_funcall(y, rb_intern(">"), 1, threshold_54bits))) {
                flg2 = 1;
                y = rb_funcall(y, rb_intern(">>"), 1, INT2FIX(1));
                r += 1;
            }
            if (RTEST(rb_funcall(y, rb_intern("odd?"), 0))) {
                flg1 = 1;
            }
            y = rb_funcall(y, rb_intern(">>"), 1, INT2FIX(1));
            r += 1;
        }
        if (flg1) {
            if (RTEST(rb_funcall(y, rb_intern("odd?"), 0)) || flg2) {
                y = rb_funcall(y, rb_intern("+"), 1, INT2FIX(1));
            }
        }
        {
            long long result_ll = NUM2LL(y);
            double result_d = (double)result_ll;
            result_d *= exp2((double)(-1074+r));
            return ( neg ? (-result_d) : result_d ) ;
        }
    }
}

#undef BUILTIN_SADD64_OVERFLOW

static inline dtype f_prod(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    dtype x,y=1;

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            y *= x;
        }
        p += stride;
    }
    return y;
}

static inline dtype f_mean(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    size_t count=0;
    dtype x,y=0;

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            y += x;
            count++;
        }
        p += stride;
    }
    return y/count;
}

static inline dtype f_var(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    size_t count=0;
    dtype x,y=0;
    dtype a,m;

    m = f_mean(n,p,stride);

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            a = x - m;
            y += a*a;
            count++;
        }
        p += stride;
    }
    return y/(count-1);
}

static inline dtype f_stddev(size_t n, char *p, ssize_t stride)
{
    return m_sqrt(f_var(n,p,stride));
}

static inline dtype f_rms(size_t n, char *p, ssize_t stride)
{
    size_t i=n;
    size_t count=0;
    dtype x,y=0;

    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x)) {
            y += x*x;
            count++;
        }
        p += stride;
    }
    return m_sqrt(y/count);
}

static inline dtype f_min(size_t n, char *p, ssize_t stride)
{
    dtype x,y;
    size_t i=n;

    y = *(dtype*)p;
    p += stride;
    i--;
    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x) && (m_isnan(y) || x<y)) {
            y = x;
        }
        p += stride;
    }
    return y;
}

static inline dtype f_max(size_t n, char *p, ssize_t stride)
{
    dtype x,y;
    size_t i=n;

    y = *(dtype*)p;
    p += stride;
    i--;
    for (; i--;) {
        x = *(dtype*)p;
        if (!m_isnan(x) && (m_isnan(y) || x>y)) {
            y = x;
        }
        p += stride;
    }
    return y;
}

static inline size_t f_min_index(size_t n, char *p, ssize_t stride)
{
    dtype x, y;
    size_t i, j=0;

    y = *(dtype*)p;
    for (i=1; i<n; i++) {
        x = *(dtype*)(p+i*stride);
        if (!m_isnan(x) && (m_isnan(y) || x<y)) {
            y = x;
            j = i;
        }
    }
    return j;
}

static inline size_t f_max_index(size_t n, char *p, ssize_t stride)
{
    dtype x, y;
    size_t i, j=0;

    y = *(dtype*)p;
    for (i=1; i<n; i++) {
        x = *(dtype*)(p+i*stride);
        if (!m_isnan(x) && (m_isnan(y) || x>y)) {
            y = x;
            j = i;
        }
    }
    return j;
}

static inline dtype f_seq(dtype x, dtype y, double c)
{
    return x + y * c;
}
