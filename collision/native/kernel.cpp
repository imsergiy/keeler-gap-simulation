#include <cmath>
#include <cstddef>

// Negative-tail ratios J_n/J_(n-1) avoid subtracting nearly equal moments.
static void moments(double h, double variance, double* result) {
    if (variance == 0.0) {
        const double x = h > 0.0 ? h : 0.0;
        result[0] = x; result[1] = x*x; result[2] = x*x*x;
        return;
    }
    constexpr double invsqrt2pi = 0.39894228040143267794;
    const double s = std::sqrt(variance), t = h/s;
    const double phi = std::exp(-0.5*t*t)*invsqrt2pi;
    if (t < -4.0) {
        if (phi == 0.0) { result[0] = result[1] = result[2] = 0.0; return; }
        const double u = -t;
        double ratio = 0.0, r1 = 0.0, r2 = 0.0, r3 = 0.0;
        for (int n = 256; n >= 1; --n) {
            ratio = static_cast<double>(n)/(u + ratio);
            if (n == 3) r3 = ratio;
            if (n == 2) r2 = ratio;
            if (n == 1) r1 = ratio;
        }
        const double j1 = r1/(u + r1);
        result[0] = s*phi*j1;
        result[1] = variance*phi*r2*j1;
        result[2] = s*variance*phi*r3*r2*j1;
        return;
    }
    const double probability = 0.5*std::erfc(-t/std::sqrt(2.0));
    result[0] = h*probability + s*phi;
    result[1] = (h*h+variance)*probability + h*s*phi;
    result[2] = (h*h*h+3.0*h*variance)*probability + (h*h+2.0*variance)*s*phi;
}

extern "C" int gaussian_normal_moments(double h, double variance, double* result) {
    if (!std::isfinite(h) || !std::isfinite(variance) || variance < 0.0) return 1;
    moments(h, variance, result);
    return 0;
}

static int evaluate_batch(
    std::size_t count, const double* C, const double* A, const double* N,
    const double* H, const double* Y, double a, double e,
    std::size_t directions, const double* normals, const double* weights,
    double* Q, double* P, double* heat, double* rate, std::size_t* bad_index,
    double* L, double* B) {
    constexpr double inv2sqrtpi = 0.28209479177387814347;
    const double one_plus_e = 1.0+e;
    for (std::size_t cell = 0; cell < count; ++cell) {
        double q[9] = {}, p[9] = {}, l[9] = {}, b[9] = {}, dissipation = 0.0, encounters = 0.0;
        const double* c = C+9*cell;
        const double* grad = A+9*cell;
        for (std::size_t direction = 0; direction < directions; ++direction) {
            const double* n = normals+3*direction;
            double cn[3] = {}, an[3] = {};
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    cn[i] += c[3*i+j]*n[j];
                    an[i] += grad[3*i+j]*n[j];
                }
            double variance = 0.0, h = 0.0;
            for (int i = 0; i < 3; ++i) {
                variance += 2.0*n[i]*cn[i];
                h += -2.0*a*n[i]*an[i];
            }
            if (variance < 0.0) { *bad_index = cell; return 1; }
            double m[3]; moments(h,variance,m);
            const double az = a*n[2]/H[cell];
            const double weight = weights[direction]*std::exp(-az*az);
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    const int ij = 3*i+j;
                    const double dyad = n[i]*n[j];
                    q[ij] += weight*(-2.0*one_plus_e*m[0]*(n[i]*cn[j]+cn[i]*n[j])
                                +0.5*one_plus_e*one_plus_e*m[2]*dyad);
                    p[ij] += weight*m[1]*dyad;
                    if (L) { l[ij] += weight*m[0]*dyad; b[ij] += weight*m[2]*dyad; }
                }
            dissipation += weight*m[2]; encounters += weight*m[0];
        }
        const double pref = N[cell]*Y[cell]*inv2sqrtpi/H[cell];
        for (int ij = 0; ij < 9; ++ij) {
            Q[9*cell+ij] = 2.0*pref*a*a*q[ij];
            P[9*cell+ij] = 2.0*one_plus_e*pref*a*a*a*p[ij];
            if (L) {
                L[9*cell+ij] = 4.0*one_plus_e*pref*a*a*l[ij];
                B[9*cell+ij] = one_plus_e*one_plus_e*pref*a*a*b[ij];
            }
        }
        heat[cell] = 0.5*(1.0-e*e)*pref*a*a*dissipation;
        rate[cell] = 4.0*pref*a*a*encounters;
    }
    return 0;
}

extern "C" int gaussian_batch(
    std::size_t count, const double* C, const double* A, const double* N,
    const double* H, const double* Y, double a, double e,
    std::size_t directions, const double* normals, const double* weights,
    double* Q, double* P, double* heat, double* rate, std::size_t* bad_index) {
    return evaluate_batch(count,C,A,N,H,Y,a,e,directions,normals,weights,Q,P,heat,rate,bad_index,nullptr,nullptr);
}

extern "C" int gaussian_batch_loss_gain(
    std::size_t count, const double* C, const double* A, const double* N,
    const double* H, const double* Y, double a, double e,
    std::size_t directions, const double* normals, const double* weights,
    double* Q, double* P, double* heat, double* rate, std::size_t* bad_index,
    double* L, double* B) {
    return evaluate_batch(count,C,A,N,H,Y,a,e,directions,normals,weights,Q,P,heat,rate,bad_index,L,B);
}
