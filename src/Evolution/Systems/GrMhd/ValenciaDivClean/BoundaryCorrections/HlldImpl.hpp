// Distributed under the MIT License.
// See LICENSE.txt for details.

// Scalar special-relativistic HLLD Riemann solver (Mignone, Ugliano &
// Bodo 2009, MNRAS 393, 1141), for an ideal-gas equation of state with the
// adiabatic index Gamma threaded in explicitly. Used pointwise by the Hlld
// boundary correction; validated against independent reference
// implementations (PLUTO, and E. Most's GRMHD code) on random states.

#pragma once
#include <array>
#include <limits>
#include <cmath>
#include <cstdlib>
#include <string>
#include <tuple>

namespace grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail {

// PLUTO Src/RMHD/hlld.c: "#ifndef HLLD_MAX_ITER / #define HLLD_MAX_ITER 20"
constexpr int HLLD_MAX_ITER = 20;

// Diagnostic counters. The oscillation excess of HLLD over HLLEM at high-order
// reconstruction (7x on ST1/MC) has two candidate causes that only a count can
// separate: the total-pressure root-find failing more often on the harder
// interface states that sharper reconstruction produces, and the physical
// admissibility gate falling back to HLL. Both replace the fan with HLL at
// isolated interfaces, which is itself a noise source. Counters are per-process
// and monotonically increasing; ratios are what matter.
struct Diagnostics {
  size_t fan_attempts = 0;
  size_t uniform_shortcut = 0;     // L == R, HLL is exact
  // Which of PLUTO's eight HLLD_Fstar conditions (hlld.c:479-490) rejects a
  // trial pressure. Counted on every HLLD_Fstar call, so treat as ratios.
  std::array<size_t, 8> term_fail{};
  // Seed disagreement: our seed comes from SpECTRE's KastaunEtAl recovery,
  // PLUTO's from its own ConsToPrim (hlld.c:241-261). HLLState::compute_ptot
  // is our direct analogue of PLUTO's route, so comparing the two isolates
  // the ONE remaining accepted divergence without changing behaviour.
  size_t seed_disagree = 0;        // |p_seed - p_hll| > 10% of p_hll
  size_t seed_compared = 0;
  size_t rootfind_failed = 0;      // secant + sampled search both failed
  // The single "rejected" counter cannot distinguish the two things that were
  // changed together (the seed, and the in-loop admissibility test), so at
  // |B| x8-x32 on CW it reported 99.9% without saying why. Split by cause:
  size_t seed_failed = 0;          // SpECTRE recovery of the HLL average failed
  size_t f0_bad = 0;               // f(p0) NaN or inadmissible at the seed
  size_t fstar_abort = 0;          // trial pressure inadmissible mid-iteration
  size_t iter_exhausted = 0;       // k > 7
  size_t resid_growing = 0;        // |f| grew after k > 4
  size_t converged = 0;            // accepted
  void reset() { *this = Diagnostics{}; }
};
inline Diagnostics& diagnostics() {
  static Diagnostics d{};
  return d;
}



// ideal-gas EOS helpers (Gamma threaded explicitly -- no mutable global state so
// this is safe to call concurrently).  Match ref_hlld IdealEOS.
inline double eos_cs2(double gamma, double rho, double eps) {
  return gamma * (gamma - 1.0) * eps / (1.0 + gamma * eps);
}
// returns p; sets eps_tot = internal-energy density rho*eps
inline double eos_press(double gamma, double& eps_tot, double rho,
                        double eps_th) {
  eps_tot = rho * eps_th;
  return (gamma - 1.0) * rho * eps_th;
}
inline double eos_eps_th(double eps) { return eps; }

enum { DENS = 0, UE, SCX, SCY, SCZ, BBX, BBY, BBZ, TAUE, NUM = 9 };
constexpr bool LEFT = true;
constexpr bool RIGHT = false;

template <bool side, int dir>
struct RecState {
  enum { RHOB = 0, EPS, WVX, WVY, WVZ, BX, BY, BZ };
  std::array<double, NUM> U{}, F{}, R{};
  std::array<double, 4> b{}, u{};
  std::array<double, 2> lambda_all{};
  double gamma{};
  double lambda{}, b2{}, press{}, rhoh{}, eps_tot{}, lorentzi{}, z2{};

  std::array<double, 2> compute_max_characteristicsSR(
      const std::array<double, 9>& P) {
    const double v_limit = 1.0e-12;
    const double b_limit = 1.0e-14;
    const double rho = P[RHOB];
    const double pgas = press;
    (void)pgas;
    const double gamma_rel_sq = 1. + z2;
    const double v_sq = z2 / gamma_rel_sq;
    const double w_gas = rhoh;
    const double cs_sq = eos_cs2(gamma, P[RHOB], P[EPS]);
    const double b_sq = b2;
    const double bbx_sq = P[BX + dir] * P[BX + dir];

    double lambda_plus_no_v, lambda_minus_no_v;
    {
      const double w_tot_inv = 1. / (w_gas + b_sq);
      const double a1 = -(b_sq + cs_sq * (w_gas + bbx_sq)) * w_tot_inv;
      const double a0 = cs_sq * bbx_sq * w_tot_inv;
      const double s2 = a1 * a1 - 4.0 * a0;
      const double s = std::sqrt(0.5 * (std::fabs(s2) + s2));
      const double lambda_sq = 0.5 * (-a1 + s);
      lambda_plus_no_v = std::sqrt(std::fabs(lambda_sq));
      lambda_minus_no_v = -lambda_plus_no_v;
    }

    const double vx = P[WVX + dir] * lorentzi;
    const double vx_sq = P[WVX + dir] * P[WVX + dir] / (1. + z2);
    double lambda_plus_no_bbx, lambda_minus_no_bbx;
    {
      const double v_dot_bb_perp = (b[0] - P[WVX + dir] * P[BX + dir]) * lorentzi;
      const double q = b_sq - cs_sq * (v_dot_bb_perp * v_dot_bb_perp);
      const double denominator_inv =
          1. / (w_gas * (cs_sq + gamma_rel_sq * (1.0 - cs_sq)) + q);
      const double a1 =
          -2.0 * w_gas * gamma_rel_sq * vx * (1.0 - cs_sq) * denominator_inv;
      const double a0 =
          (w_gas * (-cs_sq + gamma_rel_sq * vx_sq * (1.0 - cs_sq)) - q) *
          denominator_inv;
      const double s2 = (a1 * a1) - 4.0 * a0;
      const double s = std::sqrt(0.5 * (std::fabs(s2) + s2));
      bool mask_a1 = (a1 >= 0.0);
      const bool mask_sp = (s2 >= 0.0);
      mask_a1 = mask_a1 && mask_sp;
      const bool mask_am = (!mask_a1) && mask_sp;
      lambda_plus_no_bbx = 0.5 * (-a1 + s);
      lambda_minus_no_bbx = 0.5 * (-a1 - s);
      if (mask_a1) lambda_plus_no_bbx = -2.0 * a0 / (a1 + s);
      if (mask_am) lambda_minus_no_bbx = -2.0 * a0 / (a1 - s);
    }

    double lambda_plus, lambda_minus;
    {
      const double bt_sq = b[0] * b[0];
      const double bx_sq = b[1 + dir] * b[1 + dir];
      const double tmp1 = gamma_rel_sq * gamma_rel_sq * w_gas * (1.0 - cs_sq);
      const double tmp2 = gamma_rel_sq * (b_sq + w_gas * cs_sq);
      const double denominator_inv = 1. / (tmp1 + tmp2 - cs_sq * bt_sq);
      const double a3 =
          (-(4.0 * tmp1 + 2.0 * tmp2) * vx + 2.0 * cs_sq * b[0] * b[1 + dir]) *
          denominator_inv;
      const double a2 = (6.0 * tmp1 * vx_sq + tmp2 * (vx_sq - 1.0) +
                         cs_sq * (bt_sq - bx_sq)) *
                        denominator_inv;
      const double a1 = (-4.0 * tmp1 * vx_sq * vx + 2.0 * tmp2 * vx -
                         2.0 * cs_sq * b[0] * b[1 + dir]) *
                        denominator_inv;
      const double a0 = (tmp1 * vx_sq * vx_sq - tmp2 * vx_sq + cs_sq * bx_sq) *
                        denominator_inv;
      auto SQR = [](double x) { return x * x; };
      const double b2r = a2 - 0.375 * SQR(a3);
      const double b1r = a1 - 0.5 * a2 * a3 + 0.125 * a3 * SQR(a3);
      const double b0r = a0 - 0.25 * a1 * a3 + 0.0625 * a2 * SQR(a3) -
                         3.0 / 256.0 * SQR(SQR(a3));
      double y1, y2, y3, y4;
      {
        const double c2 = -b2r;
        const double c1 = -4.0 * b0r;
        const double c0 = 4.0 * b0r * b2r - SQR(b1r);
        const double q = (c2 * c2 - 3.0 * c1) / 9.0;
        const double r = (2.0 * c2 * c2 * c2 - 9.0 * c1 * c2 + 27.0 * c0) / 54.0;
        const double q3 = q * q * q;
        const double r2 = SQR(r);
        double s2 = r2 - q3;
        const bool mask_s2 = (s2 < 0.0);
        double z0;
        {
          const double s = std::sqrt(std::fabs(s2));
          const double aa =
              -std::copysign(1.0, r) * std::cbrt(std::fabs(r) + s);
          double bb = 0.;
          if (std::fabs(aa) > 0.) bb = q / aa;
          z0 = aa + bb - c2 / 3.0;
          const double theta = std::acos(r / std::sqrt(std::fabs(q3)));
          if (mask_s2)
            z0 = -2.0 * std::sqrt(std::fabs(q)) * std::cos(theta / 3.0) -
                 c2 / 3.0;
        }
        const double z0b2 = z0 - b2r;
        const double d1 = std::sqrt(0.5 * (std::fabs(z0b2) + z0b2));
        const double e1 = -d1;
        s2 = 0.25 * SQR(z0) - b0r;
        const double s = std::sqrt(0.5 * (std::fabs(s2) + s2));
        double d0 = 0.5 * z0 - s;
        double e0 = 0.5 * z0 + s;
        const bool mask_b1 = (b1r < 0);
        if (mask_b1) {
          d0 = 0.5 * z0 + s;
          e0 = 0.5 * z0 - s;
        }
        auto compute_y = [](double f1, double f0) {
          const double s2l = (f1 * f1) - 4.0 * f0;
          const double sl = std::sqrt(0.5 * (std::fabs(s2l) + s2l));
          const bool mask_s22 = (s2l >= 0.);
          const bool mask1 = (f1 < 0.0);
          const bool mask_y1 = mask1 && mask_s22;
          const bool mask_y2 = (!mask1) && mask_s22;
          double yy1 = (-f1 - sl) * 0.5;
          double yy2 = (-f1 + sl) * 0.5;
          if (mask_y1) yy1 = -2.0 * f0 / (f1 - sl);
          if (mask_y2) yy2 = -2.0 * f0 / (f1 + sl);
          return std::make_tuple(yy1, yy2);
        };
        std::tie(y1, y2) = compute_y(d1, d0);
        std::tie(y3, y4) = compute_y(e1, e0);
      }
      lambda_minus = std::min(y1, y3) - 0.25 * a3;
      lambda_plus = std::max(y2, y4) - 0.25 * a3;
    }

    if (v_sq < v_limit) {
      lambda_plus = lambda_plus_no_v;
      lambda_minus = lambda_minus_no_v;
    }
    if (b_sq < b_limit) {
      lambda_plus = lambda_plus_no_bbx;
      lambda_minus = lambda_minus_no_bbx;
    }
    const bool mask_min =
        (std::fabs(lambda_minus) < 1.e-4 && std::fabs(lambda_plus) < 1.e-4);
    if ((lambda_minus < -1.0) || (lambda_minus != lambda_minus) || mask_min)
      lambda_minus = -1.0;
    if ((lambda_plus > 1.0) || (lambda_plus != lambda_plus) || mask_min)
      lambda_plus = 1.0;
    return {lambda_plus, lambda_minus};
  }

  explicit RecState(const std::array<double, 9>& P, double gamma_) : gamma(gamma_) {
    U[BBX] = P[BX];
    U[BBY] = P[BY];
    U[BBZ] = P[BZ];
    z2 = P[WVX] * P[WVX] + P[WVY] * P[WVY] + P[WVZ] * P[WVZ];
    u[0] = std::sqrt(1. + z2);
    u[1] = P[WVX];
    u[2] = P[WVY];
    u[3] = P[WVZ];
    lorentzi = 1. / u[0];
    b[0] = P[WVX] * P[BX] + P[WVY] * P[BY] + P[WVZ] * P[BZ];
    b2 = (P[BX] * P[BX] + P[BY] * P[BY] + P[BZ] * P[BZ] + b[0] * b[0]) *
         lorentzi * lorentzi;
    b[1] = (P[BX] + b[0] * P[WVX]) * lorentzi;
    b[2] = (P[BY] + b[0] * P[WVY]) * lorentzi;
    b[3] = (P[BZ] + b[0] * P[WVZ]) * lorentzi;
    press = eos_press(gamma, eps_tot, P[RHOB], P[EPS]);
    rhoh = press + eps_tot + P[RHOB];
    U[DENS] = P[RHOB] * u[0];
    const double rhohW = (rhoh + b2) * u[0];
    const double ptot = press + 0.5 * b2;
    U[SCX] = rhohW * u[1] - b[0] * b[1];
    U[SCY] = rhohW * u[2] - b[0] * b[2];
    U[SCZ] = rhohW * u[3] - b[0] * b[3];
    U[UE] = rhohW * u[0] - ptot - b[0] * b[0];
    F[UE] = rhohW * u[1 + dir] - b[0] * b[1 + dir];
    U[TAUE] = (eps_tot + 0.5 * b2) * (1. + z2) +
              (ptot + U[DENS] / (u[0] + 1.)) * z2 - b[0] * b[0];
    F[TAUE] = (eps_tot + b2 + press) * u[0] * u[1 + dir] - b[0] * b[1 + dir] +
              P[RHOB] / (u[0] + 1.) * z2 * u[1 + dir];
    lambda_all = compute_max_characteristicsSR(P);
    const double vx = u[1 + dir] * lorentzi;
    const double rhohWvx = rhohW * vx;
    F[DENS] = U[DENS] * vx;
    F[SCX] = rhohWvx * u[1] - b[1] * b[1 + dir];
    F[SCY] = rhohWvx * u[2] - b[2] * b[1 + dir];
    F[SCZ] = rhohWvx * u[3] - b[3] * b[1 + dir];
    F[BBX] = U[BBX] * vx - U[BBX + dir] * u[1] * lorentzi;
    F[BBY] = U[BBY] * vx - U[BBX + dir] * u[2] * lorentzi;
    F[BBZ] = U[BBZ] * vx - U[BBX + dir] * u[3] * lorentzi;
    F[BBX + dir] = 0.;
    F[SCX + dir] += ptot;
  }

  // PLUTO 3d, hlld.c:226-238: R = lambda*U - F, with R[MXn] -= press.
  // PLUTO carries the normal pressure OUTSIDE the flux (sweep->press), so its
  // R[MXn] -= pL[i] is a separate line; our F already includes ptot in the
  // normal momentum (F[SCX+dir] += ptot in the constructor above), so the
  // subtraction is already contained in lambda*U - F and needs no extra term.
  // This replaces Elias' analytic expansion of the same expression, which I
  // verified component-by-component to be algebraically identical.
  void compute_jump() {
    for (int nv = 0; nv < NUM; ++nv) {
      R[nv] = lambda * U[nv] - F[nv];      // PLUTO 230-231
    }
  }
};

template <int dir>
struct RotState {
  static constexpr int BX = (dir + 0) % 3 + BBX;
  static constexpr int BY = (dir + 1) % 3 + BBX;
  static constexpr int BZ = (dir + 2) % 3 + BBX;
  static constexpr int SX = (dir + 0) % 3 + SCX;
  static constexpr int SY = (dir + 1) % 3 + SCX;
  static constexpr int SZ = (dir + 2) % 3 + SCX;
  static constexpr int VX = (dir + 0) % 3;
  static constexpr int VY = (dir + 1) % 3;
  static constexpr int VZ = (dir + 2) % 3;
  std::array<double, NUM> U{};
  std::array<double, 3> K{}, v{};
  double eta{}, rhohb2{}, lambda{};
  // PLUTO Riemann_State::fail (hlld.c:42), set by HLLD_Fstar (hlld.c:490).
  bool fail{};
  bool failed = false;

  template <bool side>
  void update(RecState<side, dir>& S, double ptot) {
    constexpr double sgneta = (side == LEFT) ? -1. : 1.;
    U[BX] = S.U[BX];
    const double mlambda = (1. - S.lambda * S.lambda);
    const double A = S.R[SX] - S.lambda * S.R[UE] + ptot * mlambda;
    const double G = S.R[BY] * S.R[BY] + S.R[BZ] * S.R[BZ];
    const double C = S.R[SY] * S.R[BY] + S.R[SZ] * S.R[BZ];
    // PLUTO's Q (hlld.c:563) appears only inside its MAPLE verification
    // comment, never in the computation; the dead local that mirrored it here
    // is gone.
    const double X = U[BX] * (A * S.lambda * U[BX] + C) -
                     (A + G) * (S.lambda * ptot + S.R[UE]);   // Eq. [30], 525
    v[VX] = (U[BX] * (A * U[BX] + C * S.lambda) - (S.R[SX] + ptot) * (G + A));
    v[VY] = (-(A + G - U[BX] * U[BX] * (1.0 - S.lambda * S.lambda)) * S.R[SY] +
             S.R[BY] * (C + U[BX] * (S.lambda * S.R[SX] - S.R[UE])));
    v[VZ] = (-(A + G - U[BX] * U[BX] * (1.0 - S.lambda * S.lambda)) * S.R[SZ] +
             S.R[BZ] * (C + U[BX] * (S.lambda * S.R[SX] - S.R[UE])));
    rhohb2 = v[VX] * S.R[SX] + v[VY] * S.R[SY] + v[VZ] * S.R[SZ];
    rhohb2 = X * S.R[UE] - rhohb2;
    rhohb2 = ptot + rhohb2 / (X * S.lambda - v[VX]);
    v[VX] /= X;
    v[VY] /= X;
    v[VZ] /= X;
    const double Ai = 1. / A;
    U[BY] = -(S.R[BY] * (S.lambda * ptot + S.R[UE]) - U[BX] * S.R[SY]) * Ai;
    U[BZ] = -(S.R[BZ] * (S.lambda * ptot + S.R[UE]) - U[BX] * S.R[SZ]) * Ai;
    // PLUTO hlld.c:541: `if (Pv->w < 0.0) return 0;` -- a negative enthalpy
    // is a failure, not something to take the absolute value of. The old
    // std::fabs() here silently manufactured a finite eta from an unphysical
    // state and carried on; PLUTO bails. The flag is read where PLUTO reads
    // its HLLD_GetRiemannState return value.
    if (rhohb2 < 0.0) {
      failed = true;
      return;
    }
    failed = false;
    double s = -1.0;                                    // PLUTO 577
    if (U[BX] > 0.) s = 1.0;
    eta = sgneta * s * std::sqrt(rhohb2);               // PLUTO 578-580: Pv->sw
    const double denom = 1. / (S.lambda * ptot + S.R[UE] + U[BX] * eta);
    K[VX] = (S.R[SX] + ptot + S.R[BX] * eta) * denom;
    K[VY] = (S.R[SY] + S.R[BY] * eta) * denom;
    K[VZ] = (S.R[SZ] + S.R[BZ] * eta) * denom;
    lambda = K[VX];
  }

  template <bool side>
  void compute_cons(RecState<side, dir>& S, double ptot) {
    update(S, ptot);
    const double lvi = 1. / (S.lambda - v[VX]);
    const double vdotB = v[VX] * U[BX] + v[VY] * U[BY] + v[VZ] * U[BZ];
    U[DENS] = S.R[DENS] * lvi;
    U[UE] = (S.R[UE] + ptot * v[VX] - vdotB * U[BX]) * lvi;
    U[SX] = (U[UE] + ptot) * v[VX] - vdotB * U[BX];
    U[SY] = (U[UE] + ptot) * v[VY] - vdotB * U[BY];
    U[SZ] = (U[UE] + ptot) * v[VZ] - vdotB * U[BZ];
    U[TAUE] = (S.R[TAUE] + ptot * v[VX] - vdotB * U[BX]) * lvi;
  }
};

template <int dir>
struct CDState {
  static constexpr int BX = (dir + 0) % 3 + BBX;
  static constexpr int BY = (dir + 1) % 3 + BBX;
  static constexpr int BZ = (dir + 2) % 3 + BBX;
  static constexpr int SX = (dir + 0) % 3 + SCX;
  static constexpr int SY = (dir + 1) % 3 + SCX;
  static constexpr int SZ = (dir + 2) % 3 + SCX;
  static constexpr int VX = (dir + 0) % 3;
  static constexpr int VY = (dir + 1) % 3;
  static constexpr int VZ = (dir + 2) % 3;
  std::array<double, NUM> UL{}, UR{};
  std::array<double, 3> vL{}, vR{};
  double lambda{};

  double update(RotState<dir>& LL, RotState<dir>& RR, double ptot) {
    const double dK = (RR.K[VX] - LL.K[VX] + 1.e-12);
    UL[BY] = RR.U[BY] * (RR.K[VX] - RR.v[VX]) - LL.U[BY] * (LL.K[VX] - LL.v[VX]) +
             RR.U[BX] * (RR.v[VY] - LL.v[VY]);
    UL[BZ] = RR.U[BZ] * (RR.K[VX] - RR.v[VX]) - LL.U[BZ] * (LL.K[VX] - LL.v[VX]) +
             RR.U[BX] * (RR.v[VZ] - LL.v[VZ]);
    UL[BX] = 0.5 * (LL.U[BX] + RR.U[BX]) * dK;
    UR[BBX] = UL[BBX];
    UR[BBY] = UL[BBY];
    UR[BBZ] = UL[BBZ];
    auto compute_v = [&](std::array<double, 3>& vv, RotState<dir>& SS,
                         std::array<double, NUM>& UU) {
      const double K2 = SS.K[0] * SS.K[0] + SS.K[1] * SS.K[1] + SS.K[2] * SS.K[2];
      const double mK2 = 1. - K2;
      const double KdotB = SS.K[0] * UU[BBX] + SS.K[1] * UU[BBY] + SS.K[2] * UU[BBZ];
      const double tmp = mK2 / (dK * SS.eta - KdotB);
      vv[VX] = SS.K[VX] - (UU[BX] * tmp);
      vv[VY] = SS.K[VY] - (UU[BY] * tmp);
      vv[VZ] = SS.K[VZ] - (UU[BZ] * tmp);
    };
    const double dKi = 1. / dK;
    compute_v(vL, LL, UL);
    compute_v(vR, RR, UR);
    UR[BX] *= dKi;
    UR[BY] *= dKi;
    UR[BZ] *= dKi;
    UL[BX] *= dKi;
    UL[BY] *= dKi;
    UL[BZ] *= dKi;
    lambda = 0.5 * (vL[VX] + vR[VX]);
    return vL[VX] - vR[VX];
  }

  void compute_cons(RotState<dir>& LL, RotState<dir>& RR, double ptot) {
    update(LL, RR, ptot);
    const double vyc = 0.5 * (vL[VY] + vR[VY]);
    const double vzc = 0.5 * (vL[VZ] + vR[VZ]);
    auto cmp_cons = [&](RotState<dir>& SS, std::array<double, NUM>& UU) {
      const double vdotB = lambda * UU[BX] + vyc * UU[BY] + vzc * UU[BZ];
      const double lvi = 1. / (SS.lambda - lambda + 1.e-12);
      UU[DENS] = SS.U[DENS] * (SS.lambda - SS.v[VX]) * lvi;
      UU[UE] = (SS.lambda * SS.U[UE] - SS.U[SX] + ptot * lambda - vdotB * UL[BX]) *
               lvi;
      UU[SX] = (UU[UE] + ptot) * lambda - vdotB * UU[BX];
      UU[SY] = (UU[UE] + ptot) * vyc - vdotB * UU[BY];
      UU[SZ] = (UU[UE] + ptot) * vzc - vdotB * UU[BZ];
      UU[TAUE] = UU[UE] - UU[DENS];
    };
    cmp_cons(LL, UL);
    cmp_cons(RR, UR);
  }
};

struct HLLState {
  std::array<double, NUM> U{}, F{};
  double gamma{};
  double ptot{}, ptot0{}, b2{};
  static constexpr double c2p_tol = 1.e-12;
  bool failed = false;

  void compute_ptot() {
    auto Uin = U;
    Uin[UE] = Uin[TAUE];
    double& press = ptot;
    press = std::max(0., ptot0);
    double press_prev = 1.e99, press_pp = 1.e109;
    double an_min = 1.e-12;
    bool mask_rhostar_neg = Uin[DENS] < 0;
    if (mask_rhostar_neg) Uin[DENS] = 1.e-15;
    const double rhostari = 1. / Uin[DENS];
    const double B2 =
        (Uin[BBX] * Uin[BBX] + Uin[BBY] * Uin[BBY] + Uin[BBZ] * Uin[BBZ]) *
        rhostari;
    const double tau_min = B2 * 0.5;
    const double taumin1 = Uin[DENS] * tau_min + 1.e-20;
    bool fail_mask = Uin[UE] < taumin1;
    if (fail_mask) Uin[UE] = taumin1;
    const double Snorm2 =
        (Uin[SCX] * Uin[SCX] + Uin[SCY] * Uin[SCY] + Uin[SCZ] * Uin[SCZ]) *
        rhostari * rhostari;
    const double e = Uin[UE] * rhostari + 1.;
    fail_mask = fail_mask || (Snorm2 > e * e);
    fail_mask = fail_mask || mask_rhostar_neg;
    const double SdotB =
        (Uin[BBX] * Uin[SCX] + Uin[BBY] * Uin[SCY] + Uin[BBZ] * Uin[SCZ]) *
        rhostari * std::sqrt(rhostari);
    const double SdotBsq = SdotB * SdotB;
    double lorentz = 1., hWi = 1., hW = 1.;
    (void)lorentz;
    (void)hWi;
    (void)hW;
    double an = (Uin[UE] + press) * rhostari + 1. + tau_min;
    an = std::max(an, an_min);
    const double tmp1 = Snorm2 * B2 - SdotBsq;
    const double dn = 0.25 * (std::fabs(tmp1) + tmp1);
    const double Pmin =
        std::max(Uin[DENS] * (std::cbrt(27. / 4. * dn) - 1. - 0.5 * B2) - Uin[UE],
                 1.e-100);
    press = std::max(press, Pmin);
    constexpr int num_it = 15;
    if (!fail_mask) {
      for (int nn = 0; nn < num_it; ++nn) {
        const double phi = std::acos(std::sqrt(27. / 4. * dn / an) / an);
        const double E1 =
            1. / 3. * an - 2. / 3. * an * std::cos(2. / 3. * phi + 2. / 3. * M_PI);
        hW = E1 - B2;
        hWi = 1. / hW;
        const double hWi2 = hWi * hWi;
        const double vp_sq = (Snorm2 + (2. * hW + B2) * SdotBsq * hWi2);
        const double Z_sq =
            std::min(std::fabs(vp_sq / (((E1 * E1) - vp_sq))), 50. * 50.);
        const double lorentz2 = 1. + Z_sq;
        const double v2 = Z_sq / lorentz2;
        lorentz = std::sqrt(lorentz2);
        const double lorentz2i = 1. / lorentz2;
        const double rho = Uin[DENS] / lorentz;
        const double tau_hydro =
            Uin[UE] - 0.5 * (B2 * (1. + v2) - SdotBsq * hWi2) * Uin[DENS];
        double eps =
            tau_hydro * lorentz2i - (Uin[DENS] / (lorentz + 1.) + press) * v2;
        eps = std::max(eps, 1.e-20);
        const double eps_th = eos_eps_th(eps);
        b2 = B2 * (1. - v2) + SdotBsq * hWi2;
        b2 *= Uin[DENS];
        press_pp = press_prev;
        press_prev = press;
        double eps_tot_dummy;
        press = eos_press(gamma, eps_tot_dummy, rho, eps_th);
        press = std::max(press, Pmin);
        an += (press - press_prev) * rhostari;
        const bool mask_conv =
            std::fabs(press - press_prev) < c2p_tol * std::max(press_prev, press);
        if (mask_conv) break;
        const double Rr = (press - press_prev) / (press_prev - press_pp);
        double Paitken = std::fabs(press_prev + (press - press_prev) / (1. - Rr));
        Paitken = std::max(Paitken, Pmin);
        if (std::fabs(Rr) < 1. && nn > 2) {
          press_pp = press_prev;
          press_prev = press;
          press = Paitken;
          an += (press - press_prev) * rhostari;
        }
        an = std::max(an, an_min);
      }
    }
    failed =
        std::fabs(press - press_prev) > c2p_tol * std::max(press_prev, press);
    press += 0.5 * b2;
  }

  HLLState() = default;

  template <int dir>
  HLLState(RecState<LEFT, dir>& LL, RecState<RIGHT, dir>& RR) {
    const double lambdai = 1. / (RR.lambda - LL.lambda);
    for (int i = 0; i < NUM; ++i) {
      U[i] = (RR.lambda * RR.U[i] - LL.lambda * LL.U[i] + LL.F[i] - RR.F[i]) *
             lambdai;
      F[i] = (RR.lambda * LL.F[i] - LL.lambda * RR.F[i] +
              RR.lambda * LL.lambda * (RR.U[i] - LL.U[i])) *
             lambdai;
    }
    const double a = RR.lambda - LL.lambda;
    const double bq = RR.R[UE] - LL.R[UE] + RR.lambda * LL.R[SCX + dir] -
                      LL.lambda * RR.R[SCX + dir];
    const double cq = LL.R[SCX + dir] * RR.R[UE] - RR.R[SCX + dir] * LL.R[UE];
    const double tmp = bq * bq - 4. * a * cq;
    const double desc = 0.5 * (std::fabs(tmp) + tmp);
    ptot0 = 0.5 * (-bq + std::sqrt(desc)) * lambdai;
  }
};

template <int dir>
struct HLLDSolver {
  RecState<LEFT, dir> LL;
  RecState<RIGHT, dir> RR;
  RotState<dir> rotL, rotR;
  CDState<dir> cd;
  HLLState hll;
  double ptot{};

  HLLDSolver(const std::array<double, 9>& PL, const std::array<double, 9>& PR,
             double gamma)
      : LL(PL, gamma), RR(PR, gamma) {
    LL.lambda = std::min(LL.lambda_all[1], RR.lambda_all[1]);
    RR.lambda = std::max(LL.lambda_all[0], RR.lambda_all[0]);
    LL.compute_jump();
    RR.compute_jump();
    hll = HLLState(LL, RR);
    hll.gamma = gamma;
  }

  // seed_from_hll maps the HLL average conserved state to a total pressure
  // (PLUTO step 3e). Returning a non-finite value means "inversion failed",
  // and we fall back as PLUTO does.
  std::tuple<std::array<double, NUM>, std::array<double, NUM>> solve(
      double ispeed = 0.) {
    return solve(ispeed, [](const std::array<double, NUM>&) {
      return std::numeric_limits<double>::quiet_NaN();
    });
  }

  template <typename SeedFn>
  std::tuple<std::array<double, NUM>, std::array<double, NUM>> solve(
      double ispeed, const SeedFn& seed_from_hll) {
    // PLUTO's HLLD_Fstar returns the jump in normal velocity across the
    // contact AND evaluates whether the resulting state is physical, storing
    // !success in PaL->fail. The iteration then abandons a trial pressure as
    // soon as it produces an unphysical intermediate state, rather than
    // discovering it afterwards. PLUTO stores it in PaL->fail; we store it
    // in rotL.fail, recomputed on every HLLD_Fstar call exactly as PLUTO does.
    auto eq48 = [&](double ptotL) {
      rotL.update(LL, ptotL);
      rotR.update(RR, ptotL);
      const double fun = cd.update(rotL, rotR, ptotL);
      // PLUTO hlld.c:479-490, all eight terms, in PLUTO's order. Mapping:
      //   PaL->Kx  = rotL.K[dir]      PaL->vx = rotL.v[dir]  (a-state velocity)
      //   vxcL     = cd.vL[dir]       PaL->w  = rotL.rhohb2
      //   PaL->Sa  = rotL.lambda      PaL->S  = LL.lambda    (fast speed)
      // PLUTO ASSIGNS here (`=`, not `*=`), discarding the two
      // HLLD_GetRiemannState return values accumulated at hlld.c:444-445; we
      // discard rotL/rotR.failed likewise. Terms 5-6 subsume w < 0 for p > 0.
      // PLUTO uses `*=`, so every term is evaluated -- no short-circuit. We
      // keep that and additionally record WHICH term rejected, which is pure
      // instrumentation: the conjunction below is identical either way.
      const bool cond[8] = {
          (cd.vL[dir] - rotL.K[dir]) > -1.e-6,      // PLUTO 479
          (rotR.K[dir] - cd.vR[dir]) > -1.e-6,      // PLUTO 480
          (LL.lambda - rotL.v[dir]) < 0.0,          // PLUTO 482
          (RR.lambda - rotR.v[dir]) > 0.0,          // PLUTO 483
          (rotR.rhohb2 - ptotL) > 0.0,              // PLUTO 485
          (rotL.rhohb2 - ptotL) > 0.0,              // PLUTO 486
          (rotL.lambda - LL.lambda) > -1.e-6,       // PLUTO 487
          (RR.lambda - rotR.lambda) > -1.e-6};      // PLUTO 488
      bool success = true;
      for (int c = 0; c < 8; ++c) {
        if (not cond[c]) {
          ++diagnostics().term_fail[static_cast<size_t>(c)];
          success = false;
        }
      }
      rotL.fail = not success;                                     // 490
      return fun;
    };
    // DEGENERATE CASE: a (near-)uniform interface. This has NO PLUTO
    // counterpart and is a deliberate, measured exception to the port.
    //
    // I removed it once on the argument that PLUTO's `fabs(f0) > 1.e-12` guard
    // (hlld.c:276) already covers it. That was WRONG: the guard only skips the
    // SECANT, it still assembles the full five-wave fan. PLUTO can afford
    // that because on an exactly uniform state every one of its intermediate
    // states collapses onto the same state and the fan is exact. Our fan
    // carries regularisers PLUTO's does not need at the same places --
    // `dK = dK + 1.e-12` (CDState::update) and `1/(Sa - Sc + 1.e-12)`
    // (CDState::compute_cons) -- and those go degenerate as L -> R, so the
    // assembled flux is noise rather than the uniform flux.
    //
    // Measured: removing this block took ST1 at N=416 with FlatPrim from
    // TV/range 1.00, hi_k 4e-4, L1 1.8e-2 to TV/range 11.7, hi_k 0.36,
    // L1 2.7e-1. FlatPrim is the worst case precisely because 1st order makes
    // L and R EXACTLY equal in smooth regions. HLL is exact on a uniform
    // interface, so returning it is both correct and lossless.
    {
      double jump_magnitude = 0.0, state_magnitude = 0.0;
      for (int k = 0; k < NUM; ++k) {
        jump_magnitude = std::max(jump_magnitude, std::fabs(RR.U[k] - LL.U[k]));
        state_magnitude = std::max(
            state_magnitude, std::max(std::fabs(LL.U[k]), std::fabs(RR.U[k])));
      }
      if (jump_magnitude <= 1.0e-12 * std::max(state_magnitude, 1.0)) {
        ++diagnostics().uniform_shortcut;
        return std::make_tuple(hll.F, hll.U);
      }
    }
    /* --------------------------------------------
       3a. Handle different cases          [PLUTO hlld.c:166-181]
       -------------------------------------------- */
    // Supersonic interfaces never see the fan at all. PLUTO returns the
    // upwind physical flux here, before the HLL average is even formed;
    // ispeed generalises PLUTO's literal 0 to a moving interface.
    if (LL.lambda >= ispeed) {          // PLUTO 171: SL[i] >= 0.0
      return {LL.F, LL.U};              // PLUTO 173-174
    }
    if (RR.lambda <= ispeed) {          // PLUTO 176: SR[i] <= 0.0
      return {RR.F, RR.U};              // PLUTO 178-179
    }

    ++diagnostics().fan_attempts;
    // ---------------------------------------------------------------------
    // Total-pressure root-find, transcribed from PLUTO Src/RMHD/hlld.c steps
    // 3e-3g, keeping PLUTO's variable names so the two can be diffed directly:
    // p0/f0 previous iterate, p/f current, dp the secant step, k the counter,
    // switch_to_hll the fallback flag, HLLD_MAX_ITER the cap.
    //
    // Two things here replaced code inherited from Elias' reference:
    //
    //  * the SEED. PLUTO uses the total pressure of the HLL AVERAGE STATE from
    //    its PRODUCTION conservative-to-primitive inversion. Ours used a
    //    bespoke 15-step fixed point that was 36-44% off on the CW test at low
    //    magnetisation, and no 7-iteration secant can recover from that. The
    //    caller now supplies the seed (see Hlld.cpp) using SpECTRE's own
    //    recovery scheme, which is the equivalent of "use the production
    //    inversion" and keeps the EOS out of this header.
    //
    //  * the CONVERGENCE TEST. The reference is vectorised, so its masks are
    //    SIMD lane predicates and it requires mask_f AND mask_x; PLUTO accepts
    //    EITHER |dp| < 1e-5 p OR |f| < 1e-6, and that difference alone made us
    //    declare failure where PLUTO converges.
    //
    // On failure we do exactly what PLUTO does -- take the HLL flux. We do NOT
    // search for another root; that idea, and why it is dangerous (eq.48 has
    // several roots plus poles, and the wrong branch gives a superluminal
    // contact), is recorded in meetings/BACKLOG.md as experiment B.
    // ---------------------------------------------------------------------
    hll.compute_ptot();
    double p0 = hll.ptot;
    {
      const double p_seed = seed_from_hll(hll.U);      // PLUTO 3e
      if (std::isfinite(p_seed) and p_seed > 0.0) {
        // Instrumentation only: how far our KastaunEtAl-based seed sits from
        // the HLLState::compute_ptot route that mirrors PLUTO's. p0 is still
        // p_seed regardless, so this cannot alter the solution.
        if (p0 > 0.0 and std::isfinite(p0)) {
          ++diagnostics().seed_compared;
          if (std::fabs(p_seed - p0) > 0.1 * p0) {
            ++diagnostics().seed_disagree;
          }
        }
        p0 = p_seed;
      } else {
        ++diagnostics().seed_failed;
        if (LL.U[BBX + dir] * LL.U[BBX + dir] / p0 < 0.01 or hll.failed) {
          p0 = hll.ptot0;                              // PLUTO's B -> 0 branch
        }
      }
    }
    const double pguess = p0;
    (void)pguess;
    bool switch_to_hll = false;
    double p = p0;
    double f0 = eq48(p0);                              // PLUTO 3f
    if (f0 != f0 or rotL.fail) {
      switch_to_hll = true;
      ++diagnostics().f0_bad;
    }
    int k = 0;                                         // PLUTO 3g
    if (std::fabs(f0) > 1.e-12 and not switch_to_hll) {
      p = 1.025 * p0;
      double f = f0;
      for (k = 1; k < HLLD_MAX_ITER; ++k) {
        f = eq48(p);
        if (f != f or rotL.fail or (k > 7) or
            (std::fabs(f) > std::fabs(f0) and k > 4)) {
          auto& dg = diagnostics();
          if (f != f or rotL.fail) {
            ++dg.fstar_abort;
          } else if (k > 7) {
            ++dg.iter_exhausted;
          } else {
            ++dg.resid_growing;
          }
          switch_to_hll = true;
          break;
        }
        const double dp = (p - p0) / (f - f0) * f;
        p0 = p;
        f0 = f;
        p -= dp;
        if (p < 0.0) {
          p = 1.e-6;
        }
        if (std::fabs(dp) < 1.e-5 * p or std::fabs(f) < 1.e-6) {
          break;
        }
      }
    } else {
      p = p0;
    }
    ptot = p;

    /* ----  too many iter ? --> use HLL ----      [PLUTO hlld.c:303-315] */
    // rotL.fail carries the verdict of the LAST HLLD_Fstar call, exactly as
    // PLUTO's PaL.fail does. PLUTO does NOT re-evaluate Fstar at the final p,
    // so neither do we -- the a/c states are rebuilt at p below instead, via
    // HLLD_GetAState / HLLD_GetCState, which is what PLUTO does too.
    //
    // An admissibility gate on the assembled flux used to sit at the end of
    // this function. It is gone: PLUTO has no counterpart, and experiment 1b
    // measured the identical test against PLUTO's own fan (hlld.c:321-353) --
    // PLUTO violates the envelope on up to 22% of interfaces while remaining
    // stable and accurate, so the condition is not a physical requirement and
    // the gate manufactured spurious fallbacks. PLUTO's whole fallback policy
    // is these two lines.
    if (rotL.fail) {                            // PLUTO 305
      switch_to_hll = true;
    }
    if (switch_to_hll) {                        // PLUTO 306
      ++diagnostics().rootfind_failed;          // PLUTO 308-310 COUNT_FAILURES
      return {hll.F, hll.U};                    // PLUTO 312-314
    }
    ++diagnostics().converged;

    /* -- ok, solution should be reliable --       [PLUTO hlld.c:317-406] */
    // PLUTO branches with plain if/else on Sa and Sc; the mask ladder that
    // used to be here came from Elias' DataVector-vectorised reference, where
    // branching is impossible. This solver is scalar, so PLUTO's control flow
    // ports directly. ispeed generalises PLUTO's literal 0 to a moving
    // interface (SpECTRE infrastructure); ispeed == 0 reproduces PLUTO.
    std::array<double, NUM> flux{}, cons{};
    if (rotL.lambda >= ispeed - 1.e-6) {          // PLUTO 359: PaL.Sa >= -1.e-6
      rotL.compute_cons(LL, ptot);                // PLUTO 361: HLLD_GetAState
      for (int i = 0; i < NUM; ++i) {             // PLUTO 365-367
        flux[i] = LL.F[i] + LL.lambda * (rotL.U[i] - LL.U[i]);
      }
      cons = rotL.U;
    } else if (rotR.lambda <= ispeed + 1.e-6) {   // PLUTO 370: PaR.Sa <= 1.e-6
      rotR.compute_cons(RR, ptot);                // PLUTO 372
      for (int i = 0; i < NUM; ++i) {             // PLUTO 376-378
        flux[i] = RR.F[i] + RR.lambda * (rotR.U[i] - RR.U[i]);
      }
      cons = rotR.U;
    } else {                                      // PLUTO 381
      // HLLD_GetCState calls HLLD_GetAState on both sides internally
      // (PLUTO 634-635, 676, 681), so both a-states are rebuilt first here.
      rotL.compute_cons(LL, ptot);
      rotR.compute_cons(RR, ptot);
      cd.compute_cons(rotL, rotR, ptot);          // PLUTO 383: HLLD_GetCState
      if (cd.lambda > ispeed) {                   // PLUTO 384: Sc > 0.0
        for (int i = 0; i < NUM; ++i) {           // PLUTO 389-392
          flux[i] = LL.F[i] + LL.lambda * (rotL.U[i] - LL.U[i]) +
                    rotL.lambda * (cd.UL[i] - rotL.U[i]);
        }
        cons = cd.UL;
      } else {                                    // PLUTO 395
        for (int i = 0; i < NUM; ++i) {           // PLUTO 400-403
          flux[i] = RR.F[i] + RR.lambda * (rotR.U[i] - RR.U[i]) +
                    rotR.lambda * (cd.UR[i] - rotR.U[i]);
        }
        cons = cd.UR;
      }
    }
    return {flux, cons};
  }
};

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail
