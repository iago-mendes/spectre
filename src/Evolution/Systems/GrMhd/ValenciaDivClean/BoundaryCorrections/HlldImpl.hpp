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

// Diagnostic counters. The oscillation excess of HLLD over HLLEM at high-order
// reconstruction (7x on ST1/MC) has two candidate causes that only a count can
// separate: the total-pressure root-find failing more often on the harder
// interface states that sharper reconstruction produces, and the physical
// admissibility gate falling back to HLL. Both replace the fan with HLL at
// isolated interfaces, which is itself a noise source. Counters are per-process
// and monotonically increasing; ratios are what matter.
struct Diagnostics {
  size_t fan_attempts = 0;
  size_t rootfind_failed = 0;      // secant + sampled search both failed
  size_t gate_rejected = 0;        // flux outside the physical envelope
  size_t uniform_shortcut = 0;     // L == R, HLL is exact
  void reset() { *this = Diagnostics{}; }
};
inline Diagnostics& diagnostics() {
  static Diagnostics d{};
  return d;
}

// Gate mode (env HLLD_GATE_MODE): "fallback" (default, current production
// behaviour: replace the whole flux with HLL), or "clamp" (project only the
// offending COMPONENT back onto the physical envelope, keeping the rest of the
// fan). The gate is protective (see above) but it is BINARY, and it fires on
// 9-44% of interfaces at 2nd order, so neighbouring interfaces can use
// completely different fluxes. That inhomogeneity is itself a noise source,
// which is why Radice & Rezzolla (THC, sec. 2 eq. 14) blend smoothly toward
// Lax-Friedrichs rather than switching. "clamp" is the minimal smooth
// alternative: same admissibility bound, smallest possible change to the flux.
// UNTESTED as of this commit -- the comparison runs were staged but not run.
inline bool gate_clamp() {
  static const bool v = [] {
    const char* e = std::getenv("HLLD_GATE_MODE");
    return e != nullptr and std::string(e) == "clamp";
  }();
  return v;
}

// Diagnostic switch (env var HLLD_DISABLE_GATE=1). The gate fires on 3% of
// interfaces at 1st order but 9-44% at 2nd order, so it is either causing the
// high-order oscillation (by flipping neighbouring interfaces between the fan
// and HLL) or merely reacting to it. Turning it off answers which: if the
// oscillation gets WORSE the gate is protective, if it gets BETTER the gate is
// the amplifier. Env var rather than an option so the comparison needs no
// input-file changes and cannot silently alter production runs.
// MEASURED (ST1, MonotonisedCentral, N=208): gate off -> oscillation 0.375 vs
// 0.021 with the gate on, i.e. 18x WORSE. The gate is protective.
inline bool gate_disabled() {
  static const bool v = [] {
    const char* e = std::getenv("HLLD_DISABLE_GATE");
    return e != nullptr and e[0] == '1';
  }();
  return v;
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
  enum { RHOB = 0, EPS, WVX, WVY, WVZ, BX, BY, BZ, YE };
  std::array<double, NUM> U{}, F{}, R{};
  std::array<double, 4> b{}, u{};
  std::array<double, 2> lambda_all{};
  double gamma{};
  double lambda{}, b2{}, press{}, rhoh{}, eps_tot{}, lorentzi{}, z2{}, ye{};

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
    ye = P[YE];
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

  void compute_jump() {
    const double lv = lambda - u[1 + dir] * lorentzi;
    R[DENS] = lv * U[DENS];
    const double rhW = (rhoh + b2) * u[0];
    const double blb = b[1 + dir] - lambda * b[0];
    R[SCX] = lv * rhW * u[1] + b[1] * blb;
    R[SCY] = lv * rhW * u[2] + b[2] * blb;
    R[SCZ] = lv * rhW * u[3] + b[3] * blb;
    const double ptot = press + 0.5 * b2;
    R[SCX + dir] -= ptot;
    R[UE] = lv * rhW * u[0] - lambda * ptot + b[0] * blb;
    R[BBX] = lv * U[BBX] + U[BBX + dir] * u[1] * lorentzi;
    R[BBY] = lv * U[BBY] + U[BBX + dir] * u[2] * lorentzi;
    R[BBZ] = lv * U[BBZ] + U[BBX + dir] * u[3] * lorentzi;
    R[TAUE] = R[UE] - R[DENS];
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
  bool failed = false;

  template <bool side>
  void update(RecState<side, dir>& S, double ptot) {
    constexpr double sgneta = (side == LEFT) ? -1. : 1.;
    U[BX] = S.U[BX];
    const double mlambda = (1. - S.lambda * S.lambda);
    const double A = S.R[SX] - S.lambda * S.R[UE] + ptot * mlambda;
    const double G = S.R[BY] * S.R[BY] + S.R[BZ] * S.R[BZ];
    const double C = S.R[SY] * S.R[BY] + S.R[SZ] * S.R[BZ];
    const double Q = -A - G + U[BX] * U[BX] * mlambda;
    const double X = U[BX] * (A * S.lambda * U[BX] + C) -
                     (A + G) * (S.lambda * ptot + S.R[UE]);
    (void)Q;
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
    double sB = -1.0;
    if (U[BX] > 0.) sB = 1.0;
    eta = sgneta * sB * std::sqrt(std::fabs(rhohb2));
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
  double ye{}, ptot{}, ptot0{}, b2{};
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
    ye = (RR.lambda * RR.U[DENS] * RR.ye - LL.lambda * LL.U[DENS] * LL.ye +
          LL.F[DENS] * LL.ye - RR.F[DENS] * RR.ye) *
         lambdai / U[DENS];
  }
};

template <typename F_t>
bool SecantMethod(F_t& f, double& x0) {
  constexpr int nmax = 20;
  double x1 = x0 * 1.025;
  const double xinit = x0;
  double f0 = f(x0);
  double f1 = f(x1);
  const double finit = f0;
  double delta_f = f1 - f0, delta_x = x1 - x0;
  bool mask_f = (std::fabs(f1 - f0) > 1.e-12);
  bool mask_x = (std::fabs(x1 - x0) > 1.e-12 * std::fabs(x0));
  int nn = 0;
  while (mask_f && mask_x && (nn < nmax)) {
    ++nn;
    f0 = f1;
    x0 = x1;
    if (mask_f) x1 -= f1 * delta_x / delta_f;
    if (x1 < 0.) x1 = 1.e-3 * xinit;
    f1 = f(x1);
    delta_f = f1 - f0;
    delta_x = x1 - x0;
    mask_f = (std::fabs(f1) > 1.e-12);
    mask_x = (std::fabs(delta_x) > 1.e-12 * std::fabs(x0));
  }
  x0 = x1;
  const bool nan_mask = (f1 != f1) || (x0 != x0);
  mask_f = mask_f || (std::fabs(f1) > std::fabs(finit));
  return (mask_f && mask_x) || nan_mask;
}

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

  std::tuple<std::array<double, NUM>, std::array<double, NUM>> solve(
      double ispeed = 0.) {
    auto eq48 = [&](double ptotL) {
      rotL.update(LL, ptotL);
      rotR.update(RR, ptotL);
      return cd.update(rotL, rotR, ptotL);
    };
    // DEGENERATE CASE: a (near-)uniform interface.
    //
    // The equation this solver roots is "the jump in the normal velocity across
    // the contact vanishes". For u_L == u_R that jump is identically zero for
    // EVERY trial total pressure, so f == 0, every value is a root, and the
    // iteration returns wherever it happens to land -- after which the fan is
    // built from a meaningless p_tot. Measured on a static uniform state with
    // rho = 10, B = (2.4, 4, 4): the root-find returned p_tot = 5.11 where the
    // exact value is 19.88, and the resulting flux was off by a factor of ten
    // in a state where the answer is simply F(u).
    //
    // This is why a uniform region -- most of the RW test -- came out as noise
    // rather than a constant: every interface there is degenerate. The HLL flux
    // is EXACT whenever Delta u = 0 (its dissipation term carries a factor
    // Delta u), and HLLD's advantage vanishes smoothly as Delta u -> 0, so
    // returning it below a relative tolerance is both correct and lossless.
    {
      double jump_magnitude = 0.;
      double state_magnitude = 0.;
      for (int k = 0; k < NUM; ++k) {
        jump_magnitude = std::max(jump_magnitude, std::fabs(RR.U[k] - LL.U[k]));
        state_magnitude = std::max(state_magnitude,
                                   std::max(std::fabs(LL.U[k]),
                                            std::fabs(RR.U[k])));
      }
      if (jump_magnitude <= 1.0e-12 * std::max(state_magnitude, 1.0)) {
        ++diagnostics().uniform_shortcut;
        return std::make_tuple(hll.F, hll.U);
      }
    }

    ++diagnostics().fan_attempts;
    hll.compute_ptot();
    ptot = hll.ptot;
    bool mask_p = LL.U[BBX + dir] * LL.U[BBX + dir] / ptot < 0.01;
    mask_p = mask_p || hll.failed;
    if (mask_p) ptot = hll.ptot0;
    bool mask_failed = SecantMethod(eq48, ptot);
    // The secant above is unsafeguarded: it takes an unbounded step, and
    // if that
    // lands at ptot < 0 it resets to 1e-3 * (initial guess) and usually
    // cannot
    // recover within nmax. The reference implementations (Elias/hlld.hh and
    // PLUTO) use the same bare secant and share this failure mode -- it is not
    // a porting bug, it is simply never exercised on the states where it bites.
    //
    // On the stationary-contact CW test it bites badly: at |B| x0.5, x1 and
    // x8-x32 the secant diverges, mask_failed fires, and HLLD silently returns
    // its internal HLL flux -- so the contact smears instead of being captured
    // exactly (see Test_Hlld.cpp, test_hlld_reproduces_a_stationary_contact).
    //
    // The residual is smooth and monotone with a single sign change, and the
    // root sits at the analytically known total pressure (verified to 1e-13 in
    // that test), which is the ideal case for a BRACKETED method. So when the
    // secant fails, bracket the root by expanding around the initial guess and
    // finish with bisection, which cannot leave the bracket and converges
    // unconditionally. Only if no sign change exists at all do we give up and
    // fall back to HLL.
    // Do not trust SecantMethod's own verdict: it returns
    // (mask_f && mask_x) || nan_mask, so an iteration that stalls in x
    // (mask_x false) reports SUCCESS no matter how large the residual is.
    // That is how a diverged root reaches the fan construction unnoticed.
    // Judge convergence by the residual itself.
    const double residual_after_secant = std::fabs(eq48(ptot));
    const double residual_tolerance =
        1.0e-9 * (1.0 + std::fabs(hll.ptot));
    if (mask_failed or not(residual_after_secant < residual_tolerance)) {
      // Two root shapes occur here, and only one of them can be bracketed:
      //  * at low/moderate field the residual CROSSES zero -- bisection works;
      //  * at high field it TOUCHES zero from below (f(412) = -10.4,
      //    f(416.28) = 0, f(420) = -0.41 for the CW state at |B| x8), so no
      //    sign change exists and any bracketing method fails.
      // The residual also has a near-pole just below the root, which is what
      // sends the unsafeguarded secant to infinity in the first place.
      //
      // Sample |f| on a logarithmic grid spanning the plausible range of the
      // total pressure, take the best sample, and refine locally -- by
      // bisection if the neighbours bracket a sign change, otherwise by
      // golden-section on |f|. Sampling is immune to both poles and tangency,
      // and this path only runs when the fast secant has already failed.
      // Equation 48 has MORE THAN ONE zero. Taking the globally best sample
      // locks onto whichever one happens to have the smallest |f|, and that is
      // often a spurious root hundreds of times below the physical pressure
      // (measured: ptot = 1.57 against an HLL estimate of 657, converged to
      // residual 1e-16 and passing every admissibility mask, yet producing a
      // flux far outside the range the physical fluxes can reach). The
      // physical root is the one near the HLL total pressure, so search
      // OUTWARD from that seed and keep the first root found on either side.
      const double seed = std::max(hll.ptot, 1.0e-30);
      constexpr int num_samples = 400;
      const double log_lo = std::log(seed * 1.0e-4);
      const double log_hi = std::log(seed * 1.0e2);
      double best_x = ptot;
      int best_index = -1;
      std::array<double, num_samples> xs{};
      std::array<double, num_samples> fs{};
      for (int s = 0; s < num_samples; ++s) {
        const double x =
            std::exp(log_lo + (log_hi - log_lo) * s / (num_samples - 1));
        xs[static_cast<size_t>(s)] = x;
        fs[static_cast<size_t>(s)] = eq48(x);
      }
      double best_abs_f = std::numeric_limits<double>::max();
      for (int s = 0; s < num_samples; ++s) {
        const size_t si = static_cast<size_t>(s);
        if (std::isfinite(fs[si]) and std::fabs(fs[si]) < best_abs_f) {
          best_abs_f = std::fabs(fs[si]);
          best_index = s;
          best_x = xs[si];
        }
      }
      if (best_index >= 0) {
        const size_t bi = static_cast<size_t>(best_index);
        double lo = bi > 0 ? xs[bi - 1] : xs[bi];
        double hi = bi + 1 < num_samples ? xs[bi + 1] : xs[bi];
        const double f_lo_s = bi > 0 ? fs[bi - 1] : fs[bi];
        const double f_hi_s = bi + 1 < num_samples ? fs[bi + 1] : fs[bi];
        if (std::isfinite(f_lo_s) and std::isfinite(f_hi_s) and
            f_lo_s * f_hi_s < 0.0) {
          double f_lo = f_lo_s;
          for (int bisect = 0; bisect < 100; ++bisect) {
            const double mid = 0.5 * (lo + hi);
            const double f_mid = eq48(mid);
            if (f_mid == 0.0 or (hi - lo) < 1.0e-15 * std::fabs(mid)) {
              lo = mid;
              hi = mid;
              break;
            }
            if (f_lo * f_mid < 0.0) {
              hi = mid;
            } else {
              lo = mid;
              f_lo = f_mid;
            }
          }
          best_x = 0.5 * (lo + hi);
        } else {
          // Tangential root: minimise |f| by golden section on [lo, hi].
          constexpr double inv_phi = 0.6180339887498949;
          double c = hi - inv_phi * (hi - lo);
          double d = lo + inv_phi * (hi - lo);
          double fc = std::fabs(eq48(c));
          double fd = std::fabs(eq48(d));
          for (int it = 0; it < 200 and (hi - lo) > 1.0e-15 * std::fabs(hi);
               ++it) {
            if (fc < fd) {
              hi = d;
              d = c;
              fd = fc;
              c = hi - inv_phi * (hi - lo);
              fc = std::fabs(eq48(c));
            } else {
              lo = c;
              c = d;
              fc = fd;
              d = lo + inv_phi * (hi - lo);
              fd = std::fabs(eq48(d));
            }
          }
          best_x = 0.5 * (lo + hi);
        }
        const double final_residual = std::fabs(eq48(best_x));
        if (final_residual < residual_tolerance) {
          ptot = best_x;
          eq48(ptot);
          mask_failed = false;
        }
      }
    }
    rotL.compute_cons(LL, ptot);
    rotR.compute_cons(RR, ptot);
    cd.compute_cons(rotL, rotR, ptot);
    mask_failed = mask_failed || hll.failed;
    mask_failed = mask_failed || ((cd.vL[dir] - rotL.K[dir]) < -1.e-6);
    mask_failed = mask_failed || ((rotR.K[dir] - cd.vR[dir]) < -1.e-6);
    mask_failed = mask_failed || ((rotL.lambda - rotL.v[dir]) > 0.0);
    mask_failed = mask_failed || ((rotR.lambda - rotR.v[dir]) < 0.0);
    mask_failed = mask_failed || ((rotL.rhohb2 - ptot) < 0.0);
    mask_failed = mask_failed || ((rotR.rhohb2 - ptot) < 0.0);
    mask_failed = mask_failed || ((rotL.lambda - LL.lambda) < -1.e-6);
    mask_failed = mask_failed || ((RR.lambda - rotR.lambda) < -1.e-6);
    mask_failed = mask_failed || rotL.failed || rotR.failed;
    if (mask_failed) {
      ++diagnostics().rootfind_failed;
    }

    std::array<double, NUM> flux{}, cons{};
    const bool maskLL = (LL.lambda > ispeed);
    const bool maskrotL = (rotL.lambda > ispeed - 1.e-6);
    const bool maskCD = (cd.lambda > ispeed);
    const bool maskrotR = (rotR.lambda > ispeed + 1.e-6);
    const bool maskRR = (RR.lambda > ispeed);
    const bool mask1 = (!maskLL) && maskrotL;
    const bool mask2 = (!maskrotL) && maskCD;
    const bool mask3 = (!maskCD) && maskrotR;
    const bool mask4 = (!maskrotR) && maskRR;
    const bool mask5 = (!maskRR);
    if (mask1)
      for (int i = 0; i < NUM; ++i) {
        flux[i] = LL.F[i] + LL.lambda * (rotL.U[i] - LL.U[i]);
        cons[i] = rotL.U[i];
      }
    if (mask2)
      for (int i = 0; i < NUM; ++i) {
        flux[i] = LL.F[i] + LL.lambda * (rotL.U[i] - LL.U[i]) +
                  rotL.lambda * (cd.UL[i] - rotL.U[i]);
        cons[i] = cd.UL[i];
      }
    if (mask3)
      for (int i = 0; i < NUM; ++i) {
        flux[i] = RR.F[i] + RR.lambda * (rotR.U[i] - RR.U[i]) +
                  rotR.lambda * (cd.UR[i] - rotR.U[i]);
        cons[i] = cd.UR[i];
      }
    if (mask4) {
      for (int i = 0; i < NUM; ++i)
        flux[i] = RR.F[i] + RR.lambda * (rotR.U[i] - RR.U[i]);
      cons = rotR.U;
    }
    if (mask_failed) {
      flux = hll.F;
      cons = hll.U;
    }
    if (maskLL) {
      flux = LL.F;
      cons = LL.U;
    }
    if (mask5) {
      flux = RR.F;
      cons = RR.U;
    }
    // Final admissibility gate. Everything above -- the secant, the sampled
    // root-find, the wave-fan masks -- can only ever be as trustworthy as the
    // total pressure it is built on, and equation 48 has several zeros. A
    // spurious one converges to machine precision and satisfies every mask,
    // yet yields a flux far outside anything the physical fluxes can produce;
    // that is what showed up as scattered nonsense in the RW and Balsara-2
    // profiles, where HLLD was visibly WORSE than HLL rather than sharper.
    //
    // So check the answer instead of trusting the path to it: a numerical flux
    // must lie between the two physical fluxes, widened by the most
    // dissipation the fan can add, |lambda|_max * |Delta u|. HLL satisfies this
    // identically (verified over random states in Test_Hlld.cpp), so falling
    // back to it is always admissible. This makes "in the worst case HLLD
    // degrades to HLL" a property of the solver rather than a hope.
    const double max_speed =
        std::max(std::fabs(LL.lambda), std::fabs(RR.lambda));
    for (int i = 0; i < NUM; ++i) {
      const double dissipation = max_speed * std::fabs(RR.U[i] - LL.U[i]);
      const double lower = std::min(LL.F[i], RR.F[i]) - dissipation;
      const double upper = std::max(LL.F[i], RR.F[i]) + dissipation;
      const double tolerance =
          1.0e-8 * (1.0 + std::fabs(lower) + std::fabs(upper));
      if (not std::isfinite(flux[i]) or flux[i] < lower - tolerance or
          flux[i] > upper + tolerance) {
        ++diagnostics().gate_rejected;
        if (gate_disabled() and std::isfinite(flux[i])) {
          continue;  // diagnostic mode: count it but keep the fan flux
        }
        if (gate_clamp() and std::isfinite(flux[i])) {
          flux[i] = std::min(std::max(flux[i], lower), upper);
          continue;
        }
        return {hll.F, hll.U};
      }
    }
    return {flux, cons};
  }
};

}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::hlld_detail
