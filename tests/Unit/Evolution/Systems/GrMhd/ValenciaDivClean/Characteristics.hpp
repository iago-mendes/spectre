// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cmath>
#include <cstddef>

#include <boost/multiprecision/cpp_bin_float.hpp>

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/TestingFramework.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace grmhd::ValenciaDivClean::TestHelpers {

namespace quad_precision {
using Quad = boost::multiprecision::cpp_bin_float_quad;

template <typename Real>
Real sq(const Real& x) {
  return x * x;
}

template <typename Real>
Real cube(const Real& x) {
  return x * x * x;
}

template <typename Real>
Real sign_of(const Real& x) {
  return x >= Real{0.0} ? Real{1.0} : Real{-1.0};
}

template <typename Real>
std::array<Real, 4> magnetosonic_quartic_coefficients(
    const Real& sound_speed_squared, const Real& normal_velocity,
    const Real& lorentz_factor, const Real& normal_magnetic_field,
    const Real& magnetic_field_dot_spatial_velocity,
    const Real& magnetic_field_squared,
    const Real& comoving_magnetic_field_squared) {
  const Real& cs2 = sound_speed_squared;
  const Real& b2scaled = comoving_magnetic_field_squared;
  const Real& Bvscaled = magnetic_field_dot_spatial_velocity;
  const Real& Bsscaled = normal_magnetic_field;
  const Real& vn = normal_velocity;
  const Real& W = lorentz_factor;
  (void)magnetic_field_squared;

  // Cache inverse denominators to avoid repeated full-precision divisions.
  Real inv_core_denom =
      Real{1.0} / (b2scaled + square(W) -
                   cs2 * (-Real{1.0} + square(Bvscaled) + square(W)));

  std::array<Real, 4> quartic_coefficients{};

  // 1 / (W^2 * (...))
  inv_core_denom /= square(W);
  quartic_coefficients[1] =
      -(Real{2.0} * Bsscaled * Bvscaled * cs2 +
        Real{2.0} * vn * square(W) *
            (-b2scaled + (-Real{1.0} + square(Bvscaled)) * cs2 -
             Real{2.0} * (-Real{1.0} + cs2) * square(vn) * square(W))) *
      inv_core_denom;
  quartic_coefficients[3] =
      -(-Real{2.0} * Bsscaled * Bvscaled * cs2 +
        Real{2.0} * vn * square(W) *
            (b2scaled + Real{2.0} * square(W) -
             cs2 * (-Real{1.0} + square(Bvscaled) + Real{2.0} * square(W)))) *
      inv_core_denom;

  // 1 / (W^4 * (...))
  inv_core_denom /= square(W);
  quartic_coefficients[0] =
      -(-square(Bsscaled) * cs2 -
        Real{2.0} * Bsscaled * Bvscaled * cs2 * vn * square(W) +
        square(vn) * square(square(W)) *
            (b2scaled - (-Real{1.0} + square(Bvscaled)) * cs2 +
             (-Real{1.0} + cs2) * square(vn) * square(W))) *
      inv_core_denom;
  quartic_coefficients[2] =
      -(square(Bsscaled) * cs2 +
        Real{2.0} * Bsscaled * Bvscaled * cs2 * vn * square(W) -
        (b2scaled - (-Real{1.0} + square(Bvscaled)) * cs2) *
            (-Real{1.0} + square(vn)) * square(square(W)) +
        Real{6.0} * (-Real{1.0} + cs2) * square(vn) * square(square(W)) *
            square(W)) *
      inv_core_denom;
  return quartic_coefficients;
}

template <typename Real>
Real quartic_value(const Real& x, const std::array<Real, 4>& c) {
  return (((x + c[3]) * x + c[2]) * x + c[1]) * x + c[0];
}

template <typename Real>
Real quartic_derivative(const Real& x, const std::array<Real, 4>& c) {
  return ((Real{4.0} * x + Real{3.0} * c[3]) * x + Real{2.0} * c[2]) * x + c[1];
}

template <typename Real>
Real find_magnetosonic_speed_from_quartic(const std::array<Real, 4>& c,
                                          const Real initial_guess) {
  // This function has been optimized by Codex to minimize memory allocation and
  // speed it up. See unoptimized::find_magnetosonic_speed_from_quartic in
  // Test_Characteristics.cpp for an easier-to-read implementation.

  constexpr size_t max_iters = 100;
  const Real tolerance = Real{"1e-30"};

  // We define the coefficients so that the quartic polynomial is
  // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
  Real x = initial_guess;
  const Real& c0 = c[0];
  const Real& c1 = c[1];
  const Real& c2 = c[2];
  const Real& c3 = c[3];

  // Find the root using Newton-Rapshon
  Real F{};
  Real dF{};
  for (size_t iter = 0; iter < max_iters; ++iter) {
    // Horner form minimizes intermediate temporary vectors in the hot loop.
    // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
    F = (((x + c3) * x + c2) * x + c1) * x + c0;

    if (abs(F) < tolerance) {
      return x;
    }

    // F'(x) = 4 x^3 + 3 c3 x^2 + 2 c2 x + c1
    dF = ((Real{4.0} * x + Real{3.0} * c3) * x + Real{2.0} * c2) * x + c1;

    // Avoid FPE from dividing by a small derivative
    if (abs(dF) < tolerance) {
      if (abs(F) < tolerance) {
        // If a point has converged and has a small derivative, then just
        // skip the Newton step.
        return x;
      } else if (abs(F) >= tolerance and abs(dF) < tolerance) {
        // If a point that hasn't converged has a small derivative, then the
        // Newton step would be unreliable, so we error out.
        CAPTURE(x);
        CAPTURE(F);
        CAPTURE(dF);
        ERROR(
            "Failed to compute magnetosonic speed from quartic: derivative "
            "is too small for a reliable Newton step. "
            "x = "
            << x << ", F = " << F << ", dF = " << dF);
        return x;
      }
    }

    // Newton step
    x -= F / dF;
  }

  F = (((x + c3) * x + c2) * x + c1) * x + c0;
  ERROR(
      "Failed to compute magnetosonic speed from quartic: exceeded maximum "
      "number of iterations. Max |F| = "
      << abs(F));
  return x;
}

std::array<Quad, 9> characteristic_speeds_mhd(
    const std::array<Quad, 3>& spatial_velocity,
    const std::array<Quad, 3>& magnetic_field, const Quad& rest_mass_density,
    const Quad& specific_internal_energy, const Quad& lorentz_factor,
    const Quad& specific_enthalpy, const std::array<std::array<Quad, 3>, 3>& g,
    const std::array<Quad, 3>& unit_normal) {
  const Quad eps_floor{"1e-30"};
  const Quad gamma_minus_one =
      abs(specific_internal_energy) > eps_floor
          ? ((specific_enthalpy - Quad{1.0}) - specific_internal_energy) /
                specific_internal_energy
          : Quad{0.0};

  const Quad pressure =
      gamma_minus_one * rest_mass_density * specific_internal_energy;
  const Quad sound_speed_squared =
      (gamma_minus_one * specific_internal_energy +
       sq(gamma_minus_one) * specific_internal_energy) /
      specific_enthalpy;

  std::array<Quad, 3> magnetic_field_one_form{};
  for (size_t i = 0; i < 3; ++i) {
    magnetic_field_one_form[i] = Quad{0.0};
    for (size_t j = 0; j < 3; ++j) {
      magnetic_field_one_form[i] += g[i][j] * magnetic_field[j];
    }
  }

  Quad normal_velocity{0.0};
  Quad normal_magnetic_field{0.0};
  Quad magnetic_field_dot_spatial_velocity{0.0};
  Quad magnetic_field_squared{0.0};
  for (size_t i = 0; i < 3; ++i) {
    normal_velocity += spatial_velocity[i] * unit_normal[i];
    normal_magnetic_field += magnetic_field[i] * unit_normal[i];
    magnetic_field_dot_spatial_velocity +=
        magnetic_field_one_form[i] * spatial_velocity[i];
    magnetic_field_squared += magnetic_field_one_form[i] * magnetic_field[i];
  }
  (void)pressure;

  const Quad comoving_magnetic_field_squared =
      magnetic_field_squared / sq(lorentz_factor) +
      sq(magnetic_field_dot_spatial_velocity);
  const Quad rho_h_star =
      rest_mass_density * specific_enthalpy + comoving_magnetic_field_squared;

  // Re-scale magnetic intermediate variables so that they are dimensionless
  const Quad inv_rho_h = Quad{1.0} / (rest_mass_density * specific_enthalpy);
  const Quad inv_sqrt_rho_h = sqrt(inv_rho_h);
  const Quad normal_magnetic_field_scaled =
      normal_magnetic_field * inv_sqrt_rho_h;
  const Quad magnetic_field_dot_spatial_velocity_scaled =
      magnetic_field_dot_spatial_velocity * inv_sqrt_rho_h;
  const Quad magnetic_field_squared_scaled = magnetic_field_squared * inv_rho_h;
  const Quad comoving_magnetic_field_squared_scaled =
      comoving_magnetic_field_squared * inv_rho_h;

  std::array<Quad, 9> characteristic_speeds{};

  // Entropy speed
  characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::Entropy] =
      normal_velocity;

  // Scalar speeds
  characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::ScalarPlus] =
      Quad{1.0};
  characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::ScalarMinus] =
      Quad{-1.0};

  const Quad alfven_1 =
      normal_velocity +
      normal_magnetic_field / sq(lorentz_factor) /
          (magnetic_field_dot_spatial_velocity + sqrt(rho_h_star));
  const Quad alfven_2 =
      normal_velocity +
      normal_magnetic_field / sq(lorentz_factor) /
          (magnetic_field_dot_spatial_velocity - sqrt(rho_h_star));
  characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus] =
      std::max(alfven_1, alfven_2);
  characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus] =
      std::min(alfven_1, alfven_2);

  // Fast magnetosonic speeds
  /*
    Find fast magnetosonic speeds via rootfinding of a quartic polynomial with
    initial guess of +1 (for positive speed) or -1 (for negative speed).
  */
  const std::array<Quad, 4> quartic_coefficients =
      magnetosonic_quartic_coefficients(
          sound_speed_squared, normal_velocity, lorentz_factor,
          normal_magnetic_field_scaled,
          magnetic_field_dot_spatial_velocity_scaled,
          magnetic_field_squared_scaled,
          comoving_magnetic_field_squared_scaled);
  characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus] = Quad{1.0};
  characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus] =
          find_magnetosonic_speed_from_quartic(
              quartic_coefficients,
              characteristic_speeds
                  [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus]);
  characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus] = Quad{-1.0};
  characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus] =
          find_magnetosonic_speed_from_quartic(
              quartic_coefficients,
              characteristic_speeds
                  [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus]);

  // Slow magnetosonic speeds
  /*
    TO-DO
  */
  // Type I threshold: set tight (quad machine epsilon scale) so the
  // bracketed root solver finds the true slow roots in the near-degenerate
  // slow-mode-compression regime, rather than snapping to vn.  This lets
  // the comparison measure the actual error introduced by the production
  // code's 1e-14 threshold.
  const Quad type_i_threshold = Quad{"1e-28"};
  // Type II thresholds stay matched to the production code's 1e-14 value.
  // When alfven coincides with a slow root the bracket [alfven_eps, vn]
  // would have same-signed endpoints, so the snap is still needed for
  // correctness; fixing that blind spot is left for future work.
  const Quad type_ii_threshold = Quad{"1e-28"};
  const Quad eps = Quad{"1e-12"};
  Quad& slow_minus = characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus];
  Quad& slow_plus = characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus];
  const Quad& vn = normal_velocity;
  const Quad& alfven_minus =
      characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus];
  const Quad& alfven_plus =
      characteristic_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus];
  const Quad& fast_minus = characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus];
  const Quad& fast_plus = characteristic_speeds
      [grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus];
  const Quad& c0 = quartic_coefficients[0];
  const Quad& c1 = quartic_coefficients[1];
  const Quad& c2 = quartic_coefficients[2];
  const Quad& c3 = quartic_coefficients[3];

  const auto evaluate_quartic = [&c0, &c1, &c2, &c3](const Quad y) {
    return (((y + c3) * y + c2) * y + c1) * y + c0;
  };

  const Quad vn_i = vn;
  const Quad alfven_minus_i = alfven_minus;
  const Quad alfven_minus_eps_i = alfven_minus_i + eps;
  const Quad alfven_plus_i = alfven_plus;
  const Quad alfven_plus_eps_i = alfven_plus_i + eps;

  const Quad N_vn = evaluate_quartic(vn_i);
  const Quad N_alfven_minus = evaluate_quartic(alfven_minus_i);
  const Quad N_alfven_minus_eps = evaluate_quartic(alfven_minus_eps_i);
  const Quad N_alfven_plus = evaluate_quartic(alfven_plus_i);
  const Quad N_alfven_plus_eps = evaluate_quartic(alfven_plus_eps_i);
  const auto bracketed_root_solve =
      [&evaluate_quartic](const Quad lower, const Quad upper,
                          const Quad f_lower, const Quad f_upper,
                          const Quad abs_tol, const Quad rel_tol) {
        Quad a = lower;
        Quad b = upper;
        Quad fa = f_lower;
        Quad fb = f_upper;
        ASSERT(fa * fb <= Quad{0.0},
               "Quad bracketed root solve requires opposite-signed endpoints. "
                   << "lower = " << a << ", upper = " << b
                   << ", f(lower) = " << fa << ", f(upper) = " << fb);
        for (size_t iter = 0; iter < 100; ++iter) {
          const Quad mid = Quad{0.5} * (a + b);
          const Quad fmid = evaluate_quartic(mid);
          if (abs(fmid) < abs_tol or
              abs(b - a) < rel_tol * std::max(Quad{1.0}, abs(mid))) {
            return mid;
          }
          if (fa * fmid <= Quad{0.0}) {
            b = mid;
            fb = fmid;
          } else {
            a = mid;
            fa = fmid;
          }
        }
        return Quad{0.5} * (a + b);
      };
  // Check if we have one of the possible degeneracies and use itto avoid
  // rootfinding for the slow roots
  if (abs(N_vn) < type_i_threshold) {
    // Type I: alfven- = slow- = entropy = slow+ = alfven+
    slow_minus = vn_i;
    slow_plus = vn_i;
  } else if (abs(N_alfven_minus) < type_ii_threshold and
             N_alfven_minus_eps > Quad{0.0}) {
    // Type II on the minus side: alfven- = slow-
    slow_minus = alfven_minus_i;
    slow_plus = -c3 - slow_minus - fast_minus - fast_plus;
  } else if (abs(N_alfven_plus) < type_ii_threshold and
             N_alfven_plus_eps < Quad{0.0}) {
    // Type II on the plus side: slow+ = alfven+
    slow_plus = alfven_plus_i;
    slow_minus = -c3 - fast_minus - fast_plus - slow_plus;
  } else {
    CAPTURE(alfven_minus_i);
    CAPTURE(alfven_minus_eps_i);
    CAPTURE(vn_i);
    CAPTURE(alfven_plus_i);
    CAPTURE(alfven_plus_eps_i);
    CAPTURE(N_alfven_minus);
    CAPTURE(N_alfven_minus_eps);
    CAPTURE(N_vn);
    slow_minus =
        bracketed_root_solve(alfven_minus_eps_i, vn_i, N_alfven_minus_eps, N_vn,
                             Quad{"5e-16"}, Quad{"5e-16"});
    slow_plus = -c3 - slow_minus - fast_minus - fast_plus;
  }

  ASSERT(abs(evaluate_quartic(slow_minus)) < Quad{"1e-12"} and
             abs(evaluate_quartic(slow_plus)) < Quad{"1e-12"},
         "Failed to find slow magnetosonic speeds: slow_minus = "
             << slow_minus << ", slow_plus = " << slow_plus
             << ", quartic(slow_minus) = " << evaluate_quartic(slow_minus)
             << ", quartic(slow_plus) = " << evaluate_quartic(slow_plus));
  return characteristic_speeds;
}

bool characteristic_eigenvectors_mhd(
    std::array<std::array<Quad, 9>, 9>* right_eigenvectors,
    std::array<std::array<Quad, 9>, 9>* left_eigenvectors,
    const std::array<Quad, 9>& characteristic_speeds,
    const std::array<Quad, 3>& spatial_velocity,
    const std::array<Quad, 3>& magnetic_field, const Quad& rest_mass_density,
    const Quad& specific_internal_energy, const Quad& lorentz_factor,
    const Quad& specific_enthalpy, const std::array<std::array<Quad, 3>, 3>& g,
    const std::array<Quad, 3>& unit_normal,
    const std::array<Quad, 3>& tangent_1,
    const std::array<Quad, 3>& tangent_2) {
  const Quad eps_floor{"1e-30"};
  const Quad gamma_minus_one =
      abs(specific_internal_energy) > eps_floor
          ? ((specific_enthalpy - Quad{1.0}) - specific_internal_energy) /
                specific_internal_energy
          : Quad{0.0};
  const Quad pressure =
      gamma_minus_one * rest_mass_density * specific_internal_energy;
  const Quad sound_speed_squared =
      (gamma_minus_one * specific_internal_energy +
       sq(gamma_minus_one) * specific_internal_energy) /
      specific_enthalpy;
  const Quad kappa =
      pressure != Quad{0.0} ? gamma_minus_one * rest_mass_density : Quad{0.0};

  std::array<Quad, 3> v_cov{};
  std::array<Quad, 3> B_cov{};
  const Quad det_g = g[0][0] * (g[1][1] * g[2][2] - g[1][2] * g[2][1]) -
                     g[0][1] * (g[1][0] * g[2][2] - g[1][2] * g[2][0]) +
                     g[0][2] * (g[1][0] * g[2][1] - g[1][1] * g[2][0]);
  const Quad inv_det_g = Quad{1.0} / det_g;
  std::array<std::array<Quad, 3>, 3> inv_g{};
  inv_g[0][0] = (g[1][1] * g[2][2] - g[1][2] * g[2][1]) * inv_det_g;
  inv_g[0][1] = (g[0][2] * g[2][1] - g[0][1] * g[2][2]) * inv_det_g;
  inv_g[0][2] = (g[0][1] * g[1][2] - g[0][2] * g[1][1]) * inv_det_g;
  inv_g[1][0] = (g[1][2] * g[2][0] - g[1][0] * g[2][2]) * inv_det_g;
  inv_g[1][1] = (g[0][0] * g[2][2] - g[0][2] * g[2][0]) * inv_det_g;
  inv_g[1][2] = (g[0][2] * g[1][0] - g[0][0] * g[1][2]) * inv_det_g;
  inv_g[2][0] = (g[1][0] * g[2][1] - g[1][1] * g[2][0]) * inv_det_g;
  inv_g[2][1] = (g[0][1] * g[2][0] - g[0][0] * g[2][1]) * inv_det_g;
  inv_g[2][2] = (g[0][0] * g[1][1] - g[0][1] * g[1][0]) * inv_det_g;
  std::array<Quad, 3> s_vec{};
  std::array<Quad, 3> tangent_1_up{};
  std::array<Quad, 3> tangent_2_up{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_cov[i] += g[i][j] * spatial_velocity[j];
      B_cov[i] += g[i][j] * magnetic_field[j];
      s_vec[i] += inv_g[i][j] * unit_normal[j];
      tangent_1_up[i] += inv_g[i][j] * tangent_1[j];
      tangent_2_up[i] += inv_g[i][j] * tangent_2[j];
    }
  }
  Quad v_n{0.0};
  Quad v_1{0.0};
  Quad v_2{0.0};
  Quad B_n{0.0};
  Quad B_1{0.0};
  Quad B_2{0.0};
  Quad B_squared{0.0};
  Quad B_dot_v{0.0};
  for (size_t i = 0; i < 3; ++i) {
    v_n += spatial_velocity[i] * unit_normal[i];
    v_1 += spatial_velocity[i] * tangent_1[i];
    v_2 += spatial_velocity[i] * tangent_2[i];
    B_n += magnetic_field[i] * unit_normal[i];
    B_1 += magnetic_field[i] * tangent_1[i];
    B_2 += magnetic_field[i] * tangent_2[i];
    B_squared += magnetic_field[i] * B_cov[i];
    B_dot_v += B_cov[i] * spatial_velocity[i];
  }

  const Quad cs2 = sound_speed_squared;
  const Quad rho = rest_mass_density;
  const Quad h = specific_enthalpy;
  const Quad W = lorentz_factor;
  const Quad b_squared = B_squared / sq(W) + sq(B_dot_v);
  const Quad rho_h_star = rho * h + b_squared;
  const Quad h_star = h + b_squared / rho;
  const Quad sqrt_rho_h_star = sqrt(rho_h_star);
  const Quad r_1 = B_dot_v + sqrt_rho_h_star;
  // const Quad r_2 = B_n * v_n - r_1;
  const Quad r_4 = B_squared + r_1 * B_dot_v * sq(W);
  const Quad B_21 = B_2 * v_1 - B_1 * v_2;
  const Quad B_31 = B_n * v_1 - B_1 * v_n;
  const Quad B_32 = B_n * v_2 - B_2 * v_n;

  std::array<std::array<Quad, 9>, 9> right{};
  std::array<std::array<Quad, 9>, 9> left{};
  for (size_t wave = 0; wave < 9; ++wave) {
    const Quad y = characteristic_speeds[wave];
    const Quad a = W * (v_n - y);
    const Quad B = B_n / W + B_dot_v * W * (v_n - y);
    const Quad G = Quad{1.0} - sq(y);
    const Quad script_G = rho * h * sq(a) - G * b_squared;
    const Quad script_G_rho = script_G / (rho * h * cs2);
    const Quad kappa_rho = kappa + rho * cs2;
    const Quad Z = rho * h * sq(W);
    const Quad K = -W * (Quad{1.0} - v_n * y);
    const Quad kappa_B =
        kappa_rho * sq(B) + (Quad{1.0} - cs2) * sq(rho * a) * h;
    const Quad kappa_Bv = kappa_B * B_dot_v - kappa_rho * rho * a * B * h_star;
    const Quad eps{"1e-14"};
    const Quad a_denom = a + eps;
    const Quad G_denom = G + eps;
    const Quad cs2_denom = cs2 + eps;

    if (wave == grmhd::ValenciaDivClean::MhdSpeed::Entropy) {
      for (size_t i = 0; i < 3; ++i) {
        right[wave][i] = h * W * (kappa - rho * cs2) * spatial_velocity[i];
        right[wave][3 + i] = Quad{0.0};
      }
      right[wave][6] = kappa;
      right[wave][7] = kappa * (h * W - Quad{1.0}) - rho * h * W * cs2;
      right[wave][8] = Quad{0.0};

      const Quad G_entropy = Quad{1.0} - sq(v_n);
      const Quad entropy_norm = Quad{1.0} / (rho * h * cs2);
      for (size_t i = 0; i < 3; ++i) {
        // S_b components
        left[wave][i] = entropy_norm * W * v_cov[i];
        // B_b components: gamma_{ba}b^a - (B_n / GW) s_b
        const Quad b_cov_i = B_cov[i] / W + W * v_cov[i] * B_dot_v;
        left[wave][3 + i] =
            entropy_norm * (b_cov_i - (B_n / (G_entropy * W)) * unit_normal[i]);
      }
      // D component
      left[wave][6] = entropy_norm * (h - W);
      // tau component
      left[wave][7] = entropy_norm * (-W);
      // phi component: W B^a v_a - B_n v_n / GW
      left[wave][8] =
          entropy_norm * (W * B_dot_v - (B_n * v_n) / (G_entropy * W));
    } else if (wave == grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus or
               wave == grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus) {
      // Choose the +-sqrt(rho h*) branch to match this wave's stored speed
      // (Eq 2.74); see the production code for the rationale.
      const Quad y_plus_branch =
          v_n + B_n / (sq(W) * (B_dot_v + sqrt_rho_h_star));
      const Quad y_minus_branch =
          v_n + B_n / (sq(W) * (B_dot_v - sqrt_rho_h_star));
      const Quad alf_sign = abs(y - y_plus_branch) <= abs(y - y_minus_branch)
                                ? Quad{1.0}
                                : Quad{-1.0};
      const Quad sqrt_rho_h_star_s = alf_sign * sqrt_rho_h_star;
      const Quad r_1_s = B_dot_v + sqrt_rho_h_star_s;
      const Quad r_4_s = B_squared + r_1_s * B_dot_v * sq(W);
      const Quad y_Alf = y;
      const Quad inv_sqrt_rho_h_star_s = Quad{1.0} / sqrt_rho_h_star_s;

      // Right eigenvector frame (s, t1, t2) components (Teukolsky Eq. 3.38).
      const Quad r_modes_s_n =
          -Quad{2.0} * sqrt_rho_h_star_s * B_21 * (B_n + r_1_s * v_n * sq(W));
      const Quad r_modes_s_1 =
          -sqrt_rho_h_star_s *
          (B_n * B_32 + B_1 * B_21 +
           r_1_s * sq(W) * (B_2 + v_1 * B_21 + v_n * B_32));
      const Quad r_modes_s_2 =
          sqrt_rho_h_star_s * (B_n * B_31 - B_2 * B_21 +
                               r_1_s * sq(W) * (B_1 - v_2 * B_21 + v_n * B_31));
      const Quad r_modes_b_1 = sqrt_rho_h_star_s * B_2 + v_2 * r_4_s;
      const Quad r_modes_b_2 = -sqrt_rho_h_star_s * B_1 - v_1 * r_4_s;

      // Left eigenvector frame components (Teukolsky Eq. 3.41).
      const Quad l_proj_s_n = inv_sqrt_rho_h_star_s * (B_21 * y_Alf);
      const Quad l_proj_s_1 = inv_sqrt_rho_h_star_s * (B_2 + B_32 * y_Alf);
      const Quad l_proj_s_2 = inv_sqrt_rho_h_star_s * (-B_1 - B_31 * y_Alf);
      const Quad l_proj_b_n = -B_21 * y_Alf;
      const Quad l_proj_b_1 = -(B_2 + B_32 * y_Alf);
      const Quad l_proj_b_2 = (B_1 + B_31 * y_Alf);

      // Rotate the frame components into the coordinate basis.
      for (size_t i = 0; i < 3; ++i) {
        right[wave][i] = r_modes_s_n * s_vec[i] +
                         r_modes_s_1 * tangent_1_up[i] +
                         r_modes_s_2 * tangent_2_up[i];
        right[wave][3 + i] =
            r_modes_b_1 * tangent_1_up[i] + r_modes_b_2 * tangent_2_up[i];
        left[wave][i] = l_proj_s_n * unit_normal[i] +
                        l_proj_s_1 * tangent_1[i] + l_proj_s_2 * tangent_2[i];
        left[wave][3 + i] = l_proj_b_n * unit_normal[i] +
                            l_proj_b_1 * tangent_1[i] +
                            l_proj_b_2 * tangent_2[i];
      }
      right[wave][6] = -rho * W * B_21;
      right[wave][7] =
          -B_21 * W * (Quad{2.0} * W * sqrt_rho_h_star_s * r_1_s - rho);
      right[wave][8] = Quad{0.0};
      left[wave][6] = inv_sqrt_rho_h_star_s * (-B_21);
      left[wave][7] = inv_sqrt_rho_h_star_s * (-B_21);
      left[wave][8] = -B_21;
    } else if (wave == grmhd::ValenciaDivClean::MhdSpeed::ScalarMinus or
               wave == grmhd::ValenciaDivClean::MhdSpeed::ScalarPlus) {
      // Scalar right eigenvector, Teukolsky Eq. (4.32), with the two author-
      // confirmed typo corrections (see the production code / paper-corrections
      // report): row 1 momentum  W^2 B kappa_B (s+yv) -> W^2 kappa_Bv (s+yv);
      // row 3 (D) 2nd term  (1-cs^2) rho a B_n -> (1-cs^2) rho^2 a B_n.
      for (size_t i = 0; i < 3; ++i) {
        right[wave][i] =
            -((y * kappa_B + Quad{2.0} * kappa_rho * a * B * B_n) *
                  magnetic_field[i] +
              sq(W) * kappa_Bv * (s_vec[i] + y * spatial_velocity[i]) -
              Quad{2.0} * W * kappa_rho * B * sq(B_n) * spatial_velocity[i]) /
            (a_denom * W);
        right[wave][3 + i] = kappa_rho * y * B * magnetic_field[i] / W +
                             (s_vec[i] - y * spatial_velocity[i]) *
                                 (kappa_rho * B * B_n +
                                  (Quad{1.0} - cs2) * sq(rho) * sq(a) * h * W) /
                                 a_denom;
      }
      // D component (Teukolsky Eq. 4.32, row 3, with rho -> rho^2 in 2nd term).
      right[wave][6] =
          kappa_rho * y * rho * B - (Quad{1.0} - cs2) * sq(rho) * a * B_n;
      right[wave][7] =
          (kappa_rho * B * (Quad{2.0} * sq(B_n) + rho * a * (a * h_star - y)) -
           kappa_B * B - Quad{2.0} * kappa_Bv * y * W +
           (Quad{1.0} - cs2) * sq(rho * a) * B_n) /
          a_denom;
      right[wave][8] = -(Quad{1.0} - cs2) * sq(rho * a) * h;

      const Quad inv_one_minus_vn2 = Quad{1.0} / (Quad{1.0} - sq(v_n) + eps);
      for (size_t i = 0; i < 3; ++i) {
        left[wave][i] = Quad{0.0};
        left[wave][3 + i] = unit_normal[i] * inv_one_minus_vn2;
      }
      left[wave][6] = Quad{0.0};
      left[wave][7] = Quad{0.0};
      left[wave][8] = y * inv_one_minus_vn2;
    } else {
      const Quad m_1s = rho * h * a * W * (B * B_dot_v - rho_h_star * a);
      const Quad m_1v = rho * h *
                        (B_n * B_dot_v * (y * a + Quad{2.0} * G * W) -
                         Quad{2.0} * a * sq(B_n) -
                         a * W *
                             (y * a * (B_squared / sq(W) + rho * h) +
                              (Quad{1.0} - Quad{1.0} / cs2) * script_G * W));
      const Quad m_1B =
          rho * h * (B * (y * a - G * W) + Quad{2.0} * B_n * (sq(a) + G)) / W;
      const Quad m_4 =
          (rho / a_denom) *
          (sq(a) * sq(B) * h - Quad{2.0} * sq(B_n) * h * (sq(a) + G) +
           sq(B) * W * (Quad{2.0} * y * a * h - G) +
           B_n * B * (G + Quad{2.0} * h * W * G - Quad{2.0} * y * a * h) +
           script_G * W * sq(a) * (h * W * (Quad{1.0} - cs2) - Quad{1.0}) /
               cs2_denom +
           rho * h * cube(a) *
               (y - Quad{2.0} * y * h_star * W + a * (W - h_star)));
      for (size_t i = 0; i < 3; ++i) {
        right[wave][i] = m_1s * s_vec[i] + m_1v * spatial_velocity[i] +
                         m_1B * magnetic_field[i];
        right[wave][3 + i] = rho * h * a *
                             (magnetic_field[i] * (Quad{1.0} - y * v_n) -
                              B_n * (s_vec[i] - y * spatial_velocity[i]));
      }
      right[wave][6] =
          -rho * B * G * B_n / a_denom - sq(rho) * a * h * (y * a - G * W);
      right[wave][7] = m_4;
      right[wave][8] = Quad{0.0};

      const Quad f_1v =
          W * (-G + B * G * B_n * W / (Z * sq(a_denom)) +
               script_G * sq(W) * (kappa + rho) / (Z * rho * cs2_denom));
      const Quad g_1B =
          script_G * kappa * W / (rho * Z * cs2_denom) - (sq(a) + G) / W;
      const Quad g_1v = B_dot_v * sq(W) * g_1B +
                        W * (a * B + script_G * B * sq(W) / (Z * a_denom));
      const Quad h_1 = -(f_1v + y * a);
      for (size_t i = 0; i < 3; ++i) {
        left[wave][i] = a * unit_normal[i] -
                        B * G * W * B_cov[i] / (Z * a_denom) + f_1v * v_cov[i];
        left[wave][3 + i] =
            B * unit_normal[i] + g_1B * B_cov[i] + g_1v * v_cov[i] +
            -(script_G_rho * kappa_rho * B / (G_denom * rho)) * unit_normal[i];
      }
      left[wave][6] =
          h_1 + script_G * (kappa - rho * cs2) / (sq(rho) * cs2_denom);
      left[wave][7] = h_1;
      left[wave][8] = (B * K * (G * rho - script_G_rho * kappa_rho) +
                       G * B_n * (rho * (sq(a) + G) - script_G_rho * kappa)) /
                      (G_denom * rho * a_denom);
    }
  }

  *right_eigenvectors = right;
  *left_eigenvectors = left;
  return true;
}

// Faithful quad port of grmhd::ValenciaDivClean::flux_jacobian_mhd (the double
// implementation in src/.../Characteristics.cpp).  Variable order of the 9x9
// output is [S_x, S_y, S_z, B^x, B^y, B^z, D, tau, phi]; entry [m][n] is row m,
// column n.  The double function calls the EoS to obtain c_s^2 and
// kappa = dp/d(eps); because the double EoS cannot be evaluated at quad
// precision, the caller supplies quad-precision sound_speed_squared and kappa
// (computed EoS-consistently, exactly as characteristic_eigenvectors_mhd
// derives them for an ideal fluid).  Every arithmetic entry below mirrors the
// double implementation with Quad substituted for double.
std::array<std::array<Quad, 9>, 9> flux_jacobian_mhd(
    const std::array<Quad, 3>& spatial_velocity,
    const std::array<Quad, 3>& magnetic_field, const Quad& rest_mass_density,
    const Quad& /*specific_internal_energy*/, const Quad& /*electron_fraction*/,
    const Quad& lorentz_factor, const Quad& specific_enthalpy,
    const std::array<std::array<Quad, 3>, 3>& spatial_metric,
    const std::array<std::array<Quad, 3>, 3>& inv_spatial_metric,
    const std::array<Quad, 3>& unit_normal, const Quad& sound_speed_squared,
    const Quad& kappa) {
  // Named quantities matching the double implementation (matrix.txt).
  const Quad& soundSpeedSquared = sound_speed_squared;
  const Quad& restMassDensity = rest_mass_density;
  const Quad& lorentzFactor = lorentz_factor;
  const Quad& specificEnthalpy = specific_enthalpy;
  const std::array<Quad, 3>& spatialVelocity = spatial_velocity;
  const std::array<Quad, 3>& magneticField = magnetic_field;
  const std::array<Quad, 3>& unitNormal = unit_normal;

  std::array<Quad, 3> unitVector{};              // s^i = gamma^{ij} n_j
  std::array<Quad, 3> spatialVelocityOneForm{};  // v_i
  std::array<Quad, 3> magneticFieldOneForm{};    // B_i
  for (size_t i = 0; i < 3; ++i) {
    unitVector[i] = Quad{0.0};
    spatialVelocityOneForm[i] = Quad{0.0};
    magneticFieldOneForm[i] = Quad{0.0};
    for (size_t j = 0; j < 3; ++j) {
      unitVector[i] += inv_spatial_metric[i][j] * unit_normal[j];
      spatialVelocityOneForm[i] += spatial_metric[i][j] * spatial_velocity[j];
      magneticFieldOneForm[i] += spatial_metric[i][j] * magnetic_field[j];
    }
  }

  Quad normalVelocity{0.0};
  Quad normalMagneticField{0.0};
  Quad magneticFieldDotVelocity{0.0};
  Quad magneticFieldSquared{0.0};
  for (size_t i = 0; i < 3; ++i) {
    normalVelocity += spatial_velocity[i] * unit_normal[i];
    normalMagneticField += magnetic_field[i] * unit_normal[i];
    magneticFieldDotVelocity += magnetic_field[i] * spatialVelocityOneForm[i];
    magneticFieldSquared += magnetic_field[i] * magneticFieldOneForm[i];
  }
  const Quad comovingMagneticFieldSquared =
      magneticFieldSquared / sq(lorentzFactor) + sq(magneticFieldDotVelocity);
  const Quad Zvar = restMassDensity * specificEnthalpy * sq(lorentzFactor);
  const Quad Dvar = restMassDensity * lorentzFactor;

  const Quad two{2.0};
  const Quad one{1.0};
  const Quad four{4.0};

  // --- generated matrix entries (As, from matrix.txt) ---
  const auto x0 = Zvar + magneticFieldSquared;
  const auto x1 = one / x0;
  const auto x2 = magneticFieldDotVelocity * unitVector[0];
  const auto x3 = two * normalMagneticField;
  const auto x4 = soundSpeedSquared - one;
  const auto x5 = restMassDensity * x4;
  const auto x6 = magneticFieldDotVelocity * x5;
  const auto x7 = kappa + restMassDensity;
  const auto x8 = magneticFieldSquared * x7;
  const auto x9 = restMassDensity * soundSpeedSquared;
  const auto x10 = kappa + x9;
  const auto x11 = Zvar * x10;
  const auto x12 = x11 + x8;
  const auto x13 = magneticField[0] * x6 + spatialVelocity[0] * x12;
  const auto x14 = sq(lorentzFactor);
  const auto x16 = magneticFieldDotVelocity * normalVelocity * x14;
  const auto x17 = two * x14;
  const auto x18 = normalMagneticField * (two - x17) + x16;
  const auto x19 = one / restMassDensity;
  const auto x23 = sq(magneticFieldDotVelocity);
  const auto x24 = x14 * x23;
  const auto x26 =
      x19 / (Zvar * soundSpeedSquared * (x14 - one) + soundSpeedSquared * x24 -
             x14 * (Zvar + comovingMagneticFieldSquared));
  const auto x27 = x18 * x26;
  const auto x29 = unitVector[0] * x0;
  const auto x31 = magneticFieldDotVelocity * normalMagneticField;
  const auto x33 = -normalVelocity * x0 + x31;
  const auto x35 = x14 * (spatialVelocityOneForm[0] * x33 + unitNormal[0] * x0);
  const auto x37 = normalVelocity * x0 - x31;
  const auto x38 = unitVector[1] * x0;
  const auto x39 = magneticField[1] * x6 + spatialVelocity[1] * x12;
  const auto x40 = magneticFieldDotVelocity * unitVector[1];
  const auto x42 = unitVector[2] * x0;
  const auto x43 = magneticField[2] * x6 + spatialVelocity[2] * x12;
  const auto x44 = magneticFieldDotVelocity * unitVector[2];
  const auto x46 = x0 + x24;
  const auto x48 = sq(restMassDensity);
  const auto x50 =
      magneticFieldDotVelocity * spatialVelocity[0] * x14 *
      (magneticFieldSquared * x10 + specificEnthalpy * x14 * x4 * x48 + x11);
  const auto x52 =
      x26 * (magneticField[0] *
                 (Zvar * kappa - Zvar * restMassDensity -
                  comovingMagneticFieldSquared * restMassDensity * x17 +
                  restMassDensity * x24 * (soundSpeedSquared + one) + x8) +
             x50);
  const auto x53 = x17 * x31;
  const auto x54 = -magneticFieldDotVelocity * normalVelocity * x17 +
                   normalMagneticField * (four * x14 - four);
  const auto x55 =
      Zvar * (kappa + restMassDensity *
                          (-soundSpeedSquared * (x17 - two) + x17 - one)) -
      x24 * x5 + x8;
  const auto x57 = x26 * x3;
  const auto x58 = x26 * (magneticField[0] * x55 + x50);
  const auto x59 = normalMagneticField * x17;
  const auto x62 = x1 / x14;
  const auto x63 =
      magneticFieldDotVelocity * spatialVelocity[1] * x14 *
      (magneticFieldSquared * x10 + specificEnthalpy * x14 * x4 * x48 + x11);
  const auto x64 =
      x26 * (magneticField[1] *
                 (Zvar * kappa - Zvar * restMassDensity -
                  comovingMagneticFieldSquared * restMassDensity * x17 +
                  restMassDensity * x24 * (soundSpeedSquared + one) + x8) +
             x63);
  const auto x66 = x26 * (magneticField[1] * x55 + x63);
  const auto x68 =
      magneticFieldDotVelocity * spatialVelocity[2] * x14 *
      (magneticFieldSquared * x10 + specificEnthalpy * x14 * x4 * x48 + x11);
  const auto x69 =
      x26 * (magneticField[2] *
                 (Zvar * kappa - Zvar * restMassDensity -
                  comovingMagneticFieldSquared * restMassDensity * x17 +
                  restMassDensity * x24 * (soundSpeedSquared + one) + x8) +
             x68);
  const auto x71 = x26 * (magneticField[2] * x55 + x68);
  const auto x73 = cube(lorentzFactor);
  const auto x75 = x0 * (Zvar * (-kappa + x9) + restMassDensity * x7 * x73);
  const auto x81 = -Zvar * lorentzFactor * soundSpeedSquared * (x14 - one) -
                   soundSpeedSquared * x23 * x73 +
                   x73 * (Zvar + comovingMagneticFieldSquared);
  const auto x86 = x0 * x14 * x7;
  const auto x87 = -soundSpeedSquared * x24 +
                   soundSpeedSquared * (-Zvar * x14 + Zvar) +
                   x14 * (Zvar + comovingMagneticFieldSquared);
  const auto x92 = x14 * (spatialVelocityOneForm[1] * x33 + unitNormal[1] * x0);
  const auto x97 = x14 * (spatialVelocityOneForm[2] * x33 + unitNormal[2] * x0);
  const auto x100 = x14 * x26;
  const auto x101 = x100 * x13;
  const auto x104 = x100 * x39;
  const auto x105 = normalVelocity * x100;
  const auto x107 = x100 * x43;
  const auto x116 =
      one / (x48 * (Zvar * lorentzFactor * soundSpeedSquared * (x14 - one) +
                    soundSpeedSquared * x23 * x73 -
                    x73 * (Zvar + comovingMagneticFieldSquared)));
  const auto x125 = Zvar * normalVelocity;
  const auto x126 = x125 + x31;
  const auto x129 = x1 / Zvar;
  const auto x130 = Dvar * x129;
  const auto x141 = two * x125 + x31;

  std::array<std::array<Quad, 9>, 9> m{};
  m[0][0] = x1 * (magneticFieldOneForm[0] *
                      (spatialVelocity[0] * x3 + x13 * x27 - x2) +
                  spatialVelocityOneForm[0] * x29 + x13 * x26 * x35 + x37);
  m[0][1] = x1 * (magneticFieldOneForm[0] *
                      (spatialVelocity[1] * x3 + x27 * x39 - x40) +
                  spatialVelocityOneForm[0] * x38 + x26 * x35 * x39);
  m[0][2] = x1 * (magneticFieldOneForm[0] *
                      (spatialVelocity[2] * x3 + x27 * x43 - x44) +
                  spatialVelocityOneForm[0] * x42 + x26 * x35 * x43);
  m[0][3] = x62 * (-magneticFieldOneForm[0] *
                       (magneticField[0] * x54 - spatialVelocity[0] * x53 +
                        unitVector[0] * x46 - x16 * x58 -
                        x57 * (magneticField[0] * x55 + x50) + x58 * x59) -
                   normalMagneticField * x46 + x35 * x52);
  m[0][4] = x62 * (-magneticFieldOneForm[0] *
                       (magneticField[1] * x54 - spatialVelocity[1] * x53 +
                        unitVector[1] * x46 - x16 * x66 -
                        x57 * (magneticField[1] * x55 + x63) + x59 * x66) +
                   x35 * x64);
  m[0][5] = x62 * (-magneticFieldOneForm[0] *
                       (magneticField[2] * x54 - spatialVelocity[2] * x53 +
                        unitVector[2] * x46 - x16 * x71 -
                        x57 * (magneticField[2] * x55 + x68) + x59 * x71) +
                   x35 * x69);
  m[0][6] = x62 *
            (magneticFieldOneForm[0] * x18 * x75 +
             x14 * (spatialVelocityOneForm[0] * x33 * x75 -
                    unitNormal[0] * x0 * (x48 * x81 - x75))) /
            (x48 * x81);
  m[0][7] = x1 * x19 *
            (magneticFieldOneForm[0] * x0 * x18 * x7 +
             spatialVelocityOneForm[0] * x33 * x86 -
             unitNormal[0] * x0 * (restMassDensity * x87 - x86)) /
            x87;
  m[0][8] = Quad{0.0};
  m[1][0] = x1 * (magneticFieldOneForm[1] *
                      (spatialVelocity[0] * x3 + x13 * x27 - x2) +
                  spatialVelocityOneForm[1] * x29 + x13 * x26 * x92);
  m[1][1] = x1 * (magneticFieldOneForm[1] *
                      (spatialVelocity[1] * x3 + x27 * x39 - x40) +
                  spatialVelocityOneForm[1] * x38 + x26 * x39 * x92 + x37);
  m[1][2] = x1 * (magneticFieldOneForm[1] *
                      (spatialVelocity[2] * x3 + x27 * x43 - x44) +
                  spatialVelocityOneForm[1] * x42 + x26 * x43 * x92);
  m[1][3] = x62 * (-magneticFieldOneForm[1] *
                       (magneticField[0] * x54 - spatialVelocity[0] * x53 +
                        unitVector[0] * x46 - x16 * x58 -
                        x57 * (magneticField[0] * x55 + x50) + x58 * x59) +
                   x52 * x92);
  m[1][4] = x62 * (-magneticFieldOneForm[1] *
                       (magneticField[1] * x54 - spatialVelocity[1] * x53 +
                        unitVector[1] * x46 - x16 * x66 -
                        x57 * (magneticField[1] * x55 + x63) + x59 * x66) -
                   normalMagneticField * x46 + x64 * x92);
  m[1][5] = x62 * (-magneticFieldOneForm[1] *
                       (magneticField[2] * x54 - spatialVelocity[2] * x53 +
                        unitVector[2] * x46 - x16 * x71 -
                        x57 * (magneticField[2] * x55 + x68) + x59 * x71) +
                   x69 * x92);
  m[1][6] = x62 *
            (magneticFieldOneForm[1] * x18 * x75 +
             x14 * (spatialVelocityOneForm[1] * x33 * x75 -
                    unitNormal[1] * x0 * (x48 * x81 - x75))) /
            (x48 * x81);
  m[1][7] = x1 * x19 *
            (magneticFieldOneForm[1] * x0 * x18 * x7 +
             spatialVelocityOneForm[1] * x33 * x86 -
             unitNormal[1] * x0 * (restMassDensity * x87 - x86)) /
            x87;
  m[1][8] = Quad{0.0};
  m[2][0] = x1 * (magneticFieldOneForm[2] *
                      (spatialVelocity[0] * x3 + x13 * x27 - x2) +
                  spatialVelocityOneForm[2] * x29 + x13 * x26 * x97);
  m[2][1] = x1 * (magneticFieldOneForm[2] *
                      (spatialVelocity[1] * x3 + x27 * x39 - x40) +
                  spatialVelocityOneForm[2] * x38 + x26 * x39 * x97);
  m[2][2] = x1 * (magneticFieldOneForm[2] *
                      (spatialVelocity[2] * x3 + x27 * x43 - x44) +
                  spatialVelocityOneForm[2] * x42 + x26 * x43 * x97 + x37);
  m[2][3] = x62 * (-magneticFieldOneForm[2] *
                       (magneticField[0] * x54 - spatialVelocity[0] * x53 +
                        unitVector[0] * x46 - x16 * x58 -
                        x57 * (magneticField[0] * x55 + x50) + x58 * x59) +
                   x52 * x97);
  m[2][4] = x62 * (-magneticFieldOneForm[2] *
                       (magneticField[1] * x54 - spatialVelocity[1] * x53 +
                        unitVector[1] * x46 - x16 * x66 -
                        x57 * (magneticField[1] * x55 + x63) + x59 * x66) +
                   x64 * x97);
  m[2][5] = x62 * (-magneticFieldOneForm[2] *
                       (magneticField[2] * x54 - spatialVelocity[2] * x53 +
                        unitVector[2] * x46 - x16 * x71 -
                        x57 * (magneticField[2] * x55 + x68) + x59 * x71) -
                   normalMagneticField * x46 + x69 * x97);
  m[2][6] = x62 *
            (magneticFieldOneForm[2] * x18 * x75 +
             x14 * (spatialVelocityOneForm[2] * x33 * x75 -
                    unitNormal[2] * x0 * (x48 * x81 - x75))) /
            (x48 * x81);
  m[2][7] = x1 * x19 *
            (magneticFieldOneForm[2] * x0 * x18 * x7 +
             spatialVelocityOneForm[2] * x33 * x86 -
             unitNormal[2] * x0 * (restMassDensity * x87 - x86)) /
            x87;
  m[2][8] = Quad{0.0};
  m[3][0] =
      x1 * (magneticFieldOneForm[0] * (-normalVelocity * x101 + unitVector[0]) +
            normalMagneticField * (spatialVelocityOneForm[0] * x101 - one));
  m[3][1] = x1 * (magneticFieldOneForm[0] * (unitVector[1] - x105 * x39) +
                  normalMagneticField * spatialVelocityOneForm[0] * x104);
  m[3][2] = x1 * (magneticFieldOneForm[0] * (unitVector[2] - x105 * x43) +
                  normalMagneticField * spatialVelocityOneForm[0] * x107);
  m[3][3] =
      x1 *
      (magneticFieldOneForm[0] * (-normalVelocity * x52 + x2) +
       spatialVelocityOneForm[0] * (normalMagneticField * x52 - x29) + x37);
  m[3][4] =
      x1 * (magneticFieldOneForm[0] * (-normalVelocity * x64 + x40) +
            spatialVelocityOneForm[0] * (normalMagneticField * x64 - x38));
  m[3][5] =
      x1 * (magneticFieldOneForm[0] * (-normalVelocity * x69 + x44) +
            spatialVelocityOneForm[0] * (normalMagneticField * x69 - x42));
  m[3][6] = x116 * (Zvar * (-kappa + x9) + restMassDensity * x7 * x73) *
            (magneticFieldOneForm[0] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[0]);
  m[3][7] = x100 * x7 *
            (magneticFieldOneForm[0] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[0]);
  m[3][8] = unitNormal[0];
  m[4][0] =
      x1 * (magneticFieldOneForm[1] * (-normalVelocity * x101 + unitVector[0]) +
            normalMagneticField * spatialVelocityOneForm[1] * x101);
  m[4][1] =
      x1 * (magneticFieldOneForm[1] * (unitVector[1] - x105 * x39) +
            normalMagneticField * (spatialVelocityOneForm[1] * x104 - one));
  m[4][2] = x1 * (magneticFieldOneForm[1] * (unitVector[2] - x105 * x43) +
                  normalMagneticField * spatialVelocityOneForm[1] * x107);
  m[4][3] =
      x1 * (magneticFieldOneForm[1] * (-normalVelocity * x52 + x2) +
            spatialVelocityOneForm[1] * (normalMagneticField * x52 - x29));
  m[4][4] =
      x1 *
      (magneticFieldOneForm[1] * (-normalVelocity * x64 + x40) +
       spatialVelocityOneForm[1] * (normalMagneticField * x64 - x38) + x37);
  m[4][5] =
      x1 * (magneticFieldOneForm[1] * (-normalVelocity * x69 + x44) +
            spatialVelocityOneForm[1] * (normalMagneticField * x69 - x42));
  m[4][6] = x116 * (Zvar * (-kappa + x9) + restMassDensity * x7 * x73) *
            (magneticFieldOneForm[1] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[1]);
  m[4][7] = x100 * x7 *
            (magneticFieldOneForm[1] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[1]);
  m[4][8] = unitNormal[1];
  m[5][0] =
      x1 * (magneticFieldOneForm[2] * (-normalVelocity * x101 + unitVector[0]) +
            normalMagneticField * spatialVelocityOneForm[2] * x101);
  m[5][1] = x1 * (magneticFieldOneForm[2] * (unitVector[1] - x105 * x39) +
                  normalMagneticField * spatialVelocityOneForm[2] * x104);
  m[5][2] =
      x1 * (magneticFieldOneForm[2] * (unitVector[2] - x105 * x43) +
            normalMagneticField * (spatialVelocityOneForm[2] * x107 - one));
  m[5][3] =
      x1 * (magneticFieldOneForm[2] * (-normalVelocity * x52 + x2) +
            spatialVelocityOneForm[2] * (normalMagneticField * x52 - x29));
  m[5][4] =
      x1 * (magneticFieldOneForm[2] * (-normalVelocity * x64 + x40) +
            spatialVelocityOneForm[2] * (normalMagneticField * x64 - x38));
  m[5][5] =
      x1 *
      (magneticFieldOneForm[2] * (-normalVelocity * x69 + x44) +
       spatialVelocityOneForm[2] * (normalMagneticField * x69 - x42) + x37);
  m[5][6] = x116 * (Zvar * (-kappa + x9) + restMassDensity * x7 * x73) *
            (magneticFieldOneForm[2] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[2]);
  m[5][7] = x100 * x7 *
            (magneticFieldOneForm[2] * normalVelocity -
             normalMagneticField * spatialVelocityOneForm[2]);
  m[5][8] = unitNormal[2];
  m[6][0] = x130 * (Zvar * unitVector[0] +
                    magneticField[0] * normalMagneticField - x101 * x126);
  m[6][1] = x130 * (Zvar * unitVector[1] +
                    magneticField[1] * normalMagneticField - x104 * x126);
  m[6][2] = x130 * (Zvar * unitVector[2] +
                    magneticField[2] * normalMagneticField - x107 * x126);
  m[6][3] =
      x130 * (Zvar * normalMagneticField * spatialVelocity[0] + Zvar * x2 +
              magneticFieldSquared * normalMagneticField * spatialVelocity[0] -
              magneticField[0] * x141 - x125 * x58 - x31 * x58);
  m[6][4] =
      x130 * (Zvar * normalMagneticField * spatialVelocity[1] + Zvar * x40 +
              magneticFieldSquared * normalMagneticField * spatialVelocity[1] -
              magneticField[1] * x141 - x125 * x66 - x31 * x66);
  m[6][5] =
      x130 * (Zvar * normalMagneticField * spatialVelocity[2] + Zvar * x44 +
              magneticFieldSquared * normalMagneticField * spatialVelocity[2] -
              magneticField[2] * x141 - x125 * x71 - x31 * x71);
  m[6][6] = x129 * (Dvar * x116 * x31 * x75 + x125 * (Dvar * x116 * x75 + x0));
  m[6][7] = Dvar * x126 * x19 * x7 /
            (Zvar * (Zvar * (-soundSpeedSquared / x14 + x4) -
                     comovingMagneticFieldSquared + soundSpeedSquared * x23));
  m[6][8] = Quad{0.0};
  m[7][0] = x129 * (-Dvar * magneticField[0] * normalMagneticField +
                    Dvar * x101 * x126 + Zvar * unitVector[0] * (-Dvar + x0));
  m[7][1] = x129 * (-Dvar * magneticField[1] * normalMagneticField +
                    Dvar * x104 * x126 + Zvar * unitVector[1] * (-Dvar + x0));
  m[7][2] = x129 * (-Dvar * magneticField[2] * normalMagneticField +
                    Dvar * x107 * x126 + Zvar * unitVector[2] * (-Dvar + x0));
  m[7][3] =
      x130 * (-Zvar * normalMagneticField * spatialVelocity[0] - Zvar * x2 -
              magneticFieldSquared * normalMagneticField * spatialVelocity[0] +
              magneticField[0] * x141 + x125 * x58 + x31 * x58);
  m[7][4] =
      x130 * (-Zvar * normalMagneticField * spatialVelocity[1] - Zvar * x40 -
              magneticFieldSquared * normalMagneticField * spatialVelocity[1] +
              magneticField[1] * x141 + x125 * x66 + x31 * x66);
  m[7][5] =
      x130 * (-Zvar * normalMagneticField * spatialVelocity[2] - Zvar * x44 -
              magneticFieldSquared * normalMagneticField * spatialVelocity[2] +
              magneticField[2] * x141 + x125 * x71 + x31 * x71);
  m[7][6] = -x129 * (Dvar * x116 * x31 * x75 + x125 * (Dvar * x116 * x75 + x0));
  m[7][7] = -Dvar * x126 * x19 * x7 /
            (Zvar * (Zvar * (-soundSpeedSquared / x14 + x4) -
                     comovingMagneticFieldSquared + soundSpeedSquared * x23));
  m[7][8] = Quad{0.0};
  m[8][0] = Quad{0.0};
  m[8][1] = Quad{0.0};
  m[8][2] = Quad{0.0};
  m[8][3] = unitVector[0];
  m[8][4] = unitVector[1];
  m[8][5] = unitVector[2];
  m[8][6] = Quad{0.0};
  m[8][7] = Quad{0.0};
  m[8][8] = Quad{0.0};
  return m;
}

}  // namespace quad_precision

/// Asymptotic (large-W) expansion for the four magnetosonic Eulerian-frame
/// speeds.  The expansion parameter is epsilon = 1/W.
///
/// Slow pair: y = c10 + c12 * eps^2 + c14 * eps^4 + O(eps^6)
/// Fast pair: y = c30 +/- c31 * eps + c32 * eps^2 +/- c33 * eps^3
///                + c34 * eps^4 + O(eps^5)
///
/// Reference: quartic_current.nb (Summary section), asymptotic expansion of
/// Eq. (2.30) from Characteristic Decomposition II.
namespace asymptotic_magnetosonic_speeds {

/// Compute the four magnetosonic speeds from the high-W asymptotic expansion.
///
/// The function works with scalars (one grid point at a time) to keep the
/// implementation simple and readable.  Call it inside a loop over points.
///
/// \param sound_speed_squared  c_s^2  (must satisfy 0 < c_s^2 < 1)
/// \param normal_velocity  s_a v^a
/// \param lorentz_factor  W
/// \param normal_magnetic_field_normalized  Bbar_s = (B^a s_a) / sqrt(rho h)
/// \param magnetic_field_dot_velocity_normalized  Bbar_v = (B^a v_a) / sqrt(rho
/// h) \param magnetic_field_squared_normalized  Bbar^2 = (B^a B_a) / (rho h)
///
/// \return array of 4 speeds indexed by MhdSpeed:
///   [FastMagnetosonicMinus, FastMagnetosonicPlus,
///    SlowMagnetosonicMinus, SlowMagnetosonicPlus]
inline std::array<double, 4> speeds(
    const double sound_speed_squared, const double normal_velocity,
    const double lorentz_factor, const double normal_magnetic_field_normalized,
    const double magnetic_field_dot_velocity_normalized,
    const double magnetic_field_squared_normalized) {
  // --- Rename inputs to match the notation in the derivation ---
  const double cs2 = sound_speed_squared;
  const double cs = std::sqrt(cs2);
  const double sv = normal_velocity;
  const double W = lorentz_factor;

  // Renormalized magnetic field projections (already divided by sqrt(rho h))
  const double Bs = normal_magnetic_field_normalized;
  const double Bv = magnetic_field_dot_velocity_normalized;
  const double Bsq = magnetic_field_squared_normalized;

  // --- Precomputed intermediate quantities ---
  const double one_minus_cs2 = 1.0 - cs2;      // 1 - c_s^2
  const double one_minus_sv2 = 1.0 - sv * sv;  // 1 - (s.v)^2

  const double Bv2 = Bv * Bv;
  const double Bs2 = Bs * Bs;
  const double Bs3 = Bs2 * Bs;

  // D = c_s^2 + (1 - c_s^2)(Bbar.v)^2
  const double D = cs2 + one_minus_cs2 * Bv2;
  const double D2 = D * D;
  const double D4 = D2 * D2;

  // A = sqrt(c_s^2 + (Bbar.v)^2)
  const double A2 = cs2 + Bv2;
  const double A = std::sqrt(A2);

  // T = sqrt(1 - (s.v)^2)
  const double T = std::sqrt(one_minus_sv2);

  // Expansion parameter
  const double eps = 1.0 / W;
  const double eps2 = eps * eps;
  const double eps3 = eps2 * eps;
  const double eps4 = eps2 * eps2;

  // cs2 - 1 (negative, used frequently in the fast coefficients)
  const double cs2_minus_1 = cs2 - 1.0;
  const double cs2_minus_1_sq = cs2_minus_1 * cs2_minus_1;

  // ============== SLOW PAIR ==============
  // Two roots obtained by the two sign choices in the denominator:
  //   denom1 = cs * Bv -/+ A
  // (branchSign = +1 gives denom1 = cs*Bv - A, branchSign = -1 gives cs*Bv + A)
  std::array<double, 4> result{};
  for (const double branch_sign : {+1.0, -1.0}) {
    const double denom1 = cs * Bv - branch_sign * A;
    const double denom2 = denom1 * denom1;
    const double denom4 = denom2 * denom2;

    const double c10 = sv;
    const double c12 = cs * Bs / denom1;
    const double c14 = (cs * Bsq * Bs) / (2.0 * branch_sign * A * denom2) -
                       (cs * cs2 * one_minus_cs2 * Bs3) /
                           (2.0 * branch_sign * A * denom4 * one_minus_sv2);

    const double y_slow = c10 + c12 * eps2 + c14 * eps4;

    // Map branch_sign to the correct MhdSpeed index.  The branch with
    // branch_sign = +1 produces the denominator cs*Bv - A (more negative for
    // positive A), giving the "minus" slow speed, and vice versa.  We just
    // store them and sort afterward.
    if (branch_sign > 0.0) {
      result[2] = y_slow;  // tentatively SlowMagnetosonicMinus
    } else {
      result[3] = y_slow;  // tentatively SlowMagnetosonicPlus
    }
  }
  // Ensure correct ordering of the slow pair
  if (result[2] > result[3]) {
    std::swap(result[2], result[3]);
  }

  // ============== FAST PAIR ==============
  // Even coefficients are the same for both fast roots.
  // Odd coefficients flip sign between the two fast roots.
  const double c30 = sv;
  const double c31 = -std::sqrt(D * one_minus_sv2 / one_minus_cs2);

  const double c32 = cs2 * Bs * Bv / D - D * sv / one_minus_cs2;

  // D^(5/2)
  const double D_sqrt = std::sqrt(D);
  const double D52 = D2 * D_sqrt;

  // N33 numerator
  const double N33 =
      cs2 * cs2_minus_1_sq * Bs2 * (cs2 + (1.0 + 2.0 * cs2) * Bv2) -
      2.0 * cs2 * cs2_minus_1 * Bs * Bv * D2 * sv -
      cs2_minus_1 * Bsq * D2 * (sv * sv - 1.0) - D4 * (2.0 * sv * sv - 1.0);

  const double c33 = N33 / (2.0 * std::pow(one_minus_cs2, 1.5) * T * D52);

  const double c34 =
      cs2 * Bs * Bv * (1.0 / cs2_minus_1 - Bsq / D2) +
      (cs2_minus_1 * Bsq + D2) * sv / cs2_minus_1_sq +
      (2.0 * cs2 * cs2 * cs2_minus_1 * Bs3 * Bv * (cs2 + (1.0 + cs2) * Bv2)) /
          (D4 * (sv * sv - 1.0));

  // The two fast roots differ by the sign of the odd-power terms
  for (const double odd_sign : {+1.0, -1.0}) {
    const double y_fast = c30 + odd_sign * c31 * eps + c32 * eps2 +
                          odd_sign * c33 * eps3 + c34 * eps4;
    if (odd_sign > 0.0) {
      result[0] = y_fast;  // tentatively FastMagnetosonicMinus
    } else {
      result[1] = y_fast;
    }
  }
  // Ensure correct ordering of the fast pair
  if (result[0] > result[1]) {
    std::swap(result[0], result[1]);
  }

  return result;
}

}  // namespace asymptotic_magnetosonic_speeds

}  // namespace grmhd::ValenciaDivClean::TestHelpers
