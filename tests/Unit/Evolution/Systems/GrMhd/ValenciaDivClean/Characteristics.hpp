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
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_cov[i] += g[i][j] * spatial_velocity[j];
      B_cov[i] += g[i][j] * magnetic_field[j];
      s_vec[i] += inv_g[i][j] * unit_normal[j];
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
      // Explicit S_b components (Rows 1, 2, 3 of the paper's array)
      right[wave][0] =
          -Quad{2.0} * sqrt_rho_h_star * B_21 * (B_n + r_1 * v_n * sq(W));
      right[wave][1] =
          -sqrt_rho_h_star * (B_n * B_32 + B_1 * B_21 +
                              r_1 * sq(W) * (B_2 + v_1 * B_21 + v_n * B_32));
      right[wave][2] =
          sqrt_rho_h_star * (B_n * B_31 - B_2 * B_21 +
                             r_1 * sq(W) * (B_1 - v_2 * B_21 + v_n * B_31));

      // Explicit B_b components (Rows 4, 5, 6 of the paper's array)
      right[wave][3] = Quad{0.0};
      right[wave][4] = sqrt_rho_h_star * B_2 + v_2 * r_4;
      right[wave][5] = -sqrt_rho_h_star * B_1 - v_1 * r_4;

      // Explicit Scalar components D and tau (Rows 7, 8 of the paper's array)
      right[wave][6] = -rho * W * B_21;
      right[wave][7] =
          -B_21 * W * (Quad{2.0} * W * sqrt_rho_h_star * r_1 - rho);

      // Appended 9th component for the Divergence Cleaning scalar phi
      right[wave][8] = Quad{0.0};

      const Quad y_Alf = y;
      const Quad alf_norm = Quad{1.0} / sqrt_rho_h_star;

      // Explicit S_b components (Rows 1, 2, 3 of the paper's array)
      left[wave][0] = alf_norm * (B_21 * y_Alf);
      left[wave][1] = alf_norm * (B_2 + B_32 * y_Alf);
      left[wave][2] = alf_norm * (-B_1 - B_31 * y_Alf);

      // Explicit B_b components (Rows 4, 5, 6 of the paper's array)
      left[wave][3] = -B_21 * y_Alf * sqrt_rho_h_star;
      left[wave][4] = -(B_2 + B_32 * y_Alf);
      left[wave][5] = (B_1 + B_31 * y_Alf);

      // Explicit Scalar components D and tau (Rows 7, 8 of the paper's array)
      left[wave][6] = alf_norm * (-B_21);
      left[wave][7] = alf_norm * (-B_21);

      // Appended 9th component for the Divergence Cleaning scalar phi
      left[wave][8] = -B_21;
    } else if (wave == grmhd::ValenciaDivClean::MhdSpeed::ScalarMinus or
               wave == grmhd::ValenciaDivClean::MhdSpeed::ScalarPlus) {
      for (size_t i = 0; i < 3; ++i) {
        right[wave][i] =
            -((y * kappa_B + Quad{2.0} * kappa_rho * a * B * B_n) *
                  magnetic_field[i] +
              sq(W) * B * kappa_B * (s_vec[i] + y * spatial_velocity[i]) -
              Quad{2.0} * W * kappa_rho * B * sq(B_n) * spatial_velocity[i]) /
            (a_denom * W);
        right[wave][3 + i] = kappa_rho * y * B * magnetic_field[i] / W +
                             (s_vec[i] - y * spatial_velocity[i]) *
                                 (kappa_rho * B * B_n +
                                  (Quad{1.0} - cs2) * sq(rho) * sq(a) * h * W) /
                                 a_denom;
      }
      right[wave][6] = kappa_rho * y * rho * B -
                       (Quad{1.0} - cs2) * sq(rho * a) * h * W / a_denom;
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
