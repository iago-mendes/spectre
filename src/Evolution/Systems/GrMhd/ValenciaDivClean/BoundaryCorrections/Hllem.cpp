// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ostream>
#include <pup.h>

#include <memory>
#include <optional>
#include <vector>

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

std::ostream& operator<<(std::ostream& os, const HllemWaves waves) {
  switch (waves) {
    case HllemWaves::Contact:
      return os << "Contact";
    case HllemWaves::ContactAlfven:
      return os << "ContactAlfven";
    case HllemWaves::ContactSlow:
      return os << "ContactSlow";
    case HllemWaves::All:
      return os << "All";
    case HllemWaves::ContactAlfvenFast:
      return os << "ContactAlfvenFast";
    case HllemWaves::AllWithFast:
      return os << "AllWithFast";
    case HllemWaves::None:
      return os << "None";
    case HllemWaves::Slow:
      return os << "Slow";
    case HllemWaves::Alfven:
      return os << "Alfven";
    default:
      ERROR("Unknown HllemWaves");
  }
}

namespace {
// Indices (MhdSpeed enum order) of the internal waves the anti-diffusion
// restores. Contact=4 (Entropy), Alfven={2,6}, Slow={3,5}; the outer fast
// waves (1,7) and GLM scalars (0,8) are the HLL / divergence-cleaning waves.
std::vector<size_t> restored_wave_indices(const HllemWaves waves) {
  switch (waves) {
    case HllemWaves::Contact:
      return {4};
    case HllemWaves::ContactAlfven:
      return {2, 4, 6};
    case HllemWaves::ContactSlow:
      return {3, 4, 5};
    case HllemWaves::All:
      return {2, 3, 4, 5, 6};
    case HllemWaves::ContactAlfvenFast:
      // fast-, Alfven-, contact, Alfven+, fast+ (no slow)
      return {1, 2, 4, 6, 7};
    case HllemWaves::AllWithFast:
      // every interior wave; only the GLM scalars stay as the outer HLL bounds
      return {1, 2, 3, 4, 5, 6, 7};
    case HllemWaves::None:
      // No anti-diffusion at all: HLLEM with every delta_k = 0, which IS the
      // HLL flux. The one consumer of this list is the per-wave loop in
      // `dg_boundary_terms`, so an empty list makes that whole block a no-op.
      return {};
    case HllemWaves::Slow:
      // ONLY the slow pair. `ContactSlow` is contact + slow, so it cannot
      // separate the two; this is the minimal set that reproduces the Del
      // Zanna jet failure.
      return {3, 5};
    case HllemWaves::Alfven:
      // ONLY the Alfven pair. On a purely axial field B_n = 0 on the radial
      // faces, the Alfven speeds coincide with the entropy speed, and the
      // speed-gap guard below drops both waves -- which is what this value
      // exists to measure rather than assume.
      return {2, 6};
    default:
      ERROR("Unknown HllemWaves");
  }
}

// --- Fan admissibility guard (see where it is applied in dg_boundary_terms) --
// SPECTRE_HLLEM_FAN_GUARD=0 turns the guard off, to reproduce runs made before
// 2026-11-09 or to compare both from one binary and one checkpoint; unset or
// any other value leaves it on. Read on every call, so a test can toggle it.
bool fan_guard_enabled() {
  const char* const value = std::getenv("SPECTRE_HLLEM_FAN_GUARD");
  return value == nullptr or std::strtol(value, nullptr, 10) != 0;
}

// The face geometry at one point: the spatial metric gamma_ij, its inverse
// and sqrt(gamma), in the order (xx, xy, xz, yy, yz, zz).
struct FanGeometry {
  std::array<double, 6> metric{};
  std::array<double, 6> inverse_metric{};
  double sqrt_det = 1.0;
};

// gamma_ij x^i x^j (or gamma^ij x_i x_j) for a symmetric 3x3 matrix stored as
// (xx, xy, xz, yy, yz, zz).
double contract(const std::array<double, 6>& m, const double x, const double y,
                const double z) {
  return m[0] * x * x + m[3] * y * y + m[5] * z * z +
         2.0 * (m[1] * x * y + m[2] * x * z + m[4] * y * z);
}

// Three necessary conditions on an RMHD state in the eigenvector ordering
// [S_x, S_y, S_z, B_x, B_y, B_z, D, Tau] of the densitized conserved variables
// (each divided by sqrt(gamma) first, so that these are the Eulerian
// observer's densities, and contracted with the spatial metric):
//   1. D > 0;
//   2. Tau >= B^2/2: the energy holds at least the field's own energy;
//   3. E = Tau + D >= sqrt(D^2 + S^2).
// Every physical state with an ideal-gas equation of state (Gamma <= 2) meets
// all three (checked on 4e5 random states with W up to 300, B up to 1e3 and
// p down to 0). They are not sufficient: the S.B constraint is not checked.
// The slack absorbs round-off, which puts cold static magnetized states
// (Tau = B^2/2 exactly) and W >> 1 states (E - |S| ~ E/(2 W^2)) at the edge.
// Returns 0 if the state passes, else the number of the first failed test.
int fan_state_failure(const std::array<double, 8>& state,
                      const FanGeometry& geometry) {
  constexpr double slack = 1.0e-12;
  const double inv_sqrt_det = 1.0 / geometry.sqrt_det;
  const double d = state[6] * inv_sqrt_det;
  const double tau = state[7] * inv_sqrt_det;
  const double s_squared =
      contract(geometry.inverse_metric, state[0], state[1], state[2]) *
      square(inv_sqrt_det);
  const double b_squared =
      contract(geometry.metric, state[3], state[4], state[5]) *
      square(inv_sqrt_det);
  const double energy = tau + d;
  if (not(d > 0.0)) {
    return 1;
  }
  if (not(tau - 0.5 * b_squared >=
          -slack * (std::abs(tau) + 0.5 * b_squared))) {
    return 2;
  }
  if (not(energy > 0.0 and
          square(energy) - square(d) - s_squared >= -slack * square(energy))) {
    return 3;
  }
  return 0;
}

// The two half-fan averages with the anti-diffusion scaled by theta:
// u_hll - theta weight_int a on the interior side and u_hll + theta weight_ext
// a on the exterior side (weight_int = lambda_max / (lambda_max - lambda_min),
// weight_ext = -lambda_min / (lambda_max - lambda_min); see dg_boundary_terms).
std::array<std::array<double, 8>, 2> fan_half_averages(
    const std::array<double, 8>& u_hll, const std::array<double, 8>& a,
    const double weight_int, const double weight_ext, const double theta) {
  std::array<std::array<double, 8>, 2> halves{};
  for (size_t n = 0; n < 8; ++n) {
    gsl::at(halves[0], n) =
        gsl::at(u_hll, n) - theta * weight_int * gsl::at(a, n);
    gsl::at(halves[1], n) =
        gsl::at(u_hll, n) + theta * weight_ext * gsl::at(a, n);
  }
  return halves;
}

// The largest theta in [0, 1] for which both half-fan averages pass
// fan_state_failure: 1 if they pass as they are, 0 if even u_hll fails (the
// face then keeps the HLL flux), otherwise found by bisection. Each condition
// holds on a convex set (D and E - sqrt(D^2 + S^2) are concave in the state,
// Tau - B^2/2 too) and the averages are affine in theta, so the passing thetas
// form one interval [0, theta_max]. Sixty halvings resolve theta to round-off,
// so the two sides of a face, which solve the same problem mirrored, agree on
// it to round-off and the flux stays conservative.
double fan_guard_theta(const std::array<double, 8>& u_hll,
                       const std::array<double, 8>& a, const double weight_int,
                       const double weight_ext, const FanGeometry& geometry) {
  const auto halves_pass = [&u_hll, &a, weight_int, weight_ext,
                            &geometry](const double theta) {
    const auto halves =
        fan_half_averages(u_hll, a, weight_int, weight_ext, theta);
    return fan_state_failure(halves[0], geometry) == 0 and
           fan_state_failure(halves[1], geometry) == 0;
  };
  if (halves_pass(1.0)) {
    return 1.0;
  }
  if (not halves_pass(0.0)) {
    return 0.0;
  }
  double lower = 0.0;
  double upper = 1.0;
  for (size_t iteration = 0; iteration < 60; ++iteration) {
    const double middle = 0.5 * (lower + upper);
    if (halves_pass(middle)) {
      lower = middle;
    } else {
      upper = middle;
    }
  }
  return lower;
}

// Engagement counters. SPECTRE_HLLEM_STATS_EVERY=N prints this thread's
// cumulative counts every N calls of dg_boundary_terms;
// SPECTRE_HLLEM_FAN_GUARD_DUMP=N prints the first N limited faces of each
// thread in full (both sides' primitives, the edge states, theta).
struct FanGuardStats {
  size_t calls = 0;
  size_t faces = 0;
  size_t limited = 0;
  size_t to_hll = 0;
  size_t dumped = 0;
  std::array<size_t, 4> first_failure{};
  double theta_sum = 0.0;
};
thread_local FanGuardStats fan_guard_stats{};

size_t env_count(const char* const name) {
  const char* const value = std::getenv(name);
  return value == nullptr ? 0
                          : static_cast<size_t>(
                                std::max(0L, std::strtol(value, nullptr, 10)));
}
}  // namespace

Hllem::Hllem(const HllemWaves waves_to_restore,
             const bool use_complementary_projection,
             const double degeneracy_tolerance,
             const double magnetic_field_magnitude_for_hydro,
             const double light_speed_density_cutoff,
             const std::optional<double> max_condition_number,
             const Options::Context& context)
    : waves_to_restore_(waves_to_restore),
      use_complementary_projection_(use_complementary_projection),
      degeneracy_tolerance_(degeneracy_tolerance),
      magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff),
      max_condition_number_(max_condition_number) {
  if (max_condition_number_.has_value() and
      not(*max_condition_number_ >= 1.0)) {
    PARSE_ERROR(context,
                "MaxConditionNumber must be at least 1 (or None), but is "
                    << *max_condition_number_);
  }
}

Hllem::Hllem(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hllem::get_clone() const {
  return std::make_unique<Hllem>(*this);
}

void Hllem::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | waves_to_restore_;
  p | use_complementary_projection_;
  p | degeneracy_tolerance_;
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
  p | max_condition_number_;
}

double Hllem::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal_vector,
    const gsl::not_null<Scalar<DataVector>*> packaged_lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_shift,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        packaged_spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> packaged_pressure,
    const gsl::not_null<Scalar<DataVector>*> packaged_lorentz_factor,
    const gsl::not_null<Scalar<DataVector>*> packaged_specific_internal_energy,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& lorentz_factor,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  // Everything `Hll` packages -- the conserved variables, their normal fluxes,
  // the light-speed and fast-magnetosonic signal speeds of each side, and the
  // interface normal and its raised form -- is packaged exactly as `Hll` does
  // it. On a curved face `dg_boundary_terms` then returns `Hll`'s flux, and on
  // a flat one the HLLEM anti-diffusion is built on the same speeds. The fast
  // speeds are each side's own (lapse, shift and spatial metric included).
  const Hll hll{magnetic_field_magnitude_for_hydro_,
                light_speed_density_cutoff_};
  const double max_speed = hll.dg_package_data(
      packaged_tilde_d, packaged_tilde_ye, packaged_tilde_tau, packaged_tilde_s,
      packaged_tilde_b, packaged_tilde_phi, packaged_normal_dot_flux_tilde_d,
      packaged_normal_dot_flux_tilde_ye, packaged_normal_dot_flux_tilde_tau,
      packaged_normal_dot_flux_tilde_s, packaged_normal_dot_flux_tilde_b,
      packaged_normal_dot_flux_tilde_phi, packaged_largest_outgoing_char_speed,
      packaged_largest_ingoing_char_speed, packaged_fast_outgoing_char_speed,
      packaged_fast_ingoing_char_speed, packaged_interface_unit_normal,
      packaged_interface_unit_normal_vector, tilde_d, tilde_ye, tilde_tau,
      tilde_s, tilde_b, tilde_phi, flux_tilde_d, flux_tilde_ye, flux_tilde_tau,
      flux_tilde_s, flux_tilde_b, flux_tilde_phi, lapse, shift,
      spatial_velocity_one_form, spatial_metric, rest_mass_density,
      electron_fraction, temperature, spatial_velocity, normal_covector,
      normal_vector, mesh_velocity, normal_dot_mesh_velocity,
      equation_of_state);

  // The face geometry, for the averaged state's eigensystem and its speeds.
  *packaged_lapse = lapse;
  *packaged_shift = shift;
  *packaged_spatial_metric = spatial_metric;

  *packaged_rest_mass_density = rest_mass_density;
  *packaged_spatial_velocity = spatial_velocity;
  *packaged_pressure = pressure;
  *packaged_lorentz_factor = lorentz_factor;
  *packaged_specific_internal_energy = specific_internal_energy;

  return max_speed;
}

void Hllem::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& fast_outgoing_char_speed_int,
    const Scalar<DataVector>& fast_ingoing_char_speed_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        interface_unit_normal_vector_int,
    const Scalar<DataVector>& lapse_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift_int,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_int,
    const Scalar<DataVector>& rest_mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& pressure_int,
    const Scalar<DataVector>& /*lorentz_factor_int*/,
    const Scalar<DataVector>& specific_internal_energy_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const Scalar<DataVector>& fast_outgoing_char_speed_ext,
    const Scalar<DataVector>& fast_ingoing_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        interface_unit_normal_vector_ext,
    const Scalar<DataVector>& lapse_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift_ext,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric_ext,
    const Scalar<DataVector>& rest_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& /*lorentz_factor_ext*/,
    const Scalar<DataVector>& specific_internal_energy_ext,
    const dg::Formulation dg_formulation,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  // The HLL part is `Hll`'s flux, on every face: its fast-magnetosonic bounds
  // (each side's own speed, with its lapse, shift and spatial metric) for the
  // MHD variables and the light-speed bounds for the divergence-cleaning
  // subsystem. HLLEM therefore has HLL's intermediate state, the integral
  // average of the fan, and with no restored waves it IS `Hll`.
  Hll::dg_boundary_terms(
      boundary_correction_tilde_d, boundary_correction_tilde_ye,
      boundary_correction_tilde_tau, boundary_correction_tilde_s,
      boundary_correction_tilde_b, boundary_correction_tilde_phi, tilde_d_int,
      tilde_ye_int, tilde_tau_int, tilde_s_int, tilde_b_int, tilde_phi_int,
      normal_dot_flux_tilde_d_int, normal_dot_flux_tilde_ye_int,
      normal_dot_flux_tilde_tau_int, normal_dot_flux_tilde_s_int,
      normal_dot_flux_tilde_b_int, normal_dot_flux_tilde_phi_int,
      largest_outgoing_char_speed_int, largest_ingoing_char_speed_int,
      fast_outgoing_char_speed_int, fast_ingoing_char_speed_int,
      interface_unit_normal_int, interface_unit_normal_vector_int, tilde_d_ext,
      tilde_ye_ext, tilde_tau_ext, tilde_s_ext, tilde_b_ext, tilde_phi_ext,
      normal_dot_flux_tilde_d_ext, normal_dot_flux_tilde_ye_ext,
      normal_dot_flux_tilde_tau_ext, normal_dot_flux_tilde_s_ext,
      normal_dot_flux_tilde_b_ext, normal_dot_flux_tilde_phi_ext,
      largest_outgoing_char_speed_ext, largest_ingoing_char_speed_ext,
      fast_outgoing_char_speed_ext, fast_ingoing_char_speed_ext,
      interface_unit_normal_ext, interface_unit_normal_vector_ext,
      dg_formulation);

  // WavesToRestore: None -- every Einfeldt coefficient is zero, which is
  // identically the HLL flux, so the flux is finished.
  const std::vector<size_t> restored_waves =
      restored_wave_indices(waves_to_restore_);
  if (restored_waves.empty()) {
    return;
  }
  const size_t num_points = get(tilde_d_int).size();

  // The two fans, exactly as `Hll` forms them: the light-speed fan of the
  // divergence-cleaning subsystem (used by the fan guard below) and the
  // fast-magnetosonic fan of the MHD variables, whose bounds lambda_L and
  // lambda_R are the ones every HLLEM coefficient below refers to.
  const DataVector lambda_max = max(0.0, get(largest_outgoing_char_speed_int),
                                    -get(largest_ingoing_char_speed_ext));
  const DataVector lambda_min = min(0.0, get(largest_ingoing_char_speed_int),
                                    -get(largest_outgoing_char_speed_ext));
  const DataVector inv_dl = 1.0 / max(lambda_max - lambda_min, 1.0e-30);
  const DataVector fast_lambda_max = max(0.0, get(fast_outgoing_char_speed_int),
                                         -get(fast_ingoing_char_speed_ext));
  const DataVector fast_lambda_min = min(0.0, get(fast_ingoing_char_speed_int),
                                         -get(fast_outgoing_char_speed_ext));
  const DataVector fast_inv_dl =
      1.0 / max(fast_lambda_max - fast_lambda_min, 1.0e-30);
  const DataVector fast_lprod = fast_lambda_max * fast_lambda_min;

  // The face geometry: the mean of the two sides' lapse, shift and spatial
  // metric (on a DG face of a smooth background they are equal), the unit
  // normal n_i and its raised form n^i.
  const Scalar<DataVector> lapse_avg{0.5 * (get(lapse_int) + get(lapse_ext))};
  tnsr::I<DataVector, 3, Frame::Inertial> shift_avg{num_points};
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric_avg{num_points};
  for (size_t i = 0; i < 3; ++i) {
    shift_avg.get(i) = 0.5 * (shift_int.get(i) + shift_ext.get(i));
    for (size_t j = i; j < 3; ++j) {
      spatial_metric_avg.get(i, j) =
          0.5 * (spatial_metric_int.get(i, j) + spatial_metric_ext.get(i, j));
    }
  }
  const auto [det_spatial_metric_avg, inv_spatial_metric_avg] =
      determinant_and_inverse(spatial_metric_avg);
  const DataVector sqrt_det_spatial_metric_int =
      sqrt(get(determinant(spatial_metric_int)));
  const DataVector sqrt_det_spatial_metric_ext =
      sqrt(get(determinant(spatial_metric_ext)));
  const DataVector sqrt_det_spatial_metric_avg =
      sqrt(get(det_spatial_metric_avg));
  // The face normal: the mean of the two sides' (opposite) normals, unit with
  // respect to the averaged metric, and raised with it. Built symmetrically,
  // as `Hll` builds its normal split, so that the int<->ext swap flips it
  // exactly and the correction stays conservative.
  tnsr::i<DataVector, 3, Frame::Inertial> unit_normal{num_points};
  for (size_t i = 0; i < 3; ++i) {
    unit_normal.get(i) = 0.5 * (interface_unit_normal_int.get(i) -
                                interface_unit_normal_ext.get(i));
  }
  {
    DataVector normal_squared{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        normal_squared += inv_spatial_metric_avg.get(i, j) *
                          unit_normal.get(i) * unit_normal.get(j);
      }
    }
    const DataVector one_over_norm = 1.0 / sqrt(normal_squared);
    for (size_t i = 0; i < 3; ++i) {
      unit_normal.get(i) *= one_over_norm;
    }
  }
  tnsr::I<DataVector, 3, Frame::Inertial> unit_normal_vector{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      unit_normal_vector.get(i) +=
          inv_spatial_metric_avg.get(i, j) * unit_normal.get(j);
    }
  }
  Scalar<DataVector> shift_dot_normal{num_points};
  dot_product(make_not_null(&shift_dot_normal), shift_avg, unit_normal);

  // --- HLLEM anti-diffusion (eigensystem at the averaged state) ---
  // FP exceptions stay ENABLED here. They used to be suppressed because the
  // analytic eigenvectors diverge at degeneracies, with the resulting inf/NaN
  // masked to "fall back to HLL at this point". That hid a real division by
  // zero (p -> 0 for atmosphere cells, Characteristics.cpp), which then killed
  // the run far away in FixConservatives on the strong-blast tests. The
  // denominators are floored at the point of division instead, so a trap here
  // now means a genuine bug rather than an expected degeneracy.

  // Averaged primitive state. The interface state the eigensystem is built at
  // must be a THERMODYNAMICALLY CONSISTENT state: averaging rho, eps and p
  // independently gives a triple that satisfies no equation of state (e.g. for
  // an ideal fluid with rho jumping 1 -> 10 at fixed p, the averaged eps is
  // 0.825 while the consistent value at the averaged density is 0.273). The
  // eigenvectors are then those of no physical state, and the projection of a
  // finite jump is correspondingly wrong -- measurably so: with the naive
  // average the anti-diffusion leaves ~18% of the HLL diffusion in place on a
  // 10:1 stationary contact (which HLLC/HLLD capture exactly), and leaks
  // anti-diffusion into TildeTau whose jump is exactly zero. See
  // Test_Hllem.cpp, test_stationary_contact_is_exact.
  //
  // We therefore average rho and p -- so that a CONTINUOUS pressure (the
  // defining property of a contact) is preserved exactly -- and derive eps from
  // the equation of state at (rho_avg, p_avg). Inverting p(rho, T) for T is
  // done with a secant iteration seeded by the two sides' own temperatures;
  // for an ideal fluid p is linear in T at fixed rho, so the first step is
  // exact.
  Scalar<DataVector> rho_avg{
      0.5 * (get(rest_mass_density_int) + get(rest_mass_density_ext))};
  Scalar<DataVector> p_avg{0.5 * (get(pressure_int) + get(pressure_ext))};
  const Scalar<DataVector> ye_avg{0.5 * (get(tilde_ye_int) / get(tilde_d_int) +
                                         get(tilde_ye_ext) / get(tilde_d_ext))};
  Scalar<DataVector> eps_avg{0.5 * (get(specific_internal_energy_int) +
                                    get(specific_internal_energy_ext))};
  {
    DataVector temperature_a =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density_int, specific_internal_energy_int, ye_avg));
    DataVector temperature_b =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density_ext, specific_internal_energy_ext, ye_avg));
    const auto pressure_residual = [&](const DataVector& temperature) {
      return DataVector{
          get(equation_of_state.pressure_from_density_and_temperature(
              rho_avg, Scalar<DataVector>{temperature}, ye_avg)) -
          get(p_avg)};
    };
    DataVector residual_a = pressure_residual(temperature_a);
    DataVector residual_b = pressure_residual(temperature_b);
    // Two secant steps: the first is exact for an ideal fluid, the second is
    // insurance for a general (non-linear in T) equation of state.
    for (size_t iteration = 0; iteration < 2; ++iteration) {
      const DataVector denominator = residual_b - residual_a;
      DataVector temperature_new = 0.5 * (temperature_a + temperature_b);
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (std::abs(denominator[pt]) > 1.0e-14) {
          temperature_new[pt] =
              temperature_b[pt] - residual_b[pt] *
                                      (temperature_b[pt] - temperature_a[pt]) /
                                      denominator[pt];
        }
        // Temperatures must stay physical; fall back to the midpoint if the
        // secant step leaves the (non-negative) physical range.
        if (not std::isfinite(temperature_new[pt]) or
            temperature_new[pt] < 0.0) {
          temperature_new[pt] = 0.5 * (temperature_a[pt] + temperature_b[pt]);
        }
      }
      temperature_a = temperature_b;
      residual_a = residual_b;
      temperature_b = temperature_new;
      residual_b = pressure_residual(temperature_b);
    }
    eps_avg =
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            rho_avg, Scalar<DataVector>{temperature_b}, ye_avg);
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (not std::isfinite(get(eps_avg)[pt]) or get(eps_avg)[pt] < 0.0) {
        get(eps_avg)[pt] = 0.5 * (get(specific_internal_energy_int)[pt] +
                                  get(specific_internal_energy_ext)[pt]);
      }
    }
  }
  tnsr::I<DataVector, 3, Frame::Inertial> v_avg{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> b_avg{num_points};
  for (size_t i = 0; i < 3; ++i) {
    v_avg.get(i) =
        0.5 * (spatial_velocity_int.get(i) + spatial_velocity_ext.get(i));
    // B^i = TildeB^i / sqrt(gamma) on each side
    b_avg.get(i) = 0.5 * (tilde_b_int.get(i) / sqrt_det_spatial_metric_int +
                          tilde_b_ext.get(i) / sqrt_det_spatial_metric_ext);
  }
  // Lorentz factor and enthalpy consistent with the averaged velocity.
  DataVector v_sq_avg{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      v_sq_avg += spatial_metric_avg.get(i, j) * v_avg.get(i) * v_avg.get(j);
    }
  }
  v_sq_avg = clamp(v_sq_avg, 0.0, 1.0 - 1.0e-10);
  Scalar<DataVector> w_avg{1.0 / sqrt(1.0 - v_sq_avg)};
  // The enthalpy handed to the eigensystem is built from the equation of
  // state's OWN pressure at (rho_avg, eps_avg), not from p_avg.
  //
  // Whenever the secant above converged, p(rho_avg, eps_avg) IS p_avg -- for
  // an ideal fluid exactly, since the first secant step is exact -- so this
  // changes nothing on the path the block was written for. It matters on the
  // fallback path just above, where a non-finite or negative inversion result
  // sends eps back to the raw arithmetic average: rho, eps and p are then
  // averaged INDEPENDENTLY and the triple satisfies no equation of state. That
  // is precisely the construction that made c_s^2 superluminal in PlutoHlld
  // and aborted a Del Zanna et al. (2003) jet run, because
  // `characteristic_speeds_mhd` forms
  //     c_s^2 = (chi + kappa p/rho^2) / h
  // with chi and kappa at (rho, eps) but h from p, and for an ideal fluid that
  // ratio tends to Gamma (Gamma - 1) as p/rho -> 0 -- above 1 for every Gamma
  // past the golden ratio 1.618. Evaluating all three of chi, kappa and h at
  // ONE state removes the mismatch identically, so c_s^2 is bounded by the
  // equation of state's own sound speed whatever eps_avg turns out to be, and
  // the fallback can no longer produce a non-hyperbolic quartic.
  //
  // Unlike PlutoHlld, HLLEM cannot simply take its bounds two-sidedly: the
  // anti-diffusive correction needs a single set of interface eigenVECTORS, so
  // it needs an interface state. The requirement is therefore that the
  // interface state be thermodynamically consistent, which is what this makes
  // it unconditionally.
  //
  // Scope, measured, so that this is not over-claimed. On the Del Zanna jet
  // the Hllem arm's OWN interface sound speed maxes at 0.5594 (r-faces) and
  // 0.6609 (z-faces) at the failing step -- never above Gamma - 1 = 0.6667,
  // exactly as the structural bound requires -- so the fallback above was very
  // probably never taken there and nothing in this block is implicated in that
  // run's abort, which is in the Lorentz-factor toms748 of FixConservatives
  // with D-tilde at its 1e-15 floor and has no wave speed on its path. (The
  // 0.6342 that appears in the record for that run is the PlutoHlld formula
  // applied to an Hllem solution, a counterfactual, and is a factor 6.3 off
  // Hllem's actual value on the worst face.) What this change buys is that the
  // fallback can no longer produce a non-state on ANY problem, and with it the
  // sound-speed clamp in characteristic_speeds_mhd can no longer fire from
  // here -- which matters more for Hllem than for PlutoHlld, because at
  // c_s^2 = 1 the slow pair collapses exactly onto the Alfven pair and the
  // speed-gap test downstream then drops every restored wave, i.e. Hllem
  // silently stops being HLLEM wherever that clamp fires.
  const Scalar<DataVector> p_at_avg_state =
      equation_of_state.pressure_from_density_and_energy(rho_avg, eps_avg,
                                                         ye_avg);
  const Scalar<DataVector> enthalpy_avg =
      hydro::relativistic_specific_enthalpy(rho_avg, eps_avg, p_at_avg_state);

  tnsr::i<DataVector, 9> mhd_speeds{num_points, 0.0};
  characteristic_speeds_mhd(make_not_null(&mhd_speeds), v_avg, b_avg, rho_avg,
                            eps_avg, w_avg, enthalpy_avg, spatial_metric_avg,
                            unit_normal, equation_of_state);

  tnsr::ij<DataVector, 9> modes{num_points, 0.0};
  tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
  // Always build the full analytic eigensystem. The per-wave anti-diffusion
  // uses the individual slow/Alfven/contact eigenvectors -- this is precisely
  // what distinguishes ContactSlow from ContactAlfven (the M&M Fig 13 knob).
  // The complementary projection is used only as a per-point fallback where
  // those eigenvectors genuinely collapse (below).
  characteristic_eigenvectors_mhd(
      make_not_null(&modes), make_not_null(&projectors), mhd_speeds, v_avg,
      b_avg, rho_avg, eps_avg, w_avg, enthalpy_avg, spatial_metric_avg,
      unit_normal, equation_of_state, false);

  // The eigenvectors do not depend on the lapse and shift: the physical
  // flux Jacobian is alpha A - beta_n I, with A the one at unit lapse and zero
  // shift, so its eigenvalues are alpha lambda_k - beta_n. These are the speeds
  // that enter the Einfeldt coefficients and the fan and degeneracy tests.
  tnsr::i<DataVector, 9> phys_speeds{num_points};
  for (size_t k = 0; k < 9; ++k) {
    phys_speeds.get(k) =
        get(lapse_avg) * mhd_speeds.get(k) - get(shift_dot_normal);
  }

  // conserved-variable jump in the eigenvector ordering
  // [S_x,S_y,S_z, B_x,B_y,B_z, D, Tau, Phi]
  std::array<DataVector, 9> du{};
  for (size_t k = 0; k < 3; ++k) {
    gsl::at(du, k) = tilde_s_ext.get(k) - tilde_s_int.get(k);
    gsl::at(du, 3 + k) = tilde_b_ext.get(k) - tilde_b_int.get(k);
  }
  du[6] = get(tilde_d_ext) - get(tilde_d_int);
  du[7] = get(tilde_tau_ext) - get(tilde_tau_int);
  du[8] = get(tilde_phi_ext) - get(tilde_phi_int);

  // Anti-diffusion uses the FAST (MHD) bounds -- consistent with the fast-speed
  // MHD baseline above. The fast waves then sit at the fan edge (delta -> 0),
  // so restoring them is a stable no-op instead of the +/-c over-restoration.
  const DataVector coeff = fast_lprod * fast_inv_dl;
  std::array<DataVector, 9> antidiff{};
  for (size_t n = 0; n < 9; ++n) {
    gsl::at(antidiff, n) = DataVector{num_points, 0.0};
  }

  // Per-wave guards. A restored wave is DROPPED at a point where its speed is
  // within DegeneracyTolerance of another wave's, where |l_k.r_k| vanishes, or
  // where its condition number exceeds the guard; its per-wave term is zero
  // there and, with the complementary projection, it is restored as part of a
  // block below. Waves outside the HLL fan are skipped but not dropped: there
  // is nothing of theirs to restore (as in PLUTO's hllem.c, the Einfeldt
  // coefficient assumes lambda_min <= lambda_k <= lambda_max, and
  // anti-diffusing an out-of-fan wave is both inconsistent and unstable; with
  // the fast bounds the fast waves land at the edge and are skipped here,
  // their delta being 0 anyway -- exactly the intended no-op).
  //
  // characteristic_eigenvectors_mhd returns biorthogonal but NOT
  // biorthonormal eigenvectors (l_k . r_k is not 1), so the spectral
  // projection of a jump onto wave k is r_k (l_k.dU)/(l_k.r_k), NOT
  // r_k (l_k.dU). Without the 1/(l_k.r_k) normalization each restored wave is
  // scaled by l_k.r_k, which over/under-restores it and leaks a residual into
  // the other characteristic fields -- producing spurious oscillations in
  // regions that should stay flat (e.g. behind the contact on the isolated
  // contact-wave test). Normalize by the diagonal, exactly as Marquina does.
  std::array<DataVector, 9> delta{};
  std::array<DataVector, 9> diagonal{};
  std::array<DataVector, 9> dropped{};
  std::array<DataVector, 9> wave_ok{};
  for (size_t k = 0; k < 9; ++k) {
    const DataVector& lam = phys_speeds.get(k);
    const DataVector lambdap = max(lam, 0.0);
    const DataVector lambdam = min(lam, 0.0);
    gsl::at(delta, k) = 1.0 - lambdam / (fast_lambda_min - 1.0e-14) -
                        lambdap / (fast_lambda_max + 1.0e-14);
    gsl::at(diagonal, k) = DataVector{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      gsl::at(diagonal, k) += projectors.get(k, n) * modes.get(k, n);
    }
    gsl::at(dropped, k) = DataVector{num_points, 0.0};
    gsl::at(wave_ok, k) = DataVector{num_points, 0.0};
  }
  for (const size_t wave : restored_waves) {
    const DataVector& lam = phys_speeds.get(wave);
    DataVector& ok = gsl::at(wave_ok, wave);
    DataVector& drop = gsl::at(dropped, wave);
    ok = 1.0;
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (lam[pt] >= fast_lambda_max[pt] or lam[pt] <= fast_lambda_min[pt]) {
        ok[pt] = 0.0;
      }
    }
    for (size_t j = 0; j < 9; ++j) {
      if (j == wave) {
        continue;
      }
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (std::abs(lam[pt] - phys_speeds.get(j)[pt]) <
            degeneracy_tolerance_) {
          ok[pt] = 0.0;
          drop[pt] = 1.0;
        }
      }
    }
    const DataVector& diag = gsl::at(diagonal, wave);
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (not std::isfinite(diag[pt]) or std::abs(diag[pt]) < 1.0e-12) {
        ok[pt] = 0.0;
        drop[pt] = 1.0;
      }
    }
    // Conditioning guard: drop the wave where the norm of its spectral
    // projector, |l| |r| / |l.r| (the eigenvalue condition number), exceeds
    // MaxConditionNumber. Unlike the floor on |l.r|, this is invariant to the
    // eigenvectors' normalization. Note what it measures on a cold gas: in
    // conserved variables the contact and slow eigenvectors differ only at
    // O(c_s), so their condition numbers are chi_e/c_s^2 and chi_e/(2 c_s^2)
    // (chi_e = dp/d(rho eps)|_rho, Gamma - 1 for an ideal gas) on EVERY
    // face -- 4e3 and 2e3 at p/rho = 1e-4 -- with the speeds well apart. The
    // guard then trips everywhere, and what follows must be right: with the
    // projection off the waves are simply not restored (HLL for them); with
    // it on they are restored as ONE block with a common coefficient (below),
    // which is the correct treatment of a nearly parallel group -- restored
    // one by one, their coefficients differ by a few per cent and, times
    // kappa, that amplifies any jump that is not a clean combination of
    // eigenvectors (a boundary mismatch, floor rows) into a transverse-field
    // drift. On a cold gas use the projection.
    if (max_condition_number_.has_value()) {
      DataVector norm_l{num_points, 0.0};
      DataVector norm_r{num_points, 0.0};
      for (size_t n = 0; n < 9; ++n) {
        norm_l += square(projectors.get(wave, n));
        norm_r += square(modes.get(wave, n));
      }
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (ok[pt] > 0.0 and not(sqrt(norm_l[pt] * norm_r[pt]) <=
                                 *max_condition_number_ * std::abs(diag[pt]))) {
          ok[pt] = 0.0;
          drop[pt] = 1.0;
        }
      }
    }
  }

  // With the complementary projection, a dropped wave's degenerate
  // neighbours -- any of the nine, restored or not -- belong to the same
  // CLUSTER: at a genuine collapse they cannot be told apart, so the block
  // below covers all of them. A restored wave within DegeneracyTolerance of a
  // dropped one has been dropped itself by the symmetric speed-gap test, so
  // the clustering changes no per-wave term.
  std::array<DataVector, 9> in_cluster = dropped;
  DataVector any_cluster{num_points, 0.0};
  if (use_complementary_projection_) {
    for (size_t k = 0; k < 9; ++k) {
      const DataVector& drop = gsl::at(dropped, k);
      for (size_t j = 0; j < 9; ++j) {
        if (j == k) {
          continue;
        }
        for (size_t pt = 0; pt < num_points; ++pt) {
          if (drop[pt] > 0.0 and
              std::abs(phys_speeds.get(k)[pt] - phys_speeds.get(j)[pt]) <
                  degeneracy_tolerance_) {
            gsl::at(in_cluster, j)[pt] = 1.0;
          }
        }
      }
    }
    for (size_t k = 0; k < 9; ++k) {
      any_cluster = max(any_cluster, gsl::at(in_cluster, k));
    }
  }

  // Per-wave anti-diffusion for the restored internal waves, each carried by
  // its own Einfeldt coefficient.
  for (const size_t wave : restored_waves) {
    const DataVector ok =
        gsl::at(wave_ok, wave) * (1.0 - gsl::at(in_cluster, wave));
    // 1/diag where ok, else 0. Selected per point: ok / (diag + (1 - ok))
    // divides 0 by 0 (FE_INVALID) at a dropped wave whose diag is exactly -1,
    // e.g. unit eigenvectors with l = -r. The same value wherever that
    // expression is finite.
    DataVector inv_diagonal{num_points, 0.0};
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (ok[pt] > 0.0) {
        inv_diagonal[pt] = 1.0 / gsl::at(diagonal, wave)[pt];
      }
    }
    DataVector ldu{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      ldu += projectors.get(wave, n) * gsl::at(du, n);
    }
    const DataVector w = coeff * gsl::at(delta, wave) * ldu * inv_diagonal;
    for (size_t n = 0; n < 9; ++n) {
      gsl::at(antidiff, n) += w * modes.get(wave, n);
    }
  }

  if (use_complementary_projection_) {
    // Complementary-projection fallback (Fedkiw-Merriman-Osher 1997) for the
    // dropped clusters. Where a restored wave collapsed above, its individual
    // eigenvector is unusable, so its per-wave term was dropped (leaving
    // plain HLL there -- exactly the M&M "HLLEM == HLL" behaviour when slow
    // modes sit on the contact). Instead, restore the cluster it belongs to
    // as ONE block via the complement of every wave OUTSIDE the cluster,
    //   P_cluster . dU = dU - sum_{k not in cluster} r_k (l_k.dU)/(l_k.r_k),
    // carried by a single Einfeldt coefficient, the smallest over the cluster:
    // at a genuine collapse the cluster shares one speed and this is that
    // speed's coefficient; where a wave was dropped on conditioning grounds
    // with its neighbours apart, no wave is restored harder than its own
    // coefficient would. Waves outside the cluster keep their own per-wave
    // terms, or none if they are not restored.
    //
    // Before 2026-11-05 the complement ran over the fast/GLM waves {0,1,7,8}
    // alone and the block replaced the whole anti-diffusion with the
    // CONTACT's coefficient. That re-restored every surviving wave: on a cold
    // tube (p/rho = 1e-4), where the guard had dropped the slow pair and the
    // contact, the Alfven pair at the fan edge was left about 1 % of its
    // upwind dissipation and the run blew up. The old and new blocks agree
    // where the dropped set is the whole fluid cluster at the contact speed
    // (the weak-field Kelvin-Helmholtz case the block was written for).
    //
    // The GLM waves {0,8} travel at the light speed, outside the fluid fan;
    // projecting them out leaves their part of dU on the HLL flux. The
    // eigenvectors are biorthogonal but not biorthonormal, so the projection
    // onto a wave is r_k (l_k.dU)/(l_k.r_k); where |l_k.r_k| < 1e-12 for a
    // wave outside the cluster the block is not applied at that point.
    std::array<DataVector, 9> cdu = du;
    DataVector complement_ok{num_points, 1.0};
    DataVector delta_block{num_points, 2.0};
    for (size_t k = 0; k < 9; ++k) {
      DataVector ldu{num_points, 0.0};
      for (size_t n = 0; n < 9; ++n) {
        ldu += projectors.get(k, n) * gsl::at(du, n);
      }
      const DataVector& diag = gsl::at(diagonal, k);
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (gsl::at(in_cluster, k)[pt] > 0.0) {
          delta_block[pt] = std::min(delta_block[pt], gsl::at(delta, k)[pt]);
          continue;
        }
        if (not std::isfinite(diag[pt]) or std::abs(diag[pt]) < 1.0e-12) {
          complement_ok[pt] = 0.0;
          continue;
        }
        const double strength = ldu[pt] / diag[pt];
        for (size_t n = 0; n < 9; ++n) {
          gsl::at(cdu, n)[pt] -= strength * modes.get(k, n)[pt];
        }
      }
    }
    // Keep the divergence-cleaning scalar phi (index 8) on the HLL flux.
    cdu[8] = DataVector{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (any_cluster[pt] > 0.0 and complement_ok[pt] > 0.0) {
          gsl::at(antidiff, n)[pt] +=
              coeff[pt] * std::max(delta_block[pt], 0.0) * gsl::at(cdu, n)[pt];
        }
      }
    }
  }

  // mask: apply the anti-diffusion only where the eigensystem is finite;
  // elsewhere keep the (conservative) HLL flux.
  DataVector mask{num_points, 1.0};
  for (size_t i = 0; i < num_points; ++i) {
    bool ok = true;
    for (size_t n = 0; ok and n < 9; ++n) {
      ok = ok and std::isfinite(gsl::at(antidiff, n)[i]);
    }
    mask[i] = ok ? 1.0 : 0.0;
  }
  for (size_t n = 0; n < 9; ++n) {
    gsl::at(antidiff, n) *= mask;
  }

  // Fan admissibility guard (2026-11-09). With the numerical flux F* across
  // the fan [lambda_min, lambda_max], integrating the conservation law over
  // each half of the fan fixes the AVERAGE state in that half, whatever the
  // fan looks like inside:
  //   interior half [lambda_min, 0]:  U_int + (F* - F_int) / lambda_min,
  //   exterior half [0, lambda_max]:  U_ext - (F_ext - F*) / lambda_max,
  // and a first-order update of a cell (CFL <= 1/2) is a convex combination of
  // its own state and the half-fan averages of its two faces. HLL's two halves
  // are both U_hll, an average of the exact fan, which is why HLL stays
  // physical. HLLEM's flux F* = F_HLL - coeff A, with
  //   A = antidiff / coeff = sum_k delta_k r_k (l_k.dU)/(l_k.r_k) (+ block),
  // gives the halves
  //   U_hll - lambda_max / (lambda_max - lambda_min) A   (interior),
  //   U_hll - lambda_min / (lambda_max - lambda_min) A   (exterior),
  // and nothing above keeps them physical. (On an exact stationary contact they
  // are the two sides' own states; the EDGES U_hll -+ A of the linear profile
  // the HLLEM flux also corresponds to overshoot both sides by design and are
  // the wrong test: they fail on a 1:10 contact.) Where a half is not physical,
  // every update that uses the face averages an unphysical state into a cell.
  // So it went on MUB09's jet at 416 x 832 (jet/run11_mub09_poster): cells next
  // to the axis emptied to the density floor, the energy of one ran away within
  // a few steps and the fixer's root find aborted, with the complementary
  // projection on (t ~ 177) and off (t ~ 112), while HLL, HLLD and HLLx ran the
  // same setup to t = 300.
  //
  // The guard scales the anti-diffusion at such a point by the largest theta
  // in [0, 1] for which both half-fan averages pass fan_state_failure;
  // theta = 0 is the HLL flux. It is the counterpart of HLLD's switch to HLL
  // where its intermediate states fail, and of the theta gate of the 2026-09
  // design notes (hlld_slow_antidiffusion). U_hll uses the fast bounds of the
  // fluid variables' baseline; the normal field, continuous across the fluid
  // waves, takes the GLM fan's middle value (the baseline's light-speed HLL
  // state, exact for the linear GLM subsystem), and A's normal field
  // component, zero for every fluid wave, is dropped.
  if (fan_guard_enabled()) {
    static const bool reported_on = [] {
      Parallel::printf("Hllem: fan admissibility guard ON\n");
      return true;
    }();
    (void)reported_on;
    FanGuardStats& stats = fan_guard_stats;
    const size_t dump_limit = env_count("SPECTRE_HLLEM_FAN_GUARD_DUMP");
    std::array<double, 8> u_hll{};
    std::array<double, 8> a{};
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (coeff[pt] == 0.0 or mask[pt] == 0.0) {
        continue;
      }
      ++stats.faces;
      const double s_min = fast_lambda_min[pt];
      const double s_max = fast_lambda_max[pt];
      const double inv_width = fast_inv_dl[pt];
      const auto hll_state = [s_min, s_max, inv_width](
                                 const double u_int, const double nf_int,
                                 const double u_ext, const double nf_ext) {
        return (s_max * u_ext - s_min * u_int + nf_ext + nf_int) * inv_width;
      };
      double bn_int = 0.0;
      double bn_ext = 0.0;
      double nf_bn_int = 0.0;
      double nf_bn_ext = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(u_hll, k) = hll_state(
            tilde_s_int.get(k)[pt], normal_dot_flux_tilde_s_int.get(k)[pt],
            tilde_s_ext.get(k)[pt], normal_dot_flux_tilde_s_ext.get(k)[pt]);
        gsl::at(u_hll, 3 + k) = hll_state(
            tilde_b_int.get(k)[pt], normal_dot_flux_tilde_b_int.get(k)[pt],
            tilde_b_ext.get(k)[pt], normal_dot_flux_tilde_b_ext.get(k)[pt]);
        const double n_k = unit_normal.get(k)[pt];
        bn_int += tilde_b_int.get(k)[pt] * n_k;
        bn_ext += tilde_b_ext.get(k)[pt] * n_k;
        nf_bn_int += normal_dot_flux_tilde_b_int.get(k)[pt] * n_k;
        nf_bn_ext += normal_dot_flux_tilde_b_ext.get(k)[pt] * n_k;
      }
      u_hll[6] =
          hll_state(get(tilde_d_int)[pt], get(normal_dot_flux_tilde_d_int)[pt],
                    get(tilde_d_ext)[pt], get(normal_dot_flux_tilde_d_ext)[pt]);
      u_hll[7] = hll_state(
          get(tilde_tau_int)[pt], get(normal_dot_flux_tilde_tau_int)[pt],
          get(tilde_tau_ext)[pt], get(normal_dot_flux_tilde_tau_ext)[pt]);
      const double bn_star = (lambda_max[pt] * bn_ext -
                              lambda_min[pt] * bn_int + nf_bn_ext + nf_bn_int) *
                             inv_dl[pt];
      for (size_t n = 0; n < 8; ++n) {
        gsl::at(a, n) = gsl::at(antidiff, n)[pt] / coeff[pt];
      }
      double bn_fast = 0.0;
      double a_bn = 0.0;
      for (size_t k = 0; k < 3; ++k) {
        bn_fast += gsl::at(u_hll, 3 + k) * unit_normal.get(k)[pt];
        a_bn += gsl::at(a, 3 + k) * unit_normal.get(k)[pt];
      }
      for (size_t k = 0; k < 3; ++k) {
        gsl::at(u_hll, 3 + k) +=
            (bn_star - bn_fast) * unit_normal_vector.get(k)[pt];
        gsl::at(a, 3 + k) -= a_bn * unit_normal_vector.get(k)[pt];
      }
      const double weight_int = s_max * inv_width;
      const double weight_ext = -s_min * inv_width;
      FanGeometry geometry{};
      for (size_t i = 0, c = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j, ++c) {
          gsl::at(geometry.metric, c) = spatial_metric_avg.get(i, j)[pt];
          gsl::at(geometry.inverse_metric, c) =
              inv_spatial_metric_avg.get(i, j)[pt];
        }
      }
      geometry.sqrt_det = sqrt_det_spatial_metric_avg[pt];
      const double theta =
          fan_guard_theta(u_hll, a, weight_int, weight_ext, geometry);
      if (theta < 1.0) {
        ++stats.limited;
        stats.theta_sum += theta;
        if (theta == 0.0) {
          ++stats.to_hll;
        }
        // the lowest-numbered test either half fails at theta = 1
        const auto halves =
            fan_half_averages(u_hll, a, weight_int, weight_ext, 1.0);
        const int failure_int = fan_state_failure(halves[0], geometry);
        const int failure_ext = fan_state_failure(halves[1], geometry);
        const int failure =
            failure_int == 0
                ? failure_ext
                : (failure_ext == 0 ? failure_int
                                    : std::min(failure_int, failure_ext));
        ++gsl::at(stats.first_failure, static_cast<size_t>(failure));
        if (stats.dumped < dump_limit) {
          ++stats.dumped;
          const auto side = [pt](const Scalar<DataVector>& rho,
                                 const Scalar<DataVector>& p,
                                 const tnsr::I<DataVector, 3>& v,
                                 const tnsr::I<DataVector, 3>& b) {
            return MakeString{} << std::scientific << std::setprecision(16)
                                << "rho " << get(rho)[pt] << " p " << get(p)[pt]
                                << " v (" << v.get(0)[pt] << ", "
                                << v.get(1)[pt] << ", " << v.get(2)[pt]
                                << ") B (" << b.get(0)[pt] << ", "
                                << b.get(1)[pt] << ", " << b.get(2)[pt] << ")";
          };
          const auto state_string = [](const std::array<double, 8>& s) {
            return MakeString{} << std::scientific << std::setprecision(6)
                                << "S (" << s[0] << ", " << s[1] << ", " << s[2]
                                << ") B (" << s[3] << ", " << s[4] << ", "
                                << s[5] << ") D " << s[6] << " Tau " << s[7];
          };
          Parallel::printf(
              "Hllem fan guard: theta %.6e (failed test %d at theta 1), "
              "normal (%g, %g, %g), fan [%.6e, %.6e]\n  int %s\n  ext %s\n"
              "  U_hll %s\n  half_int %s\n  half_ext %s\n",
              theta, failure, unit_normal.get(0)[pt], unit_normal.get(1)[pt],
              unit_normal.get(2)[pt], s_min, s_max,
              std::string(side(rest_mass_density_int, pressure_int,
                               spatial_velocity_int, tilde_b_int)),
              std::string(side(rest_mass_density_ext, pressure_ext,
                               spatial_velocity_ext, tilde_b_ext)),
              std::string(state_string(u_hll)),
              std::string(state_string(halves[0])),
              std::string(state_string(halves[1])));
        }
        for (size_t n = 0; n < 9; ++n) {
          gsl::at(antidiff, n)[pt] *= theta;
        }
      }
    }
    ++stats.calls;
    if (const size_t every = env_count("SPECTRE_HLLEM_STATS_EVERY");
        every > 0 and stats.calls % every == 0) {
      Parallel::printf(
          "Hllem fan guard (one thread, %zu calls): %zu faces, %zu limited "
          "(%.3e of faces; failed test D %zu, Tau %zu, E %zu), %zu to HLL, "
          "mean theta when limited %.4f\n",
          stats.calls, stats.faces, stats.limited,
          static_cast<double>(stats.limited) /
              static_cast<double>(std::max(stats.faces, size_t{1})),
          stats.first_failure[1], stats.first_failure[2],
          stats.first_failure[3], stats.to_hll,
          stats.theta_sum /
              static_cast<double>(std::max(stats.limited, size_t{1})));
    }
  } else {
    static const bool reported_off = [] {
      Parallel::printf(
          "Hllem: fan admissibility guard OFF (SPECTRE_HLLEM_FAN_GUARD=0)\n");
      return true;
    }();
    (void)reported_off;
  }

  // G_HLLEM = G_HLL - antidiff, so subtract from the HLL boundary correction.
  for (size_t k = 0; k < 3; ++k) {
    boundary_correction_tilde_s->get(k) -= gsl::at(antidiff, k);
    boundary_correction_tilde_b->get(k) -= gsl::at(antidiff, 3 + k);
  }
  get(*boundary_correction_tilde_d) -= antidiff[6];
  get(*boundary_correction_tilde_tau) -= antidiff[7];
  get(*boundary_correction_tilde_phi) -= antidiff[8];
}

bool operator==(const Hllem& lhs, const Hllem& rhs) {
  return lhs.waves_to_restore_ == rhs.waves_to_restore_ and
         lhs.use_complementary_projection_ ==
             rhs.use_complementary_projection_ and
         lhs.degeneracy_tolerance_ == rhs.degeneracy_tolerance_ and
         lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_ and
         lhs.max_condition_number_ == rhs.max_condition_number_;
}
bool operator!=(const Hllem& lhs, const Hllem& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hllem::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

template <>
grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves Options::
    create_from_yaml<grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves>::
        create<void>(const Options::Option& options) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "Contact") {
    return bc::HllemWaves::Contact;
  } else if (type_read == "ContactAlfven") {
    return bc::HllemWaves::ContactAlfven;
  } else if (type_read == "ContactSlow") {
    return bc::HllemWaves::ContactSlow;
  } else if (type_read == "All") {
    return bc::HllemWaves::All;
  } else if (type_read == "ContactAlfvenFast") {
    return bc::HllemWaves::ContactAlfvenFast;
  } else if (type_read == "AllWithFast") {
    return bc::HllemWaves::AllWithFast;
  } else if (type_read == "None") {
    return bc::HllemWaves::None;
  } else if (type_read == "Slow") {
    return bc::HllemWaves::Slow;
  } else if (type_read == "Alfven") {
    return bc::HllemWaves::Alfven;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to HllemWaves. Must be one of Contact, ContactAlfven, "
                     "ContactSlow, All, ContactAlfvenFast, AllWithFast, None, "
                     "Slow, or Alfven.");
}
