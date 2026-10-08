// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <cstddef>
#include <exception>
#include <ostream>
#include <pup.h>

#include <memory>
#include <optional>
#include <vector>

#include <string>

#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/lapack/geev.h>
#include <complex>
#include <cstdio>
#include <cstdlib>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HllemProbe.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/System/ParallelInfo.hpp"

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

std::ostream& operator<<(std::ostream& os,
                         const HllemEigensystem eigensystem) {
  switch (eigensystem) {
    case HllemEigensystem::Analytic:
      return os << "Analytic";
    case HllemEigensystem::Numeric:
      return os << "Numeric";
    default:
      ERROR("Unknown HllemEigensystem");
  }
}

namespace {
// Indices (MhdSpeed enum order) of the internal waves the anti-diffusion
// restores. Contact=4 (Entropy), Alfven={2,6}, Slow={3,5}; the outer fast
// waves (1,7) and GLM scalars (0,8) are the HLL / divergence-cleaning waves.
// ---- DEBUG PROBE (branch hllem_degeneracy_debug; not for merging) --------
// With SPECTRE_HLLEM_PROBE_DIR set, every call of dg_boundary_terms appends a
// summary record to <dir>/sum.<proc>.bin and, for each face point that has a
// nonzero jump AND either keeps a restored wave whose computed speed gap is
// below 1e-6 or anti-diffuses more than twice the HLL jump term, a full record
// to <dir>/rec.<proc>.bin (layout: experiments/delzanna_jet/hllem_face_probe/
// STATE.md). Nothing evolved is changed. Files are flushed every call,
// so a run that dies with an FPE keeps everything up to its last call.
constexpr size_t probe_record_size = 156;
// Summary v2 (file sum2.<proc>.bin): slots 0-19 as v1; 20-24 points with dU != 0
// where wave 2..6 is inside the fast fan; 25-29 of those whose speed gap also
// passes DegeneracyTolerance. With SPECTRE_HLLEM_PROBE_SUMMARY_ONLY set, no
// per-face records are written.
constexpr size_t probe_summary_size = 30;
struct ProbeFiles {
  std::FILE* rec = nullptr;
  std::FILE* sum = nullptr;
};
ProbeFiles& probe_files(const char* dir) {
  static thread_local ProbeFiles files{};
  if (files.rec == nullptr) {
    const std::string proc = std::to_string(sys::my_proc());
    files.rec = std::fopen(
        (std::string(dir) + "/rec." + proc + ".bin").c_str(), "ab");
    files.sum = std::fopen(
        (std::string(dir) + "/sum2." + proc + ".bin").c_str(), "ab");
    if (files.rec == nullptr or files.sum == nullptr) {
      ERROR("HLLEM probe: cannot open files in " << dir);
    }
  }
  return files;
}

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
}  // namespace

Hllem::Hllem(const HllemWaves waves_to_restore,
             const bool use_complementary_projection,
             const double degeneracy_tolerance,
             const double magnetic_field_magnitude_for_hydro,
             const double light_speed_density_cutoff,
             const HllemEigensystem eigensystem,
             const double max_projector_norm)
    : waves_to_restore_(waves_to_restore),
      use_complementary_projection_(use_complementary_projection),
      degeneracy_tolerance_(degeneracy_tolerance),
      magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff),
      eigensystem_(eigensystem),
      max_projector_norm_(max_projector_norm) {}

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
  p | eigensystem_;
  p | max_projector_norm_;
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
    const gsl::not_null<Scalar<DataVector>*>
        packaged_fast_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,
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

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& lorentz_factor,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  Scalar<DataVector> shift_dot_normal = tilde_d;
  dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
  get(*packaged_largest_outgoing_char_speed) =
      get(lapse) - get(shift_dot_normal);
  get(*packaged_largest_ingoing_char_speed) =
      -get(lapse) - get(shift_dot_normal);

  if (const bool has_b_field =
          max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
      not has_b_field and
      (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
    const size_t num_points = get(rest_mass_density).size();
    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                         ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                         ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                         ::Tags::TempScalar<6>>>
        temp_buffer{num_points};
    auto& v_dot_normal = get<::Tags::TempScalar<0>>(temp_buffer);
    auto& v_squared = get<::Tags::TempScalar<1>>(temp_buffer);
    auto& discriminant = get(get<::Tags::TempScalar<2>>(temp_buffer));
    auto& one_minus_v2_cs2 = get<::Tags::TempScalar<3>>(temp_buffer);
    auto& one_minus_cs2 = get<::Tags::TempScalar<4>>(temp_buffer);
    auto& lapse_over_one_minus_v2_cs2 = get<::Tags::TempScalar<5>>(temp_buffer);
    auto& v_dot_normal_times_one_minus_cs2 =
        get<::Tags::TempScalar<6>>(temp_buffer);
    const Scalar<DataVector> sound_speed_squared{clamp(
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature,
            Scalar<DataVector>{get(rest_mass_density).size(), 0.0})),
        0.0, 1.0)};
    dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                normal_covector);
    dot_product(make_not_null(&v_squared), spatial_velocity,
                spatial_velocity_one_form);
    get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);
    get(one_minus_v2_cs2) = 1.0 - get(v_squared) * get(sound_speed_squared);
    get(one_minus_cs2) = 1.0 - get(sound_speed_squared);
    discriminant = get(sound_speed_squared) * (1.0 - get(v_squared)) *
                   (get(one_minus_v2_cs2) -
                    get(v_dot_normal) * get(v_dot_normal) * get(one_minus_cs2));
    discriminant = max(discriminant, 0.0);
    discriminant = sqrt(discriminant);
    get(lapse_over_one_minus_v2_cs2) = get(lapse) / get(one_minus_v2_cs2);
    get(v_dot_normal_times_one_minus_cs2) =
        get(v_dot_normal) * get(one_minus_cs2);
    for (size_t i = 0; i < num_points; ++i) {
      if (get(rest_mass_density)[i] > light_speed_density_cutoff_) {
        get(*packaged_largest_outgoing_char_speed)[i] =
            get(lapse_over_one_minus_v2_cs2)[i] *
                (get(v_dot_normal_times_one_minus_cs2)[i] + discriminant[i]) -
            get(shift_dot_normal)[i];
        get(*packaged_largest_ingoing_char_speed)[i] =
            get(lapse_over_one_minus_v2_cs2)[i] *
                (get(v_dot_normal_times_one_minus_cs2)[i] - discriminant[i]) -
            get(shift_dot_normal)[i];
      }
    }
  }
  if (normal_dot_mesh_velocity.has_value()) {
    get(*packaged_largest_outgoing_char_speed) -=
        get(*normal_dot_mesh_velocity);
    get(*packaged_largest_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;
  *packaged_interface_unit_normal = normal_covector;
  get(*packaged_metric_flatness) = abs(get(lapse) - 1.0) + abs(get<0>(shift)) +
                                   abs(get<1>(shift)) + abs(get<2>(shift));

  // --- Fast-magnetosonic signal speeds for the MHD part, PER SIDE -----------
  //
  // The MHD variables travel at the fast-magnetosonic speed, not at the light
  // speed of the divergence-cleaning subsystem, so `dg_boundary_terms` needs a
  // fast bound for them. Until now this class packaged NO per-side fast speed
  // at all, and that bound was built from `characteristic_speeds_mhd`
  // evaluated once on the arithmetic-averaged interface state -- the only
  // state available at the interface. An average is an interior point of the
  // two states, so nothing stops it falling INSIDE the true signal range,
  // which is the one thing the HLL construction requires its bounds not to do.
  // That has been measured: lambda_R > lambda_exact > lambda*_R.
  //
  // `Hll` and `PlutoHlld` in this directory already package these two speeds;
  // this is the same computation, called the same way, so that the tags mean
  // the same thing in all three solvers. The speed is the direction-
  // independent estimate a^2 = c_s^2 + v_A^2 (1 - c_s^2) pushed through the
  // relativistic dispersion relation: dropping the B.n dependence makes it an
  // UPPER bound on the true fast speed, i.e. a safe (slightly dissipative) HLL
  // bound, which is what AthenaK (Mignone & Bodo Eq. 55) and PLUTO's DAVIS
  // estimate (`hll_speed.c:44-58`) also use.
  *packaged_fast_outgoing_char_speed = *packaged_largest_outgoing_char_speed;
  *packaged_fast_ingoing_char_speed = *packaged_largest_ingoing_char_speed;
  // The flat-space decomposition (and the identity metric used below) only
  // holds where the background is flat; elsewhere the fast bounds stay at the
  // light speed, and `dg_boundary_terms` returns the plain HLL flux there
  // anyway. A face is either all-flat or all-curved for the uniform
  // backgrounds this is used on, so test the whole face at once.
  if (max(get(*packaged_metric_flatness)) <= 1.0e-12) {
    const ScopedFpeState fpe(false);
    const size_t num_points = get(rest_mass_density).size();
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      flat_metric.get(i, i) = 1.0;
    }
    // In flat space TildeB = sqrt(gamma) B^i = B^i.
    std::array<DataVector, 9> fast_speeds{};
    characteristic_speeds_approximate_mhd(
        make_not_null(&fast_speeds), rest_mass_density, electron_fraction,
        specific_internal_energy, specific_enthalpy, spatial_velocity,
        lorentz_factor, tilde_b, lapse, shift, flat_metric, normal_covector,
        equation_of_state);
    // Indices 1 and 7 are the ingoing and outgoing fast-magnetosonic speeds.
    get(*packaged_fast_outgoing_char_speed) = fast_speeds[7];
    get(*packaged_fast_ingoing_char_speed) = fast_speeds[1];
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_fast_outgoing_char_speed) -= get(*normal_dot_mesh_velocity);
      get(*packaged_fast_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
    }
    // In the atmosphere keep the light speed, exactly as the Largest speeds do
    // (they already carry the mesh-velocity correction) and exactly as `Hll`
    // does. This is not merely a convention: as rho -> 0 at fixed B the
    // Alfven speed -> c, so the fast speed there really is ~c.
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (get(rest_mass_density)[pt] <= light_speed_density_cutoff_) {
        get(*packaged_fast_outgoing_char_speed)[pt] =
            get(*packaged_largest_outgoing_char_speed)[pt];
        get(*packaged_fast_ingoing_char_speed)[pt] =
            get(*packaged_largest_ingoing_char_speed)[pt];
      }
    }
  }

  *packaged_rest_mass_density = rest_mass_density;
  *packaged_spatial_velocity = spatial_velocity;
  *packaged_pressure = pressure;
  *packaged_lorentz_factor = lorentz_factor;
  *packaged_specific_internal_energy = specific_internal_energy;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);

  using std::max;
  return max(max(abs(get(*packaged_largest_outgoing_char_speed))),
             max(abs(get(*packaged_largest_ingoing_char_speed))));
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
    const Scalar<DataVector>& metric_flatness_int,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*iface_normal_ext*/,
    const Scalar<DataVector>& metric_flatness_ext,
    const Scalar<DataVector>& rest_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& /*lorentz_factor_ext*/,
    const Scalar<DataVector>& specific_internal_energy_ext,
    const dg::Formulation dg_formulation,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  // --- HLL baseline for all variables ---
  const DataVector lambda_max = max(0.0, get(largest_outgoing_char_speed_int),
                                    -get(largest_ingoing_char_speed_ext));
  const DataVector lambda_min = min(0.0, get(largest_ingoing_char_speed_int),
                                    -get(largest_outgoing_char_speed_ext));
  const DataVector inv_dl = 1.0 / (lambda_max - lambda_min);
  const DataVector lprod = lambda_max * lambda_min;
  auto hll = [&](const Scalar<DataVector>& u_int,
                 const Scalar<DataVector>& nf_i,
                 const Scalar<DataVector>& u_ext,
                 const Scalar<DataVector>& nf_e) -> DataVector {
    if (weak) {
      return DataVector{(lambda_max * get(nf_i) + lambda_min * get(nf_e) +
                         lprod * (get(u_ext) - get(u_int))) *
                        inv_dl};
    }
    return DataVector{(lambda_min * (get(nf_i) + get(nf_e)) +
                       lprod * (get(u_ext) - get(u_int))) *
                      inv_dl};
  };
  get(*boundary_correction_tilde_d) =
      hll(tilde_d_int, normal_dot_flux_tilde_d_int, tilde_d_ext,
          normal_dot_flux_tilde_d_ext);
  get(*boundary_correction_tilde_ye) =
      hll(tilde_ye_int, normal_dot_flux_tilde_ye_int, tilde_ye_ext,
          normal_dot_flux_tilde_ye_ext);
  get(*boundary_correction_tilde_tau) =
      hll(tilde_tau_int, normal_dot_flux_tilde_tau_int, tilde_tau_ext,
          normal_dot_flux_tilde_tau_ext);
  get(*boundary_correction_tilde_phi) =
      hll(tilde_phi_int, normal_dot_flux_tilde_phi_int, tilde_phi_ext,
          normal_dot_flux_tilde_phi_ext);
  for (size_t k = 0; k < 3; ++k) {
    if (weak) {
      boundary_correction_tilde_s->get(k) =
          (lambda_max * normal_dot_flux_tilde_s_int.get(k) +
           lambda_min * normal_dot_flux_tilde_s_ext.get(k) +
           lprod * (tilde_s_ext.get(k) - tilde_s_int.get(k))) *
          inv_dl;
      boundary_correction_tilde_b->get(k) =
          (lambda_max * normal_dot_flux_tilde_b_int.get(k) +
           lambda_min * normal_dot_flux_tilde_b_ext.get(k) +
           lprod * (tilde_b_ext.get(k) - tilde_b_int.get(k))) *
          inv_dl;
    } else {
      boundary_correction_tilde_s->get(k) =
          (lambda_min * (normal_dot_flux_tilde_s_int.get(k) +
                         normal_dot_flux_tilde_s_ext.get(k)) +
           lprod * (tilde_s_ext.get(k) - tilde_s_int.get(k))) *
          inv_dl;
      boundary_correction_tilde_b->get(k) =
          (lambda_min * (normal_dot_flux_tilde_b_int.get(k) +
                         normal_dot_flux_tilde_b_ext.get(k)) +
           lprod * (tilde_b_ext.get(k) - tilde_b_int.get(k))) *
          inv_dl;
    }
  }

  // --- HLLEM anti-diffusion (flat space; eigensystem at the average state) ---
  // The five-wave eigensystem is built assuming Minkowski (the regime of the
  // relativistic M&M tests). If the background is curved anywhere on the face we
  // keep the (conservative) HLL flux and skip the eigensystem entirely -- both
  // because the anti-diffusion would be masked out anyway and because building
  // the flat-space eigensystem from curved-metric primitives would feed
  // superluminal velocities into characteristic_speeds_mhd. (M&M backgrounds are
  // uniform, so a face is either all-flat or all-curved.)
  if (max(get(metric_flatness_int)) > 1.0e-12 or
      max(get(metric_flatness_ext)) > 1.0e-12) {
    return;
  }
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
  const Scalar<DataVector> ye_avg{
      0.5 * (get(tilde_ye_int) / get(tilde_d_int) +
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
              temperature_b[pt] -
              residual_b[pt] * (temperature_b[pt] - temperature_a[pt]) /
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
    eps_avg = equation_of_state
                  .specific_internal_energy_from_density_and_temperature(
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
    // B = TildeB in flat space
    b_avg.get(i) = 0.5 * (tilde_b_int.get(i) + tilde_b_ext.get(i));
  }
  // Lorentz factor and enthalpy consistent with the averaged velocity (flat).
  DataVector v_sq_avg{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    v_sq_avg += v_avg.get(i) * v_avg.get(i);
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

  // flat-space geometry
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = 1.0;
  }
  tnsr::i<DataVector, 3> unit_normal{num_points};
  for (size_t i = 0; i < 3; ++i) {
    unit_normal.get(i) = interface_unit_normal_int.get(i);
  }

  tnsr::i<DataVector, 9> mhd_speeds{num_points, 0.0};
  characteristic_speeds_mhd(make_not_null(&mhd_speeds), v_avg, b_avg, rho_avg,
                            eps_avg, w_avg, enthalpy_avg, flat_metric,
                            unit_normal, equation_of_state);

  // --- Scalar (divergence-cleaning) / MHD flux split -------------------------
  // In flat space the GLM subsystem (Phi and the NORMAL magnetic field)
  // decouples from the MHD system and propagates at the light speed; it keeps
  // the +/-c HLL flux computed in the baseline above (for the linear (Phi,B_n)
  // system HLL at
  // +/-c is identically LLF at c). The MHD variables (D, Ye, Tau, S and the
  // TANGENTIAL magnetic field) instead use the (much slower) fast-magnetosonic
  // HLL bounds from the characteristic speeds. This makes the flux far less
  // dissipative and -- crucially -- puts the fast waves AT the fan edge, so the
  // anti-diffusion restores them as a no-op (delta_fast -> 0) instead of the
  // over-restoration that blew up with the +/-c bounds. (Cf. PLUTO
  // Src/MHD/GLM/glm.c GLM_Solve.) Everything here stays
  // inside the boundary correction; the evolution system is unchanged.
  //
  // The outer MHD bounds are TWO-SIDED: the envelope of each side's own fast
  // speed and the averaged state's, which is the three-sided envelope
  //   S_min = MIN(0, MINVAL(Lambda(Q_bar)), MINVAL(Lambda(Q_L)), ...)
  // of Dumbser & Balsara's own HLLEM reference implementation (Appendix C).
  // Taking the averaged state ALONE bounds
  // nothing: an average is an interior point of the two states, so it can sit
  // inside the true signal range (measured:
  // lambda_R > lambda_exact > lambda*_R), and a symbolic derivation
  // shows no state function admits an unmargined average-state estimate at all
  // (297 of 300 random RMHD pairs violate it, largest deficit 0.5028).
  //
  // Keeping Lambda(Q_bar) IN the envelope is not decoration: the anti-diffusion
  // below reads its wave speeds from the averaged eigensystem and drops any
  // wave that leaves the fan, so dropping Q_bar from the bound would push the
  // averaged fast waves outside their own fan. The envelope only ever widens
  // the fan relative to the old bound, so the scheme is more dissipative, never
  // less -- and the per-side speeds are the same quantity `Hll` and
  // `PlutoHlld` already package.
  const DataVector fast_lambda_max =
      max(0.0, mhd_speeds.get(7), get(fast_outgoing_char_speed_int),
          -get(fast_ingoing_char_speed_ext));  // v_n + c_fast
  const DataVector fast_lambda_min =
      min(0.0, mhd_speeds.get(1), get(fast_ingoing_char_speed_int),
          -get(fast_outgoing_char_speed_ext));  // v_n - c_fast
  DataVector fast_dl = fast_lambda_max - fast_lambda_min;
  for (size_t pt = 0; pt < num_points; ++pt) {
    if (fast_dl[pt] < 1.0e-30) {
      fast_dl[pt] = 1.0e-30;
    }
  }
  const DataVector fast_inv_dl = 1.0 / fast_dl;
  const DataVector fast_lprod = fast_lambda_max * fast_lambda_min;
  const auto fast_hll = [&](const Scalar<DataVector>& u_int,
                            const Scalar<DataVector>& nf_i,
                            const Scalar<DataVector>& u_ext,
                            const Scalar<DataVector>& nf_e) -> DataVector {
    if (weak) {
      return DataVector{(fast_lambda_max * get(nf_i) +
                         fast_lambda_min * get(nf_e) +
                         fast_lprod * (get(u_ext) - get(u_int))) *
                        fast_inv_dl};
    }
    return DataVector{(fast_lambda_min * (get(nf_i) + get(nf_e)) +
                       fast_lprod * (get(u_ext) - get(u_int))) *
                      fast_inv_dl};
  };
  // fluid scalars: pure MHD -> fast bounds
  get(*boundary_correction_tilde_d) =
      fast_hll(tilde_d_int, normal_dot_flux_tilde_d_int, tilde_d_ext,
               normal_dot_flux_tilde_d_ext);
  get(*boundary_correction_tilde_ye) =
      fast_hll(tilde_ye_int, normal_dot_flux_tilde_ye_int, tilde_ye_ext,
               normal_dot_flux_tilde_ye_ext);
  get(*boundary_correction_tilde_tau) =
      fast_hll(tilde_tau_int, normal_dot_flux_tilde_tau_int, tilde_tau_ext,
               normal_dot_flux_tilde_tau_ext);
  // TildePhi keeps the +/-c HLL (divergence cleaning) from the baseline above.
  // momentum: pure MHD (no GLM coupling) -> fast bounds, all components
  for (size_t k = 0; k < 3; ++k) {
    if (weak) {
      boundary_correction_tilde_s->get(k) =
          (fast_lambda_max * normal_dot_flux_tilde_s_int.get(k) +
           fast_lambda_min * normal_dot_flux_tilde_s_ext.get(k) +
           fast_lprod * (tilde_s_ext.get(k) - tilde_s_int.get(k))) *
          fast_inv_dl;
    } else {
      boundary_correction_tilde_s->get(k) =
          (fast_lambda_min * (normal_dot_flux_tilde_s_int.get(k) +
                              normal_dot_flux_tilde_s_ext.get(k)) +
           fast_lprod * (tilde_s_ext.get(k) - tilde_s_int.get(k))) *
          fast_inv_dl;
    }
  }
  // Magnetic field: normal component belongs to the GLM subsystem (keep the
  // +/-c baseline), tangential component is MHD (fast bounds). Split, recompute
  // the tangential part with fast bounds, then recombine G(B^i) = G(B_n) n^i +
  // G(B_t).
  {
    DataVector bn_correction{num_points,
                             0.0};  // n_i G(B^i) from the +/-c baseline
    DataVector bn_int{num_points, 0.0};
    DataVector bn_ext{num_points, 0.0};
    DataVector nfbn_int{num_points, 0.0};
    DataVector nfbn_ext{num_points, 0.0};
    for (size_t k = 0; k < 3; ++k) {
      bn_correction += boundary_correction_tilde_b->get(k) * unit_normal.get(k);
      bn_int += tilde_b_int.get(k) * unit_normal.get(k);
      bn_ext += tilde_b_ext.get(k) * unit_normal.get(k);
      nfbn_int += normal_dot_flux_tilde_b_int.get(k) * unit_normal.get(k);
      nfbn_ext += normal_dot_flux_tilde_b_ext.get(k) * unit_normal.get(k);
    }
    for (size_t k = 0; k < 3; ++k) {
      const DataVector bt_int =
          tilde_b_int.get(k) - bn_int * unit_normal.get(k);
      const DataVector bt_ext =
          tilde_b_ext.get(k) - bn_ext * unit_normal.get(k);
      const DataVector nfbt_int =
          normal_dot_flux_tilde_b_int.get(k) - nfbn_int * unit_normal.get(k);
      const DataVector nfbt_ext =
          normal_dot_flux_tilde_b_ext.get(k) - nfbn_ext * unit_normal.get(k);
      DataVector g_bt{num_points, 0.0};
      if (weak) {
        g_bt = (fast_lambda_max * nfbt_int + fast_lambda_min * nfbt_ext +
                fast_lprod * (bt_ext - bt_int)) *
               fast_inv_dl;
      } else {
        g_bt = (fast_lambda_min * (nfbt_int + nfbt_ext) +
                fast_lprod * (bt_ext - bt_int)) *
               fast_inv_dl;
      }
      boundary_correction_tilde_b->get(k) =
          bn_correction * unit_normal.get(k) + g_bt;
    }
  }

  // WavesToRestore: None -- HLLEM with every Einfeldt coefficient delta_k set
  // to zero, which is identically the HLL flux (a nine-component algebraic
  // identity, independent of the state, the eigensystem and the equation of
  // state). Everything above this point IS the HLL baseline, so the flux is
  // already finished and nothing below can do anything but add zero: the
  // per-wave loop iterates over an empty list, and the complementary
  // projection -- which composes with WavesToRestore everywhere else -- only
  // fires at points where a RESTORED wave was dropped, so with no restored
  // waves it never flags a point either. None is plain HLL with the projection
  // on or off, which is why returning here rather than falling through is
  // exact, not an approximation; it also skips building an eigensystem that
  // would only be multiplied by zero.
  //
  // The reduction is to `Hll` CLOSELY BUT NOT BITWISE. `Hll` takes its outer
  // MHD bounds two-sidedly, from the two sides' own fast speeds; the envelope
  // above additionally carries the averaged state's fast speed, which can only
  // widen the fan. HLLEM-None is therefore slightly MORE dissipative than
  // `Hll` and never less, and that bound construction is the ONLY difference
  // between the two (the scalar/MHD split and the flux formula are shared).
  // A deviation of the opposite sign, or one bigger than the envelope
  // difference accounts for, is an implementation bug in the anti-diffusion,
  // not a tolerance to be relaxed. Pinned down by
  // Test_Hllem.cpp:test_no_restored_waves_reduces_to_hll.
  const std::vector<size_t> restored_waves =
      restored_wave_indices(waves_to_restore_);
  if (restored_waves.empty()) {
    return;
  }

  tnsr::ij<DataVector, 9> modes{num_points, 0.0};
  tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
  // Always build the full analytic eigensystem. The per-wave anti-diffusion
  // uses the individual slow/Alfven/contact eigenvectors -- this is precisely
  // what distinguishes ContactSlow from ContactAlfven (the M&M Fig 13 knob).
  // The complementary projection is used only as a per-point fallback where
  // those eigenvectors genuinely collapse (below).
  characteristic_eigenvectors_mhd(
      make_not_null(&modes), make_not_null(&projectors), mhd_speeds, v_avg,
      b_avg, rho_avg, eps_avg, w_avg, enthalpy_avg, flat_metric, unit_normal,
      equation_of_state, false);

  // DEBUG (hllem_degeneracy_debug): numeric characteristics. Replace the
  // restored waves' speeds and eigenvectors by LAPACK dgeev's for the flux
  // Jacobian at the SAME averaged state. The nine eigenvalues are sorted; for
  // an admissible state the characteristic speeds interlace (Anton et al.
  // 2010, Eq. 42), so sorted order IS the MhdSpeed order. An exact
  // degeneracy needs no special treatment: a semisimple multiple eigenvalue
  // comes out of dgeev split only by O(eps |A| kappa), so the speed-gap guard
  // below drops the whole cluster (HLL there), as the analytic path would
  // with exact speeds. A complex pair (dgeev's answer near a defective
  // cluster) has equal real parts, so the gap guard drops it too. A point
  // where dgeev throws, the matrix is non-finite, or some |Im lambda| > 1e-6
  // restores no wave at all (`numeric_failed`). The outer fast bounds stay
  // analytic.
  tnsr::i<DataVector, 9> numeric_speeds{};
  DataVector numeric_failed{num_points, 0.0};
  if (eigensystem_ == HllemEigensystem::Numeric) {
    const ScopedFpeState numeric_fpe(false);
    numeric_speeds = mhd_speeds;
    tnsr::II<DataVector, 3, Frame::Inertial> inv_flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      inv_flat_metric.get(i, i) = 1.0;
    }
    tnsr::iJ<DataVector, 9> jacobian{num_points};
    flux_jacobian_mhd(make_not_null(&jacobian), v_avg, b_avg, rho_avg,
                      eps_avg, ye_avg, w_avg, enthalpy_avg, flat_metric,
                      inv_flat_metric, unit_normal, equation_of_state);
    blaze::StaticMatrix<double, 9, 9> mat{};
    blaze::StaticVector<std::complex<double>, 9> eigenvalues{};
    blaze::StaticMatrix<std::complex<double>, 9, 9> left{};
    blaze::StaticMatrix<std::complex<double>, 9, 9> right{};
    for (size_t pt = 0; pt < num_points; ++pt) {
      bool ok = true;
      for (size_t row = 0; row < 9; ++row) {
        for (size_t col = 0; col < 9; ++col) {
          mat(row, col) = jacobian.get(row, col)[pt];
          ok = ok and std::isfinite(mat(row, col));
        }
      }
      if (ok) {
        try {
          blaze::geev(mat, left, eigenvalues, right);
        } catch (const std::exception& /*e*/) {
          ok = false;
        }
      }
      std::array<size_t, 9> order{};
      for (size_t i = 0; i < 9; ++i) {
        gsl::at(order, i) = i;
        ok = ok and std::isfinite(eigenvalues[i].real()) and
             std::abs(eigenvalues[i].imag()) <= 1.0e-6;
      }
      if (not ok) {
        numeric_failed[pt] = 1.0;
        continue;
      }
      std::sort(order.begin(), order.end(),
                [&eigenvalues](const size_t a, const size_t b) {
                  return eigenvalues[a].real() < eigenvalues[b].real();
                });
      for (size_t i = 0; i < 9; ++i) {
        const size_t s = gsl::at(order, i);
        numeric_speeds.get(i)[pt] = eigenvalues[s].real();
        for (size_t n = 0; n < 9; ++n) {
          // the convention of numerical_characteristics (and Marquina)
          modes.get(i, n)[pt] = right(n, s).real();
          projectors.get(i, n)[pt] = left(n, s).real();
        }
      }
    }
  }
  const tnsr::i<DataVector, 9>& restore_speeds =
      eigensystem_ == HllemEigensystem::Numeric ? numeric_speeds : mhd_speeds;

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

  // Per-wave anti-diffusion for the restored internal waves, each carried by
  // its own Einfeldt coefficient. A speed-gap guard drops a wave where its
  // speed collapses onto a neighbour and its individual analytic eigenvector is
  // ill-conditioned; those points are recorded for the complement fallback.
  DataVector restored_wave_dropped{num_points, 0.0};
  // DEBUG PROBE storage, index wave - 2 (waves 2..6)
  const char* const probe_dir = std::getenv("SPECTRE_HLLEM_PROBE_DIR");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  std::array<DataVector, 5> pr_ok{};
  std::array<DataVector, 5> pr_fan{};
  std::array<DataVector, 5> pr_gapok{};
  std::array<DataVector, 5> pr_diag{};
  std::array<DataVector, 5> pr_ldu{};
  std::array<DataVector, 5> pr_delta{};
  std::array<DataVector, 5> pr_gap{};
  std::array<std::array<DataVector, 9>, 5> pr_pdu{};
  if (probe_dir != nullptr) {
    for (size_t k = 0; k < 5; ++k) {
      gsl::at(pr_ok, k) = DataVector{num_points, 0.0};
      gsl::at(pr_fan, k) = DataVector{num_points, 0.0};
      gsl::at(pr_gapok, k) = DataVector{num_points, 0.0};
      gsl::at(pr_diag, k) = DataVector{num_points, nan};
      gsl::at(pr_ldu, k) = DataVector{num_points, nan};
      gsl::at(pr_delta, k) = DataVector{num_points, nan};
      gsl::at(pr_gap, k) = DataVector{num_points, nan};
      for (size_t n = 0; n < 9; ++n) {
        gsl::at(gsl::at(pr_pdu, k), n) = DataVector{num_points, nan};
      }
    }
  }
  DataVector kappa_dropped{num_points, 0.0};
  for (const size_t wave : restored_waves) {
    const DataVector& lam = restore_speeds.get(wave);
    const DataVector lambdap = max(lam, 0.0);
    const DataVector lambdam = min(lam, 0.0);
    const DataVector delta = 1.0 - lambdam / (fast_lambda_min - 1.0e-14) -
                             lambdap / (fast_lambda_max + 1.0e-14);
    DataVector wave_ok{num_points, 1.0};
    for (size_t pt = 0; pt < num_points; ++pt) {
      // Skip waves outside the HLL Riemann fan (as in PLUTO's hllem.c): the
      // Einfeldt coefficient assumes lambda_min <= lambda_k <= lambda_max, and
      // anti-diffusing an out-of-fan wave is both inconsistent and unstable.
      // With the fast bounds the fast waves land at the edge and are skipped
      // here (their delta is 0 anyway) -- exactly the intended no-op.
      if (lam[pt] >= fast_lambda_max[pt] or lam[pt] <= fast_lambda_min[pt]) {
        wave_ok[pt] = 0.0;
      }
    }
    for (size_t j = 0; j < 9; ++j) {
      if (j == wave) {
        continue;
      }
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (std::abs(lam[pt] - restore_speeds.get(j)[pt]) <
            degeneracy_tolerance_) {
          wave_ok[pt] = 0.0;
          restored_wave_dropped[pt] = 1.0;
        }
      }
    }
    // characteristic_eigenvectors_mhd returns biorthogonal but NOT
    // biorthonormal eigenvectors (l_k . r_k is not 1), so the spectral
    // projection of a jump onto wave k is r_k (l_k.dU)/(l_k.r_k), NOT
    // r_k (l_k.dU). Without the 1/(l_k.r_k) normalization each restored wave is
    // scaled by l_k.r_k, which over/under-restores it and leaks a residual into
    // the other characteristic fields -- producing spurious oscillations in
    // regions that should stay flat (e.g. behind the contact on the isolated
    // contact-wave test). Normalize by the diagonal here, exactly as Marquina
    // does; drop the wave where the diagonal is (near) zero (ill-conditioned).
    DataVector diagonal{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      diagonal += projectors.get(wave, n) * modes.get(wave, n);
    }
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (not std::isfinite(diagonal[pt]) or std::abs(diagonal[pt]) < 1.0e-12) {
        wave_ok[pt] = 0.0;
        restored_wave_dropped[pt] = 1.0;
      }
      if (numeric_failed[pt] > 0.0) {
        wave_ok[pt] = 0.0;
        restored_wave_dropped[pt] = 1.0;
      }
    }
    // DEBUG: conditioning guard, |l| |r| / |l.r| = ||r l^T / (l.r)||_2, the
    // norm of the wave's spectral projector (eigenvalue condition number).
    if (max_projector_norm_ < 1.0e300) {
      DataVector norm_l{num_points, 0.0};
      DataVector norm_r{num_points, 0.0};
      for (size_t n = 0; n < 9; ++n) {
        norm_l += square(projectors.get(wave, n));
        norm_r += square(modes.get(wave, n));
      }
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (wave_ok[pt] > 0.0 and
            not(sqrt(norm_l[pt] * norm_r[pt]) <=
                max_projector_norm_ * std::abs(diagonal[pt]))) {
          wave_ok[pt] = 0.0;
          restored_wave_dropped[pt] = 1.0;
          kappa_dropped[pt] += 1.0;
        }
      }
    }
    const DataVector inv_diagonal =
        wave_ok / (diagonal + (1.0 - wave_ok));  // 1/diag where ok, else 0
    DataVector ldu{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      ldu += projectors.get(wave, n) * gsl::at(du, n);
    }
    const DataVector w = coeff * delta * ldu * inv_diagonal;
    for (size_t n = 0; n < 9; ++n) {
      gsl::at(antidiff, n) += w * modes.get(wave, n);
    }
    if (probe_dir != nullptr and wave >= 2 and wave <= 6) {
      const ScopedFpeState probe_fpe(false);
      const size_t k = wave - 2;
      gsl::at(pr_ok, k) = wave_ok;
      gsl::at(pr_diag, k) = diagonal;
      gsl::at(pr_ldu, k) = ldu;
      gsl::at(pr_delta, k) = delta;
      DataVector gap{num_points, std::numeric_limits<double>::max()};
      for (size_t j = 0; j < 9; ++j) {
        if (j != wave) {
          gap = min(gap, abs(lam - restore_speeds.get(j)));
        }
      }
      gsl::at(pr_gap, k) = gap;
      for (size_t pt = 0; pt < num_points; ++pt) {
        const bool in_fan = lam[pt] < fast_lambda_max[pt] and
                            lam[pt] > fast_lambda_min[pt];
        gsl::at(pr_fan, k)[pt] = in_fan ? 1.0 : 0.0;
        gsl::at(pr_gapok, k)[pt] =
            (in_fan and gap[pt] >= degeneracy_tolerance_) ? 1.0 : 0.0;
      }
      for (size_t n = 0; n < 9; ++n) {
        gsl::at(gsl::at(pr_pdu, k), n) =
            ldu / diagonal * modes.get(wave, n);
      }
    }
  }

  if (use_complementary_projection_) {
    // Complementary-projection fallback (Fedkiw-Merriman-Osher 1997). Where a
    // restored wave collapsed above, its individual eigenvector is unusable so
    // the per-wave term was dropped (leaving plain HLL there -- exactly the M&M
    // "HLLEM == HLL" behaviour when slow modes sit on the contact). Instead,
    // restore the whole collapse-prone fluid subspace {2..6} as ONE block via
    // the complement of the well-conditioned fast/GLM waves {0,1,7,8},
    //   P_fluid . dU = dU - sum_{k in {0,1,7,8}} r_k (l_k . dU),
    // carried by a single Einfeldt coefficient at the (shared) contact speed.
    // The clustered waves share that speed at a genuine collapse, so the single
    // coefficient is accurate there; applying this block only at the collapse
    // points (rather than everywhere) avoids the over-restoration -- and
    // resulting instability -- that a blanket block complement produces in the
    // smooth regions where the fluid waves are well separated.
    const DataVector& lam_c = mhd_speeds.get(4);  // Entropy / contact
    const DataVector delta_c = 1.0 -
                               min(lam_c, 0.0) / (fast_lambda_min - 1.0e-14) -
                               max(lam_c, 0.0) / (fast_lambda_max + 1.0e-14);
    // Project du onto the fluid subspace via the complement of the
    // well-conditioned fast waves {1,7}. The GLM/divergence-cleaning waves
    // {0,8} travel at the light speed, are OUT OF the fluid HLL fan, and are
    // handled by the (diffusive) HLL flux -- projecting them out here would
    // corrupt the complement, so they are skipped per point when out of fan.
    std::array<DataVector, 9> cdu = du;
    const std::array<size_t, 4> nondegenerate_waves{{0, 1, 7, 8}};
    for (const size_t k : nondegenerate_waves) {
      const DataVector& lam_k = mhd_speeds.get(k);
      DataVector ldu{num_points, 0.0};
      for (size_t n = 0; n < 9; ++n) {
        ldu += projectors.get(k, n) * gsl::at(du, n);
      }
      for (size_t n = 0; n < 9; ++n) {
        const DataVector contrib = ldu * modes.get(k, n);
        for (size_t pt = 0; pt < num_points; ++pt) {
          if (lam_k[pt] < fast_lambda_max[pt] and
              lam_k[pt] > fast_lambda_min[pt]) {
            gsl::at(cdu, n)[pt] -= contrib[pt];
          }
        }
      }
    }
    // Keep the divergence-cleaning scalar phi (index 8) on the HLL flux.
    cdu[8] = DataVector{num_points, 0.0};
    for (size_t n = 0; n < 9; ++n) {
      const DataVector block = coeff * delta_c * gsl::at(cdu, n);
      for (size_t pt = 0; pt < num_points; ++pt) {
        if (restored_wave_dropped[pt] > 0.0) {
          gsl::at(antidiff, n)[pt] = block[pt];
        }
      }
    }
  }

  // mask: apply the anti-diffusion only where the metric is flat and the
  // eigensystem is finite; elsewhere keep the (conservative) HLL flux.
  DataVector mask{num_points, 1.0};
  for (size_t i = 0; i < num_points; ++i) {
    bool ok = get(metric_flatness_int)[i] <= 1.0e-12 and
              get(metric_flatness_ext)[i] <= 1.0e-12;
    for (size_t n = 0; ok and n < 9; ++n) {
      ok = ok and std::isfinite(gsl::at(antidiff, n)[i]);
    }
    mask[i] = ok ? 1.0 : 0.0;
  }
  for (size_t n = 0; n < 9; ++n) {
    gsl::at(antidiff, n) *= mask;
  }

  if (probe_dir != nullptr) {
    const ScopedFpeState probe_fpe(false);
    const auto& ctx = hllem_probe::context();
    std::array<DataVector, 9> hll_corr{};
    for (size_t k = 0; k < 3; ++k) {
      gsl::at(hll_corr, k) = boundary_correction_tilde_s->get(k);
      gsl::at(hll_corr, 3 + k) = boundary_correction_tilde_b->get(k);
    }
    hll_corr[6] = get(*boundary_correction_tilde_d);
    hll_corr[7] = get(*boundary_correction_tilde_tau);
    hll_corr[8] = get(*boundary_correction_tilde_phi);
    std::vector<size_t> selected{};
    const bool summary_only =
        std::getenv("SPECTRE_HLLEM_PROBE_SUMMARY_ONLY") != nullptr;
    std::array<double, probe_summary_size> summary{};
    summary.fill(0.0);
    summary[0] = ctx.valid ? ctx.time : nan;
    summary[1] = ctx.valid ? static_cast<double>(ctx.dim) : -1.0;
    summary[2] = ctx.valid ? ctx.xi_lower[0] : nan;
    summary[3] = ctx.valid ? ctx.xi_lower[1] : nan;
    summary[4] = static_cast<double>(num_points);
    for (size_t pt = 0; pt < num_points; ++pt) {
      double max_du = 0.0;
      double max_cdu = 0.0;
      double max_ad = 0.0;
      for (size_t n = 0; n < 9; ++n) {
        max_du = std::max(max_du, std::abs(gsl::at(du, n)[pt]));
        max_cdu = std::max(max_cdu, std::abs(coeff[pt] * gsl::at(du, n)[pt]));
        max_ad = std::max(max_ad, std::abs(gsl::at(antidiff, n)[pt]));
      }
      if (not(max_du > 0.0)) {
        continue;
      }
      summary[5] += 1.0;
      for (size_t k = 0; k < 5; ++k) {
        summary[20 + k] += gsl::at(pr_fan, k)[pt];
        summary[25 + k] += gsl::at(pr_gapok, k)[pt];
      }
      bool near_degenerate_kept = false;
      for (size_t k = 0; k < 5; ++k) {
        if (gsl::at(pr_ok, k)[pt] > 0.0) {
          summary[6 + k] += 1.0;
          if (gsl::at(pr_gap, k)[pt] < 1.0e-6) {
            summary[11 + k] += 1.0;
            near_degenerate_kept = true;
          }
        }
      }
      const bool amplified = max_ad > 2.0 * max_cdu;
      if (amplified) {
        summary[16] += 1.0;
      }
      if ((near_degenerate_kept or amplified) and not summary_only) {
        selected.push_back(pt);
      }
    }
    summary[17] = static_cast<double>(selected.size());
    for (size_t pt = 0; pt < num_points; ++pt) {
      summary[18] += numeric_failed[pt];
      summary[19] += kappa_dropped[pt];
    }
    ProbeFiles& files = probe_files(probe_dir);
    std::fwrite(summary.data(), sizeof(double), summary.size(), files.sum);
    if (not selected.empty()) {
      // double-precision numeric characteristics at the averaged state
      tnsr::II<DataVector, 3, Frame::Inertial> inv_flat_metric{num_points,
                                                               0.0};
      for (size_t i = 0; i < 3; ++i) {
        inv_flat_metric.get(i, i) = 1.0;
      }
      tnsr::iJ<DataVector, 9> jac{num_points};
      flux_jacobian_mhd(make_not_null(&jac), v_avg, b_avg, rho_avg, eps_avg,
                        ye_avg, w_avg, enthalpy_avg, flat_metric,
                        inv_flat_metric, unit_normal, equation_of_state);
      blaze::StaticMatrix<double, 9, 9> mat{};
      blaze::StaticVector<std::complex<double>, 9> eigs{};
      std::array<double, probe_record_size> rec{};
      for (const size_t pt : selected) {
        rec.fill(nan);
        size_t c = 0;
        const auto put = [&rec, &c](const double x) { gsl::at(rec, c++) = x; };
        // 0-4: time, dim, logical coordinates (3) of the face point
        put(summary[0]);
        put(summary[1]);
        {
          const size_t e0 = ctx.face_extents[0];
          const size_t e1 = ctx.face_extents[1];
          const std::array<size_t, 3> j{
              {pt % e0, (pt / e0) % e1, pt / (e0 * e1)}};
          for (size_t d = 0; d < 3; ++d) {
            const double lo = gsl::at(ctx.xi_lower, d);
            const double hi = gsl::at(ctx.xi_upper, d);
            const size_t e = gsl::at(ctx.face_extents, d);
            put(not ctx.valid ? nan
                : d == ctx.dim
                    ? lo + static_cast<double>(gsl::at(j, d)) * (hi - lo) /
                               static_cast<double>(e - 1)
                    : lo + (static_cast<double>(gsl::at(j, d)) + 0.5) *
                               (hi - lo) / static_cast<double>(e));
          }
        }
        // 5-19: averaged state
        put(get(rho_avg)[pt]);
        put(get(eps_avg)[pt]);
        put(get(p_at_avg_state)[pt]);
        put(get(enthalpy_avg)[pt]);
        put(get(w_avg)[pt]);
        for (size_t i = 0; i < 3; ++i) {
          put(v_avg.get(i)[pt]);
        }
        for (size_t i = 0; i < 3; ++i) {
          put(b_avg.get(i)[pt]);
        }
        for (size_t i = 0; i < 3; ++i) {
          put(unit_normal.get(i)[pt]);
        }
        put(get(ye_avg)[pt]);
        // 20-28 speeds, 29-30 fast bounds
        for (size_t i = 0; i < 9; ++i) {
          put(restore_speeds.get(i)[pt]);
        }
        put(fast_lambda_min[pt]);
        put(fast_lambda_max[pt]);
        // 31-55: per restored wave 2..6: ok, l.r, l.dU, delta, computed gap
        for (size_t k = 0; k < 5; ++k) {
          put(gsl::at(pr_ok, k)[pt]);
          put(gsl::at(pr_diag, k)[pt]);
          put(gsl::at(pr_ldu, k)[pt]);
          put(gsl::at(pr_delta, k)[pt]);
          put(gsl::at(pr_gap, k)[pt]);
        }
        // 56-64 dU, 65-73 antidiff (masked), 74-82 HLL correction
        for (size_t n = 0; n < 9; ++n) {
          put(gsl::at(du, n)[pt]);
        }
        for (size_t n = 0; n < 9; ++n) {
          put(gsl::at(antidiff, n)[pt]);
        }
        for (size_t n = 0; n < 9; ++n) {
          put(gsl::at(hll_corr, n)[pt]);
        }
        // 83-91 int, 92-100 ext: rho, p, eps, v(3), B(3)
        for (const bool interior : {true, false}) {
          put(get(interior ? rest_mass_density_int : rest_mass_density_ext)[pt]);
          put(get(interior ? pressure_int : pressure_ext)[pt]);
          put(get(interior ? specific_internal_energy_int
                           : specific_internal_energy_ext)[pt]);
          for (size_t i = 0; i < 3; ++i) {
            put((interior ? spatial_velocity_int : spatial_velocity_ext)
                    .get(i)[pt]);
          }
          for (size_t i = 0; i < 3; ++i) {
            put((interior ? tilde_b_int : tilde_b_ext).get(i)[pt]);
          }
        }
        // 101-109 geev eigenvalues (sorted real parts), 110 max |imag|
        for (size_t row = 0; row < 9; ++row) {
          for (size_t col = 0; col < 9; ++col) {
            mat(row, col) = jac.get(row, col)[pt];
          }
        }
        // A non-finite Jacobian entry, or LAPACK's non-convergence (blaze
        // throws std::runtime_error), is recorded, not propagated: NaN
        // eigenvalues and max |imag| = -1 (non-finite matrix) or -2 (throw).
        std::array<double, 9> re{};
        re.fill(nan);
        double max_imag = -1.0;
        bool finite_matrix = true;
        for (size_t row = 0; row < 9; ++row) {
          for (size_t col = 0; col < 9; ++col) {
            finite_matrix = finite_matrix and std::isfinite(mat(row, col));
          }
        }
        if (finite_matrix) {
          try {
            blaze::geev(mat, eigs);
            max_imag = 0.0;
            for (size_t i = 0; i < 9; ++i) {
              gsl::at(re, i) = eigs[i].real();
              max_imag = std::max(max_imag, std::abs(eigs[i].imag()));
            }
            std::sort(re.begin(), re.end());
          } catch (const std::exception& /*e*/) {
            re.fill(nan);
            max_imag = -2.0;
          }
        }
        for (size_t i = 0; i < 9; ++i) {
          put(gsl::at(re, i));
        }
        put(max_imag);
        // 111-155: (l.dU / l.r) r per restored wave 2..6
        for (size_t k = 0; k < 5; ++k) {
          for (size_t n = 0; n < 9; ++n) {
            put(gsl::at(gsl::at(pr_pdu, k), n)[pt]);
          }
        }
        ASSERT(c == probe_record_size, "probe record size " << c);
        std::fwrite(rec.data(), sizeof(double), rec.size(), files.rec);
      }
      std::fflush(files.rec);
    }
    std::fflush(files.sum);
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
         lhs.light_speed_density_cutoff_ ==
             rhs.light_speed_density_cutoff_ and
         lhs.eigensystem_ == rhs.eigensystem_ and
         lhs.max_projector_norm_ == rhs.max_projector_norm_;
}
bool operator!=(const Hllem& lhs, const Hllem& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hllem::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

template <>
grmhd::ValenciaDivClean::BoundaryCorrections::HllemEigensystem
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::HllemEigensystem>::
    create<void>(const Options::Option& options) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "Analytic") {
    return bc::HllemEigensystem::Analytic;
  } else if (type_read == "Numeric") {
    return bc::HllemEigensystem::Numeric;
  }
  PARSE_ERROR(options.context(), "Failed to convert \""
                                     << type_read
                                     << "\" to HllemEigensystem. Must be "
                                        "Analytic or Numeric.");
}

template <>
grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves>::
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
