// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <ostream>
#include <pup.h>

#include <memory>
#include <optional>
#include <vector>

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

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
    default:
      ERROR("Unknown HllemWaves");
  }
}
}  // namespace

Hllem::Hllem(const HllemWaves waves_to_restore,
             const bool use_complementary_projection,
             const double degeneracy_tolerance,
             const double magnetic_field_magnitude_for_hydro,
             const double light_speed_density_cutoff)
    : waves_to_restore_(waves_to_restore),
      use_complementary_projection_(use_complementary_projection),
      degeneracy_tolerance_(degeneracy_tolerance),
      magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

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
    const Scalar<DataVector>& /*electron_fraction*/,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
    const Scalar<DataVector>& metric_flatness_int,
    const Scalar<DataVector>& rest_mass_density_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
    const Scalar<DataVector>& pressure_int,
    const Scalar<DataVector>& lorentz_factor_int,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& /*iface_normal_ext*/,
    const Scalar<DataVector>& metric_flatness_ext,
    const Scalar<DataVector>& rest_mass_density_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
    const Scalar<DataVector>& pressure_ext,
    const Scalar<DataVector>& lorentz_factor_ext,
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
  // FP exceptions are disabled because the analytic eigenvectors diverge at
  // degeneracies (handled by the degeneracy guard + finiteness mask).
  const ScopedFpeState hllem_fpe_scope(false);

  // averaged primitive state
  Scalar<DataVector> rho_avg{
      0.5 * (get(rest_mass_density_int) + get(rest_mass_density_ext))};
  Scalar<DataVector> eps_avg{0.5 * (get(specific_internal_energy_int) +
                                    get(specific_internal_energy_ext))};
  Scalar<DataVector> p_avg{0.5 * (get(pressure_int) + get(pressure_ext))};
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
  const Scalar<DataVector> enthalpy_avg =
      hydro::relativistic_specific_enthalpy(rho_avg, eps_avg, p_avg);

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
  // over-restoration that blew up with the +/-c bounds. (Elias & Saul,
  // 2026-07-30; cf. PLUTO Src/MHD/GLM/glm.c GLM_Solve.) Everything here stays
  // inside the boundary correction; the evolution system is unchanged.
  const DataVector fast_lambda_max =
      max(0.0, mhd_speeds.get(7));  // v_n + c_fast
  const DataVector fast_lambda_min =
      min(0.0, mhd_speeds.get(1));  // v_n - c_fast
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
  for (const size_t wave : restored_wave_indices(waves_to_restore_)) {
    const DataVector& lam = mhd_speeds.get(wave);
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
        if (std::abs(lam[pt] - mhd_speeds.get(j)[pt]) < degeneracy_tolerance_) {
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
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hllem& lhs, const Hllem& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hllem::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

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
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to HllemWaves. Must be one of Contact, ContactAlfven, "
                     "ContactSlow, All, ContactAlfvenFast, or AllWithFast.");
}
