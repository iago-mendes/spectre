// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Hll::Hll(const double magnetic_field_magnitude_for_hydro,
         const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

Hll::Hll(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hll::get_clone() const {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double Hll::dg_package_data(
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
    const gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,

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
  {
    Scalar<DataVector> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);

    // Initialize characteristic speeds to light speed
    // (default choice)
    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);

    if (const bool has_b_field =
            max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
        not has_b_field and
        (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
      // Since we have no magnetic field (and we have grid points not in the
      // atmosphere), we can reduce the char speeds to the hydro speeds only.
      // This makes the scheme less dissipative.
      //
      // Calculate sound speed squared
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
      auto& lapse_over_one_minus_v2_cs2 =
          get<::Tags::TempScalar<5>>(temp_buffer);
      auto& v_dot_normal_times_one_minus_cs2 =
          get<::Tags::TempScalar<6>>(temp_buffer);

      const Scalar<DataVector> specific_internal_energy =
          equation_of_state
              .specific_internal_energy_from_density_and_temperature(
                  rest_mass_density, temperature, electron_fraction);
      const Scalar<DataVector> pressure =
          equation_of_state.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction);
      const Scalar<DataVector> specific_enthalpy =
          hydro::relativistic_specific_enthalpy(
              rest_mass_density, specific_internal_energy, pressure);
      const Scalar<DataVector> sound_speed_squared{
          clamp(get(equation_of_state
                        .sound_speed_squared_from_density_and_temperature(
                            rest_mass_density, temperature, electron_fraction)),
                0.0, 1.0)};

      // Compute v_dot_normal, v^i n_i
      dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                  normal_covector);

      // Compute v^2=v^i v_i
      dot_product(make_not_null(&v_squared), spatial_velocity,
                  spatial_velocity_one_form);
      get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);

      // Calculate characteristic speeds in inertial frame
      //
      // Ideally we'd use the Lorentz factor instead of 1-v^2, but I (Nils
      // Deppe) don't have the bandwidth to change this right now.
      get(one_minus_v2_cs2) = 1.0 - get(v_squared) * get(sound_speed_squared);
      get(one_minus_cs2) = 1.0 - get(sound_speed_squared);
      discriminant =
          get(sound_speed_squared) * (1.0 - get(v_squared)) *
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

    // Correct for mesh velocity
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  // --- Fast-magnetosonic signal speeds for the MHD part of the system --------
  // In flat space the divergence-cleaning (Phi, B_n) subsystem decouples and
  // travels at the light speed (the Largest speeds above); the MHD variables
  // travel at the slower fast-magnetosonic speed. We package that here so the
  // boundary correction can use the tighter (less dissipative) bounds for the
  // MHD part while keeping +/-c for the scalar/normal-field part. The fast
  // speed uses the standard angle-independent upper bound a_f^2 = c_s^2 + c_A^2
  // (1 - c_s^2) (a safe HLL bound), fed through the same relativistic
  // dispersion as the hydro branch. In curved space (BBH etc.) the flat
  // decomposition does not hold, so the fast bounds fall back to the light
  // speed (no change from the current HLL there).
  {
    const size_t num_points = get(rest_mass_density).size();
    Scalar<DataVector> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    const DataVector cs2 = clamp(
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction)),
        0.0, 1.0);
    DataVector v_dot_normal{num_points, 0.0};
    DataVector v_squared{num_points, 0.0};
    DataVector b_squared{num_points, 0.0};  // comoving b^2 = B^2/W^2 + (B.v)^2
    DataVector b_dot_v{num_points, 0.0};
    DataVector b_norm_sq{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      v_dot_normal += spatial_velocity.get(i) * normal_covector.get(i);
      v_squared += spatial_velocity.get(i) * spatial_velocity_one_form.get(i);
      b_norm_sq += tilde_b.get(i) * tilde_b.get(i);
      b_dot_v += tilde_b.get(i) * spatial_velocity_one_form.get(i);
    }
    v_squared = clamp(v_squared, 0.0, 1.0 - 1.0e-8);
    const DataVector w2 = square(get(lorentz_factor));
    b_squared = b_norm_sq / w2 + square(b_dot_v);
    // relativistic Alfven + fast speeds
    const DataVector rho_h = get(rest_mass_density) * get(specific_enthalpy);
    const DataVector ca2 = b_squared / (b_squared + rho_h + 1.0e-30);
    const DataVector a2 = clamp(cs2 + ca2 * (1.0 - cs2), 0.0, 1.0);
    DataVector discriminant =
        a2 * (1.0 - v_squared) *
        ((1.0 - v_squared * a2) - v_dot_normal * v_dot_normal * (1.0 - a2));
    discriminant = sqrt(max(discriminant, 0.0));
    const DataVector lapse_over = get(lapse) / (1.0 - v_squared * a2);
    get(*packaged_fast_outgoing_char_speed) =
        lapse_over * (v_dot_normal * (1.0 - a2) + discriminant) -
        get(shift_dot_normal);
    get(*packaged_fast_ingoing_char_speed) =
        lapse_over * (v_dot_normal * (1.0 - a2) - discriminant) -
        get(shift_dot_normal);
    // Curved-space / atmosphere fallback: use the light speed (= Largest).
    get(*packaged_metric_flatness) = abs(get(lapse) - 1.0) +
                                     abs(get<0>(shift)) + abs(get<1>(shift)) +
                                     abs(get<2>(shift));
    const DataVector& flatness = get(*packaged_metric_flatness);
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (flatness[pt] > 1.0e-12 or
          get(rest_mass_density)[pt] <= light_speed_density_cutoff_) {
        get(*packaged_fast_outgoing_char_speed)[pt] =
            get(*packaged_largest_outgoing_char_speed)[pt];
        get(*packaged_fast_ingoing_char_speed)[pt] =
            get(*packaged_largest_ingoing_char_speed)[pt];
      }
    }
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_fast_outgoing_char_speed) -= get(*normal_dot_mesh_velocity);
      get(*packaged_fast_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
    }
  }
  *packaged_interface_unit_normal = normal_covector;

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

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

void Hll::dg_boundary_terms(
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
    const dg::Formulation dg_formulation) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  // Light-speed (divergence-cleaning) HLL bounds: used for Phi and the normal
  // magnetic field, which form the decoupled GLM subsystem at +/-c.
  const DataVector lambda_max = max(0., get(largest_outgoing_char_speed_int),
                                    -get(largest_ingoing_char_speed_ext));
  const DataVector lambda_min = min(0., get(largest_ingoing_char_speed_int),
                                    -get(largest_outgoing_char_speed_ext));
  // Fast-magnetosonic HLL bounds: used for the MHD variables (D, Ye, Tau, S and
  // the tangential magnetic field), which travel slower than light. In curved
  // space these were set equal to the light speed in dg_package_data, so the
  // split reduces to the standard HLL flux there.
  const DataVector fast_max = max(0., get(fast_outgoing_char_speed_int),
                                  -get(fast_ingoing_char_speed_ext));
  const DataVector fast_min = min(0., get(fast_ingoing_char_speed_int),
                                  -get(fast_outgoing_char_speed_ext));

  // HLL flux for one conserved component given the bounds l_max, l_min.
  const auto hll = [&weak, &num_points](
                       const DataVector& l_max, const DataVector& l_min,
                       const DataVector& u_int, const DataVector& nf_int,
                       const DataVector& u_ext,
                       const DataVector& nf_ext) -> DataVector {
    DataVector dl = l_max - l_min;
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (dl[pt] < 1.0e-30) {
        dl[pt] = 1.0e-30;
      }
    }
    const DataVector lprod = l_max * l_min;
    if (weak) {
      return (l_max * nf_int + l_min * nf_ext + lprod * (u_ext - u_int)) / dl;
    }
    return (l_min * (nf_int + nf_ext) + lprod * (u_ext - u_int)) / dl;
  };

  // Fluid scalars: pure MHD -> fast bounds.
  get(*boundary_correction_tilde_d) = hll(
      fast_max, fast_min, get(tilde_d_int), get(normal_dot_flux_tilde_d_int),
      get(tilde_d_ext), get(normal_dot_flux_tilde_d_ext));
  get(*boundary_correction_tilde_ye) = hll(
      fast_max, fast_min, get(tilde_ye_int), get(normal_dot_flux_tilde_ye_int),
      get(tilde_ye_ext), get(normal_dot_flux_tilde_ye_ext));
  get(*boundary_correction_tilde_tau) =
      hll(fast_max, fast_min, get(tilde_tau_int),
          get(normal_dot_flux_tilde_tau_int), get(tilde_tau_ext),
          get(normal_dot_flux_tilde_tau_ext));
  // Divergence-cleaning scalar Phi: light speed.
  get(*boundary_correction_tilde_phi) =
      hll(lambda_max, lambda_min, get(tilde_phi_int),
          get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
          get(normal_dot_flux_tilde_phi_ext));
  // Momentum: pure MHD -> fast bounds.
  for (size_t i = 0; i < 3; ++i) {
    boundary_correction_tilde_s->get(i) =
        hll(fast_max, fast_min, tilde_s_int.get(i),
            normal_dot_flux_tilde_s_int.get(i), tilde_s_ext.get(i),
            normal_dot_flux_tilde_s_ext.get(i));
  }
  // Magnetic field: the NORMAL component is part of the GLM subsystem (light
  // speed), the TANGENTIAL component is MHD (fast bounds). Decompose along the
  // interface normal, treat each part with its own bounds, and recombine
  // G(B^i) = G(B_n) n^i + G(B_t^i). (In flat space n is a unit covector and the
  // metric is the identity, so raising/lowering the normal is trivial.)
  {
    const auto& n = interface_unit_normal_int;
    DataVector bn_int{num_points, 0.0};
    DataVector bn_ext{num_points, 0.0};
    DataVector nfbn_int{num_points, 0.0};
    DataVector nfbn_ext{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      bn_int += tilde_b_int.get(i) * n.get(i);
      bn_ext += tilde_b_ext.get(i) * n.get(i);
      nfbn_int += normal_dot_flux_tilde_b_int.get(i) * n.get(i);
      nfbn_ext += normal_dot_flux_tilde_b_ext.get(i) * n.get(i);
    }
    const DataVector g_bn =
        hll(lambda_max, lambda_min, bn_int, nfbn_int, bn_ext, nfbn_ext);
    // The decomposition uses n as both covector and (raised) vector, which is
    // only valid in flat space; where the background is curved fall back to the
    // plain (light-speed) HLL flux for B, which keeps the scheme conservative.
    for (size_t i = 0; i < 3; ++i) {
      const DataVector bt_int = tilde_b_int.get(i) - bn_int * n.get(i);
      const DataVector bt_ext = tilde_b_ext.get(i) - bn_ext * n.get(i);
      const DataVector nfbt_int =
          normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n.get(i);
      const DataVector nfbt_ext =
          normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n.get(i);
      const DataVector g_bt =
          hll(fast_max, fast_min, bt_int, nfbt_int, bt_ext, nfbt_ext);
      const DataVector g_split = g_bn * n.get(i) + g_bt;
      const DataVector g_plain =
          hll(lambda_max, lambda_min, tilde_b_int.get(i),
              normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
              normal_dot_flux_tilde_b_ext.get(i));
      for (size_t pt = 0; pt < num_points; ++pt) {
        boundary_correction_tilde_b->get(i)[pt] =
            (get(metric_flatness_int)[pt] > 1.0e-12 or
             get(metric_flatness_ext)[pt] > 1.0e-12)
                ? g_plain[pt]
                : g_split[pt];
      }
    }
  }
}

bool operator==(const Hll& lhs, const Hll& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hll& lhs, const Hll& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hll::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
