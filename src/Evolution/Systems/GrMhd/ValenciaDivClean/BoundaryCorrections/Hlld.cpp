// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hlld.hpp"

#include "Parallel/Printf/Printf.hpp"

#include <array>
#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/HlldImpl.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

namespace {
// Orthonormal triad (n, t1, t2) from a unit normal, in flat (Euclidean) space.
void make_triad(const std::array<double, 3>& n, std::array<double, 3>* t1,
                std::array<double, 3>* t2) {
  const std::array<double, 3> helper =
      std::abs(n[0]) < 0.9 ? std::array<double, 3>{1.0, 0.0, 0.0}
                           : std::array<double, 3>{0.0, 1.0, 0.0};
  const double hdotn = helper[0] * n[0] + helper[1] * n[1] + helper[2] * n[2];
  std::array<double, 3> a{helper[0] - hdotn * n[0], helper[1] - hdotn * n[1],
                          helper[2] - hdotn * n[2]};
  const double inv = 1.0 / std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
  *t1 = {a[0] * inv, a[1] * inv, a[2] * inv};
  // t2 = n x t1
  *t2 = {n[1] * (*t1)[2] - n[2] * (*t1)[1], n[2] * (*t1)[0] - n[0] * (*t1)[2],
         n[0] * (*t1)[1] - n[1] * (*t1)[0]};
}
}  // namespace

Hlld::Hlld(const double magnetic_field_magnitude_for_hydro,
           const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

Hlld::Hlld(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hlld::get_clone() const {
  return std::make_unique<Hlld>(*this);
}

void Hlld::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double Hlld::dg_package_data(
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
  {
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
      auto& lapse_over_one_minus_v2_cs2 =
          get<::Tags::TempScalar<5>>(temp_buffer);
      auto& v_dot_normal_times_one_minus_cs2 =
          get<::Tags::TempScalar<6>>(temp_buffer);
      const Scalar<DataVector> sound_speed_squared{clamp(
          get(equation_of_state
                  .sound_speed_squared_from_density_and_temperature(
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
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;
  *packaged_interface_unit_normal = normal_covector;
  // Flatness indicator: the five-wave fan below assumes flat space, so measure
  // the metric's departure from Minkowski using the (analytic, un-reconstructed)
  // background lapse and shift -- zero for the Minkowski backgrounds of the
  // relativistic M&M tests, nonzero otherwise (where we fall back to HLL).
  // We deliberately do NOT use sqrt(det gamma)=TildeD/(rho W): at reconstructed
  // FD faces TildeD and rho,W are reconstructed independently so that ratio is
  // not 1 even in flat space. A curved spatial metric with lapse=1, shift=0 is
  // not a target regime and is not separately guarded here.
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

void Hlld::dg_boundary_terms(
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
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  // --- HLL baseline (used for TildeYe, TildePhi, and as a fallback) ---
  //
  // SCALAR / MHD SPLIT (the same one HLL and HLLEM use; Teukolsky MHD Eq. 4.35).
  // In flat space the GLM subsystem decouples: Phi and the NORMAL magnetic field
  // propagate at the light speed, everything else (D, Ye, Tau, S, and the
  // TANGENTIAL magnetic field) at the fast-magnetosonic speed. Using +/-c for
  // the MHD variables makes this baseline far more dissipative than it needs to
  // be, so the HLLD *logic* is applied to the MHD part and the GLM part is
  // carried separately at +/-c.
  //
  // Note this baseline is not merely cosmetic for HLLD: it supplies TildeYe and
  // TildePhi outright, it seeds TildeD/S/B/Tau, and it is what a point falls
  // back to if the fan solve returns something non-finite.
  DataVector lambda_max = max(0.0, get(largest_outgoing_char_speed_int),
                              -get(largest_ingoing_char_speed_ext));
  DataVector lambda_min = min(0.0, get(largest_ingoing_char_speed_int),
                              -get(largest_outgoing_char_speed_ext));

  // Fast-magnetosonic bounds at the ARITHMETIC-AVERAGE interface state, exactly
  // as in HLL/HLLEM (a single interface eigensystem, so the bounds are the same
  // on mirror-image faces). In curved space the flat decomposition does not
  // hold, so we keep the light bounds and the scheme reduces to plain HLL.
  DataVector fast_max = lambda_max;
  DataVector fast_min = lambda_min;
  const bool flat_face = max(get(metric_flatness_int)) <= 1.0e-12 and
                         max(get(metric_flatness_ext)) <= 1.0e-12;
  if (flat_face) {
    const ScopedFpeState fpe(false);
    const Scalar<DataVector> rho_avg{
        0.5 * (get(rest_mass_density_int) + get(rest_mass_density_ext))};
    const Scalar<DataVector> eps_avg{0.5 * (get(specific_internal_energy_int) +
                                            get(specific_internal_energy_ext))};
    const Scalar<DataVector> p_avg{0.5 *
                                   (get(pressure_int) + get(pressure_ext))};
    tnsr::I<DataVector, 3, Frame::Inertial> v_avg{num_points};
    tnsr::I<DataVector, 3, Frame::Inertial> b_avg{num_points};
    for (size_t i = 0; i < 3; ++i) {
      v_avg.get(i) =
          0.5 * (spatial_velocity_int.get(i) + spatial_velocity_ext.get(i));
      b_avg.get(i) = 0.5 * (tilde_b_int.get(i) + tilde_b_ext.get(i));
    }
    DataVector v_sq_avg{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      v_sq_avg += v_avg.get(i) * v_avg.get(i);
    }
    v_sq_avg = clamp(v_sq_avg, 0.0, 1.0 - 1.0e-10);
    const Scalar<DataVector> w_avg{1.0 / sqrt(1.0 - v_sq_avg)};
    const Scalar<DataVector> enthalpy_avg =
        hydro::relativistic_specific_enthalpy(rho_avg, eps_avg, p_avg);
    tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      flat_metric.get(i, i) = 1.0;
    }
    tnsr::i<DataVector, 9> mhd_speeds{num_points, 0.0};
    characteristic_speeds_mhd(make_not_null(&mhd_speeds), v_avg, b_avg, rho_avg,
                              eps_avg, w_avg, enthalpy_avg, flat_metric,
                              interface_unit_normal_int, equation_of_state);
    fast_max = max(0.0, mhd_speeds.get(7));  // v_n + c_fast
    fast_min = min(0.0, mhd_speeds.get(1));  // v_n - c_fast
  }

  // HLL flux for one component given explicit bounds.
  const auto hll_with = [&weak, &num_points](
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
  // MHD variables -> fast bounds; kept as a lambda with the old signature so the
  // call sites below are unchanged.
  auto hll =
      [&](const Scalar<DataVector>& nf_int, const Scalar<DataVector>& u_int,
          const Scalar<DataVector>& nf_ext,
          const Scalar<DataVector>& u_ext) -> DataVector {
    return hll_with(fast_max, fast_min, get(u_int), get(nf_int), get(u_ext),
                    get(nf_ext));
  };
  get(*boundary_correction_tilde_ye) =
      hll(normal_dot_flux_tilde_ye_int, tilde_ye_int,
          normal_dot_flux_tilde_ye_ext, tilde_ye_ext);
  // Divergence-cleaning scalar Phi belongs to the GLM subsystem: light speed.
  get(*boundary_correction_tilde_phi) =
      hll_with(lambda_max, lambda_min, get(tilde_phi_int),
               get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
               get(normal_dot_flux_tilde_phi_ext));
  // HLLD overrides TildeD/S/B/Tau below; seed with HLL so any skipped point is
  // still filled.
  get(*boundary_correction_tilde_d) =
      hll(normal_dot_flux_tilde_d_int, tilde_d_int, normal_dot_flux_tilde_d_ext,
          tilde_d_ext);
  get(*boundary_correction_tilde_tau) =
      hll(normal_dot_flux_tilde_tau_int, tilde_tau_int,
          normal_dot_flux_tilde_tau_ext, tilde_tau_ext);
  // Momentum: pure MHD -> fast bounds.
  for (size_t k = 0; k < 3; ++k) {
    boundary_correction_tilde_s->get(k) =
        hll_with(fast_max, fast_min, tilde_s_int.get(k),
                 normal_dot_flux_tilde_s_int.get(k), tilde_s_ext.get(k),
                 normal_dot_flux_tilde_s_ext.get(k));
  }
  // Magnetic field: the NORMAL component belongs to the GLM subsystem (light
  // speed), the TANGENTIAL component is MHD (fast bounds). Split along the
  // interface normal, use the appropriate bounds for each part, and recombine
  // G(B^i) = G(B_n) n^i + G(B_t^i). The decomposition treats n as both covector
  // and raised vector, which holds only in flat space, so on a curved face we
  // keep the plain light-speed HLL flux (which stays conservative).
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
        hll_with(lambda_max, lambda_min, bn_int, nfbn_int, bn_ext, nfbn_ext);
    for (size_t i = 0; i < 3; ++i) {
      const DataVector bt_int = tilde_b_int.get(i) - bn_int * n.get(i);
      const DataVector bt_ext = tilde_b_ext.get(i) - bn_ext * n.get(i);
      const DataVector nfbt_int =
          normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n.get(i);
      const DataVector nfbt_ext =
          normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n.get(i);
      const DataVector g_bt =
          hll_with(fast_max, fast_min, bt_int, nfbt_int, bt_ext, nfbt_ext);
      const DataVector g_split = g_bn * n.get(i) + g_bt;
      const DataVector g_plain = hll_with(
          lambda_max, lambda_min, tilde_b_int.get(i),
          normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
          normal_dot_flux_tilde_b_ext.get(i));
      for (size_t pt = 0; pt < num_points; ++pt) {
        boundary_correction_tilde_b->get(i)[pt] =
            flat_face ? g_split[pt] : g_plain[pt];
      }
    }
  }

  // Periodic diagnostic report. Ratios (not absolute counts) are the signal:
  // if gate_rejected / fan_attempts is ~0 then the admissibility gate is NOT
  // responsible for the high-order oscillation excess, and likewise for the
  // root-find. Printed on a geometric schedule to keep the log small.
  {
    auto& diag = hlld_detail::diagnostics();
    static size_t next_report = 100000;
    if (diag.fan_attempts > next_report) {
      Parallel::printf(
          "HLLD diag: attempts=%zu rootfind_failed=%zu (%.4f%%) "
          "gate_rejected=%zu (%.4f%%) uniform=%zu\n",
          diag.fan_attempts, diag.rootfind_failed,
          100.0 * static_cast<double>(diag.rootfind_failed) /
              static_cast<double>(std::max<size_t>(diag.fan_attempts, 1)),
          diag.gate_rejected,
          100.0 * static_cast<double>(diag.gate_rejected) /
              static_cast<double>(std::max<size_t>(diag.fan_attempts, 1)),
          diag.uniform_shortcut);
      next_report *= 4;
    }
  }

  // --- pointwise HLLD for TildeD, TildeS, TildeB, TildeTau ---
  // The five-wave solver internally produces inf/nan for unphysical trial states
  // (handled by its own masks + the isfinite fallback below), so disable FP
  // exception trapping for the duration of the fan reconstruction.
  const ScopedFpeState hlld_fpe_scope(false);
  namespace hd = hlld_detail;
  for (size_t i = 0; i < num_points; ++i) {
    if (get(metric_flatness_int)[i] > 1.0e-12 or
        get(metric_flatness_ext)[i] > 1.0e-12) {
      continue;  // curved metric: keep the (conservative) HLL flux
    }
    const double rho_i = get(rest_mass_density_int)[i];
    const double eps_i = get(specific_internal_energy_int)[i];
    const double p_i = get(pressure_int)[i];
    if (rho_i <= 0.0 or eps_i <= 0.0 or p_i <= 0.0) {
      continue;  // keep HLL where the ideal-gas relation is undefined
    }
    // ideal-gas index from the interior primitives (constant for an ideal EoS)
    const double gamma = 1.0 + p_i / (rho_i * eps_i);

    const std::array<double, 3> n{get<0>(interface_unit_normal_int)[i],
                                  get<1>(interface_unit_normal_int)[i],
                                  get<2>(interface_unit_normal_int)[i]};
    std::array<double, 3> t1{}, t2{};
    make_triad(n, &t1, &t2);
    auto proj = [&](const tnsr::I<DataVector, 3, Frame::Inertial>& v,
                    size_t comp) {
      const std::array<double, 3> vv{v.get(0)[i], v.get(1)[i], v.get(2)[i]};
      const std::array<double, 3>& e = comp == 0 ? n : (comp == 1 ? t1 : t2);
      return vv[0] * e[0] + vv[1] * e[1] + vv[2] * e[2];
    };
    auto build = [&](const Scalar<DataVector>& rho,
                     const Scalar<DataVector>& eps,
                     const tnsr::I<DataVector, 3, Frame::Inertial>& vel,
                     const tnsr::I<DataVector, 3, Frame::Inertial>& bfield,
                     const Scalar<DataVector>& w) {
      const double lorentz = get(w)[i];
      return std::array<double, 9>{get(rho)[i],
                                   get(eps)[i],
                                   lorentz * proj(vel, 0),
                                   lorentz * proj(vel, 1),
                                   lorentz * proj(vel, 2),
                                   proj(bfield, 0),
                                   proj(bfield, 1),
                                   proj(bfield, 2),
                                   0.0};
    };
    const auto p_int_arr =
        build(rest_mass_density_int, specific_internal_energy_int,
              spatial_velocity_int, tilde_b_int, lorentz_factor_int);
    const auto p_ext_arr =
        build(rest_mass_density_ext, specific_internal_energy_ext,
              spatial_velocity_ext, tilde_b_ext, lorentz_factor_ext);
    const auto [flux, cons] =
        hd::HLLDSolver<0>(p_int_arr, p_ext_arr, gamma).solve(0.0);
    bool all_finite = true;
    for (size_t k = 0; k < hd::NUM; ++k) {
      all_finite = all_finite and std::isfinite(flux[k]);
    }
    if (not all_finite) {
      continue;  // keep HLL if the fan solve produced a non-finite result
    }
    // numerical flux G = n . F_HLLD, rotated back to the inertial frame
    const double g_d = flux[hd::DENS];
    const double g_tau = flux[hd::TAUE];
    std::array<double, 3> g_s{}, g_b{};
    for (size_t k = 0; k < 3; ++k) {
      g_s[k] =
          flux[hd::SCX] * n[k] + flux[hd::SCY] * t1[k] + flux[hd::SCZ] * t2[k];
      g_b[k] =
          flux[hd::BBX] * n[k] + flux[hd::BBY] * t1[k] + flux[hd::BBZ] * t2[k];
    }
    // strong form returns G - F_int; weak form returns G
    const double sub_d = weak ? 0.0 : get(normal_dot_flux_tilde_d_int)[i];
    const double sub_tau = weak ? 0.0 : get(normal_dot_flux_tilde_tau_int)[i];
    get(*boundary_correction_tilde_d)[i] = g_d - sub_d;
    get(*boundary_correction_tilde_tau)[i] = g_tau - sub_tau;
    for (size_t k = 0; k < 3; ++k) {
      boundary_correction_tilde_s->get(k)[i] =
          g_s[k] - (weak ? 0.0 : normal_dot_flux_tilde_s_int.get(k)[i]);
      boundary_correction_tilde_b->get(k)[i] =
          g_b[k] - (weak ? 0.0 : normal_dot_flux_tilde_b_int.get(k)[i]);
    }
  }
}

bool operator==(const Hlld& lhs, const Hlld& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hlld& lhs, const Hlld& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hlld::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
