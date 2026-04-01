// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::test_detail {

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro_unoptimized(
    const gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  if ((*char_speeds)[0].size() != num_grid_points) {
    for (auto& cs : (*char_speeds)) {
      cs = DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      temp_tensors{num_grid_points};

  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);

  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }

  // Calculate the characteristic speed for non-degenerate ones
  Scalar<DataVector>& denom = get<::Tags::TempScalar<1>>(temp_tensors);
  get(denom) = 1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<2>>(temp_tensors);
  get(first_term) =
      (1.0 - get(sound_speed_squared)) * get(normal_velocity) / get(denom);

  Scalar<DataVector>& second_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(second_term) =
      sqrt(get(sound_speed_squared)) *
      sqrt(get(denom) - get(normal_velocity) * get(normal_velocity) *
                            (1 - get(sound_speed_squared))) /
      (get(lorentz_factor) * get(denom));

  // Degenerate eigenvalue (normal dot velocity)
  (*char_speeds)[HydroSpeed::NormalDotVelocity] = get(normal_velocity);
  (*char_speeds)[HydroSpeed::LambdaPlus] = get(first_term) + get(second_term);
  (*char_speeds)[HydroSpeed::LambdaMinus] = get(first_term) - get(second_term);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 3> characteristic_speeds_hydro_unoptimized(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::array<DataVector, 3> char_speeds{};
  characteristic_speeds_hydro_unoptimized(
      make_not_null(&char_speeds), spatial_velocity, rest_mass_density,
      specific_internal_energy, specific_enthalpy, electron_fraction,
      lorentz_factor, unit_normal, spatial_metric, equation_of_state);
  return char_speeds;
}

template <size_t ThermodynamicDim>
void eigenvectors_hydro_unoptimized(
    const gsl::not_null<std::array<tnsr::i<DataVector, 6, Frame::Inertial>,
                                   6>*>& right_eigenvectors,
    const gsl::not_null<std::array<tnsr::I<DataVector, 6, Frame::Inertial>,
                                   6>*>& left_eigenvectors,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();

  auto allocate_and_zero = [num_grid_points](auto& vec_array) {
    if (vec_array[0].get(0).size() != num_grid_points) {
      for (auto& vec : vec_array) {
        for (size_t a = 0; a < 6; ++a) {
          vec.get(a) = DataVector(num_grid_points, 0.0);
        }
      }
    } else {
      for (auto& vec : vec_array) {
        for (size_t a = 0; a < 6; ++a) {
          vec.get(a) = 0.0;
        }
      }
    }
  };
  allocate_and_zero(*right_eigenvectors);
  allocate_and_zero(*left_eigenvectors);

  Scalar<DataVector> det_spatial_metric{num_grid_points};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{num_grid_points};
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inv_spatial_metric), spatial_metric);

  tnsr::i<DataVector, 3, Frame::Inertial> tangent_one_form_1{num_grid_points};
  orthonormal_oneform(make_not_null(&tangent_one_form_1), unit_normal,
                      inv_spatial_metric);

  tnsr::i<DataVector, 3, Frame::Inertial> tangent_one_form_2{num_grid_points};
  orthonormal_oneform(make_not_null(&tangent_one_form_2), unit_normal,
                      tangent_one_form_1, spatial_metric, det_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> unit_normal_vector{num_grid_points};
  raise_or_lower_index(make_not_null(&unit_normal_vector), unit_normal,
                       inv_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> tangent_vector_1{num_grid_points};
  raise_or_lower_index(make_not_null(&tangent_vector_1), tangent_one_form_1,
                       inv_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> tangent_vector_2{num_grid_points};
  raise_or_lower_index(make_not_null(&tangent_vector_2), tangent_one_form_2,
                       inv_spatial_metric);

  Scalar<DataVector> v_dot_tangent_1{num_grid_points};
  dot_product(make_not_null(&v_dot_tangent_1), tangent_one_form_1,
              spatial_velocity);

  Scalar<DataVector> v_dot_tangent_2{num_grid_points};
  dot_product(make_not_null(&v_dot_tangent_2), tangent_one_form_2,
              spatial_velocity);

  Scalar<DataVector> normal_velocity{num_grid_points};
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  const DataVector one_minus_normal_velocity_squared =
      1.0 - get(normal_velocity) * get(normal_velocity);

  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{
      num_grid_points};
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);

  Scalar<DataVector> spatial_velocity_squared{num_grid_points};
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_one_form);

  Scalar<DataVector> sound_speed_squared{num_grid_points};
  Scalar<DataVector> kappa{num_grid_points};
  Scalar<DataVector> zeta{num_grid_points};
  Scalar<DataVector> pressure{num_grid_points};

  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(equation_of_state
                 .kappa_times_p_over_rho_squared_from_density_and_energy(
                     rest_mass_density, specific_internal_energy))) /
        get(specific_enthalpy);
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
    // So, we're currently using the equations from an ideal fluid EoS to set
    // kappa, assuming the same adiabatic index as used in the tests. This
    // approach will need to be improved during the code review process..
    const double adiabatic_index = 1.5;
    const Scalar<DataVector> chi =
        tenex::evaluate(specific_internal_energy() * (adiabatic_index - 1.0));
    const Scalar<DataVector> kappa_times_p_over_rho_squared = tenex::evaluate(
        square(adiabatic_index - 1.0) * specific_internal_energy());
    const DataVector sound_speed_squared_ideal_fluid =
        (get(chi) + get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    ASSERT(max(abs(get(sound_speed_squared) -
                   sound_speed_squared_ideal_fluid)) < 1e-10,
           "The ideal fluid approximation for kappa is not valid.");
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  }

  // This is for the case for zeta = 0.
  const double zeta_max_abs = max(abs(get(zeta)));

  const DataVector sound_speed = sqrt(get(sound_speed_squared));
  const DataVector W_squared = square(get(lorentz_factor));

  // Variables in the ordering (D, S_i, tau, DYe)
  // RIGHT eigenvectors

  // R1 and R2
  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[R1].get(i + 1) =
        get(specific_enthalpy) *
        (tangent_one_form_1.get(i) + 2.0 * W_squared * get(v_dot_tangent_1) *
                                         spatial_velocity_one_form.get(i));

    (*right_eigenvectors)[R2].get(i + 1) =
        get(specific_enthalpy) *
        (tangent_one_form_2.get(i) + 2.0 * W_squared * get(v_dot_tangent_2) *
                                         spatial_velocity_one_form.get(i));
  }

  (*right_eigenvectors)[R1].get(0) = get(lorentz_factor) * get(v_dot_tangent_1);
  (*right_eigenvectors)[R2].get(0) = get(lorentz_factor) * get(v_dot_tangent_2);

  (*right_eigenvectors)[R1].get(4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_1);

  (*right_eigenvectors)[R2].get(4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_2);

  (*right_eigenvectors)[R1].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R1].get(0);
  (*right_eigenvectors)[R2].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R2].get(0);

  // R3
  const DataVector common_R3 =
      get(specific_enthalpy) * get(lorentz_factor) *
      (get(kappa) - get(rest_mass_density) * get(sound_speed_squared));
  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[R3].get(i + 1) =
        common_R3 * spatial_velocity_one_form.get(i);
  }
  (*right_eigenvectors)[R3].get(0) = get(kappa);
  (*right_eigenvectors)[R3].get(4) = common_R3 - get(kappa);
  (*right_eigenvectors)[R3].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R3].get(0);

  // R4
  if (zeta_max_abs < 1e-14) {
    (*right_eigenvectors)[R4].get(5) = 1.0;
  } else {
    for (size_t i = 0; i < 3; ++i) {
      (*right_eigenvectors)[R4].get(i + 1) = spatial_velocity_one_form.get(i);
    }
    (*right_eigenvectors)[R4].get(4) = 1.0;
    (*right_eigenvectors)[R4].get(5) =
        -get(kappa) / (get(zeta) * get(lorentz_factor));
  }

  // R±
  const DataVector denom =
      get(lorentz_factor) *
      sqrt(1.0 - get(spatial_velocity_squared) * get(sound_speed_squared) -
           get(normal_velocity) * get(normal_velocity) *
               (1.0 - get(sound_speed_squared)));

  const DataVector sound_speed_over_denom = sound_speed / denom;

  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[Rplus].get(i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) +
         sound_speed_over_denom * unit_normal.get(i));

    (*right_eigenvectors)[Rminus].get(i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) -
         sound_speed_over_denom * unit_normal.get(i));
  }

  (*right_eigenvectors)[Rplus].get(0) = 1.0;
  (*right_eigenvectors)[Rminus].get(0) = 1.0;

  (*right_eigenvectors)[Rplus].get(4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 + sound_speed * get(normal_velocity) / denom) -
      1.0;

  (*right_eigenvectors)[Rminus].get(4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 - sound_speed * get(normal_velocity) / denom) -
      1.0;

  (*right_eigenvectors)[Rplus].get(5) = get(electron_fraction);
  (*right_eigenvectors)[Rminus].get(5) = get(electron_fraction);

  // LEFT eigenvectors
  const DataVector prefactor_L12 =
      1.0 / (get(specific_enthalpy) * one_minus_normal_velocity_squared);

  // L1
  (*left_eigenvectors)[L1].get(0) = -get(v_dot_tangent_1) * prefactor_L12;
  (*left_eigenvectors)[L1].get(4) = -get(v_dot_tangent_1) * prefactor_L12;
  for (size_t i = 0; i < 3; ++i) {
    (*left_eigenvectors)[L1].get(i + 1) =
        (get(v_dot_tangent_1) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         one_minus_normal_velocity_squared * tangent_vector_1.get(i)) *
        prefactor_L12;
  }

  // L2
  (*left_eigenvectors)[L2].get(0) = -get(v_dot_tangent_2) * prefactor_L12;
  (*left_eigenvectors)[L2].get(4) = -get(v_dot_tangent_2) * prefactor_L12;
  for (size_t i = 0; i < 3; ++i) {
    (*left_eigenvectors)[L2].get(i + 1) =
        (get(v_dot_tangent_2) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         one_minus_normal_velocity_squared * tangent_vector_2.get(i)) *
        prefactor_L12;
  }

  // L3
  {
    const DataVector prefactor_L3 =
        1.0 / (get(rest_mass_density) * get(specific_enthalpy) *
               get(sound_speed_squared));

    const DataVector h_minus_one =
        get(specific_internal_energy) + get(pressure) / get(rest_mass_density);

    const DataVector W_minus_one = get(spatial_velocity_squared) *
                                   square(get(lorentz_factor)) /
                                   (get(lorentz_factor) + 1.0);

    const DataVector h_minus_W = h_minus_one - W_minus_one;

    (*left_eigenvectors)[L3].get(0) =
        (h_minus_W + get(zeta) * get(electron_fraction) / get(kappa)) *
        prefactor_L3;

    for (size_t i = 0; i < 3; ++i) {
      (*left_eigenvectors)[L3].get(i + 1) =
          (get(lorentz_factor) * spatial_velocity.get(i)) * prefactor_L3;
    }

    (*left_eigenvectors)[L3].get(4) = (-get(lorentz_factor)) * prefactor_L3;
    (*left_eigenvectors)[L3].get(5) = (-get(zeta) / get(kappa)) * prefactor_L3;
  }

  // L4
  {
    if (zeta_max_abs < 1e-14) {
      (*left_eigenvectors)[L4].get(0) = -get(electron_fraction);
      (*left_eigenvectors)[L4].get(5) = 1.0;
    } else {
      const DataVector prefactor_L4 =
          get(zeta) * get(lorentz_factor) / get(kappa);
      (*left_eigenvectors)[L4].get(0) = prefactor_L4 * get(electron_fraction);
      (*left_eigenvectors)[L4].get(5) = -prefactor_L4;
    }
  }

  // L±
  {
    const DataVector a =
        square(get(lorentz_factor)) * one_minus_normal_velocity_squared *
        (get(kappa) + get(rest_mass_density) * get(sound_speed_squared));

    const DataVector c_plus = get(rest_mass_density) * sound_speed *
                              (sound_speed + get(normal_velocity) * denom);
    const DataVector c_minus = get(rest_mass_density) * sound_speed *
                               (sound_speed - get(normal_velocity) * denom);

    const DataVector b_plus = a - c_plus;
    const DataVector b_minus = a - c_minus;

    const DataVector k_term =
        get(kappa) - get(rest_mass_density) * get(sound_speed_squared) +
        get(zeta) * get(electron_fraction) / get(specific_enthalpy);

    const DataVector prefactor_Lpm =
        1.0 / (2.0 * get(rest_mass_density) * get(specific_enthalpy) *
               get(lorentz_factor) * get(sound_speed_squared) *
               one_minus_normal_velocity_squared);

    // S_i
    for (size_t i = 0; i < 3; ++i) {
      (*left_eigenvectors)[Lplus].get(i + 1) =
          (-a * spatial_velocity.get(i) +
           get(rest_mass_density) * sound_speed *
               (sound_speed * get(normal_velocity) + denom) *
               unit_normal_vector.get(i)) *
          prefactor_Lpm;

      (*left_eigenvectors)[Lminus].get(i + 1) =
          (-a * spatial_velocity.get(i) +
           get(rest_mass_density) * sound_speed *
               (sound_speed * get(normal_velocity) - denom) *
               unit_normal_vector.get(i)) *
          prefactor_Lpm;
    }

    // D
    (*left_eigenvectors)[Lplus].get(0) =
        (b_plus - get(specific_enthalpy) * get(lorentz_factor) * k_term *
                      one_minus_normal_velocity_squared) *
        prefactor_Lpm;

    (*left_eigenvectors)[Lminus].get(0) =
        (b_minus - get(specific_enthalpy) * get(lorentz_factor) * k_term *
                       one_minus_normal_velocity_squared) *
        prefactor_Lpm;

    // tau
    (*left_eigenvectors)[Lplus].get(4) = b_plus * prefactor_Lpm;
    (*left_eigenvectors)[Lminus].get(4) = b_minus * prefactor_Lpm;

    // DYe
    (*left_eigenvectors)[Lplus].get(5) =
        (get(zeta) * get(lorentz_factor) * one_minus_normal_velocity_squared) *
        prefactor_Lpm;
    (*left_eigenvectors)[Lminus].get(5) =
        (get(zeta) * get(lorentz_factor) * one_minus_normal_velocity_squared) *
        prefactor_Lpm;
  }
}

template <size_t ThermodynamicDim>
void flux_jacobian_hydro_unoptimized(
    gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(equation_of_state
                 .kappa_times_p_over_rho_squared_from_density_and_energy(
                     rest_mass_density, specific_internal_energy))) /
        get(specific_enthalpy);
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    const Scalar<DataVector> pressure =
        equation_of_state.pressure_from_density_and_energy(
            rest_mass_density, specific_internal_energy);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    // The following computation works for a general 3D EoS, but it doesn't
    // allow getting kappa.
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    // So, we're currently using the equations from an ideal fluid EoS to set
    // kappa, assuming the same adiabatic index as used in the tests. This
    // approach will need to be improved during the code review process..
    const double adiabatic_index = 1.5;
    const Scalar<DataVector> chi =
        tenex::evaluate(specific_internal_energy() * (adiabatic_index - 1.0));
    const Scalar<DataVector> kappa_times_p_over_rho_squared = tenex::evaluate(
        square(adiabatic_index - 1.0) * specific_internal_energy());
    const DataVector sound_speed_squared_ideal_fluid =
        (get(chi) + get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    ASSERT(max(abs(get(sound_speed_squared) -
                   sound_speed_squared_ideal_fluid)) < 1e-10,
           "The ideal fluid approximation for kappa is not valid.");
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    // For now, we assume that we are at compositional equilibrium, so we set
    // zeta to zero.
    get(zeta) = 0.0;
  }

  // Intermediate variables
  const auto Z = tenex::evaluate(rest_mass_density() * specific_enthalpy() *
                                 square(lorentz_factor()));
  const auto D = tenex::evaluate(rest_mass_density() * lorentz_factor());
  const auto normal_velocity =
      tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
  const auto unit_vector = tenex::evaluate<ti::I>(
      inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  const auto spatial_velocity_one_form = tenex::evaluate<ti::i>(
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));
  const auto mixed_spatial_metric = tenex::evaluate<ti::I, ti::j>(
      inv_spatial_metric(ti::I, ti::K) * spatial_metric(ti::k, ti::j));

  // Derivatives of Z
  const auto dzdD = tenex::evaluate(
      -((lorentz_factor() *
         (kappa() * (-specific_enthalpy() + lorentz_factor()) -
          zeta() * electron_fraction() +
          (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
              rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  const auto dzds = tenex::evaluate<ti::I>(
      (spatial_velocity(ti::I) * square(lorentz_factor()) *
       (kappa() + sound_speed_squared() * rest_mass_density())) /
      ((-square(lorentz_factor()) +
        sound_speed_squared() * (-1. + square(lorentz_factor()))) *
       rest_mass_density()));
  const auto dzdtau = tenex::evaluate(
      -((square(lorentz_factor()) * (kappa() + rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  const auto dzdye = tenex::evaluate(
      (zeta() * lorentz_factor()) /
      ((square(lorentz_factor()) -
        sound_speed_squared() * (-1. + square(lorentz_factor()))) *
       rest_mass_density()));

  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      ((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z);
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(0, B + 1) =
        (get(D) * (unit_vector.get(B) - dzds.get(B) * get(normal_velocity))) /
        get(Z);
  }
  characteristic_matrix->get(0, 4) =
      -((get(D) * get(dzdtau) * get(normal_velocity)) / get(Z));
  characteristic_matrix->get(0, 5) =
      -((get(D) * get(dzdye) * get(normal_velocity)) / get(Z));
  for (size_t c = 0; c < 3; ++c) {
    characteristic_matrix->get(c + 1, 0) =
        (-1.0 + get(dzdD)) * unit_normal.get(c) -
        get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(c);
    for (size_t B = 0; B < 3; ++B) {
      characteristic_matrix->get(c + 1, B + 1) =
          mixed_spatial_metric.get(B, c) * get(normal_velocity) +
          unit_vector.get(B) * spatial_velocity_one_form.get(c) +
          dzds.get(B) *
              (unit_normal.get(c) -
               get(normal_velocity) * spatial_velocity_one_form.get(c));
    }
    characteristic_matrix->get(c + 1, 4) =
        (-1.0 + get(dzdtau)) * unit_normal.get(c) -
        get(dzdtau) * get(normal_velocity) * spatial_velocity_one_form.get(c);
    characteristic_matrix->get(c + 1, 5) =
        get(dzdye) * (unit_normal.get(c) -
                      get(normal_velocity) * spatial_velocity_one_form.get(c));
  }
  characteristic_matrix->get(4, 0) =
      -(((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z));
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(4, B + 1) =
        ((get(Z) - get(D)) * unit_vector.get(B) +
         get(D) * dzds.get(B) * get(normal_velocity)) /
        get(Z);
  }
  characteristic_matrix->get(4, 4) =
      (get(D) * get(dzdtau) * get(normal_velocity)) / get(Z);
  characteristic_matrix->get(4, 5) =
      (get(D) * get(dzdye) * get(normal_velocity)) / get(Z);
  characteristic_matrix->get(5, 0) =
      -((get(D) * get(electron_fraction) * get(dzdD) * get(normal_velocity)) /
        get(Z));
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(5, B + 1) =
        (get(D) * get(electron_fraction) *
         (unit_vector.get(B) - dzds.get(B) * get(normal_velocity))) /
        get(Z);
  }
  characteristic_matrix->get(5, 4) =
      -((get(D) * get(electron_fraction) * get(dzdtau) * get(normal_velocity)) /
        get(Z));
  characteristic_matrix->get(5, 5) =
      ((get(Z) - get(D) * get(electron_fraction) * get(dzdye)) *
       get(normal_velocity)) /
      get(Z);
}

}  // namespace grmhd::ValenciaDivClean::test_detail

namespace {

using HydroSpeed = grmhd::ValenciaDivClean::HydroSpeed;
using HydroVectorR = grmhd::ValenciaDivClean::HydroVectorR;
using HydroVectorL = grmhd::ValenciaDivClean::HydroVectorL;

void test_characteristic_speeds(const DataVector& /*used_for_size*/) {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  // const double max_value = 1.0 / sqrt(3);
  // pypp::check_with_random_values<7>(
  //     &grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd<3>,
  //     "CharacteristicSpeeds", "CharacteristicSpeeds",
  //     {{{0.0, 1.0},
  //       {-1.0, 1.0},
  //       {-max_value, max_value},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {-max_value, max_value}}},
  //     used_for_size);
}

void test_with_normal_along_coordinate_axes(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density(rest_mass_density);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy,
      eos.pressure_from_density(rest_mass_density));

  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto lapse = gr_helper::random_lapse(nn_gen, used_for_size);
  const auto shift = gr_helper::random_shift<3>(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);

  const auto magnetic_field = helper::random_magnetic_field(
      nn_gen, eos.pressure_from_density(rest_mass_density), spatial_metric);
  const auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(spatial_velocity, magnetic_field, spatial_metric);
  const DataVector comoving_magnetic_field_squared =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  const Scalar<DataVector> alfven_speed_squared{
      comoving_magnetic_field_squared /
      (comoving_magnetic_field_squared +
       get(rest_mass_density) * get(specific_enthalpy))};
  const Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(specific_enthalpy)};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    const Approx custom_approx = Approx::custom().epsilon(1.0e-10);
    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd(
            rest_mass_density, electron_fraction, specific_internal_energy,
            specific_enthalpy, spatial_velocity, lorentz_factor, magnetic_field,
            lapse, shift, spatial_metric, normal, eos_base),
        (pypp::call<std::array<DataVector, 9>>(
            "CharacteristicSpeeds", "CharacteristicSpeeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)),
        custom_approx);
  }
}

void test_hydro_analytic_eigenvectors(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Define equation of state
  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  // Compute derived quantities
  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  // Get kappa (as in your original test) and set zeta = 0 for now
  const Scalar<DataVector> kappa_times_p_over_rho_squared =
      base_eos.kappa_times_p_over_rho_squared_from_density_and_energy(
          rest_mass_density, specific_internal_energy);

  Scalar<DataVector> kappa{};
  get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
               square(get(rest_mass_density));

  const Scalar<DataVector> zeta{DataVector(used_for_size.size(), 0.0)};

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    // Analytic eigenvectors (RIGHT + LEFT)
    constexpr size_t matrix_size = 6;
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        right_eigenvectors{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        left_eigenvectors{};
    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&right_eigenvectors), make_not_null(&left_eigenvectors),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
        spatial_metric, *eos_3d);

    // Analytic characteristic speeds
    const std::array<DataVector, 3> analytic_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d);

    // Assemble eigenvalues in YOUR eigenvector ordering: (degenerate x4, +, -)
    std::array<Scalar<DataVector>, matrix_size> all_eigenvalues;
    const size_t num_points = used_for_size.size();
    for (size_t i = 0; i < matrix_size; ++i) {
      get(gsl::at(all_eigenvalues, i)).destructive_resize(num_points);
    }
    for (size_t i = 0; i < 4; ++i) {
      get(gsl::at(all_eigenvalues, i)) =
          analytic_speeds[HydroSpeed::NormalDotVelocity];
    }
    get(gsl::at(all_eigenvalues, 4)) = analytic_speeds[HydroSpeed::LambdaPlus];
    get(gsl::at(all_eigenvalues, 5)) = analytic_speeds[HydroSpeed::LambdaMinus];

    // Get characteristic matrix to check eigensystem relations
    tnsr::iJ<DataVector, 6> characteristic_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *eos_3d);

    constexpr double tolerance = 1e-10;

    // Check scaled eigensystem relation for each eigenvalue/eigenvector
    for (size_t i = 0; i < matrix_size; ++i) {
      const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);

      const tnsr::i<DataVector, 6>& right_eigenvector =
          gsl::at(right_eigenvectors, i);
      const Scalar<DataVector> right_residual =
          magnitude(tenex::evaluate<ti::i>(
              characteristic_matrix(ti::i, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::i)));
      const Scalar<DataVector> right_norm = magnitude(right_eigenvector);

      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(left_eigenvectors, i);
      const Scalar<DataVector> left_residual = magnitude(tenex::evaluate<ti::I>(
          left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::I) -
          eigenvalue() * left_eigenvector(ti::I)));
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);

      double max_scaled_error = 0.0;
      for (size_t point = 0; point < used_for_size.size(); ++point) {
        const double r_scale = std::max(1.0, std::abs(get(right_norm)[point]));
        const double l_scale = std::max(1.0, std::abs(get(left_norm)[point]));
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(right_residual)[point]) / r_scale);
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(left_residual)[point]) / l_scale);
      }

      CHECK(max_scaled_error < tolerance);
    }

    // Orthonormality check
    for (size_t i = 0; i < matrix_size; ++i) {
      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(left_eigenvectors, i);
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);

      for (size_t j = 0; j < matrix_size; ++j) {
        const tnsr::i<DataVector, 6>& right_eigenvector =
            gsl::at(right_eigenvectors, j);
        const Scalar<DataVector> right_norm = magnitude(right_eigenvector);

        const Scalar<DataVector> dot_ij =
            tenex::evaluate(left_eigenvector(ti::J) * right_eigenvector(ti::j));

        const double target = (i == j ? 1.0 : 0.0);

        double max_scaled_error = 0.0;
        for (size_t point = 0; point < used_for_size.size(); ++point) {
          const double err = std::abs(get(dot_ij)[point] - target);
          const double scale =
              std::max(1.0, std::abs(get(left_norm)[point]) *
                                std::abs(get(right_norm)[point]));
          max_scaled_error = std::max(max_scaled_error, err / scale);
        }

        CHECK(max_scaled_error < tolerance);
      }
    }
  }
}

void test_hydro_characteristic_speed(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);

  const EquationsOfState::IdealFluid<true> base_eos(4.0 / 3.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  const auto temperature = eos_3d->temperature_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);

  const auto sound_speed_squared =
      eos_3d->sound_speed_squared_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy,
      eos_3d->pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy, electron_fraction));

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const Approx custom_approx = Approx::custom().epsilon(1.0e-10);

    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d),
        (pypp::call<std::array<DataVector, 3>>(
            "CharacteristicSpeeds", "characteristic_speeds_hydro",
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            lorentz_factor, unit_normal)),
        custom_approx);
  }
}

/**
 * Tests that the characteristics remain the same for an opposite unit normal
 * as long as we (1) flip the sign of the eigenvalues and (2) swap the outgoing
 * and ingoing acoustic waves (R+ and R- / L+ and L-).
 */
void test_hydro_characteristics_symmetry(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto equation_of_state_2d =
      EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();

  const auto pressure = equation_of_state_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  constexpr double tolerance = 1.0e-12;
  const Approx custom_approx = Approx::custom().epsilon(tolerance);

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);
    auto unit_normal_opposite = unit_normal;
    for (size_t i = 0; i < 3; ++i) {
      unit_normal_opposite.get(i) *= -1.0;
    }

    const std::array<DataVector, 3> eigenvalues =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *equation_of_state_3d);
    const std::array<DataVector, 3> eigenvalues_opposite =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor,
            unit_normal_opposite, spatial_metric, *equation_of_state_3d);

    CHECK_ITERABLE_CUSTOM_APPROX(
        eigenvalues[HydroSpeed::NormalDotVelocity],
        -eigenvalues_opposite[HydroSpeed::NormalDotVelocity], custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(eigenvalues[HydroSpeed::LambdaPlus],
                                 -eigenvalues_opposite[HydroSpeed::LambdaMinus],
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(eigenvalues[HydroSpeed::LambdaMinus],
                                 -eigenvalues_opposite[HydroSpeed::LambdaPlus],
                                 custom_approx);

    constexpr size_t matrix_size = 6;
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        right_eigenvectors{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        left_eigenvectors{};
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        right_eigenvectors_opposite{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        left_eigenvectors_opposite{};

    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&right_eigenvectors), make_not_null(&left_eigenvectors),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
        spatial_metric, *equation_of_state_3d);
    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&right_eigenvectors_opposite),
        make_not_null(&left_eigenvectors_opposite), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal_opposite, spatial_metric,
        *equation_of_state_3d);

    auto check_swap = [&](const auto& vec_plus, const auto& vec_minus) {
      double max_error = 0.0;
      for (size_t component = 0; component < matrix_size; ++component) {
        const DataVector diff =
            vec_plus.get(component) - vec_minus.get(component);
        const DataVector sum =
            vec_plus.get(component) + vec_minus.get(component);
        for (size_t point = 0; point < used_for_size.size(); ++point) {
          const double point_error =
              std::min(std::abs(diff[point]), std::abs(sum[point]));
          max_error = std::max(max_error, point_error);
        }
      }
      CHECK(max_error < tolerance);
    };

    check_swap(gsl::at(right_eigenvectors, HydroVectorR::Rplus),
               gsl::at(right_eigenvectors_opposite, HydroVectorR::Rminus));
    check_swap(gsl::at(right_eigenvectors, HydroVectorR::Rminus),
               gsl::at(right_eigenvectors_opposite, HydroVectorR::Rplus));
    check_swap(gsl::at(left_eigenvectors, HydroVectorL::Lplus),
               gsl::at(left_eigenvectors_opposite, HydroVectorL::Lminus));
    check_swap(gsl::at(left_eigenvectors, HydroVectorL::Lminus),
               gsl::at(left_eigenvectors_opposite, HydroVectorL::Lplus));
  }
}

void test_hydro_eigenvectors_identity(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Define equation of state
  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  // Compute derived quantities
  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    // Analytic eigenvectors (RIGHT + LEFT)
    constexpr size_t matrix_size = 6;
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        right_eigenvectors{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        left_eigenvectors{};
    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&right_eigenvectors), make_not_null(&left_eigenvectors),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
        spatial_metric, *eos_3d);

    constexpr double tolerance = 1e-12;

    // Check that right_eigenvectors * left_eigenvectors = Identity matrix
    // This evaluates \sum_k R^{(k)}_i L^{(k)}_j = \delta_{ij}
    for (size_t i = 0; i < matrix_size; ++i) {
      for (size_t j = 0; j < matrix_size; ++j) {
        const double target = (i == j ? 1.0 : 0.0);

        double max_scaled_error = 0.0;
        for (size_t point = 0; point < used_for_size.size(); ++point) {
          double dot_ij = 0.0;
          double scale = 1.0;

          for (size_t k = 0; k < matrix_size; ++k) {
            const double product =
                gsl::at(right_eigenvectors, k).get(i)[point] *
                gsl::at(left_eigenvectors, k).get(j)[point];
            dot_ij += product;
            scale += std::abs(product);
          }

          const double err = std::abs(dot_ij - target);
          max_scaled_error = std::max(max_scaled_error, err / scale);
        }

        CHECK(max_scaled_error < tolerance);
      }
    }
  }
}

void test_hydro_numerical_eigensystem(const DataVector& used_for_size) {
  // Initialize number generator
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity = TestHelpers::hydro::random_velocity(
      nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density =
      TestHelpers::hydro::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      TestHelpers::hydro::random_specific_internal_energy(nn_gen,
                                                          used_for_size);
  const auto electron_fraction =
      TestHelpers::hydro::random_electron_fraction(nn_gen, used_for_size);

  // Define equation of state
  const auto equation_of_state_2d =
      EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();

  // Compute derived quantities
  const auto pressure = equation_of_state_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal and normal velocity in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);
    const auto normal_velocity =
        tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));

    // Initialize containers for all eigenvalues and eigenvectors
    constexpr size_t matrix_size = 6;
    std::array<Scalar<DataVector>, matrix_size> all_eigenvalues;
    std::array<tnsr::i<DataVector, matrix_size>, matrix_size>
        all_right_eigenvectors;
    std::array<tnsr::I<DataVector, matrix_size>, matrix_size>
        all_left_eigenvectors;
    const size_t num_points = used_for_size.size();
    for (size_t i = 0; i < matrix_size; ++i) {
      get(gsl::at(all_eigenvalues, i)).destructive_resize(num_points);
      for (size_t k = 0; k < matrix_size; ++k) {
        gsl::at(all_right_eigenvectors, i)
            .get(k)
            .destructive_resize(num_points);
        gsl::at(all_left_eigenvectors, i).get(k).destructive_resize(num_points);
      }
    }

    // Solve numerical eigensystem
    grmhd::ValenciaDivClean::numerical_eigensystem(
        make_not_null(&all_eigenvalues), make_not_null(&all_right_eigenvectors),
        make_not_null(&all_left_eigenvectors), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Get analytic characteristic speeds to check the numeric eigenvalues
    const std::array<DataVector, 3> analytic_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *equation_of_state_3d);

    // Count degenerate eigenvalues and check the other speeds
    // Note 1: We expect 4 degenerate eigenvalues equal to the normal velocity.
    // Note 2: The characteristic matrix becomes more defective for larger
    //         Lorentz boosts. With the default random Lorentz factor generator,
    //         the largest Lorentz factor is ~20, which leads to an eigenvalue
    //         error of ~1e-10.
    constexpr double eigenvalue_tolerance = 1e-9;
    for (size_t point = 0; point < num_points; ++point) {
      int number_of_degenerate_eigenvalues = 0;
      bool found_lambda_plus = false;
      bool found_lambda_minus = false;
      for (size_t i = 0; i < 6; ++i) {
        const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);
        const double diff_with_normal_velocity =
            std::abs(get(eigenvalue)[point] - get(normal_velocity)[point]);
        const double diff_with_lambda_plus =
            std::abs(get(eigenvalue)[point] -
                     analytic_speeds[HydroSpeed::LambdaPlus][point]);
        const double diff_with_lambda_minus =
            std::abs(get(eigenvalue)[point] -
                     analytic_speeds[HydroSpeed::LambdaMinus][point]);
        if (diff_with_normal_velocity < eigenvalue_tolerance) {
          number_of_degenerate_eigenvalues += 1;
        } else if (diff_with_lambda_plus < eigenvalue_tolerance) {
          CHECK_FALSE(found_lambda_plus);
          found_lambda_plus = true;
        } else if (diff_with_lambda_minus < eigenvalue_tolerance) {
          CHECK_FALSE(found_lambda_minus);
          found_lambda_minus = true;
        } else {
          FAIL(
              "Found an eigenvalue that does not match any expected "
              "characteristic speed.\n"
              "Lorentz factor: "
              << get(lorentz_factor)[point]
              << "\n"
                 "Differences with analytic eigenvalues: "
              << diff_with_normal_velocity << ", " << diff_with_lambda_plus
              << ", " << diff_with_lambda_minus);
        }
      }
      CHECK(number_of_degenerate_eigenvalues == 4);
      CHECK(found_lambda_plus);
      CHECK(found_lambda_minus);
    }

    // Get characteristic matrix to check eigensystem relations
    tnsr::iJ<DataVector, 6> characteristic_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Check eigensystem relation for each eigenvalue/eigenvector
    constexpr double numeric_tolerance = 1e-12;
    for (size_t i = 0; i < 6; ++i) {
      const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);

      const tnsr::i<DataVector, 6>& right_eigenvector =
          gsl::at(all_right_eigenvectors, i);
      const Scalar<DataVector> right_eigensystem_error =
          magnitude(tenex::evaluate<ti::i>(
              characteristic_matrix(ti::i, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::i)));

      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(all_left_eigenvectors, i);
      const Scalar<DataVector> left_eigensystem_error =
          magnitude(tenex::evaluate<ti::I>(
              left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::I) -
              eigenvalue() * left_eigenvector(ti::I)));

      double eigensystem_error = 0.0;
      for (size_t point = 0; point < used_for_size.size(); ++point) {
        eigensystem_error = std::max(
            eigensystem_error, std::abs(get(right_eigensystem_error)[point]));
        eigensystem_error = std::max(
            eigensystem_error, std::abs(get(left_eigensystem_error)[point]));
      }
      CHECK(eigensystem_error < numeric_tolerance);
    }
  }
}

void test_hydro_characteristic_match_unoptimized(
    const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  constexpr size_t matrix_size = 6;
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12);
  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    const auto optimized_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d);
    const auto unoptimized_speeds = grmhd::ValenciaDivClean::test_detail::
        characteristic_speeds_hydro_unoptimized<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d);
    CHECK_ITERABLE_CUSTOM_APPROX(optimized_speeds, unoptimized_speeds,
                                 custom_approx);

    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        optimized_outgoing_fields{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        optimized_ingoing_fields{};
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        unoptimized_outgoing_fields{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        unoptimized_ingoing_fields{};

    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&optimized_outgoing_fields),
        make_not_null(&optimized_ingoing_fields), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal, spatial_metric,
        *eos_3d);
    grmhd::ValenciaDivClean::test_detail::eigenvectors_hydro_unoptimized<3>(
        make_not_null(&unoptimized_outgoing_fields),
        make_not_null(&unoptimized_ingoing_fields), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal, spatial_metric,
        *eos_3d);

    for (size_t vector_index = 0; vector_index < matrix_size; ++vector_index) {
      for (size_t component = 0; component < matrix_size; ++component) {
        CHECK_ITERABLE_CUSTOM_APPROX(
            gsl::at(optimized_outgoing_fields, vector_index).get(component),
            gsl::at(unoptimized_outgoing_fields, vector_index).get(component),
            custom_approx);
        CHECK_ITERABLE_CUSTOM_APPROX(
            gsl::at(optimized_ingoing_fields, vector_index).get(component),
            gsl::at(unoptimized_ingoing_fields, vector_index).get(component),
            custom_approx);
      }
    }

    tnsr::iJ<DataVector, 6> optimized_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    tnsr::iJ<DataVector, 6> unoptimized_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro<3>(
        make_not_null(&optimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
    grmhd::ValenciaDivClean::test_detail::flux_jacobian_hydro_unoptimized<3>(
        make_not_null(&unoptimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
    CHECK_ITERABLE_CUSTOM_APPROX(optimized_matrix, unoptimized_matrix,
                                 custom_approx);
  }
}

void run_hydro_characteristic_benchmarks(const bool enable) {
  if (not enable) {
    return;
  }

  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const DataVector used_for_size(512);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  std::array<DataVector, 3> optimized_speeds{};
  std::array<DataVector, 3> unoptimized_speeds{};
  BENCHMARK("characteristic_speeds_hydro") {
    grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
        make_not_null(&optimized_speeds), spatial_velocity, rest_mass_density,
        specific_internal_energy, specific_enthalpy, electron_fraction,
        lorentz_factor, unit_normal, spatial_metric, *eos_3d);
  };
  BENCHMARK("characteristic_speeds_hydro_unoptimized") {
    grmhd::ValenciaDivClean::test_detail::
        characteristic_speeds_hydro_unoptimized<3>(
            make_not_null(&unoptimized_speeds), spatial_velocity,
            rest_mass_density, specific_internal_energy, specific_enthalpy,
            electron_fraction, lorentz_factor, unit_normal, spatial_metric,
            *eos_3d);
  };

  constexpr size_t matrix_size = 6;
  std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
      optimized_outgoing_fields{};
  std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
      optimized_ingoing_fields{};
  std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
      unoptimized_outgoing_fields{};
  std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
      unoptimized_ingoing_fields{};

  BENCHMARK("eigenvectors_hydro") {
    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&optimized_outgoing_fields),
        make_not_null(&optimized_ingoing_fields), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal, spatial_metric,
        *eos_3d);
  };
  BENCHMARK("eigenvectors_hydro_unoptimized") {
    grmhd::ValenciaDivClean::test_detail::eigenvectors_hydro_unoptimized<3>(
        make_not_null(&unoptimized_outgoing_fields),
        make_not_null(&unoptimized_ingoing_fields), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal, spatial_metric,
        *eos_3d);
  };

  tnsr::iJ<DataVector, 6> optimized_matrix =
      make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
  tnsr::iJ<DataVector, 6> unoptimized_matrix =
      make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
  BENCHMARK("flux_jacobian_hydro") {
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro<3>(
        make_not_null(&optimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
  BENCHMARK("flux_jacobian_hydro_unoptimized") {
    grmhd::ValenciaDivClean::test_detail::flux_jacobian_hydro_unoptimized<3>(
        make_not_null(&unoptimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  test_with_normal_along_coordinate_axes(dv);
  test_hydro_characteristic_speed(dv);
  test_hydro_characteristics_symmetry(dv);
  test_hydro_eigenvectors_identity(dv);
  test_hydro_numerical_eigensystem(dv);
  test_hydro_analytic_eigenvectors(dv);
  test_hydro_characteristic_match_unoptimized(dv);

  run_hydro_characteristic_benchmarks(true);

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
