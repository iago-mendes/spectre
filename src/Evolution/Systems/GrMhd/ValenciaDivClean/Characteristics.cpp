// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <stdexcept>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void compute_characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> pchar_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  // Mapping of indices between GRMHD char speeds and relativistic Euler char
  // speeds arrays.
  //
  // GRMHD     Rel Euler
  //   1           0
  //   2           1
  //   3           2
  //   4           3
  //   5           1
  //   6           1
  //   7           4
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>>>
      temp_tensors{num_grid_points};

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  {
    Scalar<DataVector>& normal_shift = get<::Tags::TempScalar<0>>(temp_tensors);
    dot_product(make_not_null(&normal_shift), normal, shift);
    char_speeds[0] = -1.0 * get(normal_shift);
    char_speeds[1] = char_speeds[0];
  }
  Scalar<DataVector>& scaled_sound_speed_squared =
      get<::Tags::TempScalar<5>>(temp_tensors);
  get(scaled_sound_speed_squared) =
      get(sound_speed_squared) +
      get(alfven_speed_squared) * (1.0 - get(sound_speed_squared));
  // Dim-fold degenerate eigenvalue, reuse normal_shift allocation
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  char_speeds[2] = char_speeds[1] + get(lapse) * get(normal_velocity);
  char_speeds[3] = char_speeds[2];
  char_speeds[4] = char_speeds[3];
  char_speeds[5] = char_speeds[2];
  char_speeds[6] = char_speeds[2];

  Scalar<DataVector>& one_minus_v_sqrd_cs_sqrd =
      get<::Tags::TempScalar<1>>(temp_tensors);
  get(one_minus_v_sqrd_cs_sqrd) =
      1.0 - get(spatial_velocity_squared) * get(scaled_sound_speed_squared);
  Scalar<DataVector>& vn_times_one_minus_cs_sqrd =
      get<::Tags::TempScalar<2>>(temp_tensors);
  get(vn_times_one_minus_cs_sqrd) =
      get(normal_velocity) * (1.0 - get(scaled_sound_speed_squared));

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(first_term) = get(lapse) / get(one_minus_v_sqrd_cs_sqrd);
  Scalar<DataVector>& second_term = get<::Tags::TempScalar<4>>(temp_tensors);
  get(second_term) =
      get(first_term) * sqrt(get(scaled_sound_speed_squared)) *
      sqrt((1.0 - get(spatial_velocity_squared)) *
           (get(one_minus_v_sqrd_cs_sqrd) -
            get(normal_velocity) * get(vn_times_one_minus_cs_sqrd)));
  get(first_term) *= get(vn_times_one_minus_cs_sqrd);

  char_speeds[7] = char_speeds[1] + get(first_term) + get(second_term);
  char_speeds[1] += get(first_term) - get(second_term);

  char_speeds[8] = char_speeds[0] + get(lapse);
  char_speeds[0] -= get(lapse);
}
}  // namespace

namespace grmhd::ValenciaDivClean {

template <size_t ThermodynamicDim>
void characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  // Remaining places to reduce allocations:
  // - EoS calls: 2 allocations
  // - Pass temp pointer to Rel Euler: 1 allocation
  // - Return a DataVectorArray (not yet implemented): 9 allocations
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0>(shift).size()};

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);
  const auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(
          &get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors)),
      magnetic_field, spatial_metric);
  const auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
              temp_tensors)),
      magnetic_field, spatial_velocity_one_form);
  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  const auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(
                  temp_tensors)),
              magnetic_field, magnetic_field_one_form);
  const auto& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(
      temp_tensors)) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));

  // reuse magnetic_field_squared allocation for Alfven speed squared
  const auto& alfven_speed_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors)) =
      get(comoving_magnetic_field_squared) /
      (get(comoving_magnetic_field_squared) +
       get(rest_mass_density) * get(specific_enthalpy));

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

  compute_characteristic_speeds_approximate_mhd(
      char_speeds, lapse, shift, spatial_velocity, spatial_velocity_squared,
      sound_speed_squared, alfven_speed_squared, unit_normal);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds_approximate_mhd(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::array<DataVector, 9> char_speeds{};
  characteristic_speeds_approximate_mhd(
      make_not_null(&char_speeds), rest_mass_density, electron_fraction,
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
  return char_speeds;
}

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_grid_points) {
    for (size_t i = 0; i < 3; ++i) {
      characteristic_speeds->get(i) = DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::Temperature<DataVector>,
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
    Scalar<DataVector>& temperature =
        get<hydro::Tags::Temperature<DataVector>>(temp_tensors);
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
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

  // Degenerate characteristic speed (normal dot velocity)
  characteristic_speeds->get(HydroSpeed::NormalDotVelocity) =
      get(normal_velocity);
  characteristic_speeds->get(HydroSpeed::LambdaPlus) =
      get(first_term) + get(second_term);
  characteristic_speeds->get(HydroSpeed::LambdaMinus) =
      get(first_term) - get(second_term);
}


template <size_t ThermodynamicDim>
tnsr::i<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  tnsr::i<DataVector, 3> characteristic_speeds{};
  characteristic_speeds_hydro(make_not_null(&characteristic_speeds),
                              spatial_velocity, rest_mass_density,
                              specific_internal_energy, electron_fraction,
                              lorentz_factor, specific_enthalpy, spatial_metric,
                              unit_normal, equation_of_state);
  return characteristic_speeds;
}

void magnetosonic_quartic_coefficients(
    gsl::not_null<tnsr::i<DataVector, 4>*> quartic_coefficients,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& normal_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& normal_magnetic_field,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& comoving_magnetic_field_squared) {
  // The expressions in this function have been optimized by Codex. See
  // unoptimized::magnetosonic_quartic_coefficients in Test_Characteristics.cpp
  // for the original expressions, which were directly compared against the
  // results from a Mathematica notebook.

  const DataVector& cs2 = get(sound_speed_squared);
  const DataVector& b2scaled = get(comoving_magnetic_field_squared);
  const DataVector& Bvscaled = get(magnetic_field_dot_spatial_velocity);
  const DataVector& Bsscaled = get(normal_magnetic_field);
  const DataVector& vn = get(normal_velocity);
  const DataVector& W = get(lorentz_factor);
  (void)magnetic_field_squared;

  Variables<tmpl::list<::Tags::TempScalar<0>>> temp_tensors{cs2.size()};
  DataVector& inv_denom = get(get<::Tags::TempScalar<0>>(temp_tensors));
  inv_denom = 1.0 / (square(W) * (b2scaled + square(W) * (1.0 - cs2) +
                                  cs2 * (1.0 - square(Bvscaled))));

  get<1>(*quartic_coefficients) =
      -(2.0 * Bsscaled * Bvscaled * cs2 +
        2.0 * vn * square(W) *
            (-b2scaled + (-1.0 + square(Bvscaled)) * cs2 -
             2.0 * (-1.0 + cs2) * square(vn) * square(W))) *
      inv_denom;
  get<3>(*quartic_coefficients) =
      -(-2.0 * Bsscaled * Bvscaled * cs2 +
        2.0 * vn * square(W) *
            (b2scaled + 2.0 * square(W) * (1.0 - cs2) +
             cs2 * (1.0 - square(Bvscaled)))) *
      inv_denom;

  inv_denom /= square(W);
  get<0>(*quartic_coefficients) =
      -(-square(Bsscaled) * cs2 -
        2.0 * Bsscaled * Bvscaled * cs2 * vn * square(W) +
        square(vn) * square(square(W)) *
            (b2scaled - (-1.0 + square(Bvscaled)) * cs2 +
             (-1.0 + cs2) * square(vn) * square(W))) *
      inv_denom;
  get<2>(*quartic_coefficients) =
      -(square(Bsscaled) * cs2 +
        2.0 * Bsscaled * Bvscaled * cs2 * vn * square(W) -
        (b2scaled - (-1.0 + square(Bvscaled)) * cs2) * (-1.0 + square(vn)) *
            square(square(W)) +
        6.0 * (-1.0 + cs2) * square(vn) * square(square(W)) * square(W)) *
      inv_denom;
}

void find_magnetosonic_speed_from_quartic(
    gsl::not_null<DataVector*> magnetosonic_speed,
    const tnsr::i<DataVector, 4>& quartic_coefficients) {
  // This function has been optimized by Codex to minimize memory allocation and
  // speed it up. See unoptimized::find_magnetosonic_speed_from_quartic in
  // Test_Characteristics.cpp for an easier-to-read implementation.

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
      temp_tensors{magnetosonic_speed->size()};

  constexpr size_t max_iters = 100;
  constexpr double tolerance = 1.0e-14;

  // We define the coefficients so that the quartic polynomial is
  // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
  DataVector& x = *magnetosonic_speed;
  const DataVector& c0 = get<0>(quartic_coefficients);
  const DataVector& c1 = get<1>(quartic_coefficients);
  const DataVector& c2 = get<2>(quartic_coefficients);
  const DataVector& c3 = get<3>(quartic_coefficients);

  // Find the root using Newton-Rapshon
  DataVector& F = get(get<::Tags::TempScalar<0>>(temp_tensors));
  DataVector& dF = get(get<::Tags::TempScalar<1>>(temp_tensors));
  for (size_t iter = 0; iter < max_iters; ++iter) {
    // Horner form minimizes intermediate temporary vectors in the hot loop.
    // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
    F = (((x + c3) * x + c2) * x + c1) * x + c0;

    if (max(abs(F)) < tolerance) {
      return;
    }

    // F'(x) = 4 x^3 + 3 c3 x^2 + 2 c2 x + c1
    dF = ((4.0 * x + 3.0 * c3) * x + 2.0 * c2) * x + c1;

    // Avoid FPE from dividing by a small derivative
    if (min(abs(dF)) < tolerance) {
      // It's possible for some points to have converged and others not.
      for (size_t point = 0; point < x.size(); ++point) {
        if (abs(F[point]) < tolerance) {
          // If a point has converged and has a small derivative, then just
          // skip the Newton step.
          F[point] = 0.0;
          dF[point] = 1.0;
        } else if (abs(F[point]) >= tolerance and abs(dF[point]) < tolerance) {
          // If a point that hasn't converged has a small derivative, then the
          // Newton step would be unreliable, so we error out.
          CAPTURE_FOR_ERROR(x);
          CAPTURE_FOR_ERROR(F);
          CAPTURE_FOR_ERROR(dF);
          ERROR(
              "Failed to compute magnetosonic speed from quartic: derivative "
              "is too small for a reliable Newton step. "
              "x = "
              << x[point] << ", F = " << F[point] << ", dF = " << dF[point]);
          return;
        }
      }
    }

    // Newton step
    x -= F / dF;
  }

  F = (((x + c3) * x + c2) * x + c1) * x + c0;
  ERROR(
      "Failed to compute magnetosonic speed from quartic: exceeded maximum "
      "number of iterations. Max |F| = "
      << max(abs(F)));
}

template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    const gsl::not_null<tnsr::i<DataVector, 9>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const SlowMagnetosonicSpeedMethod slow_speed_method) {
  const size_t num_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_points) {
    for (size_t i = 0; i < 9; ++i) {
      characteristic_speeds->get(i) = DataVector(num_points, 0.0);
    }
  }

  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::Tempi<0, 4>>>
      temp_tensors{num_points};

  // Get sound speed from EoS
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
    ERROR(
        "Characteristic speeds for MHD with a 3D equation of state are not "
        "implemented yet.");
  }

  // Scalar speeds
  get<MhdSpeed::ScalarPlus>(*characteristic_speeds) = +1.0;
  get<MhdSpeed::ScalarMinus>(*characteristic_speeds) = -1.0;

  // Entropy speed
  /*
    This is just the normal component of the fluid velocity. Since this quantity
    will be used later on, we also define normal_velocity, which points to the
    same DataVector as the entropy speed.
  */
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  get<MhdSpeed::Entropy>(*characteristic_speeds) = get(normal_velocity);

  // Alfven speeds
  // Intermediate magnetic variables
  Scalar<DataVector>& normal_magnetic_field =
      get<::Tags::TempScalar<1>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_magnetic_field),
                  magnetic_field(ti::I) * unit_normal(ti::i));
  tnsr::i<DataVector, 3>& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(make_not_null(&magnetic_field_one_form),
                         spatial_metric(ti::i, ti::j) * magnetic_field(ti::J));
  Scalar<DataVector>& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_dot_spatial_velocity),
                  magnetic_field_one_form(ti::i) * spatial_velocity(ti::I));
  Scalar<DataVector>& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_squared),
                  magnetic_field_one_form(ti::i) * magnetic_field(ti::I));
  Scalar<DataVector>& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(comoving_magnetic_field_squared) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  Scalar<DataVector>& rho_h_star = get<::Tags::TempScalar<3>>(temp_tensors);
  get(rho_h_star) = get(rest_mass_density) * get(specific_enthalpy) +
                    get(comoving_magnetic_field_squared);
  // Compute both speeds with both signs and then assign max/min to be alfven
  // plus/minus to ensure correct ordering.
  Scalar<DataVector>& alfven_1 = get<::Tags::TempScalar<4>>(temp_tensors);
  DataVector& alfven_plus_speed =
      get<MhdSpeed::AlfvenPlus>(*characteristic_speeds);
  DataVector& alfven_minus_speed =
      get<MhdSpeed::AlfvenMinus>(*characteristic_speeds);
  get(alfven_1) =
      get(normal_velocity) +
      get(normal_magnetic_field) / square(get(lorentz_factor)) /
          (get(magnetic_field_dot_spatial_velocity) + sqrt(get(rho_h_star)));
  alfven_minus_speed =
      get(normal_velocity) +
      get(normal_magnetic_field) / square(get(lorentz_factor)) /
          (get(magnetic_field_dot_spatial_velocity) - sqrt(get(rho_h_star)));
  alfven_plus_speed = max(get(alfven_1), alfven_minus_speed);
  alfven_minus_speed = min(get(alfven_1), alfven_minus_speed);

  // Polynomial coefficients for magnetosonic speeds
  // Re-scale magnetic intermediate variables so that they are dimensionless
  // Note: TempScalar<3> and TempScalar<4> are available to be used again here
  Scalar<DataVector>& inv_rho_h = get<::Tags::TempScalar<3>>(temp_tensors);
  Scalar<DataVector>& inv_sqrt_rho_h = get<::Tags::TempScalar<4>>(temp_tensors);
  get(inv_rho_h) = 1.0 / (get(rest_mass_density) * get(specific_enthalpy));
  get(inv_sqrt_rho_h) = sqrt(get(inv_rho_h));
  get(normal_magnetic_field) *= get(inv_sqrt_rho_h);
  get(magnetic_field_dot_spatial_velocity) *= get(inv_sqrt_rho_h);
  get(magnetic_field_squared) *= get(inv_rho_h);
  get(comoving_magnetic_field_squared) *= get(inv_rho_h);
  tnsr::i<DataVector, 4>& quartic_coefficients =
      get<::Tags::Tempi<0, 4>>(temp_tensors);
  magnetosonic_quartic_coefficients(
      make_not_null(&quartic_coefficients), sound_speed_squared,
      normal_velocity, lorentz_factor, normal_magnetic_field,
      magnetic_field_dot_spatial_velocity, magnetic_field_squared,
      comoving_magnetic_field_squared);

  // Fast magnetosonic speeds
  /*
    Find fast magnetosonic speeds via rootfinding of a quartic polynomial with
    initial guess of +1 (for positive speed) or -1 (for negative speed).
  */

  get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds) = 1.0;
  find_magnetosonic_speed_from_quartic(
      make_not_null(
          &get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds)),
      quartic_coefficients);
  get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds) = -1.0;
  find_magnetosonic_speed_from_quartic(
      make_not_null(
          &get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds)),
      quartic_coefficients);

  // Slow magnetosonic speeds
  constexpr double tolerance = 1.0e-14;
  constexpr double eps = 1.0e-12;
  constexpr double discriminant_tolerance = 1.0e-3;
  DataVector& slow_minus =
      get<MhdSpeed::SlowMagnetosonicMinus>(*characteristic_speeds);
  DataVector& slow_plus =
      get<MhdSpeed::SlowMagnetosonicPlus>(*characteristic_speeds);
  const DataVector& vn = get(normal_velocity);
  const DataVector& alfven_minus =
      get<MhdSpeed::AlfvenMinus>(*characteristic_speeds);
  const DataVector& alfven_plus =
      get<MhdSpeed::AlfvenPlus>(*characteristic_speeds);
  const DataVector& fast_minus =
      get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds);
  const DataVector& fast_plus =
      get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds);
  const DataVector& c0 = get<0>(quartic_coefficients);
  const DataVector& c1 = get<1>(quartic_coefficients);
  const DataVector& c2 = get<2>(quartic_coefficients);
  const DataVector& c3 = get<3>(quartic_coefficients);

  const auto evaluate_quartic = [&c0, &c1, &c2, &c3](const double y,
                                                     const size_t point) {
    return (((y + c3[point]) * y + c2[point]) * y + c1[point]) * y + c0[point];
  };

  for (size_t point = 0; point < num_points; ++point) {
    const double vn_i = vn[point];
    const double alfven_minus_i = alfven_minus[point];
    const double alfven_minus_eps_i = alfven_minus_i + eps;
    const double alfven_plus_i = alfven_plus[point];
    const double alfven_plus_eps_i = alfven_plus_i + eps;

    const double N_vn = evaluate_quartic(vn_i, point);
    const double N_alfven_minus = evaluate_quartic(alfven_minus_i, point);
    const double N_alfven_minus_eps =
        evaluate_quartic(alfven_minus_eps_i, point);
    const double N_alfven_plus = evaluate_quartic(alfven_plus_i, point);
    const double N_alfven_plus_eps = evaluate_quartic(alfven_plus_eps_i, point);
    // Check if we have one of the possible degeneracies and use it to avoid
    // rootfinding / reduced-quadratic solve for the slow roots
    if (std::abs(N_vn) < tolerance) {
      // Type I: alfven- = slow- = entropy = slow+ = alfven+
      slow_minus[point] = vn_i;
      slow_plus[point] = vn_i;
    } else if (std::abs(N_alfven_minus) < tolerance and
               N_alfven_minus_eps > 0.0) {
      // Type II on the minus side: alfven- = slow-
      slow_minus[point] = alfven_minus_i;
      slow_plus[point] =
          -c3[point] - slow_minus[point] - fast_minus[point] - fast_plus[point];
    } else if (std::abs(N_alfven_plus) < tolerance and
               N_alfven_plus_eps < 0.0) {
      // Type II on the plus side: slow+ = alfven+
      slow_plus[point] = alfven_plus_i;
      slow_minus[point] =
          -c3[point] - fast_minus[point] - fast_plus[point] - slow_plus[point];
    } else {
      if (slow_speed_method == SlowMagnetosonicSpeedMethod::ReducedQuadratic or
          slow_speed_method ==
              SlowMagnetosonicSpeedMethod::ReducedQuadraticThenNewton) {
        const double b_i = c3[point] + fast_minus[point] + fast_plus[point];
        const double c_i = c2[point] +
                           b_i * (fast_minus[point] + fast_plus[point]) -
                           fast_minus[point] * fast_plus[point];
        double discriminant_i = square(b_i) - 4.0 * c_i;
        ASSERT(discriminant_i >= -discriminant_tolerance,
               "Failed to compute slow magnetosonic speeds: reduced quadratic "
               "has negative discriminant below tolerance. discriminant = "
                   << discriminant_i << ", tolerance = "
                   << discriminant_tolerance << ", point = " << point);
        // Clamp discriminant to zero if it's slightly negative due to
        // numerical error.
        discriminant_i = std::max(discriminant_i, 0.0);

        // The cancellation-safe root of y^2 + b y + c = 0.
        const double q_i =
            -0.5 * (b_i + (b_i >= 0.0 ? 1.0 : -1.0) * sqrt(discriminant_i));

        slow_plus[point] = q_i;
        slow_minus[point] = -c3[point] - fast_minus[point] - fast_plus[point] -
                            slow_plus[point];
        if (slow_speed_method ==
            SlowMagnetosonicSpeedMethod::ReducedQuadraticThenNewton) {
          // Refine slow roots with one-variable Newton solves initialized at
          // reduced-quadratic estimates.
          DataVector seed_minus{1, slow_minus[point]};
          DataVector seed_plus{1, slow_plus[point]};
          find_magnetosonic_speed_from_quartic(make_not_null(&seed_minus),
                                               quartic_coefficients);
          find_magnetosonic_speed_from_quartic(make_not_null(&seed_plus),
                                               quartic_coefficients);
          slow_minus[point] = seed_minus[0];
          slow_plus[point] = seed_plus[0];
        }
        if (slow_minus[point] > slow_plus[point]) {
          std::swap(slow_minus[point], slow_plus[point]);
        }
      } else {
        CAPTURE_FOR_ERROR(alfven_minus_i);
        CAPTURE_FOR_ERROR(alfven_minus_eps_i);
        CAPTURE_FOR_ERROR(vn_i);
        CAPTURE_FOR_ERROR(alfven_plus_i);
        CAPTURE_FOR_ERROR(alfven_plus_eps_i);
        CAPTURE_FOR_ERROR(N_alfven_minus);
        CAPTURE_FOR_ERROR(N_alfven_minus_eps);
        CAPTURE_FOR_ERROR(N_vn);
        slow_minus[point] = RootFinder::toms748(
            [&evaluate_quartic, point](const double y) {
              return evaluate_quartic(y, point);
            },
            alfven_minus_eps_i, vn_i, N_alfven_minus_eps, N_vn,
            0.05 * tolerance, 0.05 * tolerance, /* max_iterations */ 100);
        slow_plus[point] = -c3[point] - slow_minus[point] - fast_minus[point] -
                           fast_plus[point];
      }
    }

    ASSERT(std::abs(evaluate_quartic(slow_minus[point], point)) <
                   10.0 * tolerance and
               std::abs(evaluate_quartic(slow_plus[point], point)) <
                   10.0 * tolerance,
           "Failed to find slow magnetosonic speeds: slow_minus = "
               << slow_minus[point] << ", slow_plus = " << slow_plus[point]
               << ", quartic(slow_minus) = "
               << evaluate_quartic(slow_minus[point], point)
               << ", quartic(slow_plus) = "
               << evaluate_quartic(slow_plus[point], point));
  }
}

template <size_t ThermodynamicDim>
void characteristic_eigenvectors_mhd(
    const gsl::not_null<tnsr::ij<DataVector, 9>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 9>*> characteristic_projectors,
    const tnsr::i<DataVector, 9>& characteristic_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_points = get(lorentz_factor).size();

  Scalar<DataVector> sound_speed_squared{num_points};
  Scalar<DataVector> kappa{num_points};
  Scalar<DataVector> pressure{num_points};
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
  } else if constexpr (ThermodynamicDim == 3) {
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "characteristic_eigenvectors_mhd currently only supports 3D EoSs "
          "in equilibrium.");
    }
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.0)};
    pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
  }

  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv_spatial_metric.first;
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto tangent_1 = orthonormal_oneform(unit_normal, inv_spatial_metric);
  const auto tangent_2 = orthonormal_oneform(
      unit_normal, tangent_1, spatial_metric, det_spatial_metric);
  Variables<tmpl::list<
      ::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>, ::Tags::TempI<0, 3>,
      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
      ::Tags::TempScalar<12>, ::Tags::TempScalar<13>, ::Tags::TempScalar<14>,
      ::Tags::TempScalar<15>, ::Tags::TempScalar<16>, ::Tags::TempScalar<17>,
      ::Tags::TempScalar<18>, ::Tags::TempScalar<19>, ::Tags::TempScalar<20>,
      ::Tags::TempScalar<21>, ::Tags::TempScalar<22>, ::Tags::TempScalar<23>,
      ::Tags::TempScalar<24>, ::Tags::TempScalar<25>, ::Tags::TempScalar<26>,
      ::Tags::TempScalar<27>, ::Tags::TempScalar<28>, ::Tags::TempScalar<29>,
      ::Tags::TempScalar<30>, ::Tags::TempScalar<31>, ::Tags::TempScalar<32>,
      ::Tags::TempScalar<33>, ::Tags::TempScalar<34>, ::Tags::TempScalar<35>,
      ::Tags::TempScalar<36>, ::Tags::TempScalar<37>, ::Tags::TempScalar<38>,
      ::Tags::TempScalar<39>>>
      temp_tensors{num_points};

  auto& v_cov = get<::Tags::Tempi<0, 3>>(temp_tensors);
  auto& B_cov = get<::Tags::Tempi<1, 3>>(temp_tensors);
  auto& s_vec = get<::Tags::TempI<0, 3>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    v_cov.get(i) = 0.0;
    B_cov.get(i) = 0.0;
    s_vec.get(i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      v_cov.get(i) += spatial_metric.get(i, j) * spatial_velocity.get(j);
      B_cov.get(i) += spatial_metric.get(i, j) * magnetic_field.get(j);
      s_vec.get(i) += inv_spatial_metric.get(i, j) * unit_normal.get(j);
    }
  }

  auto& v_n = get<::Tags::TempScalar<0>>(temp_tensors);
  auto& v_1 = get<::Tags::TempScalar<1>>(temp_tensors);
  auto& v_2 = get<::Tags::TempScalar<2>>(temp_tensors);
  auto& B_n = get<::Tags::TempScalar<3>>(temp_tensors);
  auto& B_1 = get<::Tags::TempScalar<4>>(temp_tensors);
  auto& B_2 = get<::Tags::TempScalar<5>>(temp_tensors);
  get(v_n) = 0.0;
  get(v_1) = 0.0;
  get(v_2) = 0.0;
  get(B_n) = 0.0;
  get(B_1) = 0.0;
  get(B_2) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(v_n) += spatial_velocity.get(i) * unit_normal.get(i);
    get(v_1) += spatial_velocity.get(i) * tangent_1.get(i);
    get(v_2) += spatial_velocity.get(i) * tangent_2.get(i);
    get(B_n) += magnetic_field.get(i) * unit_normal.get(i);
    get(B_1) += magnetic_field.get(i) * tangent_1.get(i);
    get(B_2) += magnetic_field.get(i) * tangent_2.get(i);
  }

  auto& B_squared = get<::Tags::TempScalar<6>>(temp_tensors);
  auto& B_dot_v = get<::Tags::TempScalar<7>>(temp_tensors);
  get(B_squared) = 0.0;
  get(B_dot_v) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(B_squared) += magnetic_field.get(i) * B_cov.get(i);
    get(B_dot_v) += B_cov.get(i) * spatial_velocity.get(i);
  }

  const DataVector& rho = get(rest_mass_density);
  const DataVector& h = get(specific_enthalpy);
  const DataVector& W = get(lorentz_factor);
  const DataVector& cs2 = get(sound_speed_squared);
  const DataVector& kappa_i = get(kappa);

  auto& b_squared = get<::Tags::TempScalar<8>>(temp_tensors);
  auto& rho_h_star = get<::Tags::TempScalar<9>>(temp_tensors);
  auto& h_star = get<::Tags::TempScalar<10>>(temp_tensors);
  auto& sqrt_rho_h_star = get<::Tags::TempScalar<11>>(temp_tensors);
  auto& r_1 = get<::Tags::TempScalar<12>>(temp_tensors);
  auto& r_2 = get<::Tags::TempScalar<13>>(temp_tensors);
  auto& r_4 = get<::Tags::TempScalar<14>>(temp_tensors);
  auto& B_21 = get<::Tags::TempScalar<15>>(temp_tensors);
  auto& B_31 = get<::Tags::TempScalar<16>>(temp_tensors);
  auto& B_32 = get<::Tags::TempScalar<17>>(temp_tensors);
  get(b_squared) = get(B_squared) / square(W) + square(get(B_dot_v));
  get(rho_h_star) = rho * h + get(b_squared);
  get(h_star) = h + get(b_squared) / rho;
  get(sqrt_rho_h_star) = sqrt(get(rho_h_star));
  get(r_1) = get(B_dot_v) + get(sqrt_rho_h_star);
  get(r_2) = get(B_n) * get(v_n) - get(r_1);
  get(r_4) = get(B_squared) + get(r_1) * get(B_dot_v) * square(W);
  get(B_21) = get(B_2) * get(v_1) - get(B_1) * get(v_2);
  get(B_31) = get(B_n) * get(v_1) - get(B_1) * get(v_n);
  get(B_32) = get(B_n) * get(v_2) - get(B_2) * get(v_n);

  auto& a = get<::Tags::TempScalar<20>>(temp_tensors);
  auto& B = get<::Tags::TempScalar<21>>(temp_tensors);
  auto& G = get<::Tags::TempScalar<22>>(temp_tensors);
  auto& script_G = get<::Tags::TempScalar<23>>(temp_tensors);
  auto& script_G_rho = get<::Tags::TempScalar<24>>(temp_tensors);
  auto& kappa_rho = get<::Tags::TempScalar<25>>(temp_tensors);
  auto& Z = get<::Tags::TempScalar<26>>(temp_tensors);
  auto& K = get<::Tags::TempScalar<27>>(temp_tensors);
  auto& kappa_B = get<::Tags::TempScalar<28>>(temp_tensors);
  auto& kappa_Bv = get<::Tags::TempScalar<29>>(temp_tensors);
  auto& m_1s = get<::Tags::TempScalar<30>>(temp_tensors);
  auto& m_1v = get<::Tags::TempScalar<31>>(temp_tensors);
  auto& m_1B = get<::Tags::TempScalar<32>>(temp_tensors);
  auto& m_4 = get<::Tags::TempScalar<33>>(temp_tensors);
  auto& f_1v = get<::Tags::TempScalar<34>>(temp_tensors);
  auto& g_1B = get<::Tags::TempScalar<35>>(temp_tensors);
  auto& g_1v = get<::Tags::TempScalar<36>>(temp_tensors);
  auto& h_1 = get<::Tags::TempScalar<37>>(temp_tensors);
  auto& G_entropy = get<::Tags::TempScalar<38>>(temp_tensors);

  for (size_t wave = 0; wave < 9; ++wave) {
    const DataVector& y = characteristic_speeds.get(wave);
    get(a) = W * (get(v_n) - y);
    get(B) = get(B_n) / W + get(B_dot_v) * W * (get(v_n) - y);
    get(G) = 1.0 - square(y);
    get(script_G) = rho * h * square(get(a)) - get(G) * get(b_squared);
    get(script_G_rho) = get(script_G) / (rho * h * cs2);
    get(kappa_rho) = kappa_i + rho * cs2;
    get(Z) = rho * h * square(W);
    get(K) = -W * (1.0 - get(v_n) * y);
    get(kappa_B) = get(kappa_rho) * square(get(B)) +
                   (1.0 - cs2) * square(rho * get(a)) * h;
    get(kappa_Bv) = get(kappa_B) * get(B_dot_v) -
                    get(kappa_rho) * rho * get(a) * get(B) * get(h_star);
    const DataVector a_denom = get(a) + 1.0e-14;
    const DataVector G_denom = get(G) + 1.0e-14;
    const DataVector cs2_denom = cs2 + 1.0e-14;

    if (wave == MhdSpeed::Entropy) {
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            h * W * (kappa_i - rho * cs2) * spatial_velocity.get(i);
        characteristic_modes->get(wave, 3 + i) = 0.0;
      }
      characteristic_modes->get(wave, 6) = kappa_i;
      characteristic_modes->get(wave, 7) =
          kappa_i * (h * W - 1.0) - rho * h * W * cs2;
      characteristic_modes->get(wave, 8) = 0.0;

      get(G_entropy) = 1.0 - square(get(v_n));
      const DataVector entropy_norm = 1.0 / (rho * h * cs2);
      for (size_t i = 0; i < 3; ++i) {
        // S_b components
        characteristic_projectors->get(wave, i) =
            entropy_norm * W * v_cov.get(i);
        // B_b components: gamma_{ba}b^a - (B_n / GW) s_b
        const DataVector b_cov_i =
            B_cov.get(i) / W + W * v_cov.get(i) * get(B_dot_v);
        characteristic_projectors->get(wave, 3 + i) =
            entropy_norm *
            (b_cov_i - (get(B_n) / (get(G_entropy) * W)) * unit_normal.get(i));
      }
      // D component
      characteristic_projectors->get(wave, 6) = entropy_norm * (h - W);
      // tau component
      characteristic_projectors->get(wave, 7) = entropy_norm * (-W);
      // phi component: W B^a v_a - B_n v_n / GW
      characteristic_projectors->get(wave, 8) =
          entropy_norm *
          (W * get(B_dot_v) - (get(B_n) * get(v_n)) / (get(G_entropy) * W));
    } else if (wave == MhdSpeed::AlfvenMinus or wave == MhdSpeed::AlfvenPlus) {
      // Explicit S_b components (Rows 1, 2, 3 of the paper's array)
      characteristic_modes->get(wave, 0) =
          -2.0 * get(sqrt_rho_h_star) * get(B_21) *
          (get(B_n) + get(r_1) * get(v_n) * square(W));
      characteristic_modes->get(wave, 1) =
          -get(sqrt_rho_h_star) *
          (get(B_n) * get(B_32) + get(B_1) * get(B_21) +
           get(r_1) * square(W) *
               (get(B_2) + get(v_1) * get(B_21) + get(v_n) * get(B_32)));
      characteristic_modes->get(wave, 2) =
          get(sqrt_rho_h_star) *
          (get(B_n) * get(B_31) - get(B_2) * get(B_21) +
           get(r_1) * square(W) *
               (get(B_1) - get(v_2) * get(B_21) + get(v_n) * get(B_31)));

      // Explicit B_b components (Rows 4, 5, 6 of the paper's array)
      characteristic_modes->get(wave, 3) = 0.0;
      characteristic_modes->get(wave, 4) =
          get(sqrt_rho_h_star) * get(B_2) + get(v_2) * get(r_4);
      characteristic_modes->get(wave, 5) =
          -get(sqrt_rho_h_star) * get(B_1) - get(v_1) * get(r_4);

      // Explicit Scalar components D and tau (Rows 7, 8 of the paper's array)
      characteristic_modes->get(wave, 6) = -rho * W * get(B_21);
      characteristic_modes->get(wave, 7) =
          -get(B_21) * W * (2.0 * W * get(sqrt_rho_h_star) * get(r_1) - rho);

      // Appended 9th component for the Divergence Cleaning scalar phi
      characteristic_modes->get(wave, 8) = 0.0;

      const DataVector& y_Alf = y;
      const DataVector alf_norm = 1.0 / get(sqrt_rho_h_star);

      // Explicit S_b components (Rows 1, 2, 3 of the paper's array)
      characteristic_projectors->get(wave, 0) = alf_norm * (get(B_21) * y_Alf);
      characteristic_projectors->get(wave, 1) =
          alf_norm * (get(B_2) + get(B_32) * y_Alf);
      characteristic_projectors->get(wave, 2) =
          alf_norm * (-get(B_1) - get(B_31) * y_Alf);

      // Explicit B_b components (Rows 4, 5, 6 of the paper's array)
      characteristic_projectors->get(wave, 3) =
          -get(B_21) * y_Alf * get(sqrt_rho_h_star);
      characteristic_projectors->get(wave, 4) = -(get(B_2) + get(B_32) * y_Alf);
      characteristic_projectors->get(wave, 5) = (get(B_1) + get(B_31) * y_Alf);

      // Explicit Scalar components D and tau (Rows 7, 8 of the paper's array)
      characteristic_projectors->get(wave, 6) = alf_norm * (-get(B_21));
      characteristic_projectors->get(wave, 7) = alf_norm * (-get(B_21));

      // Appended 9th component for the Divergence Cleaning scalar phi
      characteristic_projectors->get(wave, 8) = -get(B_21);
    } else if (wave == MhdSpeed::ScalarMinus or wave == MhdSpeed::ScalarPlus) {
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            -((y * get(kappa_B) +
               2.0 * get(kappa_rho) * get(a) * get(B) * get(B_n)) *
                  magnetic_field.get(i) +
              square(W) * get(B) * get(kappa_B) *
                  (s_vec.get(i) + y * spatial_velocity.get(i)) -
              2.0 * W * get(kappa_rho) * get(B) * square(get(B_n)) *
                  spatial_velocity.get(i)) /
            (a_denom * W);
        characteristic_modes->get(wave, 3 + i) =
            get(kappa_rho) * y * get(B) * magnetic_field.get(i) / W +
            (s_vec.get(i) - y * spatial_velocity.get(i)) *
                (get(kappa_rho) * get(B) * get(B_n) +
                 (1.0 - cs2) * square(rho) * square(get(a)) * h * W) /
                a_denom;
      }
      characteristic_modes->get(wave, 6) =
          get(kappa_rho) * y * rho * get(B) -
          (1.0 - cs2) * square(rho * get(a)) * h * W / a_denom;
      characteristic_modes->get(wave, 7) =
          (get(kappa_rho) * get(B) *
               (2.0 * square(get(B_n)) +
                rho * get(a) * (get(a) * get(h_star) - y)) -
           get(kappa_B) * get(B) - 2.0 * get(kappa_Bv) * y * W +
           (1.0 - cs2) * square(rho * get(a)) * get(B_n)) /
          a_denom;
      characteristic_modes->get(wave, 8) =
          -(1.0 - cs2) * square(rho * get(a)) * h;

      const DataVector inv_one_minus_vn2 =
          1.0 / ((1.0 - square(get(v_n))) + 1.0e-14);
      for (size_t i = 0; i < 3; ++i) {
        characteristic_projectors->get(wave, i) = 0.0;
        characteristic_projectors->get(wave, 3 + i) =
            unit_normal.get(i) * inv_one_minus_vn2;
      }
      characteristic_projectors->get(wave, 6) = 0.0;
      characteristic_projectors->get(wave, 7) = 0.0;
      characteristic_projectors->get(wave, 8) = y * inv_one_minus_vn2;
    } else {
      get(m_1s) = rho * h * get(a) * W *
                  (get(B) * get(B_dot_v) - get(rho_h_star) * get(a));
      get(m_1v) = rho * h *
                  (get(B_n) * get(B_dot_v) * (y * get(a) + 2.0 * get(G) * W) -
                   2.0 * get(a) * square(get(B_n)) -
                   get(a) * W *
                       (y * get(a) * (get(B_squared) / square(W) + rho * h) +
                        (1.0 - 1.0 / cs2) * get(script_G) * W));
      get(m_1B) = rho * h *
                  (get(B) * (y * get(a) - get(G) * W) +
                   2.0 * get(B_n) * (square(get(a)) + get(G))) /
                  W;
      get(m_4) =
          (rho / a_denom) *
          (square(get(a)) * square(get(B)) * h -
           2.0 * square(get(B_n)) * h * (square(get(a)) + get(G)) +
           square(get(B)) * W * (2.0 * y * get(a) * h - get(G)) +
           get(B_n) * get(B) *
               (get(G) + 2.0 * h * W * get(G) - 2.0 * y * get(a) * h) +
           get(script_G) * W * square(get(a)) * (h * W * (1.0 - cs2) - 1.0) /
               cs2_denom +
           rho * h * cube(get(a)) *
               (y - 2.0 * y * get(h_star) * W + get(a) * (W - get(h_star))));
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            get(m_1s) * s_vec.get(i) + get(m_1v) * spatial_velocity.get(i) +
            get(m_1B) * magnetic_field.get(i);
        characteristic_modes->get(wave, 3 + i) =
            rho * h * get(a) *
            (magnetic_field.get(i) * (1.0 - y * get(v_n)) -
             get(B_n) * (s_vec.get(i) - y * spatial_velocity.get(i)));
      }
      characteristic_modes->get(wave, 6) =
          -rho * get(B) * get(G) * get(B_n) / a_denom -
          square(rho) * get(a) * h * (y * get(a) - get(G) * W);
      characteristic_modes->get(wave, 7) = get(m_4);
      characteristic_modes->get(wave, 8) = 0.0;

      get(f_1v) =
          W * (-get(G) +
               get(B) * get(G) * get(B_n) * W / (get(Z) * square(a_denom)) +
               get(script_G) * square(W) * (kappa_i + rho) /
                   (get(Z) * rho * cs2_denom));
      get(g_1B) = get(script_G) * kappa_i * W / (rho * get(Z) * cs2_denom) -
                  (square(get(a)) + get(G)) / W;
      get(g_1v) = get(B_dot_v) * square(W) * get(g_1B) +
                  W * (get(a) * get(B) +
                       get(script_G) * get(B) * square(W) / (get(Z) * a_denom));
      get(h_1) = -(get(f_1v) + y * get(a));
      for (size_t i = 0; i < 3; ++i) {
        characteristic_projectors->get(wave, i) =
            get(a) * unit_normal.get(i) -
            get(B) * get(G) * W * B_cov.get(i) / (get(Z) * a_denom) +
            get(f_1v) * v_cov.get(i);
        characteristic_projectors->get(wave, 3 + i) =
            get(B) * unit_normal.get(i) + get(g_1B) * B_cov.get(i) +
            get(g_1v) * v_cov.get(i) -
            (get(script_G_rho) * get(kappa_rho) * get(B) / (G_denom * rho)) *
                unit_normal.get(i);
      }
      characteristic_projectors->get(wave, 6) =
          get(h_1) +
          get(script_G) * (kappa_i - rho * cs2) / (square(rho) * cs2_denom);
      characteristic_projectors->get(wave, 7) = get(h_1);
      characteristic_projectors->get(wave, 8) =
          (get(B) * get(K) *
               (get(G) * rho - get(script_G_rho) * get(kappa_rho)) +
           get(G) * get(B_n) *
               (rho * (square(get(a)) + get(G)) -
                get(script_G_rho) * kappa_i)) /
          (G_denom * rho * a_denom);
    }
  }
}

template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,
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
  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempI<0, 3>, ::Tags::Tempi<0, 3>,
      ::Tags::TempI<1, 3>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  Scalar<DataVector>& pressure =
      get<hydro::Tags::Pressure<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<2>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    // For non-equilibrium 3D EoSs we do not have direct access to kappa and we
    // don't know how to specify zeta, both of which are needed
    // for the expressions here. So, we currently only support equilibrium 3D
    // EoSs, for which we get kappa from the underlying 2D EoS and set zeta to
    // 0.
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "flux_jacobian_hydro currently only supports 3D EoSs in "
          "equilibrium.");
    }
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<2>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    // For now, we assume that we are at compositional equilibrium, so we set
    // zeta to zero.
    get(zeta) = 0.0;
  }

  // The expressions in this function have been iteratively optimized by Codex
  // with 3 main goals:
  //   1. Avoid catastrophic cancellations by rewriting terms like W-1 to, e.g.,
  //      v^2 W^2 / (W + 1). Known tricks were listed in a skill used by Codex.
  //   2. Avoid repeated calculations by storing intermediate results in
  //      temporary variables and reusing previous allocations when possible.
  //   3. Try different rearrangements and run benchmark to find the best
  //      optimization.

  // Intermediate variables
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<3>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  tnsr::I<DataVector, 3>& unit_vector = get<::Tags::TempI<0, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(make_not_null(&unit_vector),
                         inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  tnsr::i<DataVector, 3>& spatial_velocity_one_form =
      get<::Tags::Tempi<0, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(
      make_not_null(&spatial_velocity_one_form),
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));

  // Cancellation-safe thermodynamic / kinematic differences
  Scalar<DataVector>& h_minus_1 = get<::Tags::TempScalar<9>>(temp_tensors);
  get(h_minus_1) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  Scalar<DataVector>& velocity_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  tenex::evaluate(make_not_null(&velocity_squared),
                  spatial_velocity(ti::I) * spatial_velocity_one_form(ti::i));
  Scalar<DataVector>& lorentz_factor_squared =
      get<::Tags::TempScalar<11>>(temp_tensors);
  get(lorentz_factor_squared) = square(get(lorentz_factor));
  Scalar<DataVector>& W_minus_1 = get<::Tags::TempScalar<12>>(temp_tensors);
  get(W_minus_1) = get(velocity_squared) * get(lorentz_factor_squared) /
                   (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& h_minus_W = get<::Tags::TempScalar<13>>(temp_tensors);
  get(h_minus_W) = get(h_minus_1) - get(W_minus_1);

  // Derivatives of Z
  Scalar<DataVector>& inv_derivative_denom =
      get<::Tags::TempScalar<7>>(temp_tensors);
  get(inv_derivative_denom) =
      1.0 / ((-get(lorentz_factor_squared) + get(sound_speed_squared) *
                                                 get(velocity_squared) *
                                                 get(lorentz_factor_squared)) *
             get(rest_mass_density));
  Scalar<DataVector>& dzdD = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdD),
      -(lorentz_factor() *
        (-kappa() * h_minus_W() - zeta() * electron_fraction() +
         (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
             rest_mass_density()) *
        inv_derivative_denom()));
  tnsr::I<DataVector, 3>& dzds = get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dzds),
      (spatial_velocity(ti::I) * lorentz_factor_squared() *
       (kappa() + sound_speed_squared() * rest_mass_density()) *
       inv_derivative_denom()));
  Scalar<DataVector>& dzdtau = get<::Tags::TempScalar<5>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdtau),
                  -(lorentz_factor_squared() * (kappa() + rest_mass_density()) *
                    inv_derivative_denom()));
  Scalar<DataVector>& dzdye = get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdye),
                  -(zeta() * lorentz_factor() * inv_derivative_denom()));

  // Common factors used repeatedly in the characteristic matrix entries
  Scalar<DataVector>& D_over_Z = get<::Tags::TempScalar<8>>(temp_tensors);
  get(D_over_Z) = 1.0 / (get(specific_enthalpy) * get(lorentz_factor));
  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      (1.0 - get(D_over_Z) * get(dzdD)) * get(normal_velocity);
  characteristic_matrix->get(0, 1) =
      get(D_over_Z) * (unit_vector.get(0) - dzds.get(0) * get(normal_velocity));
  characteristic_matrix->get(0, 2) =
      get(D_over_Z) * (unit_vector.get(1) - dzds.get(1) * get(normal_velocity));
  characteristic_matrix->get(0, 3) =
      get(D_over_Z) * (unit_vector.get(2) - dzds.get(2) * get(normal_velocity));
  characteristic_matrix->get(4, 1) =
      unit_vector.get(0) - characteristic_matrix->get(0, 1);
  characteristic_matrix->get(4, 2) =
      unit_vector.get(1) - characteristic_matrix->get(0, 2);
  characteristic_matrix->get(4, 3) =
      unit_vector.get(2) - characteristic_matrix->get(0, 3);
  characteristic_matrix->get(5, 1) =
      get(electron_fraction) * characteristic_matrix->get(0, 1);
  characteristic_matrix->get(5, 2) =
      get(electron_fraction) * characteristic_matrix->get(0, 2);
  characteristic_matrix->get(5, 3) =
      get(electron_fraction) * characteristic_matrix->get(0, 3);
  characteristic_matrix->get(0, 4) =
      -(get(D_over_Z) * get(dzdtau) * get(normal_velocity));
  characteristic_matrix->get(0, 5) =
      -(get(D_over_Z) * get(dzdye) * get(normal_velocity));
  characteristic_matrix->get(1, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(0) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(0);
  characteristic_matrix->get(2, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(1) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(1);
  characteristic_matrix->get(3, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(2) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(2);

  characteristic_matrix->get(1, 1) =
      get(normal_velocity) +
      unit_vector.get(0) * spatial_velocity_one_form.get(0) +
      dzds.get(0) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(0) +
      dzds.get(1) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(0) +
      dzds.get(2) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(1) +
      dzds.get(0) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 2) =
      get(normal_velocity) +
      unit_vector.get(1) * spatial_velocity_one_form.get(1) +
      dzds.get(1) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(1) +
      dzds.get(2) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(2) +
      dzds.get(0) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(2) +
      dzds.get(1) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 3) =
      get(normal_velocity) +
      unit_vector.get(2) * spatial_velocity_one_form.get(2) +
      dzds.get(2) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));

  characteristic_matrix->get(1, 4) =
      -unit_normal.get(0) +
      get(dzdtau) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 4) =
      -unit_normal.get(1) +
      get(dzdtau) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 4) =
      -unit_normal.get(2) +
      get(dzdtau) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(1, 5) =
      get(dzdye) * (unit_normal.get(0) -
                    get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 5) =
      get(dzdye) * (unit_normal.get(1) -
                    get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 5) =
      get(dzdye) * (unit_normal.get(2) -
                    get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(4, 0) = -characteristic_matrix->get(0, 0);
  characteristic_matrix->get(4, 4) = -characteristic_matrix->get(0, 4);
  characteristic_matrix->get(4, 5) = -characteristic_matrix->get(0, 5);
  characteristic_matrix->get(5, 0) = -get(electron_fraction) * get(D_over_Z) *
                                     get(dzdD) * get(normal_velocity);
  characteristic_matrix->get(5, 4) =
      get(electron_fraction) * characteristic_matrix->get(0, 4);
  characteristic_matrix->get(5, 5) =
      (1.0 - get(electron_fraction) * get(D_over_Z) * get(dzdye)) *
      get(normal_velocity);
}

template <size_t ThermodynamicDim>
void numerical_characteristics(
    const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,
    const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,
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
  // Build characteristic matrix
  Variables<tmpl::list<::Tags::TempiJ<0, 6>>> temp_tensors{
      get<0, 0>(spatial_metric).size()};
  tnsr::iJ<DataVector, 6>& characteristic_matrix =
      get<::Tags::TempiJ<0, 6>>(temp_tensors);
  flux_jacobian_hydro(make_not_null(&characteristic_matrix), spatial_velocity,
                      rest_mass_density, specific_internal_energy,
                      electron_fraction,
                      /* other helpful quantities */
                      lorentz_factor, specific_enthalpy, spatial_metric,
                      inv_spatial_metric, unit_normal, equation_of_state);

  // Allocate memory to work with blaze::geev (outside of loop to save
  // time/memory)
  constexpr size_t matrix_size = 6;
  blaze::StaticMatrix<double, matrix_size, matrix_size> blaze_point_matrix{};
  blaze::StaticVector<blaze::complex<double>, matrix_size>
      blaze_complex_eigenvalues{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_L{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_R{};
  std::array<double, matrix_size> blaze_real_eigenvalues{};

  // Loop over each grid point
  const size_t num_points = get<0, 0>(spatial_metric).size();
  for (size_t point = 0; point < num_points; ++point) {
    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        blaze_point_matrix(row, col) =
            characteristic_matrix.get(row, col)[point];
      }
    }

    // Solve eigensystem using blaze:geev
    blaze::geev(blaze_point_matrix, blaze_complex_L, blaze_complex_eigenvalues,
                blaze_complex_R);
    const double tolerance = 1.0e-12;
    for (size_t i = 0; i < matrix_size; ++i) {
      ASSERT(std::abs(blaze_complex_eigenvalues.at(i).imag()) < tolerance,
             "Complex eigenvalue: "
                 << blaze_complex_eigenvalues.at(i).real() << " + "
                 << blaze_complex_eigenvalues.at(i).imag() << " i.");
      gsl::at(blaze_real_eigenvalues, i) =
          blaze_complex_eigenvalues.at(i).real();
    }

// We found cases in which blaze::geev returns a wrong eigenvalue. As a sanity
// check, we use GSL to compute the eigenvalues for the same matrix and check
// that they are the same.
#ifdef SPECTRE_DEBUG
    // Allocate memory to work with GSL:
    // Workspace for computing eigenvalues and eigenvectors
    gsl_eigen_nonsymm_workspace* gsl_workspace =
        gsl_eigen_nonsymm_alloc(matrix_size);
    // Containers for the right eigensystem (A * R = lambda * R)
    gsl_matrix* gsl_point_matrix = gsl_matrix_alloc(matrix_size, matrix_size);
    gsl_vector_complex* gsl_complex_eigenvalues =
        gsl_vector_complex_alloc(matrix_size);
    gsl_matrix_complex* gsl_complex_right_eigenvectors =
        gsl_matrix_complex_alloc(matrix_size, matrix_size);
    std::array<double, matrix_size> gsl_real_eigenvalues{};

    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        const double entry = characteristic_matrix.get(row, col)[point];
        gsl_matrix_set(gsl_point_matrix, row, col, entry);
      }
    }

    // Solve for eigenvalues using GSL
    gsl_eigen_nonsymm(gsl_point_matrix, gsl_complex_eigenvalues, gsl_workspace);
    for (size_t i = 0; i < matrix_size; ++i) {
      gsl::at(gsl_real_eigenvalues, i) =
          GSL_REAL(gsl_vector_complex_get(gsl_complex_eigenvalues, i));
    }

    // Sort both blaze's and GSL's eigenvalues to facilitate comparison
    std::array<double, matrix_size> sorted_blaze_real_eigenvalues =
        blaze_real_eigenvalues;
    std::sort(sorted_blaze_real_eigenvalues.begin(),
              sorted_blaze_real_eigenvalues.end());
    std::sort(gsl_real_eigenvalues.begin(), gsl_real_eigenvalues.end());

    // Compare eigenvalues
    for (size_t i = 0; i < matrix_size; ++i) {
      if (UNLIKELY(std::abs(gsl::at(sorted_blaze_real_eigenvalues, i) -
                            gsl::at(gsl_real_eigenvalues, i)) > 1.e-6)) {
        std::ostringstream matrix_stream;
        matrix_stream << "Point matrix:\n";
        for (size_t row = 0; row < matrix_size; ++row) {
          for (size_t col = 0; col < matrix_size; ++col) {
            matrix_stream << blaze_point_matrix(row, col) << " ";
          }
          matrix_stream << "\n";
        }
        matrix_stream << "Blaze eigenvalue: "
                      << gsl::at(sorted_blaze_real_eigenvalues, i)
                      << ", GSL eigenvalue: "
                      << gsl::at(gsl_real_eigenvalues, i) << "\n";

        ERROR(
            "The eigensolvers from blaze and GSL found different eigenvalues "
            "for the same matrix.\n"
            << matrix_stream.str());
      }
    }

    // Free GSL allocations
    gsl_eigen_nonsymm_free(gsl_workspace);
    gsl_matrix_free(gsl_point_matrix);
    gsl_vector_complex_free(gsl_complex_eigenvalues);
    gsl_matrix_complex_free(gsl_complex_right_eigenvectors);
#endif

    // Check and save results
    // Track which indices have been processed to avoid double-counting partners
    std::array<bool, matrix_size> processed_right_eigenvectors{};
    std::array<bool, matrix_size> processed_left_eigenvectors{};
    // Note: we're looping through the ith eigenvalue/vectors, not the ith row!
    for (size_t i = 0; i < matrix_size; ++i) {
      // Store eigenvalue
      characteristic_speeds->get(i)[point] = gsl::at(blaze_real_eigenvalues, i);

      // For each PAIR of degenerate eigenvalues, it is possible that GSL
      // returns a PAIR of complex eigenvectors that are complex conjugates of
      // each other. Here, we check if the ith right/left eigenvector is
      // complex.
      bool right_eigenvector_is_complex = false;
      bool left_eigenvector_is_complex = false;
      for (size_t k = 0; k < matrix_size; ++k) {
        if (std::abs(blaze_complex_R(k, i).imag()) > tolerance) {
          right_eigenvector_is_complex = true;
        }

        if (std::abs(blaze_complex_L(k, i).imag()) > tolerance) {
          left_eigenvector_is_complex = true;
        }
      }

      // If either eigenvector is complex, then we know that the vectors formed
      // by their real and imaginary parts are also linearly-independent
      // eigenvectors. Here, we use this fact to build real-valued left and
      // right eigenvectors.
      if (not gsl::at(processed_right_eigenvectors, i)) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (right_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_R(k, i);
              const auto conjugate_component = blaze_complex_R(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not right_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex right eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (right_eigenvector_is_complex) {
          ASSERT(
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith right eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_modes->get(i, k)[point] = blaze_complex_R(k, i).real();
          if (right_eigenvector_is_complex) {
            characteristic_modes->get(conjugate_i, k)[point] =
                blaze_complex_R(k, i).imag();
          }
        }
        gsl::at(processed_right_eigenvectors, i) = true;
        if (right_eigenvector_is_complex) {
          gsl::at(processed_right_eigenvectors, conjugate_i) = true;
        }
      }

      // Same as above, but for left eigenvectors
      if (not gsl::at(processed_left_eigenvectors, i)) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (left_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_L(k, i);
              const auto conjugate_component = blaze_complex_L(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not left_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex left eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (left_eigenvector_is_complex) {
          ASSERT(
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith left eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_projectors->get(i, k)[point] =
              blaze_complex_L(k, i).real();
          if (left_eigenvector_is_complex) {
            characteristic_projectors->get(conjugate_i, k)[point] =
                blaze_complex_L(k, i).imag();
          }
        }
        gsl::at(processed_left_eigenvectors, i) = true;
        if (left_eigenvector_is_complex) {
          gsl::at(processed_left_eigenvectors, conjugate_i) = true;
        }
      }
    }
  }
}

namespace Tags {

template <size_t ThermodynamicDim>
void CharacteristicSpeedsCompute::function(
    const gsl::not_null<return_type*> result,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& /* electron_fraction */,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  characteristic_speeds_approximate_mhd<ThermodynamicDim>(
      result, rest_mass_density, /*electron_fraction*/ {},
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FUNCTION_INSTANTIATION(r, data)                                     \
  template void CharacteristicSpeedsCompute::function<DIM(data)>(           \
      const gsl::not_null<return_type*> result,                             \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& electron_fraction,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, DIM(data)>&             \
          equation_of_state);

GENERATE_INSTANTIATIONS(FUNCTION_INSTANTIATION, (1, 2, 3))
#undef DIM
#undef FUNCTION_INSTANTIATION

}  // namespace Tags

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template std::array<DataVector, 9>                                           \
  characteristic_speeds_approximate_mhd<GET_DIM(data)>(                        \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_approximate_mhd<GET_DIM(data)>(          \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,             \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template tnsr::i<DataVector, 3> characteristic_speeds_hydro<GET_DIM(data)>(  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_hydro<GET_DIM(data)>(                    \
      const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_mhd<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 9>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state,                                                   \
      SlowMagnetosonicSpeedMethod slow_speed_method);                          \
  template void characteristic_eigenvectors_mhd<GET_DIM(data)>(                \
      const gsl::not_null<tnsr::ij<DataVector, 9>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 9>*> characteristic_projectors, \
      const tnsr::i<DataVector, 9>& characteristic_speeds,                     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void flux_jacobian_hydro<GET_DIM(data)>(                            \
      const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void numerical_characteristics<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,      \
      const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors, \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
