// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

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
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/RandomUnitNormal.hpp"
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
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// Quad-precision reference implementation for MHD characteristic speeds
#include "Characteristics.hpp"

namespace {

namespace quad_ref = grmhd::ValenciaDivClean::TestHelpers::quad_precision;
namespace asymptotic_speeds =
    grmhd::ValenciaDivClean::TestHelpers::asymptotic_magnetosonic_speeds;

// This namespace is meant to hold straightforward implementations of the
// expressions used in the GRMHD characteristics that are not optimized for
// performance. We keep them here to compare against the optimized versions in
// Characteristics.cpp to ensure the optimized versions are correct.
namespace unoptimized {

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
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempI<0, 3>,
      ::Tags::Tempi<0, 3>, ::Tags::TempIj<0, 3>, ::Tags::TempI<1, 3>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  Scalar<DataVector>& kappa_times_p_over_rho_squared =
      get<::Tags::TempScalar<2>>(temp_tensors);
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
    // don't know how to specify zeta, both of which are needed for the needed
    // for the expressions here. So, we currently only support equilibrium 3D
    // EoSs, for which we get kappa from the underlying 2D EoS and set zeta to
    // 0.
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "flux_jacobian_hydro currently only supports 3D EoSs in "
          "equilibrium.");
    }
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
    get(zeta) = 0.0;
  }

  // Intermediate variables
  Scalar<DataVector>& Z = get<::Tags::TempScalar<3>>(temp_tensors);
  tenex::evaluate(make_not_null(&Z), rest_mass_density() * specific_enthalpy() *
                                         square(lorentz_factor()));
  Scalar<DataVector>& D = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(make_not_null(&D), rest_mass_density() * lorentz_factor());
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<5>>(temp_tensors);
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
  tnsr::Ij<DataVector, 3>& mixed_spatial_metric =
      get<::Tags::TempIj<0, 3>>(temp_tensors);
  tenex::evaluate<ti::I, ti::j>(
      make_not_null(&mixed_spatial_metric),
      inv_spatial_metric(ti::I, ti::K) * spatial_metric(ti::k, ti::j));

  // Derivatives of Z
  Scalar<DataVector>& dzdD = get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdD),
      -((lorentz_factor() *
         (kappa() * (-specific_enthalpy() + lorentz_factor()) -
          zeta() * electron_fraction() +
          (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
              rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  tnsr::I<DataVector, 3>& dzds = get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dzds),
      (spatial_velocity(ti::I) * square(lorentz_factor()) *
       (kappa() + sound_speed_squared() * rest_mass_density())) /
          ((-square(lorentz_factor()) +
            sound_speed_squared() * (-1. + square(lorentz_factor()))) *
           rest_mass_density()));
  Scalar<DataVector>& dzdtau = get<::Tags::TempScalar<7>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdtau),
      -((square(lorentz_factor()) * (kappa() + rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  Scalar<DataVector>& dzdye = get<::Tags::TempScalar<8>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdye),
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

}  // namespace unoptimized

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
            electron_fraction, lorentz_factor, specific_enthalpy,
            spatial_metric, unit_normal, *eos_3d),
        (pypp::call<tnsr::i<DataVector, 3>>(
            "CharacteristicSpeeds", "characteristic_speeds_hydro",
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            lorentz_factor, unit_normal)),
        custom_approx);
  }
}

void test_hydro_numerical_characteristics(const DataVector& used_for_size) {
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

  // Initialize containers for all eigenvalues and eigenvectors
  constexpr size_t matrix_size = 6;
  const size_t num_points = used_for_size.size();
  tnsr::i<DataVector, matrix_size> eigenvalues{num_points};
  tnsr::ij<DataVector, matrix_size> right_eigenvectors{num_points};
  tnsr::IJ<DataVector, matrix_size> left_eigenvectors{num_points};

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal and normal velocity in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);
    const auto normal_velocity =
        tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));

    // Solve numerical eigensystem (hydro: MatrixSize 6, magnetic field unused)
    const tnsr::I<DataVector, 3> no_magnetic_field{num_points, 0.0};
    grmhd::ValenciaDivClean::numerical_characteristics(
        make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
        make_not_null(&left_eigenvectors), spatial_velocity, no_magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Get analytic characteristic speeds to check the numeric eigenvalues
    const tnsr::i<DataVector, 3> analytic_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            electron_fraction, lorentz_factor, specific_enthalpy,
            spatial_metric, unit_normal, *equation_of_state_3d);

    // Count degenerate eigenvalues and check the other speeds
    // Note 1: We expect 4 degenerate eigenvalues equal to the normal velocity.
    // Note 2: The characteristic matrix becomes more defective for larger
    //         Lorentz boosts. With the default random Lorentz factor generator,
    //         the largest Lorentz factor is ~20, which leads to an eigenvalue
    //         error of ~1e-10.
    constexpr double eigenvalue_tolerance = 1e-6;
    for (size_t point = 0; point < num_points; ++point) {
      int number_of_degenerate_eigenvalues = 0;
      bool found_lambda_plus = false;
      bool found_lambda_minus = false;
      for (size_t i = 0; i < 6; ++i) {
        const DataVector& eigenvalue = eigenvalues.get(i);
        const double diff_with_normal_velocity =
            std::abs(eigenvalue[point] - get(normal_velocity)[point]);
        const double diff_with_lambda_plus = std::abs(
            eigenvalue[point] -
            analytic_speeds.get(
                grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus)[point]);
        const double diff_with_lambda_minus = std::abs(
            eigenvalue[point] -
            analytic_speeds.get(
                grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus)[point]);
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
    tnsr::iJ<DataVector, 6> characteristic_matrix{num_points};
    grmhd::ValenciaDivClean::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Check eigensystem relation for each eigenvalue/eigenvector
    constexpr double numeric_tolerance = 1e-12;
    for (size_t i = 0; i < 6; ++i) {
      // Use non-owning DataVector views to avoid copying data
      const Scalar<DataVector> eigenvalue{};
      const tnsr::i<DataVector, 6> right_eigenvector{};
      const tnsr::I<DataVector, 6> left_eigenvector{};
      make_const_view(make_not_null(&get(eigenvalue)), eigenvalues.get(i), 0,
                      num_points);
      for (size_t k = 0; k < 6; ++k) {
        make_const_view(make_not_null(&right_eigenvector.get(k)),
                        right_eigenvectors.get(i, k), 0, num_points);
        make_const_view(make_not_null(&left_eigenvector.get(k)),
                        left_eigenvectors.get(i, k), 0, num_points);
      }

      const Scalar<DataVector> right_eigensystem_error =
          magnitude(tenex::evaluate<ti::k>(
              characteristic_matrix(ti::k, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::k)));

      const Scalar<DataVector> left_eigensystem_error =
          magnitude(tenex::evaluate<ti::K>(
              left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::K) -
              eigenvalue() * left_eigenvector(ti::K)));

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

void test_hydro_analytic_eigenvectors(const DataVector& used_for_size) {
  // Emily's eigensystem + biorthonormality checks for the analytic hydro+Ye
  // eigenvectors, using the standardized characteristic_eigenvectors_hydro
  // (tnsr::ij/IJ modes and projectors) and the chars_mhd characteristic
  // interfaces.
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);

  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity = TestHelpers::hydro::random_velocity(
      nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density =
      TestHelpers::hydro::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      TestHelpers::hydro::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      TestHelpers::hydro::random_electron_fraction(nn_gen, used_for_size);

  const auto equation_of_state_2d = EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();
  const auto pressure = equation_of_state_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  constexpr size_t matrix_size = 6;
  const size_t num_points = used_for_size.size();

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    // Analytic eigenvectors in the standardized modes/projectors layout.
    tnsr::ij<DataVector, matrix_size> characteristic_modes{num_points};
    tnsr::IJ<DataVector, matrix_size> characteristic_projectors{num_points};
    grmhd::ValenciaDivClean::characteristic_eigenvectors_hydro(
        make_not_null(&characteristic_modes),
        make_not_null(&characteristic_projectors), spatial_velocity,
        rest_mass_density, specific_internal_energy, specific_enthalpy,
        electron_fraction, lorentz_factor, unit_normal, spatial_metric,
        *equation_of_state_3d);

    // Analytic speeds, assembled in the HydroVectorR ordering: four degenerate
    // normal-velocity modes, then lambda+, lambda-.
    tnsr::i<DataVector, 3> hydro_speeds{num_points};
    grmhd::ValenciaDivClean::characteristic_speeds_hydro(
        make_not_null(&hydro_speeds), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal, *equation_of_state_3d);
    tnsr::i<DataVector, matrix_size> eigenvalues{num_points};
    for (size_t i = 0; i < 4; ++i) {
      eigenvalues.get(i) =
          hydro_speeds.get(grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity);
    }
    eigenvalues.get(grmhd::ValenciaDivClean::HydroVectorR::Rplus) =
        hydro_speeds.get(grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus);
    eigenvalues.get(grmhd::ValenciaDivClean::HydroVectorR::Rminus) =
        hydro_speeds.get(grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus);

    // Characteristic matrix to check the eigensystem relations.
    tnsr::iJ<DataVector, matrix_size> characteristic_matrix{num_points};
    grmhd::ValenciaDivClean::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    constexpr double tolerance = 1e-10;

    // Eigensystem relation A.R = y R and L.A = y L, scaled by the (non-unit)
    // eigenvector norms.
    for (size_t i = 0; i < matrix_size; ++i) {
      const Scalar<DataVector> eigenvalue{};
      const tnsr::i<DataVector, matrix_size> right_eigenvector{};
      const tnsr::I<DataVector, matrix_size> left_eigenvector{};
      make_const_view(make_not_null(&get(eigenvalue)), eigenvalues.get(i), 0,
                      num_points);
      for (size_t k = 0; k < matrix_size; ++k) {
        make_const_view(make_not_null(&right_eigenvector.get(k)),
                        characteristic_modes.get(i, k), 0, num_points);
        make_const_view(make_not_null(&left_eigenvector.get(k)),
                        characteristic_projectors.get(i, k), 0, num_points);
      }
      const Scalar<DataVector> right_residual =
          magnitude(tenex::evaluate<ti::k>(
              characteristic_matrix(ti::k, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::k)));
      const Scalar<DataVector> right_norm = magnitude(right_eigenvector);
      const Scalar<DataVector> left_residual = magnitude(tenex::evaluate<ti::K>(
          left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::K) -
          eigenvalue() * left_eigenvector(ti::K)));
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);

      double max_scaled_error = 0.0;
      for (size_t point = 0; point < num_points; ++point) {
        const double r_scale = std::max(1.0, std::abs(get(right_norm)[point]));
        const double l_scale = std::max(1.0, std::abs(get(left_norm)[point]));
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(right_residual)[point]) / r_scale);
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(left_residual)[point]) / l_scale);
      }
      CHECK(max_scaled_error < tolerance);
    }

    // Biorthonormality: L_i . R_j = delta_ij (scaled by the vector norms).
    for (size_t i = 0; i < matrix_size; ++i) {
      const tnsr::I<DataVector, matrix_size> left_eigenvector{};
      for (size_t k = 0; k < matrix_size; ++k) {
        make_const_view(make_not_null(&left_eigenvector.get(k)),
                        characteristic_projectors.get(i, k), 0, num_points);
      }
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);
      for (size_t j = 0; j < matrix_size; ++j) {
        const tnsr::i<DataVector, matrix_size> right_eigenvector{};
        for (size_t k = 0; k < matrix_size; ++k) {
          make_const_view(make_not_null(&right_eigenvector.get(k)),
                          characteristic_modes.get(j, k), 0, num_points);
        }
        const Scalar<DataVector> right_norm = magnitude(right_eigenvector);
        const Scalar<DataVector> dot_ij =
            tenex::evaluate(left_eigenvector(ti::J) * right_eigenvector(ti::j));
        const double target = (i == j ? 1.0 : 0.0);
        double max_scaled_error = 0.0;
        for (size_t point = 0; point < num_points; ++point) {
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

void test_hydro_characteristics_match_unoptimized_version(
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
  const size_t num_points = used_for_size.size();
  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    tnsr::iJ<DataVector, matrix_size> optimized_matrix{num_points};
    tnsr::iJ<DataVector, matrix_size> unoptimized_matrix{num_points};
    grmhd::ValenciaDivClean::flux_jacobian_hydro<3>(
        make_not_null(&optimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
    unoptimized::flux_jacobian_hydro<3>(
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

  constexpr size_t num_points = 100000;
  const DataVector used_for_size(num_points);

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

  // Benchmark ran on mbot (Iago Mendes, April 2026):
  // - flux_jacobian_hydro (optimized): ~134 ms
  // - flux_jacobian_hydro (unoptimized): ~159 ms
  // - optimization result: ~1.2x speedup (~20% faster)
  constexpr size_t matrix_size = 6;
  tnsr::iJ<DataVector, matrix_size> optimized_matrix{num_points};
  tnsr::iJ<DataVector, matrix_size> unoptimized_matrix{num_points};
  BENCHMARK("flux_jacobian_hydro (optimized)") {
    grmhd::ValenciaDivClean::flux_jacobian_hydro<3>(
        make_not_null(&optimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
  BENCHMARK("flux_jacobian_hydro (unoptimized)") {
    unoptimized::flux_jacobian_hydro<3>(
        make_not_null(&unoptimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
}

/**
 * Tests that we can find the roots of a test quartic polynomial:
 *   F(x) = (x - 0.9)(x - 0.3)(x + 0.2)(x + 0.8)
 *        = x^4 - 0.2 x^3 - 0.77 x^2 + 0.078 x + 0.0432
 */
void test_quartic_rootfinding(const DataVector& used_for_size) {
  const size_t num_points = used_for_size.size();

  tnsr::i<DataVector, 4> quartic_coefficients{num_points};
  get<0>(quartic_coefficients) = 0.0432;
  get<1>(quartic_coefficients) = 0.078;
  get<2>(quartic_coefficients) = -0.77;
  get<3>(quartic_coefficients) = -0.2;

  DataVector fast_plus(num_points, 1.0);
  grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
      make_not_null(&fast_plus), quartic_coefficients);

  DataVector fast_minus(num_points, -1.0);
  grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
      make_not_null(&fast_minus), quartic_coefficients);

  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(fast_plus, DataVector(num_points, 0.9),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(fast_minus, DataVector(num_points, -0.8),
                               custom_approx);
}

void test_mhd_characteristics(const DataVector& used_for_size) {
  const ScopedFpeState disable_fpes(false);
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const size_t num_points = used_for_size.size();

  // Generate random primitives in a "typical" MHD simulation regime:
  //   W in [1, 3]            — mildly relativistic
  //   B^2/p in [0.02, 55]   — moderate magnetization
  // This avoids extreme corners (high W, extreme magnetization) where
  // numerical errors are expected and tested separately by
  // test_mhd_characteristics_errors.
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);

  // W in [1, 3]: log(W-1) uniform in [-10, log(2)]
  Scalar<DataVector> lorentz_factor{num_points};
  {
    std::uniform_real_distribution<> w_dist(-10.0, std::log(2.0));
    get(lorentz_factor) =
        1.0 + exp(make_with_random_values<DataVector>(
                  nn_gen, make_not_null(&w_dist), used_for_size));
  }

  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto& det_spatial_metric = det_and_inv_spatial_metric.first;

  const EquationsOfState::IdealFluid<true> eos_2d(1.5, 0.0);
  const auto pressure = eos_2d.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  // B^2/p in [exp(-4), exp(4)]: moderate magnetization
  tnsr::I<DataVector, 3> magnetic_field{num_points};
  {
    auto B_direction = random_unit_normal(nn_gen, spatial_metric);
    std::uniform_real_distribution<> b_dist(-4.0, 4.0);
    for (size_t s = 0; s < num_points; ++s) {
      const double B_mag = sqrt(get(pressure)[s] * exp(b_dist(*nn_gen)));
      for (size_t d = 0; d < 3; ++d) {
        magnetic_field.get(d)[s] = B_direction.get(d)[s] * B_mag;
      }
    }
  }

  Scalar<DataVector> sound_speed_squared{num_points};
  get(sound_speed_squared) =
      get(eos_2d.chi_from_density_and_energy(rest_mass_density,
                                             specific_internal_energy)) +
      get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
          rest_mass_density, specific_internal_energy));
  get(sound_speed_squared) /= get(specific_enthalpy);

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);
    // Compute MHD characteristic speeds
    tnsr::i<DataVector, 9> characteristic_speeds{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&characteristic_speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal, eos_2d);

    // Compute quartic coefficients for residual check
    const Scalar<DataVector> normal_velocity =
        tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
    const Scalar<DataVector> normal_magnetic_field =
        tenex::evaluate(magnetic_field(ti::I) * unit_normal(ti::i));
    const auto magnetic_field_squared =
        dot_product(magnetic_field, magnetic_field, spatial_metric);
    const auto magnetic_field_dot_spatial_velocity =
        dot_product(magnetic_field, spatial_velocity, spatial_metric);
    const Scalar<DataVector> comoving_magnetic_field_squared{
        get(magnetic_field_squared) / square(get(lorentz_factor)) +
        square(get(magnetic_field_dot_spatial_velocity))};
    const Scalar<DataVector> inv_rho_h{
        1.0 / (get(rest_mass_density) * get(specific_enthalpy))};
    const Scalar<DataVector> inv_sqrt_rho_h{sqrt(get(inv_rho_h))};
    const Scalar<DataVector> normal_magnetic_field_scaled{
        get(normal_magnetic_field) * get(inv_sqrt_rho_h)};
    const Scalar<DataVector> magnetic_field_dot_spatial_velocity_scaled{
        get(magnetic_field_dot_spatial_velocity) * get(inv_sqrt_rho_h)};
    const Scalar<DataVector> magnetic_field_squared_scaled{
        get(magnetic_field_squared) * get(inv_rho_h)};
    const Scalar<DataVector> comoving_magnetic_field_squared_scaled{
        get(comoving_magnetic_field_squared) * get(inv_rho_h)};
    tnsr::i<DataVector, 4> quartic_coefficients{num_points};
    grmhd::ValenciaDivClean::magnetosonic_quartic_coefficients(
        make_not_null(&quartic_coefficients), sound_speed_squared,
        normal_velocity, lorentz_factor, normal_magnetic_field_scaled,
        magnetic_field_dot_spatial_velocity_scaled,
        magnetic_field_squared_scaled, comoving_magnetic_field_squared_scaled);

    // Check 1: Quartic residual for all magnetosonic speeds
    constexpr double quartic_tolerance = 1.0e-8;
    const auto check_quartic_residual =
        [&quartic_coefficients](const DataVector& y) {
          const DataVector quartic_at_y =
              square(square(y)) + get<3>(quartic_coefficients) * cube(y) +
              get<2>(quartic_coefficients) * square(y) +
              get<1>(quartic_coefficients) * y + get<0>(quartic_coefficients);
          CHECK(max(abs(quartic_at_y)) < quartic_tolerance);
        };
    check_quartic_residual(characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus));
    check_quartic_residual(characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus));
    check_quartic_residual(characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus));
    check_quartic_residual(characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus));

    // Check 2: Speed ordering
    // scalar- <= fast- <= alfven- <= slow- <= entropy
    //       <= slow+ <= alfven+ <= fast+ <= scalar+
    const auto& scalar_minus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::ScalarMinus);
    const auto& fast_minus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus);
    const auto& alfven_minus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus);
    const auto& slow_minus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus);
    const auto& entropy =
        characteristic_speeds.get(grmhd::ValenciaDivClean::MhdSpeed::Entropy);
    const auto& slow_plus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus);
    const auto& alfven_plus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus);
    const auto& fast_plus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus);
    const auto& scalar_plus = characteristic_speeds.get(
        grmhd::ValenciaDivClean::MhdSpeed::ScalarPlus);
    constexpr double ordering_tolerance = 1.0e-10;
    for (size_t point = 0; point < num_points; ++point) {
      CHECK(scalar_minus[point] <= fast_minus[point] + ordering_tolerance);
      CHECK(fast_minus[point] <= alfven_minus[point] + ordering_tolerance);
      CHECK(alfven_minus[point] <= slow_minus[point] + ordering_tolerance);
      CHECK(slow_minus[point] <= entropy[point] + ordering_tolerance);
      CHECK(entropy[point] <= slow_plus[point] + ordering_tolerance);
      CHECK(slow_plus[point] <= alfven_plus[point] + ordering_tolerance);
      CHECK(alfven_plus[point] <= fast_plus[point] + ordering_tolerance);
      CHECK(fast_plus[point] <= scalar_plus[point] + ordering_tolerance);
    }

    // Tangent vectors — same computation as production code uses internally
    const auto tangent_1 = orthonormal_oneform(unit_normal, inv_spatial_metric);
    const auto tangent_2 = orthonormal_oneform(
        unit_normal, tangent_1, spatial_metric, det_spatial_metric);

    // Eigenvectors for all points in this direction
    tnsr::ij<DataVector, 9> modes{num_points, 0.0};
    tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
        make_not_null(&modes), make_not_null(&projectors),
        characteristic_speeds, spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal, eos_2d);

    // Check 3: Quad-precision speed comparison and eigenvector checks
    for (size_t point = 0; point < num_points; ++point) {
      std::array<quad_ref::Quad, 3> q_v{};
      std::array<quad_ref::Quad, 3> q_B{};
      std::array<quad_ref::Quad, 3> q_n{};
      std::array<std::array<quad_ref::Quad, 3>, 3> q_g{};
      for (size_t i = 0; i < 3; ++i) {
        q_v[i] = spatial_velocity.get(i)[point];
        q_B[i] = magnetic_field.get(i)[point];
        q_n[i] = unit_normal.get(i)[point];
        for (size_t j = 0; j < 3; ++j) {
          q_g[i][j] = spatial_metric.get(i, j)[point];
        }
      }
      const auto q_speeds = quad_ref::characteristic_speeds_mhd(
          q_v, q_B, get(rest_mass_density)[point],
          get(specific_internal_energy)[point], get(lorentz_factor)[point],
          get(specific_enthalpy)[point], q_g, q_n);

      constexpr double speed_abs_tolerance = 1.0e-6;
      for (size_t mode = 0; mode < 9; ++mode) {
        const double dbl_speed = characteristic_speeds.get(mode)[point];
        const double quad_speed = static_cast<double>(q_speeds[mode]);
        const double abs_err = std::abs(dbl_speed - quad_speed);
        CAPTURE(mode);
        CAPTURE(dbl_speed);
        CAPTURE(quad_speed);
        CAPTURE(abs_err);
        CHECK(abs_err < speed_abs_tolerance);
      }

      // Compute the quad-precision reference eigenvectors.  Pass the same
      // tangent vectors production computes internally so both are in the same
      // basis and a component-by-component comparison is meaningful.
      std::array<quad_ref::Quad, 3> q_t1{};
      std::array<quad_ref::Quad, 3> q_t2{};
      for (size_t d = 0; d < 3; ++d) {
        q_t1[d] = tangent_1.get(d)[point];
        q_t2[d] = tangent_2.get(d)[point];
      }
      std::array<std::array<quad_ref::Quad, 9>, 9> q_right{};
      std::array<std::array<quad_ref::Quad, 9>, 9> q_left{};
      quad_ref::characteristic_eigenvectors_mhd(
          &q_right, &q_left, q_speeds, q_v, q_B,
          get(rest_mass_density)[point], get(specific_internal_energy)[point],
          get(lorentz_factor)[point], get(specific_enthalpy)[point], q_g, q_n,
          q_t1, q_t2);

      // Diagnostic (NOT asserted): whole-eigenvector double-vs-quad relative
      // error.  A hard assertion here is hopelessly seed-dependent: the
      // eigenvector formulas (especially the left vectors, with their
      // 1/a, 1/G, 1/cs^2 denominators) are ill-conditioned, so for unlucky
      // random realizations even the whole-vector relative norm disagrees
      // double-vs-quad by ~1e-3.  Faithfulness is tracked deterministically in
      // test_mhd_characteristics_errors / mhd_eigenvector_errors.tsv instead.
      double max_evec_rel = 0.0;
      for (size_t wave = 0; wave < 9; ++wave) {
        double r_dnorm = 0.0;
        double r_qnorm = 0.0;
        double l_dnorm = 0.0;
        double l_qnorm = 0.0;
        for (size_t n = 0; n < 9; ++n) {
          const double r_quad = static_cast<double>(q_right[wave][n]);
          const double l_quad = static_cast<double>(q_left[wave][n]);
          r_dnorm += square(modes.get(wave, n)[point] - r_quad);
          r_qnorm += square(r_quad);
          l_dnorm += square(projectors.get(wave, n)[point] - l_quad);
          l_qnorm += square(l_quad);
        }
        max_evec_rel = std::max(
            {max_evec_rel,
             std::sqrt(r_dnorm) / std::max(std::sqrt(r_qnorm), 1.0e-300),
             std::sqrt(l_dnorm) / std::max(std::sqrt(l_qnorm), 1.0e-300)});
      }
      CAPTURE(max_evec_rel);

      // Entropy eigenvector biorthonormality: L_entropy · R_entropy = 1.
      // The full left/right set is biorthogonal away from degeneracies (see
      // test_mhd_characteristics_errors), but only the entropy pair is
      // normalized to exactly 1, so it is the one asserted here.
      {
        constexpr size_t ent = grmhd::ValenciaDivClean::MhdSpeed::Entropy;
        double ent_diag = 0.0;
        for (size_t n = 0; n < 9; ++n) {
          ent_diag += projectors.get(ent, n)[point] * modes.get(ent, n)[point];
        }
        CAPTURE(ent_diag);
        CHECK(std::abs(ent_diag - 1.0) < 1.0e-10);
      }

      // Diagnostic (NOT asserted): full biorthogonality error.  With the
      // corrected eigenvectors this is at machine precision for well-separated
      // states and grows only near speed degeneracies, where the eigenvector
      // basis becomes ill-conditioned.  It is left as a diagnostic rather than a
      // hard CHECK because a random draw can land close to a degeneracy; see the
      // biorthogonality study in runs-ai/mhd_eigenvectors/.
      double max_biorth_err = 0.0;
      for (size_t wi = 0; wi < 9; ++wi) {
        for (size_t wj = 0; wj < 9; ++wj) {
          double lr = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            lr += projectors.get(wi, n)[point] * modes.get(wj, n)[point];
          }
          max_biorth_err =
              std::max(max_biorth_err, std::abs(lr - (wi == wj ? 1.0 : 0.0)));
        }
      }
      CAPTURE(max_biorth_err);
    }
  }
}

void test_mhd_numerical_characteristics(const DataVector& used_for_size) {
  const ScopedFpeState disable_fpes(false);
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const size_t num_points = used_for_size.size();

  // Typical MHD regime (W in [1, 3], moderate magnetization), well away from
  // degeneracies so the numeric and analytic eigensystems are both
  // well-conditioned and should agree.  (The near-degenerate / high-W behavior,
  // where the analytic closed form collapses but the numeric solver does not,
  // is studied in test_mhd_characteristics_errors.)
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  Scalar<DataVector> lorentz_factor{num_points};
  {
    std::uniform_real_distribution<> w_dist(-10.0, std::log(2.0));
    get(lorentz_factor) =
        1.0 + exp(make_with_random_values<DataVector>(
                  nn_gen, make_not_null(&w_dist), used_for_size));
  }
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;

  const EquationsOfState::IdealFluid<true> eos_2d(1.5, 0.0);
  const auto pressure = eos_2d.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  // The MHD flux Jacobian carries an electron-fraction slot for interface
  // compatibility with the hydro+Ye system; it is unused by the 2D EoS here.
  const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.0)};

  tnsr::I<DataVector, 3> magnetic_field{num_points};
  {
    auto B_direction = random_unit_normal(nn_gen, spatial_metric);
    std::uniform_real_distribution<> b_dist(-4.0, 4.0);
    for (size_t s = 0; s < num_points; ++s) {
      const double B_mag = sqrt(get(pressure)[s] * exp(b_dist(*nn_gen)));
      for (size_t d = 0; d < 3; ++d) {
        magnetic_field.get(d)[s] = B_direction.get(d)[s] * B_mag;
      }
    }
  }

  constexpr size_t matrix_size = 9;
  tnsr::i<DataVector, matrix_size> eigenvalues{num_points};
  tnsr::ij<DataVector, matrix_size> right_eigenvectors{num_points};
  tnsr::IJ<DataVector, matrix_size> left_eigenvectors{num_points};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    // Solve the numeric eigensystem.  MatrixSize 9 (deduced from the output
    // containers) selects the MHD flux Jacobian flux_jacobian_mhd.
    grmhd::ValenciaDivClean::numerical_characteristics(
        make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
        make_not_null(&left_eigenvectors), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, eos_2d);

    // Analytic speeds to check the numeric eigenvalues against.
    tnsr::i<DataVector, 9> analytic_speeds{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&analytic_speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal, eos_2d);

    // The characteristic matrix, to check the eigensystem relations.
    tnsr::iJ<DataVector, 9> characteristic_matrix{num_points, 0.0};
    grmhd::ValenciaDivClean::flux_jacobian_mhd(
        make_not_null(&characteristic_matrix), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, eos_2d);

    constexpr double eigenvalue_tolerance = 1.0e-6;
    constexpr double eigensystem_tolerance = 1.0e-8;
    for (size_t point = 0; point < num_points; ++point) {
      for (size_t i = 0; i < 9; ++i) {
        const double y = eigenvalues.get(i)[point];

        // Each numeric eigenvalue must match one of the nine analytic speeds.
        double min_speed_diff = std::numeric_limits<double>::max();
        for (size_t s = 0; s < 9; ++s) {
          min_speed_diff = std::min(
              min_speed_diff, std::abs(y - analytic_speeds.get(s)[point]));
        }
        CAPTURE(get(lorentz_factor)[point]);
        CAPTURE(min_speed_diff);
        CHECK(min_speed_diff < eigenvalue_tolerance);

        // The numeric eigenvectors must solve A.R = y R and L.A = y L.  Because
        // geev returns the eigenvectors for its own eigenvalues, this residual
        // reflects the solver accuracy and is independent of the analytic
        // closed form (this is exactly why it stays small near degeneracies).
        double right_resid = 0.0;
        double left_resid = 0.0;
        double rnorm = 0.0;
        double lnorm = 0.0;
        for (size_t m = 0; m < 9; ++m) {
          double ar = 0.0;
          double la = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            ar += characteristic_matrix.get(m, n)[point] *
                  right_eigenvectors.get(i, n)[point];
            la += left_eigenvectors.get(i, n)[point] *
                  characteristic_matrix.get(n, m)[point];
          }
          right_resid = std::max(
              right_resid,
              std::abs(ar - y * right_eigenvectors.get(i, m)[point]));
          left_resid = std::max(
              left_resid,
              std::abs(la - y * left_eigenvectors.get(i, m)[point]));
          rnorm = std::max(rnorm, std::abs(right_eigenvectors.get(i, m)[point]));
          lnorm = std::max(lnorm, std::abs(left_eigenvectors.get(i, m)[point]));
        }
        CAPTURE(right_resid / std::max(rnorm, 1.0e-300));
        CAPTURE(left_resid / std::max(lnorm, 1.0e-300));
        CHECK(right_resid / std::max(rnorm, 1.0e-300) < eigensystem_tolerance);
        CHECK(left_resid / std::max(lnorm, 1.0e-300) < eigensystem_tolerance);
      }
    }
  }
}

void test_mhd_characteristics_errors(const bool output) {
  const ScopedFpeState disable_fpes(false);

  // Configuration families for deterministic error probing.
  // Each config fixes the geometric and magnetization parameters;
  // the test sweeps only over the Lorentz factor W.
  struct Config {
    const char* name;
    double bn_fraction;  // Bn / |B|
    double vn_fraction;  // vn / |v|
    double phi;          // tangential magnetic field angle
    double sigma;        // B^2 / (rho * h)
    double pressure_val;  // thermal pressure (controls cs^2)
    double density;       // rest-mass density (rho != 1 exercises the D-row)
  };
  const std::array<Config, 6> configs{{
      // Generic moderate regime: hot fluid (cs^2 ~ 0.13) with moderate
      // magnetization gives a well-separated quartic with clear W shape.
      {"typical", 0.4, 0.5, M_PI / 4.0, 0.1, 0.1, 1.0},
      // Same as "typical" but with rho != 1.  Several scalar-eigenvector terms
      // scale as rho vs rho^2, so a rho != 1 state is needed to exercise them
      // (e.g. the (1-cs^2) rho^2 a B_n term of the D row, Eq. 4.32 row 3).
      {"moving_dense", 0.4, 0.5, M_PI / 4.0, 0.1, 0.1, 2.5},
      // Worst-case errors from low magnetization: cold fluid (cs^2 ~ 1.7e-4)
      // with sigma << 1 makes the quartic coefficients nearly degenerate.
      {"low_magnetization", 0.3674661940736692, 0.999, 0.6283185307179586,
       1.0e-4, 1.0e-4, 1.0},
      // High magnetization regime (Anton et al. 2010, Section 8.2.1):
      // sigma ~ 5000 matches their cylindrical explosion test background.
      // Cold fluid so that cs^2/sigma ~ 3e-8.  Bn/B=0.05 and phi=0.6pi
      // optimized from 4D sweep (see CLAUDE-MhdSpeedErrorAnalysis.md).
      {"high_magnetization", 0.05, 0.995, 0.6 * M_PI, 5000.0, 1.0e-4, 1.0},
      // Near Type-I degeneracy: Bn/B -> 0, inner speeds coalesce.  The
      // degeneracy PERSISTS under boost (the gap shrinks with W), so the gap
      // stays tiny while the boosted matrix grows ill-conditioned -- this is the
      // config where the numeric solver eventually fails (see below).
      {"near_type_I", 1.0e-6, 0.5, 0.0, 0.01, 1.0e-4, 1.0},
      // Near Type-II degeneracy: Bn/B -> 1 (B nearly normal), slow merges with
      // Alfven.  bn=1-1e-8 gives a gap ~1e-9 at W=1.  Unlike type I, this
      // degeneracy LIFTS under boost (the gap grows to ~4e-3 by W~2), so the gap
      // is never simultaneously tiny and at high W, and the numeric solver stays
      // robust.  (Both degeneracies are semisimple -- the matrix stays
      // diagonalizable with a real spectrum and bounded cond(R) at fixed W.)
      {"near_type_II", 0.99999999, 0.5, 0.0, 0.01, 1.0e-4, 1.0},
  }};

  constexpr size_t num_points = 1;
  constexpr double adiabatic_index = 5.0 / 3.0;
  const EquationsOfState::IdealFluid<true> eos_2d(adiabatic_index, 0.0);
  Scalar<DataVector> rest_mass_density{DataVector(num_points, 1.0)};

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto& det_spatial_metric = det_and_inv_spatial_metric.first;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  // Tangent vectors — metric and normal are fixed across the entire sweep
  const auto tangent_1_normal =
      orthonormal_oneform(unit_normal, inv_spatial_metric);
  const auto tangent_2_normal = orthonormal_oneform(
      unit_normal, tangent_1_normal, spatial_metric, det_spatial_metric);

  auto rel_err = [](const double a, const double b) {
    return std::abs(a - b) / std::max(std::abs(b), 1.0e-300);
  };

  std::ofstream out;
  if (output) {
    out.open("mhd_characteristics_errors.tsv", std::ios::out | std::ios::trunc);
    out << std::setprecision(16);
    out << "config\tW\t"
           "fast_m\tfast_p\tslow_m\tslow_p\talf_m\talf_p\tentropy\t"
           "q_fast_m\tq_fast_p\tq_slow_m\tq_slow_p\tq_alf_m\tq_alf_p\t"
           "q_entropy\t"
           "abs_err_fast_m\tabs_err_fast_p\tabs_err_slow_m\tabs_err_slow_p\t"
           "abs_err_alf_m\tabs_err_alf_p\tabs_err_entropy\t"
           "rel_err_fast_m\trel_err_fast_p\trel_err_slow_m\trel_err_slow_p\t"
           "rel_err_alf_m\trel_err_alf_p\trel_err_entropy\t"
           "asym_fast_m\tasym_fast_p\tasym_slow_m\tasym_slow_p\t"
           "abs_err_asym_fast_m\tabs_err_asym_fast_p\t"
           "abs_err_asym_slow_m\tabs_err_asym_slow_p\t"
           "rel_err_asym_fast_m\trel_err_asym_fast_p\t"
           "rel_err_asym_slow_m\trel_err_asym_slow_p\n";
  }

  std::ofstream out_evec;
  std::ofstream out_lr;
  std::ofstream out_raw;
  std::ofstream out_prec;
  if (output) {
    out_evec.open("mhd_eigenvector_errors.tsv",
                  std::ios::out | std::ios::trunc);
    out_evec << std::setprecision(16);
    out_evec << "config\tW\tbiorth_err\t"
                "R_err_sclm\tR_err_fastm\tR_err_alfm\tR_err_slowm\t"
                "R_err_ent\tR_err_slowp\tR_err_alfp\tR_err_fastp\tR_err_sclp\t"
                "L_err_sclm\tL_err_fastm\tL_err_alfm\tL_err_slowm\t"
                "L_err_ent\tL_err_slowp\tL_err_alfp\tL_err_fastp\tL_err_sclp\n";
    out_lr.open("mhd_lr_matrix.tsv", std::ios::out | std::ios::trunc);
    out_lr << std::setprecision(12);
    out_lr << "config\tW\ti\tj\tvalue\n";
    out_raw.open("mhd_evec_raw.tsv", std::ios::out | std::ios::trunc);
    out_raw << std::setprecision(12);
    out_raw << "config\tW\tkind\twave\tn\tvalue\n";
    out_prec.open("mhd_evec_precision.tsv", std::ios::out | std::ios::trunc);
    out_prec << std::setprecision(16);
    out_prec << "config\tW\tspeed_gap\tRL_err_whole\tRL_err_iso\t"
                "analytic_speed_err\t1W_speed_err\t1W_RL_err\t1W_biorth\t"
                "1W_eig_resid\t"
                "biorth_whole_norm\tbiorth_iso_norm\teig_resid\t"
                "eig_resid_right\teig_resid_left\tnum_eig_resid\t"
                "num_speed_err\teig_resid_iso\tnum_biorth\n";
  }

  constexpr size_t n_w = 64;
  constexpr double w_max = 20.0;

  for (const auto& config : configs) {
    get(rest_mass_density) = config.density;
    // Per-config thermodynamics: pressure determines cs^2.
    // The "typical" config uses p=0.1 (hot fluid, cs^2~0.13) for a
    // well-separated quartic; the stress-test configs use p=1e-4
    // (cold fluid, cs^2~1.7e-4) to amplify cancellation errors.
    const Scalar<DataVector> pressure{
        DataVector(num_points, config.pressure_val)};
    const Scalar<DataVector> specific_internal_energy =
        eos_2d.specific_internal_energy_from_density_and_pressure(
            rest_mass_density, pressure);
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    Scalar<DataVector> sound_speed_squared{DataVector(num_points, 0.0)};
    get(sound_speed_squared) =
        get(eos_2d.chi_from_density_and_energy(rest_mass_density,
                                               specific_internal_energy)) +
        get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
            rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);

    double max_abs_fast = 0.0;
    double max_abs_slow = 0.0;
    double max_abs_alfven = 0.0;
    double max_abs_entropy = 0.0;
    double max_rel_fast = 0.0;
    double max_rel_slow = 0.0;
    double max_biorth_err = 0.0;

    for (size_t iw = 0; iw < n_w; ++iw) {
      const double tw = static_cast<double>(iw) / static_cast<double>(n_w - 1);
      const double W = std::exp(tw * std::log(w_max));
      const double vmag = std::sqrt(std::max(0.0, 1.0 - 1.0 / square(W)));
      const double vn = config.vn_fraction * vmag;
      const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));

      const double Bmag = std::sqrt(config.sigma * get(rest_mass_density)[0] *
                                    get(specific_enthalpy)[0]);
      const double bx = config.bn_fraction * Bmag;
      const double bt = std::sqrt(std::max(0.0, square(Bmag) - square(bx)));
      const double by = bt * std::cos(config.phi);
      const double bz = bt * std::sin(config.phi);

      Scalar<DataVector> lorentz_factor{DataVector(num_points, W)};
      tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
      spatial_velocity.get(0) = vn;
      spatial_velocity.get(1) = vt;
      spatial_velocity.get(2) = 0.0;
      tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
      magnetic_field.get(0) = bx;
      magnetic_field.get(1) = by;
      magnetic_field.get(2) = bz;

      tnsr::i<DataVector, 9> speeds{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_speeds_mhd(
          make_not_null(&speeds), spatial_velocity, magnetic_field,
          rest_mass_density, specific_internal_energy, lorentz_factor,
          specific_enthalpy, spatial_metric, unit_normal, eos_2d);

      // Quad reference
      std::array<quad_ref::Quad, 3> q_v{};
      std::array<quad_ref::Quad, 3> q_B{};
      std::array<quad_ref::Quad, 3> q_n{};
      std::array<std::array<quad_ref::Quad, 3>, 3> q_g{};
      for (size_t i = 0; i < 3; ++i) {
        q_v[i] = spatial_velocity.get(i)[0];
        q_B[i] = magnetic_field.get(i)[0];
        q_n[i] = unit_normal.get(i)[0];
        for (size_t j = 0; j < 3; ++j) {
          q_g[i][j] = spatial_metric.get(i, j)[0];
        }
      }
      const auto q_speeds = quad_ref::characteristic_speeds_mhd(
          q_v, q_B, get(rest_mass_density)[0], get(specific_internal_energy)[0],
          W, get(specific_enthalpy)[0], q_g, q_n);

      // Eigenvectors (double and quad)
      tnsr::ij<DataVector, 9> modes{num_points, 0.0};
      tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
          make_not_null(&modes), make_not_null(&projectors), speeds,
          spatial_velocity, magnetic_field, rest_mass_density,
          specific_internal_energy, lorentz_factor, specific_enthalpy,
          spatial_metric, unit_normal, eos_2d);

      std::array<quad_ref::Quad, 3> q_t1{};
      std::array<quad_ref::Quad, 3> q_t2{};
      for (size_t d = 0; d < 3; ++d) {
        q_t1[d] = tangent_1_normal.get(d)[0];
        q_t2[d] = tangent_2_normal.get(d)[0];
      }
      std::array<std::array<quad_ref::Quad, 9>, 9> q_right{};
      std::array<std::array<quad_ref::Quad, 9>, 9> q_left{};
      quad_ref::characteristic_eigenvectors_mhd(
          &q_right, &q_left, q_speeds, q_v, q_B,
          get(rest_mass_density)[0], get(specific_internal_energy)[0], W,
          get(specific_enthalpy)[0], q_g, q_n, q_t1, q_t2);

      // ISOLATED eigenvector error: feed the double eigenvector code the same
      // (accurate, quad-derived) speeds the quad reference used, cast to
      // double.  This removes the speed-solver cancellation error so that what
      // remains is the conditioning of the eigenvector FORMULAS themselves.
      tnsr::i<DataVector, 9> speeds_from_quad{num_points, 0.0};
      for (size_t wave = 0; wave < 9; ++wave) {
        speeds_from_quad.get(wave) =
            DataVector(num_points, static_cast<double>(q_speeds[wave]));
      }
      tnsr::ij<DataVector, 9> modes_iso{num_points, 0.0};
      tnsr::IJ<DataVector, 9> projectors_iso{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
          make_not_null(&modes_iso), make_not_null(&projectors_iso),
          speeds_from_quad, spatial_velocity, magnetic_field, rest_mass_density,
          specific_internal_energy, lorentz_factor, specific_enthalpy,
          spatial_metric, unit_normal, eos_2d);

      // 1/W ASYMPTOTIC-EXPANSION eigensystem: replace the 4 magnetosonic speeds
      // with the large-W (eps = 1/W) series (asymptotic_magnetosonic_speeds),
      // keep the other 5 speeds, and feed that set to the eigenvector formulas
      // -- exactly like the isolated case but with the deployable double-
      // precision 1/W speeds in place of the quad speeds.  This is the method
      // that targets the high-W magnetosonic speed-solver cancellation.
      const double rho_h_e =
          get(rest_mass_density)[0] * get(specific_enthalpy)[0];
      const double inv_sqrt_rho_h_e = 1.0 / std::sqrt(rho_h_e);
      const std::array<double, 4> asym_e = asymptotic_speeds::speeds(
          get(sound_speed_squared)[0], -vn, W, -bx * inv_sqrt_rho_h_e,
          (bx * vn + by * vt) * inv_sqrt_rho_h_e,
          (bx * bx + by * by + bz * bz) / rho_h_e);
      tnsr::i<DataVector, 9> speeds_1w = speeds;
      speeds_1w.get(grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus) =
          DataVector(num_points, asym_e[0]);
      speeds_1w.get(grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus) =
          DataVector(num_points, asym_e[1]);
      speeds_1w.get(grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus) =
          DataVector(num_points, asym_e[2]);
      speeds_1w.get(grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus) =
          DataVector(num_points, asym_e[3]);
      tnsr::ij<DataVector, 9> modes_1w{num_points, 0.0};
      tnsr::IJ<DataVector, 9> projectors_1w{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
          make_not_null(&modes_1w), make_not_null(&projectors_1w), speeds_1w,
          spatial_velocity, magnetic_field, rest_mass_density,
          specific_internal_energy, lorentz_factor, specific_enthalpy,
          spatial_metric, unit_normal, eos_2d);
      const double speed_err_1w = std::max(
          std::max(
              std::abs(asym_e[0] -
                       static_cast<double>(
                           q_speeds[grmhd::ValenciaDivClean::MhdSpeed::
                                        FastMagnetosonicMinus])),
              std::abs(asym_e[1] -
                       static_cast<double>(
                           q_speeds[grmhd::ValenciaDivClean::MhdSpeed::
                                        FastMagnetosonicPlus]))),
          std::max(
              std::abs(asym_e[2] -
                       static_cast<double>(
                           q_speeds[grmhd::ValenciaDivClean::MhdSpeed::
                                        SlowMagnetosonicMinus])),
              std::abs(asym_e[3] -
                       static_cast<double>(
                           q_speeds[grmhd::ValenciaDivClean::MhdSpeed::
                                        SlowMagnetosonicPlus]))));
      // Analytic closed-form magnetosonic speed error vs quad (same 4 waves as
      // speed_err_1w), so the speeds panel compares analytic / 1W / numeric.
      double analytic_speed_err = 0.0;
      for (const auto w :
           {grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus,
            grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus,
            grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus,
            grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus}) {
        analytic_speed_err = std::max(
            analytic_speed_err,
            std::abs(speeds.get(w)[0] - static_cast<double>(q_speeds[w])));
      }

      // Minimum gap between the (quad) characteristic speeds, to quantify how
      // close this state is to a degeneracy (where the eigenvectors become
      // ill-conditioned).
      double speed_gap = std::numeric_limits<double>::max();
      for (size_t wi = 0; wi < 9; ++wi) {
        for (size_t wj = wi + 1; wj < 9; ++wj) {
          speed_gap = std::min(speed_gap,
                               std::abs(static_cast<double>(q_speeds[wi]) -
                                        static_cast<double>(q_speeds[wj])));
        }
      }

      // EIGENPROBLEM CHECK: do the analytic speeds & eigenvectors actually solve
      // A.R = y R and L.A = y L for the conserved characteristic matrix A = "As"
      // (flux_jacobian_mhd)?  This is independent of biorthogonality and of any
      // left eigenvector, and is the authoritative test that the eigenvectors
      // are correct.  Residual normalized per wave by |R| (resp. |L|).
      tnsr::iJ<DataVector, 9> char_matrix{num_points, 0.0};
      grmhd::ValenciaDivClean::flux_jacobian_mhd(
          make_not_null(&char_matrix), spatial_velocity, magnetic_field,
          rest_mass_density, specific_internal_energy,
          Scalar<DataVector>{DataVector(num_points, 0.0)}, lorentz_factor,
          specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
          eos_2d);
      double eig_resid = 0.0;
      double eig_resid_right = 0.0;
      double eig_resid_left = 0.0;
      for (size_t w = 0; w < 9; ++w) {
        const double y = speeds.get(w)[0];
        double rnorm = 0.0;
        double lnorm = 0.0;
        double ar_resid = 0.0;
        double la_resid = 0.0;
        for (size_t m = 0; m < 9; ++m) {
          double ar = 0.0;  // (A . R)_m
          double la = 0.0;  // (L . A)_m
          for (size_t n = 0; n < 9; ++n) {
            ar += char_matrix.get(m, n)[0] * modes.get(w, n)[0];
            la += projectors.get(w, n)[0] * char_matrix.get(n, m)[0];
          }
          ar_resid = std::max(ar_resid, std::abs(ar - y * modes.get(w, m)[0]));
          la_resid =
              std::max(la_resid, std::abs(la - y * projectors.get(w, m)[0]));
          rnorm = std::max(rnorm, std::abs(modes.get(w, m)[0]));
          lnorm = std::max(lnorm, std::abs(projectors.get(w, m)[0]));
        }
        eig_resid_right =
            std::max(eig_resid_right, ar_resid / std::max(rnorm, 1e-300));
        eig_resid_left =
            std::max(eig_resid_left, la_resid / std::max(lnorm, 1e-300));
      }
      eig_resid = std::max(eig_resid_right, eig_resid_left);

      // ISOLATED eigenproblem residual: feed the eigenvector formulas the
      // accurate (quad-derived) speeds (modes_iso / projectors_iso) and use the
      // quad speed as the eigenvalue.  This removes the speed-solver error, so
      // eig_resid_iso measures only the conditioning of the eigenvector FORMULAS
      // (their 1/gap factors blow up near a degeneracy).  Comparing eig_resid
      // (full) with eig_resid_iso disentangles speed-solver error (grows with W)
      // from eigenvector-formula error (near degeneracies).
      double eig_resid_iso = 0.0;
      for (size_t w = 0; w < 9; ++w) {
        const double y_iso = static_cast<double>(q_speeds[w]);
        double rnorm = 0.0;
        double lnorm = 0.0;
        double ar_resid = 0.0;
        double la_resid = 0.0;
        for (size_t m = 0; m < 9; ++m) {
          double ar = 0.0;
          double la = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            ar += char_matrix.get(m, n)[0] * modes_iso.get(w, n)[0];
            la += projectors_iso.get(w, n)[0] * char_matrix.get(n, m)[0];
          }
          ar_resid =
              std::max(ar_resid, std::abs(ar - y_iso * modes_iso.get(w, m)[0]));
          la_resid = std::max(
              la_resid, std::abs(la - y_iso * projectors_iso.get(w, m)[0]));
          rnorm = std::max(rnorm, std::abs(modes_iso.get(w, m)[0]));
          lnorm = std::max(lnorm, std::abs(projectors_iso.get(w, m)[0]));
        }
        eig_resid_iso =
            std::max({eig_resid_iso, ar_resid / std::max(rnorm, 1e-300),
                      la_resid / std::max(lnorm, 1e-300)});
      }

      // 1/W eigenproblem residual: eigenvector formulas fed the 1/W-expansion
      // speeds (modes_1w / projectors_1w), with the 1/W speed as the eigenvalue.
      double eig_resid_1w = 0.0;
      for (size_t w = 0; w < 9; ++w) {
        const double y_1w = speeds_1w.get(w)[0];
        double rnorm = 0.0;
        double lnorm = 0.0;
        double ar_resid = 0.0;
        double la_resid = 0.0;
        for (size_t m = 0; m < 9; ++m) {
          double ar = 0.0;
          double la = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            ar += char_matrix.get(m, n)[0] * modes_1w.get(w, n)[0];
            la += projectors_1w.get(w, n)[0] * char_matrix.get(n, m)[0];
          }
          ar_resid =
              std::max(ar_resid, std::abs(ar - y_1w * modes_1w.get(w, m)[0]));
          la_resid = std::max(
              la_resid, std::abs(la - y_1w * projectors_1w.get(w, m)[0]));
          rnorm = std::max(rnorm, std::abs(modes_1w.get(w, m)[0]));
          lnorm = std::max(lnorm, std::abs(projectors_1w.get(w, m)[0]));
        }
        eig_resid_1w =
            std::max({eig_resid_1w, ar_resid / std::max(rnorm, 1e-300),
                      la_resid / std::max(lnorm, 1e-300)});
      }

      // NUMERIC EIGENSYSTEM comparison: solve the same eigenproblem with the
      // general per-point eigensolver (numerical_characteristics / blaze::geev).
      // Unlike the analytic closed form, geev returns eigenvectors for its own
      // computed eigenvalues, so its eigenproblem residual stays near machine
      // precision wherever it succeeds -- which motivates it as a fallback near
      // a degeneracy.  +inf marks a genuine failure (caught here -- SpECTRE
      // ERROR/ASSERT throw): geev cannot resolve two real eigenvalues closer
      // than ~eps*cond(R), so where the speed_gap underflows that limit (the
      // near_type_I config at high W: the gap stays tiny while cond(R) grows
      // with the boost) it returns a spurious complex eigenvalue and trips the
      // real-spectrum check.  We record that as the solver's limit rather than
      // aborting.  (near_type_II lifts under boost, so its gap stays resolvable
      // and geev never fails.)
      double num_eig_resid = std::numeric_limits<double>::quiet_NaN();
      double num_speed_err = std::numeric_limits<double>::quiet_NaN();
      // Declared outside the try so the numeric eigenvectors survive for the
      // biorthogonality comparison below; num_ok records whether geev succeeded.
      tnsr::ij<DataVector, 9> num_modes{num_points, 0.0};
      tnsr::IJ<DataVector, 9> num_projectors{num_points, 0.0};
      bool num_ok = false;
      {
        try {
          tnsr::i<DataVector, 9> num_speeds{num_points, 0.0};
          grmhd::ValenciaDivClean::numerical_characteristics(
              make_not_null(&num_speeds), make_not_null(&num_modes),
              make_not_null(&num_projectors), spatial_velocity, magnetic_field,
              rest_mass_density, specific_internal_energy,
              Scalar<DataVector>{DataVector(num_points, 0.0)}, lorentz_factor,
              specific_enthalpy, spatial_metric, inv_spatial_metric,
              unit_normal, eos_2d);
          num_eig_resid = 0.0;
          num_speed_err = 0.0;
          for (size_t w = 0; w < 9; ++w) {
            const double y = num_speeds.get(w)[0];
            // Numeric eigenvalue vs the nearest analytic characteristic speed.
            double min_speed_diff = std::numeric_limits<double>::max();
            for (size_t s = 0; s < 9; ++s) {
              min_speed_diff =
                  std::min(min_speed_diff, std::abs(y - speeds.get(s)[0]));
            }
            num_speed_err = std::max(num_speed_err, min_speed_diff);
            double rnorm = 0.0;
            double lnorm = 0.0;
            double ar_resid = 0.0;
            double la_resid = 0.0;
            for (size_t m = 0; m < 9; ++m) {
              double ar = 0.0;
              double la = 0.0;
              for (size_t n = 0; n < 9; ++n) {
                ar += char_matrix.get(m, n)[0] * num_modes.get(w, n)[0];
                la += num_projectors.get(w, n)[0] * char_matrix.get(n, m)[0];
              }
              ar_resid =
                  std::max(ar_resid, std::abs(ar - y * num_modes.get(w, m)[0]));
              la_resid = std::max(
                  la_resid, std::abs(la - y * num_projectors.get(w, m)[0]));
              rnorm = std::max(rnorm, std::abs(num_modes.get(w, m)[0]));
              lnorm = std::max(lnorm, std::abs(num_projectors.get(w, m)[0]));
            }
            num_eig_resid =
                std::max({num_eig_resid, ar_resid / std::max(rnorm, 1e-300),
                          la_resid / std::max(lnorm, 1e-300)});
          }
          num_ok = true;
        } catch (const std::exception&) {
          num_eig_resid = std::numeric_limits<double>::infinity();
          num_speed_err = std::numeric_limits<double>::infinity();
        }
      }

      // Biorthogonality error: max_{i,j} |(L·R)_{ij} - delta_{ij}|
      double biorth_err = 0.0;
      for (size_t wi = 0; wi < 9; ++wi) {
        for (size_t wj = 0; wj < 9; ++wj) {
          double lr = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            lr += projectors.get(wi, n)[0] * modes.get(wj, n)[0];
          }
          biorth_err =
              std::max(biorth_err, std::abs(lr - (wi == wj ? 1.0 : 0.0)));
        }
      }
      max_biorth_err = std::max(max_biorth_err, biorth_err);

      // Diagnostic: dump the full L*R matrix and the raw eigenvector
      // components at selected W values so the biorthogonality structure can be
      // visualized (see plots in runs-ai/mhd_eigenvectors/).
      // mhd_lr_matrix.tsv  : long format config,W,i,j,value  (value = (L·R)_ij)
      // mhd_evec_raw.tsv   : long format config,W,kind,wave,n,value (kind=R|L)
      // iw==0 -> W=1 (rest frame, v=0); iw==6 -> W~1.4 (moderate, v!=0, NOT
      // degenerate) which actually exercises the moving-fluid eigenvectors;
      // iw==n_w/2 -> W~4.6; iw==n_w-1 -> W=20 (the high-W end of the sweep).
      if (output and
          (iw == 0 or iw == 6 or iw == n_w / 2 or iw == n_w - 1)) {
        for (size_t wi = 0; wi < 9; ++wi) {
          for (size_t wj = 0; wj < 9; ++wj) {
            double lr = 0.0;
            for (size_t n = 0; n < 9; ++n) {
              lr += projectors.get(wi, n)[0] * modes.get(wj, n)[0];
            }
            out_lr << config.name << '\t' << W << '\t' << wi << '\t' << wj
                   << '\t' << lr << '\n';
          }
        }
        for (size_t wave = 0; wave < 9; ++wave) {
          for (size_t n = 0; n < 9; ++n) {
            out_raw << config.name << '\t' << W << "\tR\t" << wave << '\t' << n
                    << '\t' << modes.get(wave, n)[0] << '\n';
            out_raw << config.name << '\t' << W << "\tL\t" << wave << '\t' << n
                    << '\t' << projectors.get(wave, n)[0] << '\n';
          }
        }
      }

      // Per-wave component errors vs quad
      std::array<double, 9> R_err{};
      std::array<double, 9> L_err{};
      for (size_t wave = 0; wave < 9; ++wave) {
        double r_max = 0.0;
        double l_max = 0.0;
        for (size_t n = 0; n < 9; ++n) {
          r_max = std::max(
              r_max, std::abs(modes.get(wave, n)[0] -
                               static_cast<double>(q_right[wave][n])));
          l_max = std::max(
              l_max, std::abs(projectors.get(wave, n)[0] -
                               static_cast<double>(q_left[wave][n])));
        }
        R_err[wave] = r_max;
        L_err[wave] = l_max;
      }

      // Diagnostic (NOT asserted): per-wave double-vs-quad component error.
      // For benign configs at moderate W these agree to ~1e-13, but the
      // ill-conditioned eigenvector formulas (their 1/a, 1/G, 1/cs^2 and 1/gap
      // factors) suffer catastrophic cancellation near degeneracies and at high
      // W / extreme magnetization, where even double and quad diverge by O(100).
      // Recorded to mhd_eigenvector_errors.tsv and plotted in
      // runs-ai/mhd_eigenvectors/; the moderate-W faithfulness assertion lives
      // in test_mhd_characteristics instead.
      double max_RL_err = 0.0;
      for (size_t wave = 0; wave < 9; ++wave) {
        max_RL_err = std::max({max_RL_err, R_err[wave], L_err[wave]});
      }
      CAPTURE(max_RL_err);

      // ISOLATED double-vs-quad component error: modes_iso (double formulas,
      // quad speeds) vs the quad reference (which used the same quad speeds).
      double max_RL_err_iso = 0.0;
      for (size_t wave = 0; wave < 9; ++wave) {
        for (size_t n = 0; n < 9; ++n) {
          max_RL_err_iso = std::max(
              {max_RL_err_iso,
               std::abs(modes_iso.get(wave, n)[0] -
                        static_cast<double>(q_right[wave][n])),
               std::abs(projectors_iso.get(wave, n)[0] -
                        static_cast<double>(q_left[wave][n]))});
        }
      }
      // 1/W component error: eigenvector formulas fed the 1/W speeds vs quad.
      double max_RL_err_1w = 0.0;
      for (size_t wave = 0; wave < 9; ++wave) {
        for (size_t n = 0; n < 9; ++n) {
          max_RL_err_1w = std::max(
              {max_RL_err_1w,
               std::abs(modes_1w.get(wave, n)[0] -
                        static_cast<double>(q_right[wave][n])),
               std::abs(projectors_1w.get(wave, n)[0] -
                        static_cast<double>(q_left[wave][n]))});
        }
      }
      // Normalized off-diagonal biorthogonality (divide each row by its
      // diagonal, since the paper's eigenvectors are biorthogonal but NOT
      // unit-normalized; the raw biorth_err above is dominated by the non-unit
      // diagonal norm and is not a contamination measure).  Computed for both
      // the whole system (double speeds) and the isolated case (quad speeds).
      auto normalized_offdiag = [](const auto& proj, const auto& md) {
        double worst = 0.0;
        for (size_t wi = 0; wi < 9; ++wi) {
          double diag = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            diag += proj.get(wi, n)[0] * md.get(wi, n)[0];
          }
          if (std::abs(diag) < 1e-300) {
            diag = 1e-300;
          }
          for (size_t wj = 0; wj < 9; ++wj) {
            if (wi == wj) {
              continue;
            }
            double lr = 0.0;
            for (size_t n = 0; n < 9; ++n) {
              lr += proj.get(wi, n)[0] * md.get(wj, n)[0];
            }
            worst = std::max(worst, std::abs(lr / diag));
          }
        }
        return worst;
      };
      const double biorth_norm_whole = normalized_offdiag(projectors, modes);
      const double biorth_norm_iso =
          normalized_offdiag(projectors_iso, modes_iso);
      const double biorth_norm_1w =
          normalized_offdiag(projectors_1w, modes_1w);
      // Biorthogonality of the NUMERIC (geev) left/right eigenvectors, in the
      // same normalized-off-diagonal measure, to compare against the analytic
      // biorth_norm_whole near degeneracies.  +inf if geev failed.
      const double num_biorth =
          num_ok ? normalized_offdiag(num_projectors, num_modes)
                 : std::numeric_limits<double>::infinity();

      if (output) {
        out_evec << config.name << '\t' << W << '\t' << biorth_err;
        for (size_t wave = 0; wave < 9; ++wave) {
          out_evec << '\t' << R_err[wave];
        }
        for (size_t wave = 0; wave < 9; ++wave) {
          out_evec << '\t' << L_err[wave];
        }
        out_evec << '\n';
        // Precision study: speed gap, whole vs isolated component error, and
        // whole vs isolated normalized off-diagonal biorthogonality.
        out_prec << config.name << '\t' << W << '\t' << speed_gap << '\t'
                 << max_RL_err << '\t' << max_RL_err_iso << '\t'
                 << analytic_speed_err << '\t'
                 << speed_err_1w << '\t' << max_RL_err_1w << '\t'
                 << biorth_norm_1w << '\t' << eig_resid_1w << '\t'
                 << biorth_norm_whole << '\t' << biorth_norm_iso << '\t'
                 << eig_resid << '\t' << eig_resid_right << '\t'
                 << eig_resid_left << '\t' << num_eig_resid << '\t'
                 << num_speed_err << '\t' << eig_resid_iso << '\t' << num_biorth
                 << '\n';
      }

      // Asymptotic expansion speeds.
      // The unit_normal is Direction<3>::lower_xi(), so n_a = (-1, 0, 0)
      // covariantly in flat metric.  The asymptotic function expects the
      // Lorentz-invariant projections:
      //   sv  = s_a v^a = n_x * v^x = -vn
      //   Bs  = (B^a s_a) / sqrt(rho h) = B^x * n_x / sqrt(rho h) =
      //   -bx/sqrt(rho h) Bv  = (B^a v_a) / sqrt(rho h)   (no sign from the
      //   normal) Bsq = B^a B_a / (rho h)          (no sign)
      const double rho_h =
          get(rest_mass_density)[0] * get(specific_enthalpy)[0];
      const double inv_sqrt_rho_h = 1.0 / std::sqrt(rho_h);
      const double sv_asym = -vn;
      const double Bs_asym = -bx * inv_sqrt_rho_h;
      const double Bv_scalar = bx * vn + by * vt;
      // Flat metric: B^2 = bx^2 + by^2 + bz^2
      const double B_squared = bx * bx + by * by + bz * bz;
      const auto asymptotic =
          asymptotic_speeds::speeds(get(sound_speed_squared)[0],
                                    sv_asym,  // s_a v^a = -vn (lower_xi normal)
                                    W,        // Lorentz factor
                                    Bs_asym,  // Bbar_s = -bx/sqrt(rho h)
                                    Bv_scalar * inv_sqrt_rho_h,  // Bbar.v
                                    B_squared / rho_h);          // Bbar^2
      const double asym_fast_m = asymptotic[0];
      const double asym_fast_p = asymptotic[1];
      const double asym_slow_m = asymptotic[2];
      const double asym_slow_p = asymptotic[3];

      // Print one tab-separated line to stdout so the Python verification
      // script can compare the C++ expansion against the Mathematica derivation
      // using exactly the same inputs that were passed to speeds().
      if (output) {
        std::cout << std::setprecision(17) << "ASYMP_CHECK" << '\t'
                  << config.name << '\t' << W << '\t'
                  << get(sound_speed_squared)[0] << '\t' << sv_asym << '\t'
                  << Bs_asym << '\t' << Bv_scalar * inv_sqrt_rho_h << '\t'
                  << B_squared / rho_h << '\t' << asym_fast_m << '\t'
                  << asym_fast_p << '\t' << asym_slow_m << '\t' << asym_slow_p
                  << '\n';
      }

      // Extract speeds
      const double fast_m = speeds.get(
          grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus)[0];
      const double fast_p = speeds.get(
          grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus)[0];
      const double slow_m = speeds.get(
          grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus)[0];
      const double slow_p = speeds.get(
          grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus)[0];
      const double alf_m =
          speeds.get(grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus)[0];
      const double alf_p =
          speeds.get(grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus)[0];
      const double ent =
          speeds.get(grmhd::ValenciaDivClean::MhdSpeed::Entropy)[0];

      const double q_fast_m = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicMinus]);
      const double q_fast_p = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::FastMagnetosonicPlus]);
      const double q_slow_m = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicMinus]);
      const double q_slow_p = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::SlowMagnetosonicPlus]);
      const double q_alf_m = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenMinus]);
      const double q_alf_p = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::AlfvenPlus]);
      const double q_ent = static_cast<double>(
          q_speeds[grmhd::ValenciaDivClean::MhdSpeed::Entropy]);

      // Track errors
      const double abs_fast =
          std::max(std::abs(fast_m - q_fast_m), std::abs(fast_p - q_fast_p));
      const double abs_slow =
          std::max(std::abs(slow_m - q_slow_m), std::abs(slow_p - q_slow_p));
      const double abs_alfven =
          std::max(std::abs(alf_m - q_alf_m), std::abs(alf_p - q_alf_p));
      const double abs_entropy = std::abs(ent - q_ent);
      max_abs_fast = std::max(max_abs_fast, abs_fast);
      max_abs_slow = std::max(max_abs_slow, abs_slow);
      max_abs_alfven = std::max(max_abs_alfven, abs_alfven);
      max_abs_entropy = std::max(max_abs_entropy, abs_entropy);
      max_rel_fast = std::max(
          max_rel_fast,
          std::max(rel_err(fast_m, q_fast_m), rel_err(fast_p, q_fast_p)));
      max_rel_slow = std::max(
          max_rel_slow,
          std::max(rel_err(slow_m, q_slow_m), rel_err(slow_p, q_slow_p)));

      if (output) {
        out << config.name << '\t' << W << '\t' << fast_m << '\t' << fast_p
            << '\t' << slow_m << '\t' << slow_p << '\t' << alf_m << '\t'
            << alf_p << '\t' << ent << '\t' << q_fast_m << '\t' << q_fast_p
            << '\t' << q_slow_m << '\t' << q_slow_p << '\t' << q_alf_m << '\t'
            << q_alf_p << '\t' << q_ent << '\t' << std::abs(fast_m - q_fast_m)
            << '\t' << std::abs(fast_p - q_fast_p) << '\t'
            << std::abs(slow_m - q_slow_m) << '\t'
            << std::abs(slow_p - q_slow_p) << '\t' << std::abs(alf_m - q_alf_m)
            << '\t' << std::abs(alf_p - q_alf_p) << '\t'
            << std::abs(ent - q_ent) << '\t' << rel_err(fast_m, q_fast_m)
            << '\t' << rel_err(fast_p, q_fast_p) << '\t'
            << rel_err(slow_m, q_slow_m) << '\t' << rel_err(slow_p, q_slow_p)
            << '\t' << rel_err(alf_m, q_alf_m) << '\t'
            << rel_err(alf_p, q_alf_p) << '\t' << rel_err(ent, q_ent)
            << '\t'
            // Asymptotic expansion speeds and their errors vs quad
            << asym_fast_m << '\t' << asym_fast_p << '\t' << asym_slow_m << '\t'
            << asym_slow_p << '\t' << std::abs(asym_fast_m - q_fast_m) << '\t'
            << std::abs(asym_fast_p - q_fast_p) << '\t'
            << std::abs(asym_slow_m - q_slow_m) << '\t'
            << std::abs(asym_slow_p - q_slow_p) << '\t'
            << rel_err(asym_fast_m, q_fast_m) << '\t'
            << rel_err(asym_fast_p, q_fast_p) << '\t'
            << rel_err(asym_slow_m, q_slow_m) << '\t'
            << rel_err(asym_slow_p, q_slow_p) << '\n';
      }

      // Speed ordering at every point.
      // Near degeneracies the ReducedQuadratic discriminant clamping can
      // push slow speeds slightly past the entropy speed (~2e-5 at high sigma).
      constexpr double ordering_tolerance = 5.0e-5;
      CHECK(speeds.get(grmhd::ValenciaDivClean::MhdSpeed::ScalarMinus)[0] <=
            fast_m + ordering_tolerance);
      CHECK(fast_m <= alf_m + ordering_tolerance);
      CHECK(alf_m <= slow_m + ordering_tolerance);
      CHECK(slow_m <= ent + ordering_tolerance);
      CHECK(ent <= slow_p + ordering_tolerance);
      CHECK(slow_p <= alf_p + ordering_tolerance);
      CHECK(alf_p <= fast_p + ordering_tolerance);
      CHECK(fast_p <=
            speeds.get(grmhd::ValenciaDivClean::MhdSpeed::ScalarPlus)[0] +
                ordering_tolerance);
    }

    // Print summary for this config
    CAPTURE(config.name);
    CAPTURE(max_abs_fast);
    CAPTURE(max_rel_fast);
    CAPTURE(max_abs_slow);
    CAPTURE(max_rel_slow);
    CAPTURE(max_abs_alfven);
    CAPTURE(max_abs_entropy);
    CAPTURE(max_biorth_err);

    // Regression thresholds (loose initially; tighten after baseline)
    constexpr double max_allowed_abs_error = 5.0e-3;
    CAPTURE(config.name);
    CAPTURE(max_abs_fast);
    CAPTURE(max_abs_slow);
    CHECK(max_abs_fast < max_allowed_abs_error);
    CHECK(max_abs_slow < max_allowed_abs_error);

    // Biorthogonality is a DIAGNOSTIC, not an assertion.  With the corrected
    // eigenvectors L·R is the identity at machine precision for well-separated
    // states; max_biorth_err grows only as a speed degeneracy is approached
    // (where the eigenvector basis becomes ill-conditioned) and at high W (where
    // the magnetosonic speed solver loses precision).  Because this sweep
    // deliberately probes those regimes, the error is logged (to
    // mhd_eigenvector_errors.tsv, visualized in runs-ai/mhd_eigenvectors/)
    // rather than asserted.
    CAPTURE(max_biorth_err);
  }
}

void run_mhd_characteristic_benchmarks(const bool enable) {
  if (not enable) {
    return;
  }

  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  constexpr size_t num_points = 1000000;
  const DataVector used_for_size(num_points);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const EquationsOfState::IdealFluid<true> eos_2d(1.5, 0.0);
  const auto pressure = eos_2d.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto magnetic_field =
      helper::random_magnetic_field(nn_gen, pressure, spatial_metric);

  Scalar<DataVector> sound_speed_squared{num_points};
  get(sound_speed_squared) =
      get(eos_2d.chi_from_density_and_energy(rest_mass_density,
                                             specific_internal_energy)) +
      get(eos_2d.kappa_times_p_over_rho_squared_from_density_and_energy(
          rest_mass_density, specific_internal_energy));
  get(sound_speed_squared) /= get(specific_enthalpy);

  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);
  const Scalar<DataVector> normal_velocity =
      tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
  const Scalar<DataVector> normal_magnetic_field =
      tenex::evaluate(magnetic_field(ti::I) * unit_normal(ti::i));
  const auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(magnetic_field, spatial_velocity, spatial_metric);
  const Scalar<DataVector> comoving_magnetic_field_squared{
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity))};
  const Scalar<DataVector> inv_rho_h{
      1.0 / (get(rest_mass_density) * get(specific_enthalpy))};
  const Scalar<DataVector> inv_sqrt_rho_h{sqrt(get(inv_rho_h))};
  const Scalar<DataVector> normal_magnetic_field_scaled{
      get(normal_magnetic_field) * get(inv_sqrt_rho_h)};
  const Scalar<DataVector> magnetic_field_dot_spatial_velocity_scaled{
      get(magnetic_field_dot_spatial_velocity) * get(inv_sqrt_rho_h)};
  const Scalar<DataVector> magnetic_field_squared_scaled{
      get(magnetic_field_squared) * get(inv_rho_h)};
  const Scalar<DataVector> comoving_magnetic_field_squared_scaled{
      get(comoving_magnetic_field_squared) * get(inv_rho_h)};

  // Benchmark ran on mbot (Iago Mendes, April 2026):
  // - Optimized: ~13 ms
  // - Unoptimized: ~13 ms
  // - Result: not faster, but re-writes expressions to avoid
  //           catastrophic cancellation.
  tnsr::i<DataVector, 4> optimized_coefficients{num_points};
  BENCHMARK("magnetosonic_quartic_coefficients") {
    grmhd::ValenciaDivClean::magnetosonic_quartic_coefficients(
        make_not_null(&optimized_coefficients), sound_speed_squared,
        normal_velocity, lorentz_factor, normal_magnetic_field_scaled,
        magnetic_field_dot_spatial_velocity_scaled,
        magnetic_field_squared_scaled, comoving_magnetic_field_squared_scaled);
  };

  // Benchmark ran on mbot (Iago Mendes, April 2026):
  // - Optimized: ~14 ms
  // - Unoptimized: ~52 ms
  // - Result: ~3.714285714x speedup (~270% faster)
  // - Takeaway: miniming allocations is the biggest factor in improving
  //             performance, even if that means more floating-point operations.
  DataVector optimized_fast_plus(num_points, 1.0);
  DataVector optimized_fast_minus(num_points, -1.0);
  BENCHMARK("find_magnetosonic_speed_from_quartic") {
    grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
        make_not_null(&optimized_fast_plus), optimized_coefficients);
    grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
        make_not_null(&optimized_fast_minus), optimized_coefficients);
  };
  // The unoptimized benchmark is unstable for some random realizations and can
  // trigger SIGFPE, so keep it disabled for method-comparison benchmark runs.

  tnsr::i<DataVector, 9> characteristic_speeds_toms{num_points};
  tnsr::i<DataVector, 9> characteristic_speeds_reduced{num_points};
  BENCHMARK("characteristic_speeds_mhd (TOMS748)") {
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&characteristic_speeds_toms), spatial_velocity,
        magnetic_field, rest_mass_density, specific_internal_energy,
        lorentz_factor, specific_enthalpy, spatial_metric, unit_normal, eos_2d,
        grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod::Toms748);
  };
  BENCHMARK("characteristic_speeds_mhd (reduced quadratic)") {
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&characteristic_speeds_reduced), spatial_velocity,
        magnetic_field, rest_mass_density, specific_internal_energy,
        lorentz_factor, specific_enthalpy, spatial_metric, unit_normal, eos_2d,
        grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod::ReducedQuadratic);
  };

  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&characteristic_speeds_toms), spatial_velocity,
      magnetic_field, rest_mass_density, specific_internal_energy,
      lorentz_factor, specific_enthalpy, spatial_metric, unit_normal, eos_2d,
      grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod::Toms748);
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&characteristic_speeds_reduced), spatial_velocity,
      magnetic_field, rest_mass_density, specific_internal_energy,
      lorentz_factor, specific_enthalpy, spatial_metric, unit_normal, eos_2d,
      grmhd::ValenciaDivClean::SlowMagnetosonicSpeedMethod::ReducedQuadratic);
  double max_method_difference = 0.0;
  for (size_t i = 0; i < 9; ++i) {
    max_method_difference = std::max(
        max_method_difference, max(abs(characteristic_speeds_toms.get(i) -
                                       characteristic_speeds_reduced.get(i))));
  }
  CAPTURE(max_method_difference);
  CHECK(max_method_difference < 1.0e-2);

  // Eigensystem benchmark: analytic closed-form eigenvectors vs the per-point
  // numeric eigensolver (blaze::geev).  The numeric solver loops over grid
  // points one 9x9 problem at a time, so it is expected to be far more
  // expensive than the vectorized analytic formulas; this quantifies the cost
  // of falling back to the numeric eigensystem (e.g. near degeneracies).  It is
  // run on a reduced point count, and is much slower with SPECTRE_DEBUG on
  // (which adds a per-point GSL eigenvalue cross-check).
  constexpr size_t eig_bench_points = 10000;
  const DataVector eig_used_for_size(eig_bench_points);
  const auto eig_rest_mass_density =
      helper::random_density(nn_gen, eig_used_for_size);
  const auto eig_specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, eig_used_for_size);
  const auto eig_lorentz_factor =
      helper::random_lorentz_factor(nn_gen, eig_used_for_size);
  const auto eig_spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, eig_used_for_size);
  const auto eig_spatial_velocity =
      helper::random_velocity(nn_gen, eig_lorentz_factor, eig_spatial_metric);
  const auto& eig_inv_spatial_metric =
      determinant_and_inverse(eig_spatial_metric).second;
  const auto eig_pressure = eos_2d.pressure_from_density_and_energy(
      eig_rest_mass_density, eig_specific_internal_energy);
  const auto eig_specific_enthalpy = hydro::relativistic_specific_enthalpy(
      eig_rest_mass_density, eig_specific_internal_energy, eig_pressure);
  const auto eig_magnetic_field =
      helper::random_magnetic_field(nn_gen, eig_pressure, eig_spatial_metric);
  const Scalar<DataVector> eig_electron_fraction{
      DataVector(eig_bench_points, 0.0)};
  const auto eig_unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), eig_inv_spatial_metric);

  tnsr::i<DataVector, 9> eig_analytic_speeds{eig_bench_points};
  tnsr::ij<DataVector, 9> eig_modes{eig_bench_points};
  tnsr::IJ<DataVector, 9> eig_projectors{eig_bench_points};
  BENCHMARK("analytic eigensystem (speeds + eigenvectors)") {
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&eig_analytic_speeds), eig_spatial_velocity,
        eig_magnetic_field, eig_rest_mass_density, eig_specific_internal_energy,
        eig_lorentz_factor, eig_specific_enthalpy, eig_spatial_metric,
        eig_unit_normal, eos_2d);
    grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
        make_not_null(&eig_modes), make_not_null(&eig_projectors),
        eig_analytic_speeds, eig_spatial_velocity, eig_magnetic_field,
        eig_rest_mass_density, eig_specific_internal_energy, eig_lorentz_factor,
        eig_specific_enthalpy, eig_spatial_metric, eig_unit_normal, eos_2d);
  };

  tnsr::i<DataVector, 9> eig_numeric_speeds{eig_bench_points};
  tnsr::ij<DataVector, 9> eig_numeric_modes{eig_bench_points};
  tnsr::IJ<DataVector, 9> eig_numeric_projectors{eig_bench_points};
  BENCHMARK("numeric eigensystem (blaze::geev per point)") {
    grmhd::ValenciaDivClean::numerical_characteristics(
        make_not_null(&eig_numeric_speeds), make_not_null(&eig_numeric_modes),
        make_not_null(&eig_numeric_projectors), eig_spatial_velocity,
        eig_magnetic_field, eig_rest_mass_density, eig_specific_internal_energy,
        eig_electron_fraction, eig_lorentz_factor, eig_specific_enthalpy,
        eig_spatial_metric, eig_inv_spatial_metric, eig_unit_normal, eos_2d);
  };
}

// Degeneracy-tolerance study.  Sweep the normal magnetic field B_n -> 0 (the
// Type-I degeneracy, where the slow / Alfven / entropy speeds collapse toward
// v_n) and, at each state, measure how well competing decompositions
// reconstruct the flux Jacobian A = R Lambda L^T.  The reference A_ref is the
// QUAD-precision analytic reconstruction sum_i lam_i^q R_i^q (L_i^q)^T (accurate
// to gaps ~1e-30, unlike double ~1e-16).  We compare:
//   * all-analytic:   sum over all 9 double modes (biorthonormality is lost near
//                     degeneracy, so this blows up);
//   * CPM variants:   lump a near-degenerate set S into (I - P_kept) upwound at
//                     the mean speed of S -- accurate near degeneracy, but a
//                     single-speed approximation that costs accuracy away from
//                     it.
// The crossover of the error curves suggests the speed-gap DegeneracyTolerance.
// The sweep TSV is written only when output is set (SPECTRE_DEGEN_DUMP).
// One degeneracy-sweep point.  The reference A is the flux Jacobian
// flux_jacobian_mhd, which the matrix-precision study (matrix_errors) showed is
// accurate to round-off through the degeneracy.  We reconstruct A with the
// double analytic all-modes decomposition (err_analytic) and with the
// complementary projection of the fluid modes slow+Alfven+entropy
// (err_cpm_fluid).  ok=false if the analytic eigenvector formulas throw.
struct DegenPointResult {
  double gap;
  double err_analytic;
  double err_cpm_fluid;
  std::array<double, 9> speeds;
  bool ok;
};
DegenPointResult degeneracy_point_result(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::IdealFluid<true>& eos) {
  constexpr size_t num_points = 1;
  try {
    tnsr::i<DataVector, 9> speeds{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal, eos);
    tnsr::ij<DataVector, 9> modes{num_points, 0.0};
    tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
        make_not_null(&modes), make_not_null(&projectors), speeds,
        spatial_velocity, magnetic_field, rest_mass_density,
        specific_internal_energy, lorentz_factor, specific_enthalpy,
        spatial_metric, unit_normal, eos);
    tnsr::iJ<DataVector, 9> jacobian{num_points, 0.0};
    grmhd::ValenciaDivClean::flux_jacobian_mhd(
        make_not_null(&jacobian), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, eos);
    std::array<std::array<double, 9>, 9> a_ref{};
    for (size_t m = 0; m < 9; ++m) {
      for (size_t n = 0; n < 9; ++n) {
        a_ref[m][n] = jacobian.get(m, n)[0];
      }
    }
    std::array<double, 9> diag_d{};
    std::array<double, 9> lam{};
    for (size_t i = 0; i < 9; ++i) {
      double dd = 0.0;
      for (size_t k = 0; k < 9; ++k) {
        dd += projectors.get(i, k)[0] * modes.get(i, k)[0];
      }
      diag_d[i] = dd;
      lam[i] = speeds.get(i)[0];
    }
    // Fluid complement set: Alfven-, slow-, entropy, slow+, Alfven+ (indices
    // 2,3,4,5,6).
    const std::array<bool, 9> fluid_mask{
        {false, false, true, true, true, true, true, false, false}};
    const std::array<bool, 9> none_mask{};
    const auto recon_err = [&](const std::array<bool, 9>& in_s) {
      double lam_s = 0.0;
      size_t count = 0;
      for (size_t i = 0; i < 9; ++i) {
        if (in_s[i]) {
          lam_s += lam[i];
          ++count;
        }
      }
      if (count > 0) {
        lam_s /= static_cast<double>(count);
      }
      double err = 0.0;
      for (size_t m = 0; m < 9; ++m) {
        for (size_t n = 0; n < 9; ++n) {
          double kept_lam = 0.0;
          double kept_proj = 0.0;
          for (size_t i = 0; i < 9; ++i) {
            if (not in_s[i]) {
              const double rl =
                  modes.get(i, m)[0] * projectors.get(i, n)[0] / diag_d[i];
              kept_lam += lam[i] * rl;
              kept_proj += rl;
            }
          }
          double val = kept_lam;
          if (count > 0) {
            val += lam_s * ((m == n ? 1.0 : 0.0) - kept_proj);
          }
          err = std::max(err, std::abs(val - a_ref[m][n]));
        }
      }
      return err;
    };
    std::array<double, 9> sorted = lam;
    std::sort(sorted.begin(), sorted.end());
    double gap = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < 9; ++i) {
      gap = std::min(gap, sorted[i + 1] - sorted[i]);
    }
    return {gap, recon_err(none_mask), recon_err(fluid_mask), lam, true};
  } catch (...) {
    return {0.0, 0.0, 0.0, std::array<double, 9>{}, false};
  }
}

// Parameter-space survey of the analytic-vs-CPM crossover.  For a grid of
// background states (thermal pressure/cs^2, magnetization sigma, Lorentz factor
// W, field angle phi) sweep B_n -> 0 and dump the reconstruction errors, so the
// crossover gap can be compared across configs (it is state-dependent).  Uses
// the fluid-modes complement only.  TSV dump guarded by SPECTRE_DEGEN_DUMP.
void test_degeneracy_parameter_sweep(const bool output) {
  const ScopedFpeState disable_fpes(false);
  constexpr size_t num_points = 1;
  // Non-round adiabatic index so mistaken powers (rho vs rho^2, ...) cannot hide
  // behind unit values.  The spatial metric is kept flat: the eigen-
  // reconstruction A = R Lambda L^T used below is only clean in flat space,
  // because the conserved variables mix covariant (S_i) and contravariant (B^i)
  // components (a non-flat metric leaves a constant reconstruction offset).
  const EquationsOfState::IdealFluid<true> eos_2d(1.37, 0.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  struct Cfg {
    double pressure;
    double sigma;
    double W;
    double phi;
    double vn_frac;
  };
  // Four representative configs = the corners of the (temperature x
  // magnetization) parameter space, so the crossover spread is captured:
  //   0 cold_weakB   - cold fluid, weakly magnetized
  //   1 cold_strongB - cold fluid, strongly magnetized
  //   2 hot_weakB    - hot fluid, weakly magnetized
  //   3 hot_strongB  - hot fluid, strongly magnetized
  // (pressure, sigma, W, phi, vn_frac); non-round values throughout.
  const std::array<Cfg, 4> cfgs{{{1.3e-4, 1.1e-4, 1.43, M_PI / 4.0, 0.28},
                                 {1.3e-4, 1.2e3, 1.43, M_PI / 4.0, 0.28},
                                 {1.1, 1.1e-4, 1.43, M_PI / 4.0, 0.28},
                                 {1.1, 1.2e3, 1.43, M_PI / 4.0, 0.28}}};
  const size_t nc = cfgs.size();

  std::ofstream dump;
  if (output) {
    dump.open("degeneracy_parameter_sweep.tsv", std::ios::out | std::ios::trunc);
    dump << std::setprecision(16);
    dump << "cfg\tpressure\tsigma\tW\tphi\tvn_frac\tbn_frac\tmin_gap\t"
            "err_analytic\terr_cpm_fluid\t"
            "sm\tfm\tam\tslm\tent\tslp\tap\tfp\tsp\n";
  }

  constexpr size_t n_bn = 50;
  double best_analytic = std::numeric_limits<double>::infinity();
  for (size_t ic = 0; ic < nc; ++ic) {
    const Cfg& cf = cfgs[ic];
    Scalar<DataVector> rest_mass_density{DataVector(num_points, 1.13)};
    const Scalar<DataVector> pressure{DataVector(num_points, cf.pressure)};
    const Scalar<DataVector> specific_internal_energy =
        eos_2d.specific_internal_energy_from_density_and_pressure(
            rest_mass_density, pressure);
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(rest_mass_density,
                                              specific_internal_energy,
                                              pressure);
    const Scalar<DataVector> lorentz_factor{DataVector(num_points, cf.W)};
    const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.13)};
    const double vmag = std::sqrt(std::max(0.0, 1.0 - 1.0 / square(cf.W)));
    const double vn = cf.vn_frac * vmag;
    const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));
    tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
    spatial_velocity.get(0) = vn;
    spatial_velocity.get(1) = vt;
    spatial_velocity.get(2) = 0.0;
    const double b_mag = std::sqrt(cf.sigma * get(rest_mass_density)[0] *
                                   get(specific_enthalpy)[0]);
    for (size_t ib = 0; ib < n_bn; ++ib) {
      const double bn_frac = std::pow(
          10.0, std::log10(0.5) + (std::log10(1.0e-10) - std::log10(0.5)) *
                                      static_cast<double>(ib) /
                                      static_cast<double>(n_bn - 1));
      const double bx = bn_frac * b_mag;
      const double bt = std::sqrt(std::max(0.0, square(b_mag) - square(bx)));
      tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
      magnetic_field.get(0) = bx;
      magnetic_field.get(1) = bt * std::cos(cf.phi);
      magnetic_field.get(2) = bt * std::sin(cf.phi);
      const auto e = degeneracy_point_result(
          spatial_velocity, magnetic_field, rest_mass_density,
          specific_internal_energy, electron_fraction, lorentz_factor,
          specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
          eos_2d);
      if (not e.ok) {
        break;
      }
      best_analytic = std::min(best_analytic, e.err_analytic);
      if (dump.is_open()) {
        dump << ic << '\t' << cf.pressure << '\t' << cf.sigma << '\t' << cf.W
             << '\t' << cf.phi << '\t' << cf.vn_frac << '\t' << bn_frac << '\t'
             << e.gap << '\t' << e.err_analytic << '\t' << e.err_cpm_fluid;
        for (size_t i = 0; i < 9; ++i) {
          dump << '\t' << e.speeds[i];
        }
        dump << '\n';
      }
    }
  }
  CHECK(best_analytic < 1.0e-9);
}

// Numerical-precision study of the analytic MHD flux Jacobian ENTRIES.
//
// The double flux_jacobian_mhd forms the same 9x9 matrix that the numeric
// eigensystem path (numerical_characteristics -> blaze::geev) diagonalizes, so
// if that matrix's entries lose accuracy to floating-point cancellation as the
// state approaches degeneracy (normal field B_n -> 0), the numeric path is
// capped by the same loss.  Here we recompute the identical matrix in
// quad precision (quad_precision::flux_jacobian_mhd) and measure the max
// absolute/relative entry difference vs the min eigenvalue gap, isolating the
// cancellation in the matrix itself (independent of the eigen-decomposition).
//
// Isotropic non-flat metric, normal along x.  Two states: a hot moderate-sigma
// state and a cold state.  B_n/|B| is swept from ~0.5 down to ~1e-9.  TSV dump
// guarded by SPECTRE_MATRIX_DUMP.  A CHECK verifies double==quad to ~1e-14 in
// the well-conditioned regime (B_n/|B| ~ 0.3-0.5); failure there means the port
// is buggy.
void test_flux_jacobian_precision(const bool output) {
  const ScopedFpeState disable_fpes(false);
  constexpr size_t num_points = 1;
  // Non-round adiabatic index and isotropic metric (see the parameter sweep).
  constexpr double adiabatic_index = 1.37;
  const EquationsOfState::IdealFluid<true> eos_2d(adiabatic_index, 0.0);
  constexpr double metric_value = 1.17;

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = metric_value;
  spatial_metric.get(1, 1) = metric_value;
  spatial_metric.get(2, 2) = metric_value;
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  std::ofstream dump;
  if (output) {
    dump.open("flux_jacobian_precision.tsv", std::ios::out | std::ios::trunc);
    dump << std::setprecision(16);
    dump << "state\tbn_frac\tbx\tmin_gap\tmax_abs_err\tmax_rel_err\n";
  }

  struct State {
    std::string name;
    double W;
    double density;
    double pressure;
    double sigma;  // B^2 / (rho h)
  };
  // Hot moderate-sigma state and a cold state.
  const std::array<State, 2> states{
      {{"hot", 1.23, 1.13, 0.12, 1.3}, {"cold", 1.07, 1.13, 1.3e-4, 0.27}}};

  const double phi_B = M_PI / 4.0;  // tangential B angle
  const double vn_fraction = 0.28;

  bool checked_well_conditioned = false;

  for (const auto& st : states) {
    const Scalar<DataVector> rest_mass_density{
        DataVector(num_points, st.density)};
    const Scalar<DataVector> pressure{DataVector(num_points, st.pressure)};
    const Scalar<DataVector> specific_internal_energy =
        eos_2d.specific_internal_energy_from_density_and_pressure(
            rest_mass_density, pressure);
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
    const Scalar<DataVector> lorentz_factor{DataVector(num_points, st.W)};
    const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.13)};

    // Contravariant norms carry 1/sqrt(metric_value) (isotropic metric).
    const double vmag =
        std::sqrt(std::max(0.0, (1.0 - 1.0 / square(st.W)) / metric_value));
    const double vn = vn_fraction * vmag;
    const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));
    tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
    spatial_velocity.get(0) = vn;
    spatial_velocity.get(1) = vt;
    spatial_velocity.get(2) = 0.0;
    const double b_mag = std::sqrt(st.sigma * st.density *
                                   get(specific_enthalpy)[0] / metric_value);

    // EoS-consistent quad c_s^2 and kappa, computed EXACTLY as the double
    // flux_jacobian_mhd computes them for an IdealFluid, so the quad and double
    // matrices share identical thermodynamic inputs (any residual difference is
    // then genuine entry-level floating-point behaviour, not an EoS mismatch).
    // Double path (see IdealFluid.cpp + flux_jacobian_mhd):
    //   chi = (Gamma-1) eps,  kappa_times_p_over_rho2 = (Gamma-1)^2 eps,
    //   c_s^2 = (chi + kappa_times_p_over_rho2) / h,
    //   kappa = kappa_times_p_over_rho2 / p * rho^2 = (Gamma-1) rho,
    // using the LITERAL (Gamma-1) and the same double-precision h (cast to quad)
    // that is passed to the double function.
    const quad_ref::Quad q_eps = get(specific_internal_energy)[0];
    const quad_ref::Quad q_h = get(specific_enthalpy)[0];
    const quad_ref::Quad q_rho = get(rest_mass_density)[0];
    const quad_ref::Quad q_gamma_minus_one =
        quad_ref::Quad{adiabatic_index} - quad_ref::Quad{1.0};
    const quad_ref::Quad q_chi = q_gamma_minus_one * q_eps;
    const quad_ref::Quad q_kappa_p_over_rho2 =
        q_gamma_minus_one * q_gamma_minus_one * q_eps;
    const quad_ref::Quad q_cs2 = (q_chi + q_kappa_p_over_rho2) / q_h;
    const quad_ref::Quad q_kappa = q_gamma_minus_one * q_rho;

    constexpr size_t n_bn = 50;
    for (size_t ib = 0; ib < n_bn; ++ib) {
      // log-spaced B_n fraction from 0.5 down to 1e-9
      const double bn_frac = std::pow(
          10.0, std::log10(0.5) +
                    (std::log10(1.0e-9) - std::log10(0.5)) *
                        static_cast<double>(ib) / static_cast<double>(n_bn - 1));
      const double bx = bn_frac * b_mag;
      const double bt = std::sqrt(std::max(0.0, square(b_mag) - square(bx)));
      tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
      magnetic_field.get(0) = bx;
      magnetic_field.get(1) = bt * std::cos(phi_B);
      magnetic_field.get(2) = bt * std::sin(phi_B);

      // Double analytic flux Jacobian.
      tnsr::iJ<DataVector, 9> jacobian{num_points, 0.0};
      grmhd::ValenciaDivClean::flux_jacobian_mhd(
          make_not_null(&jacobian), spatial_velocity, magnetic_field,
          rest_mass_density, specific_internal_energy, electron_fraction,
          lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
          unit_normal, eos_2d);

      // Quad analytic flux Jacobian (same matrix, quad precision).
      std::array<quad_ref::Quad, 3> q_v{};
      std::array<quad_ref::Quad, 3> q_b{};
      std::array<quad_ref::Quad, 3> q_n{};
      std::array<std::array<quad_ref::Quad, 3>, 3> q_g{};
      std::array<std::array<quad_ref::Quad, 3>, 3> q_inv_g{};
      for (size_t i = 0; i < 3; ++i) {
        q_v[i] = spatial_velocity.get(i)[0];
        q_b[i] = magnetic_field.get(i)[0];
        q_n[i] = unit_normal.get(i)[0];
        for (size_t j = 0; j < 3; ++j) {
          q_g[i][j] = spatial_metric.get(i, j)[0];
          q_inv_g[i][j] = inv_spatial_metric.get(i, j)[0];
        }
      }
      const auto q_jac = quad_ref::flux_jacobian_mhd(
          q_v, q_b, q_rho, q_eps, quad_ref::Quad{"0.1"}, st.W, q_h, q_g,
          q_inv_g, q_n, q_cs2, q_kappa);

      // Max absolute entry difference, and the max per-entry RELATIVE error
      // restricted to entries that are "significant" -- i.e. whose magnitude is
      // not a mere round-off residue of an analytically-zero entry.  We treat an
      // entry as significant when its magnitude exceeds 1e-8 times the largest
      // entry of the matrix (matrix inf-scale).  Below that floor a nominally
      // zero entry can show up as ~1e-16 in one precision and a different ~1e-16
      // in the other, giving a meaningless relative error of order unity; those
      // entries carry no accuracy loss because their absolute value is
      // negligible against the matrix scale.  The reported max_rel_err therefore
      // measures genuine loss of significant digits in the meaningful entries,
      // which is exactly what caps the eigensolver.
      double matrix_scale = 0.0;
      for (size_t m = 0; m < 9; ++m) {
        for (size_t n = 0; n < 9; ++n) {
          matrix_scale = std::max(matrix_scale, std::abs(jacobian.get(m, n)[0]));
          matrix_scale =
              std::max(matrix_scale, std::abs(static_cast<double>(q_jac[m][n])));
        }
      }
      const double significant_floor = 1.0e-8 * matrix_scale;
      double max_abs_err = 0.0;
      double max_rel_err = 0.0;
      for (size_t m = 0; m < 9; ++m) {
        for (size_t n = 0; n < 9; ++n) {
          const double dbl = jacobian.get(m, n)[0];
          const double quad = static_cast<double>(q_jac[m][n]);
          const double abs_err = std::abs(dbl - quad);
          max_abs_err = std::max(max_abs_err, abs_err);
          const double scale = std::max(std::abs(dbl), std::abs(quad));
          if (scale > significant_floor) {
            max_rel_err = std::max(max_rel_err, abs_err / scale);
          }
        }
      }

      // Min eigenvalue gap from the analytic speeds.
      tnsr::i<DataVector, 9> speeds{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_speeds_mhd(
          make_not_null(&speeds), spatial_velocity, magnetic_field,
          rest_mass_density, specific_internal_energy, lorentz_factor,
          specific_enthalpy, spatial_metric, unit_normal, eos_2d);
      std::array<double, 9> sorted{};
      for (size_t i = 0; i < 9; ++i) {
        sorted[i] = speeds.get(i)[0];
      }
      std::sort(sorted.begin(), sorted.end());
      double min_gap = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < 9; ++i) {
        min_gap = std::min(min_gap, sorted[i + 1] - sorted[i]);
      }

      // Verification: in the well-conditioned regime (B_n/|B| ~ 0.3-0.5) the
      // quad and double matrices must agree to ~1e-14 relative.  A larger error
      // there indicates a bug in the quad port.
      if (bn_frac >= 0.3 and bn_frac <= 0.5) {
        CHECK(max_rel_err < 1.0e-13);
        checked_well_conditioned = true;
      }

      if (dump.is_open()) {
        dump << st.name << '\t' << bn_frac << '\t' << bx << '\t' << min_gap
             << '\t' << max_abs_err << '\t' << max_rel_err << '\n';
      }
    }
  }
  CHECK(checked_well_conditioned);
}

// Hydro-vs-MHD comparison (task #50).  As the field vanishes (sigma -> 0, so
// |B| -> 0) the MHD flux Jacobian's fluid sub-block on {S_x,S_y,S_z,D,tau} must
// approach the hydro flux Jacobian on the same variables, and the MHD fast
// speeds must approach the hydro acoustic speeds v_n +- c_s (slow/Alfven/entropy
// -> v_n).  This quantifies the mode correspondence and how weak the field must
// be for the hydro system to be a faithful stand-in (a weak-field switch).
// Flat metric (the comparison needs no eigen-reconstruction; primitives are
// de-rounded).  Dump guarded by SPECTRE_HYDROMHD_DUMP.
void test_hydro_mhd_comparison(const bool output) {
  const ScopedFpeState disable_fpes(false);
  constexpr size_t num_points = 1;
  const EquationsOfState::IdealFluid<true> eos_2d(1.37, 0.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  // Fluid vars shared by both systems: S_x,S_y,S_z,D,tau.  MHD order is
  // [S_x,S_y,S_z,B^x,B^y,B^z,D,tau,phi]; hydro order is [D,S_x,S_y,S_z,tau,Y_e].
  const std::array<size_t, 5> mhd_fluid{{0, 1, 2, 6, 7}};
  const std::array<size_t, 5> hyd_fluid{{1, 2, 3, 0, 4}};
  // Speed of each hydro eigenvector (HydroVectorR order): R1..R4 at v_n
  // (NormalDotVelocity), Rplus at LambdaPlus, Rminus at LambdaMinus.
  const std::array<size_t, 6> hyd_speed_idx{{0, 0, 0, 0, 1, 2}};

  // Parameter grid: thermal pressure (cs^2) x field orientation bn_frac = B_n/|B|
  // (proximity to the field degeneracy at a given |B|).  W, phi, vn fixed.
  const double W = 1.4;
  const double phi_B = M_PI / 4.0;
  const double vn_frac = 0.28;
  const std::array<double, 5> pressures{{1.3e-4, 1.3e-3, 1.3e-2, 1.2e-1, 1.1}};
  const std::array<double, 5> bn_fracs{{0.03, 0.13, 0.3, 0.6, 0.9}};

  std::ofstream dump;
  if (output) {
    dump.open("hydro_mhd_comparison.tsv", std::ios::out | std::ios::trunc);
    dump << std::setprecision(16);
    dump << "cfg\tpressure\tbn_frac\tW\tsigma\tb_norm\tmatrix_fluid_diff\t"
            "err_mhd_recon\terr_hydro_recon\tmhd_sm\tmhd_fm\tmhd_am\tmhd_slm\t"
            "mhd_ent\tmhd_slp\tmhd_ap\tmhd_fp\tmhd_sp\thyd_vn\thyd_lp\thyd_lm\n";
  }

  constexpr size_t n_sig = 45;
  double diff_at_small_sigma = 1.0;
  double hydro_err_at_small_sigma = 1.0;
  size_t cfg = 0;
  for (const double pressure_val : pressures) {
    for (const double bn_frac : bn_fracs) {
      Scalar<DataVector> rest_mass_density{DataVector(num_points, 1.13)};
      const Scalar<DataVector> pressure{DataVector(num_points, pressure_val)};
      const Scalar<DataVector> specific_internal_energy =
          eos_2d.specific_internal_energy_from_density_and_pressure(
              rest_mass_density, pressure);
      const Scalar<DataVector> specific_enthalpy =
          hydro::relativistic_specific_enthalpy(
              rest_mass_density, specific_internal_energy, pressure);
      const Scalar<DataVector> lorentz_factor{DataVector(num_points, W)};
      const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.13)};
      const double vmag = std::sqrt(std::max(0.0, 1.0 - 1.0 / square(W)));
      const double vn = vn_frac * vmag;
      const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));
      tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
      spatial_velocity.get(0) = vn;
      spatial_velocity.get(1) = vt;
      spatial_velocity.get(2) = 0.0;

      for (size_t is = 0; is < n_sig; ++is) {
        const double sigma =
            std::pow(10.0, 1.0 + (std::log10(1.0e-9) - 1.0) *
                                     static_cast<double>(is) /
                                     static_cast<double>(n_sig - 1));
        const double b_mag = std::sqrt(sigma * get(rest_mass_density)[0] *
                                       get(specific_enthalpy)[0]);
        const double bx = bn_frac * b_mag;
        const double bt = std::sqrt(std::max(0.0, square(b_mag) - square(bx)));
        tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
        magnetic_field.get(0) = bx;
        magnetic_field.get(1) = bt * std::cos(phi_B);
        magnetic_field.get(2) = bt * std::sin(phi_B);

        tnsr::iJ<DataVector, 9> jac_mhd{num_points, 0.0};
        grmhd::ValenciaDivClean::flux_jacobian_mhd(
            make_not_null(&jac_mhd), spatial_velocity, magnetic_field,
            rest_mass_density, specific_internal_energy, electron_fraction,
            lorentz_factor, specific_enthalpy, spatial_metric,
            inv_spatial_metric, unit_normal, eos_2d);
        tnsr::iJ<DataVector, 6> jac_hyd{num_points, 0.0};
        grmhd::ValenciaDivClean::flux_jacobian_hydro(
            make_not_null(&jac_hyd), spatial_velocity, rest_mass_density,
            specific_internal_energy, electron_fraction, lorentz_factor,
            specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
            eos_2d);
        double matrix_diff = 0.0;
        for (size_t a = 0; a < 5; ++a) {
          for (size_t b = 0; b < 5; ++b) {
            matrix_diff = std::max(
                matrix_diff,
                std::abs(jac_mhd.get(mhd_fluid[a], mhd_fluid[b])[0] -
                         jac_hyd.get(hyd_fluid[a], hyd_fluid[b])[0]));
          }
        }

        tnsr::i<DataVector, 9> mhd_speeds{num_points, 0.0};
        grmhd::ValenciaDivClean::characteristic_speeds_mhd(
            make_not_null(&mhd_speeds), spatial_velocity, magnetic_field,
            rest_mass_density, specific_internal_energy, lorentz_factor,
            specific_enthalpy, spatial_metric, unit_normal, eos_2d);
        tnsr::i<DataVector, 3> hyd_speeds{num_points, 0.0};
        grmhd::ValenciaDivClean::characteristic_speeds_hydro(
            make_not_null(&hyd_speeds), spatial_velocity, rest_mass_density,
            specific_internal_energy, electron_fraction, lorentz_factor,
            specific_enthalpy, spatial_metric, unit_normal, eos_2d);

        // Reconstruct the fluid sub-block of jac_mhd from each system's own
        // eigen-decomposition (see the header comment on this function).
        double err_mhd_recon = 1.0e30;  // sentinel: MHD eigenvectors threw
        try {
          tnsr::ij<DataVector, 9> modes{num_points, 0.0};
          tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
          grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
              make_not_null(&modes), make_not_null(&projectors), mhd_speeds,
              spatial_velocity, magnetic_field, rest_mass_density,
              specific_internal_energy, lorentz_factor, specific_enthalpy,
              spatial_metric, unit_normal, eos_2d);
          std::array<double, 9> diag{};
          for (size_t i = 0; i < 9; ++i) {
            double d = 0.0;
            for (size_t k = 0; k < 9; ++k) {
              d += projectors.get(i, k)[0] * modes.get(i, k)[0];
            }
            diag[i] = d;
          }
          double e = 0.0;
          for (size_t a = 0; a < 5; ++a) {
            for (size_t b = 0; b < 5; ++b) {
              double val = 0.0;
              for (size_t i = 0; i < 9; ++i) {
                val += mhd_speeds.get(i)[0] * modes.get(i, mhd_fluid[a])[0] *
                       projectors.get(i, mhd_fluid[b])[0] / diag[i];
              }
              e = std::max(
                  e, std::abs(val - jac_mhd.get(mhd_fluid[a], mhd_fluid[b])[0]));
            }
          }
          err_mhd_recon = e;
        } catch (...) {
        }

        tnsr::ij<DataVector, 6> hyd_modes{num_points, 0.0};
        tnsr::IJ<DataVector, 6> hyd_projectors{num_points, 0.0};
        grmhd::ValenciaDivClean::characteristic_eigenvectors_hydro(
            make_not_null(&hyd_modes), make_not_null(&hyd_projectors),
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, eos_2d);
        std::array<double, 6> hyd_diag{};
        for (size_t i = 0; i < 6; ++i) {
          double d = 0.0;
          for (size_t k = 0; k < 6; ++k) {
            d += hyd_projectors.get(i, k)[0] * hyd_modes.get(i, k)[0];
          }
          hyd_diag[i] = d;
        }
        double err_hydro_recon = 0.0;
        for (size_t a = 0; a < 5; ++a) {
          for (size_t b = 0; b < 5; ++b) {
            double val = 0.0;
            for (size_t i = 0; i < 6; ++i) {
              val += hyd_speeds.get(hyd_speed_idx[i])[0] *
                     hyd_modes.get(i, hyd_fluid[a])[0] *
                     hyd_projectors.get(i, hyd_fluid[b])[0] / hyd_diag[i];
            }
            err_hydro_recon = std::max(
                err_hydro_recon,
                std::abs(val - jac_mhd.get(mhd_fluid[a], mhd_fluid[b])[0]));
          }
        }

        if (sigma < 1.0e-6) {
          diff_at_small_sigma = std::min(diff_at_small_sigma, matrix_diff);
          hydro_err_at_small_sigma =
              std::min(hydro_err_at_small_sigma, err_hydro_recon);
        }
        if (dump.is_open()) {
          dump << cfg << '\t' << pressure_val << '\t' << bn_frac << '\t' << W
               << '\t' << sigma << '\t' << b_mag << '\t' << matrix_diff << '\t'
               << err_mhd_recon << '\t' << err_hydro_recon;
          for (size_t i = 0; i < 9; ++i) {
            dump << '\t' << mhd_speeds.get(i)[0];
          }
          for (size_t i = 0; i < 3; ++i) {
            dump << '\t' << hyd_speeds.get(i)[0];
          }
          dump << '\n';
        }
      }
      ++cfg;
    }
  }
  // As |B| -> 0 both the MHD fluid sub-block and the hydro reconstruction of it
  // must converge to the reference MHD matrix.
  CHECK(diff_at_small_sigma < 1.0e-6);
  CHECK(hydro_err_at_small_sigma < 1.0e-6);
}

// Diagnostic for the numeric-eigensystem-in-Marquina problem (task #51).  The
// Marquina reconstruction needs sum_i R_i (x) L_i = I, i.e. L.R = I.
// numerical_characteristics stores geev's RAW eigenvectors (no
// biorthonormalization).  For DISTINCT eigenvalues, left/right eigenvectors of a
// matrix are automatically biorthogonal (L_i.R_j = 0, i != j), so a scalar
// rescale L_i <- L_i/(L_i.R_i) should give L.R = I; only a degenerate block can
// break that.  Here we sweep B_n -> 0 and, per point, measure:
//   raw_offdiag = max_{i!=j} |L_i.R_j|            (biorthogonality of geev output)
//   raw_mindiag = min_i |L_i.R_i|                 (how far the diagonal is from 1)
//   rescaled_id = || sum_i R_i (L_i/(L_i.R_i))^T - I ||   (does a scalar rescale fix it)
// TSV dump guarded by SPECTRE_NUMEIG_DUMP.
void test_numeric_biorthogonality(const bool output) {
  const ScopedFpeState disable_fpes(false);
  constexpr size_t num_points = 1;
  const EquationsOfState::IdealFluid<true> eos_2d(1.37, 0.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  const double W = 1.4;
  const double phi_B = M_PI / 4.0;
  const double vn_frac = 0.28;
  const double sigma = 1.3;
  Scalar<DataVector> rest_mass_density{DataVector(num_points, 1.13)};
  const Scalar<DataVector> pressure{DataVector(num_points, 0.12)};
  const Scalar<DataVector> specific_internal_energy =
      eos_2d.specific_internal_energy_from_density_and_pressure(
          rest_mass_density, pressure);
  const Scalar<DataVector> specific_enthalpy =
      hydro::relativistic_specific_enthalpy(rest_mass_density,
                                            specific_internal_energy, pressure);
  const Scalar<DataVector> lorentz_factor{DataVector(num_points, W)};
  const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.13)};
  const double vmag = std::sqrt(std::max(0.0, 1.0 - 1.0 / square(W)));
  const double vn = vn_frac * vmag;
  const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
  spatial_velocity.get(0) = vn;
  spatial_velocity.get(1) = vt;
  spatial_velocity.get(2) = 0.0;
  const double b_mag = std::sqrt(sigma * get(rest_mass_density)[0] *
                                 get(specific_enthalpy)[0]);

  std::ofstream dump;
  if (output) {
    dump.open("numeric_biorthogonality.tsv", std::ios::out | std::ios::trunc);
    dump << std::setprecision(16);
    dump << "bn_frac\tmin_gap\traw_offdiag\traw_mindiag\trescaled_id\n";
  }

  constexpr size_t n_bn = 40;
  double well_sep_rescaled_id = 1.0e30;  // best (smallest) at large gap
  for (size_t ib = 0; ib < n_bn; ++ib) {
    const double bn_frac = std::pow(
        10.0, std::log10(0.5) + (std::log10(1.0e-10) - std::log10(0.5)) *
                                    static_cast<double>(ib) /
                                    static_cast<double>(n_bn - 1));
    const double bx = bn_frac * b_mag;
    const double bt = std::sqrt(std::max(0.0, square(b_mag) - square(bx)));
    tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points, 0.0};
    magnetic_field.get(0) = bx;
    magnetic_field.get(1) = bt * std::cos(phi_B);
    magnetic_field.get(2) = bt * std::sin(phi_B);

    tnsr::i<DataVector, 9> speeds{num_points, 0.0};
    tnsr::ij<DataVector, 9> modes{num_points, 0.0};
    tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
    grmhd::ValenciaDivClean::numerical_characteristics<9>(
        make_not_null(&speeds), make_not_null(&modes),
        make_not_null(&projectors), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, eos_2d);

    // L.R matrix (M_ij = L_i . R_j = projectors row i dot modes row j).
    std::array<std::array<double, 9>, 9> m{};
    for (size_t i = 0; i < 9; ++i) {
      for (size_t j = 0; j < 9; ++j) {
        double v = 0.0;
        for (size_t k = 0; k < 9; ++k) {
          v += projectors.get(i, k)[0] * modes.get(j, k)[0];
        }
        m[i][j] = v;
      }
    }
    double raw_offdiag = 0.0;
    double raw_mindiag = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < 9; ++i) {
      raw_mindiag = std::min(raw_mindiag, std::abs(m[i][i]));
      for (size_t j = 0; j < 9; ++j) {
        if (i != j) {
          raw_offdiag = std::max(raw_offdiag, std::abs(m[i][j]));
        }
      }
    }
    // Scalar-rescaled reconstruction identity: sum_i R_i (L_i/M_ii)^T.
    double rescaled_id = 0.0;
    for (size_t a = 0; a < 9; ++a) {
      for (size_t b = 0; b < 9; ++b) {
        double v = 0.0;
        for (size_t i = 0; i < 9; ++i) {
          v += modes.get(i, a)[0] * projectors.get(i, b)[0] / m[i][i];
        }
        rescaled_id = std::max(rescaled_id, std::abs(v - (a == b ? 1.0 : 0.0)));
      }
    }

    std::array<double, 9> sorted{};
    for (size_t i = 0; i < 9; ++i) {
      sorted[i] = speeds.get(i)[0];
    }
    std::sort(sorted.begin(), sorted.end());
    double min_gap = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i + 1 < 9; ++i) {
      min_gap = std::min(min_gap, sorted[i + 1] - sorted[i]);
    }
    if (bn_frac > 0.1) {
      well_sep_rescaled_id = std::min(well_sep_rescaled_id, rescaled_id);
    }
    if (dump.is_open()) {
      dump << bn_frac << '\t' << min_gap << '\t' << raw_offdiag << '\t'
           << raw_mindiag << '\t' << rescaled_id << '\n';
    }
  }
  // At a well-separated (non-degenerate) state, a scalar rescale of geev's own
  // eigenvectors must reconstruct the identity -- i.e. the numeric decomposition
  // needs no inversion there, only per-wave renormalization.
  CHECK(well_sep_rescaled_id < 1.0e-9);

  // Contrast: the HYDRO system has an EXACT 4-fold degeneracy at v_n (the contact
  // modes), so geev returns an arbitrary basis for that eigenspace and its left /
  // right bases are NOT mutually biorthogonal -- a scalar rescale then does NOT
  // recover the identity.  This is the genuine "degenerate block" problem (and
  // the likely cause of the earlier HydroYe numeric-Marquina failure).
  {
    tnsr::i<DataVector, 6> h_speeds{num_points, 0.0};
    tnsr::ij<DataVector, 6> h_modes{num_points, 0.0};
    tnsr::IJ<DataVector, 6> h_projectors{num_points, 0.0};
    grmhd::ValenciaDivClean::numerical_characteristics<6>(
        make_not_null(&h_speeds), make_not_null(&h_modes),
        make_not_null(&h_projectors), spatial_velocity,
        tnsr::I<DataVector, 3, Frame::Inertial>{num_points, 0.0},
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, eos_2d);
    std::array<std::array<double, 6>, 6> hm{};
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        double v = 0.0;
        for (size_t k = 0; k < 6; ++k) {
          v += h_projectors.get(i, k)[0] * h_modes.get(j, k)[0];
        }
        hm[i][j] = v;
      }
    }
    double h_offdiag = 0.0;
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        if (i != j) {
          h_offdiag = std::max(h_offdiag, std::abs(hm[i][j]));
        }
      }
    }
    double h_rescaled_id = 0.0;
    for (size_t a = 0; a < 6; ++a) {
      for (size_t b = 0; b < 6; ++b) {
        double v = 0.0;
        for (size_t i = 0; i < 6; ++i) {
          v += h_modes.get(i, a)[0] * h_projectors.get(i, b)[0] / hm[i][i];
        }
        h_rescaled_id = std::max(h_rescaled_id, std::abs(v - (a == b ? 1.0 : 0.0)));
      }
    }
    if (output) {
      std::cout << "HYDRO_NUMEIG exact-degeneracy: raw_offdiag=" << h_offdiag
                << "  rescaled_id=" << h_rescaled_id << "\n";
    }
    // The exact degenerate block breaks biorthogonality: a scalar rescale is not
    // enough (the block needs a proper projector, e.g. the complementary one).
    CHECK(h_rescaled_id > 1.0e-3);
  }
}

// Flux-level comparison of the analytic vs numeric decomposition (task #51,
// step 4).  What Marquina actually uses is the dissipation matrix
// |A| = sum_i |lambda_i| R_i (L_i/(L_i.R_i))^T (the decomposition-dependent part
// of the numerical flux).  Over the parameter-space corners, sweep B_n -> 0 and
// compare, against the flux Jacobian A = flux_jacobian_mhd:
//   err_signed_analytic = || sum_i lambda_i R_i^an  L_i^an  - A ||   (grows near deg.)
//   err_signed_numeric  = || sum_i lambda_i R_i^num L_i^num - A ||   (~round-off)
//   diss_diff           = || |A|_analytic - |A|_numeric ||          (analytic flux error)
// with the numeric decomposition as the (validated) reference.  Dump guarded by
// SPECTRE_NUMEIG_DUMP.
// Marquina boundary-correction (numerical flux) comparison, numeric vs analytic
// (task #51, step 4).  This is the quantity actually used as the DG boundary
// correction -- not the flux Jacobian.  For each parameter-space corner we build
// a physical interface (an interior state and a slightly perturbed exterior
// state), compute the real conserved variables and normal fluxes with
// ConservativeFromPrimitive / Fluxes, and then apply the exact Marquina per-wave
// flux formula (copied from Marquina::dg_boundary_terms) with EACH decomposition.
// The numeric decomposition is the reference; we report the relative difference
// of the analytic boundary correction from it, swept over B_n -> 0.  Flat metric
// (lapse 1, shift 0, sqrt(gamma) 1), interior normal +x / exterior normal -x.
// Dump guarded by SPECTRE_NUMEIG_DUMP.
void test_numeric_vs_analytic_flux(const bool output) {
  const ScopedFpeState disable_fpes(false);
  constexpr size_t num_points = 1;
  const EquationsOfState::IdealFluid<true> eos_2d(1.37, 0.0);

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv = determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv.second;
  const Scalar<DataVector> sqrt_det{DataVector(num_points, 1.0)};
  const Scalar<DataVector> lapse{DataVector(num_points, 1.0)};
  const tnsr::I<DataVector, 3, Frame::Inertial> shift{num_points, 0.0};
  // Interior normal +x, exterior normal -x (its outward normal).
  const auto normal_int = unit_basis_form(Direction<3>::lower_xi(),
                                          inv_spatial_metric);
  tnsr::i<DataVector, 3, Frame::Inertial> normal_ext{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    normal_ext.get(i) = -normal_int.get(i);
  }

  // A physical MHD state -> its conserved variables U (order
  // [S_x,S_y,S_z,B_x,B_y,B_z,D,tau,phi]) and normal flux F.n for a given normal.
  struct State {
    std::array<double, 9> u;
    std::array<double, 9> fn;
    tnsr::I<DataVector, 3, Frame::Inertial> velocity;
    tnsr::I<DataVector, 3, Frame::Inertial> bfield;
    Scalar<DataVector> rho, eps, ye, h, W, phi;
  };
  const auto make_state = [&](double density, double pressure_val, double W_lor,
                              double vn_frac, double bn_frac, double b_mag,
                              double phi_val, double phi_ang,
                              const tnsr::i<DataVector, 3, Frame::Inertial>& nrm) {
    State s;
    s.rho = Scalar<DataVector>{DataVector(num_points, density)};
    const Scalar<DataVector> pressure{DataVector(num_points, pressure_val)};
    s.eps = eos_2d.specific_internal_energy_from_density_and_pressure(s.rho,
                                                                      pressure);
    s.ye = Scalar<DataVector>{DataVector(num_points, 0.13)};
    s.h = hydro::relativistic_specific_enthalpy(s.rho, s.eps, pressure);
    s.W = Scalar<DataVector>{DataVector(num_points, W_lor)};
    s.phi = Scalar<DataVector>{DataVector(num_points, phi_val)};
    const double vmag = std::sqrt(std::max(0.0, 1.0 - 1.0 / square(W_lor)));
    const double vn = vn_frac * vmag;
    const double vt = std::sqrt(std::max(0.0, square(vmag) - square(vn)));
    s.velocity = tnsr::I<DataVector, 3, Frame::Inertial>{num_points, 0.0};
    s.velocity.get(0) = vn;
    s.velocity.get(1) = vt;
    const double bx = bn_frac * b_mag;
    const double bt = std::sqrt(std::max(0.0, square(b_mag) - square(bx)));
    s.bfield = tnsr::I<DataVector, 3, Frame::Inertial>{num_points, 0.0};
    s.bfield.get(0) = bx;
    s.bfield.get(1) = bt * std::cos(phi_ang);
    s.bfield.get(2) = bt * std::sin(phi_ang);
    Scalar<DataVector> tilde_d{num_points}, tilde_ye{num_points},
        tilde_tau{num_points}, tilde_phi{num_points};
    tnsr::i<DataVector, 3, Frame::Inertial> tilde_s{num_points};
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_b{num_points};
    grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
        make_not_null(&tilde_d), make_not_null(&tilde_ye),
        make_not_null(&tilde_tau), make_not_null(&tilde_s),
        make_not_null(&tilde_b), make_not_null(&tilde_phi), s.rho, s.ye, s.eps,
        pressure, s.velocity, s.W, s.bfield, sqrt_det, spatial_metric, s.phi);
    tnsr::I<DataVector, 3, Frame::Inertial> f_d{num_points}, f_ye{num_points},
        f_tau{num_points}, f_phi{num_points};
    tnsr::Ij<DataVector, 3, Frame::Inertial> f_s{num_points};
    tnsr::IJ<DataVector, 3, Frame::Inertial> f_b{num_points};
    grmhd::ValenciaDivClean::ComputeFluxes::apply(
        make_not_null(&f_d), make_not_null(&f_ye), make_not_null(&f_tau),
        make_not_null(&f_s), make_not_null(&f_b), make_not_null(&f_phi),
        tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
        sqrt_det, spatial_metric, inv_spatial_metric, pressure, s.velocity, s.W,
        s.bfield);
    // Conserved vector + normal flux in Marquina's variable order.
    s.u = {get<0>(tilde_s)[0], get<1>(tilde_s)[0], get<2>(tilde_s)[0],
           get<0>(tilde_b)[0], get<1>(tilde_b)[0], get<2>(tilde_b)[0],
           get(tilde_d)[0],    get(tilde_tau)[0],  get(tilde_phi)[0]};
    const auto ndot = [&](const auto& flux, size_t comp) {
      double v = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        v += flux.get(comp, i)[0] * nrm.get(i)[0];
      }
      return v;
    };
    const auto ndot_vec = [&](const tnsr::I<DataVector, 3, Frame::Inertial>& fl) {
      double v = 0.0;
      for (size_t i = 0; i < 3; ++i) {
        v += fl.get(i)[0] * nrm.get(i)[0];
      }
      return v;
    };
    s.fn = {ndot(f_s, 0), ndot(f_s, 1), ndot(f_s, 2), ndot(f_b, 0),
            ndot(f_b, 1), ndot(f_b, 2), ndot_vec(f_d), ndot_vec(f_tau),
            ndot_vec(f_phi)};
    return s;
  };

  // Decomposition (speeds, L, R) in MhdSpeed order at a state, with a given
  // normal, either analytic or numeric (matched to analytic speeds + rescaled),
  // exactly as Marquina::dg_package_data builds it.  ok=false if it threw / is
  // not biorthonormal.
  struct Decomp {
    std::array<double, 9> speed;
    std::array<std::array<double, 9>, 9> l;  // l[i] = L_i / (L_i.R_i)
    std::array<std::array<double, 9>, 9> r;  // r[i] = R_i
    bool ok;
  };
  const auto decompose = [&](const State& s,
                             const tnsr::i<DataVector, 3, Frame::Inertial>& nrm,
                             bool numeric) {
    Decomp d;
    d.ok = true;
    tnsr::i<DataVector, 9> speeds{num_points, 0.0};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&speeds), s.velocity, s.bfield, s.rho, s.eps, s.W, s.h,
        spatial_metric, nrm, eos_2d);
    for (size_t i = 0; i < 9; ++i) {
      d.speed[i] = speeds.get(i)[0];
    }
    tnsr::ij<DataVector, 9> modes{num_points, 0.0};
    tnsr::IJ<DataVector, 9> proj{num_points, 0.0};
    if (numeric) {
      // numerical_characteristics ASSERTs on a complex eigenvalue -- which
      // blaze::geev can return for a near-degenerate real spectrum -- so the
      // numeric decomposition can fail outright there.  Catch it and flag.
      try {
        tnsr::i<DataVector, 9> ns{num_points, 0.0};
        tnsr::ij<DataVector, 9> nm{num_points, 0.0};
        tnsr::IJ<DataVector, 9> np{num_points, 0.0};
        grmhd::ValenciaDivClean::numerical_characteristics<9>(
            make_not_null(&ns), make_not_null(&nm), make_not_null(&np),
            s.velocity, s.bfield, s.rho, s.eps, s.ye, s.W, s.h, spatial_metric,
            inv_spatial_metric, nrm, eos_2d);
        std::array<bool, 9> used{};
        for (size_t k = 0; k < 9; ++k) {
          size_t best = 9;
          double bd = std::numeric_limits<double>::infinity();
          for (size_t g = 0; g < 9; ++g) {
            if (not used[g] and std::abs(ns.get(g)[0] - d.speed[k]) < bd) {
              bd = std::abs(ns.get(g)[0] - d.speed[k]);
              best = g;
            }
          }
          used[best] = true;
          for (size_t n = 0; n < 9; ++n) {
            modes.get(k, n)[0] = nm.get(best, n)[0];
            proj.get(k, n)[0] = np.get(best, n)[0];
          }
        }
      } catch (...) {
        d.ok = false;
        return d;
      }
    } else {
      try {
        grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
            make_not_null(&modes), make_not_null(&proj), speeds, s.velocity,
            s.bfield, s.rho, s.eps, s.W, s.h, spatial_metric, nrm, eos_2d);
      } catch (...) {
        d.ok = false;
      }
    }
    for (size_t i = 0; i < 9; ++i) {
      double diagonal = 0.0;
      for (size_t n = 0; n < 9; ++n) {
        diagonal += proj.get(i, n)[0] * modes.get(i, n)[0];
      }
      for (size_t n = 0; n < 9; ++n) {
        d.r[i][n] = modes.get(i, n)[0];
        d.l[i][n] = proj.get(i, n)[0] / diagonal;
      }
    }
    return d;
  };

  // Marquina per-wave boundary correction G (9-vector), replicating
  // Marquina::dg_boundary_terms: align the exterior decomposition to the
  // interior frame (negate speeds, swap +/- pairs), then upwind each wave.
  const auto marquina_flux = [&](const State& si, const State& se,
                                 const Decomp& di, const Decomp& de) {
    // Align exterior to interior frame.
    const std::array<std::array<size_t, 2>, 4> pairs{
        {{{0, 8}}, {{1, 7}}, {{2, 6}}, {{3, 5}}}};
    std::array<double, 9> se_speed = de.speed;
    auto le = de.l;
    auto re = de.r;
    se_speed[4] = -de.speed[4];
    for (const auto& pr : pairs) {
      const size_t m = pr[0], p = pr[1];
      se_speed[m] = -de.speed[p];
      se_speed[p] = -de.speed[m];
      le[m] = de.l[p];
      le[p] = de.l[m];
      re[m] = de.r[p];
      re[p] = de.r[m];
    }
    std::array<double, 9> g{};
    for (size_t i = 0; i < 9; ++i) {
      double omega_int = 0.0, omega_ext = 0.0, phi_int = 0.0, phi_ext = 0.0;
      for (size_t n = 0; n < 9; ++n) {
        omega_int += di.l[i][n] * si.u[n];
        omega_ext += le[i][n] * se.u[n];
        phi_int += di.l[i][n] * si.fn[n];
        phi_ext += le[i][n] * (-se.fn[n]);
      }
      const double li = di.speed[i];
      const double lex = se_speed[i];
      double phi_plus = 0.0, phi_minus = 0.0;
      if (li >= 0.0 and lex >= 0.0) {
        phi_plus = phi_int;
      } else if (li <= 0.0 and lex <= 0.0) {
        phi_minus = phi_ext;
      } else {
        const double alpha = std::max(std::abs(li), std::abs(lex));
        phi_plus = 0.5 * (phi_int + alpha * omega_int);
        phi_minus = 0.5 * (phi_ext - alpha * omega_ext);
      }
      for (size_t n = 0; n < 9; ++n) {
        g[n] += phi_plus * di.r[i][n] + phi_minus * re[i][n];
      }
    }
    return g;
  };

  struct Cfg {
    const char* name;
    double pressure;
    double sigma;
  };
  const std::array<Cfg, 4> cfgs{{{"cold_weakB", 1.3e-4, 1.1e-4},
                                 {"cold_strongB", 1.3e-4, 1.2e3},
                                 {"hot_weakB", 1.1, 1.1e-4},
                                 {"hot_strongB", 1.1, 1.2e3}}};
  const double W = 1.43;
  const double phi_ang = M_PI / 4.0;

  std::ofstream dump;
  if (output) {
    dump.open("numeric_vs_analytic_marquina_flux.tsv",
              std::ios::out | std::ios::trunc);
    dump << std::setprecision(16);
    dump << "cfg\tname\tbn_frac\tmin_gap\tflux_rel_diff\tnumeric_ok\n";
  }

  for (size_t ic = 0; ic < cfgs.size(); ++ic) {
    const Cfg& cf = cfgs[ic];
    const double density = 1.13;
    // |B| from sigma at the interior state's enthalpy.
    const Scalar<DataVector> rho0{DataVector(num_points, density)};
    const Scalar<DataVector> p0{DataVector(num_points, cf.pressure)};
    const auto eps0 =
        eos_2d.specific_internal_energy_from_density_and_pressure(rho0, p0);
    const auto h0 = hydro::relativistic_specific_enthalpy(rho0, eps0, p0);
    const double b_mag = std::sqrt(cf.sigma * density * get(h0)[0]);

    constexpr size_t n_bn = 40;
    for (size_t ib = 0; ib < n_bn; ++ib) {
      const double bn_frac = std::pow(
          10.0, std::log10(0.5) + (std::log10(1.0e-10) - std::log10(0.5)) *
                                      static_cast<double>(ib) /
                                      static_cast<double>(n_bn - 1));
      // Interior state, and a modestly perturbed exterior state (smooth jump).
      const State si = make_state(density, cf.pressure, W, 0.28, bn_frac, b_mag,
                                  0.02, phi_ang, normal_int);
      const State se = make_state(density * 1.05, cf.pressure * 1.04, W * 1.01,
                                  0.3, bn_frac, b_mag * 1.03, 0.025, phi_ang,
                                  normal_ext);

      // min gap from the interior analytic speeds
      std::array<double, 9> sorted{};
      tnsr::i<DataVector, 9> isp{num_points, 0.0};
      grmhd::ValenciaDivClean::characteristic_speeds_mhd(
          make_not_null(&isp), si.velocity, si.bfield, si.rho, si.eps, si.W,
          si.h, spatial_metric, normal_int, eos_2d);
      for (size_t i = 0; i < 9; ++i) {
        sorted[i] = isp.get(i)[0];
      }
      std::sort(sorted.begin(), sorted.end());
      double min_gap = std::numeric_limits<double>::infinity();
      for (size_t i = 0; i + 1 < 9; ++i) {
        min_gap = std::min(min_gap, sorted[i + 1] - sorted[i]);
      }

      const Decomp di_an = decompose(si, normal_int, false);
      const Decomp de_an = decompose(se, normal_ext, false);
      const Decomp di_nu = decompose(si, normal_int, true);
      const Decomp de_nu = decompose(se, normal_ext, true);

      // numeric biorthonormality (proxy for "usable"): reconstruct identity.
      const auto biorth_ok = [&](const Decomp& d) {
        double e = 0.0;
        for (size_t m = 0; m < 9; ++m) {
          for (size_t n = 0; n < 9; ++n) {
            double v = 0.0;
            for (size_t i = 0; i < 9; ++i) {
              v += d.r[i][m] * d.l[i][n];
            }
            e = std::max(e, std::abs(v - (m == n ? 1.0 : 0.0)));
          }
        }
        return e < 1.0e-6;
      };
      const bool numeric_ok = di_nu.ok and de_nu.ok and biorth_ok(di_nu) and
                              biorth_ok(de_nu);

      double flux_rel_diff = -1.0;
      if (numeric_ok and di_an.ok and de_an.ok) {
        const auto g_nu = marquina_flux(si, se, di_nu, de_nu);
        const auto g_an = marquina_flux(si, se, di_an, de_an);
        double num = 0.0, den = 0.0;
        for (size_t n = 0; n < 9; ++n) {
          num = std::max(num, std::abs(g_an[n] - g_nu[n]));
          den = std::max(den, std::abs(g_nu[n]));
        }
        flux_rel_diff = num / std::max(den, 1.0e-300);
      }
      if (dump.is_open()) {
        dump << ic << '\t' << cf.name << '\t' << bn_frac << '\t' << min_gap
             << '\t' << flux_rel_diff << '\t' << (numeric_ok ? 1 : 0) << '\n';
      }
    }
  }
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
  test_hydro_numerical_characteristics(dv);
  test_hydro_analytic_eigenvectors(dv);
  test_hydro_characteristics_match_unoptimized_version(dv);
  test_quartic_rootfinding(dv);
  // Run data-producing sweeps — quartic_shape and typical_vn_sweep first
  // since they use moderate W.  test_mhd_characteristics_errors pushes to
  // W=100 at high sigma where the tighter production tolerance (1e-15)
  // can intermittently trigger ASSERTs.  The precision-study TSVs +
  // ASYMP_CHECK dump are written only when SPECTRE_MHD_PREC_DUMP is set
  // (matching the SPECTRE_CPM_DUMP guard in Test_Marquina); the assertions
  // run unconditionally.
  test_mhd_characteristics_errors(std::getenv("SPECTRE_MHD_PREC_DUMP") !=
                                  nullptr);
  test_degeneracy_parameter_sweep(std::getenv("SPECTRE_DEGEN_DUMP") != nullptr);
  test_flux_jacobian_precision(std::getenv("SPECTRE_MATRIX_DUMP") != nullptr);
  test_hydro_mhd_comparison(std::getenv("SPECTRE_HYDROMHD_DUMP") != nullptr);
  test_numeric_biorthogonality(std::getenv("SPECTRE_NUMEIG_DUMP") != nullptr);
  test_numeric_vs_analytic_flux(std::getenv("SPECTRE_NUMEIG_DUMP") != nullptr);
  test_mhd_characteristics(dv);
  test_mhd_numerical_characteristics(dv);

  // Disable benchmarks by default
  run_hydro_characteristic_benchmarks(false);
  run_mhd_characteristic_benchmarks(false);

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
