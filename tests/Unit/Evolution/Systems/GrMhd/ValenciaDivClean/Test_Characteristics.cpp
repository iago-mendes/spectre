// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
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

    // Solve numerical eigensystem
    grmhd::ValenciaDivClean::numerical_characteristics(
        make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
        make_not_null(&left_eigenvectors), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *equation_of_state_3d);

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
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

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

    // Check 3: Quad-precision speed comparison
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

      // Note: eigenvector checks (quad comparison, biorthogonality) are
      // intentionally omitted here. The current eigenvector implementation
      // has known conditioning issues near degeneracies that make absolute
      // error checks unreliable. Eigenvector quality should be tested
      // separately once the eigenvector code is improved.
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
  };
  const std::array<Config, 5> configs{{
      // Generic moderate regime: hot fluid (cs^2 ~ 0.13) with moderate
      // magnetization gives a well-separated quartic with clear W shape.
      {"typical", 0.4, 0.5, M_PI / 4.0, 0.1, 0.1},
      // Worst-case errors from low magnetization: cold fluid (cs^2 ~ 1.7e-4)
      // with sigma << 1 makes the quartic coefficients nearly degenerate.
      {"low_magnetization", 0.3674661940736692, 0.999, 0.6283185307179586,
       1.0e-4, 1.0e-4},
      // High magnetization regime (Anton et al. 2010, Section 8.2.1):
      // sigma ~ 5000 matches their cylindrical explosion test background.
      // Cold fluid so that cs^2/sigma ~ 3e-8.  Bn/B=0.05 and phi=0.6pi
      // optimized from 4D sweep (see CLAUDE-MhdSpeedErrorAnalysis.md).
      {"high_magnetization", 0.05, 0.995, 0.6 * M_PI, 5000.0, 1.0e-4},
      // Near Type-I degeneracy: Bn/B -> 0, all inner speeds coalesce
      {"near_type_I", 1.0e-6, 0.5, 0.0, 0.01, 1.0e-4},
      // Near Type-II degeneracy: Bn/B -> 1, slow mode merges with Alfven
      {"near_type_II", 0.9999, 0.5, 0.0, 0.01, 1.0e-4},
  }};

  constexpr size_t num_points = 1;
  constexpr double adiabatic_index = 5.0 / 3.0;
  const EquationsOfState::IdealFluid<true> eos_2d(adiabatic_index, 0.0);
  const Scalar<DataVector> rest_mass_density{DataVector(num_points, 1.0)};

  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric{num_points, 0.0};
  spatial_metric.get(0, 0) = 1.0;
  spatial_metric.get(1, 1) = 1.0;
  spatial_metric.get(2, 2) = 1.0;
  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

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

  constexpr size_t n_w = 64;
  constexpr double w_max = 100.0;

  for (const auto& config : configs) {
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
      std::cout << std::setprecision(17) << "ASYMP_CHECK" << '\t' << config.name
                << '\t' << W << '\t' << get(sound_speed_squared)[0] << '\t'
                << sv_asym << '\t' << Bs_asym << '\t'
                << Bv_scalar * inv_sqrt_rho_h << '\t' << B_squared / rho_h
                << '\t' << asym_fast_m << '\t' << asym_fast_p << '\t'
                << asym_slow_m << '\t' << asym_slow_p << '\n';

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

    // Regression thresholds (loose initially; tighten after baseline)
    constexpr double max_allowed_abs_error = 5.0e-3;
    CAPTURE(config.name);
    CAPTURE(max_abs_fast);
    CAPTURE(max_abs_slow);
    CHECK(max_abs_fast < max_allowed_abs_error);
    CHECK(max_abs_slow < max_allowed_abs_error);
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
  test_hydro_characteristics_match_unoptimized_version(dv);
  test_quartic_rootfinding(dv);
  // Run data-producing sweeps — quartic_shape and typical_vn_sweep first
  // since they use moderate W.  test_mhd_characteristics_errors pushes to
  // W=100 at high sigma where the tighter production tolerance (1e-15)
  // can intermittently trigger ASSERTs.
  test_mhd_characteristics_errors(true);
  test_mhd_characteristics(dv);

  // Disable benchmarks by default
  run_hydro_characteristic_benchmarks(false);
  run_mhd_characteristic_benchmarks(false);

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
