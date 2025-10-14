// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_characteristic_speeds(const DataVector& /*used_for_size*/) {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  // const double max_value = 1.0 / sqrt(3);
  // pypp::check_with_random_values<7>(
  //     &grmhd::ValenciaDivClean::characteristic_speeds<3>, "TestFunctions",
  //     "characteristic_speeds",
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
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
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
  Scalar<DataVector> alfven_speed_squared{
      comoving_magnetic_field_squared /
      (comoving_magnetic_field_squared +
       get(rest_mass_density) * get(specific_enthalpy))};
  Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(specific_enthalpy)};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    Approx custom_approx = Approx::custom().epsilon(1.0e-10);
    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::approx::characteristic_speeds(
            rest_mass_density, electron_fraction, specific_internal_energy,
            specific_enthalpy, spatial_velocity, lorentz_factor, magnetic_field,
            lapse, shift, spatial_metric, normal, eos_base),
        (pypp::call<std::array<DataVector, 9>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)),
        custom_approx);
  }
}

// TODO: debug random test failure
void test_hydro_numerical_eigensystem(const DataVector& used_for_size) {
  // Initialize number generator
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);

  // Generate random variables used to get primitive variables
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(nn_gen, used_for_size);
  const auto equation_of_state = EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto rest_mass_density =
      TestHelpers::hydro::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      TestHelpers::hydro::random_specific_internal_energy(nn_gen,
                                                          used_for_size);
  const auto pressure = equation_of_state.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);

  // Generate random primitive variables
  const auto spatial_velocity = TestHelpers::hydro::random_velocity(
      nn_gen, lorentz_factor, spatial_metric);
  // rest_mass_density already defined above
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const auto& direction = Direction<3>::all_directions()[0];
  const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

  const auto normal_velocity =
      tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
  // std::cout << "\tnormal_velocity = " << std::defaultfloat << normal_velocity
  // << std::endl;

  // std::cout << "Starting numerical eigensystem test..." << std::endl;

  // std::cout << std::endl << "Random variables:" << std::endl
  //           << "\tspatial_velocity = " << spatial_velocity << std::endl
  //           << "\trest_mass_density = " << rest_mass_density << std::endl
  //           << "\tspecific_internal_energy = " << specific_internal_energy
  //           << std::endl
  //           << "\tlorentz_factor = " << lorentz_factor << std::endl
  //           << "\tspatial_metric = " << spatial_metric << std::endl
  //           << "\tinv_spatial_metric = " << inv_spatial_metric << std::endl
  //           << "\tunit_normal = " << unit_normal << std::endl
  //           << "\tspecific_enthalpy = " << specific_enthalpy << std::endl;

  // Solve numerical eigensystem
  const auto eigensystem = grmhd::ValenciaDivClean::numerical_eigensystem(
      spatial_velocity, rest_mass_density, specific_internal_energy,
      lorentz_factor, spatial_metric, inv_spatial_metric, unit_normal,
      equation_of_state, specific_enthalpy);
  const auto& eigenvalues = eigensystem.first;
  const auto& right_eigenvectors = eigensystem.second.first;
  const auto& left_eigenvectors = eigensystem.second.second;

  // std::cout << "\tEigenvalues: " << eigenvalues << std::endl;

  // std::cout << "... finished numerical eigensystem test!" << std::endl;

  // std::cout << "Building expected matrix..." << std::endl;
  tnsr::iJ<DataVector, 5> expected_characteristic_matrix =
      make_with_value<tnsr::iJ<DataVector, 5>>(spatial_metric, 0.0);
  grmhd::ValenciaDivClean::detail::flux_jacobian_hydro(
      make_not_null(&expected_characteristic_matrix), spatial_velocity,
      rest_mass_density, specific_internal_energy,
      /* other helpful quantities */
      lorentz_factor, spatial_metric, inv_spatial_metric, unit_normal,
      equation_of_state, specific_enthalpy);
  // std::cout << "... built expected matrix!" << std::endl;

  // Loop over each point and verify solution
  double old_error = 0.0;
  double error = 0.0;
  const size_t num_points = used_for_size.size();
  constexpr size_t matrix_size = 5;
  for (size_t i = 0; i < num_points; ++i) {
    std::cout << "point " << i << ":" << std::endl;

    // Reconstruct characteristic matrix
    // std::cout << "Getting point expected matrix..." << std::endl;
    Matrix point_expected_characteristic_matrix(matrix_size, matrix_size, 0.0);
    for (size_t c = 0; c < matrix_size; ++c) {
      for (size_t B = 0; B < matrix_size; ++B) {
        // point_expected_characteristic_matrix(B,c) =
        // expected_characteristic_matrix.get(B,c)[i];
        point_expected_characteristic_matrix(c, B) =
            expected_characteristic_matrix.get(c, B)[i];
      }
    }

    // Reconstruct the eigenvector matrices (L, R) and diagonal eigenvalue
    // matrix (D)
    // std::cout << "Getting L, R, D..." << std::endl;
    Matrix R(matrix_size, matrix_size);
    Matrix L(matrix_size, matrix_size);
    Matrix D(matrix_size, matrix_size, 0.0);
    for (size_t j = 0; j < matrix_size;
         ++j) {  // Loop over eigenvalues / eigenvectors
      D(j, j) = eigenvalues[j][i];
      for (size_t k = 0; k < matrix_size; ++k) {  // Loop over vector components
        R(k, j) = right_eigenvectors[j].get(k)[i];
        L(k, j) = left_eigenvectors[j].get(k)[i];
      }
    }

    // Renormalize left eigenvectors so that L^T * R = I
    // std::cout << "Renormalizing eigenvectors..." << std::endl;
    Matrix L_rescaled = L;
    for (size_t j = 0; j < matrix_size; ++j) {
      const auto r_j = blaze::column(R, j);
      const auto l_j = blaze::column(L, j);
      const double dot = blaze::trans(l_j) * r_j;
      //   const double ratio = blaze::norm(l_j) / blaze::norm(r_j);
      // std::cout << "\tl_" << j << " . x_" << j << " = " << dot << std::endl;
      // std::cout << "\tl_" << l_j << " . x_" << j << " = " << dot <<
      // std::endl;
      blaze::column(L_rescaled, j) /= dot;
    }

    int degenerate_eigenvalues_count = 0;
    for (size_t j = 0; j < matrix_size; ++j) {
      const double diff = std::abs(eigenvalues[j][i] - get(normal_velocity)[i]);
      std::cout << "\t| (y_" << j << " - v_n) / v_n | = " << std::scientific
                << diff << std::endl;
      if (diff < 1e-10) {
        degenerate_eigenvalues_count += 1;
      }
    }
    CHECK(degenerate_eigenvalues_count == 3);
    // for (size_t j = 0; j < matrix_size; ++j) {
    //   std::cout << "\tx_" << j << " = (" << blaze::column(R, j)[0] << ", " <<
    //   blaze::column(R, j)[1] << ", " << blaze::column(R, j)[2] << ", " <<
    //   blaze::column(R, j)[3] << ", " << blaze::column(R, j)[4] << ")" <<
    //   std::endl;
    // }
    // for (size_t j = 0; j < matrix_size; ++j) {
    //   std::cout << "\tl_" << j << " = (" << blaze::column(L, j)[0] << ", " <<
    //   blaze::column(L, j)[1] << ", " << blaze::column(L, j)[2] << ", " <<
    //   blaze::column(L, j)[3] << ", " << blaze::column(L, j)[4] << ")" <<
    //   std::endl;
    //   // std::cout << "\tl_" << j << " = " << blaze::column(L, j) <<
    //   std::endl;
    // }

    // 1. Check the right eigenvector equation: A * R = R * D
    const Matrix AR = point_expected_characteristic_matrix * R;
    const Matrix RD = R * D;

    // 2. Check the left eigenvector equation: trans(L) * A = D * trans(L)
    const Matrix LT = trans(L);
    const Matrix LTA = LT * point_expected_characteristic_matrix;
    const Matrix DLT = D * LT;

    // Verify reconstructed identity and characteristic matrix
    // std::cout << "Verifying result..." << std::endl;
    const Matrix reconstructed_identity = blaze::trans(L_rescaled) * R;
    const Matrix reconstructed_matrix = R * D * blaze::trans(L_rescaled);
    Approx approx = Approx::custom().epsilon(1e-10).scale(1.0);
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        const double delta_ij = (row == col) ? 1.0 : 0.0;
        // CHECK(reconstructed_identity(row, col) == approx(delta_ij));
        // CHECK(reconstructed_matrix(row, col) ==
        //       approx(point_expected_characteristic_matrix(row, col)));
        old_error = std::max(
            old_error, std::abs(reconstructed_identity(row, col) - delta_ij));
        old_error =
            std::max(old_error,
                     std::abs(reconstructed_matrix(row, col) -
                              point_expected_characteristic_matrix(row, col)));
        error = std::max(error, std::abs(AR(row, col) - RD(row, col)));
        error = std::max(error, std::abs(LTA(row, col) - DLT(row, col)));
      }
    }
  }
  //   std::cout << "Old error = " << old_error << std::endl;
  std::cout << "Error = " << std::scientific << error << std::endl;
}

struct SomeEosType {};

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  // test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  // test_with_normal_along_coordinate_axes(dv);
  for (size_t i = 0; i < 100; ++i) {
    std::cout << std::endl << "Test " << i + 1 << std::endl;
    test_hydro_numerical_eigensystem(dv);
  }

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
