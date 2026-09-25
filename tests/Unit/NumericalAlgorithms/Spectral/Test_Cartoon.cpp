// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/DifferentiationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/ModalToNodalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/NodalToModalMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/QuadratureWeights.hpp"

// The Basis::Cartoon should never be called to generate collocation points or
// weights, so it errors on everything

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonAxialSymmetry.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::collocation_points<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
  CHECK_THROWS_WITH(
      (Spectral::quadrature_weights<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::differentiation_matrix<
          Spectral::Basis::Cartoon, Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.ModalToNodal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::modal_to_nodal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonAxialSymmetry.NodalToModal",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::nodal_to_modal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::AxialSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.PointsAndWeights",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::collocation_points<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
  CHECK_THROWS_WITH(
      (Spectral::quadrature_weights<Spectral::Basis::Cartoon,
                                    Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonTranslationalSymmetry.DiffMatrix",
    "[NumericalAlgorithms][Spectral][Unit]") {
  // The translational Cartoon is the exception to the rule above. The axial
  // and spherical ones are differentiated by the `cartoon_*` operators, which
  // handle the non-Cartoon directions themselves, so asking them for a
  // differentiation matrix means something has gone wrong and they error. A
  // translationally collapsed direction has no geometric terms and is
  // differentiated by the ordinary Cartesian operators, which DO ask for this
  // matrix. It is the 1x1 zero matrix: one grid point, and a logical
  // derivative that vanishes identically.
  const Matrix& diff_matrix = Spectral::differentiation_matrix<
      Spectral::Basis::Cartoon, Spectral::Quadrature::TranslationalSymmetry>(1);
  CHECK(diff_matrix.rows() == 1);
  CHECK(diff_matrix.columns() == 1);
  CHECK(diff_matrix(0, 0) == 0.0);
  // Collocation points remain an error for it, as for every Cartoon
  // quadrature -- the zero matrix is returned without consulting them.
  CHECK_THROWS_WITH(
      (Spectral::collocation_points<
          Spectral::Basis::Cartoon,
          Spectral::Quadrature::TranslationalSymmetry>(1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE("Unit.Numerical.Spectral.CartoonSphericalSymmetry.DiffMatrix",
                  "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::differentiation_matrix<
          Spectral::Basis::Cartoon, Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.ModalToNodal",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::modal_to_nodal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}

SPECTRE_TEST_CASE(
    "Unit.Numerical.Spectral.CartoonSphericalSymmetry.NodalToModal",
    "[NumericalAlgorithms][Spectral][Unit]") {
  CHECK_THROWS_WITH(
      (Spectral::nodal_to_modal_matrix<Spectral::Basis::Cartoon,
                                       Spectral::Quadrature::SphericalSymmetry>(
          1)),
      Catch::Matchers::ContainsSubstring(
          "Invalid to compute collocation points and weights for a Cartoon "
          "basis."));
}
