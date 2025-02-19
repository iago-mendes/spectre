// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <algorithm>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/IndexIterator.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/NumericalAlgorithms/SphericalHarmonics/YlmTestFunctions.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Literals.hpp"

namespace {
void test_definite_integral_0d() {
  const double value = 1.234;
  const DataVector data(1, value);
  CHECK(value == approx(definite_integral(data, Mesh<0>{})));
}

void test_definite_integral_1d(const Mesh<1>& mesh) {
  const DataVector& x = Spectral::collocation_points(mesh);
  DataVector integrand(mesh.number_of_grid_points());
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t s = 0; s < integrand.size(); ++s) {
      integrand[s] = pow(x[s], a);
    }
    if (0 == a % 2) {
      CHECK(2.0 / (a + 1.0) == approx(definite_integral(integrand, mesh)));
    } else {
      CHECK(0.0 == approx(definite_integral(integrand, mesh)));
    }
  }
}

// Test for finite difference methods using midpoint rule.
void test_midpoint_integral_1d(const Mesh<1>& mesh) {
  const DataVector& x = Spectral::collocation_points(mesh);
  const size_t number_of_points = mesh.number_of_grid_points();
  DataVector integrand(number_of_points);
  for (size_t a = 0; a < 3; ++a) {
    for (size_t s = 0; s < integrand.size(); ++s) {
      integrand[s] = pow(x[s], a);
    }
    if (0 == a) {
      CHECK(2.0 == approx(definite_integral(integrand, mesh)));
    }
    if (1 == a) {
      CHECK(0.0 == approx(definite_integral(integrand, mesh)));
    }
    if (2 ==a) {
      // Correct error should be \f$2/(3N^2)\f$ for N points in the mesh
      CHECK(2.0/3.0*(1.0-1.0/(number_of_points*number_of_points)) ==
            approx(definite_integral(integrand, mesh)));
    }
  }
}


void test_definite_integral_2d(const Mesh<2>& mesh) {
  const DataVector& x = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& y = Spectral::collocation_points(mesh.slice_through(1));
  DataVector integrand(mesh.number_of_grid_points());
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t b = 0; b < mesh.extents(1); ++b) {
      for (IndexIterator<2> index_it(mesh.extents()); index_it; ++index_it) {
        integrand[index_it.collapsed_index()] =
            pow(x[index_it()[0]], a) * pow(y[index_it()[1]], b);
      }
      if (0 == a % 2 and 0 == b % 2) {
        CHECK(4.0 / ((a + 1.0) * (b + 1.0)) ==
              approx(definite_integral(integrand, mesh)));
      } else {
        CHECK(0.0 == approx(definite_integral(integrand, mesh)));
      }
    }
  }
}

void test_definite_integral_3d(const Mesh<3>& mesh) {
  const DataVector& x = Spectral::collocation_points(mesh.slice_through(0));
  const DataVector& y = Spectral::collocation_points(mesh.slice_through(1));
  const DataVector& z = Spectral::collocation_points(mesh.slice_through(2));
  DataVector integrand(mesh.number_of_grid_points());
  for (size_t a = 0; a < mesh.extents(0); ++a) {
    for (size_t b = 0; b < mesh.extents(1); ++b) {
      for (size_t c = 0; c < mesh.extents(2); ++c) {
        for (IndexIterator<3> index_it(mesh.extents()); index_it; ++index_it) {
          integrand[index_it.collapsed_index()] = pow(x[index_it()[0]], a) *
                                                  pow(y[index_it()[1]], b) *
                                                  pow(z[index_it()[2]], c);
        }
        if (0 == a % 2 and 0 == b % 2 and 0 == c % 2) {
          CHECK(8.0 / ((a + 1.0) * (b + 1.0) * (c + 1.0)) ==
                approx(definite_integral(integrand, mesh)));
        } else {
          CHECK(0.0 == approx(definite_integral(integrand, mesh)));
        }
      }
    }
  }
}

void test_definite_integral_spherical_shell(const size_t n_r, const size_t L) {
  CAPTURE(n_r);
  CAPTURE(L);
  const Mesh<3> mesh{
      {n_r, L + 1, 2 * L + 1},
      {Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
       Spectral::Basis::SphericalHarmonic},
      {Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
       Spectral::Quadrature::Equiangular}};
  const auto xi_vector = logical_coordinates(mesh);
  const DataVector r = xi_vector.get(0) + 2.0;
  for (size_t pow_nx = 0; pow_nx <= L; ++pow_nx) {
    CAPTURE(pow_nx);
    for (size_t pow_ny = 0; pow_ny <= L - pow_nx; ++pow_ny) {
      CAPTURE(pow_ny);
      for (size_t pow_nz = 0; pow_nz <= L - pow_nx - pow_ny; ++pow_nz) {
        CAPTURE(pow_nz);
        const YlmTestFunctions::ProductOfPolynomials y_lm{
            n_r, L, L, pow_nx, pow_ny, pow_nz};

        const DataVector integrand = r * y_lm.f();
        CHECK(4.0 * y_lm.definite_integral() ==
              approx(definite_integral(integrand, mesh)));
      }
    }
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Numerical.LinearOperators.DefiniteIntegral",
                  "[NumericalAlgorithms][LinearOperators][Unit]") {
  test_definite_integral_0d();

  constexpr size_t min_extents =
      Spectral::minimum_number_of_points<Spectral::Basis::Legendre,
                                         Spectral::Quadrature::GaussLobatto>;
  constexpr size_t max_extents =
      Spectral::maximum_number_of_points<Spectral::Basis::Legendre>;
  for (size_t n0 = min_extents; n0 <= max_extents; ++n0) {
    test_definite_integral_1d(Mesh<1>{n0, Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto});
  }
  for (size_t n0 = min_extents; n0 <= max_extents; ++n0) {
    for (size_t n1 = min_extents; n1 <= max_extents - 1; ++n1) {
      test_definite_integral_2d(Mesh<2>{{{n0, n1}},
                                        Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto});
    }
  }
  for (size_t n0 = min_extents; n0 <= std::min(6_st, max_extents); ++n0) {
    for (size_t n1 = min_extents; n1 <= std::min(7_st, max_extents); ++n1) {
      for (size_t n2 = min_extents; n2 <= std::min(8_st, max_extents); ++n2) {
        test_definite_integral_3d(Mesh<3>{{{n0, n1, n2}},
                                          Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto});
      }
    }
  }

  for (size_t n_r = 2; n_r < 5; ++n_r) {
    for (size_t L = 2; L < 9; ++L) {
      test_definite_integral_spherical_shell(n_r, L);
    }
  }

  // Test finite difference integral
  constexpr size_t min_extents_fd =
      Spectral::minimum_number_of_points<Spectral::Basis::FiniteDifference,
                                         Spectral::Quadrature::CellCentered>;
  constexpr size_t max_extents_fd =
    Spectral::maximum_number_of_points<Spectral::Basis::FiniteDifference>;
  for (size_t n0 = min_extents_fd; n0 <= max_extents_fd; ++n0) {
    test_midpoint_integral_1d(Mesh<1>{n0, Spectral::Basis::FiniteDifference,
                                      Spectral::Quadrature::CellCentered});
  }
}
