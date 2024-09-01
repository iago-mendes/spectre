// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <iostream>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

namespace {

std::ofstream output_file("Test_CenterOfMass.output");
using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

/**
 * This test shifts the isotropic Schwarzschild solution and checks that the
 * center of mass corresponds to the coordinate shift.
 */
void test_center_of_mass_surface_integral(const double distance, const size_t L,
                                          const size_t P) {
  // Get Schwarzschild solution in isotropic coordinates.
  // const double mass = 1;
  // const Xcts::Solutions::Schwarzschild solution(
  //     mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
  // const double horizon_radius = 0.5 * mass;

  const double mass = 1.;
  const double horizon_radius = 2. * mass;
  const double boost_speed = 0.5;
  const std::array<double, 3> boost_velocity({0., 0., boost_speed});
  const std::array<double, 3> dimensionless_spin({0., 0., 0.});
  const std::array<double, 3> center({0., 0., 0.});
  const KerrSchild solution(mass, dimensionless_spin, center, boost_velocity);

  // Define z-shift applied to the coordinates.
  const double z_shift = 0. * mass;

  // Set up domain
  const size_t h_refinement = L;
  const size_t p_refinement = P;
  const domain::creators::Sphere shell{
      /* inner_radius */ z_shift + 2 * horizon_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ {},
      /* radial_distribution */ domain::CoordinateMaps::Distribution::Inverse};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<3> mesh{p_refinement + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize "reduced" integral
  tnsr::I<double, 3> total_integral({0., 0., 0.});
  tnsr::I<double, 3> surface_integral({0., 0., 0.});
  tnsr::I<double, 3> volume_integral({0., 0., 0.});

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Get 3D coordinates
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = logical_to_inertial_map(logical_coords);
    const auto jacobian = logical_to_inertial_map.jacobian(logical_coords);
    const auto det_jacobian = determinant(jacobian);
    const auto inv_jacobian =
        logical_to_inertial_map.inv_jacobian(logical_coords);

    // Shift coordinates used to get analytic solution
    auto shifted_coords = inertial_coords;
    shifted_coords.get(2) -= z_shift;

    // Get required fields
    const auto shifted_fields = solution.variables(
        shifted_coords,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            ::Tags::deriv<
                Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3,
                                               Frame::Inertial>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(shifted_fields);
    const auto& deriv_conformal_factor =
        get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(shifted_fields);
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            shifted_fields);
    const auto& deriv_conformal_metric = get<::Tags::deriv<
        Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(shifted_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            shifted_fields);

    // Compute Euclidean unit normal vector
    const auto r_coordinate = magnitude(inertial_coords);
    const auto euclidean_unit_normal =
        tenex::evaluate<ti::I>(inertial_coords(ti::I) / r_coordinate());

    const auto volume_integrand = Xcts::center_of_mass_volume_integrand(
        conformal_factor, deriv_conformal_factor, euclidean_unit_normal,
        deriv_conformal_metric);
    for (int I = 0; I < 3; I++) {
      total_integral.get(I) += definite_integral(
          volume_integrand.get(I) * get(det_jacobian),
          mesh);
      volume_integral.get(I) +=
          definite_integral(volume_integrand.get(I) * get(det_jacobian), mesh);
    }

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the inner boundary
      if (boundary_direction != Direction<3>::lower_zeta()) {
        continue;
      }

      // Get interface coordinates.
      const auto face_logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto face_inv_jacobian =
          logical_to_inertial_map.inv_jacobian(face_logical_coords);

      // Slice required fields to the interface
      const size_t slice_index =
          index_to_slice_at(mesh.extents(), boundary_direction);
      const auto& face_conformal_factor =
          data_on_slice(conformal_factor, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_inv_conformal_metric =
          data_on_slice(inv_conformal_metric, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_euclidean_unit_normal =
          data_on_slice(euclidean_unit_normal, mesh.extents(),
                        boundary_direction.dimension(), slice_index);

      // Compute Euclidean area element
      const auto area_element = euclidean_area_element(
          face_inv_jacobian, boundary_direction);

      // Compute Euclidean face normal
      auto euclidean_face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude = magnitude(euclidean_face_normal);
      for (size_t d = 0; d < 3; ++d) {
        euclidean_face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Integrate
      const auto surface_integrand = Xcts::center_of_mass_surface_integrand(
          face_conformal_factor, face_euclidean_unit_normal);
      const auto contracted_integrand = tenex::evaluate<ti::I>(
          -surface_integrand(ti::I, ti::J) * euclidean_face_normal(ti::j));
      for (int I = 0; I < 3; I++) {
        total_integral.get(I) += definite_integral(
            contracted_integrand.get(I) * get(area_element),
            face_mesh);
        surface_integral.get(I) += definite_integral(
            contracted_integrand.get(I) * get(area_element), face_mesh);
      }
    }
  }

  // Check result
  // auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  // CHECK(get<0>(total_integral) == custom_approx(0.));
  // CHECK(get<1>(total_integral) == custom_approx(0.));
  // CHECK(get<2>(total_integral) / mass == custom_approx(z_shift));

  output_file << std::setprecision(16)           //
              << distance                        //
              << ", "                            //
              << L                               //
              << ", "                            //
              << P                               //
              << ", "                            //
              << get(magnitude(total_integral))  //
              << ", "                            //
              << z_shift                         //
              << std::endl;                      //

  std::cout << std::setprecision(16)     //
            << "\t Center of Mass \t"    //
            << distance                  //
            << ", "                      //
            << get<2>(surface_integral)  //
            << ", "                      //
            << get<2>(volume_integral)   //
            << ", "                      //
            << get<2>(total_integral)    //
            << " == "                    //
            << z_shift                   //
            << std::endl;                //
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.CenterOfMass",
                  "[Unit][PointwiseFunctions]") {
  // Test that integral converges with distance.
  // for (const double distance : std::array<double, 3>({1.e3, 1.e4, 1.e5})) {
  // for (const double distance :
  //      std::array<double, 5>({1.e1, 1.e2, 1.e3, 1.e5, 1.e10})) {
  for (const double distance : std::array<double, 8>(
           {1.e1, 1.e2, 1.e3, 1.e4, 1.e5, 1.e6, 1.e7, 1.e8})) {
    for (size_t L = 0; L <= 2; L++) {
      for (size_t P = 2; P <= 15; P++) {
        test_center_of_mass_surface_integral(distance, L, P);
      }
    }
  }
}
