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
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"
#include "PointwiseFunctions/Xcts/AdmMass.hpp"

namespace {

using Schwarzschild = Xcts::Solutions::Schwarzschild;
using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

template <typename Solution>
void test_mass_surface_integral(const double distance, const double mass,
                                const double boost_speed,
                                const Solution& solution,
                                const double horizon_radius) {
  // Set up domain
  const size_t h_refinement = 1;
  const size_t p_refinement = 6;
  domain::creators::Sphere shell{
      /* inner_radius */ 2 * horizon_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      /* initial_refinement */ h_refinement,
      /* initial_number_of_grid_points */ p_refinement + 1,
      /* use_equiangular_map */ true,
      /* equatorial_compression */ {},
      /* radial_partitioning */ {},
      /* radial_distribution */
      domain::CoordinateMaps::Distribution::Logarithmic};
  // /* radial_partitioning */ std::vector<double>{{60.}},
  // /* radial_distribution */
  // std::vector<domain::CoordinateMaps::Distribution>{{
      // domain::CoordinateMaps::Distribution::Logarithmic,
      // domain::CoordinateMaps::Distribution::Inverse}}};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<3> mesh{p_refinement + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{p_refinement + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize "reduced" integral.
  Scalar<double> total_integral(0.);
  Scalar<double> surface_integral(0.);
  Scalar<double> volume_integral(0.);

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

    // Get required fields
    const auto background_fields = solution.variables(
        inertial_coords, mesh, inv_jacobian,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            ::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalRicciScalar<DataVector>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
                DataVector>,
            Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            ::Tags::deriv<
                Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>,
            ::Tags::deriv<Xcts::Tags::ConformalChristoffelSecondKind<
                              DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(background_fields);
    const auto& deriv_conformal_factor =
        get<::Tags::deriv<Xcts::Tags::ConformalFactorMinusOne<DataVector>,
                          tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& conformal_ricci_scalar =
        get<Xcts::Tags::ConformalRicciScalar<DataVector>>(background_fields);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);
    const auto& longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
        get<Xcts::Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<
            DataVector>>(background_fields);
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& deriv_conformal_metric = get<::Tags::deriv<
        Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& conformal_christoffel_second_kind =
        get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>>(
            background_fields);
    const auto& deriv_conformal_christoffel_second_kind =
        get<::Tags::deriv<Xcts::Tags::ConformalChristoffelSecondKind<
                              DataVector, 3, Frame::Inertial>,
                          tmpl::size_t<3>, Frame::Inertial>>(background_fields);
    const auto& conformal_christoffel_contracted =
        get<Xcts::Tags::ConformalChristoffelContracted<DataVector, 3,
                                                       Frame::Inertial>>(
            background_fields);

    const auto deriv_inv_conformal_metric =
        tenex::evaluate<ti::i, ti::J, ti::K>(
            inv_conformal_metric(ti::J, ti::L) *
                inv_conformal_metric(ti::K, ti::M) *
                (deriv_conformal_metric(ti::i, ti::l, ti::m) -
                 conformal_christoffel_second_kind(ti::N, ti::i, ti::l) *
                     conformal_metric(ti::n, ti::m) -
                 conformal_christoffel_second_kind(ti::N, ti::i, ti::m) *
                     conformal_metric(ti::l, ti::n)) -
            conformal_christoffel_second_kind(ti::J, ti::i, ti::l) *
                inv_conformal_metric(ti::L, ti::K) -
            conformal_christoffel_second_kind(ti::K, ti::i, ti::l) *
                inv_conformal_metric(ti::J, ti::L));

    // Evaluate volume integral.
    const auto volume_integrand = Xcts::adm_mass_volume_integrand(
        conformal_factor, conformal_ricci_scalar, trace_extrinsic_curvature,
        longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
        inv_conformal_metric, deriv_inv_conformal_metric,
        conformal_christoffel_second_kind,
        deriv_conformal_christoffel_second_kind);
    total_integral.get() += definite_integral(
        get(volume_integrand) * get(det_jacobian),
        mesh);
    volume_integral.get() +=
        definite_integral(get(volume_integrand) * get(det_jacobian), mesh);

    // Loop over external boundaries.
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the inner boundary.
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
      const auto& face_deriv_conformal_factor =
          data_on_slice(deriv_conformal_factor, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_conformal_metric =
          data_on_slice(conformal_metric, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_inv_conformal_metric =
          data_on_slice(inv_conformal_metric, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_conformal_christoffel_second_kind =
          data_on_slice(conformal_christoffel_second_kind, mesh.extents(),
                        boundary_direction.dimension(), slice_index);
      const auto& face_conformal_christoffel_contracted =
          data_on_slice(conformal_christoffel_contracted, mesh.extents(),
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

      // Evaluate surface integral.
      const auto surface_integrand = Xcts::adm_mass_surface_integrand(
          face_deriv_conformal_factor, face_inv_conformal_metric,
          face_conformal_christoffel_second_kind,
          face_conformal_christoffel_contracted);
      const auto contracted_integrand = tenex::evaluate(
          -surface_integrand(ti::I) * euclidean_face_normal(ti::i));

      // Compute contribution to surface integral
      total_integral.get() += definite_integral(
          get(contracted_integrand) * get(area_element), face_mesh);
      surface_integral.get() += definite_integral(
          get(contracted_integrand) * get(area_element), face_mesh);
    }
  }

  // Check result
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));
  // auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  // CHECK(get(total_integral) == custom_approx(lorentz_factor * mass));

  std::cout << std::setprecision(16)  //
            << "\t ADM Mass \t"       //
            << distance               //
            << ", "                   //
            << get(surface_integral)  //
            << ", "                   //
            << get(volume_integral)   //
            << ", "                   //
            << get(total_integral)    //
            << " == "                 //
            << lorentz_factor * mass  //
            << std::endl;             //
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.AdmMass",
                  "[Unit][PointwiseFunctions]") {
  // Test that integral converges with two analytic solutions.
  {
    INFO("Boosted Kerr-Schild");
    const double mass = 1.;
    const double horizon_radius = 2. * mass;
    const double boost_speed = 0.;
    const std::array<double, 3> boost_velocity({0., 0., boost_speed});
    const std::array<double, 3> dimensionless_spin({0., 0., 0.});
    const std::array<double, 3> center({0., 0., 0.});
    const KerrSchild solution(mass, dimensionless_spin, center, boost_velocity);
    // for (const double distance : std::array<double, 3>({1.e3, 1.e5, 1.e10}))
    // {
    for (const double distance :
         std::array<double, 4>({1.e3, .2e4, 1.e5, 1.e10})) {
      test_mass_surface_integral(distance, mass, boost_speed, solution,
                                 horizon_radius);
    }
  }
  {
    INFO("Isotropic Schwarzschild");
    const double mass = 1.;
    const double horizon_radius = 0.5 * mass;
    const double boost_speed = 0.;
    const Schwarzschild solution(
        mass, Xcts::Solutions::SchwarzschildCoordinates::Isotropic);
    for (const double distance :
         std::array<double, 4>({1.e3, 1.e4, 1.e5, 1.e10})) {
      test_mass_surface_integral(distance, mass, boost_speed, solution,
                                 horizon_radius);
    }
  }
}
