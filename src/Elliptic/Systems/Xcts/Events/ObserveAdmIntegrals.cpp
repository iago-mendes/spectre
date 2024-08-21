// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Xcts/Events/ObserveAdmIntegrals.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"
#include "PointwiseFunctions/Xcts/AdmMass.hpp"
#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

namespace Events {

void local_adm_integrals(
    gsl::not_null<Scalar<double>*> adm_mass,
    gsl::not_null<tnsr::I<double, 3>*> adm_linear_momentum,
    gsl::not_null<tnsr::I<double, 3>*> center_of_mass,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3>& deriv_conformal_factor,
    const tnsr::ii<DataVector, 3>& conformal_metric,
    const tnsr::ijj<DataVector, 3>& deriv_conformal_metric,
    const tnsr::II<DataVector, 3>& inv_conformal_metric,
    const tnsr::Ijj<DataVector, 3>& conformal_christoffel_second_kind,
    const tnsr::iJkk<DataVector, 3>& deriv_conformal_christoffel_second_kind,
    const tnsr::i<DataVector, 3>& conformal_christoffel_contracted,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::II<DataVector, 3>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature,
    const Scalar<DataVector>& trace_extrinsic_curvature,
    const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& conformal_ricci_scalar,
    const tnsr::II<DataVector, 3>& longitudinal_shift_excess,
    const tnsr::II<DataVector, 3>&
        longitudinal_shift_background_minus_dt_conformal_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>& inv_jacobian,
    const Mesh<3>& mesh, const Element<3>& element,
    const DirectionMap<3, tnsr::i<DataVector, 3>>& conformal_face_normals,
    const DirectionMap<3, tnsr::I<DataVector, 3>>&
        conformal_face_normal_vectors) {
  // Initialize integrals to 0
  adm_mass->get() = 0;
  for (int I = 0; I < 3; I++) {
    adm_linear_momentum->get(I) = 0;
    center_of_mass->get(I) = 0;
  }

  // Compute the inverse extrinsic curvature
  tnsr::II<DataVector, 3> inv_extrinsic_curvature;
  tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_extrinsic_curvature),
                                inv_spatial_metric(ti::I, ti::K) *
                                    inv_spatial_metric(ti::J, ti::L) *
                                    extrinsic_curvature(ti::k, ti::l));

  const auto linear_momentum_surface_integrand =
      Xcts::adm_linear_momentum_surface_integrand(
          conformal_factor, inv_spatial_metric, inv_extrinsic_curvature,
          trace_extrinsic_curvature);

  const auto r_coordinate = magnitude(inertial_coords);
  const auto euclidean_unit_normal_vector =
      tenex::evaluate<ti::I>(inertial_coords(ti::I) / r_coordinate());

  const auto deriv_inv_conformal_metric = tenex::evaluate<ti::i, ti::J, ti::K>(
      inv_conformal_metric(ti::J, ti::L) * inv_conformal_metric(ti::K, ti::M) *
          (deriv_conformal_metric(ti::i, ti::l, ti::m) -
           conformal_christoffel_second_kind(ti::N, ti::i, ti::l) *
               conformal_metric(ti::n, ti::m) -
           conformal_christoffel_second_kind(ti::N, ti::i, ti::m) *
               conformal_metric(ti::l, ti::n)) -
      conformal_christoffel_second_kind(ti::J, ti::i, ti::l) *
          inv_conformal_metric(ti::L, ti::K) -
      conformal_christoffel_second_kind(ti::K, ti::i, ti::l) *
          inv_conformal_metric(ti::J, ti::L));

  const auto longitudinal_shift_minus_dt_conformal_metric_over_lapse_square =
      tenex::evaluate(conformal_metric(ti::i, ti::k) *
                      conformal_metric(ti::j, ti::l) *
                      (longitudinal_shift_excess(ti::I, ti::J) +
                       longitudinal_shift_background_minus_dt_conformal_metric(
                           ti::I, ti::J)) *
                      (longitudinal_shift_excess(ti::K, ti::L) +
                       longitudinal_shift_background_minus_dt_conformal_metric(
                           ti::K, ti::L)) /
                      square(lapse()));

  // Get volume integrands.
  const auto mass_volume_integrand = Xcts::adm_mass_volume_integrand(
      conformal_factor, conformal_ricci_scalar, trace_extrinsic_curvature,
      longitudinal_shift_minus_dt_conformal_metric_over_lapse_square,
      inv_conformal_metric, deriv_inv_conformal_metric,
      conformal_christoffel_second_kind,
      deriv_conformal_christoffel_second_kind);
  const auto linear_momentum_volume_integrand =
      Xcts::adm_linear_momentum_volume_integrand(
          linear_momentum_surface_integrand, conformal_factor,
          deriv_conformal_factor, conformal_metric, inv_conformal_metric,
          conformal_christoffel_second_kind, conformal_christoffel_contracted);
  const auto center_of_mass_volume_integrand =
      Xcts::center_of_mass_volume_integrand(
          conformal_factor, deriv_conformal_factor,
          euclidean_unit_normal_vector, deriv_conformal_metric);

  // Evaluate volume integrals.
  const auto det_jacobian =
      Scalar<DataVector>(1. / get(determinant(inv_jacobian)));
  adm_mass->get() +=
      definite_integral(get(mass_volume_integrand) * get(det_jacobian), mesh);
  for (int I = 0; I < 3; I++) {
    adm_linear_momentum->get(I) += definite_integral(
        linear_momentum_volume_integrand.get(I) * get(det_jacobian), mesh);
    center_of_mass->get(I) += definite_integral(
        center_of_mass_volume_integrand.get(I) * get(det_jacobian), mesh);
  }

  // Loop over external boundaries
  for (const auto boundary_direction : element.external_boundaries()) {
    // Skip interfaces not at the inner boundary
    if (boundary_direction != Direction<3>::lower_zeta()) {
      continue;
    }

    // Project fields to the boundary
    const auto face_conformal_factor = dg::project_tensor_to_boundary(
        conformal_factor, mesh, boundary_direction);
    const auto face_deriv_conformal_factor = dg::project_tensor_to_boundary(
        deriv_conformal_factor, mesh, boundary_direction);
    const auto face_conformal_metric = dg::project_tensor_to_boundary(
        conformal_metric, mesh, boundary_direction);
    const auto face_inv_conformal_metric = dg::project_tensor_to_boundary(
        inv_conformal_metric, mesh, boundary_direction);
    const auto face_conformal_christoffel_second_kind =
        dg::project_tensor_to_boundary(conformal_christoffel_second_kind, mesh,
                                       boundary_direction);
    const auto face_conformal_christoffel_contracted =
        dg::project_tensor_to_boundary(conformal_christoffel_contracted, mesh,
                                       boundary_direction);
    const auto face_spatial_metric = dg::project_tensor_to_boundary(
        spatial_metric, mesh, boundary_direction);
    const auto face_inv_spatial_metric = dg::project_tensor_to_boundary(
        inv_spatial_metric, mesh, boundary_direction);
    const auto face_extrinsic_curvature = dg::project_tensor_to_boundary(
        extrinsic_curvature, mesh, boundary_direction);
    const auto face_trace_extrinsic_curvature = dg::project_tensor_to_boundary(
        trace_extrinsic_curvature, mesh, boundary_direction);
    // This projection could be avoided by using
    // domain::Tags::DetSurfaceJacobian from the DataBox, which is computed
    // directly on the face (not projected). That would be better on Gauss
    // meshes that have no grid point at the boundary. The DetSurfaceJacobian
    // could then be multiplied by the (conformal) metric determinant to form
    // the area element. Note that the DetSurfaceJacobian is computed using the
    // conformal metric.
    const auto face_inv_jacobian =
        dg::project_tensor_to_boundary(inv_jacobian, mesh, boundary_direction);
    const auto face_euclidean_unit_normal_vector =
        dg::project_tensor_to_boundary(euclidean_unit_normal_vector, mesh,
                                       boundary_direction);

    // Get interface mesh and normal
    const auto& face_mesh = mesh.slice_away(boundary_direction.dimension());

    // Compute Euclidean face normal
    auto euclidean_face_normal = conformal_face_normals.at(boundary_direction);
    const auto face_normal_magnitude = magnitude(euclidean_face_normal);
    for (size_t d = 0; d < 3; ++d) {
      euclidean_face_normal.get(d) /= get(face_normal_magnitude);
    }

    // Compute Euclidean area element
    const auto area_element =
        euclidean_area_element(face_inv_jacobian, boundary_direction);

    // Get surface integrands.
    const auto face_mass_surface_integrand = Xcts::adm_mass_surface_integrand(
        face_deriv_conformal_factor, face_inv_conformal_metric,
        face_conformal_christoffel_second_kind,
        face_conformal_christoffel_contracted);
    const auto face_linear_momentum_surface_integrand =
        dg::project_tensor_to_boundary(linear_momentum_surface_integrand, mesh,
                                       boundary_direction);
    const auto face_center_of_mass_surface_integrand =
        Xcts::center_of_mass_surface_integrand(
            face_conformal_factor, face_euclidean_unit_normal_vector);

    // Contract surface integrands with face normal.
    const auto contracted_mass_integrand = tenex::evaluate(
        -face_mass_surface_integrand(ti::I) * euclidean_face_normal(ti::i));
    const auto contracted_linear_momentum_integrand = tenex::evaluate<ti::I>(
        -face_linear_momentum_surface_integrand(ti::I, ti::J) *
        euclidean_face_normal(ti::j));
    const auto contracted_center_of_mass_integrand = tenex::evaluate<ti::I>(
        -face_center_of_mass_surface_integrand(ti::I, ti::J) *
        euclidean_face_normal(ti::j));

    // Take integrals
    adm_mass->get() += definite_integral(
        get(contracted_mass_integrand) * get(area_element), face_mesh);
    for (int I = 0; I < 3; I++) {
      adm_linear_momentum->get(I) += definite_integral(
          contracted_linear_momentum_integrand.get(I) * get(area_element),
          face_mesh);
      center_of_mass->get(I) += definite_integral(
          contracted_center_of_mass_integrand.get(I) * get(area_element),
          face_mesh);
    }
  }
}

PUP::able::PUP_ID ObserveAdmIntegrals::my_PUP_ID = 0;  // NOLINT

}  // namespace Events
