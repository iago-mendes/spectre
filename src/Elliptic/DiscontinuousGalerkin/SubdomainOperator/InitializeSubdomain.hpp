// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/FaceNormal.hpp"
#include "Domain/Tags/Faces.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "Elliptic/DiscontinuousGalerkin/Initialization.hpp"
#include "Elliptic/DiscontinuousGalerkin/SubdomainOperator/Tags.hpp"
#include "Elliptic/Systems/GetFluxesComputer.hpp"
#include "Elliptic/Utilities/ApplyAt.hpp"
#include "Elliptic/Utilities/GetAnalyticData.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/OverlapHelpers.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
struct GlobalCache;
}  // namespace Parallel
namespace tuples {
template <typename... Tags>
struct TaggedTuple;
}  // namespace tuples
/// \endcond

/// Actions related to the DG subdomain operator
namespace elliptic::dg::subdomain_operator::Actions {

namespace detail {
// Initialize the geometry of a neighbor into which an overlap extends
template <size_t Dim>
struct InitializeOverlapGeometry {
  using return_tags =
      tmpl::list<elliptic::dg::subdomain_operator::Tags::ExtrudingExtent,
                 elliptic::dg::subdomain_operator::Tags::NeighborMortars<
                     domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>, Dim>,
                 elliptic::dg::subdomain_operator::Tags::NeighborMortars<
                     domain::Tags::Mesh<Dim - 1>, Dim>,
                 elliptic::dg::subdomain_operator::Tags::NeighborMortars<
                     ::Tags::MortarSize<Dim - 1>, Dim>>;
  using argument_tags =
      tmpl::list<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                 domain::Tags::NeighborMesh<Dim>>;
  void operator()(
      gsl::not_null<size_t*> extruding_extent,
      gsl::not_null<::dg::MortarMap<Dim, Scalar<DataVector>>*>
          neighbor_face_normal_magnitudes,
      gsl::not_null<::dg::MortarMap<Dim, Mesh<Dim - 1>>*>
          neighbor_mortar_meshes,
      gsl::not_null<::dg::MortarMap<Dim, ::dg::MortarSize<Dim - 1>>*>
          neighbor_mortar_sizes,
      const Element<Dim>& element, const Mesh<Dim>& mesh,
      const DirectionalIdMap<Dim, Mesh<Dim>>& neighbor_meshes,
      const ElementId<Dim>& element_id, const Direction<Dim>& overlap_direction,
      const size_t max_overlap) const;
};
}  // namespace detail

/*!
 * \brief Initialize the geometry for the DG subdomain operator
 *
 * Initializes tags that define the geometry of overlap regions with neighboring
 * elements. The data needs to be updated if the geometry of neighboring
 * elements changes.
 *
 * Note that the geometry depends on the system and on the choice of background
 * through the normalization of face normals, which involves a metric.
 *
 * The `FromInitialDomain` template parameter controls how the
 * next-to-nearest-neighbor information is acquired:
 * - If `FromInitialDomain = true`, uses the initial parameters from the input
 *   file. This can be used at the beginning of a simulation or if the domain
 *   never changes (no AMR).
 * - If `FromInitialDomain = false`, uses the mesh, element, and neighbor meshes
 *   of each neighbor. This typically requires communication to exchange this
 *   data between neighbors each time the domain changes, e.g. with AMR.
 *
 * DataBox:
 * - Uses:
 *   - `BackgroundTag`
 *   - `domain::Tags::Element<Dim>`
 *   - If `FromInitialDomain = true`:
 *     - `domain::Tags::InitialExtents<Dim>`
 *     - `domain::Tags::InitialRefinementLevels<Dim>`
 *     - `elliptic::dg::Tags::Quadrature`
 *   - If `FromInitialDomain = false`:
 *     - `Overlaps<domain::Tags::Mesh<Dim>>`
 *     - `Overlaps<domain::Tags::Element<Dim>>`
 *     - `Overlaps<domain::Tags::NeighborMesh<Dim>>`
 *   - `domain::Tags::Domain<Dim>`
 *   - `domain::Tags::FunctionsOfTime`
 *   - `LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>`
 * - Adds: Many tags prefixed with `LinearSolver::Schwarz::Tags::Overlaps`. See
 *   `elliptic::dg::Actions::InitializeDomain` and
 *   `elliptic::dg::Actions::initialize_operator` for a complete list.
 */
template <typename System, typename BackgroundTag, typename OptionsGroup,
          bool FromInitialDomain = true>
struct InitializeSubdomain {
 private:
  static constexpr size_t Dim = System::volume_dim;
  static constexpr bool is_curved =
      not std::is_same_v<typename System::inv_metric_tag, void>;
  static constexpr bool has_background_fields =
      not std::is_same_v<typename System::background_fields, tmpl::list<>>;

  using InitializeGeometry =
      tmpl::conditional_t<FromInitialDomain,
                          elliptic::dg::InitializeGeometry<Dim>,
                          elliptic::dg::ProjectGeometry<Dim>>;
  using InitializeOverlapGeometry = detail::InitializeOverlapGeometry<Dim>;
  using InitializeFacesAndMortars = elliptic::dg::InitializeFacesAndMortars<
      Dim, typename System::inv_metric_tag, BackgroundTag>;

  template <typename Tag>
  using overlaps_tag =
      LinearSolver::Schwarz::Tags::Overlaps<Tag, Dim, OptionsGroup>;
  // Only slice those background fields to internal boundaries that are
  // necessary for the DG operator, i.e. the background fields in the
  // System::fluxes_computer::argument_tags
  using fluxes_non_background_args =
      tmpl::list_difference<elliptic::get_fluxes_argument_tags<System, true>,
                            typename System::background_fields>;
  using background_fields_internal =
      tmpl::list_difference<elliptic::get_fluxes_argument_tags<System, true>,
                            fluxes_non_background_args>;
  // Slice all background fields to external boundaries for use in boundary
  // conditions
  using background_fields_external = typename System::background_fields;

 public:
  using simple_tags_from_options =
      tmpl::list<domain::Tags::InitialExtents<Dim>,
                 domain::Tags::InitialRefinementLevels<Dim>>;
  using const_global_cache_tags =
      tmpl::list<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>;
  using simple_tags = db::wrap_tags_in<
      overlaps_tag,
      tmpl::append<
          typename InitializeGeometry::return_tags,
          typename InitializeFacesAndMortars::return_tags,
          typename InitializeOverlapGeometry::return_tags,
          tmpl::conditional_t<
              has_background_fields,
              tmpl::list<::Tags::Variables<typename System::background_fields>>,
              tmpl::list<>>,
          domain::make_faces_tags<Dim, typename System::background_fields>>>;
  using compute_tags = tmpl::list<>;

  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& /*element_id*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& element = db::get<domain::Tags::Element<Dim>>(box);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const LinearSolver::Schwarz::OverlapId<Dim> overlap_id{direction,
                                                               neighbor_id};
        // Initialize background-agnostic geometry on overlaps
        elliptic::util::mutate_apply_at<
            db::wrap_tags_in<overlaps_tag,
                             typename InitializeGeometry::return_tags>,
            tmpl::append<
                db::wrap_tags_in<overlaps_tag,
                                 tmpl::list_difference<
                                     typename InitializeGeometry::argument_tags,
                                     typename InitializeGeometry::volume_tags>>,
                typename InitializeGeometry::volume_tags>,
            typename InitializeGeometry::volume_tags>(
            InitializeGeometry{}, make_not_null(&box), overlap_id, neighbor_id);
        // Initialize subdomain-specific tags on overlaps
        elliptic::util::mutate_apply_at<
            db::wrap_tags_in<overlaps_tag,
                             typename InitializeOverlapGeometry::return_tags>,
            db::wrap_tags_in<overlaps_tag,
                             typename InitializeOverlapGeometry::argument_tags>,
            tmpl::list<>>(
            InitializeOverlapGeometry{}, make_not_null(&box), overlap_id,
            neighbor_id, direction_from_neighbor,
            db::get<LinearSolver::Schwarz::Tags::MaxOverlap<OptionsGroup>>(
                box));
        // Initialize faces and mortars on overlaps
        elliptic::util::mutate_apply_at<
            db::wrap_tags_in<overlaps_tag,
                             typename InitializeFacesAndMortars::return_tags>,
            tmpl::append<
                db::wrap_tags_in<
                    overlaps_tag,
                    tmpl::list_difference<
                        typename InitializeFacesAndMortars::argument_tags,
                        typename InitializeFacesAndMortars::volume_tags>>,
                typename InitializeFacesAndMortars::volume_tags>,
            typename InitializeFacesAndMortars::volume_tags>(
            InitializeFacesAndMortars{}, make_not_null(&box), overlap_id);
        if constexpr (has_background_fields) {
          // Background fields
          initialize_background_fields(make_not_null(&box), overlap_id);
        }
        // Faces on the other side of the overlapped element's mortars
        initialize_remote_faces(make_not_null(&box), overlap_id);
      }  // neighbors in direction
    }  // directions
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }

 private:
  template <typename DbTagsList>
  static void initialize_background_fields(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) {
    const auto& background = db::get<BackgroundTag>(*box);
    DirectionMap<Dim, Variables<typename System::background_fields>>
        face_background_fields{};
    elliptic::util::mutate_apply_at<
        tmpl::list<overlaps_tag<
            ::Tags::Variables<typename System::background_fields>>>,
        db::wrap_tags_in<
            overlaps_tag,
            tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>,
                       domain::Tags::Mesh<Dim>,
                       domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>>,
        tmpl::list<>>(
        [&background, &face_background_fields, &box](
            const gsl::not_null<Variables<typename System::background_fields>*>
                background_fields,
            const tnsr::I<DataVector, Dim>& inertial_coords,
            const Mesh<Dim>& mesh,
            const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                                  Frame::Inertial>& inv_jacobian) {
          *background_fields = elliptic::util::get_analytic_data<
              typename System::background_fields>(
              background, *box, inertial_coords, mesh, inv_jacobian);
          for (const auto& direction : Direction<Dim>::all_directions()) {
            // Slice the background fields to the face instead of evaluating
            // them on the face coords to avoid re-computing them, and because
            // this is also what the DG operator currently does. The result is
            // the same on Gauss-Lobatto grids, but may need adjusting when
            // adding support for Gauss grids.
            face_background_fields[direction].initialize(
                mesh.slice_away(direction.dimension()).number_of_grid_points());
            ::dg::project_contiguous_data_to_boundary(
                make_not_null(&face_background_fields[direction]),
                *background_fields, mesh, direction);
          }
        },
        box, overlap_id);
    // Move face background fields into DataBox
    const auto mutate_assign_face_background_field =
        [&box, &overlap_id, &face_background_fields](
            auto tag_v, const Direction<Dim>& direction) {
          using tag = tmpl::type_from<std::decay_t<decltype(tag_v)>>;
          db::mutate<overlaps_tag<domain::Tags::Faces<Dim, tag>>>(
              [&face_background_fields, &overlap_id,
               &direction](const auto stored_value) {
                (*stored_value)[overlap_id][direction] =
                    get<tag>(face_background_fields.at(direction));
              },
              box);
        };
    const auto& element =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(*box).at(overlap_id);
    for (const auto& direction : element.internal_boundaries()) {
      tmpl::for_each<background_fields_internal>(
          [&mutate_assign_face_background_field, &direction](auto tag_v) {
            mutate_assign_face_background_field(tag_v, direction);
          });
    }
    for (const auto& direction : element.external_boundaries()) {
      tmpl::for_each<background_fields_external>(
          [&mutate_assign_face_background_field, &direction](auto tag_v) {
            mutate_assign_face_background_field(tag_v, direction);
          });
    }
  }

  // Faces on the other side of the overlapped element's mortars
  template <typename DbTagsList>
  static void initialize_remote_faces(
      const gsl::not_null<db::DataBox<DbTagsList>*> box,
      const LinearSolver::Schwarz::OverlapId<Dim>& overlap_id) {
    const auto& element =
        db::get<overlaps_tag<domain::Tags::Element<Dim>>>(*box).at(overlap_id);
    const auto& domain = db::get<domain::Tags::Domain<Dim>>(*box);
    const auto& functions_of_time =
        db::get<domain::Tags::FunctionsOfTime>(*box);
    const auto& neighbor_meshes =
        db::get<overlaps_tag<domain::Tags::NeighborMesh<Dim>>>(*box).at(
            overlap_id);
    for (const auto& [direction, neighbors] : element.neighbors()) {
      const auto& orientation = neighbors.orientation();
      const auto direction_from_neighbor = orientation(direction.opposite());
      for (const auto& neighbor_id : neighbors) {
        const ::dg::MortarId<Dim> mortar_id{direction, neighbor_id};
        const auto neighbor_face_mesh =
            neighbor_meshes.at(mortar_id).slice_away(
                direction_from_neighbor.dimension());
        const auto neighbor_face_logical_coords = interface_logical_coordinates(
            neighbor_face_mesh, direction_from_neighbor);
        const auto& neighbor_block = domain.blocks()[neighbor_id.block_id()];
        const ElementMap<Dim, Frame::Inertial> neighbor_element_map{
            neighbor_id, neighbor_block};
        const auto neighbor_face_normal = unnormalized_face_normal(
            neighbor_face_mesh,
            neighbor_element_map.inv_jacobian(neighbor_face_logical_coords, 0.,
                                              functions_of_time),
            direction_from_neighbor);
        using neighbor_face_normal_magnitudes_tag = overlaps_tag<
            elliptic::dg::subdomain_operator::Tags::NeighborMortars<
                domain::Tags::UnnormalizedFaceNormalMagnitude<Dim>, Dim>>;
        if constexpr (is_curved) {
          const auto& background = db::get<BackgroundTag>(*box);
          const auto neighbor_face_inertial_coords = neighbor_element_map(
              neighbor_face_logical_coords, 0., functions_of_time);
          const auto inv_metric_on_face = get<typename System::inv_metric_tag>(
              elliptic::util::get_analytic_data<
                  tmpl::list<typename System::inv_metric_tag>>(
                  background, *box, neighbor_face_inertial_coords));
          elliptic::util::mutate_apply_at<
              tmpl::list<neighbor_face_normal_magnitudes_tag>, tmpl::list<>,
              tmpl::list<>>(
              [&neighbor_face_normal,
               &inv_metric_on_face](const auto neighbor_face_normal_magnitude) {
                magnitude(neighbor_face_normal_magnitude, neighbor_face_normal,
                          inv_metric_on_face);
              },
              box, std::make_tuple(overlap_id, mortar_id));
        } else {
          elliptic::util::mutate_apply_at<
              tmpl::list<neighbor_face_normal_magnitudes_tag>, tmpl::list<>,
              tmpl::list<>>(
              [&neighbor_face_normal](
                  const auto neighbor_face_normal_magnitude) {
                magnitude(neighbor_face_normal_magnitude, neighbor_face_normal);
              },
              box, std::make_tuple(overlap_id, mortar_id));
        }
      }  // neighbors
    }  // internal directions
  }
};

}  // namespace elliptic::dg::subdomain_operator::Actions
