// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalyticInNozzle.hpp"

#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DgSubcell/GhostZoneLogicalCoordinates.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/FiniteDifference/Reconstructor.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryConditions {
namespace {
// The points (of a face or a ghost zone) inside the nozzle, |x^0| <= radius.
std::vector<size_t> points_in_nozzle(
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const double nozzle_radius) {
  std::vector<size_t> points{};
  const DataVector& radial_coordinate = get<0>(coords);
  for (size_t point = 0; point < radial_coordinate.size(); ++point) {
    if (std::abs(radial_coordinate[point]) <= nozzle_radius) {
      points.push_back(point);
    }
  }
  return points;
}

// Copy the given points of every component of `source` into `destination`.
template <typename TensorType>
void copy_points(const gsl::not_null<TensorType*> destination,
                 const TensorType& source, const std::vector<size_t>& points) {
  for (size_t component = 0; component < destination->size(); ++component) {
    for (const size_t point : points) {
      (*destination)[component][point] = source[component][point];
    }
  }
}

// Fill copies of `outputs` with `compute` (called with pointers to the
// copies), then copy the nozzle points of the copies into `outputs`.
// Starting from copies of the outputs leaves every output that `compute` does
// not set at its previous value.
template <typename Compute, typename... Tensors>
std::optional<std::string> overwrite_points(
    const std::vector<size_t>& points, const Compute& compute,
    const gsl::not_null<Tensors*>... outputs) {
  std::tuple<Tensors...> values{*outputs...};
  std::optional<std::string> error = std::apply(
      [&compute](auto&... value) { return compute(make_not_null(&value)...); },
      values);
  std::apply(
      [&points, &outputs...](const auto&... value) {
        (copy_points(outputs, value, points), ...);
      },
      values);
  return error;
}
}  // namespace

DirichletAnalyticInNozzle::DirichletAnalyticInNozzle(
    CkMigrateMessage* const msg)
    : BoundaryCondition(msg) {}

DirichletAnalyticInNozzle::DirichletAnalyticInNozzle(
    std::unique_ptr<evolution::initial_data::InitialData> analytic_prescription,
    const double nozzle_radius)
    : dirichlet_analytic_(std::move(analytic_prescription)),
      nozzle_radius_(nozzle_radius) {}

std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>
DirichletAnalyticInNozzle::get_clone() const {
  return std::make_unique<DirichletAnalyticInNozzle>(*this);
}

void DirichletAnalyticInNozzle::pup(PUP::er& p) {
  BoundaryCondition::pup(p);
  dirichlet_analytic_.pup(p);
  p | nozzle_radius_;
}

// NOLINTNEXTLINE
PUP::able::PUP_ID DirichletAnalyticInNozzle::my_PUP_ID = 0;

std::optional<std::string> DirichletAnalyticInNozzle::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<tnsr::ii<DataVector, 3, Frame::Inertial>*>
        spatial_metric,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_temperature,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

    const double time) const {
  // The outflow state on the whole face.
  std::optional<std::string> error = HydroFreeOutflow::dg_ghost(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
      tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      lapse, shift, spatial_velocity_one_form, spatial_metric,
      rest_mass_density, electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric,
      face_mesh_velocity, outward_directed_normal_covector,
      outward_directed_normal_vector, interior_rest_mass_density,
      interior_electron_fraction, interior_specific_internal_energy,
      interior_spatial_velocity, interior_magnetic_field,
      interior_lorentz_factor, interior_pressure, interior_temperature,
      interior_shift, interior_lapse, interior_inv_spatial_metric);
  // The analytic state on the face's nozzle points.
  const std::vector<size_t> points = points_in_nozzle(coords, nozzle_radius_);
  if (points.empty()) {
    return error;
  }
  std::optional<std::string> dirichlet_error = overwrite_points(
      points,
      [&](const auto... outputs) {
        return dirichlet_analytic_.dg_ghost(
            outputs..., face_mesh_velocity, outward_directed_normal_covector,
            outward_directed_normal_vector, coords, time);
      },
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
      tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      lapse, shift, spatial_velocity_one_form, spatial_metric,
      rest_mass_density, electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric);
  return error.has_value() ? error : dirichlet_error;
}

std::optional<std::string> DirichletAnalyticInNozzle::dg_ghost(
    const gsl::not_null<Scalar<DataVector>*> tilde_d,
    const gsl::not_null<Scalar<DataVector>*> tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_b,
    const gsl::not_null<Scalar<DataVector>*> tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_ye_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const gsl::not_null<Scalar<DataVector>*> lapse,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> shift,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        spatial_velocity_one_form,
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
        inv_spatial_metric,

    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
        face_mesh_velocity,
    const tnsr::i<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        outward_directed_normal_vector,

    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,
    const Scalar<DataVector>& interior_lorentz_factor,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_temperature,

    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& interior_inv_spatial_metric,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,

    const double time) const {
  std::optional<std::string> error = HydroFreeOutflow::dg_ghost(
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
      tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      lapse, shift, spatial_velocity_one_form, rest_mass_density,
      electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric,
      face_mesh_velocity, outward_directed_normal_covector,
      outward_directed_normal_vector, interior_rest_mass_density,
      interior_electron_fraction, interior_specific_internal_energy,
      interior_spatial_velocity, interior_magnetic_field,
      interior_lorentz_factor, interior_pressure, interior_temperature,
      interior_shift, interior_lapse, interior_inv_spatial_metric);
  const std::vector<size_t> points = points_in_nozzle(coords, nozzle_radius_);
  if (points.empty()) {
    return error;
  }
  std::optional<std::string> dirichlet_error = overwrite_points(
      points,
      [&](const auto... outputs) {
        return dirichlet_analytic_.dg_ghost(
            outputs..., face_mesh_velocity, outward_directed_normal_covector,
            outward_directed_normal_vector, coords, time);
      },
      tilde_d, tilde_ye, tilde_tau, tilde_s, tilde_b, tilde_phi, tilde_d_flux,
      tilde_ye_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,
      lapse, shift, spatial_velocity_one_form, rest_mass_density,
      electron_fraction, temperature, spatial_velocity,
      specific_internal_energy, pressure, lorentz_factor, inv_spatial_metric);
  return error.has_value() ? error : dirichlet_error;
}

void DirichletAnalyticInNozzle::fd_ghost(
    const gsl::not_null<Scalar<DataVector>*> rest_mass_density,
    const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const gsl::not_null<Scalar<DataVector>*> temperature,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        lorentz_factor_times_spatial_velocity,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        magnetic_field,
    const gsl::not_null<Scalar<DataVector>*> divergence_cleaning_field,

    const gsl::not_null<std::optional<Variables<db::wrap_tags_in<
        Flux, typename grmhd::ValenciaDivClean::System::flux_variables>>>*>
        cell_centered_ghost_fluxes,

    const Direction<3>& direction,

    // fd_interior_temporary_tags
    const Mesh<3>& subcell_mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_shift,
    const Scalar<DataVector>& interior_lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& interior_spatial_metric,

    // fd_interior_primitive_variables_tags
    const Scalar<DataVector>& interior_rest_mass_density,
    const Scalar<DataVector>& interior_electron_fraction,
    const Scalar<DataVector>& interior_temperature,
    const Scalar<DataVector>& interior_pressure,
    const Scalar<DataVector>& interior_specific_internal_energy,
    const Scalar<DataVector>& interior_lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& interior_magnetic_field,

    // fd_gridless_tags
    const double time,
    const std::unordered_map<
        std::string,
        std::unique_ptr<::domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const ElementMap<3, Frame::Grid>& logical_to_grid_map,
    const domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 3>&
        grid_to_inertial_map,
    const fd::Reconstructor& reconstructor) const {
  // The outflow state on the whole ghost zone.
  HydroFreeOutflow::fd_ghost(
      rest_mass_density, electron_fraction, temperature,
      lorentz_factor_times_spatial_velocity, magnetic_field,
      divergence_cleaning_field, cell_centered_ghost_fluxes, direction,
      subcell_mesh, interior_shift, interior_lapse, interior_spatial_metric,
      interior_rest_mass_density, interior_electron_fraction,
      interior_temperature, interior_pressure,
      interior_specific_internal_energy, interior_lorentz_factor,
      interior_spatial_velocity, interior_magnetic_field, reconstructor);

  // The ghost cells inside the nozzle, from the ghost-cell coordinates
  // DirichletAnalytic::fd_ghost evaluates its prescription at.
  const auto ghost_inertial_coords = grid_to_inertial_map(
      logical_to_grid_map(
          evolution::dg::subcell::fd::ghost_zone_logical_coordinates(
              subcell_mesh, reconstructor.ghost_zone_size(), direction)),
      time, functions_of_time);
  const std::vector<size_t> points =
      points_in_nozzle(ghost_inertial_coords, nozzle_radius_);
  if (points.empty()) {
    return;
  }

  // The analytic state on the whole ghost zone, copied in at those cells.
  auto dirichlet_fluxes = *cell_centered_ghost_fluxes;
  overwrite_points(
      points,
      [&](const auto... outputs) -> std::optional<std::string> {
        dirichlet_analytic_.fd_ghost(
            outputs..., make_not_null(&dirichlet_fluxes), direction,
            subcell_mesh, time, functions_of_time, logical_to_grid_map,
            grid_to_inertial_map, reconstructor);
        return std::nullopt;
      },
      rest_mass_density, electron_fraction, temperature,
      lorentz_factor_times_spatial_velocity, magnetic_field,
      divergence_cleaning_field);
  if (cell_centered_ghost_fluxes->has_value()) {
    auto& fluxes = cell_centered_ghost_fluxes->value();
    const auto& analytic_fluxes = dirichlet_fluxes.value();
    const size_t number_of_points = fluxes.number_of_grid_points();
    constexpr size_t number_of_components =
        std::decay_t<decltype(fluxes)>::number_of_independent_components;
    for (size_t component = 0; component < number_of_components; ++component) {
      for (const size_t point : points) {
        fluxes.data()[component * number_of_points + point] =
            analytic_fluxes.data()[component * number_of_points + point];
      }
    }
  }
}
}  // namespace grmhd::ValenciaDivClean::BoundaryConditions
