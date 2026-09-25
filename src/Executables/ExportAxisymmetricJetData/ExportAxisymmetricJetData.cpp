// Distributed under the MIT License.
// See LICENSE.txt for details.

// Evaluate the Del Zanna Fig. 6 jet initial data on a real CartoonCylinder
// domain and write CSV. Nothing here reimplements the selector: the values
// come from grmhd::AnalyticData::AxisymmetricJet and, for the side-by-side
// comparison, grmhd::AnalyticData::SlabJet, evaluated on inertial coordinates
// produced by the domain creator that the evolution will use.

#include <array>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/CartoonCylinder.hpp"
#include "Domain/Creators/TimeDependence/None.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Structure/CreateInitialMesh.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/CartoonGhost.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/DirichletAnalytic.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/HydroFreeOutflow.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AxisymmetricJet.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SlabJet.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {

// The Fig. 6 / Paper I parameters. Jet (rho, v_z, v_r, p) = (0.1, 0.99, 0,
// 0.01), ambient (10, 0, 0, 0.01), B_z = 0.1 everywhere, Gamma = 5/3.
// papers/2002-del-zanna-bucciantini-hydro/tex/delzannal1.tex:860-866 and
// papers/2002-del-zanna-et-al/tex/delzannal2.tex:1344-1347.
constexpr double adiabatic_index = 5. / 3.;
constexpr double ambient_density = 10.;
constexpr double ambient_pressure = 0.01;
constexpr double jet_density = 0.1;
constexpr double jet_pressure = 0.01;
constexpr double jet_speed = 0.99;
constexpr double inlet_radius = 1.;
constexpr double nozzle_length = 1.;
constexpr double axial_field = 0.1;

grmhd::AnalyticData::AxisymmetricJet make_axisymmetric_jet(
    const double smoothing_width) {
  return grmhd::AnalyticData::AxisymmetricJet{
      adiabatic_index,
      ambient_density,
      ambient_pressure,
      0.,
      jet_density,
      jet_pressure,
      0.,
      std::array<double, 3>{{0., jet_speed, 0.}},
      inlet_radius,
      nozzle_length,
      smoothing_width,
      std::array<double, 3>{{0., axial_field, 0.}}};
}

grmhd::AnalyticData::SlabJet make_slab_jet() {
  return grmhd::AnalyticData::SlabJet{
      adiabatic_index,
      ambient_density,
      ambient_pressure,
      0.,
      jet_density,
      jet_pressure,
      0.,
      std::array<double, 3>{{0., jet_speed, 0.}},
      inlet_radius,
      std::array<double, 3>{{0., axial_field, 0.}}};
}

// The production domain. Boundary conditions are the real ones the evolution
// uses, so that the resolved-boundary report is faithful.
domain::creators::CartoonCylinder make_cylinder(
    const std::array<size_t, 2>& refinement,
    const std::array<size_t, 2>& points) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryConditions;
  auto outflow = []() {
    return std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>{
        std::make_unique<bc::HydroFreeOutflow>()};
  };
  auto inflow = []() {
    return std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>{
        std::make_unique<bc::DirichletAnalytic>(
            std::make_unique<grmhd::AnalyticData::AxisymmetricJet>(
                make_axisymmetric_jet(0.)))};
  };
  return domain::creators::CartoonCylinder{
      std::array<double, 2>{{0., 0.}},
      std::array<double, 2>{{8., 20.}},
      refinement,
      points,
      std::array<domain::CoordinateMaps::Distribution, 2>{
          {domain::CoordinateMaps::Distribution::Linear,
           domain::CoordinateMaps::Distribution::Linear}},
      std::make_unique<domain::creators::time_dependence::None<3>>(),
      std::array<std::array<std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>,
                            2>,
                 2>{{{{outflow(), outflow()}}, {{inflow(), outflow()}}}},
      std::make_unique<bc::CartoonGhost>()};
}

// Inertial coordinates of every grid point of every element, DG or subcell.
tnsr::I<DataVector, 3, Frame::Inertial> all_coordinates(
    const domain::creators::CartoonCylinder& cylinder, const bool subcell) {
  const auto domain_3d = cylinder.create_domain();
  const auto& block = domain_3d.blocks()[0];
  const auto element_ids =
      initial_element_ids<3>(0, cylinder.initial_refinement_levels()[0]);
  std::vector<std::array<double, 3>> points{};
  for (const auto& element_id : element_ids) {
    auto mesh = domain::create_initial_mesh(
        cylinder.initial_extents(), block, element_id,
        Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
    if (subcell) {
      // The finite-difference grid the evolution actually runs on: cell
      // centred, 2N-1 cells per element per dimension.
      mesh = Mesh<3>{
          {{2 * mesh.extents(0) - 1, 2 * mesh.extents(1) - 1, 1}},
          {{Spectral::Basis::FiniteDifference,
            Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon}},
          {{Spectral::Quadrature::CellCentered,
            Spectral::Quadrature::CellCentered,
            Spectral::Quadrature::AxialSymmetry}}};
    }
    const ElementMap<3, Frame::Inertial> element_map{
        element_id, block.stationary_map().get_clone()};
    const auto inertial = element_map(logical_coordinates(mesh));
    for (size_t i = 0; i < get<0>(inertial).size(); ++i) {
      points.push_back(
          {{get<0>(inertial)[i], get<1>(inertial)[i], get<2>(inertial)[i]}});
    }
  }
  tnsr::I<DataVector, 3, Frame::Inertial> result{points.size()};
  for (size_t i = 0; i < points.size(); ++i) {
    for (size_t d = 0; d < 3; ++d) {
      result.get(d)[i] = gsl::at(points[i], d);
    }
  }
  return result;
}

using dump_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                             hydro::Tags::Pressure<DataVector>,
                             hydro::Tags::SpatialVelocity<DataVector, 3>,
                             hydro::Tags::MagneticField<DataVector, 3>,
                             hydro::Tags::LorentzFactor<DataVector>,
                             hydro::Tags::SpecificEnthalpy<DataVector>>;

template <typename AnalyticData>
void write_csv(const std::string& filename, const AnalyticData& data,
               const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  const auto vars = data.variables(coords, dump_tags{});
  const auto& rho = get<hydro::Tags::RestMassDensity<DataVector>>(vars);
  const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(vars);
  const auto& velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars);
  const auto& b_field = get<hydro::Tags::MagneticField<DataVector, 3>>(vars);
  const auto& lorentz = get<hydro::Tags::LorentzFactor<DataVector>>(vars);
  const auto& enthalpy = get<hydro::Tags::SpecificEnthalpy<DataVector>>(vars);
  std::ofstream out{filename};
  out << std::setprecision(17);
  out << "r,z,rho,p,v_r,v_z,B_r,B_z,W,h\n";
  for (size_t i = 0; i < get<0>(coords).size(); ++i) {
    out << get<0>(coords)[i] << ',' << get<1>(coords)[i] << ',' << get(rho)[i]
        << ',' << get(pressure)[i] << ',' << get<0>(velocity)[i] << ','
        << get<1>(velocity)[i] << ',' << get<0>(b_field)[i] << ','
        << get<1>(b_field)[i] << ',' << get(lorentz)[i] << ','
        << get(enthalpy)[i] << '\n';
  }
  std::cout << "wrote " << filename << " (" << get<0>(coords).size()
            << " points)\n";
}
}  // namespace

// Charm looks for this function, but this is a plain command-line tool built
// without a main module, so it is empty -- the same stub that
// tests/Unit/TestMain.cpp uses for the same reason.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmissing-declarations"
extern "C" void CkRegisterMainModule(void) {}
#pragma GCC diagnostic pop

int main(int argc, char** argv) {
  const std::string out_dir = argc > 1 ? argv[1] : ".";
  const std::array<size_t, 2> refinement{{5, 6}};
  const std::array<size_t, 2> points{{3, 3}};

  const auto cylinder = make_cylinder(refinement, points);
  const auto jet = make_axisymmetric_jet(0.);
  const auto slab = make_slab_jet();

  // ---- F3.1(a,b): the (z,r) plane on the real FD grid -------------------
  const auto subcell_coords = all_coordinates(cylinder, true);
  write_csv(out_dir + "/axisymmetric_jet_subcell.csv", jet, subcell_coords);
  write_csv(out_dir + "/slab_jet_subcell.csv", slab, subcell_coords);

  // The DG collocation grid too, for the grid overlay.
  const auto dg_coords = all_coordinates(cylinder, false);
  write_csv(out_dir + "/axisymmetric_jet_dg.csv", jet, dg_coords);

  // ---- F3.1(c): radial cut at z = 0.5 -----------------------------------
  {
    const size_t num_cut = 801;
    tnsr::I<DataVector, 3, Frame::Inertial> cut{num_cut, 0.};
    for (size_t i = 0; i < num_cut; ++i) {
      get<0>(cut)[i] =
          8. * static_cast<double>(i) / static_cast<double>(num_cut - 1);
      get<1>(cut)[i] = 0.5;
    }
    write_csv(out_dir + "/radial_cut_z0.5_sharp.csv", jet, cut);
    write_csv(out_dir + "/radial_cut_z0.5_smoothed.csv",
              make_axisymmetric_jet(0.05), cut);
  }

  // ---- F3.2(c): the lower-z inflow face ---------------------------------
  // These are the coordinates DirichletAnalytic evaluates the prescription on.
  // One boundary-condition entry must give beam values for r <= 1 and ambient
  // for r > 1.
  {
    const size_t num_face = 401;
    tnsr::I<DataVector, 3, Frame::Inertial> face{num_face, 0.};
    for (size_t i = 0; i < num_face; ++i) {
      get<0>(face)[i] =
          8. * static_cast<double>(i) / static_cast<double>(num_face - 1);
      get<1>(face)[i] = 0.;
    }
    write_csv(out_dir + "/inflow_face_z0.csv", jet, face);
    // And a ghost-zone slab below the face, which is where the FD ghost
    // points actually live.
    tnsr::I<DataVector, 3, Frame::Inertial> ghost{num_face, 0.};
    for (size_t i = 0; i < num_face; ++i) {
      get<0>(ghost)[i] =
          8. * static_cast<double>(i) / static_cast<double>(num_face - 1);
      get<1>(ghost)[i] = -0.05;
    }
    write_csv(out_dir + "/inflow_ghost_zm0.05.csv", jet, ghost);
  }

  // ---- F3.2(b): resolved boundary conditions from the BUILT domain ------
  {
    std::ofstream out{out_dir + "/resolved_boundary_conditions.txt"};
    const auto bcs = cylinder.external_boundary_conditions();
    out << "Resolved external boundary conditions, from "
           "CartoonCylinder::external_boundary_conditions()\n";
    out << "x^0 = cylindrical radius r in [0,8]; x^1 = symmetry axis z in "
           "[0,20]\n\n";
    for (const auto& [direction, bc] : bcs[0]) {
      std::string face;
      if (direction == Direction<3>::lower_xi()) {
        face = "lower r  (r = 0,  the axis)";
      } else if (direction == Direction<3>::upper_xi()) {
        face = "upper r  (r = 8)";
      } else if (direction == Direction<3>::lower_eta()) {
        face = "lower z  (z = 0,  the nozzle)";
      } else {
        face = "upper z  (z = 20)";
      }
      std::string name = "UNKNOWN";
      if (dynamic_cast<
              const grmhd::ValenciaDivClean::BoundaryConditions::CartoonGhost*>(
              bc.get()) != nullptr) {
        name = "CartoonGhost";
      } else if (dynamic_cast<const grmhd::ValenciaDivClean::
                                  BoundaryConditions::DirichletAnalytic*>(
                     bc.get()) != nullptr) {
        name = "DirichletAnalytic(AxisymmetricJet)";
      } else if (dynamic_cast<const grmhd::ValenciaDivClean::
                                  BoundaryConditions::HydroFreeOutflow*>(
                     bc.get()) != nullptr) {
        name = "HydroFreeOutflow";
      }
      out << face << "  ->  " << name << '\n';
    }
    out << "\nMesh, element (0,0):\n";
    const auto domain_3d = cylinder.create_domain();
    const auto mesh = domain::create_initial_mesh(
        cylinder.initial_extents(), domain_3d.blocks()[0], ElementId<3>{0},
        Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
    out << mesh << '\n';
    out << "\nInitialRefinement = [" << refinement[0] << ", " << refinement[1]
        << "], InitialGridPoints = [" << points[0] << ", " << points[1]
        << "]\n";
    out << "FD cells: r = " << (1u << refinement[0]) * (2 * points[0] - 1)
        << " over [0,8],  z = " << (1u << refinement[1]) * (2 * points[1] - 1)
        << " over [0,20]\n";
  }
  std::cout << "done\n";
  return 0;
}
