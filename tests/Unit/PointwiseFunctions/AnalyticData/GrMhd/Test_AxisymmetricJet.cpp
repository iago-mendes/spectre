// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
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
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AxisymmetricJet.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/SlabJet.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {

// The Del Zanna et al. (2003) Fig. 6 parameters, read from
// papers/2002-del-zanna-et-al/tex/delzannal2.tex:1341-1348.
constexpr double adiabatic_index = 5. / 3.;
constexpr double ambient_density = 10.;
constexpr double ambient_pressure = 0.01;
constexpr double ambient_electron_fraction = 0.;
constexpr double jet_density = 0.1;
constexpr double jet_pressure = 0.01;
constexpr double jet_electron_fraction = 0.;
constexpr double jet_speed = 0.99;
constexpr double inlet_radius = 1.;
constexpr double nozzle_length = 1.;
constexpr double axial_field = 0.1;

std::array<double, 3> jet_velocity() { return {{0., jet_speed, 0.}}; }
std::array<double, 3> magnetic_field() { return {{0., axial_field, 0.}}; }

grmhd::AnalyticData::AxisymmetricJet make_jet(const double smoothing_width) {
  return grmhd::AnalyticData::AxisymmetricJet{
      adiabatic_index,       ambient_density,
      ambient_pressure,      ambient_electron_fraction,
      jet_density,           jet_pressure,
      jet_electron_fraction, jet_velocity(),
      inlet_radius,          nozzle_length,
      smoothing_width,       magnetic_field()};
}

// Real inertial coordinates from a CartoonCylinder domain covering the Fig. 6
// box, r in [0,8] and z in [0,20], at the refinement the production run uses.
// Building the domain rather than writing coordinates by hand is the point of
// this helper: the failure mode being guarded against is a selector that does
// not match the domain, and hand-written coordinates hide it.
//
// The refinement matters. The nozzle is 1/8 of the domain in r and 1/20 of it
// in z, so an unrefined block can contain no collocation point inside the
// nozzle at all. At InitialRefinement [3, 4] the element touching the origin
// spans exactly r in [0,1] and z in [0,1.25].
tnsr::I<DataVector, 3, Frame::Inertial> cartoon_cylinder_coordinates() {
  using TestBc =
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<3>;
  auto make_bc = [](const Direction<3>& direction) {
    return std::unique_ptr<domain::BoundaryConditions::BoundaryCondition>{
        std::make_unique<TestBc>(direction, 0)};
  };
  const domain::creators::CartoonCylinder cylinder{
      std::array<double, 2>{{0., 0.}},
      std::array<double, 2>{{8., 20.}},
      std::array<size_t, 2>{{3, 4}},
      std::array<size_t, 2>{{5, 5}},
      std::array<domain::CoordinateMaps::Distribution, 2>{
          {domain::CoordinateMaps::Distribution::Linear,
           domain::CoordinateMaps::Distribution::Linear}},
      std::make_unique<domain::creators::time_dependence::None<3>>(),
      std::array<std::array<std::unique_ptr<
                                domain::BoundaryConditions::BoundaryCondition>,
                            2>,
                 2>{{{{make_bc(Direction<3>::lower_xi()),
                       make_bc(Direction<3>::upper_xi())}},
                     {{make_bc(Direction<3>::lower_eta()),
                       make_bc(Direction<3>::upper_eta())}}}},
      std::make_unique<TestHelpers::domain::BoundaryConditions::
                           TestCartoonBoundaryCondition<3>>()};

  const auto domain_3d = cylinder.create_domain();
  const auto& block = domain_3d.blocks()[0];
  const auto element_ids =
      initial_element_ids<3>(0, cylinder.initial_refinement_levels()[0]);
  std::vector<std::array<double, 3>> points{};
  for (const auto& element_id : element_ids) {
    const auto mesh = domain::create_initial_mesh(
        cylinder.initial_extents(), block, element_id,
        Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto);
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

// Number of grid points at which `data` returns the jet rest-mass density,
// and the number at which it returns the ambient one.
template <typename AnalyticData>
std::array<size_t, 2> count_jet_and_ambient_points(
    const AnalyticData& data,
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords) {
  const auto density =
      get<hydro::Tags::RestMassDensity<DataVector>>(data.variables(
          coords, tmpl::list<hydro::Tags::RestMassDensity<DataVector>>{}));
  size_t num_jet = 0;
  size_t num_ambient = 0;
  for (size_t i = 0; i < get(density).size(); ++i) {
    if (get(density)[i] == approx(jet_density)) {
      ++num_jet;
    } else if (get(density)[i] == approx(ambient_density)) {
      ++num_ambient;
    }
  }
  return {{num_jet, num_ambient}};
}

void test_selector_on_real_domain() {
  INFO("Selector, on coordinates a CartoonCylinder domain really produces");
  const auto coords = cartoon_cylinder_coordinates();
  const size_t num_points = get<0>(coords).size();
  REQUIRE(num_points > 0);
  // The domain is r in [0,8], z in [0,20]: every point has a non-negative
  // radius. This is exactly why SlabJet's `x <= 0` selector cannot work here.
  CHECK(min(get<0>(coords)) >= 0.);

  const auto jet = make_jet(0.);
  const auto counts = count_jet_and_ambient_points(jet, coords);
  CAPTURE(counts[0]);
  CAPTURE(counts[1]);
  CAPTURE(num_points);
  // The whole point: a non-empty subset is jet, and the remainder is ambient.
  CHECK(counts[0] > 0);
  CHECK(counts[1] > 0);
  CHECK(counts[0] + counts[1] == num_points);

  // Every jet point is inside the nozzle and every ambient point outside it.
  const auto profile = jet.nozzle_profile(coords);
  for (size_t i = 0; i < num_points; ++i) {
    const bool inside = get<0>(coords)[i] <= inlet_radius and
                        get<1>(coords)[i] <= nozzle_length;
    CHECK(profile[i] == (inside ? 1. : 0.));
  }

  {
    INFO(
        "SlabJet selects nothing on this domain -- the silent failure this "
        "test exists to catch");
    const grmhd::AnalyticData::SlabJet slab_jet{
        adiabatic_index,           ambient_density, ambient_pressure,
        ambient_electron_fraction, jet_density,     jet_pressure,
        jet_electron_fraction,     jet_velocity(),  inlet_radius,
        magnetic_field()};
    const auto slab_counts = count_jet_and_ambient_points(slab_jet, coords);
    CAPTURE(slab_counts[0]);
    // Same assertion as above (`counts[0] > 0`) would FAIL here.
    CHECK(slab_counts[0] == 0);
    CHECK(slab_counts[1] == num_points);
  }
}

void test_values_against_the_paper() {
  INFO("Values, against Del Zanna et al. (2003) Fig. 6");
  const auto jet = make_jet(0.);
  // One point inside the nozzle and one just outside it in each direction.
  // The inside point is the PRE-FILL check: Paper I states the beam is
  // "located at $r\\leq 1$ and $z\\leq 1$" at t=0
  // (papers/2002-del-zanna-bucciantini-hydro/tex/delzannal1.tex:860), i.e. the
  // nozzle volume is initial data, separately from the z=0 face being held
  // constant for all time (:855-858). Without the pre-fill the jet head lags
  // the paper by one nozzle-crossing time.
  tnsr::I<DataVector, 3, Frame::Inertial> x{3_st, 0.};
  get<0>(x) = DataVector{{0.5, 1.5, 0.5}};  // radius
  get<1>(x) = DataVector{{0.5, 0.5, 1.5}};  // symmetry axis
  const auto vars = jet.variables(
      x, tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                    hydro::Tags::Pressure<DataVector>,
                    hydro::Tags::SpatialVelocity<DataVector, 3>,
                    hydro::Tags::MagneticField<DataVector, 3>,
                    hydro::Tags::SpecificInternalEnergy<DataVector>,
                    hydro::Tags::LorentzFactor<DataVector>>{});
  const auto& rho = get<hydro::Tags::RestMassDensity<DataVector>>(vars);
  const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(vars);
  const auto& velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars);
  const auto& b_field = get<hydro::Tags::MagneticField<DataVector, 3>>(vars);

  // Beam: rho = 0.1, p = 0.01, v_z = 0.99.
  CHECK(get(rho)[0] == approx(jet_density));
  CHECK(get(pressure)[0] == approx(jet_pressure));
  CHECK(get<0>(velocity)[0] == approx(0.));
  CHECK(get<1>(velocity)[0] == approx(jet_speed));
  CHECK(get<2>(velocity)[0] == approx(0.));
  // Ambient: rho = 10, p = 0.01, v = 0. Both points outside the nozzle.
  for (const size_t i : {1_st, 2_st}) {
    CHECK(get(rho)[i] == approx(ambient_density));
    CHECK(get(pressure)[i] == approx(ambient_pressure));
    CHECK(get<0>(velocity)[i] == approx(0.));
    CHECK(get<1>(velocity)[i] == approx(0.));
    CHECK(get<2>(velocity)[i] == approx(0.));
  }
  // B_z = 0.1 everywhere, purely axial, beam and ambient alike.
  for (size_t i = 0; i < 3; ++i) {
    CHECK(get<0>(b_field)[i] == approx(0.));
    CHECK(get<1>(b_field)[i] == approx(axial_field));
    CHECK(get<2>(b_field)[i] == approx(0.));
  }
  // Gamma = 5/3, checked through the ideal-gas relation
  // epsilon = p / ((Gamma - 1) rho) on the paper's own states:
  // beam    0.01 / ((2/3) * 0.1) = 0.15
  // ambient 0.01 / ((2/3) * 10 ) = 0.0015
  const auto& specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(vars);
  CHECK(get(specific_internal_energy)[0] == approx(0.15));
  CHECK(get(specific_internal_energy)[1] == approx(0.0015));
  CHECK(get(specific_internal_energy)[0] ==
        approx(jet_pressure / ((adiabatic_index - 1.) * jet_density)));
}

void test_closed_form_beam_quantities() {
  INFO("Closed form, independent of the code");
  const auto jet = make_jet(0.);
  tnsr::I<DataVector, 3, Frame::Inertial> x{1_st, 0.};
  get<0>(x) = DataVector{{0.5}};
  get<1>(x) = DataVector{{0.5}};
  const auto vars =
      jet.variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                  hydro::Tags::LorentzFactor<DataVector>,
                                  hydro::Tags::SpatialVelocity<DataVector, 3>,
                                  hydro::Tags::MagneticField<DataVector, 3>>{});
  const auto& lorentz_factor =
      get<hydro::Tags::LorentzFactor<DataVector>>(vars);
  const auto& velocity = get<hydro::Tags::SpatialVelocity<DataVector, 3>>(vars);
  const auto& b_field = get<hydro::Tags::MagneticField<DataVector, 3>>(vars);
  const auto& rho = get<hydro::Tags::RestMassDensity<DataVector>>(vars);

  // W = 1 / sqrt(1 - 0.99^2) = 7.088812050083354
  const double expected_lorentz_factor = 1. / sqrt(1. - square(jet_speed));
  CHECK(expected_lorentz_factor == approx(7.088812050083354));
  CHECK(get(lorentz_factor)[0] == approx(expected_lorentz_factor));

  // The field is parallel to the flow, so |b|^2 = B^2/W^2 + (v.B)^2 = B_z^2
  // exactly, every W cancelling, and sigma = B_z^2 / rho = 0.01 / 0.1 = 0.1.
  double b_squared = 0.;
  double v_dot_b = 0.;
  for (size_t d = 0; d < 3; ++d) {
    b_squared += square(b_field.get(d)[0]);
    v_dot_b += velocity.get(d)[0] * b_field.get(d)[0];
  }
  const double comoving_b_squared =
      b_squared / square(get(lorentz_factor)[0]) + square(v_dot_b);
  CHECK(comoving_b_squared == approx(square(axial_field)));
  CHECK(comoving_b_squared / get(rho)[0] == approx(0.1));

  // Relativistic Mach number, Paper I's own definition and value
  // (delzannal1.tex:873): M = gamma v / (gamma_cs c_s) with
  // c_s^2 = Gamma p / w and the PLAIN enthalpy w = rho + Gamma p / (Gamma - 1)
  // -- no magnetic term, since Paper I is unmagnetized. This single number
  // pins Gamma, the sound-speed convention and the beam state at once.
  const double enthalpy =
      jet_density + adiabatic_index * jet_pressure / (adiabatic_index - 1.);
  CHECK(enthalpy == approx(0.125));
  const double sound_speed_squared = adiabatic_index * jet_pressure / enthalpy;
  CHECK(sound_speed_squared == approx(0.13333333333333333));
  const double lorentz_factor_of_sound_speed =
      1. / sqrt(1. - sound_speed_squared);
  CHECK(lorentz_factor_of_sound_speed == approx(1.0741723110591654));
  const double mach_number =
      get(lorentz_factor)[0] * jet_speed /
      (lorentz_factor_of_sound_speed * sqrt(sound_speed_squared));
  CHECK(mach_number == approx(17.892265530925521));
  // The paper quotes 17.9 (delzannal1.tex:873); agreement to its precision.
  CHECK(fabs(mach_number - 17.9) < 0.05);
}

void test_tag_completeness() {
  INFO("Every tag DirichletAnalytic asks for");
  const auto jet = make_jet(0.);
  tnsr::I<DataVector, 3, Frame::Inertial> x{2_st, 0.};
  get<0>(x) = DataVector{{0.5, 2.0}};
  get<1>(x) = DataVector{{0.5, 2.0}};
  // This list is copied from the dg_ghost() call in
  // Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryConditions/
  // DirichletAnalytic.cpp. A missing tag would otherwise only show up when a
  // job runs.
  const auto vars = jet.variables(
      x, tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                    hydro::Tags::ElectronFraction<DataVector>,
                    hydro::Tags::SpecificInternalEnergy<DataVector>,
                    hydro::Tags::Pressure<DataVector>,
                    hydro::Tags::Temperature<DataVector>,
                    hydro::Tags::SpatialVelocity<DataVector, 3>,
                    hydro::Tags::LorentzFactor<DataVector>,
                    hydro::Tags::MagneticField<DataVector, 3>,
                    hydro::Tags::DivergenceCleaningField<DataVector>,
                    gr::Tags::SpatialMetric<DataVector, 3>,
                    gr::Tags::InverseSpatialMetric<DataVector, 3>,
                    gr::Tags::SqrtDetSpatialMetric<DataVector>,
                    gr::Tags::Lapse<DataVector>,
                    gr::Tags::Shift<DataVector, 3>>{});
  // Minkowski background, as the paper is special relativistic.
  CHECK(get(get<gr::Tags::Lapse<DataVector>>(vars))[0] == approx(1.));
  CHECK(get(get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars))[0] ==
        approx(1.));
  CHECK(get<0>(get<gr::Tags::Shift<DataVector, 3>>(vars))[0] == approx(0.));
  CHECK(get<0, 0>(get<gr::Tags::SpatialMetric<DataVector, 3>>(vars))[0] ==
        approx(1.));
  // Divergence cleaning field starts at zero.
  CHECK(get(get<hydro::Tags::DivergenceCleaningField<DataVector>>(vars))[0] ==
        approx(0.));
  // Temperature is finite and follows the ideal-gas relation
  // T = (Gamma - 1) * epsilon for this equation of state.
  const auto& temperature = get<hydro::Tags::Temperature<DataVector>>(vars);
  const auto& specific_internal_energy =
      get<hydro::Tags::SpecificInternalEnergy<DataVector>>(vars);
  for (size_t i = 0; i < 2; ++i) {
    CHECK(get(temperature)[i] ==
          approx((adiabatic_index - 1.) * get(specific_internal_energy)[i]));
  }
  CHECK(get(get<hydro::Tags::ElectronFraction<DataVector>>(vars))[0] ==
        approx(jet_electron_fraction));
}

void test_smoothing() {
  INFO("Smoothing width zero reproduces the sharp selector exactly");
  const auto coords = cartoon_cylinder_coordinates();
  const auto sharp = make_jet(0.);
  const auto sharp_profile = sharp.nozzle_profile(coords);
  for (size_t i = 0; i < sharp_profile.size(); ++i) {
    CHECK((sharp_profile[i] == 0. or sharp_profile[i] == 1.));
    const bool inside = get<0>(coords)[i] <= inlet_radius and
                        get<1>(coords)[i] <= nozzle_length;
    CHECK(sharp_profile[i] == (inside ? 1. : 0.));
  }
  // A positive width is a sensitivity knob with no oracle; all that is
  // asserted is that it is a genuine blend, bounded and monotone in the
  // right sense.
  const auto smoothed = make_jet(0.05);
  const auto smoothed_profile = smoothed.nozzle_profile(coords);
  bool found_intermediate = false;
  for (size_t i = 0; i < smoothed_profile.size(); ++i) {
    CHECK(smoothed_profile[i] >= 0.);
    CHECK(smoothed_profile[i] <= 1.);
    if (smoothed_profile[i] > 1.e-8 and smoothed_profile[i] < 1. - 1.e-8) {
      found_intermediate = true;
    }
  }
  CHECK(found_intermediate);
  CHECK(sharp != smoothed);

  {
    INFO(
        "Smoothing touches only rho and v, because p and B do not jump. "
        "Paper I names exactly those two: 'density and velocity jumps are "
        "actually smoothed' (delzannal1.tex:869)");
    tnsr::I<DataVector, 3, Frame::Inertial> edge{3_st, 0.};
    // Straddling the nozzle edge at r = 1.
    get<0>(edge) = DataVector{{0.95, 1.0, 1.05}};
    get<1>(edge) = DataVector{{0.5, 0.5, 0.5}};
    const auto vars = smoothed.variables(
        edge, tmpl::list<hydro::Tags::Pressure<DataVector>,
                         hydro::Tags::MagneticField<DataVector, 3>,
                         hydro::Tags::RestMassDensity<DataVector>>{});
    const auto& pressure = get<hydro::Tags::Pressure<DataVector>>(vars);
    const auto& b_field = get<hydro::Tags::MagneticField<DataVector, 3>>(vars);
    const auto& density = get<hydro::Tags::RestMassDensity<DataVector>>(vars);
    for (size_t i = 0; i < 3; ++i) {
      // Uniform through the smoothing layer, so the blend cannot disturb them.
      CHECK(get(pressure)[i] == approx(ambient_pressure));
      CHECK(get<1>(b_field)[i] == approx(axial_field));
    }
    // Density genuinely varies across the layer.
    CHECK(get(density)[0] < get(density)[2]);
    CHECK(get(density)[1] > jet_density);
    CHECK(get(density)[1] < ambient_density);
  }
}

void test_creation_and_semantics() {
  INFO("Option parsing, serialization and semantics");
  register_classes_with_charm<grmhd::AnalyticData::AxisymmetricJet>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_data =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::AxisymmetricJet>(
          "AxisymmetricJet:\n"
          "  AdiabaticIndex: 1.6666666666666667\n"
          "  AmbientDensity: 10.\n"
          "  AmbientPressure: 0.01\n"
          "  AmbientElectronFraction: 0.\n"
          "  JetDensity: 0.1\n"
          "  JetPressure: 0.01\n"
          "  JetElectronFraction: 0.\n"
          "  JetVelocity: [0., 0.99, 0.]\n"
          "  InletRadius: 1.\n"
          "  NozzleLength: 1.\n"
          "  SmoothingWidth: 0.\n"
          "  MagneticField: [0., 0.1, 0.]\n")
          ->get_clone();
  const auto deserialized = serialize_and_deserialize(option_data);
  const auto& parsed =
      dynamic_cast<const grmhd::AnalyticData::AxisymmetricJet&>(*deserialized);
  CHECK(parsed == make_jet(0.));
  CHECK(parsed != make_jet(0.05));

  auto to_move = make_jet(0.);
  test_move_semantics(std::move(to_move), make_jet(0.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.AxisymmetricJet",
                  "[Unit][PointwiseFunctions]") {
  test_creation_and_semantics();
  test_selector_on_real_domain();
  test_values_against_the_paper();
  test_closed_form_beam_quantities();
  test_tag_completeness();
  test_smoothing();
}
