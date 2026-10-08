// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/Printf/Printf.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hllem.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/TestCreation.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

// HLLEM is the HLL flux plus an eigenvector-based anti-diffusion that restores
// selected intermediate waves; in flat space it is active and in curved space
// it falls back to the (conservative) HLL flux, so this checks conservation,
// factory creation, serialization and option equality across wave-set choices.
// The anti-diffusion itself is exercised / compared against PLUTO on flat-space
// runs outside the unit tests.
namespace {
namespace helpers = TestHelpers::evolution::dg;
namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;

// One side of an interface: primitives, conserved variables, and the normal
// dot fluxes, all built from primitives in flat space.
struct InterfaceState {
  Scalar<DataVector> rest_mass_density{};
  Scalar<DataVector> electron_fraction{};
  Scalar<DataVector> specific_internal_energy{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> temperature{};
  Scalar<DataVector> lorentz_factor{};
  tnsr::I<DataVector, 3> spatial_velocity{};
  tnsr::i<DataVector, 3> spatial_velocity_one_form{};
  tnsr::I<DataVector, 3> magnetic_field{};

  Scalar<DataVector> tilde_d{};
  Scalar<DataVector> tilde_ye{};
  Scalar<DataVector> tilde_tau{};
  tnsr::i<DataVector, 3> tilde_s{};
  tnsr::I<DataVector, 3> tilde_b{};
  Scalar<DataVector> tilde_phi{};

  tnsr::I<DataVector, 3> flux_tilde_d{};
  tnsr::I<DataVector, 3> flux_tilde_ye{};
  tnsr::I<DataVector, 3> flux_tilde_tau{};
  tnsr::Ij<DataVector, 3> flux_tilde_s{};
  tnsr::IJ<DataVector, 3> flux_tilde_b{};
  tnsr::I<DataVector, 3> flux_tilde_phi{};
};

// Build a flat-space state at rest (v = 0) with the given density, uniform
// pressure and uniform magnetic field. With v = 0 and p, B continuous across
// the interface, the physical fluxes on the two sides are IDENTICAL and the
// jump is a pure contact (entropy) jump carried by TildeD and TildeTau.
InterfaceState make_state_at_rest(const double rest_mass_density,
                                 const double pressure,
                                 const std::array<double, 3>& magnetic_field,
                                 const double adiabatic_index,
                                 const size_t num_points) {
  InterfaceState state{};
  state.rest_mass_density = Scalar<DataVector>{num_points, rest_mass_density};
  state.electron_fraction = Scalar<DataVector>{num_points, 0.5};
  state.pressure = Scalar<DataVector>{num_points, pressure};
  const double specific_internal_energy =
      pressure / ((adiabatic_index - 1.0) * rest_mass_density);
  state.specific_internal_energy =
      Scalar<DataVector>{num_points, specific_internal_energy};
  // Ideal fluid: T = (Gamma - 1) * epsilon
  state.temperature = Scalar<DataVector>{
      num_points, (adiabatic_index - 1.0) * specific_internal_energy};
  state.lorentz_factor = Scalar<DataVector>{num_points, 1.0};
  state.spatial_velocity = tnsr::I<DataVector, 3>{num_points, 0.0};
  state.spatial_velocity_one_form = tnsr::i<DataVector, 3>{num_points, 0.0};
  state.magnetic_field = tnsr::I<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    state.magnetic_field.get(i) =
        DataVector{num_points, gsl::at(magnetic_field, i)};
  }

  const Scalar<DataVector> sqrt_det_spatial_metric{num_points, 1.0};
  const Scalar<DataVector> divergence_cleaning_field{num_points, 0.0};
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto spatial_metric = tnsr::ii<DataVector, 3>{num_points, 0.0};
  auto inv_spatial_metric = tnsr::II<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = DataVector{num_points, 1.0};
    inv_spatial_metric.get(i, i) = DataVector{num_points, 1.0};
  }

  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&state.tilde_d), make_not_null(&state.tilde_ye),
      make_not_null(&state.tilde_tau), make_not_null(&state.tilde_s),
      make_not_null(&state.tilde_b), make_not_null(&state.tilde_phi),
      state.rest_mass_density, state.electron_fraction,
      state.specific_internal_energy, state.pressure, state.spatial_velocity,
      state.lorentz_factor, state.magnetic_field, sqrt_det_spatial_metric,
      spatial_metric, divergence_cleaning_field);

  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&state.flux_tilde_d), make_not_null(&state.flux_tilde_ye),
      make_not_null(&state.flux_tilde_tau), make_not_null(&state.flux_tilde_s),
      make_not_null(&state.flux_tilde_b), make_not_null(&state.flux_tilde_phi),
      state.tilde_d, state.tilde_ye, state.tilde_tau, state.tilde_s,
      state.tilde_b, state.tilde_phi, lapse, shift, sqrt_det_spatial_metric,
      spatial_metric, inv_spatial_metric, state.pressure,
      state.spatial_velocity, state.lorentz_factor, state.magnetic_field);
  return state;
}

// Everything dg_package_data hands to dg_boundary_terms for one side.
struct Packaged {
  Scalar<DataVector> tilde_d{}, tilde_ye{}, tilde_tau{}, tilde_phi{};
  tnsr::i<DataVector, 3> tilde_s{};
  tnsr::I<DataVector, 3> tilde_b{};
  Scalar<DataVector> nf_tilde_d{}, nf_tilde_ye{}, nf_tilde_tau{}, nf_tilde_phi{};
  tnsr::i<DataVector, 3> nf_tilde_s{};
  tnsr::I<DataVector, 3> nf_tilde_b{};
  Scalar<DataVector> largest_outgoing{}, largest_ingoing{}, fast_outgoing{},
      fast_ingoing{}, metric_flatness{};
  tnsr::i<DataVector, 3> interface_unit_normal{};
  Scalar<DataVector> rest_mass_density{}, pressure{}, lorentz_factor{},
      specific_internal_energy{};
  tnsr::I<DataVector, 3> spatial_velocity{};
};

Packaged package_interface_state(
    const bc::Hllem& solver, const InterfaceState& state,
    const tnsr::i<DataVector, 3>& normal_covector,
    const tnsr::I<DataVector, 3>& normal_vector, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3>& shift,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  Packaged packaged{};
  solver.dg_package_data(
      make_not_null(&packaged.tilde_d), make_not_null(&packaged.tilde_ye),
      make_not_null(&packaged.tilde_tau), make_not_null(&packaged.tilde_s),
      make_not_null(&packaged.tilde_b), make_not_null(&packaged.tilde_phi),
      make_not_null(&packaged.nf_tilde_d), make_not_null(&packaged.nf_tilde_ye),
      make_not_null(&packaged.nf_tilde_tau), make_not_null(&packaged.nf_tilde_s),
      make_not_null(&packaged.nf_tilde_b), make_not_null(&packaged.nf_tilde_phi),
      make_not_null(&packaged.largest_outgoing),
      make_not_null(&packaged.largest_ingoing),
      make_not_null(&packaged.fast_outgoing),
      make_not_null(&packaged.fast_ingoing),
      make_not_null(&packaged.interface_unit_normal),
      make_not_null(&packaged.metric_flatness),
      make_not_null(&packaged.rest_mass_density),
      make_not_null(&packaged.spatial_velocity),
      make_not_null(&packaged.pressure),
      make_not_null(&packaged.lorentz_factor),
      make_not_null(&packaged.specific_internal_energy), state.tilde_d,
      state.tilde_ye, state.tilde_tau, state.tilde_s, state.tilde_b,
      state.tilde_phi, state.flux_tilde_d, state.flux_tilde_ye,
      state.flux_tilde_tau, state.flux_tilde_s, state.flux_tilde_b,
      state.flux_tilde_phi, lapse, shift, state.spatial_velocity_one_form,
      state.rest_mass_density, state.electron_fraction, state.temperature,
      state.spatial_velocity, state.specific_internal_energy, state.pressure,
      state.lorentz_factor, normal_covector, normal_vector, {}, {},
      equation_of_state);
  return packaged;
}

// The same for `Hll`, whose `dg_package_data` takes the same inputs and fills
// the same first eighteen outputs (it does not package the primitives that
// `Hllem` needs for its averaged interface state).
Packaged package_interface_state_hll(
    const bc::Hll& solver, const InterfaceState& state,
    const tnsr::i<DataVector, 3>& normal_covector,
    const tnsr::I<DataVector, 3>& normal_vector,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  Packaged packaged{};
  solver.dg_package_data(
      make_not_null(&packaged.tilde_d), make_not_null(&packaged.tilde_ye),
      make_not_null(&packaged.tilde_tau), make_not_null(&packaged.tilde_s),
      make_not_null(&packaged.tilde_b), make_not_null(&packaged.tilde_phi),
      make_not_null(&packaged.nf_tilde_d), make_not_null(&packaged.nf_tilde_ye),
      make_not_null(&packaged.nf_tilde_tau),
      make_not_null(&packaged.nf_tilde_s), make_not_null(&packaged.nf_tilde_b),
      make_not_null(&packaged.nf_tilde_phi),
      make_not_null(&packaged.largest_outgoing),
      make_not_null(&packaged.largest_ingoing),
      make_not_null(&packaged.fast_outgoing),
      make_not_null(&packaged.fast_ingoing),
      make_not_null(&packaged.interface_unit_normal),
      make_not_null(&packaged.metric_flatness), state.tilde_d, state.tilde_ye,
      state.tilde_tau, state.tilde_s, state.tilde_b, state.tilde_phi,
      state.flux_tilde_d, state.flux_tilde_ye, state.flux_tilde_tau,
      state.flux_tilde_s, state.flux_tilde_b, state.flux_tilde_phi, lapse,
      shift, state.spatial_velocity_one_form, state.rest_mass_density,
      state.electron_fraction, state.temperature, state.spatial_velocity,
      state.specific_internal_energy, state.pressure, state.lorentz_factor,
      normal_covector, normal_vector, {}, {}, equation_of_state);
  return packaged;
}

// A whole single-point boundary correction, flattened so that two solvers can
// be compared component by component: d, ye, tau, phi, s_i, b_i.
std::array<double, 10> flatten_correction(
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau, const Scalar<DataVector>& tilde_phi,
    const tnsr::i<DataVector, 3>& tilde_s,
    const tnsr::I<DataVector, 3>& tilde_b) {
  return std::array<double, 10>{
      {get(tilde_d)[0], get(tilde_ye)[0], get(tilde_tau)[0], get(tilde_phi)[0],
       tilde_s.get(0)[0], tilde_s.get(1)[0], tilde_s.get(2)[0],
       tilde_b.get(0)[0], tilde_b.get(1)[0], tilde_b.get(2)[0]}};
}

// As make_state_at_rest but with a non-zero spatial velocity. The strong-blast
// crashes happen mid-evolution, where the fluid is moving fast, so the at-rest
// states alone do not probe the failing corner.
InterfaceState make_state(const double rest_mass_density, const double pressure,
                          const std::array<double, 3>& velocity,
                          const std::array<double, 3>& magnetic_field,
                          const double adiabatic_index,
                          const size_t num_points) {
  InterfaceState state = make_state_at_rest(rest_mass_density, pressure,
                                            magnetic_field, adiabatic_index,
                                            num_points);
  double v_squared = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    v_squared += gsl::at(velocity, i) * gsl::at(velocity, i);
  }
  const double lorentz_factor = 1.0 / sqrt(1.0 - v_squared);
  state.lorentz_factor = Scalar<DataVector>{num_points, lorentz_factor};
  for (size_t i = 0; i < 3; ++i) {
    state.spatial_velocity.get(i) =
        DataVector{num_points, gsl::at(velocity, i)};
    state.spatial_velocity_one_form.get(i) =
        DataVector{num_points, gsl::at(velocity, i)};
  }

  const Scalar<DataVector> sqrt_det_spatial_metric{num_points, 1.0};
  const Scalar<DataVector> divergence_cleaning_field{num_points, 0.0};
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto spatial_metric = tnsr::ii<DataVector, 3>{num_points, 0.0};
  auto inv_spatial_metric = tnsr::II<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric.get(i, i) = DataVector{num_points, 1.0};
    inv_spatial_metric.get(i, i) = DataVector{num_points, 1.0};
  }
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&state.tilde_d), make_not_null(&state.tilde_ye),
      make_not_null(&state.tilde_tau), make_not_null(&state.tilde_s),
      make_not_null(&state.tilde_b), make_not_null(&state.tilde_phi),
      state.rest_mass_density, state.electron_fraction,
      state.specific_internal_energy, state.pressure, state.spatial_velocity,
      state.lorentz_factor, state.magnetic_field, sqrt_det_spatial_metric,
      spatial_metric, divergence_cleaning_field);
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&state.flux_tilde_d), make_not_null(&state.flux_tilde_ye),
      make_not_null(&state.flux_tilde_tau), make_not_null(&state.flux_tilde_s),
      make_not_null(&state.flux_tilde_b), make_not_null(&state.flux_tilde_phi),
      state.tilde_d, state.tilde_ye, state.tilde_tau, state.tilde_s,
      state.tilde_b, state.tilde_phi, lapse, shift, sqrt_det_spatial_metric,
      spatial_metric, inv_spatial_metric, state.pressure,
      state.spatial_velocity, state.lorentz_factor, state.magnetic_field);
  return state;
}

// Sweep the corner the strong-blast runs actually explore: fast flow, extreme
// pressure ratio, and a small-or-zero tangential field. Reports the FIRST
// combination that produces a non-finite boundary correction, which is the
// state to hand to the debugger.
void test_strong_blast_states_stay_finite() {
  const size_t num_points = 1;
  const double adiabatic_index = 4.0 / 3.0;
  const auto equation_of_state =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_int) = DataVector{num_points, 1.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_int) = DataVector{num_points, 1.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_ext) = DataVector{num_points, -1.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_ext) = DataVector{num_points, -1.0};

  size_t nonfinite_count = 0;
  for (const double normal_velocity : {0.0, 0.5, 0.9, 0.99, -0.9}) {
    for (const double tangential_velocity : {0.0, 0.3, 0.7}) {
      for (const double tangential_b : {0.0, 1.0e-9, 1.0e-3, 0.1, 1.0}) {
        for (const double pressure_ratio : {1.0, 100.0, 1000.0}) {
          const double v_squared = normal_velocity * normal_velocity +
                                   tangential_velocity * tangential_velocity;
          if (v_squared >= 0.99) {
            continue;
          }
          // The two sides must differ in velocity as well: a blast forms
          // interfaces where the two sides move at each other, which is the
          // configuration a matched-velocity sweep never reaches.
          const std::array<double, 3> velocity_int{
              {normal_velocity, tangential_velocity, 0.0}};
          const std::array<double, 3> velocity_ext{
              {-normal_velocity, tangential_velocity, 0.0}};
          const std::array<double, 3> magnetic_field{
              {1.0, tangential_b, 0.0}};
          const auto state_int =
              make_state(1.0, pressure_ratio, velocity_int, magnetic_field,
                         adiabatic_index, num_points);
          const auto state_ext =
              make_state(0.1, 1.0, velocity_ext, magnetic_field,
                         adiabatic_index, num_points);
          const bc::Hllem solver{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30,
                                 1.0e-8};
          const auto interior = package_interface_state(
              solver, state_int, normal_covector_int, normal_vector_int, lapse,
              shift, *equation_of_state);
          const auto exterior = package_interface_state(
              solver, state_ext, normal_covector_ext, normal_vector_ext, lapse,
              shift, *equation_of_state);
          Scalar<DataVector> c_d{num_points, 0.0}, c_ye{num_points, 0.0},
              c_tau{num_points, 0.0}, c_phi{num_points, 0.0};
          tnsr::i<DataVector, 3> c_s{num_points, 0.0};
          tnsr::I<DataVector, 3> c_b{num_points, 0.0};
          solver.dg_boundary_terms(
              make_not_null(&c_d), make_not_null(&c_ye), make_not_null(&c_tau),
              make_not_null(&c_s), make_not_null(&c_b), make_not_null(&c_phi),
              interior.tilde_d, interior.tilde_ye, interior.tilde_tau,
              interior.tilde_s, interior.tilde_b, interior.tilde_phi,
              interior.nf_tilde_d, interior.nf_tilde_ye, interior.nf_tilde_tau,
              interior.nf_tilde_s, interior.nf_tilde_b, interior.nf_tilde_phi,
              interior.largest_outgoing, interior.largest_ingoing,
              interior.fast_outgoing, interior.fast_ingoing,
              interior.interface_unit_normal, interior.metric_flatness,
              interior.rest_mass_density, interior.spatial_velocity,
              interior.pressure, interior.lorentz_factor,
              interior.specific_internal_energy, exterior.tilde_d,
              exterior.tilde_ye, exterior.tilde_tau, exterior.tilde_s,
              exterior.tilde_b, exterior.tilde_phi, exterior.nf_tilde_d,
              exterior.nf_tilde_ye, exterior.nf_tilde_tau, exterior.nf_tilde_s,
              exterior.nf_tilde_b, exterior.nf_tilde_phi,
              exterior.largest_outgoing, exterior.largest_ingoing,
              exterior.fast_outgoing, exterior.fast_ingoing,
              exterior.interface_unit_normal, exterior.metric_flatness,
              exterior.rest_mass_density, exterior.spatial_velocity,
              exterior.pressure, exterior.lorentz_factor,
              exterior.specific_internal_energy,
              ::dg::Formulation::StrongInertial, *equation_of_state);
          bool finite = std::isfinite(get(c_d)[0]) and
                        std::isfinite(get(c_tau)[0]) and
                        std::isfinite(get(c_phi)[0]);
          for (size_t i = 0; i < 3; ++i) {
            finite = finite and std::isfinite(c_s.get(i)[0]) and
                     std::isfinite(c_b.get(i)[0]);
          }
          if (not finite) {
            ++nonfinite_count;
            CAPTURE(normal_velocity);
            CAPTURE(tangential_velocity);
            CAPTURE(tangential_b);
            CAPTURE(pressure_ratio);
            CHECK(finite);
          }
        }
      }
    }
  }
  CHECK(nonfinite_count == 0);
}


// WHEN do the denominator floors in the characteristic decomposition actually
// bind? A floor that never binds is inert; one that binds says the eigensystem
// is being evaluated where it is not meaningful. This sweeps representative
// regimes and reports the count for each, which is the diagnostic that tells us
// whether a solver's answer in that regime can be trusted.
void report_when_denominator_floors_bind() {
  const size_t num_points = 1;
  const double adiabatic_index = 4.0 / 3.0;
  const auto equation_of_state =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto n_cov = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(n_cov) = DataVector{num_points, 1.0};
  auto n_vec = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(n_vec) = DataVector{num_points, 1.0};
  auto n_cov_e = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(n_cov_e) = DataVector{num_points, -1.0};
  auto n_vec_e = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(n_vec_e) = DataVector{num_points, -1.0};
  const bc::Hllem solver{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8};

  struct Regime {
    std::string name;
    double density;
    double pressure;
    std::array<double, 3> velocity;
    std::array<double, 3> magnetic_field;
  };
  const std::vector<Regime> regimes{
      {"ordinary fluid", 1.0, 1.0, {{0.2, 0.1, 0.0}}, {{1.0, 0.5, 0.3}}},
      {"strong blast", 1.0, 1000.0, {{0.0, 0.0, 0.0}}, {{1.0, 0.5, 0.3}}},
      {"zero tangential B", 1.0, 1.0, {{0.2, 0.0, 0.0}}, {{1.0, 0.0, 0.0}}},
      {"ultrarelativistic", 1.0, 1.0, {{0.99, 0.0, 0.0}}, {{1.0, 0.5, 0.3}}},
      {"near-atmosphere", 1.0e-12, 1.0e-14, {{0.0, 0.0, 0.0}}, {{1.0, 0.5, 0.3}}},
      {"cold (p -> 0)", 1.0, 1.0e-20, {{0.1, 0.0, 0.0}}, {{1.0, 0.5, 0.3}}}};

  for (const auto& regime : regimes) {
    const auto state_int =
        make_state(regime.density, regime.pressure, regime.velocity,
                   regime.magnetic_field, adiabatic_index, num_points);
    const auto state_ext =
        make_state(0.5 * regime.density, regime.pressure, regime.velocity,
                   regime.magnetic_field, adiabatic_index, num_points);
    grmhd::ValenciaDivClean::reset_denominator_floor_count();
    const auto interior = package_interface_state(
        solver, state_int, n_cov, n_vec, lapse, shift, *equation_of_state);
    const auto exterior = package_interface_state(
        solver, state_ext, n_cov_e, n_vec_e, lapse, shift, *equation_of_state);
    Scalar<DataVector> c_d{num_points, 0.0}, c_ye{num_points, 0.0},
        c_tau{num_points, 0.0}, c_phi{num_points, 0.0};
    tnsr::i<DataVector, 3> c_s{num_points, 0.0};
    tnsr::I<DataVector, 3> c_b{num_points, 0.0};
    solver.dg_boundary_terms(
        make_not_null(&c_d), make_not_null(&c_ye), make_not_null(&c_tau),
        make_not_null(&c_s), make_not_null(&c_b), make_not_null(&c_phi),
        interior.tilde_d, interior.tilde_ye, interior.tilde_tau,
        interior.tilde_s, interior.tilde_b, interior.tilde_phi,
        interior.nf_tilde_d, interior.nf_tilde_ye, interior.nf_tilde_tau,
        interior.nf_tilde_s, interior.nf_tilde_b, interior.nf_tilde_phi,
        interior.largest_outgoing, interior.largest_ingoing,
        interior.fast_outgoing, interior.fast_ingoing,
        interior.interface_unit_normal, interior.metric_flatness,
        interior.rest_mass_density, interior.spatial_velocity,
        interior.pressure, interior.lorentz_factor,
        interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
        exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
        exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
        exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
        exterior.nf_tilde_phi, exterior.largest_outgoing,
        exterior.largest_ingoing, exterior.fast_outgoing,
        exterior.fast_ingoing, exterior.interface_unit_normal,
        exterior.metric_flatness, exterior.rest_mass_density,
        exterior.spatial_velocity, exterior.pressure, exterior.lorentz_factor,
        exterior.specific_internal_energy, ::dg::Formulation::StrongInertial,
        *equation_of_state);
    Parallel::printf("  floors bound %5zu times : %s\n",
                     grmhd::ValenciaDivClean::denominator_floor_count(),
                     regime.name);
  }
}

// HLLEM crashes with a floating-point exception on strong-blast problems whose
// tangential magnetic field (nearly) vanishes: Komissarov shock tube 1
// (B = (1,0,0), so B_t is EXACTLY zero) and Balsara test 3 (B_t/B_n ~ 0.1 on
// the right state) both die, while Balsara test 2 (B_t/B_n ~ 0.14) survives.
// B_t -> 0 is a known degeneracy of the RMHD eigensystem: the Alfven and slow
// waves collapse onto the entropy wave and the analytic eigenvectors become
// singular. The solver is supposed to survive this via its degeneracy guard, so
// pin that down: the boundary correction must stay FINITE all the way to
// B_t = 0 exactly, for every wave set.
void test_vanishing_tangential_field_stays_finite() {
  const size_t num_points = 1;
  const double adiabatic_index = 4.0 / 3.0;
  const auto equation_of_state =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();

  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_int) = DataVector{num_points, 1.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_int) = DataVector{num_points, 1.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_ext) = DataVector{num_points, -1.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_ext) = DataVector{num_points, -1.0};

  // Komissarov shock tube 1 pressures/densities, with the tangential field
  // taken down to zero. B_t = 0 is the exact Komissarov ST1 configuration.
  for (const double tangential_b :
       {1.0, 1.0e-1, 1.0e-3, 1.0e-6, 1.0e-9, 1.0e-13, 0.0}) {
    const std::array<double, 3> magnetic_field{{1.0, tangential_b, 0.0}};
    const auto state_int =
        make_state_at_rest(1.0, 1000.0, magnetic_field, adiabatic_index,
                           num_points);
    const auto state_ext =
        make_state_at_rest(0.1, 1.0, magnetic_field, adiabatic_index,
                           num_points);
    for (const auto waves :
         {bc::HllemWaves::All, bc::HllemWaves::ContactSlow,
          bc::HllemWaves::ContactAlfven, bc::HllemWaves::AllWithFast}) {
      const bc::Hllem solver{waves, false, 1.0e-10, 1.0e-30, 1.0e-8};
      CAPTURE(tangential_b);
      CAPTURE(waves);
      const auto interior = package_interface_state(
          solver, state_int, normal_covector_int, normal_vector_int, lapse,
          shift, *equation_of_state);
      const auto exterior = package_interface_state(
          solver, state_ext, normal_covector_ext, normal_vector_ext, lapse,
          shift, *equation_of_state);

      Scalar<DataVector> correction_tilde_d{num_points, 0.0};
      Scalar<DataVector> correction_tilde_ye{num_points, 0.0};
      Scalar<DataVector> correction_tilde_tau{num_points, 0.0};
      tnsr::i<DataVector, 3> correction_tilde_s{num_points, 0.0};
      tnsr::I<DataVector, 3> correction_tilde_b{num_points, 0.0};
      Scalar<DataVector> correction_tilde_phi{num_points, 0.0};
      solver.dg_boundary_terms(
          make_not_null(&correction_tilde_d),
          make_not_null(&correction_tilde_ye),
          make_not_null(&correction_tilde_tau),
          make_not_null(&correction_tilde_s),
          make_not_null(&correction_tilde_b),
          make_not_null(&correction_tilde_phi), interior.tilde_d,
          interior.tilde_ye, interior.tilde_tau, interior.tilde_s,
          interior.tilde_b, interior.tilde_phi, interior.nf_tilde_d,
          interior.nf_tilde_ye, interior.nf_tilde_tau, interior.nf_tilde_s,
          interior.nf_tilde_b, interior.nf_tilde_phi,
          interior.largest_outgoing, interior.largest_ingoing,
          interior.fast_outgoing, interior.fast_ingoing,
          interior.interface_unit_normal, interior.metric_flatness,
          interior.rest_mass_density, interior.spatial_velocity,
          interior.pressure, interior.lorentz_factor,
          interior.specific_internal_energy, exterior.tilde_d,
          exterior.tilde_ye, exterior.tilde_tau, exterior.tilde_s,
          exterior.tilde_b, exterior.tilde_phi, exterior.nf_tilde_d,
          exterior.nf_tilde_ye, exterior.nf_tilde_tau, exterior.nf_tilde_s,
          exterior.nf_tilde_b, exterior.nf_tilde_phi,
          exterior.largest_outgoing, exterior.largest_ingoing,
          exterior.fast_outgoing, exterior.fast_ingoing,
          exterior.interface_unit_normal, exterior.metric_flatness,
          exterior.rest_mass_density, exterior.spatial_velocity,
          exterior.pressure, exterior.lorentz_factor,
          exterior.specific_internal_energy,
          ::dg::Formulation::StrongInertial, *equation_of_state);

      CHECK(std::isfinite(get(correction_tilde_d)[0]));
      CHECK(std::isfinite(get(correction_tilde_tau)[0]));
      CHECK(std::isfinite(get(correction_tilde_phi)[0]));
      for (size_t i = 0; i < 3; ++i) {
        CHECK(std::isfinite(correction_tilde_s.get(i)[0]));
        CHECK(std::isfinite(correction_tilde_b.get(i)[0]));
      }
    }
  }
}


// `Slow` and `Alfven` restore ONE pair each -- and this asserts WHICH pair,
// rather than merely that something changed.
//
// The lever is that the HLLEM anti-diffusion is a SUM over the restored waves,
//
//   G(S) = G_HLL - K * sum_{k in S} delta_k (l_k . dU) r_k ,
//
// with each wave's term depending only on that wave. (The per-wave speed-gap
// and conditioning guards are also per-wave: a wave is dropped at a point by
// comparing its own speed against all nine speeds, never against the restored
// SET. The one part that is not additive is the complementary projection, so
// these runs use UseComplementaryProjection = false.) The wave sets are then
// linearly related, and with restored_wave_indices as claimed --
// Contact = {4}, Slow = {3,5}, Alfven = {2,6}, ContactSlow = {3,4,5},
// ContactAlfven = {2,4,6}, All = {2,3,4,5,6}, None = {} -- the identities
//
//   ContactSlow   = Contact + Slow   - None
//   ContactAlfven = Contact + Alfven - None
//   All           = Contact + Slow   + Alfven - 2 * None
//
// hold component by component to roundoff. They pin the index sets down: give
// `Slow` the Alfven pair, or one slow root and one fast one, or three waves
// instead of two, and the first two identities break immediately.
void test_single_wave_pairs(
    const InterfaceState& state_int, const InterfaceState& state_ext,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const std::array<double, 3>& normal_direction, const std::string& what,
    const bool expect_degenerate_pairs) {
  const size_t num_points = 1;
  CAPTURE(what);
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    normal_covector_int.get(i) =
        DataVector{num_points, gsl::at(normal_direction, i)};
    normal_vector_int.get(i) =
        DataVector{num_points, gsl::at(normal_direction, i)};
    normal_covector_ext.get(i) =
        DataVector{num_points, -gsl::at(normal_direction, i)};
    normal_vector_ext.get(i) =
        DataVector{num_points, -gsl::at(normal_direction, i)};
  }

  const auto run = [&](const bc::HllemWaves waves) {
    const bc::Hllem solver{waves, false, 1.0e-10, 1.0e-30, 1.0e-8};
    const auto interior = package_interface_state(
        solver, state_int, normal_covector_int, normal_vector_int, lapse, shift,
        equation_of_state);
    const auto exterior = package_interface_state(
        solver, state_ext, normal_covector_ext, normal_vector_ext, lapse, shift,
        equation_of_state);
    Scalar<DataVector> correction_tilde_d{num_points, 0.0};
    Scalar<DataVector> correction_tilde_ye{num_points, 0.0};
    Scalar<DataVector> correction_tilde_tau{num_points, 0.0};
    tnsr::i<DataVector, 3> correction_tilde_s{num_points, 0.0};
    tnsr::I<DataVector, 3> correction_tilde_b{num_points, 0.0};
    Scalar<DataVector> correction_tilde_phi{num_points, 0.0};
    solver.dg_boundary_terms(
        make_not_null(&correction_tilde_d), make_not_null(&correction_tilde_ye),
        make_not_null(&correction_tilde_tau),
        make_not_null(&correction_tilde_s), make_not_null(&correction_tilde_b),
        make_not_null(&correction_tilde_phi), interior.tilde_d,
        interior.tilde_ye, interior.tilde_tau, interior.tilde_s,
        interior.tilde_b, interior.tilde_phi, interior.nf_tilde_d,
        interior.nf_tilde_ye, interior.nf_tilde_tau, interior.nf_tilde_s,
        interior.nf_tilde_b, interior.nf_tilde_phi, interior.largest_outgoing,
        interior.largest_ingoing, interior.fast_outgoing, interior.fast_ingoing,
        interior.interface_unit_normal, interior.metric_flatness,
        interior.rest_mass_density, interior.spatial_velocity,
        interior.pressure, interior.lorentz_factor,
        interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
        exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
        exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
        exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
        exterior.nf_tilde_phi, exterior.largest_outgoing,
        exterior.largest_ingoing, exterior.fast_outgoing, exterior.fast_ingoing,
        exterior.interface_unit_normal, exterior.metric_flatness,
        exterior.rest_mass_density, exterior.spatial_velocity,
        exterior.pressure, exterior.lorentz_factor,
        exterior.specific_internal_energy, ::dg::Formulation::StrongInertial,
        equation_of_state);
    return flatten_correction(correction_tilde_d, correction_tilde_ye,
                              correction_tilde_tau, correction_tilde_phi,
                              correction_tilde_s, correction_tilde_b);
  };

  const auto none = run(bc::HllemWaves::None);
  const auto contact = run(bc::HllemWaves::Contact);
  const auto slow = run(bc::HllemWaves::Slow);
  const auto alfven = run(bc::HllemWaves::Alfven);
  const auto contact_slow = run(bc::HllemWaves::ContactSlow);
  const auto contact_alfven = run(bc::HllemWaves::ContactAlfven);
  const auto all = run(bc::HllemWaves::All);

  // The scale every comparison below is measured against: the largest
  // anti-diffusion any wave set applies on this interface. If this were tiny
  // the identities would be trivially satisfied, so it is checked, not assumed.
  double scale = 0.0;
  for (size_t n = 0; n < 10; ++n) {
    scale = std::max(scale, std::abs(gsl::at(all, n) - gsl::at(none, n)));
    scale = std::max(scale, std::abs(gsl::at(none, n)));
  }
  CAPTURE(scale);
  CHECK(scale > 1.0e-8);
  Approx sum_approx = Approx::custom().epsilon(1.0e-11).scale(scale);

  double slow_deviation = 0.0;
  double alfven_deviation = 0.0;
  for (size_t n = 0; n < 10; ++n) {
    CAPTURE(n);
    CHECK(std::isfinite(gsl::at(slow, n)));
    CHECK(std::isfinite(gsl::at(alfven, n)));
    CHECK(gsl::at(contact_slow, n) ==
          sum_approx(gsl::at(contact, n) + gsl::at(slow, n) -
                     gsl::at(none, n)));
    CHECK(gsl::at(contact_alfven, n) ==
          sum_approx(gsl::at(contact, n) + gsl::at(alfven, n) -
                     gsl::at(none, n)));
    CHECK(gsl::at(all, n) ==
          sum_approx(gsl::at(contact, n) + gsl::at(slow, n) +
                     gsl::at(alfven, n) - 2.0 * gsl::at(none, n)));
    slow_deviation =
        std::max(slow_deviation, std::abs(gsl::at(slow, n) - gsl::at(none, n)));
    alfven_deviation = std::max(
        alfven_deviation, std::abs(gsl::at(alfven, n) - gsl::at(none, n)));
  }
  CAPTURE(slow_deviation);
  CAPTURE(alfven_deviation);

  if (expect_degenerate_pairs) {
    // B_n = 0 is Anton et al.'s TYPE I degeneracy, and the measured result is
    // stronger than the one this case was written to check. It is not only the
    // Alfven pair that collapses onto the entropy speed there:
    // alfven- = slow- = entropy = slow+ = alfven+, so the DegeneracyTolerance
    // speed-gap guard drops the SLOW pair as well, at every point. Both
    // `Alfven` and `Slow` are then EXACTLY `None` -- checked here bit for bit,
    // not to a tolerance.
    //
    // That is worth stating plainly, because "B_n = 0 makes the Alfven waves
    // inert" reads as if it singled the Alfven waves out. It does not. On a
    // face whose normal is perpendicular to the field, HLLEM restores the
    // contact and nothing else whatever `WavesToRestore` says. The Del Zanna
    // jet's slow-wave failure therefore cannot be happening on such a face; it
    // needs faces where B_n != 0, which is the other case below.
    for (size_t n = 0; n < 10; ++n) {
      CAPTURE(n);
      CHECK(gsl::at(alfven, n) == gsl::at(none, n));
      CHECK(gsl::at(slow, n) == gsl::at(none, n));
      CHECK(gsl::at(contact_slow, n) == gsl::at(contact, n));
      CHECK(gsl::at(contact_alfven, n) == gsl::at(contact, n));
      CHECK(gsl::at(all, n) == gsl::at(contact, n));
    }
  } else {
    // The slow pair is live here: `Slow` is not a relabelling of `None`.
    CHECK(slow_deviation > 1.0e-3 * scale);
    CHECK(alfven_deviation > 1.0e-3 * scale);
    // ... and the two pairs are not the same pair under another name.
    double slow_vs_alfven = 0.0;
    for (size_t n = 0; n < 10; ++n) {
      slow_vs_alfven = std::max(
          slow_vs_alfven, std::abs(gsl::at(slow, n) - gsl::at(alfven, n)));
    }
    CAPTURE(slow_vs_alfven);
    CHECK(slow_vs_alfven > 1.0e-3 * scale);
  }
}

// The HLLEM anti-diffusion is built so that a wave sitting exactly at
// lambda = 0 gets the full Einfeldt weight delta = 1 (Mattia & Mignone 2022,
// eq. 61), which exactly cancels the HLL diffusion for a jump lying entirely
// along that wave's eigenvector. An isolated STATIONARY CONTACT is precisely
// that situation: v = 0 with p and B continuous, so the physical fluxes agree
// on the two sides and the whole jump is the entropy mode. HLLEM must then
// return the common physical flux exactly, while HLL smears it. This is the
// property that makes HLLEM competitive with HLLC/HLLD on contact-dominated
// problems, so it is worth pinning down in a unit test.
void test_stationary_contact_is_exact(const double density_ratio,
                                      const bool require_exact) {
  const size_t num_points = 1;
  const double adiabatic_index = 5.0 / 3.0;
  const auto equation_of_state =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();

  // Density jumps by `density_ratio`; pressure, velocity (zero) and the
  // magnetic field (with a non-zero normal component) are continuous.
  const double pressure = 1.0;
  const std::array<double, 3> magnetic_field{{0.5, 0.3, 0.2}};
  const auto state_int = make_state_at_rest(1.0, pressure, magnetic_field,
                                            adiabatic_index, num_points);
  const auto state_ext = make_state_at_rest(
      density_ratio, pressure, magnetic_field, adiabatic_index, num_points);

  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};

  // Interface normal along x for the interior; the exterior packages its data
  // with the opposite (its own outward) normal.
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_int) = DataVector{num_points, 1.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_int) = DataVector{num_points, 1.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_ext) = DataVector{num_points, -1.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_ext) = DataVector{num_points, -1.0};


  const auto package = [&](const bc::Hllem& solver, const InterfaceState& state,
                           const tnsr::i<DataVector, 3>& normal_covector,
                           const tnsr::I<DataVector, 3>& normal_vector) {
    Packaged packaged{};
    solver.dg_package_data(
        make_not_null(&packaged.tilde_d), make_not_null(&packaged.tilde_ye),
        make_not_null(&packaged.tilde_tau), make_not_null(&packaged.tilde_s),
        make_not_null(&packaged.tilde_b), make_not_null(&packaged.tilde_phi),
        make_not_null(&packaged.nf_tilde_d),
        make_not_null(&packaged.nf_tilde_ye),
        make_not_null(&packaged.nf_tilde_tau),
        make_not_null(&packaged.nf_tilde_s),
        make_not_null(&packaged.nf_tilde_b),
        make_not_null(&packaged.nf_tilde_phi),
        make_not_null(&packaged.largest_outgoing),
        make_not_null(&packaged.largest_ingoing),
        make_not_null(&packaged.fast_outgoing),
        make_not_null(&packaged.fast_ingoing),
        make_not_null(&packaged.interface_unit_normal),
        make_not_null(&packaged.metric_flatness),
        make_not_null(&packaged.rest_mass_density),
        make_not_null(&packaged.spatial_velocity),
        make_not_null(&packaged.pressure),
        make_not_null(&packaged.lorentz_factor),
        make_not_null(&packaged.specific_internal_energy), state.tilde_d,
        state.tilde_ye, state.tilde_tau, state.tilde_s, state.tilde_b,
        state.tilde_phi, state.flux_tilde_d, state.flux_tilde_ye,
        state.flux_tilde_tau, state.flux_tilde_s, state.flux_tilde_b,
        state.flux_tilde_phi, lapse, shift, state.spatial_velocity_one_form,
        state.rest_mass_density, state.electron_fraction, state.temperature,
        state.spatial_velocity, state.specific_internal_energy, state.pressure,
        state.lorentz_factor, normal_covector, normal_vector, {}, {},
        *equation_of_state);
    return packaged;
  };

  // Restoring the contact must give the exact flux; restoring only the Alfven
  // waves must NOT (the contact jump is then left to the HLL diffusion). That
  // contrast is the point of the test: it shows the cancellation comes from the
  // restored contact eigenvector and not from the test being trivial.
  for (const auto waves :
       {bc::HllemWaves::Contact, bc::HllemWaves::ContactSlow,
        bc::HllemWaves::ContactAlfven, bc::HllemWaves::All}) {
    const bc::Hllem solver{waves, false, 1.0e-10, 1.0e-30, 1.0e-8};
    const auto interior =
        package_interface_state(solver, state_int, normal_covector_int,
                                normal_vector_int, lapse, shift,
                                *equation_of_state);
    const auto exterior =
        package_interface_state(solver, state_ext, normal_covector_ext,
                                normal_vector_ext, lapse, shift,
                                *equation_of_state);

    Scalar<DataVector> correction_tilde_d{num_points, 0.0};
    Scalar<DataVector> correction_tilde_ye{num_points, 0.0};
    Scalar<DataVector> correction_tilde_tau{num_points, 0.0};
    tnsr::i<DataVector, 3> correction_tilde_s{num_points, 0.0};
    tnsr::I<DataVector, 3> correction_tilde_b{num_points, 0.0};
    Scalar<DataVector> correction_tilde_phi{num_points, 0.0};

    solver.dg_boundary_terms(
        make_not_null(&correction_tilde_d), make_not_null(&correction_tilde_ye),
        make_not_null(&correction_tilde_tau),
        make_not_null(&correction_tilde_s), make_not_null(&correction_tilde_b),
        make_not_null(&correction_tilde_phi), interior.tilde_d,
        interior.tilde_ye, interior.tilde_tau, interior.tilde_s,
        interior.tilde_b, interior.tilde_phi, interior.nf_tilde_d,
        interior.nf_tilde_ye, interior.nf_tilde_tau, interior.nf_tilde_s,
        interior.nf_tilde_b, interior.nf_tilde_phi, interior.largest_outgoing,
        interior.largest_ingoing, interior.fast_outgoing,
        interior.fast_ingoing, interior.interface_unit_normal,
        interior.metric_flatness, interior.rest_mass_density,
        interior.spatial_velocity, interior.pressure, interior.lorentz_factor,
        interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
        exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
        exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
        exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
        exterior.nf_tilde_phi, exterior.largest_outgoing,
        exterior.largest_ingoing, exterior.fast_outgoing,
        exterior.fast_ingoing, exterior.interface_unit_normal,
        exterior.metric_flatness, exterior.rest_mass_density,
        exterior.spatial_velocity, exterior.pressure, exterior.lorentz_factor,
        exterior.specific_internal_energy, ::dg::Formulation::StrongInertial,
        *equation_of_state);

    // The HLL diffusion that must be cancelled: coeff * (u_ext - u_int) in
    // TildeD. If this were tiny the test would pass trivially.
    // In the STRONG formulation the interior flux is already subtracted, so
    // the boundary correction of an exactly-resolved interface is ZERO: the
    // two sides carry the same physical flux (nf_ext = -nf_int) and only the
    // diffusion term survives. HLLEM's anti-diffusion must cancel it.
    const DataVector lambda_max =
        max(0.0, get(interior.largest_outgoing), -get(exterior.largest_ingoing));
    const DataVector lambda_min =
        min(0.0, get(interior.largest_ingoing), -get(exterior.largest_outgoing));
    const DataVector hll_diffusion_tilde_d =
        lambda_max * lambda_min / (lambda_max - lambda_min) *
        (get(exterior.tilde_d) - get(interior.tilde_d));
    // Residuals relative to the HLL diffusion this interface would otherwise
    // suffer: 0 means the contact is preserved exactly, 1 means no better than
    // HLL. TildeTau is normalized the same way even though its own HLL
    // diffusion vanishes here (the state has Delta(TildeTau) = 0 exactly), so a
    // non-zero TildeTau residual is anti-diffusion LEAKING out of the density
    // jump into the energy -- an error plain HLL does not make.
    const double scale = fabs(hll_diffusion_tilde_d[0]);
    const double residual_tilde_d = fabs(get(correction_tilde_d)[0]) / scale;
    const double residual_tilde_tau = fabs(get(correction_tilde_tau)[0]) / scale;
    CAPTURE(waves);
    CAPTURE(density_ratio);
    CAPTURE(scale);
    CAPTURE(residual_tilde_d);
    CAPTURE(residual_tilde_tau);
    // Guard against a trivially-passing test: plain HLL really does smear this
    // interface by an O(1) amount.
    CHECK(scale > 0.1 * fabs(get(exterior.tilde_d)[0] - get(interior.tilde_d)[0]));
    if (require_exact) {
      CHECK(residual_tilde_d < 1.0e-10);
      CHECK(residual_tilde_tau < 1.0e-10);
    }
  }
}

// The outer MHD bounds must be built from the two sides' OWN fast
// magnetosonic speeds, not from the averaged interface state alone. An average
// is an interior point of the two states, so it can sit INSIDE the true signal
// range -- the one thing the HLL construction requires its bounds not to do --
// and umbrella G's derivation shows no state function admits an unmargined
// average-state estimate at all. This checks the half of that which is
// checkable at a single interface: each side packages ITS OWN fast speeds, so
// `dg_boundary_terms` has something two-sided to take a min/max over.
void test_fast_speeds_are_packaged_per_side() {
  const size_t num_points = 1;
  const double adiabatic_index = 5.0 / 3.0;
  const auto equation_of_state =
      EquationsOfState::IdealFluid<true>{adiabatic_index}.promote_to_3d_eos();
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector) = DataVector{num_points, 1.0};
  auto normal_vector = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector) = DataVector{num_points, 1.0};

  // A dense (slow) side and a light (fast) side at the same pressure and
  // velocity: the two fast speeds differ by O(1), which is the configuration in
  // which a single averaged-state estimate cannot bound both.
  const std::array<double, 3> magnetic_field{{0.5, 0.3, 0.2}};
  const std::array<double, 3> velocity{{0.2, 0.0, 0.0}};
  const auto state_dense = make_state(10.0, 1.0, velocity, magnetic_field,
                                      adiabatic_index, num_points);
  const auto state_light = make_state(0.1, 1.0, velocity, magnetic_field,
                                      adiabatic_index, num_points);

  const bc::Hllem solver{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8};
  const auto dense = package_interface_state(solver, state_dense,
                                             normal_covector, normal_vector,
                                             lapse, shift, *equation_of_state);
  const auto light = package_interface_state(solver, state_light,
                                             normal_covector, normal_vector,
                                             lapse, shift, *equation_of_state);

  // That side's own fast speeds, computed here rather than read from the
  // solver: {ingoing, outgoing}.
  const auto own_fast_speeds = [&](const InterfaceState& state) {
    tnsr::ii<DataVector, 3> flat_metric{num_points, 0.0};
    for (size_t i = 0; i < 3; ++i) {
      flat_metric.get(i, i) = DataVector{num_points, 1.0};
    }
    const Scalar<DataVector> specific_enthalpy =
        hydro::relativistic_specific_enthalpy(state.rest_mass_density,
                                              state.specific_internal_energy,
                                              state.pressure);
    std::array<DataVector, 9> speeds{};
    grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd(
        make_not_null(&speeds), state.rest_mass_density,
        state.electron_fraction, state.specific_internal_energy,
        specific_enthalpy, state.spatial_velocity, state.lorentz_factor,
        state.magnetic_field, lapse, shift, flat_metric, normal_covector,
        *equation_of_state);
    return std::array<double, 2>{{speeds[1][0], speeds[7][0]}};
  };
  const auto fast_dense = own_fast_speeds(state_dense);
  const auto fast_light = own_fast_speeds(state_light);

  CHECK(get(dense.fast_ingoing)[0] == approx(fast_dense[0]));
  CHECK(get(dense.fast_outgoing)[0] == approx(fast_dense[1]));
  CHECK(get(light.fast_ingoing)[0] == approx(fast_light[0]));
  CHECK(get(light.fast_outgoing)[0] == approx(fast_light[1]));

  // The two sides really do disagree, so a bound that ignores one of them is
  // not a bound; without this the checks above could pass trivially.
  CHECK(fast_light[1] - fast_dense[1] > 0.1);
  CHECK(fast_dense[0] - fast_light[0] > 0.1);

  // And these are the MHD speeds, not the +/-c divergence-cleaning ones the
  // Largest tags carry.
  CHECK(get(dense.fast_outgoing)[0] < get(dense.largest_outgoing)[0]);
  CHECK(get(dense.fast_ingoing)[0] > get(dense.largest_ingoing)[0]);
}

// `Hllem` with `WavesToRestore: None` is HLLEM with every Einfeldt coefficient
// delta_k set to zero, and that is identically the HLL flux: an algebraic
// identity in all nine components, for an arbitrary invertible eigenvector
// matrix R and arbitrary lambda_-, lambda_+, U and F, with no state, no
// eigensystem and no equation of state entering it. `None` therefore doubles
// as a consistency check on the anti-diffusion -- if the solver does not reduce
// to `Hll`, the HLLEM implementation is wrong, and THAT is the finding.
//
// The reduction is close but NOT bitwise, and the one permitted difference is
// known in advance: `Hll` builds its outer MHD bounds two-sidedly from the two
// sides' own fast speeds, while `Hllem` uses the three-sided envelope that also
// carries the averaged interface state's fast speed. The envelope can only
// widen the fan, so HLLEM-None is slightly MORE dissipative than `Hll`, never
// less.
//
// So this test does not compare against a tolerance pulled out of the air. It
// reconstructs BOTH bound sets here and predicts the correction exactly:
//   * the prediction with the NARROW (two-sided) bounds must reproduce `Hll`,
//     which is what validates the reference implementation below, and
//   * the prediction with the WIDE (three-sided) bounds must reproduce
//     `Hllem`-`None`.
// Anything the envelope difference does not account for -- of either sign --
// fails. The dissipation coefficient is checked to have moved in the
// more-dissipative direction, and every other wave set is checked to deviate
// from `Hll` by orders of magnitude more than `None` does, so the test would
// fail if any wave were left restored.
void test_no_restored_waves_reduces_to_hll(
    const InterfaceState& state_int, const InterfaceState& state_ext,
    const double adiabatic_index,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  const size_t num_points = 1;
  // The two implementations differ only in the order of a couple of
  // floating-point operations (`* (1/dl)` against `/ dl`), so roundoff is
  // the only slack the identity checks below need.
  Approx local_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_int) = DataVector{num_points, 1.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_int) = DataVector{num_points, 1.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  get<0>(normal_covector_ext) = DataVector{num_points, -1.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  get<0>(normal_vector_ext) = DataVector{num_points, -1.0};

  const bc::Hll hll_solver{1.0e-30, 1.0e-8};
  const bc::Hllem hllem_none{bc::HllemWaves::None, false, 1.0e-10, 1.0e-30,
                             1.0e-8};

  const auto hllem_int = package_interface_state(
      hllem_none, state_int, normal_covector_int, normal_vector_int, lapse,
      shift, equation_of_state);
  const auto hllem_ext = package_interface_state(
      hllem_none, state_ext, normal_covector_ext, normal_vector_ext, lapse,
      shift, equation_of_state);
  const auto hll_int = package_interface_state_hll(
      hll_solver, state_int, normal_covector_int, normal_vector_int, lapse,
      shift, equation_of_state);
  const auto hll_ext = package_interface_state_hll(
      hll_solver, state_ext, normal_covector_ext, normal_vector_ext, lapse,
      shift, equation_of_state);

  // Premise 1: the two solvers package the SAME data, so the comparison below
  // is of the two flux formulas and nothing else. (If the packaged speeds
  // differed, an agreement -- or a disagreement -- would say nothing about the
  // anti-diffusion.)
  CHECK(get(hllem_int.largest_outgoing)[0] ==
        approx(get(hll_int.largest_outgoing)[0]));
  CHECK(get(hllem_int.largest_ingoing)[0] ==
        approx(get(hll_int.largest_ingoing)[0]));
  CHECK(get(hllem_int.fast_outgoing)[0] ==
        approx(get(hll_int.fast_outgoing)[0]));
  CHECK(get(hllem_int.fast_ingoing)[0] == approx(get(hll_int.fast_ingoing)[0]));
  CHECK(get(hllem_ext.fast_outgoing)[0] ==
        approx(get(hll_ext.fast_outgoing)[0]));
  CHECK(get(hllem_ext.fast_ingoing)[0] == approx(get(hll_ext.fast_ingoing)[0]));
  // Premise 2: the background really is flat, so `Hllem` takes its anti-
  // diffusion path rather than the curved-space HLL fallback -- otherwise the
  // agreement below would be the fallback's, and would say nothing.
  CHECK(get(hllem_int.metric_flatness)[0] < 1.0e-12);
  CHECK(get(hllem_ext.metric_flatness)[0] < 1.0e-12);

  // The solvers' own bounds, rebuilt here.
  // Narrow (what `Hll` uses): the two sides' own fast speeds.
  const double fast_max_narrow = std::max(
      {0.0, get(hllem_int.fast_outgoing)[0], -get(hllem_ext.fast_ingoing)[0]});
  const double fast_min_narrow = std::min(
      {0.0, get(hllem_int.fast_ingoing)[0], -get(hllem_ext.fast_outgoing)[0]});
  const double light_max = std::max({0.0, get(hllem_int.largest_outgoing)[0],
                                     -get(hllem_ext.largest_ingoing)[0]});
  const double light_min = std::min({0.0, get(hllem_int.largest_ingoing)[0],
                                     -get(hllem_ext.largest_outgoing)[0]});

  // Wide (what `Hllem` uses): the same envelope plus the averaged interface
  // state's own fast speeds. The averaged state is rebuilt exactly as
  // `Hllem::dg_boundary_terms` builds it -- rho and p averaged, eps taken from
  // the equation of state at (rho_avg, p_avg), which for this ideal fluid is
  // the closed form below (the solver's secant is exact in one step there).
  const double rho_avg = 0.5 * (get(state_int.rest_mass_density)[0] +
                                get(state_ext.rest_mass_density)[0]);
  const double p_avg =
      0.5 * (get(state_int.pressure)[0] + get(state_ext.pressure)[0]);
  const double eps_avg = p_avg / ((adiabatic_index - 1.0) * rho_avg);
  const Scalar<DataVector> rho_avg_dv{num_points, rho_avg};
  const Scalar<DataVector> eps_avg_dv{num_points, eps_avg};
  const Scalar<DataVector> ye_avg_dv{
      num_points, 0.5 * (get(state_int.electron_fraction)[0] +
                         get(state_ext.electron_fraction)[0])};
  tnsr::I<DataVector, 3> v_avg{num_points, 0.0};
  tnsr::I<DataVector, 3> b_avg{num_points, 0.0};
  double v_sq_avg = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    v_avg.get(i) =
        DataVector{num_points, 0.5 * (state_int.spatial_velocity.get(i)[0] +
                                      state_ext.spatial_velocity.get(i)[0])};
    // B = TildeB in flat space, and `Hllem` averages TildeB.
    b_avg.get(i) = DataVector{num_points, 0.5 * (state_int.tilde_b.get(i)[0] +
                                                 state_ext.tilde_b.get(i)[0])};
    v_sq_avg += v_avg.get(i)[0] * v_avg.get(i)[0];
  }
  const Scalar<DataVector> w_avg{num_points, 1.0 / sqrt(1.0 - v_sq_avg)};
  const Scalar<DataVector> enthalpy_avg = hydro::relativistic_specific_enthalpy(
      rho_avg_dv, eps_avg_dv,
      equation_of_state.pressure_from_density_and_energy(rho_avg_dv, eps_avg_dv,
                                                         ye_avg_dv));
  tnsr::ii<DataVector, 3> flat_metric{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    flat_metric.get(i, i) = DataVector{num_points, 1.0};
  }
  tnsr::i<DataVector, 9> speeds_avg{num_points, 0.0};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds_avg), v_avg, b_avg, rho_avg_dv, eps_avg_dv, w_avg,
      enthalpy_avg, flat_metric, hllem_int.interface_unit_normal,
      equation_of_state);
  const double fast_max_wide = std::max(fast_max_narrow, speeds_avg.get(7)[0]);
  const double fast_min_wide = std::min(fast_min_narrow, speeds_avg.get(1)[0]);

  // The envelope only ever widens the fan, hence only ever adds dissipation:
  // the HLL diffusion coefficient -l_max l_min / (l_max - l_min) must not go
  // DOWN. A `None` that were less dissipative than `Hll` would be a bug.
  const auto dissipation_coefficient = [](const double l_max,
                                          const double l_min) {
    return -l_max * l_min / std::max(l_max - l_min, 1.0e-30);
  };
  CHECK(fast_max_wide >= fast_max_narrow);
  CHECK(fast_min_wide <= fast_min_narrow);
  CHECK(dissipation_coefficient(fast_max_wide, fast_min_wide) >=
        dissipation_coefficient(fast_max_narrow, fast_min_narrow) - 1.0e-15);

  // Reference HLL flux (strong formulation) for one component, given bounds.
  const auto hll_component = [](const double l_max, const double l_min,
                                const double u_int, const double nf_int,
                                const double u_ext, const double nf_ext) {
    const double dl = std::max(l_max - l_min, 1.0e-30);
    return (l_min * (nf_int + nf_ext) + l_max * l_min * (u_ext - u_int)) / dl;
  };
  // The whole boundary correction, flattened: d, ye, tau, phi, s_i, b_i. The
  // GLM subsystem (Phi and the NORMAL magnetic field) rides the light bounds;
  // the MHD variables (D, Ye, Tau, S and the TANGENTIAL magnetic field) ride
  // the fast bounds. Both solvers do this; only the fast bounds differ.
  const auto reference_correction = [&](const double fast_max,
                                        const double fast_min) {
    std::array<double, 10> correction{};
    correction[0] =
        hll_component(fast_max, fast_min, get(hllem_int.tilde_d)[0],
                      get(hllem_int.nf_tilde_d)[0], get(hllem_ext.tilde_d)[0],
                      get(hllem_ext.nf_tilde_d)[0]);
    correction[1] =
        hll_component(fast_max, fast_min, get(hllem_int.tilde_ye)[0],
                      get(hllem_int.nf_tilde_ye)[0], get(hllem_ext.tilde_ye)[0],
                      get(hllem_ext.nf_tilde_ye)[0]);
    correction[2] = hll_component(
        fast_max, fast_min, get(hllem_int.tilde_tau)[0],
        get(hllem_int.nf_tilde_tau)[0], get(hllem_ext.tilde_tau)[0],
        get(hllem_ext.nf_tilde_tau)[0]);
    correction[3] = hll_component(
        light_max, light_min, get(hllem_int.tilde_phi)[0],
        get(hllem_int.nf_tilde_phi)[0], get(hllem_ext.tilde_phi)[0],
        get(hllem_ext.nf_tilde_phi)[0]);
    for (size_t i = 0; i < 3; ++i) {
      gsl::at(correction, 4 + i) = hll_component(
          fast_max, fast_min, hllem_int.tilde_s.get(i)[0],
          hllem_int.nf_tilde_s.get(i)[0], hllem_ext.tilde_s.get(i)[0],
          hllem_ext.nf_tilde_s.get(i)[0]);
    }
    double bn_int = 0.0;
    double bn_ext = 0.0;
    double nfbn_int = 0.0;
    double nfbn_ext = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      const double n_i = hllem_int.interface_unit_normal.get(i)[0];
      bn_int += hllem_int.tilde_b.get(i)[0] * n_i;
      bn_ext += hllem_ext.tilde_b.get(i)[0] * n_i;
      nfbn_int += hllem_int.nf_tilde_b.get(i)[0] * n_i;
      nfbn_ext += hllem_ext.nf_tilde_b.get(i)[0] * n_i;
    }
    const double g_bn =
        hll_component(light_max, light_min, bn_int, nfbn_int, bn_ext, nfbn_ext);
    for (size_t i = 0; i < 3; ++i) {
      const double n_i = hllem_int.interface_unit_normal.get(i)[0];
      const double g_bt = hll_component(
          fast_max, fast_min, hllem_int.tilde_b.get(i)[0] - bn_int * n_i,
          hllem_int.nf_tilde_b.get(i)[0] - nfbn_int * n_i,
          hllem_ext.tilde_b.get(i)[0] - bn_ext * n_i,
          hllem_ext.nf_tilde_b.get(i)[0] - nfbn_ext * n_i);
      gsl::at(correction, 7 + i) = g_bn * n_i + g_bt;
    }
    return correction;
  };
  const std::array<double, 10> predicted_hll =
      reference_correction(fast_max_narrow, fast_min_narrow);
  const std::array<double, 10> predicted_none =
      reference_correction(fast_max_wide, fast_min_wide);

  // Run the two solvers on the same packaged data.
  const auto run_hll = [&]() {
    Scalar<DataVector> c_d{num_points, 0.0}, c_ye{num_points, 0.0},
        c_tau{num_points, 0.0}, c_phi{num_points, 0.0};
    tnsr::i<DataVector, 3> c_s{num_points, 0.0};
    tnsr::I<DataVector, 3> c_b{num_points, 0.0};
    bc::Hll::dg_boundary_terms(
        make_not_null(&c_d), make_not_null(&c_ye), make_not_null(&c_tau),
        make_not_null(&c_s), make_not_null(&c_b), make_not_null(&c_phi),
        hll_int.tilde_d, hll_int.tilde_ye, hll_int.tilde_tau, hll_int.tilde_s,
        hll_int.tilde_b, hll_int.tilde_phi, hll_int.nf_tilde_d,
        hll_int.nf_tilde_ye, hll_int.nf_tilde_tau, hll_int.nf_tilde_s,
        hll_int.nf_tilde_b, hll_int.nf_tilde_phi, hll_int.largest_outgoing,
        hll_int.largest_ingoing, hll_int.fast_outgoing, hll_int.fast_ingoing,
        hll_int.interface_unit_normal, hll_int.metric_flatness, hll_ext.tilde_d,
        hll_ext.tilde_ye, hll_ext.tilde_tau, hll_ext.tilde_s, hll_ext.tilde_b,
        hll_ext.tilde_phi, hll_ext.nf_tilde_d, hll_ext.nf_tilde_ye,
        hll_ext.nf_tilde_tau, hll_ext.nf_tilde_s, hll_ext.nf_tilde_b,
        hll_ext.nf_tilde_phi, hll_ext.largest_outgoing, hll_ext.largest_ingoing,
        hll_ext.fast_outgoing, hll_ext.fast_ingoing,
        hll_ext.interface_unit_normal, hll_ext.metric_flatness,
        ::dg::Formulation::StrongInertial);
    return flatten_correction(c_d, c_ye, c_tau, c_phi, c_s, c_b);
  };
  const auto run_hllem = [&](const bc::HllemWaves waves,
                             const bool complementary_projection) {
    const bc::Hllem solver{waves, complementary_projection, 1.0e-10, 1.0e-30,
                           1.0e-8};
    Scalar<DataVector> c_d{num_points, 0.0}, c_ye{num_points, 0.0},
        c_tau{num_points, 0.0}, c_phi{num_points, 0.0};
    tnsr::i<DataVector, 3> c_s{num_points, 0.0};
    tnsr::I<DataVector, 3> c_b{num_points, 0.0};
    solver.dg_boundary_terms(
        make_not_null(&c_d), make_not_null(&c_ye), make_not_null(&c_tau),
        make_not_null(&c_s), make_not_null(&c_b), make_not_null(&c_phi),
        hllem_int.tilde_d, hllem_int.tilde_ye, hllem_int.tilde_tau,
        hllem_int.tilde_s, hllem_int.tilde_b, hllem_int.tilde_phi,
        hllem_int.nf_tilde_d, hllem_int.nf_tilde_ye, hllem_int.nf_tilde_tau,
        hllem_int.nf_tilde_s, hllem_int.nf_tilde_b, hllem_int.nf_tilde_phi,
        hllem_int.largest_outgoing, hllem_int.largest_ingoing,
        hllem_int.fast_outgoing, hllem_int.fast_ingoing,
        hllem_int.interface_unit_normal, hllem_int.metric_flatness,
        hllem_int.rest_mass_density, hllem_int.spatial_velocity,
        hllem_int.pressure, hllem_int.lorentz_factor,
        hllem_int.specific_internal_energy, hllem_ext.tilde_d,
        hllem_ext.tilde_ye, hllem_ext.tilde_tau, hllem_ext.tilde_s,
        hllem_ext.tilde_b, hllem_ext.tilde_phi, hllem_ext.nf_tilde_d,
        hllem_ext.nf_tilde_ye, hllem_ext.nf_tilde_tau, hllem_ext.nf_tilde_s,
        hllem_ext.nf_tilde_b, hllem_ext.nf_tilde_phi,
        hllem_ext.largest_outgoing, hllem_ext.largest_ingoing,
        hllem_ext.fast_outgoing, hllem_ext.fast_ingoing,
        hllem_ext.interface_unit_normal, hllem_ext.metric_flatness,
        hllem_ext.rest_mass_density, hllem_ext.spatial_velocity,
        hllem_ext.pressure, hllem_ext.lorentz_factor,
        hllem_ext.specific_internal_energy, ::dg::Formulation::StrongInertial,
        equation_of_state);
    return flatten_correction(c_d, c_ye, c_tau, c_phi, c_s, c_b);
  };

  const std::array<double, 10> hll_correction = run_hll();
  const std::array<double, 10> none_correction =
      run_hllem(bc::HllemWaves::None, false);

  // The scale every deviation is measured against: the HLL diffusion this
  // interface actually suffers. If it were tiny the test would pass trivially.
  double scale = 0.0;
  for (size_t n = 0; n < 10; ++n) {
    scale = std::max(scale, std::abs(gsl::at(hll_correction, n)));
  }
  CAPTURE(scale);
  CHECK(scale > 1.0e-3);

  // The reference implementation above reproduces `Hll` exactly, which is what
  // licenses using it to predict `Hllem`-`None`.
  for (size_t n = 0; n < 10; ++n) {
    CAPTURE(n);
    CHECK(gsl::at(hll_correction, n) ==
          local_approx(gsl::at(predicted_hll, n)));
  }
  // And `Hllem`-`None` is that same HLL flux with the three-sided bounds -- no
  // anti-diffusion anywhere in it.
  double none_deviation = 0.0;
  double envelope_difference = 0.0;
  for (size_t n = 0; n < 10; ++n) {
    CAPTURE(n);
    CHECK(gsl::at(none_correction, n) ==
          local_approx(gsl::at(predicted_none, n)));
    none_deviation = std::max(
        none_deviation,
        std::abs(gsl::at(none_correction, n) - gsl::at(hll_correction, n)));
    envelope_difference = std::max(
        envelope_difference,
        std::abs(gsl::at(predicted_none, n) - gsl::at(predicted_hll, n)));
  }
  CAPTURE(fast_max_narrow);
  CAPTURE(fast_max_wide);
  CAPTURE(fast_min_narrow);
  CAPTURE(fast_min_wide);
  CAPTURE(none_deviation);
  CAPTURE(envelope_difference);
  // Nothing beyond the envelope difference, in either direction.
  CHECK(none_deviation <= envelope_difference + 1.0e-12 * scale);
  // Reported rather than just asserted: this is the measurement that says HOW
  // far HLLEM-None is from HLL on this interface, and how much of that the
  // three-sided envelope accounts for.
  Parallel::printf(
      "  HLLEM(None) vs HLL: max deviation %.3e, envelope difference %.3e, "
      "HLL diffusion scale %.3e\n",
      none_deviation, envelope_difference, scale);

  // `UseComplementaryProjection` composes with `WavesToRestore` everywhere
  // else, but the complement fallback only fires where a RESTORED wave was
  // dropped, so with no restored waves it can never fire: `None` is the same
  // flux with the projection on.
  const std::array<double, 10> none_with_projection =
      run_hllem(bc::HllemWaves::None, true);
  for (size_t n = 0; n < 10; ++n) {
    CAPTURE(n);
    CHECK(gsl::at(none_with_projection, n) ==
          local_approx(gsl::at(none_correction, n)));
  }

  // DISCRIMINATION. A test that would also pass with the waves still switched
  // on is worthless, so check that every other wave set moves the flux by
  // orders of magnitude more than the envelope does. If `None` accidentally
  // left a wave restored, or if the empty-list path were not reached, this is
  // the check that fails.
  for (const auto waves :
       {bc::HllemWaves::Contact, bc::HllemWaves::ContactAlfven,
        bc::HllemWaves::ContactSlow, bc::HllemWaves::All,
        bc::HllemWaves::AllWithFast}) {
    const std::array<double, 10> restored = run_hllem(waves, false);
    double deviation = 0.0;
    for (size_t n = 0; n < 10; ++n) {
      deviation = std::max(deviation, std::abs(gsl::at(restored, n) -
                                               gsl::at(hll_correction, n)));
    }
    CAPTURE(waves);
    CAPTURE(deviation);
    CHECK(deviation > 1.0e-3 * scale);
    CHECK(deviation > 100.0 * none_deviation);
  }
}

// DEBUG (hllem_degeneracy_debug): Eigensystem Numeric (dgeev of the flux
// Jacobian) against Analytic on one interface. On a non-degenerate interface
// the two eigensystems span the same eigenspaces, so every wave set must give
// the same boundary correction to round-off; on a B_n = 0 interface (Type I,
// exactly degenerate) Numeric must drop every restored internal wave, so All
// equals None.
std::array<double, 10> run_hllem(
    const bc::Hllem& solver, const InterfaceState& state_int,
    const InterfaceState& state_ext,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const std::array<double, 3>& normal_direction) {
  const size_t num_points = 1;
  const Scalar<DataVector> lapse{num_points, 1.0};
  const tnsr::I<DataVector, 3> shift{num_points, 0.0};
  auto normal_covector_int = tnsr::i<DataVector, 3>{num_points, 0.0};
  auto normal_vector_int = tnsr::I<DataVector, 3>{num_points, 0.0};
  auto normal_covector_ext = tnsr::i<DataVector, 3>{num_points, 0.0};
  auto normal_vector_ext = tnsr::I<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    normal_covector_int.get(i) =
        DataVector{num_points, gsl::at(normal_direction, i)};
    normal_vector_int.get(i) =
        DataVector{num_points, gsl::at(normal_direction, i)};
    normal_covector_ext.get(i) =
        DataVector{num_points, -gsl::at(normal_direction, i)};
    normal_vector_ext.get(i) =
        DataVector{num_points, -gsl::at(normal_direction, i)};
  }
  const auto interior =
      package_interface_state(solver, state_int, normal_covector_int,
                              normal_vector_int, lapse, shift,
                              equation_of_state);
  const auto exterior =
      package_interface_state(solver, state_ext, normal_covector_ext,
                              normal_vector_ext, lapse, shift,
                              equation_of_state);
  Scalar<DataVector> c_d{num_points, 0.0};
  Scalar<DataVector> c_ye{num_points, 0.0};
  Scalar<DataVector> c_tau{num_points, 0.0};
  tnsr::i<DataVector, 3> c_s{num_points, 0.0};
  tnsr::I<DataVector, 3> c_b{num_points, 0.0};
  Scalar<DataVector> c_phi{num_points, 0.0};
  solver.dg_boundary_terms(
      make_not_null(&c_d), make_not_null(&c_ye), make_not_null(&c_tau),
      make_not_null(&c_s), make_not_null(&c_b), make_not_null(&c_phi),
      interior.tilde_d, interior.tilde_ye, interior.tilde_tau,
      interior.tilde_s, interior.tilde_b, interior.tilde_phi,
      interior.nf_tilde_d, interior.nf_tilde_ye, interior.nf_tilde_tau,
      interior.nf_tilde_s, interior.nf_tilde_b, interior.nf_tilde_phi,
      interior.largest_outgoing, interior.largest_ingoing,
      interior.fast_outgoing, interior.fast_ingoing,
      interior.interface_unit_normal, interior.metric_flatness,
      interior.rest_mass_density, interior.spatial_velocity,
      interior.pressure, interior.lorentz_factor,
      interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
      exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
      exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
      exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
      exterior.nf_tilde_phi, exterior.largest_outgoing,
      exterior.largest_ingoing, exterior.fast_outgoing, exterior.fast_ingoing,
      exterior.interface_unit_normal, exterior.metric_flatness,
      exterior.rest_mass_density, exterior.spatial_velocity,
      exterior.pressure, exterior.lorentz_factor,
      exterior.specific_internal_energy, ::dg::Formulation::StrongInertial,
      equation_of_state);
  return flatten_correction(c_d, c_ye, c_tau, c_phi, c_s, c_b);
}

void test_numeric_eigensystem(
    const InterfaceState& state_int, const InterfaceState& state_ext,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
    const std::array<double, 3>& normal_direction, const std::string& what,
    const bool exactly_degenerate) {
  CAPTURE(what);
  const auto run = [&](const bc::HllemWaves waves,
                       const bc::HllemEigensystem eigensystem) {
    return run_hllem(bc::Hllem{waves, false, 1.0e-10, 1.0e-30, 1.0e-8,
                               eigensystem, 1.0e300},
                     state_int, state_ext, equation_of_state,
                     normal_direction);
  };
  const auto none = run(bc::HllemWaves::None, bc::HllemEigensystem::Analytic);
  for (const auto waves :
       {bc::HllemWaves::Contact, bc::HllemWaves::ContactSlow,
        bc::HllemWaves::ContactAlfven, bc::HllemWaves::All,
        bc::HllemWaves::AllWithFast}) {
    CAPTURE(waves);
    const auto analytic = run(waves, bc::HllemEigensystem::Analytic);
    const auto numeric = run(waves, bc::HllemEigensystem::Numeric);
    double scale = 0.0;
    for (size_t n = 0; n < 10; ++n) {
      scale = std::max(scale, std::abs(gsl::at(none, n)));
      scale = std::max(scale,
                       std::abs(gsl::at(analytic, n) - gsl::at(none, n)));
    }
    CAPTURE(scale);
    CHECK(scale > 1.0e-8);
    Approx approx = Approx::custom().epsilon(1.0e-10).scale(scale);
    for (size_t n = 0; n < 10; ++n) {
      CAPTURE(n);
      CHECK(std::isfinite(gsl::at(numeric, n)));
      if (exactly_degenerate and waves != bc::HllemWaves::AllWithFast) {
        // every restored internal wave sits in the five-fold cluster
        CHECK(gsl::at(numeric, n) == approx(gsl::at(none, n)));
      } else if (not exactly_degenerate) {
        CHECK(gsl::at(numeric, n) == approx(gsl::at(analytic, n)));
      }
    }
  }
  // the option value round-trips and is distinct
  CHECK(bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                  bc::HllemEigensystem::Numeric, 1.0e300} !=
        bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                  bc::HllemEigensystem::Analytic, 1.0e300});
  CHECK(bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                  bc::HllemEigensystem::Numeric, 1.0e4} !=
        bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                  bc::HllemEigensystem::Numeric, 1.0e300});
}

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hllem",
                  "[Unit][GrMhd]") {
  PUPable_reg(bc::Hllem);
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::IdealFluid<true>{4.0 / 3.0}.promote_to_3d_eos()};

  for (const auto waves : {bc::HllemWaves::Contact, bc::HllemWaves::ContactSlow,
                           bc::HllemWaves::ContactAlfven, bc::HllemWaves::All,
                           bc::HllemWaves::None, bc::HllemWaves::Slow,
                           bc::HllemWaves::Alfven}) {
    TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
        make_not_null(&gen), bc::Hllem{waves, true, 1.0e-10, 1.0e-30, 1.0e-8},
        Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
        volume_data, ranges);
  }

  const auto hllem =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::Hllem>(
          "Hllem:\n"
          "  WavesToRestore: ContactSlow\n"
      "  UseComplementaryProjection: true\n"
          "  DegeneracyTolerance: 1.0e-4\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n"
          "  Eigensystem: Analytic\n"
          "  MaxProjectorNorm: 1.0e300\n");
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), dynamic_cast<const bc::Hllem&>(*hllem),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // `None` is reachable from an input file, and it really is a distinct
  // option value (not silently aliased onto another wave set).
  const auto hllem_none =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::Hllem>(
          "Hllem:\n"
          "  WavesToRestore: None\n"
          "  UseComplementaryProjection: false\n"
          "  DegeneracyTolerance: 1.0e-4\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n"
          "  Eigensystem: Analytic\n"
          "  MaxProjectorNorm: 1.0e300\n");
  CHECK_FALSE(dynamic_cast<const bc::Hllem&>(*hllem_none) !=
              bc::Hllem{bc::HllemWaves::None, false, 1.0e-4, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::None, false, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::Contact, false, 1.0e-10, 1.0e-30, 1.0e-8});

  // `Slow` and `Alfven` are reachable from an input file too, and are distinct
  // option values -- from each other and from every set that contains them.
  const auto hllem_slow =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::Hllem>(
          "Hllem:\n"
          "  WavesToRestore: Slow\n"
          "  UseComplementaryProjection: false\n"
          "  DegeneracyTolerance: 1.0e-4\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n"
          "  Eigensystem: Analytic\n"
          "  MaxProjectorNorm: 1.0e300\n");
  CHECK_FALSE(dynamic_cast<const bc::Hllem&>(*hllem_slow) !=
              bc::Hllem{bc::HllemWaves::Slow, false, 1.0e-4, 1.0e-30, 1.0e-8});
  const auto hllem_alfven =
      TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                         bc::Hllem>(
          "Hllem:\n"
          "  WavesToRestore: Alfven\n"
          "  UseComplementaryProjection: false\n"
          "  DegeneracyTolerance: 1.0e-4\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n"
          "  Eigensystem: Analytic\n"
          "  MaxProjectorNorm: 1.0e300\n");
  CHECK_FALSE(
      dynamic_cast<const bc::Hllem&>(*hllem_alfven) !=
      bc::Hllem{bc::HllemWaves::Alfven, false, 1.0e-4, 1.0e-30, 1.0e-8});
  for (const auto other :
       {bc::HllemWaves::None, bc::HllemWaves::Contact,
        bc::HllemWaves::ContactSlow, bc::HllemWaves::ContactAlfven,
        bc::HllemWaves::All, bc::HllemWaves::Alfven}) {
    CHECK(bc::Hllem{bc::HllemWaves::Slow, false, 1.0e-10, 1.0e-30, 1.0e-8} !=
          bc::Hllem{other, false, 1.0e-10, 1.0e-30, 1.0e-8});
  }
  for (const auto other :
       {bc::HllemWaves::None, bc::HllemWaves::Contact,
        bc::HllemWaves::ContactSlow, bc::HllemWaves::ContactAlfven,
        bc::HllemWaves::All, bc::HllemWaves::Slow}) {
    CHECK(bc::Hllem{bc::HllemWaves::Alfven, false, 1.0e-10, 1.0e-30, 1.0e-8} !=
          bc::Hllem{other, false, 1.0e-10, 1.0e-30, 1.0e-8});
  }

  // What `Slow` and `Alfven` actually restore, asserted by the additivity of
  // the per-wave anti-diffusion. Two interfaces: a generic one where both
  // pairs are live, and one whose field is purely tangential to the face
  // (B_n = 0), where Type I degeneracy collapses the Alfven AND the slow pair
  // onto the entropy speed and the guard drops all four.
  {
    const double pair_adiabatic_index = 5.0 / 3.0;
    const auto pair_eos =
        EquationsOfState::IdealFluid<true>{pair_adiabatic_index}
            .promote_to_3d_eos();
    test_single_wave_pairs(
        make_state(1.0, 1.0, {{0.2, -0.3, 0.1}}, {{0.5, 0.3, 0.2}},
                   pair_adiabatic_index, 1),
        make_state(4.0, 0.2, {{-0.1, 0.25, -0.4}}, {{0.3, -0.2, 0.4}},
                   pair_adiabatic_index, 1),
        *pair_eos, {{1.0, 0.0, 0.0}}, "generic interface, B_n != 0", false);
    test_single_wave_pairs(
        make_state(1.0, 1.0, {{0.0, 0.2, -0.3}}, {{0.0, 0.4, 0.2}},
                   pair_adiabatic_index, 1),
        make_state(4.0, 0.2, {{0.0, -0.1, 0.25}}, {{0.0, 0.3, -0.2}},
                   pair_adiabatic_index, 1),
        *pair_eos, {{1.0, 0.0, 0.0}}, "purely tangential field, B_n = 0",
        true);
    test_numeric_eigensystem(
        make_state(1.0, 1.0, {{0.2, -0.3, 0.1}}, {{0.5, 0.3, 0.2}},
                   pair_adiabatic_index, 1),
        make_state(4.0, 0.2, {{-0.1, 0.25, -0.4}}, {{0.3, -0.2, 0.4}},
                   pair_adiabatic_index, 1),
        *pair_eos, {{1.0, 0.0, 0.0}}, "numeric: generic interface", false);
    test_numeric_eigensystem(
        make_state(1.0, 1.0, {{0.0, 0.2, -0.3}}, {{0.0, 0.4, 0.2}},
                   pair_adiabatic_index, 1),
        make_state(4.0, 0.2, {{0.0, -0.1, 0.25}}, {{0.0, 0.3, -0.2}},
                   pair_adiabatic_index, 1),
        *pair_eos, {{1.0, 0.0, 0.0}}, "numeric: B_n = 0", true);
    const auto numeric_from_yaml =
        TestHelpers::test_factory_creation<evolution::BoundaryCorrection,
                                           bc::Hllem>(
            "Hllem:\n"
            "  WavesToRestore: All\n"
            "  UseComplementaryProjection: false\n"
            "  DegeneracyTolerance: 1.0e-10\n"
            "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
            "  LightSpeedDensityCutoff: 1.0e-8\n"
            "  Eigensystem: Numeric\n"
            "  MaxProjectorNorm: 1.0e4\n");
    CHECK_FALSE(dynamic_cast<const bc::Hllem&>(*numeric_from_yaml) !=
                bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                          bc::HllemEigensystem::Numeric, 1.0e4});
  }

  CHECK_FALSE(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
              bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::ContactSlow, true, 1.0e-10, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::All, true, 1.0e-9, 1.0e-30, 1.0e-8});
  CHECK(bc::Hllem{bc::HllemWaves::All, true, 1.0e-10, 1.0e-30, 1.0e-8} !=
        bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8});
  // per-wave (non-CPM) path is also conservative
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      bc::Hllem{bc::HllemWaves::All, false, 1.0e-3, 1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);
  // DEBUG: the numeric eigensystem and the conditioning guard are too
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      bc::Hllem{bc::HllemWaves::All, false, 1.0e-10, 1.0e-30, 1.0e-8,
                bc::HllemEigensystem::Numeric, 1.0e4},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // HLLEM must preserve a stationary contact exactly at ANY jump strength, not
  // just in the linear limit -- that is the property that puts it on the same
  // footing as HLLC/HLLD on contact-dominated problems. It only holds because
  // the eigensystem is built at a thermodynamically consistent average state;
  // with independently averaged (rho, eps, p) the 10:1 case leaves ~18% of the
  // HLL diffusion in place.
  for (const double density_ratio :
       {1.0 + 1.0e-6, 1.0 + 1.0e-3, 1.1, 2.0, 10.0}) {
    test_stationary_contact_is_exact(density_ratio, true);
  }

  // `None` restores nothing, so it must BE the HLL flux -- the consistency
  // check on the anti-diffusion. Checked on a stationary contact (where the
  // restored contact wave changes the flux by 100% of the HLL diffusion, so
  // the discrimination is unmistakable) and on a moving, asymmetric, fully
  // three-dimensional interface.
  {
    const double reduction_adiabatic_index = 5.0 / 3.0;
    const auto reduction_eos =
        EquationsOfState::IdealFluid<true>{reduction_adiabatic_index}
            .promote_to_3d_eos();
    const std::array<double, 3> uniform_magnetic_field{{0.5, 0.3, 0.2}};
    test_no_restored_waves_reduces_to_hll(
        make_state_at_rest(1.0, 1.0, uniform_magnetic_field,
                           reduction_adiabatic_index, 1),
        make_state_at_rest(10.0, 1.0, uniform_magnetic_field,
                           reduction_adiabatic_index, 1),
        reduction_adiabatic_index, *reduction_eos);
    test_no_restored_waves_reduces_to_hll(
        make_state(1.0, 1.0, {{0.3, 0.1, -0.05}}, {{0.5, 0.3, 0.2}},
                   reduction_adiabatic_index, 1),
        make_state(0.125, 0.1, {{-0.2, 0.05, 0.1}}, {{0.5, -0.4, 0.15}},
                   reduction_adiabatic_index, 1),
        reduction_adiabatic_index, *reduction_eos);
  }

  test_fast_speeds_are_packaged_per_side();
  report_when_denominator_floors_bind();
  test_vanishing_tangential_field_stays_finite();
  test_strong_blast_states_stay_finite();
}
}  // namespace

// DEBUG (hllem_degeneracy_debug, umbrella A2): the face where the Del Zanna jet
// at DegeneracyTolerance 1e-10 goes wrong (the nozzle rim, r = 1, first cell
// row; probe records of experiments/delzanna_jet/hllem_face_probe, r11 at
// t = 0.908605 and r22 at t = 0.444207). The analytic eigenvectors are
// evaluated on the recorded averaged state twice: at SpECTRE's own double
// speeds (what the run did) and at the 60-digit oracle speeds rounded to
// double. The restored anti-diffusion direction sum_k delta_k (l.dU/l.r) r
// over the five kept internal waves is printed relative to |dU|; the oracle's
// own value (exact eigensystem) is 1.569 (r11) and 0.7332 (r22). This
// separates the speed error from round-off in the eigenvector formulas.
namespace {
struct OnsetFace {
  std::string name;
  double rho, eps, w, h;
  std::array<double, 3> v, b, n;
  std::array<double, 9> du, speeds_double, speeds_oracle;
  std::array<double, 5> delta;
  double oracle_value;
};

double onset_face_antidiffusion(
    const OnsetFace& face, const std::array<double, 9>& speeds,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) {
  const size_t num_points = 1;
  tnsr::I<DataVector, 3, Frame::Inertial> v{num_points};
  tnsr::I<DataVector, 3, Frame::Inertial> b{num_points};
  tnsr::i<DataVector, 3> n{num_points};
  tnsr::ii<DataVector, 3, Frame::Inertial> metric{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    v.get(i) = gsl::at(face.v, i);
    b.get(i) = gsl::at(face.b, i);
    n.get(i) = gsl::at(face.n, i);
    metric.get(i, i) = 1.0;
  }
  tnsr::i<DataVector, 9> lam{num_points};
  for (size_t k = 0; k < 9; ++k) {
    lam.get(k) = gsl::at(speeds, k);
  }
  tnsr::ij<DataVector, 9> modes{num_points, 0.0};
  tnsr::IJ<DataVector, 9> projectors{num_points, 0.0};
  grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd(
      make_not_null(&modes), make_not_null(&projectors), lam, v, b,
      Scalar<DataVector>{num_points, face.rho},
      Scalar<DataVector>{num_points, face.eps},
      Scalar<DataVector>{num_points, face.w},
      Scalar<DataVector>{num_points, face.h}, metric, n, equation_of_state,
      false);
  std::array<double, 9> sum{};
  double du_norm = 0.0;
  for (size_t m = 0; m < 9; ++m) {
    du_norm += square(gsl::at(face.du, m));
  }
  for (size_t k = 2; k <= 6; ++k) {
    double lr = 0.0;
    double ldu = 0.0;
    for (size_t m = 0; m < 9; ++m) {
      lr += projectors.get(k, m)[0] * modes.get(k, m)[0];
      ldu += projectors.get(k, m)[0] * gsl::at(face.du, m);
    }
    for (size_t m = 0; m < 9; ++m) {
      gsl::at(sum, m) +=
          gsl::at(face.delta, k - 2) * ldu / lr * modes.get(k, m)[0];
    }
  }
  double sum_norm = 0.0;
  for (size_t m = 0; m < 9; ++m) {
    sum_norm += square(gsl::at(sum, m));
  }
  return std::sqrt(sum_norm / du_norm);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.HllemOnsetFaceDebug",
    "[Unit][GrMhd]") {
  const auto eos =
      EquationsOfState::IdealFluid<true>{1.6666666666666667}
          .promote_to_3d_eos();
  const std::array<OnsetFace, 5> faces{{
      {"r11 t=0.908605",
       0.0739495689947152,
       4.1023613607110926e-11,
       1.601853001252123,
       1.0000000000683726,
       {{-0.05116278175661949, 0.779525872675541, 0.0}},
       {{0.00278603535805616, 0.08398031806776235, 0.0}},
       {{-1.0, -0.0, -0.0}},
       {{-0.18497596655023824, 6.038612320205473, 0.0,
         -0.004523156336791348, -0.13773474610353464, 0.0,
         0.5838771806964465, 5.435743842823654, 0.003367510859097023}},
       {{-1.0, -0.13928721262125857, 0.048059416529945165,
         0.051162701484856764, 0.05116278175661949, 0.051162862029309815,
         0.05611554607193812, 0.2379628798632529, 1.0}},
       {{-1.0, -0.13928721262125215, 0.04805941652994516,
         0.0511626942564261, 0.05116278175661949, 0.05116286925773412,
         0.05611554607193812, 0.23796287986325287, 1.0}},
       {{0.898925812937611, 0.8923992667019697, 0.8923990978817103,
         0.8923989290595002, 0.881982895126304}},
       1.569},
      {"r22 t=0.444207",
       0.04081258816629474,
       6.998697404971646e-11,
       1.8818672361803628,
       1.000000000116645,
       {{-0.03628591428785676, 0.8463516215222865, 0.0}},
       {{0.0027237308471549228, 0.0847106112069555, 0.0}},
       {{-1.0, -0.0, -0.0}},
       {{-0.07907010954432901, 4.2053738446693165, 0.0,
         -0.0005836930236142153, -0.26424205686560154, 0.0,
         0.3840084708241124, 3.8000530658914156, 0.011406889864836216}},
       {{-1.0, -0.18246362986265566, 0.03363938890301627,
         0.036285834109478235, 0.03628591428785676, 0.036285994467432345,
         0.04150311145989474, 0.25161262840343335, 1.0}},
       {{-1.0, -0.18246362986265557, 0.03363938890301627,
         0.036285834104437885, 0.03628591428785676, 0.036285994472472674,
         0.04150311145989474, 0.2516126284034333, 1.0}},
       {{0.9487841411417267, 0.9447549370274345, 0.9447548149561171,
         0.9447546928829773, 0.9368116494375865}},
       0.7332},
      {"r11 warm t=0.046151 r=0.7273 z=0.1705",
       0.09999824442625359,
       0.1499870287246698,
       7.088933473682282,
       1.2499783812077832,
       {{1.4886464166628985e-07, 0.9900003442994056, 0.0}},
       {{6.53467670814331e-05, 0.10002948841721129, 0.0}},
       {{-1.0, -0.0, -0.0}},
       {{8.562793787599554e-06, -1.118493919705088e-08, 0.0, -8.645580893004155e-05, -1.765717530027855e-05, 0.0, 0.0, -1.7774237299761353e-06, 8.620909356009006e-05}},
       {{-1.0, -0.06809927881126582, -2.9366077412369604e-06, -2.433463195864813e-06, -1.4886464166628985e-07, 3.4895440632046945e-06, 4.696045518453119e-06, 0.06809763493068052, 1.0}},
       {{-1.0, -0.06809927880703849, -2.936607741236954e-06, -2.529190954153832e-06, -1.4886464166628985e-07, 3.5852718276624e-06, 4.696045518453109e-06, 0.06809763492644702, 1.0}},
       {{0.9999579437668894, 0.9999651494838776, 0.9999978680550405, 0.9999500247516144, 0.999932745929851}},
       2.674},
      {"r11 warm t=0.046151 r=1.0909 z=0.5114",
       10.005414371093302,
       0.0015075546033608148,
       1.0000035344096445,
       1.0025125910056012,
       {{0.0016969365807974707, -0.0020467506086178184, 0.0}},
       {{-4.285304317982756e-05, 0.0962214144871043, 0.0}},
       {{-1.0, -0.0, -0.0}},
       {{0.0319509350730833, -0.039638485840919425, 0.0, 0.00040382735156674186, -0.007443935011635511, 0.0, 0.0001349631944957963, -0.0006362114913396888, -0.002732978419911954}},
       {{-1.0, -0.052599100500320704, -0.0017104600711831284, -0.0017077773327704858, -0.0016969365807974707, -0.001686094745160557, -0.0016834114085899562, 0.049214021430991864, 1.0}},
       {{-1.0, -0.05259910049678032, -0.0017104600711831284, -0.0017077955488618542, -0.0016969365807974707, -0.0016860765283845723, -0.0016834114085899562, 0.04921402142676686, 1.0}},
       {{0.9680079284145733, 0.9680581057678996, 0.9682608688250464, 0.968463652150806, 0.968513840691948}},
       5.495},
      {"r11 warm t=0.074996 r=1.0909 z=1.1932",
       9.99961253058517,
       0.0014999727004265408,
       1.00000000000318,
       1.002499954500711,
       {{2.4853492458673613e-06, 4.2756008938835445e-07, 0.0}},
       {{3.0567170218039805e-05, 0.10054344328856227, 0.0}},
       {{-1.0, -0.0, -0.0}},
       {{-1.670140058830047e-05, -3.16439214576129e-06, 0.0, -8.56690138745834e-05, 0.0010283448978338633, 0.0, -4.271605291705782e-11, 0.00010339067528796972, 0.0004982695957971671}},
       {{-1.0, -0.05165716054652363, -1.2134800090065717e-05, -1.009017457654435e-05, -2.4853492458673613e-06, 5.11947624780926e-06, 7.164101860691043e-06, 0.05165220311071793, 1.0}},
       {{-1.0, -0.05165716054476314, -1.2134800090065719e-05, -1.0102124663355852e-05, -2.4853492458673613e-06, 5.131426335089906e-06, 7.164101860691044e-06, 0.05165220310895697, 1.0}},
       {{0.9997655380961462, 0.9998050432208297, 0.999951979454824, 0.9999010780056022, 0.9998615703619229}},
       33.56},
  }};
  for (const auto& face : faces) {
    const double with_double =
        onset_face_antidiffusion(face, face.speeds_double, *eos);
    const double with_oracle =
        onset_face_antidiffusion(face, face.speeds_oracle, *eos);
    Parallel::printf(
        "HLLEM onset face %s: |sum delta P dU|/|dU| analytic eigenvectors at "
        "double speeds %.6e, at oracle speeds %.6e, exact (oracle) %.4e\n",
        face.name, with_double, with_oracle, face.oracle_value);
    CHECK(std::isfinite(with_double));
    CHECK(std::isfinite(with_oracle));
  }
}
