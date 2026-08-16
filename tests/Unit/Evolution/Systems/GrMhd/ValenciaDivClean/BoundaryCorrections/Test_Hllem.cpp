// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Parallel/Printf/Printf.hpp"

#include <array>
#include <cstddef>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
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
  Scalar<DataVector> largest_outgoing{}, largest_ingoing{}, metric_flatness{};
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
              interior.interface_unit_normal, interior.metric_flatness,
              interior.rest_mass_density, interior.spatial_velocity,
              interior.pressure, interior.lorentz_factor,
              interior.specific_internal_energy, exterior.tilde_d,
              exterior.tilde_ye, exterior.tilde_tau, exterior.tilde_s,
              exterior.tilde_b, exterior.tilde_phi, exterior.nf_tilde_d,
              exterior.nf_tilde_ye, exterior.nf_tilde_tau, exterior.nf_tilde_s,
              exterior.nf_tilde_b, exterior.nf_tilde_phi,
              exterior.largest_outgoing, exterior.largest_ingoing,
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
        interior.interface_unit_normal, interior.metric_flatness,
        interior.rest_mass_density, interior.spatial_velocity,
        interior.pressure, interior.lorentz_factor,
        interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
        exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
        exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
        exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
        exterior.nf_tilde_phi, exterior.largest_outgoing,
        exterior.largest_ingoing, exterior.interface_unit_normal,
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
          interior.interface_unit_normal, interior.metric_flatness,
          interior.rest_mass_density, interior.spatial_velocity,
          interior.pressure, interior.lorentz_factor,
          interior.specific_internal_energy, exterior.tilde_d,
          exterior.tilde_ye, exterior.tilde_tau, exterior.tilde_s,
          exterior.tilde_b, exterior.tilde_phi, exterior.nf_tilde_d,
          exterior.nf_tilde_ye, exterior.nf_tilde_tau, exterior.nf_tilde_s,
          exterior.nf_tilde_b, exterior.nf_tilde_phi,
          exterior.largest_outgoing, exterior.largest_ingoing,
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
        interior.largest_ingoing, interior.interface_unit_normal,
        interior.metric_flatness, interior.rest_mass_density,
        interior.spatial_velocity, interior.pressure, interior.lorentz_factor,
        interior.specific_internal_energy, exterior.tilde_d, exterior.tilde_ye,
        exterior.tilde_tau, exterior.tilde_s, exterior.tilde_b,
        exterior.tilde_phi, exterior.nf_tilde_d, exterior.nf_tilde_ye,
        exterior.nf_tilde_tau, exterior.nf_tilde_s, exterior.nf_tilde_b,
        exterior.nf_tilde_phi, exterior.largest_outgoing,
        exterior.largest_ingoing, exterior.interface_unit_normal,
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

  for (const auto waves :
       {bc::HllemWaves::Contact, bc::HllemWaves::ContactSlow,
        bc::HllemWaves::ContactAlfven, bc::HllemWaves::All}) {
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
          "  DegeneracyTolerance: 1.0e-10\n"
          "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
          "  LightSpeedDensityCutoff: 1.0e-8\n");
  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen), dynamic_cast<const bc::Hllem&>(*hllem),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

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

  report_when_denominator_floors_bind();
  test_vanishing_tangential_field_stays_finite();
  test_strong_blast_states_stay_finite();
}
}  // namespace
