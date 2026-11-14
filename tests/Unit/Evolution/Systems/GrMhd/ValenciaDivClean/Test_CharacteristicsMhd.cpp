// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace {
namespace MhdSpeed = grmhd::ValenciaDivClean::MhdSpeed;

/**
 * Tests that we can find the roots of a test quartic polynomial:
 *   F(x) = (x - 0.9)(x - 0.3)(x + 0.2)(x + 0.8)
 *        = x^4 - 0.2 x^3 - 0.77 x^2 + 0.078 x + 0.0432
 */
void test_quartic_rootfinding(const DataVector& used_for_size) {
  const size_t num_points = used_for_size.size();

  tnsr::i<DataVector, 4> quartic_coefficients{num_points};
  get<0>(quartic_coefficients) = 0.0432;
  get<1>(quartic_coefficients) = 0.078;
  get<2>(quartic_coefficients) = -0.77;
  get<3>(quartic_coefficients) = -0.2;

  DataVector fast_plus(num_points, 1.0);
  grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
      make_not_null(&fast_plus), quartic_coefficients);

  DataVector fast_minus(num_points, -1.0);
  grmhd::ValenciaDivClean::find_magnetosonic_speed_from_quartic(
      make_not_null(&fast_minus), quartic_coefficients);

  const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(fast_plus, DataVector(num_points, 0.9),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(fast_minus, DataVector(num_points, -0.8),
                               custom_approx);
}

// Check that each speed is a zero of its factor of the characteristic
// determinant, evaluated from its definition (see the documentation of
// `characteristic_speeds_mhd`), and that the speeds are in increasing order.
void check_speeds_solve_characteristic_determinant(
    const tnsr::i<DataVector, 7>& characteristic_speeds,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const tnsr::I<DataVector, 3>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal) {
  const double tolerance = 1.0e-10;
  const size_t num_points = get(rest_mass_density).size();

  // Build the quantities that enter the determinant
  const DataVector normal_velocity =
      get(dot_product(spatial_velocity, unit_normal));
  const DataVector normal_magnetic_field =
      get(dot_product(magnetic_field, unit_normal));
  const auto magnetic_field_one_form =
      raise_or_lower_index(magnetic_field, spatial_metric);
  const DataVector magnetic_field_dot_spatial_velocity =
      get(dot_product(magnetic_field_one_form, spatial_velocity));
  const DataVector magnetic_field_squared =
      get(dot_product(magnetic_field_one_form, magnetic_field));
  const DataVector& lorentz = get(lorentz_factor);
  const DataVector comoving_magnetic_field_squared =
      magnetic_field_squared / square(lorentz) +
      square(magnetic_field_dot_spatial_velocity);
  const DataVector rho_h = get(rest_mass_density) * get(specific_enthalpy);
  const DataVector rho_h_star = rho_h + comoving_magnetic_field_squared;
  const DataVector& cs2 = get(sound_speed_squared);

  // The factors a, B (calligraphic) and G of the determinant at a speed
  const auto factors = [&](const DataVector& speed) {
    return std::array<DataVector, 3>{
        {lorentz * (normal_velocity - speed),
         normal_magnetic_field / lorentz -
             lorentz * magnetic_field_dot_spatial_velocity *
                 (speed - normal_velocity),
         1.0 - square(speed)}};
  };
  // Check that |residual| <= tolerance * scale at every point
  const auto check_relative_zero = [&tolerance, &num_points](
                                       const DataVector& residual,
                                       const DataVector& scale,
                                       const std::string& what) {
    CAPTURE(what);
    for (size_t point = 0; point < num_points; ++point) {
      CAPTURE(residual[point]);
      CAPTURE(scale[point]);
      CHECK(std::abs(residual[point]) <= tolerance * scale[point]);
    }
  };

  // Check the entropy speed
  CHECK(get<MhdSpeed::Entropy>(characteristic_speeds) == normal_velocity);

  // Check that the Alfven speeds solve B^2 = rho h^* a^2
  for (const auto speed_index : {MhdSpeed::AlfvenMinus, MhdSpeed::AlfvenPlus}) {
    const auto [a, cal_b, g] = factors(characteristic_speeds.get(speed_index));
    check_relative_zero(square(cal_b) - rho_h_star * square(a),
                        square(cal_b) + rho_h_star * square(a), "Alfven");
  }

  // Check that the magnetosonic speeds solve N = 0
  for (const auto speed_index :
       {MhdSpeed::FastMagnetosonicMinus, MhdSpeed::SlowMagnetosonicMinus,
        MhdSpeed::SlowMagnetosonicPlus, MhdSpeed::FastMagnetosonicPlus}) {
    const auto [a, cal_b, g] = factors(characteristic_speeds.get(speed_index));
    const DataVector a2 = square(a);
    const DataVector n_polynomial =
        a2 * g * comoving_magnetic_field_squared - rho_h * square(a2) +
        cs2 * (rho_h * a2 * (a2 + g) - square(cal_b) * g);
    const DataVector scale =
        abs(a2 * g * comoving_magnetic_field_squared) + rho_h * square(a2) +
        cs2 * (rho_h * a2 * (a2 + abs(g)) + square(cal_b) * abs(g));
    check_relative_zero(n_polynomial, scale, "magnetosonic");
  }

  // Check that the speeds are in increasing order
  for (size_t i = 0; i + 1 < 7; ++i) {
    CAPTURE(i);
    CHECK(max(characteristic_speeds.get(i) -
              characteristic_speeds.get(i + 1)) <= tolerance);
  }
}

// Check the speeds on random states, for each kind of equation of state.
template <size_t ThermodynamicDim>
void test_speeds_solve_characteristic_determinant(
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Build a random state
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  Scalar<DataVector> specific_internal_energy{};
  Scalar<DataVector> pressure{};
  Scalar<DataVector> sound_speed_squared{};
  if constexpr (ThermodynamicDim == 1) {
    specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density(
            rest_mass_density);
    pressure = equation_of_state.pressure_from_density(rest_mass_density);
    sound_speed_squared = Scalar<DataVector>{
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density))};
  } else {
    specific_internal_energy =
        helper::random_specific_internal_energy(nn_gen, used_for_size);
    pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
    sound_speed_squared = Scalar<DataVector>{
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy))};
  }
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  get(sound_speed_squared) /= get(specific_enthalpy);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto magnetic_field =
      helper::random_magnetic_field(nn_gen, pressure, spatial_metric);

  for (const auto& direction : Direction<3>::all_directions()) {
    CAPTURE(direction);
    const auto unit_normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    // Compute the speeds
    tnsr::i<DataVector, 7> characteristic_speeds{used_for_size.size()};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&characteristic_speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, unit_normal,
        equation_of_state);

    check_speeds_solve_characteristic_determinant(
        characteristic_speeds, spatial_velocity, magnetic_field,
        rest_mass_density, lorentz_factor, specific_enthalpy,
        sound_speed_squared, spatial_metric, unit_normal);
  }
}

// Check that, without a magnetic field, the fast speeds are the hydro sound
// speeds and the Alfven and slow speeds are the normal velocity.
void test_hydro_limit(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Build a random state without a magnetic field
  const EquationsOfState::IdealFluid<true> equation_of_state(4.0 / 3.0);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy,
      equation_of_state.pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy));
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const tnsr::I<DataVector, 3> magnetic_field{used_for_size.size(), 0.0};

  for (const auto& direction : Direction<3>::all_directions()) {
    CAPTURE(direction);
    const auto unit_normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    // Compute the MHD and the hydro speeds
    tnsr::i<DataVector, 7> mhd_speeds{used_for_size.size()};
    grmhd::ValenciaDivClean::characteristic_speeds_mhd(
        make_not_null(&mhd_speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, unit_normal,
        equation_of_state);
    const auto hydro_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            electron_fraction, lorentz_factor, specific_enthalpy,
            spatial_metric, unit_normal, equation_of_state);

    // Check that they agree
    const auto& normal_velocity =
        get<grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity>(
            hydro_speeds);
    const Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<MhdSpeed::FastMagnetosonicMinus>(mhd_speeds),
        get<grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus>(hydro_speeds),
        custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        get<MhdSpeed::FastMagnetosonicPlus>(mhd_speeds),
        get<grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus>(hydro_speeds),
        custom_approx);
    for (const auto speed_index :
         {MhdSpeed::AlfvenMinus, MhdSpeed::SlowMagnetosonicMinus,
          MhdSpeed::Entropy, MhdSpeed::SlowMagnetosonicPlus,
          MhdSpeed::AlfvenPlus}) {
      CAPTURE(speed_index);
      CHECK_ITERABLE_CUSTOM_APPROX(mhd_speeds.get(speed_index), normal_velocity,
                                   custom_approx);
    }
  }
}

// Check that an equilibrium 3D equation of state gives the speeds of the 2D
// equation of state it is built from.
void test_three_dimensional_equation_of_state(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Build a random state
  const EquationsOfState::IdealFluid<true> equation_of_state_2d(5.0 / 3.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto pressure = equation_of_state_2d.pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto magnetic_field =
      helper::random_magnetic_field(nn_gen, pressure, spatial_metric);
  const auto unit_normal = unit_basis_form(
      Direction<3>::upper_xi(), determinant_and_inverse(spatial_metric).second);

  // Compute the speeds with both equations of state
  tnsr::i<DataVector, 7> speeds_2d{used_for_size.size()};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds_2d), spatial_velocity, magnetic_field,
      rest_mass_density, specific_internal_energy, electron_fraction,
      lorentz_factor, specific_enthalpy, spatial_metric, unit_normal,
      equation_of_state_2d);
  tnsr::i<DataVector, 7> speeds_3d{used_for_size.size()};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds_3d), spatial_velocity, magnetic_field,
      rest_mass_density, specific_internal_energy, electron_fraction,
      lorentz_factor, specific_enthalpy, spatial_metric, unit_normal,
      *equation_of_state_3d);

  // Check that they agree
  const Approx custom_approx = Approx::custom().epsilon(1.0e-10).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(speeds_3d, speeds_2d, custom_approx);
}

// The characteristic speeds at one point in flat space, with an ideal fluid.
tnsr::i<DataVector, 7> speeds_in_flat_space(
    const double rest_mass_density, const double specific_internal_energy,
    const double specific_enthalpy, const double lorentz_factor,
    const std::array<double, 3>& velocity, const std::array<double, 3>& b_field,
    const std::array<double, 3>& normal,
    const double adiabatic_index = 5.0 / 3.0) {
  const EquationsOfState::IdealFluid<true> eos(adiabatic_index, 0.0);
  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{1_st, 0.0};
  tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{1_st, 0.0};
  tnsr::i<DataVector, 3> unit_normal{1_st, 0.0};
  tnsr::ii<DataVector, 3, Frame::Inertial> flat_metric{1_st, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity.get(i) = gsl::at(velocity, i);
    magnetic_field.get(i) = gsl::at(b_field, i);
    unit_normal.get(i) = gsl::at(normal, i);
    flat_metric.get(i, i) = 1.0;
  }
  tnsr::i<DataVector, 7> speeds{1_st, 0.0};
  grmhd::ValenciaDivClean::characteristic_speeds_mhd(
      make_not_null(&speeds), spatial_velocity, magnetic_field,
      Scalar<DataVector>{DataVector{1_st, rest_mass_density}},
      Scalar<DataVector>{DataVector{1_st, specific_internal_energy}},
      Scalar<DataVector>{DataVector{1_st, 0.1}},
      Scalar<DataVector>{DataVector{1_st, lorentz_factor}},
      Scalar<DataVector>{DataVector{1_st, specific_enthalpy}}, flat_metric,
      unit_normal, eos);
  return speeds;
}

// The characteristic speeds must INTERLACE. For any admissible state the seven
// physical RMHD speeds satisfy
//
//   lambda_f^- <= lambda_a^- <= lambda_s^- <= lambda_e
//              <= lambda_s^+ <= lambda_a^+ <= lambda_f^+
//
// -- Anton, Miralles, Marti, Ibanez, Aloy & Mimica (2010), Eq. (42), which is
// also the order of the `MhdSpeed` labels in Characteristics.hpp. NOTE
// THE ORDER: the Alfven speed sits OUTSIDE the slow speed and INSIDE the fast
// one.
//
// The quartic has four roots and |Q| is ~0 at every one of them, so a residual
// check cannot tell whether the slow slots hold the slow roots. Ordering is
// what tells the roots apart.
//
// The sweep deliberately includes the degeneracies: B = 0, B_n = 0 (Type I,
// where alfven+- = slow+- = entropy), purely normal B (Type II), a nearly cold
// state and a nearly force-free one, at W up to 7, on two normals.
void test_speeds_interlace() {
  const std::array<std::array<double, 3>, 6> velocities{{{{0.0, 0.0, 0.0}},
                                                         {{0.4, 0.0, 0.0}},
                                                         {{-0.6, 0.3, 0.1}},
                                                         {{0.1, -0.2, 0.9}},
                                                         {{0.99, 0.0, 0.0}},
                                                         {{0.2, -0.5, -0.2}}}};
  const std::array<std::array<double, 3>, 7> magnetic_fields{
      {{{0.0, 0.0, 0.0}},
       {{1.0e-8, 0.0, 0.0}},
       {{0.0, 3.0, 0.0}},
       {{2.0, 0.0, 0.0}},
       {{-0.3, 0.1, 0.2}},
       {{5.0, -4.0, 3.0}},
       {{0.05, 0.02, -0.01}}}};
  const std::array<std::array<double, 2>, 4> rho_and_pressure{
      {{{1.0, 1.0}}, {{1.0, 1.0e-6}}, {{1.0e-3, 1.0}}, {{5.0, 0.5}}}};
  const double skew = 1.0 / sqrt(3.0);
  const std::array<std::array<double, 3>, 2> normals{
      {{{1.0, 0.0, 0.0}}, {{skew, skew, skew}}}};

  // How far out of order the chain may be, per state. For a monic quartic,
  // |Q'(root)| is the product of the distances to the other three roots, so a
  // cluster of nearly coincident speeds drives it down like the cube of the
  // cluster width, and the root cannot be LOCATED more finely than
  // noise/|Q'|. One state in this very sweep (cs^2 = 1.67e-4,
  // b^2/(rho h) = 2.4e-5, W = 3.3) puts all four magnetosonic roots inside
  // 2.5e-3, where that floor is 9.8e-6. A constant slack tight enough to be
  // useful on separated roots is a false alarm there.
  //
  // For a monic quartic whose roots all lie inside the light cone the
  // coefficients are bounded by 6 in magnitude, which bounds the Horner noise.
  const double quartic_noise =
      8.0 * std::numeric_limits<double>::epsilon() * 6.0;
  const auto root_uncertainty =
      [&quartic_noise](const double root, const double other_1,
                       const double other_2, const double other_3) {
        const double derivative =
            std::abs((root - other_1) * (root - other_2) * (root - other_3));
        return derivative > 0.0 ? quartic_noise / derivative
                                : std::numeric_limits<double>::infinity();
      };

  size_t states_checked = 0;
  for (const double adiabatic_index : {4.0 / 3.0, 5.0 / 3.0}) {
    for (const auto& normal : normals) {
      for (const auto& velocity : velocities) {
        double v_squared = 0.0;
        for (size_t i = 0; i < 3; ++i) {
          v_squared += square(gsl::at(velocity, i));
        }
        REQUIRE(v_squared < 1.0);
        const double lorentz = 1.0 / sqrt(1.0 - v_squared);
        for (const auto& b_field : magnetic_fields) {
          for (const auto& rho_p : rho_and_pressure) {
            const double rho = rho_p[0];
            const double pressure = rho_p[1];
            const double eps = pressure / ((adiabatic_index - 1.0) * rho);
            const auto speeds = speeds_in_flat_space(
                rho, eps, 1.0 + eps + pressure / rho, lorentz, velocity,
                b_field, normal, adiabatic_index);

            const double fast_minus =
                get<MhdSpeed::FastMagnetosonicMinus>(speeds)[0];
            const double alfven_minus = get<MhdSpeed::AlfvenMinus>(speeds)[0];
            const double slow_minus =
                get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0];
            const double entropy = get<MhdSpeed::Entropy>(speeds)[0];
            const double slow_plus =
                get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0];
            const double alfven_plus = get<MhdSpeed::AlfvenPlus>(speeds)[0];
            const double fast_plus =
                get<MhdSpeed::FastMagnetosonicPlus>(speeds)[0];
            CAPTURE(adiabatic_index);
            CAPTURE(rho);
            CAPTURE(pressure);
            CAPTURE(lorentz);
            CAPTURE(fast_minus);
            CAPTURE(alfven_minus);
            CAPTURE(slow_minus);
            CAPTURE(entropy);
            CAPTURE(slow_plus);
            CAPTURE(alfven_plus);
            CAPTURE(fast_plus);
            const double slack = std::max(
                1.0e-10, 4.0 * (root_uncertainty(slow_minus, fast_minus,
                                                 slow_plus, fast_plus) +
                                root_uncertainty(slow_plus, fast_minus,
                                                 slow_minus, fast_plus)));
            CAPTURE(slack);
            CHECK(fast_minus <= alfven_minus + slack);
            CHECK(alfven_minus <= slow_minus + slack);
            CHECK(slow_minus <= entropy + slack);
            CHECK(entropy <= slow_plus + slack);
            CHECK(slow_plus <= alfven_plus + slack);
            CHECK(alfven_plus <= fast_plus + slack);
            CHECK(std::abs(fast_minus) <= 1.0);
            CHECK(std::abs(fast_plus) <= 1.0);
            ++states_checked;
          }
        }
      }
    }
  }
  CHECK(states_checked == 2 * 2 * 6 * 7 * 4);
}

// Regression test for an abort of a Del Zanna et al. (2003) jet run, "reduced
// quadratic has negative discriminant below tolerance. discriminant =
// -7.8677713958712506894".
//
// The point of the test is that NEITHER of the two states below is bad. Each
// one is an admissible RMHD state whose magnetosonic quartic has four real
// roots strictly inside the light cone. What is not a state is an interface
// value that averages rho, eps and p between them INDEPENDENTLY, because
// `characteristic_speeds_mhd` then forms
// c_s^2 = (chi + kappa p/rho^2)/h with chi and kappa from (rho_avg, eps_avg)
// but h from p_avg. For an ideal fluid that ratio tends to Gamma(Gamma-1) as
// p_avg/rho_avg -> 0, which EXCEEDS 1 for every Gamma above the golden ratio
// 1.618 -- so Gamma = 5/3 is exposed and Gamma = 4/3 is not.
//
// The states here are the jet's own cocoon/ambient contact in miniature: a hot
// rarefied cell (rho = 0.01, eps = 45) against a cold dense one (rho = 5,
// eps = 0.15). Their independent average has c_s^2 = 1.0568, and the quartic
// then puts its extremal roots at -1.6163 and +1.2338, OUTSIDE the light cone,
// where the +/-1 Newton seeds no longer find the fast pair.
//
// So `characteristic_speeds_mhd` must not solve that quartic at all: the
// sound-speed check at its top reports the non-state instead of repairing it.
void test_superluminal_interface_sound_speed() {
  const double adiabatic_index = 5.0 / 3.0;

  const double rho_left = 0.01;
  const double pressure_left = 0.3;
  const double rho_right = 5.0;
  const double pressure_right = 0.5;
  const double eps_left = pressure_left / ((adiabatic_index - 1.0) * rho_left);
  const double eps_right =
      pressure_right / ((adiabatic_index - 1.0) * rho_right);
  CHECK(eps_left == approx(45.0));
  CHECK(eps_right == approx(0.15));

  // The velocity is continuous across the interface (as across a contact); the
  // magnetic field is not.
  const std::array<double, 3> velocity{{0.2, -0.9, -0.2}};
  const std::array<double, 3> b_field_left{{-0.3, 0.1, 0.2}};
  const std::array<double, 3> b_field_right{{0.1, 0.3, -0.2}};
  const double v_squared = 0.2 * 0.2 + 0.9 * 0.9 + 0.2 * 0.2;
  REQUIRE(v_squared < 1.0);
  const double lorentz = 1.0 / sqrt(1.0 - v_squared);
  const std::array<double, 3> normal{{1.0, 0.0, 0.0}};

  const auto check_physical = [](const tnsr::i<DataVector, 7>& speeds,
                                 const std::string& what) {
    CAPTURE(what);
    for (size_t i = 0; i < 7; ++i) {
      CAPTURE(i);
      CAPTURE(speeds.get(i)[0]);
      CHECK(std::isfinite(speeds.get(i)[0]));
      CHECK(std::abs(speeds.get(i)[0]) <= 1.0 + 1.0e-12);
    }
    for (size_t i = 0; i + 1 < 7; ++i) {
      CAPTURE(i);
      CHECK(speeds.get(i)[0] <= speeds.get(i + 1)[0] + 1.0e-12);
    }
  };

  // 1. Each side, on its own, is an admissible state: a consistent (rho, eps,
  //    p) triple, four real magnetosonic speeds inside the light cone.
  check_physical(speeds_in_flat_space(rho_left, eps_left,
                                      1.0 + eps_left + pressure_left / rho_left,
                                      lorentz, velocity, b_field_left, normal),
                 "left state");
  check_physical(
      speeds_in_flat_space(rho_right, eps_right,
                           1.0 + eps_right + pressure_right / rho_right,
                           lorentz, velocity, b_field_right, normal),
      "right state");

  // 2. Their independently averaged interface value is not a state: its
  //    sound speed is superluminal.
  const double rho_avg = 0.5 * (rho_left + rho_right);
  const double eps_avg = 0.5 * (eps_left + eps_right);
  const double p_avg = 0.5 * (pressure_left + pressure_right);
  std::array<double, 3> b_field_avg{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(b_field_avg, i) =
        0.5 * (gsl::at(b_field_left, i) + gsl::at(b_field_right, i));
  }
  const double enthalpy_avg = 1.0 + eps_avg + p_avg / rho_avg;
  const double naive_sound_speed_squared =
      adiabatic_index * (adiabatic_index - 1.0) * eps_avg / enthalpy_avg;
  CAPTURE(naive_sound_speed_squared);
  CHECK(naive_sound_speed_squared > 1.0);
  // ... while the state that IS consistent at (rho_avg, eps_avg) is nowhere
  // near the light cone. The gap is the defect, not the magnitude of c_s^2.
  const double consistent_sound_speed_squared =
      adiabatic_index * (adiabatic_index - 1.0) * eps_avg /
      (1.0 + adiabatic_index * eps_avg);
  CHECK(consistent_sound_speed_squared < 0.67);

  // 3. The interface call is refused, with the offending state in the message.
  CHECK_THROWS_WITH(
      speeds_in_flat_space(rho_avg, eps_avg, enthalpy_avg, lorentz, velocity,
                           b_field_avg, normal),
      Catch::Matchers::ContainsSubstring(
          "The sound speed squared is outside [0, 1)"));
}

// The slow pair near the slow/entropy degeneracy. Two radial faces of the
// static ambient of the Del Zanna et al. (2003) jet: v = 0, B_r ~ 3e-5 and
// 9e-9. The expected slow speeds are the roots of the same magnetosonic
// quartic, computed with mpmath at 60 digits; v_n = 0, so they are +-mu.
void test_slow_speeds_near_degeneracy() {
  struct Face {
    double rho, eps, h;
    std::array<double, 3> b;
    double exact_mu;
  };
  const std::array<Face, 2> faces{
      {{10.00000147497365,
        0.0014992494598037663,
        1.002498749099673,
        {{2.997194571166376e-05, 0.10007501873759235, 0.0}},
        7.480833985711296863266183e-06},
       {10.0,
        0.001499999483693053,
        1.0024999991394885,
        {{-8.90989296669351e-09, 0.1000000516306809, 0.0}},
        2.224693536551416198672102e-09}}};
  const std::array<double, 3> v{{0.0, 0.0, 0.0}};
  const std::array<double, 3> n{{-1.0, 0.0, 0.0}};
  for (const auto& face : faces) {
    CAPTURE(face.exact_mu);
    const auto speeds =
        speeds_in_flat_space(face.rho, face.eps, face.h, 1.0, v, face.b, n);
    const double slow_minus = get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0];
    const double slow_plus = get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0];
    CAPTURE(slow_minus);
    CAPTURE(slow_plus);
    // accurate relative to the distance from v_n = 0
    CHECK(std::abs(slow_minus + face.exact_mu) <= 1.0e-6 * face.exact_mu);
    CHECK(std::abs(slow_plus - face.exact_mu) <= 1.0e-6 * face.exact_mu);
  }
  // A Balsara-1 state (Gamma = 2, normal z, B_n ~ 2e-7) where dividing out the
  // fast roots gives slow_minus = slow_plus and an unguarded Newton step
  // leaves the light cone. Slow speeds computed with mpmath at 60 digits.
  {
    const auto speeds = speeds_in_flat_space(
        0.6953676820816324, 0.6997190337712661, 2.3994380675425324,
        1.0474884323238665,
        {{0.29653056001033645, -0.02617931983258507, 1.2840727654936489e-07}},
        {{0.500000427349988, 0.7161462537884182, 1.9845159331757503e-07}},
        {{0.0, 0.0, 1.0}}, 2.0);
    const double vn = 1.284072765493648850494078e-07;
    const double exact_minus = 1.322726028842970510357819e-08;
    const double exact_plus = 2.272776457004750699634155e-07;
    const double slow_minus = get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0];
    const double slow_plus = get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0];
    CAPTURE(slow_minus);
    CAPTURE(slow_plus);
    CHECK(std::abs(slow_minus - exact_minus) <=
          1.0e-6 * std::abs(exact_minus - vn));
    CHECK(std::abs(slow_plus - exact_plus) <=
          1.0e-6 * std::abs(exact_plus - vn));
  }
  // A subnormal B_n (3.5e-323 after scaling), on a Balsara-1 transverse face
  // at rest. The slow pair is v_n exactly.
  {
    const auto speeds =
        speeds_in_flat_space(1.0, 1.0, 3.0, 1.0, {{0.0, 0.0, 0.0}},
                             {{0.5, 1.0, 3.458459520888726e-323 * sqrt(3.0)}},
                             {{0.0, 0.0, 1.0}}, 2.0);
    CHECK(get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0] == 0.0);
    CHECK(get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0] == 0.0);
  }
  // B_n = 0: the slow pair is v_n exactly (at rest and moving)
  for (const auto& velocity : {std::array<double, 3>{{0.0, 0.0, 0.0}},
                               std::array<double, 3>{{0.3, 0.2, 0.1}}}) {
    double v_squared = 0.0;
    for (const double component : velocity) {
      v_squared += square(component);
    }
    const double lorentz = 1.0 / sqrt(1.0 - v_squared);
    const double rho = 1.3;
    const double eps = 0.4;
    const double h = 1.0 + eps + (5.0 / 3.0 - 1.0) * eps;
    const auto speeds = speeds_in_flat_space(
        rho, eps, h, lorentz, velocity, {{0.0, 0.7, -0.2}}, {{1.0, 0.0, 0.0}});
    CAPTURE(velocity[0]);
    CHECK(get<MhdSpeed::SlowMagnetosonicMinus>(speeds)[0] == velocity[0]);
    CHECK(get<MhdSpeed::SlowMagnetosonicPlus>(speeds)[0] == velocity[0]);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.CharacteristicsMhd",
                  "[Unit][Evolution]") {
  const DataVector used_for_size(5);
  test_quartic_rootfinding(used_for_size);
  test_speeds_solve_characteristic_determinant(
      EquationsOfState::PolytropicFluid<true>(0.001, 4.0 / 3.0), used_for_size);
  test_speeds_solve_characteristic_determinant(
      EquationsOfState::IdealFluid<true>(4.0 / 3.0), used_for_size);
  test_speeds_solve_characteristic_determinant(
      EquationsOfState::IdealFluid<true>(5.0 / 3.0), used_for_size);
  test_hydro_limit(used_for_size);
  test_three_dimensional_equation_of_state(used_for_size);
  test_speeds_interlace();
  test_superluminal_interface_sound_speed();
  test_slow_speeds_near_degeneracy();
}
