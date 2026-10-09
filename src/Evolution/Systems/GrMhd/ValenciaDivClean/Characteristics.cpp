// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <exception>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <limits>
#include <stdexcept>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/RootFinding/TOMS748.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
namespace detail {
// Diagnostic counter for how often the denominator floors in the characteristic
// decomposition actually BIND. A floor that never binds is inert; one that binds
// constantly says the eigensystem is being evaluated where it is not meaningful
// (atmosphere cells, or a genuine wave degeneracy), which is exactly when a
// solver that uses the eigenbasis AS its flux (Marquina) cannot be trusted.
size_t& denominator_floor_counter() {
  static size_t counter = 0;
  return counter;
}
}  // namespace detail

size_t denominator_floor_count() { return detail::denominator_floor_counter(); }
void reset_denominator_floor_count() { detail::denominator_floor_counter() = 0; }
}  // namespace grmhd::ValenciaDivClean

namespace {
void compute_characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> pchar_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  // Mapping of indices between GRMHD char speeds and relativistic Euler char
  // speeds arrays.
  //
  // GRMHD     Rel Euler
  //   1           0
  //   2           1
  //   3           2
  //   4           3
  //   5           1
  //   6           1
  //   7           4
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>>>
      temp_tensors{num_grid_points};

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  {
    Scalar<DataVector>& normal_shift = get<::Tags::TempScalar<0>>(temp_tensors);
    dot_product(make_not_null(&normal_shift), normal, shift);
    char_speeds[0] = -1.0 * get(normal_shift);
    char_speeds[1] = char_speeds[0];
  }
  Scalar<DataVector>& scaled_sound_speed_squared =
      get<::Tags::TempScalar<5>>(temp_tensors);
  get(scaled_sound_speed_squared) =
      get(sound_speed_squared) +
      get(alfven_speed_squared) * (1.0 - get(sound_speed_squared));
  // Dim-fold degenerate eigenvalue, reuse normal_shift allocation
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  char_speeds[2] = char_speeds[1] + get(lapse) * get(normal_velocity);
  char_speeds[3] = char_speeds[2];
  char_speeds[4] = char_speeds[3];
  char_speeds[5] = char_speeds[2];
  char_speeds[6] = char_speeds[2];

  Scalar<DataVector>& one_minus_v_sqrd_cs_sqrd =
      get<::Tags::TempScalar<1>>(temp_tensors);
  get(one_minus_v_sqrd_cs_sqrd) =
      1.0 - get(spatial_velocity_squared) * get(scaled_sound_speed_squared);
  Scalar<DataVector>& vn_times_one_minus_cs_sqrd =
      get<::Tags::TempScalar<2>>(temp_tensors);
  get(vn_times_one_minus_cs_sqrd) =
      get(normal_velocity) * (1.0 - get(scaled_sound_speed_squared));

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(first_term) = get(lapse) / get(one_minus_v_sqrd_cs_sqrd);
  Scalar<DataVector>& second_term = get<::Tags::TempScalar<4>>(temp_tensors);
  get(second_term) =
      get(first_term) * sqrt(get(scaled_sound_speed_squared)) *
      sqrt((1.0 - get(spatial_velocity_squared)) *
           (get(one_minus_v_sqrd_cs_sqrd) -
            get(normal_velocity) * get(vn_times_one_minus_cs_sqrd)));
  get(first_term) *= get(vn_times_one_minus_cs_sqrd);

  char_speeds[7] = char_speeds[1] + get(first_term) + get(second_term);
  char_speeds[1] += get(first_term) - get(second_term);

  char_speeds[8] = char_speeds[0] + get(lapse);
  char_speeds[0] -= get(lapse);
}

// Guard a denominator against vanishing WITHOUT changing its sign.
//
// The `denominator + 1.0e-14` idiom previously used here has two defects.
// First it is SIGN-BREAKING: a denominator of -1e-15 becomes +9e-15, so the
// term it divides flips sign instead of being suppressed. Second, 1e-14 is
// dimensionally arbitrary -- negligible against a pressure of 1e3, and
// completely dominant against a pressure of 1e-20 -- so it silently sets the
// answer for atmosphere cells while doing nothing where the denominator is
// genuinely small but physical.
//
// Instead floor the MAGNITUDE, keeping the sign, relative to the natural scale
// of the quantity (with a 1.0 backstop so the floor never itself degenerates).
// Where the denominator is healthy this is exactly the identity.
DataVector floored_denominator(const DataVector& denominator,
                               const DataVector& scale,
                               const double relative_floor = 1.0e-12) {
  DataVector result = denominator;
  for (size_t i = 0; i < result.size(); ++i) {
    const double floor_value =
        relative_floor * std::max(std::abs(scale[i]), 1.0);
    if (std::abs(result[i]) < floor_value) {
      result[i] = result[i] < 0.0 ? -floor_value : floor_value;
      ++grmhd::ValenciaDivClean::detail::denominator_floor_counter();
    }
  }
  return result;
}

DataVector floored_denominator(const DataVector& denominator,
                               const double scale = 1.0,
                               const double relative_floor = 1.0e-12) {
  DataVector result = denominator;
  const double floor_value = relative_floor * std::max(std::abs(scale), 1.0);
  for (size_t i = 0; i < result.size(); ++i) {
    if (std::abs(result[i]) < floor_value) {
      result[i] = result[i] < 0.0 ? -floor_value : floor_value;
      ++grmhd::ValenciaDivClean::detail::denominator_floor_counter();
    }
  }
  return result;
}

// The floor on the interlacing slack, for well-separated roots where the
// conditioning estimate below is meaningless.
constexpr double interlacing_slack_floor = 1.0e-10;

// How finely a root of the magnetosonic quartic can be LOCATED AT ALL.
//
// A perturbation of size `noise` in the evaluation of Q displaces a root x by
// noise/|Q'(x)|, and for a monic quartic |Q'(x)| is the product of the
// distances from x to the other three roots. In a cluster of nearly coincident
// speeds it therefore collapses like the CUBE of the cluster width, and the
// roots cannot be ordered more finely than that no matter how the solve is
// done. Cold, weakly magnetized, fast-moving fluid sits in exactly that
// cluster, and that is the Del Zanna jet's cocoon.
//
// MEASURED, on a state the unit-test sweep generates (cs^2 = 1.67e-4,
// b^2/(rho h) = 2.4e-5, W = 3.3): all four magnetosonic roots lie inside
// 2.5e-3, |Q'| at the slow roots is 9.8e-10, the achievable root accuracy is
// 9.8e-6, and the slow-minus root comes back 4.8e-6 BELOW the Alfven speed --
// half the noise floor. That is not a misidentified root, it is two speeds
// that are equal to within what double precision can resolve, and a fixed
// tolerance cannot tell the two situations apart. Hence the slack is stated
// relative to the conditioning.
double quartic_root_uncertainty(const double c0, const double c1,
                                const double c2, const double c3,
                                const double x) {
  const double derivative =
      std::abs(((4.0 * x + 3.0 * c3) * x + 2.0 * c2) * x + c1);
  if (not(derivative > 0.0)) {
    // A genuine multiple root: the ordering carries no information at all.
    return std::numeric_limits<double>::infinity();
  }
  const double noise =
      8.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(c0), std::abs(c1), std::abs(c2), std::abs(c3)});
  return noise / derivative;
}

// The slack the interlacing chain is checked against at one point. The factor
// of four is margin over the estimate above; on the measured state it leaves
// the healthy violation sixteen times inside the slack, while a fast root
// substituted into a slow slot there overshoots it by eighteen times, so the
// check still discriminates where it is loosest.
double interlacing_tolerance(const double c0, const double c1, const double c2,
                             const double c3, const double slow_minus,
                             const double slow_plus) {
  return std::max(interlacing_slack_floor,
                  4.0 * (quartic_root_uncertainty(c0, c1, c2, c3, slow_minus) +
                         quartic_root_uncertainty(c0, c1, c2, c3, slow_plus)));
}

// The physical interlacing of the RMHD characteristic speeds,
//
//   lambda_f^- <= lambda_a^- <= lambda_s^- <= lambda_e
//              <= lambda_s^+ <= lambda_a^+ <= lambda_f^+,
//
// which is Anton, Miralles, Marti, Ibanez, Aloy & Mimica (2010), Eq. (42)
// (papers/2010-anton-et-al-eigenvectors), and is also the ordering the
// `MhdSpeed` enum in Characteristics.hpp is documented with. NOTE THE ORDER:
// the Alfven speed sits OUTSIDE the slow speed and INSIDE the fast one. It is
// easy to write the chain down with slow and Alfven swapped, and a check
// asserting that swapped chain would fire on every ordinary state.
//
// This is the check the residual test structurally cannot do. The quartic has
// four roots and |Q(root)| is ~0 at ANY of them, so verifying the residual at
// the returned "slow" pair says nothing about WHICH pair came back: a
// deflation that hands the fast roots (or one fast and one slow) back as the
// slow pair passes the residual check exactly. The ORDERING is what identifies
// the roots, and nothing else here does.
//
// Returns the largest amount by which a member of the chain exceeds its
// successor, and 0.0 when the chain is ordered.
double interlacing_violation(const double fast_minus, const double alfven_minus,
                             const double slow_minus, const double entropy,
                             const double slow_plus, const double alfven_plus,
                             const double fast_plus) {
  const std::array<double, 7> chain{{fast_minus, alfven_minus, slow_minus,
                                     entropy, slow_plus, alfven_plus,
                                     fast_plus}};
  double worst = 0.0;
  for (size_t i = 0; i + 1 < chain.size(); ++i) {
    worst = std::max(worst, gsl::at(chain, i) - gsl::at(chain, i + 1));
  }
  return worst;
}

}  // namespace

namespace grmhd::ValenciaDivClean {


template <size_t ThermodynamicDim>
void characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  // Remaining places to reduce allocations:
  // - EoS calls: 2 allocations
  // - Pass temp pointer to Rel Euler: 1 allocation
  // - Return a DataVectorArray (not yet implemented): 9 allocations
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0>(shift).size()};

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);
  const auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(
          &get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors)),
      magnetic_field, spatial_metric);
  const auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
              temp_tensors)),
      magnetic_field, spatial_velocity_one_form);
  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  const auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(
                  temp_tensors)),
              magnetic_field, magnetic_field_one_form);
  const auto& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(
      temp_tensors)) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));

  // reuse magnetic_field_squared allocation for Alfven speed squared
  const auto& alfven_speed_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors)) =
      get(comoving_magnetic_field_squared) /
      (get(comoving_magnetic_field_squared) +
       get(rest_mass_density) * get(specific_enthalpy));

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }

  compute_characteristic_speeds_approximate_mhd(
      char_speeds, lapse, shift, spatial_velocity, spatial_velocity_squared,
      sound_speed_squared, alfven_speed_squared, unit_normal);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds_approximate_mhd(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::array<DataVector, 9> char_speeds{};
  characteristic_speeds_approximate_mhd(
      make_not_null(&char_speeds), rest_mass_density, electron_fraction,
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
  return char_speeds;
}

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_grid_points) {
    for (size_t i = 0; i < 3; ++i) {
      characteristic_speeds->get(i) = DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::Temperature<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      temp_tensors{num_grid_points};

  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);

  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    Scalar<DataVector>& temperature =
        get<hydro::Tags::Temperature<DataVector>>(temp_tensors);
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }

  // Calculate the characteristic speed for non-degenerate ones
  Scalar<DataVector>& denom = get<::Tags::TempScalar<1>>(temp_tensors);
  get(denom) = 1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<2>>(temp_tensors);
  get(first_term) =
      (1.0 - get(sound_speed_squared)) * get(normal_velocity) / get(denom);

  Scalar<DataVector>& second_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(second_term) =
      sqrt(get(sound_speed_squared)) *
      sqrt(get(denom) - get(normal_velocity) * get(normal_velocity) *
                            (1 - get(sound_speed_squared))) /
      (get(lorentz_factor) * get(denom));

  // Degenerate characteristic speed (normal dot velocity)
  characteristic_speeds->get(HydroSpeed::NormalDotVelocity) =
      get(normal_velocity);
  characteristic_speeds->get(HydroSpeed::LambdaPlus) =
      get(first_term) + get(second_term);
  characteristic_speeds->get(HydroSpeed::LambdaMinus) =
      get(first_term) - get(second_term);
}


template <size_t ThermodynamicDim>
tnsr::i<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  tnsr::i<DataVector, 3> characteristic_speeds{};
  characteristic_speeds_hydro(make_not_null(&characteristic_speeds),
                              spatial_velocity, rest_mass_density,
                              specific_internal_energy, electron_fraction,
                              lorentz_factor, specific_enthalpy, spatial_metric,
                              unit_normal, equation_of_state);
  return characteristic_speeds;
}

void magnetosonic_quartic_coefficients(
    gsl::not_null<tnsr::i<DataVector, 4>*> quartic_coefficients,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& normal_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& normal_magnetic_field,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& comoving_magnetic_field_squared) {
  // The expressions in this function have been optimized by Codex. See
  // unoptimized::magnetosonic_quartic_coefficients in Test_Characteristics.cpp
  // for the original expressions, which were directly compared against the
  // results from a Mathematica notebook.

  const DataVector& cs2 = get(sound_speed_squared);
  const DataVector& b2scaled = get(comoving_magnetic_field_squared);
  const DataVector& Bvscaled = get(magnetic_field_dot_spatial_velocity);
  const DataVector& Bsscaled = get(normal_magnetic_field);
  const DataVector& vn = get(normal_velocity);
  const DataVector& W = get(lorentz_factor);
  (void)magnetic_field_squared;

  Variables<tmpl::list<::Tags::TempScalar<0>>> temp_tensors{cs2.size()};
  DataVector& inv_denom = get(get<::Tags::TempScalar<0>>(temp_tensors));
  inv_denom = 1.0 / (square(W) * (b2scaled + square(W) * (1.0 - cs2) +
                                  cs2 * (1.0 - square(Bvscaled))));

  get<1>(*quartic_coefficients) =
      -(2.0 * Bsscaled * Bvscaled * cs2 +
        2.0 * vn * square(W) *
            (-b2scaled + (-1.0 + square(Bvscaled)) * cs2 -
             2.0 * (-1.0 + cs2) * square(vn) * square(W))) *
      inv_denom;
  get<3>(*quartic_coefficients) =
      -(-2.0 * Bsscaled * Bvscaled * cs2 +
        2.0 * vn * square(W) *
            (b2scaled + 2.0 * square(W) * (1.0 - cs2) +
             cs2 * (1.0 - square(Bvscaled)))) *
      inv_denom;

  inv_denom /= square(W);
  get<0>(*quartic_coefficients) =
      -(-square(Bsscaled) * cs2 -
        2.0 * Bsscaled * Bvscaled * cs2 * vn * square(W) +
        square(vn) * square(square(W)) *
            (b2scaled - (-1.0 + square(Bvscaled)) * cs2 +
             (-1.0 + cs2) * square(vn) * square(W))) *
      inv_denom;
  get<2>(*quartic_coefficients) =
      -(square(Bsscaled) * cs2 +
        2.0 * Bsscaled * Bvscaled * cs2 * vn * square(W) -
        (b2scaled - (-1.0 + square(Bvscaled)) * cs2) * (-1.0 + square(vn)) *
            square(square(W)) +
        6.0 * (-1.0 + cs2) * square(vn) * square(square(W)) * square(W)) *
      inv_denom;
}

void find_magnetosonic_speed_from_quartic(
    gsl::not_null<DataVector*> magnetosonic_speed,
    const tnsr::i<DataVector, 4>& quartic_coefficients) {
  // This function has been optimized by Codex to minimize memory allocation and
  // speed it up. See unoptimized::find_magnetosonic_speed_from_quartic in
  // Test_Characteristics.cpp for an easier-to-read implementation.

  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
      temp_tensors{magnetosonic_speed->size()};

  constexpr size_t max_iters = 100;
  constexpr double tolerance = 1.0e-14;

  // We define the coefficients so that the quartic polynomial is
  // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
  DataVector& x = *magnetosonic_speed;
  const DataVector& c0 = get<0>(quartic_coefficients);
  const DataVector& c1 = get<1>(quartic_coefficients);
  const DataVector& c2 = get<2>(quartic_coefficients);
  const DataVector& c3 = get<3>(quartic_coefficients);

  // Find the root using Newton-Rapshon
  DataVector& F = get(get<::Tags::TempScalar<0>>(temp_tensors));
  DataVector& dF = get(get<::Tags::TempScalar<1>>(temp_tensors));

  // Evaluating F by Horner cannot resolve below the cancellation floor
  //   eps * sum_k |c_k x^k|,
  // because at a root the O(|c_k x^k|) terms must cancel to zero. The quartic
  // coefficients grow with magnetisation, so at high sigma that floor can
  // exceed the fixed 1e-14 target -- the old test `max(abs(F)) < tolerance`
  // then becomes UNSATISFIABLE and Newton spins in round-off noise until the
  // iteration cap and ERRORs out. Observed on a magnetization sweep:
  //   sigma=20 stalled at |F| = 5.5e-13 (only ~55x the target),
  //   sigma=10 stalled at |F| = 4.2e-8.
  // Accept a point once |F| reaches the larger of `tolerance` and that
  // achievable floor. This is strictly MORE permissive than the old test, so
  // every state that converged before still converges; it only stops us
  // demanding accuracy that double precision cannot deliver.
  constexpr double eps_mach = std::numeric_limits<double>::epsilon();
  const auto achievable_tolerance = [&c0, &c1, &c2, &c3,
                                    tolerance](const double x_point,
                                                         const size_t point) {
    const double ax = std::abs(x_point);
    const double sum_abs =
        (((ax + std::abs(c3[point])) * ax + std::abs(c2[point])) * ax +
         std::abs(c1[point])) *
            ax +
        std::abs(c0[point]);
    return std::max(tolerance, 8.0 * eps_mach * sum_abs);
  };

  for (size_t iter = 0; iter < max_iters; ++iter) {
    // Horner form minimizes intermediate temporary vectors in the hot loop.
    // F(x) = x^4 + c3 x^3 + c2 x^2 + c1 x + c0
    F = (((x + c3) * x + c2) * x + c1) * x + c0;

    bool all_converged = true;
    for (size_t point = 0; point < x.size(); ++point) {
      if (std::abs(F[point]) >= achievable_tolerance(x[point], point)) {
        all_converged = false;
        break;
      }
    }
    if (all_converged) {
      return;
    }

    // F'(x) = 4 x^3 + 3 c3 x^2 + 2 c2 x + c1
    dF = ((4.0 * x + 3.0 * c3) * x + 2.0 * c2) * x + c1;

    // Freeze points that have already reached their achievable accuracy, so a
    // single hard point cannot keep stepping -- and degrading -- the ones that
    // are already as good as double precision allows.
    for (size_t point = 0; point < x.size(); ++point) {
      if (std::abs(F[point]) < achievable_tolerance(x[point], point)) {
        F[point] = 0.0;
        dF[point] = 1.0;
      }
    }

    // Avoid FPE from dividing by a small derivative
    if (min(abs(dF)) < tolerance) {
      // It's possible for some points to have converged and others not.
      for (size_t point = 0; point < x.size(); ++point) {
        if (abs(F[point]) < tolerance) {
          // If a point has converged and has a small derivative, then just
          // skip the Newton step.
          F[point] = 0.0;
          dF[point] = 1.0;
        } else if (abs(F[point]) >= tolerance and abs(dF[point]) < tolerance) {
          // If a point that hasn't converged has a small derivative, then the
          // Newton step would be unreliable, so we error out.
          CAPTURE_FOR_ERROR(x);
          CAPTURE_FOR_ERROR(F);
          CAPTURE_FOR_ERROR(dF);
          ERROR(
              "Failed to compute magnetosonic speed from quartic: derivative "
              "is too small for a reliable Newton step. "
              "x = "
              << x[point] << ", F = " << F[point] << ", dF = " << dF[point]);
          return;
        }
      }
    }

    // Newton step
    x -= F / dF;
  }

  F = (((x + c3) * x + c2) * x + c1) * x + c0;
  // Report the achievable floor alongside |F| so a residual failure says
  // whether the target was reachable at all.
  size_t worst_point = 0;
  double worst_ratio = 0.0;
  for (size_t point = 0; point < x.size(); ++point) {
    const double ratio =
        std::abs(F[point]) / achievable_tolerance(x[point], point);
    if (ratio > worst_ratio) {
      worst_ratio = ratio;
      worst_point = point;
    }
  }
  ERROR(
      "Failed to compute magnetosonic speed from quartic: exceeded maximum "
      "number of iterations. Max |F| = "
      << max(abs(F)) << ", worst |F|/achievable_floor = " << worst_ratio
      << " at x = " << x[worst_point]
      << " (floor = " << achievable_tolerance(x[worst_point], worst_point)
      << ")");
}

template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    const gsl::not_null<tnsr::i<DataVector, 9>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const SlowMagnetosonicSpeedMethod slow_speed_method) {
  const size_t num_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_points) {
    for (size_t i = 0; i < 9; ++i) {
      characteristic_speeds->get(i) = DataVector(num_points, 0.0);
    }
  }

  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::Tempi<0, 4>>>
      temp_tensors{num_points};

  // Get sound speed from EoS
  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    // As for flux_jacobian_mhd / characteristic_eigenvectors_mhd, only
    // equilibrium 3D EoSs are supported: the 2-argument chi and
    // kappa_times_p_over_rho_squared come from the underlying 2D EoS.
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "Characteristic speeds for MHD with a 3D equation of state currently "
          "only support equilibrium EoSs.");
    }
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  }

  // --- admissibility guard on the sound speed --------------------------------
  // Ideal RMHD is hyperbolic -- and the magnetosonic quartic below has four
  // real roots, all inside the light cone -- only for 0 <= c_s^2 < 1 (Anile
  // 1989; Del Zanna, Bucciantini & Londrillo, A&A 400, 397 (2003), Sect. 2.2,
  // their Eq. 16). Every physical equation of state respects that, but a
  // state that is not consistent with its equation of state need not: if rho,
  // eps and p are averaged INDEPENDENTLY across a face to build one interface
  // eigensystem, the triple satisfies no equation of state, and
  // since c_s^2 is formed here as (chi + kappa p/rho^2)/h with chi and kappa
  // evaluated at (rho_avg, eps_avg) but h built from p_avg, it can come out
  // ABOVE 1. For an ideal fluid the ceiling of that mismatch is
  // Gamma (Gamma - 1), which exceeds 1 for every Gamma above the golden ratio
  // 1.618 -- so Gamma = 5/3 is exposed and Gamma = 4/3 is not. Reaching it
  // needs eps_avg >~ 9 with p_avg/rho_avg small, i.e. a strong contact between
  // a hot rarefied cell and a cold dense one: exactly the cocoon/ambient
  // interface of the Del Zanna jet.
  //
  // What a superluminal c_s^2 does downstream is not a rounding problem.
  // Measured on 300,000 random states with geometrically consistent (v, B, n):
  //     c_s^2 <= 1     -> reduced-quadratic discriminant in [-6.2e-7, +4.00],
  //                       which is exactly the round-off the clamp further
  //                       down was written for (the -6.2e-7 floor is from an
  //                       adversarial search over admissible states; a random
  //                       sweep at fixed c_s^2 = 0.2 reaches only -3.6e-8);
  //     c_s^2 = 1.001  -> discriminant down to -7.2e+03, 1.2% of states below
  //                       -1e-3; c_s^2 = 1.111 -> down to -1.9e+03, 24%.
  // The practical onset is not exactly c_s^2 = 1 but c_s^2 - 1 ~ 0.25/W^2:
  // the quartic's leading coefficient carries W^2 (1 - c_s^2), so a state with
  // W <= 1.5 cannot be broken by this mechanism at all even at the Gamma = 5/3
  // ceiling 1.1111 (0 of 20,000 draws), while at the jet's W ~ 7 the onset is
  // an overshoot of 5.2e-3. Gamma > 1.618 makes c_s^2 > 1 REACHABLE; W decides
  // whether it breaks anything. Both are necessary, neither is sufficient.
  // The quartic either acquires a complex-conjugate pair (>90% of the
  // failures) or pushes its extremal roots outside [-1, 1]. The latter also
  // destroys the property that makes the +/-1 seeds in
  // `find_magnetosonic_speed_from_quartic` land on the FAST pair -- for x
  // beyond every root the quartic's derivatives alternate in sign, so Newton
  // converges monotonically to the nearest root, and once a root passes +/-1
  // that guarantee is gone and BOTH seeds can converge to the same root. The
  // deflation below is then performed with fast_minus == fast_plus. That is
  // the origin of the
  //   "reduced quadratic has negative discriminant ... discriminant = -7.8678"
  // abort of a Del Zanna et al. (2003) jet run: a discriminant is
  // (slow_plus - slow_minus)^2 when the deflation is valid, so it can only lie
  // in [-4, +4] for roots inside the light cone, and -7.87 is outside that
  // band -- it cannot come from a valid deflation at all, however inaccurate.
  //
  // So a sound speed outside [0, 1) is not repaired here: the quartic below
  // would no longer be the RMHD one, and saturating c_s^2 would hide the
  // caller that produced it. Every caller must pass a state consistent with
  // its equation of state -- each face's own state, or an interface state
  // whose pressure and enthalpy come from the equation of state at the
  // averaged (rho, eps) -- and for a causal equation of state such a state
  // always has 0 <= c_s^2 < 1.
  for (size_t point = 0; point < num_points; ++point) {
    const double cs2 = get(sound_speed_squared)[point];
    if (not(cs2 >= 0.0 and cs2 < 1.0)) {
      ERROR("The sound speed squared is outside [0, 1): c_s^2 = "
            << cs2 << " at point " << point
            << ", rest-mass density = " << get(rest_mass_density)[point]
            << ", specific internal energy = "
            << get(specific_internal_energy)[point]
            << ", specific enthalpy = " << get(specific_enthalpy)[point]
            << ". Either the equation of state is acausal here, or the "
               "caller passed a state that is not consistent with it.");
    }
  }

  // Scalar speeds
  get<MhdSpeed::ScalarPlus>(*characteristic_speeds) = +1.0;
  get<MhdSpeed::ScalarMinus>(*characteristic_speeds) = -1.0;

  // Entropy speed
  /*
    This is just the normal component of the fluid velocity. Since this quantity
    will be used later on, we also define normal_velocity, which points to the
    same DataVector as the entropy speed.
  */
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  get<MhdSpeed::Entropy>(*characteristic_speeds) = get(normal_velocity);

  // Alfven speeds
  // Intermediate magnetic variables
  Scalar<DataVector>& normal_magnetic_field =
      get<::Tags::TempScalar<1>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_magnetic_field),
                  magnetic_field(ti::I) * unit_normal(ti::i));
  tnsr::i<DataVector, 3>& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(make_not_null(&magnetic_field_one_form),
                         spatial_metric(ti::i, ti::j) * magnetic_field(ti::J));
  Scalar<DataVector>& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_dot_spatial_velocity),
                  magnetic_field_one_form(ti::i) * spatial_velocity(ti::I));
  Scalar<DataVector>& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_squared),
                  magnetic_field_one_form(ti::i) * magnetic_field(ti::I));
  Scalar<DataVector>& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(comoving_magnetic_field_squared) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  Scalar<DataVector>& rho_h_star = get<::Tags::TempScalar<3>>(temp_tensors);
  get(rho_h_star) = get(rest_mass_density) * get(specific_enthalpy) +
                    get(comoving_magnetic_field_squared);
  // Compute both speeds with both signs and then assign max/min to be alfven
  // plus/minus to ensure correct ordering.
  Scalar<DataVector>& alfven_1 = get<::Tags::TempScalar<4>>(temp_tensors);
  DataVector& alfven_plus_speed =
      get<MhdSpeed::AlfvenPlus>(*characteristic_speeds);
  DataVector& alfven_minus_speed =
      get<MhdSpeed::AlfvenMinus>(*characteristic_speeds);
  get(alfven_1) =
      get(normal_velocity) +
      get(normal_magnetic_field) / square(get(lorentz_factor)) /
          (get(magnetic_field_dot_spatial_velocity) + sqrt(get(rho_h_star)));
  alfven_minus_speed =
      get(normal_velocity) +
      get(normal_magnetic_field) / square(get(lorentz_factor)) /
          (get(magnetic_field_dot_spatial_velocity) - sqrt(get(rho_h_star)));
  alfven_plus_speed = max(get(alfven_1), alfven_minus_speed);
  alfven_minus_speed = min(get(alfven_1), alfven_minus_speed);

  // Polynomial coefficients for magnetosonic speeds
  // Re-scale magnetic intermediate variables so that they are dimensionless
  // Note: TempScalar<3> and TempScalar<4> are available to be used again here
  Scalar<DataVector>& inv_rho_h = get<::Tags::TempScalar<3>>(temp_tensors);
  Scalar<DataVector>& inv_sqrt_rho_h = get<::Tags::TempScalar<4>>(temp_tensors);
  get(inv_rho_h) = 1.0 / (get(rest_mass_density) * get(specific_enthalpy));
  get(inv_sqrt_rho_h) = sqrt(get(inv_rho_h));
  get(normal_magnetic_field) *= get(inv_sqrt_rho_h);
  get(magnetic_field_dot_spatial_velocity) *= get(inv_sqrt_rho_h);
  get(magnetic_field_squared) *= get(inv_rho_h);
  get(comoving_magnetic_field_squared) *= get(inv_rho_h);
  tnsr::i<DataVector, 4>& quartic_coefficients =
      get<::Tags::Tempi<0, 4>>(temp_tensors);
  magnetosonic_quartic_coefficients(
      make_not_null(&quartic_coefficients), sound_speed_squared,
      normal_velocity, lorentz_factor, normal_magnetic_field,
      magnetic_field_dot_spatial_velocity, magnetic_field_squared,
      comoving_magnetic_field_squared);

  // Fast magnetosonic speeds
  /*
    Find fast magnetosonic speeds via rootfinding of a quartic polynomial with
    initial guess of +1 (for positive speed) or -1 (for negative speed).
  */

  get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds) = 1.0;
  find_magnetosonic_speed_from_quartic(
      make_not_null(
          &get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds)),
      quartic_coefficients);
  get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds) = -1.0;
  find_magnetosonic_speed_from_quartic(
      make_not_null(
          &get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds)),
      quartic_coefficients);

  // Slow magnetosonic speeds
  constexpr double tolerance = 1.0e-14;
  constexpr double eps = 1.0e-12;
  constexpr double discriminant_tolerance = 1.0e-3;
  DataVector& slow_minus =
      get<MhdSpeed::SlowMagnetosonicMinus>(*characteristic_speeds);
  DataVector& slow_plus =
      get<MhdSpeed::SlowMagnetosonicPlus>(*characteristic_speeds);
  const DataVector& vn = get(normal_velocity);
  const DataVector& alfven_minus =
      get<MhdSpeed::AlfvenMinus>(*characteristic_speeds);
  const DataVector& alfven_plus =
      get<MhdSpeed::AlfvenPlus>(*characteristic_speeds);
  const DataVector& fast_minus =
      get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds);
  const DataVector& fast_plus =
      get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds);
  const DataVector& c0 = get<0>(quartic_coefficients);
  const DataVector& c1 = get<1>(quartic_coefficients);
  const DataVector& c2 = get<2>(quartic_coefficients);
  const DataVector& c3 = get<3>(quartic_coefficients);

  const auto evaluate_quartic = [&c0, &c1, &c2, &c3](const double y,
                                                     const size_t point) {
    return (((y + c3[point]) * y + c2[point]) * y + c1[point]) * y + c0[point];
  };

  for (size_t point = 0; point < num_points; ++point) {
    const double vn_i = vn[point];
    const double alfven_minus_i = alfven_minus[point];
    const double alfven_minus_eps_i = alfven_minus_i + eps;
    const double alfven_plus_i = alfven_plus[point];
    const double alfven_plus_eps_i = alfven_plus_i + eps;

    const double N_vn = evaluate_quartic(vn_i, point);
    const double N_alfven_minus = evaluate_quartic(alfven_minus_i, point);
    const double N_alfven_minus_eps =
        evaluate_quartic(alfven_minus_eps_i, point);
    // Every point takes the general solve. Testing whether v_n or an Alfven
    // speed happens to be a root of the quartic (a Type I or Type II
    // degeneracy) and taking the slow pair from it instead saves one sqrt,
    // but cannot establish WHICH root it found: the quartic has four roots and
    // |Q| ~ 0 at each of them, so near a degeneracy such a test accepts a
    // fast or Alfven root as a slow one. The general solve needs no such
    // classification. It is accurate to round-off as a slow root approaches
    // the fast one, and at a Type I degeneracy its error is about half the
    // unresolved gap, which is the correct answer for a pair degenerate to
    // that precision.
    if (slow_speed_method == SlowMagnetosonicSpeedMethod::ReducedQuadratic or
        slow_speed_method ==
            SlowMagnetosonicSpeedMethod::ReducedQuadraticThenNewton or
        slow_speed_method ==
            SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving) {
      const double b_i = c3[point] + fast_minus[point] + fast_plus[point];
      const double c_i = c2[point] +
                         b_i * (fast_minus[point] + fast_plus[point]) -
                         fast_minus[point] * fast_plus[point];
      double discriminant_i = square(b_i) - 4.0 * c_i;
      // The discriminant is (slow_plus - slow_minus)^2 whenever the
      // deflation by the two fast roots is valid, so for an admissible state
      // -- four real roots inside the light cone -- it lies in [0, 4]. A
      // value outside that band is a CERTIFICATE that one of two hypotheses
      // failed: either the state handed in was not admissible, or the +/-1
      // Newton seeds did not land on the extremal pair. It is not a property
      // of the quantity itself (the quantity this line computes is
      // unbounded, which is why it came out at -7.87), and it is not an
      // accuracy problem: the double-precision floor over admissible states
      // is -6.2e-7 (adversarial Nelder-Mead over Gram-consistent states with
      // W up to 1.3e6; a 300,000-state random sweep restricted to cs^2 = 0.2
      // gives the narrower -3.6e-8, which is the number NOT to quote), so
      // the default 1e-3 tolerance is 1600 floors away, not 28,000.
      // The state is printed because the
      // usual cause is upstream -- a caller handing over an interface state
      // that is not a state (see the sound-speed guard above) -- and one
      // production run is expensive enough that the numbers have to come out
      // with the abort.
      ASSERT(discriminant_i >= -discriminant_tolerance,
             "Failed to compute slow magnetosonic speeds: reduced quadratic "
             "has negative discriminant below tolerance. discriminant = "
                 << discriminant_i << ", tolerance = "
                 << discriminant_tolerance << ", point = " << point
                 << ". State at this point: cs^2 = "
                 << get(sound_speed_squared)[point] << ", v_n = " << vn_i
                 << ", W = " << get(lorentz_factor)[point]
                 << ", B_n/sqrt(rho h) = "
                 << get(normal_magnetic_field)[point]
                 << ", (B.v)/sqrt(rho h) = "
                 << get(magnetic_field_dot_spatial_velocity)[point]
                 << ", b^2/(rho h) = "
                 << get(comoving_magnetic_field_squared)[point]
                 << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
                 << c0[point] << ", c1 = " << c1[point] << ", c2 = "
                 << c2[point] << ", c3 = " << c3[point]
                 << ". Fast roots returned by Newton: fast_minus = "
                 << fast_minus[point] << ", fast_plus = " << fast_plus[point]
                 << " (quartic there: "
                 << evaluate_quartic(fast_minus[point], point) << ", "
                 << evaluate_quartic(fast_plus[point], point)
                 << "); Alfven speeds " << alfven_minus_i << ", "
                 << alfven_plus_i << ".");
      // The SAME certificate on the other tail, which until now was not
      // checked at all.
      //
      // The check above is one-sided, and an invalid deflation does not
      // preferentially produce a NEGATIVE discriminant: measured over 40,000
      // random Gram-consistent states per row, a superluminal interface state
      // gives `disc > 4` in 21.7% of draws at cs^2 = 1.001, 8.6% at 1.0597
      // (the jet's own overshoot) and 7.5% at the Gamma = 5/3 ceiling
      // 1.1111, against 13.2/67.5/71.5% for `disc < -1e-3`. On that branch
      // the old code took a sqrt of a large positive number and returned
      // "slow magnetosonic speeds" of up to |lambda| ~ 2e4 -- twenty thousand
      // times the speed of light -- with no message in any build.
      //
      // `PlutoHlld` never reads the slow pair, which is why the jet aborted
      // rather than silently corrupting: it happened to land on the tail that
      // is checked. `Hllem` (its speed gaps and eigenvector build) and
      // `Marquina` DO read it, so for those two this was a live silent path.
      // The bound is the same one as below: disc = (slow_plus - slow_minus)^2
      // and both slow roots are inside the light cone for an admissible
      // state, so disc <= 4, with equality only in the limit slow_minus =
      // -slow_plus = -1.
      ASSERT(discriminant_i <= 4.0 + discriminant_tolerance,
             "Failed to compute slow magnetosonic speeds: reduced quadratic "
             "has a discriminant above 4, which no valid deflation can "
             "produce. discriminant = "
                 << discriminant_i << ", tolerance = "
                 << discriminant_tolerance << ", point = " << point
                 << ". State at this point: cs^2 = "
                 << get(sound_speed_squared)[point] << ", v_n = " << vn_i
                 << ", W = " << get(lorentz_factor)[point]
                 << ", B_n/sqrt(rho h) = "
                 << get(normal_magnetic_field)[point]
                 << ", (B.v)/sqrt(rho h) = "
                 << get(magnetic_field_dot_spatial_velocity)[point]
                 << ", b^2/(rho h) = "
                 << get(comoving_magnetic_field_squared)[point]
                 << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
                 << c0[point] << ", c1 = " << c1[point] << ", c2 = "
                 << c2[point] << ", c3 = " << c3[point]
                 << ". Fast roots returned by Newton: fast_minus = "
                 << fast_minus[point] << ", fast_plus = " << fast_plus[point]
                 << " (quartic there: "
                 << evaluate_quartic(fast_minus[point], point) << ", "
                 << evaluate_quartic(fast_plus[point], point)
                 << "); Alfven speeds " << alfven_minus_i << ", "
                 << alfven_plus_i << ".");
      // Round-off can leave a valid near-degenerate deflation slightly
      // negative (the measured floor over admissible states is -6.2e-7), so
      // floor it at zero before the square root. A value outside the band
      // itself is not repaired: the ASSERTs above report it.
      discriminant_i = std::max(discriminant_i, 0.0);

      // The cancellation-safe root of y^2 + b y + c = 0.
      const double q_i =
          -0.5 * (b_i + (b_i >= 0.0 ? 1.0 : -1.0) * sqrt(discriminant_i));

      slow_plus[point] = q_i;
      slow_minus[point] = -c3[point] - fast_minus[point] - fast_plus[point] -
                          slow_plus[point];
      if (slow_speed_method ==
          SlowMagnetosonicSpeedMethod::ReducedQuadraticThenNewton) {
        // Refine slow roots with one-variable Newton solves initialized at
        // reduced-quadratic estimates.
        DataVector seed_minus{1, slow_minus[point]};
        DataVector seed_plus{1, slow_plus[point]};
        find_magnetosonic_speed_from_quartic(make_not_null(&seed_minus),
                                             quartic_coefficients);
        find_magnetosonic_speed_from_quartic(make_not_null(&seed_plus),
                                             quartic_coefficients);
        slow_minus[point] = seed_minus[0];
        slow_plus[point] = seed_plus[0];
      }
      if (slow_minus[point] > slow_plus[point]) {
        std::swap(slow_minus[point], slow_plus[point]);
      }
    } else {
      CAPTURE_FOR_ERROR(alfven_minus_i);
      CAPTURE_FOR_ERROR(alfven_minus_eps_i);
      CAPTURE_FOR_ERROR(vn_i);
      CAPTURE_FOR_ERROR(alfven_plus_i);
      CAPTURE_FOR_ERROR(alfven_plus_eps_i);
      CAPTURE_FOR_ERROR(N_alfven_minus);
      CAPTURE_FOR_ERROR(N_alfven_minus_eps);
      CAPTURE_FOR_ERROR(N_vn);
      slow_minus[point] = RootFinder::toms748(
          [&evaluate_quartic, point](const double y) {
            return evaluate_quartic(y, point);
          },
          alfven_minus_eps_i, vn_i, N_alfven_minus_eps, N_vn,
          0.05 * tolerance, 0.05 * tolerance, /* max_iterations */ 100);
      slow_plus[point] = -c3[point] - slow_minus[point] - fast_minus[point] -
                         fast_plus[point];
    }

    // ---- slow pair: re-solve in the comoving variable mu = lambda - v_n ----
    // The deflation above inherits the fast roots' error (Newton accepts them
    // at an ABSOLUTE |Q| < 1e-14, i.e. up to 1e-14/|Q'(fast)|, ~2e-12 on warm
    // faces), and a near-double slow pair amplifies it by
    // 1/(slow_plus - slow_minus): with B_n -> 0 the slow speeds come out
    // ~1e-8 off against a slow-entropy gap of 1e-6..1e-5, which makes the
    // analytic slow eigenvectors wrong (the HLLEM "ambient blobs" of the Del
    // Zanna jet; measured in research experiments/slow_root_accuracy). The
    // magnetosonic quartic written in mu (Anton et al. 2010, N_4, times
    // c_s^2/(rho h)),
    //   F(mu) = c_s^2 Bc^2 - (c_s^2 + b^2) W^2 mu^2 + (1 - c_s^2) W^4 mu^4/G,
    //   Bc = B_n/W - (B.v) W mu,   G = 1 - (v_n + mu)^2,
    // has no O(1) cancellation near mu = 0, so its slow roots are located to
    // relative round-off. Seeds: the closed form that zeroes its first two
    // terms (exact when the mu^4 term is negligible, i.e. near the
    // slow/entropy degeneracy; exactly 0 at B_n = 0) and the deflation result
    // (good when the slow root is far from v_n). The seed with the smaller
    // Newton step goes first; if its root leaves this slot's interlacing
    // interval the other seed is tried, and if neither lands the deflation
    // result is kept.
    if (slow_speed_method ==
        SlowMagnetosonicSpeedMethod::ReducedQuadraticComoving) {
      const double cs2_i = get(sound_speed_squared)[point];
      const double w_i = get(lorentz_factor)[point];
      const double w2_i = square(w_i);
      // B quantities are already divided by sqrt(rho h) here
      const double bn_i = get(normal_magnetic_field)[point];
      const double bv_i = get(magnetic_field_dot_spatial_velocity)[point];
      const double b2_i = get(comoving_magnetic_field_squared)[point];
      const double bn_over_w = bn_i / w_i;
      const double bv_w = bv_i * w_i;
      const double mu2_coef = (cs2_i + b2_i) * w2_i;
      const double mu4_coef = (1.0 - cs2_i) * square(w2_i);
      const auto newton_step = [&](const double mu) {
        const double bc = bn_over_w - bv_w * mu;
        const double lam = vn_i + mu;
        const double inv_g = 1.0 / ((1.0 - lam) * (1.0 + lam));
        const double mu2 = square(mu);
        const double f = cs2_i * square(bc) - mu2_coef * mu2 +
                         mu4_coef * square(mu2) * inv_g;
        const double df =
            -2.0 * cs2_i * bc * bv_w - 2.0 * mu2_coef * mu +
            mu4_coef * mu2 * inv_g * (4.0 * mu + 2.0 * mu2 * lam * inv_g);
        if (f == 0.0) {
          return 0.0;  // at a root (e.g. B_n = 0, mu = 0)
        }
        // a stationary point off a root gives no usable step (e.g. a
        // deflation that returned slow_minus = slow_plus = v_n with B.v = 0)
        return df == 0.0 ? std::numeric_limits<double>::infinity() : f / df;
      };
      const double cs_i = sqrt(cs2_i);
      const double t_i = sqrt(cs2_i + b2_i);
      const double num = cs_i * bn_i / w2_i;
      // cs (B.v) -/+ t never vanishes: t^2 - cs^2 (B.v)^2 >= cs^2 > 0
      const double seed_1 = num / (cs_i * bv_i + t_i);
      const double seed_2 = num / (cs_i * bv_i - t_i);
      const std::array<double, 2> closed{
          {std::min(seed_1, seed_2), std::max(seed_1, seed_2)}};
      const std::array<double, 2> deflated{
          {slow_minus[point] - vn_i, slow_plus[point] - vn_i}};
      const double slack =
          4.0 * std::numeric_limits<double>::epsilon() *
          (std::abs(alfven_minus_i) + std::abs(alfven_plus_i) + std::abs(vn_i));
      const std::array<double, 2> lower{
          {alfven_minus_i - vn_i - slack, -slack}};
      const std::array<double, 2> upper{{slack, alfven_plus_i - vn_i + slack}};
      std::array<double, 2> polished{};
      for (size_t j = 0; j < 2; ++j) {
        const std::array<double, 2> seeds{
            {gsl::at(closed, j), gsl::at(deflated, j)}};
        const std::array<double, 2> first_steps{
            {newton_step(seeds[0]), newton_step(seeds[1])}};
        const std::array<size_t, 2> order =
            std::abs(first_steps[0]) <= std::abs(first_steps[1])
                ? std::array<size_t, 2>{{0, 1}}
                : std::array<size_t, 2>{{1, 0}};
        gsl::at(polished, j) = vn_i + gsl::at(deflated, j);
        for (const size_t c : order) {
          double mu = gsl::at(seeds, c);
          double step = gsl::at(first_steps, c);
          for (size_t iter = 0;
               iter < 4 and step != 0.0 and std::isfinite(step); ++iter) {
            mu -= step;
            if (std::abs(step) <=
                4.0 * std::numeric_limits<double>::epsilon() * std::abs(mu)) {
              break;
            }
            step = newton_step(mu);
          }
          if (mu >= gsl::at(lower, j) and mu <= gsl::at(upper, j)) {
            gsl::at(polished, j) = vn_i + mu;
            break;
          }
        }
      }
      slow_minus[point] = std::min(polished[0], polished[1]);
      slow_plus[point] = std::max(polished[0], polished[1]);
    }

    ASSERT(std::abs(evaluate_quartic(slow_minus[point], point)) <
                   10.0 * tolerance and
               std::abs(evaluate_quartic(slow_plus[point], point)) <
                   10.0 * tolerance,
           "Failed to find slow magnetosonic speeds: slow_minus = "
               << slow_minus[point] << ", slow_plus = " << slow_plus[point]
               << ", quartic(slow_minus) = "
               << evaluate_quartic(slow_minus[point], point)
               << ", quartic(slow_plus) = "
               << evaluate_quartic(slow_plus[point], point));
    // ---- interlacing / root identity ----------------------------------
    // The ASSERT above verifies that the returned slow pair SOLVES the
    // quartic. It cannot verify that they are the SLOW pair: the quartic has
    // four roots and the residual is ~0 at every one of them, so a deflation
    // that returns the fast roots -- or one fast and one slow -- in the slow
    // slots satisfies it exactly. Nothing else in this function checks which
    // root is which.
    //
    // The ordering does. For an admissible state the seven physical speeds
    // interlace as lambda_f^- <= lambda_a^- <= lambda_s^- <= lambda_e <=
    // lambda_s^+ <= lambda_a^+ <= lambda_f^+ (Anton et al. 2010, Eq. (42),
    // papers/2010-anton-et-al-eigenvectors; the same chain the `MhdSpeed`
    // enum is documented with in Characteristics.hpp). A fast root in a slow
    // slot breaks it by the fast/slow gap, which is many orders of magnitude
    // above the slack.
    //
    // Placed BEFORE the clamps, like its siblings, so a debug build reports
    // the raw values a broken deflation produced rather than repaired ones.
    ASSERT(interlacing_violation(fast_minus[point], alfven_minus[point],
                                 slow_minus[point], vn_i, slow_plus[point],
                                 alfven_plus[point], fast_plus[point]) <=
               interlacing_tolerance(c0[point], c1[point], c2[point],
                                     c3[point], slow_minus[point],
                                     slow_plus[point]),
           "Slow magnetosonic speeds solve the quartic but are not the SLOW "
           "roots: the characteristic speeds do not interlace. Required "
           "(Anton et al. 2010, Eq. 42): fast- <= alfven- <= slow- <= "
           "entropy <= slow+ <= alfven+ <= fast+, slack "
               << interlacing_tolerance(c0[point], c1[point], c2[point],
                                        c3[point], slow_minus[point],
                                        slow_plus[point])
               << " (four times the quartic's own root-location noise floor, "
                  "which is what a cluster of nearly coincident speeds "
                  "allows). Got fast- = " << fast_minus[point]
               << ", alfven- = " << alfven_minus[point] << ", slow- = "
               << slow_minus[point] << ", entropy = " << vn_i
               << ", slow+ = " << slow_plus[point] << ", alfven+ = "
               << alfven_plus[point] << ", fast+ = " << fast_plus[point]
               << "; worst inversion = "
               << interlacing_violation(
                      fast_minus[point], alfven_minus[point],
                      slow_minus[point], vn_i, slow_plus[point],
                      alfven_plus[point], fast_plus[point])
               << ", point = " << point
               << ". State at this point: cs^2 = "
               << get(sound_speed_squared)[point] << ", v_n = " << vn_i
               << ", W = " << get(lorentz_factor)[point]
               << ", B_n/sqrt(rho h) = "
               << get(normal_magnetic_field)[point]
               << ", (B.v)/sqrt(rho h) = "
               << get(magnetic_field_dot_spatial_velocity)[point]
               << ", b^2/(rho h) = "
               << get(comoving_magnetic_field_squared)[point]
               << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
               << c0[point] << ", c1 = " << c1[point] << ", c2 = "
               << c2[point] << ", c3 = " << c3[point]
               << ". Quartic at the returned slow roots: "
               << evaluate_quartic(slow_minus[point], point) << ", "
               << evaluate_quartic(slow_plus[point], point)
               << " (both ~0 -- which is precisely why the residual check "
                  "above passed).");

    // This function is the flat-space (special-relativistic) one -- it takes
    // no lapse and no shift -- so every root of the quartic lies in [-1, 1]
    // for an admissible state. A slow speed outside the light cone is not a
    // speed.
    ASSERT(std::abs(slow_minus[point]) <= 1.0 and
               std::abs(slow_plus[point]) <= 1.0,
           "Slow magnetosonic speeds outside the light cone: slow- = "
               << slow_minus[point] << ", slow+ = " << slow_plus[point]
               << ", point = " << point);
  }
}

template <size_t ThermodynamicDim>
void characteristic_eigenvectors_mhd(
    const gsl::not_null<tnsr::ij<DataVector, 9>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 9>*> characteristic_projectors,
    const tnsr::i<DataVector, 9>& characteristic_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    const bool skip_fluid_subspace) {
  const size_t num_points = get(lorentz_factor).size();

  Scalar<DataVector> sound_speed_squared{num_points};
  Scalar<DataVector> kappa{num_points};
  Scalar<DataVector> pressure{num_points};
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
  } else if constexpr (ThermodynamicDim == 3) {
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "characteristic_eigenvectors_mhd currently only supports 3D EoSs "
          "in equilibrium.");
    }
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    const Scalar<DataVector> electron_fraction{DataVector(num_points, 0.0)};
    pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction);
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
  }

  const auto det_and_inv_spatial_metric =
      determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv_spatial_metric.first;
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto tangent_1 = orthonormal_oneform(unit_normal, inv_spatial_metric);
  const auto tangent_2 = orthonormal_oneform(
      unit_normal, tangent_1, spatial_metric, det_spatial_metric);
  // Raised (vector) forms of the tangent one-forms.  The Alfven eigenvectors
  // are given in the paper as components in the orthonormal frame
  // (s, tangent_1, tangent_2); to express them in the coordinate basis used by
  // the conserved variables we expand the frame vectors, e.g. for the
  // (vector-valued) right eigenvector S^i = S_n s^i + S_1 t_1^i + S_2 t_2^i.
  // Right eigenvectors use the raised frame (s_vec, tangent_*_up); left
  // eigenvectors use the lowered frame (unit_normal, tangent_*), matching the
  // index convention of the other waves.
  tnsr::I<DataVector, 3> tangent_1_up{num_points};
  tnsr::I<DataVector, 3> tangent_2_up{num_points};
  for (size_t i = 0; i < 3; ++i) {
    tangent_1_up.get(i) = 0.0;
    tangent_2_up.get(i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      tangent_1_up.get(i) += inv_spatial_metric.get(i, j) * tangent_1.get(j);
      tangent_2_up.get(i) += inv_spatial_metric.get(i, j) * tangent_2.get(j);
    }
  }
  Variables<tmpl::list<
      ::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>, ::Tags::TempI<0, 3>,
      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
      ::Tags::TempScalar<12>, ::Tags::TempScalar<13>, ::Tags::TempScalar<14>,
      ::Tags::TempScalar<15>, ::Tags::TempScalar<16>, ::Tags::TempScalar<17>,
      ::Tags::TempScalar<18>, ::Tags::TempScalar<19>, ::Tags::TempScalar<20>,
      ::Tags::TempScalar<21>, ::Tags::TempScalar<22>, ::Tags::TempScalar<23>,
      ::Tags::TempScalar<24>, ::Tags::TempScalar<25>, ::Tags::TempScalar<26>,
      ::Tags::TempScalar<27>, ::Tags::TempScalar<28>, ::Tags::TempScalar<29>,
      ::Tags::TempScalar<30>, ::Tags::TempScalar<31>, ::Tags::TempScalar<32>,
      ::Tags::TempScalar<33>, ::Tags::TempScalar<34>, ::Tags::TempScalar<35>,
      ::Tags::TempScalar<36>, ::Tags::TempScalar<37>, ::Tags::TempScalar<38>,
      ::Tags::TempScalar<39>>>
      temp_tensors{num_points};

  auto& v_cov = get<::Tags::Tempi<0, 3>>(temp_tensors);
  auto& B_cov = get<::Tags::Tempi<1, 3>>(temp_tensors);
  auto& s_vec = get<::Tags::TempI<0, 3>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    v_cov.get(i) = 0.0;
    B_cov.get(i) = 0.0;
    s_vec.get(i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      v_cov.get(i) += spatial_metric.get(i, j) * spatial_velocity.get(j);
      B_cov.get(i) += spatial_metric.get(i, j) * magnetic_field.get(j);
      s_vec.get(i) += inv_spatial_metric.get(i, j) * unit_normal.get(j);
    }
  }

  auto& v_n = get<::Tags::TempScalar<0>>(temp_tensors);
  auto& v_1 = get<::Tags::TempScalar<1>>(temp_tensors);
  auto& v_2 = get<::Tags::TempScalar<2>>(temp_tensors);
  auto& B_n = get<::Tags::TempScalar<3>>(temp_tensors);
  auto& B_1 = get<::Tags::TempScalar<4>>(temp_tensors);
  auto& B_2 = get<::Tags::TempScalar<5>>(temp_tensors);
  get(v_n) = 0.0;
  get(v_1) = 0.0;
  get(v_2) = 0.0;
  get(B_n) = 0.0;
  get(B_1) = 0.0;
  get(B_2) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(v_n) += spatial_velocity.get(i) * unit_normal.get(i);
    get(v_1) += spatial_velocity.get(i) * tangent_1.get(i);
    get(v_2) += spatial_velocity.get(i) * tangent_2.get(i);
    get(B_n) += magnetic_field.get(i) * unit_normal.get(i);
    get(B_1) += magnetic_field.get(i) * tangent_1.get(i);
    get(B_2) += magnetic_field.get(i) * tangent_2.get(i);
  }

  auto& B_squared = get<::Tags::TempScalar<6>>(temp_tensors);
  auto& B_dot_v = get<::Tags::TempScalar<7>>(temp_tensors);
  get(B_squared) = 0.0;
  get(B_dot_v) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    get(B_squared) += magnetic_field.get(i) * B_cov.get(i);
    get(B_dot_v) += B_cov.get(i) * spatial_velocity.get(i);
  }

  const DataVector& rho = get(rest_mass_density);
  const DataVector& h = get(specific_enthalpy);
  const DataVector& W = get(lorentz_factor);
  const DataVector& cs2 = get(sound_speed_squared);
  const DataVector& kappa_i = get(kappa);

  auto& b_squared = get<::Tags::TempScalar<8>>(temp_tensors);
  auto& rho_h_star = get<::Tags::TempScalar<9>>(temp_tensors);
  auto& h_star = get<::Tags::TempScalar<10>>(temp_tensors);
  auto& sqrt_rho_h_star = get<::Tags::TempScalar<11>>(temp_tensors);
  auto& r_1 = get<::Tags::TempScalar<12>>(temp_tensors);
  auto& r_2 = get<::Tags::TempScalar<13>>(temp_tensors);
  auto& r_4 = get<::Tags::TempScalar<14>>(temp_tensors);
  auto& B_21 = get<::Tags::TempScalar<15>>(temp_tensors);
  auto& B_31 = get<::Tags::TempScalar<16>>(temp_tensors);
  auto& B_32 = get<::Tags::TempScalar<17>>(temp_tensors);
  get(b_squared) = get(B_squared) / square(W) + square(get(B_dot_v));
  get(rho_h_star) = rho * h + get(b_squared);
  get(h_star) = h + get(b_squared) / floored_denominator(rho);
  get(sqrt_rho_h_star) = sqrt(get(rho_h_star));
  get(r_1) = get(B_dot_v) + get(sqrt_rho_h_star);
  get(r_2) = get(B_n) * get(v_n) - get(r_1);
  get(r_4) = get(B_squared) + get(r_1) * get(B_dot_v) * square(W);
  get(B_21) = get(B_2) * get(v_1) - get(B_1) * get(v_2);
  get(B_31) = get(B_n) * get(v_1) - get(B_1) * get(v_n);
  get(B_32) = get(B_n) * get(v_2) - get(B_2) * get(v_n);

  auto& a = get<::Tags::TempScalar<20>>(temp_tensors);
  auto& B = get<::Tags::TempScalar<21>>(temp_tensors);
  auto& G = get<::Tags::TempScalar<22>>(temp_tensors);
  auto& script_G = get<::Tags::TempScalar<23>>(temp_tensors);
  auto& script_G_rho = get<::Tags::TempScalar<24>>(temp_tensors);
  auto& kappa_rho = get<::Tags::TempScalar<25>>(temp_tensors);
  auto& Z = get<::Tags::TempScalar<26>>(temp_tensors);
  auto& K = get<::Tags::TempScalar<27>>(temp_tensors);
  auto& kappa_B = get<::Tags::TempScalar<28>>(temp_tensors);
  auto& kappa_Bv = get<::Tags::TempScalar<29>>(temp_tensors);
  auto& m_1s = get<::Tags::TempScalar<30>>(temp_tensors);
  auto& m_1v = get<::Tags::TempScalar<31>>(temp_tensors);
  auto& m_1B = get<::Tags::TempScalar<32>>(temp_tensors);
  auto& m_4 = get<::Tags::TempScalar<33>>(temp_tensors);
  auto& f_1v = get<::Tags::TempScalar<34>>(temp_tensors);
  auto& g_1B = get<::Tags::TempScalar<35>>(temp_tensors);
  auto& g_1v = get<::Tags::TempScalar<36>>(temp_tensors);
  auto& h_1 = get<::Tags::TempScalar<37>>(temp_tensors);
  auto& G_entropy = get<::Tags::TempScalar<38>>(temp_tensors);

  for (size_t wave = 0; wave < 9; ++wave) {
    // Optimization (skip-degenerate): when the caller will unconditionally
    // complement the collapse-prone fluid subspace (MhdSpeed 2-6 = Alfven-,
    // slow-, entropy, slow+, Alfven+) -- i.e. AlwaysComplementaryProjection --
    // there is no need to build those (expensive, ill-conditioned) analytic
    // eigenvectors: they get zeroed and reconstructed by complement anyway.
    // Zero their rows and skip, giving a bit-identical result with less work.
    if (skip_fluid_subspace and wave >= 2 and wave <= 6) {
      for (size_t comp = 0; comp < 9; ++comp) {
        characteristic_modes->get(wave, comp) = DataVector(num_points, 0.0);
        characteristic_projectors->get(wave, comp) =
            DataVector(num_points, 0.0);
      }
      continue;
    }
    const DataVector& y = characteristic_speeds.get(wave);
    get(a) = W * (get(v_n) - y);
    get(B) = get(B_n) / W + get(B_dot_v) * W * (get(v_n) - y);
    get(G) = 1.0 - square(y);
    get(script_G) = rho * h * square(get(a)) - get(G) * get(b_squared);
    // Same atmosphere limit as the kappa guard above: p -> 0 drives cs2 -> 0,
    // so this denominator vanishes for a cell at the floor. Mask the division
    // rather than let it raise an FPE (which the HLLEM caller used to suppress).
    get(script_G_rho) =
        get(script_G) / floored_denominator(rho * h * cs2, rho * h);
    get(kappa_rho) = kappa_i + rho * cs2;
    get(Z) = rho * h * square(W);
    get(K) = -W * (1.0 - get(v_n) * y);
    get(kappa_B) = get(kappa_rho) * square(get(B)) +
                   (1.0 - cs2) * square(rho * get(a)) * h;
    get(kappa_Bv) = get(kappa_B) * get(B_dot_v) -
                    get(kappa_rho) * rho * get(a) * get(B) * get(h_star);
    const DataVector a_denom = floored_denominator(get(a), W);
    const DataVector G_denom = floored_denominator(get(G));
    const DataVector cs2_denom = floored_denominator(cs2);

    if (wave == MhdSpeed::Entropy) {
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            h * W * (kappa_i - rho * cs2) * spatial_velocity.get(i);
        characteristic_modes->get(wave, 3 + i) = 0.0;
      }
      characteristic_modes->get(wave, 6) = kappa_i;
      characteristic_modes->get(wave, 7) =
          kappa_i * (h * W - 1.0) - rho * h * W * cs2;
      characteristic_modes->get(wave, 8) = 0.0;

      get(G_entropy) = 1.0 - square(get(v_n));
      const DataVector entropy_norm =
          1.0 / floored_denominator(rho * h * cs2, rho * h);
      for (size_t i = 0; i < 3; ++i) {
        // S_b components
        characteristic_projectors->get(wave, i) =
            entropy_norm * W * v_cov.get(i);
        // B_b components: gamma_{ba}b^a - (B_n / GW) s_b
        const DataVector b_cov_i =
            B_cov.get(i) / W + W * v_cov.get(i) * get(B_dot_v);
        characteristic_projectors->get(wave, 3 + i) =
            entropy_norm *
            (b_cov_i -
             (get(B_n) / floored_denominator(get(G_entropy) * W, W)) *
                 unit_normal.get(i));
      }
      // D component
      characteristic_projectors->get(wave, 6) = entropy_norm * (h - W);
      // tau component
      characteristic_projectors->get(wave, 7) = entropy_norm * (-W);
      // phi component: W B^a v_a - B_n v_n / GW
      characteristic_projectors->get(wave, 8) =
          entropy_norm *
          (W * get(B_dot_v) -
           (get(B_n) * get(v_n)) /
               floored_denominator(get(G_entropy) * W, W));
    } else if (wave == MhdSpeed::AlfvenMinus or wave == MhdSpeed::AlfvenPlus) {
      // The two Alfven eigenvectors differ by the sign of sqrt(rho h*)
      // (Teukolsky, "Characteristic Decomposition ... II. MHD", the
      // eigenvector for B = -+sqrt(rho h*) a); the quantities r_1 and r_4 must
      // carry the same sign.  That sign must match the one carried by this
      // wave's OWN stored speed, which (Teukolsky Eq. 2.74) is
      //   y = v_n + B_n / (W^2 (B.v + sigma sqrt(rho h*))),  sigma = +-1.
      // The speeds are stored ordered by magnitude (max -> AlfvenPlus,
      // min -> AlfvenMinus); that ordering does NOT track sigma when B_n < 0
      // (e.g. for a normal pointing along -x), so choosing sigma from the wave
      // label is wrong there and breaks biorthogonality.  Instead pick sigma
      // per grid point by matching the stored speed y to its sqrt(rho h*)
      // branch.  (Using the same sign for both waves makes the two Alfven
      // eigenvectors identical and also breaks biorthogonality.)
      const DataVector y_plus_branch =
          get(v_n) +
          get(B_n) / floored_denominator(
                         square(W) * (get(B_dot_v) + get(sqrt_rho_h_star)),
                         square(W) * get(sqrt_rho_h_star));
      const DataVector y_minus_branch =
          get(v_n) +
          get(B_n) / floored_denominator(
                         square(W) * (get(B_dot_v) - get(sqrt_rho_h_star)),
                         square(W) * get(sqrt_rho_h_star));
      DataVector alf_sign{num_points};
      for (size_t point = 0; point < num_points; ++point) {
        alf_sign[point] = (std::abs(y[point] - y_plus_branch[point]) <=
                           std::abs(y[point] - y_minus_branch[point]))
                              ? 1.0
                              : -1.0;
      }
      const DataVector sqrt_rho_h_star_s = alf_sign * get(sqrt_rho_h_star);
      const DataVector r_1_s = get(B_dot_v) + sqrt_rho_h_star_s;
      const DataVector r_4_s =
          get(B_squared) + r_1_s * get(B_dot_v) * square(W);

      const DataVector& y_Alf = y;
      const DataVector inv_sqrt_rho_h_star_s =
          1.0 / floored_denominator(sqrt_rho_h_star_s);

      // The paper gives the Alfven eigenvector's S and B blocks as components
      // in the orthonormal frame (s, tangent_1, tangent_2) (Teukolsky Eq. 3.38
      // / Eq. 2.78, "the first 6 components are the (s, t_(1), t_(2))
      // components").  We must rotate them into the coordinate basis used by
      // the conserved variables by expanding the frame vectors, e.g.
      // S^i = S_n s^i + S_1 t_1^i + S_2 t_2^i.  Right eigenvectors use the
      // raised frame (s_vec, tangent_*_up); left eigenvectors use the lowered
      // frame (unit_normal, tangent_*) -- the same index convention as the
      // entropy/magnetosonic/scalar waves.

      // Right eigenvector frame components (Teukolsky Eq. 3.38, sign branch).
      const DataVector r_modes_s_n =  // S along s
          -2.0 * sqrt_rho_h_star_s * get(B_21) *
          (get(B_n) + r_1_s * get(v_n) * square(W));
      const DataVector r_modes_s_1 =  // S along tangent_1
          -sqrt_rho_h_star_s *
          (get(B_n) * get(B_32) + get(B_1) * get(B_21) +
           r_1_s * square(W) *
               (get(B_2) + get(v_1) * get(B_21) + get(v_n) * get(B_32)));
      const DataVector r_modes_s_2 =  // S along tangent_2
          sqrt_rho_h_star_s *
          (get(B_n) * get(B_31) - get(B_2) * get(B_21) +
           r_1_s * square(W) *
               (get(B_1) - get(v_2) * get(B_21) + get(v_n) * get(B_31)));
      // B along s vanishes; B along tangent_1, tangent_2:
      const DataVector r_modes_b_1 = sqrt_rho_h_star_s * get(B_2) + get(v_2) * r_4_s;
      const DataVector r_modes_b_2 =
          -sqrt_rho_h_star_s * get(B_1) - get(v_1) * r_4_s;

      // Left eigenvector frame components (Teukolsky Eq. 3.41, sign branch).
      const DataVector l_proj_s_n = inv_sqrt_rho_h_star_s * (get(B_21) * y_Alf);
      const DataVector l_proj_s_1 =
          inv_sqrt_rho_h_star_s * (get(B_2) + get(B_32) * y_Alf);
      const DataVector l_proj_s_2 =
          inv_sqrt_rho_h_star_s * (-get(B_1) - get(B_31) * y_Alf);
      const DataVector l_proj_b_n = -get(B_21) * y_Alf;
      const DataVector l_proj_b_1 = -(get(B_2) + get(B_32) * y_Alf);
      const DataVector l_proj_b_2 = (get(B_1) + get(B_31) * y_Alf);

      // Rotate the frame components into the coordinate basis.
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            r_modes_s_n * s_vec.get(i) + r_modes_s_1 * tangent_1_up.get(i) +
            r_modes_s_2 * tangent_2_up.get(i);
        characteristic_modes->get(wave, 3 + i) =
            r_modes_b_1 * tangent_1_up.get(i) +
            r_modes_b_2 * tangent_2_up.get(i);  // B along s is zero
        characteristic_projectors->get(wave, i) =
            l_proj_s_n * unit_normal.get(i) + l_proj_s_1 * tangent_1.get(i) +
            l_proj_s_2 * tangent_2.get(i);
        characteristic_projectors->get(wave, 3 + i) =
            l_proj_b_n * unit_normal.get(i) + l_proj_b_1 * tangent_1.get(i) +
            l_proj_b_2 * tangent_2.get(i);
      }

      // Scalar components D and tau (Rows 7, 8 of the paper's array) and the
      // appended divergence-cleaning phi component (frame-independent).
      characteristic_modes->get(wave, 6) = -rho * W * get(B_21);
      characteristic_modes->get(wave, 7) =
          -get(B_21) * W * (2.0 * W * sqrt_rho_h_star_s * r_1_s - rho);
      characteristic_modes->get(wave, 8) = 0.0;
      characteristic_projectors->get(wave, 6) =
          inv_sqrt_rho_h_star_s * (-get(B_21));
      characteristic_projectors->get(wave, 7) =
          inv_sqrt_rho_h_star_s * (-get(B_21));
      characteristic_projectors->get(wave, 8) = -get(B_21);
    } else if (wave == MhdSpeed::ScalarMinus or wave == MhdSpeed::ScalarPlus) {
      // Scalar (divergence-cleaning) right eigenvector, Teukolsky Eq. (4.32),
      // with the two typo corrections confirmed by the author (S. A. Teukolsky,
      // private communication).  The printed Eq. (4.32) is NOT a right
      // eigenvector for v != 0:
      //   row 1 (momentum S^i): the term  W^2 B kappa_B (s^i + y v^i)  should be
      //                          W^2 kappa_Bv (s^i + y v^i);
      //   row 3 (D), 2nd term:  (1 - cs^2) rho a B_n  should be
      //                          (1 - cs^2) rho^2 a B_n   (dimensional fix; only
      //                          visible for rho != 1).
      // Verified against the conserved characteristic matrix (flux_jacobian_mhd):
      // A.R = y R to machine precision (including rho != 1).
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            -((y * get(kappa_B) +
               2.0 * get(kappa_rho) * get(a) * get(B) * get(B_n)) *
                  magnetic_field.get(i) +
              square(W) * get(kappa_Bv) *
                  (s_vec.get(i) + y * spatial_velocity.get(i)) -
              2.0 * W * get(kappa_rho) * get(B) * square(get(B_n)) *
                  spatial_velocity.get(i)) /
            (a_denom * W);
        characteristic_modes->get(wave, 3 + i) =
            get(kappa_rho) * y * get(B) * magnetic_field.get(i) / W +
            (s_vec.get(i) - y * spatial_velocity.get(i)) *
                (get(kappa_rho) * get(B) * get(B_n) +
                 (1.0 - cs2) * square(rho) * square(get(a)) * h * W) /
                a_denom;
      }
      // D component (Teukolsky Eq. 4.32, row 3, with rho -> rho^2 in 2nd term).
      characteristic_modes->get(wave, 6) =
          get(kappa_rho) * y * rho * get(B) -
          (1.0 - cs2) * square(rho) * get(a) * get(B_n);
      characteristic_modes->get(wave, 7) =
          (get(kappa_rho) * get(B) *
               (2.0 * square(get(B_n)) +
                rho * get(a) * (get(a) * get(h_star) - y)) -
           get(kappa_B) * get(B) - 2.0 * get(kappa_Bv) * y * W +
           (1.0 - cs2) * square(rho * get(a)) * get(B_n)) /
          a_denom;
      characteristic_modes->get(wave, 8) =
          -(1.0 - cs2) * square(rho * get(a)) * h;

      const DataVector inv_one_minus_vn2 =
          1.0 / floored_denominator(1.0 - square(get(v_n)));
      for (size_t i = 0; i < 3; ++i) {
        characteristic_projectors->get(wave, i) = 0.0;
        characteristic_projectors->get(wave, 3 + i) =
            unit_normal.get(i) * inv_one_minus_vn2;
      }
      characteristic_projectors->get(wave, 6) = 0.0;
      characteristic_projectors->get(wave, 7) = 0.0;
      characteristic_projectors->get(wave, 8) = y * inv_one_minus_vn2;
    } else {
      get(m_1s) = rho * h * get(a) * W *
                  (get(B) * get(B_dot_v) - get(rho_h_star) * get(a));
      get(m_1v) = rho * h *
                  (get(B_n) * get(B_dot_v) * (y * get(a) + 2.0 * get(G) * W) -
                   2.0 * get(a) * square(get(B_n)) -
                   get(a) * W *
                       (y * get(a) * (get(B_squared) / square(W) + rho * h) +
                        (1.0 - 1.0 / cs2_denom) * get(script_G) * W));
      get(m_1B) = rho * h *
                  (get(B) * (y * get(a) - get(G) * W) +
                   2.0 * get(B_n) * (square(get(a)) + get(G))) /
                  W;
      get(m_4) =
          (rho / a_denom) *
          (square(get(a)) * square(get(B)) * h -
           2.0 * square(get(B_n)) * h * (square(get(a)) + get(G)) +
           square(get(B)) * W * (2.0 * y * get(a) * h - get(G)) +
           get(B_n) * get(B) *
               (get(G) + 2.0 * h * W * get(G) - 2.0 * y * get(a) * h) +
           get(script_G) * W * square(get(a)) * (h * W * (1.0 - cs2) - 1.0) /
               cs2_denom +
           rho * h * cube(get(a)) *
               (y - 2.0 * y * get(h_star) * W + get(a) * (W - get(h_star))));
      for (size_t i = 0; i < 3; ++i) {
        characteristic_modes->get(wave, i) =
            get(m_1s) * s_vec.get(i) + get(m_1v) * spatial_velocity.get(i) +
            get(m_1B) * magnetic_field.get(i);
        characteristic_modes->get(wave, 3 + i) =
            rho * h * get(a) *
            (magnetic_field.get(i) * (1.0 - y * get(v_n)) -
             get(B_n) * (s_vec.get(i) - y * spatial_velocity.get(i)));
      }
      characteristic_modes->get(wave, 6) =
          -rho * get(B) * get(G) * get(B_n) / a_denom -
          square(rho) * get(a) * h * (y * get(a) - get(G) * W);
      characteristic_modes->get(wave, 7) = get(m_4);
      characteristic_modes->get(wave, 8) = 0.0;

      get(f_1v) =
          W * (-get(G) +
               get(B) * get(G) * get(B_n) * W /
                   floored_denominator(get(Z) * square(a_denom), get(Z)) +
               get(script_G) * square(W) * (kappa_i + rho) /
                   (get(Z) * rho * cs2_denom));
      get(g_1B) = get(script_G) * kappa_i * W /
                      floored_denominator(rho * get(Z) * cs2_denom,
                                          rho * get(Z)) -
                  (square(get(a)) + get(G)) / W;
      get(g_1v) = get(B_dot_v) * square(W) * get(g_1B) +
                  W * (get(a) * get(B) +
                       get(script_G) * get(B) * square(W) / (get(Z) * a_denom));
      get(h_1) = -(get(f_1v) + y * get(a));
      for (size_t i = 0; i < 3; ++i) {
        characteristic_projectors->get(wave, i) =
            get(a) * unit_normal.get(i) -
            get(B) * get(G) * W * B_cov.get(i) / (get(Z) * a_denom) +
            get(f_1v) * v_cov.get(i);
        characteristic_projectors->get(wave, 3 + i) =
            get(B) * unit_normal.get(i) + get(g_1B) * B_cov.get(i) +
            get(g_1v) * v_cov.get(i) -
            (get(script_G_rho) * get(kappa_rho) * get(B) / (G_denom * rho)) *
                unit_normal.get(i);
      }
      characteristic_projectors->get(wave, 6) =
          get(h_1) +
          get(script_G) * (kappa_i - rho * cs2) / (square(rho) * cs2_denom);
      characteristic_projectors->get(wave, 7) = get(h_1);
      characteristic_projectors->get(wave, 8) =
          (get(B) * get(K) *
               (get(G) * rho - get(script_G_rho) * get(kappa_rho)) +
           get(G) * get(B_n) *
               (rho * (square(get(a)) + get(G)) -
                get(script_G_rho) * kappa_i)) /
          (G_denom * rho * a_denom);
    }
  }
}

template <size_t ThermodynamicDim>
void characteristic_eigenvectors_hydro(
    const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  // Zero the outputs (only a subset of the 6x6 entries are nonzero).
  for (size_t wave = 0; wave < 6; ++wave) {
    for (size_t comp = 0; comp < 6; ++comp) {
      characteristic_modes->get(wave, comp) =
          DataVector(num_grid_points, 0.0);
      characteristic_projectors->get(wave, comp) =
          DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
      hydro::Tags::SpatialVelocitySquared<DataVector>,
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Temperature<DataVector>, hydro::Tags::Pressure<DataVector>,
      hydro::Tags::LorentzFactorSquared<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempII<0, 3>, ::Tags::Tempi<0, 3>, ::Tags::Tempi<1, 3>,
      ::Tags::TempI<0, 3>, ::Tags::TempI<1, 3>, ::Tags::TempI<2, 3>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempScalar<14>, ::Tags::TempScalar<15>,
      ::Tags::TempScalar<16>, ::Tags::TempScalar<17>, ::Tags::TempScalar<18>,
      ::Tags::TempScalar<19>, ::Tags::TempScalar<20>, ::Tags::TempScalar<21>,
      ::Tags::TempScalar<22>, ::Tags::TempScalar<23>, ::Tags::TempScalar<24>,
      ::Tags::TempScalar<25>, ::Tags::TempScalar<26>, ::Tags::TempScalar<27>,
      ::Tags::TempScalar<28>, ::Tags::TempScalar<29>>>
      temp_tensors{num_grid_points};

  Scalar<DataVector>& det_spatial_metric =
      get<::Tags::TempScalar<0>>(temp_tensors);
  auto& inv_spatial_metric = get<::Tags::TempII<0, 3>>(temp_tensors);
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inv_spatial_metric), spatial_metric);

  auto& tangent_one_form_1 = get<::Tags::Tempi<0, 3>>(temp_tensors);
  orthonormal_oneform(make_not_null(&tangent_one_form_1), unit_normal,
                      inv_spatial_metric);

  auto& tangent_one_form_2 = get<::Tags::Tempi<1, 3>>(temp_tensors);
  orthonormal_oneform(make_not_null(&tangent_one_form_2), unit_normal,
                      tangent_one_form_1, spatial_metric, det_spatial_metric);

  auto& unit_normal_vector = get<::Tags::TempI<0, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&unit_normal_vector), unit_normal,
                       inv_spatial_metric);

  auto& tangent_vector_1 = get<::Tags::TempI<1, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&tangent_vector_1), tangent_one_form_1,
                       inv_spatial_metric);

  auto& tangent_vector_2 = get<::Tags::TempI<2, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&tangent_vector_2), tangent_one_form_2,
                       inv_spatial_metric);

  Scalar<DataVector>& v_dot_tangent_1 =
      get<::Tags::TempScalar<1>>(temp_tensors);
  dot_product(make_not_null(&v_dot_tangent_1), tangent_one_form_1,
              spatial_velocity);

  Scalar<DataVector>& v_dot_tangent_2 =
      get<::Tags::TempScalar<2>>(temp_tensors);
  dot_product(make_not_null(&v_dot_tangent_2), tangent_one_form_2,
              spatial_velocity);

  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<3>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  Scalar<DataVector>& one_minus_normal_velocity_squared =
      get<::Tags::TempScalar<4>>(temp_tensors);
  get(one_minus_normal_velocity_squared) = 1.0 - square(get(normal_velocity));

  auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);

  auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_one_form);

  auto& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<5>>(temp_tensors);
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<6>>(temp_tensors);
  auto& pressure = get<hydro::Tags::Pressure<DataVector>>(temp_tensors);

  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(equation_of_state
                 .kappa_times_p_over_rho_squared_from_density_and_energy(
                     rest_mass_density, specific_internal_energy))) /
        get(specific_enthalpy);
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<8>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    // For non-equilibrium 3D EoSs we do not have direct access to kappa and we
    // don't know how to specify zeta, so we currently only support equilibrium
    // 3D EoSs, for which kappa comes from the underlying 2D EoS and zeta = 0.
    // This matches characteristic_speeds_{hydro,mhd}, flux_jacobian_hydro, and
    // characteristic_eigenvectors_mhd; it replaces an earlier placeholder that
    // hard-coded the adiabatic index (only valid when the EoS happened to use
    // that same index).
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "characteristic_eigenvectors_hydro currently only supports 3D EoSs "
          "in equilibrium.");
    }
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<8>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
    get(zeta) = 0.0;
  }

  // This is for the case for zeta = 0.
  const double zeta_max_abs = max(abs(get(zeta)));

  Scalar<DataVector>& sound_speed = get<::Tags::TempScalar<10>>(temp_tensors);
  get(sound_speed) = sqrt(get(sound_speed_squared));
  auto& W_squared =
      get<hydro::Tags::LorentzFactorSquared<DataVector>>(temp_tensors);
  get(W_squared) = square(get(lorentz_factor));

  // Variables in the ordering (D, S_i, tau, DYe)
  // RIGHT eigenvectors

  // R1 and R2
  for (size_t i = 0; i < 3; ++i) {
    characteristic_modes->get(R1, i + 1) =
        get(specific_enthalpy) * (tangent_one_form_1.get(i) +
                                  2.0 * get(W_squared) * get(v_dot_tangent_1) *
                                      spatial_velocity_one_form.get(i));

    characteristic_modes->get(R2, i + 1) =
        get(specific_enthalpy) * (tangent_one_form_2.get(i) +
                                  2.0 * get(W_squared) * get(v_dot_tangent_2) *
                                      spatial_velocity_one_form.get(i));
  }

  characteristic_modes->get(R1, 0) = get(lorentz_factor) * get(v_dot_tangent_1);
  characteristic_modes->get(R2, 0) = get(lorentz_factor) * get(v_dot_tangent_2);

  characteristic_modes->get(R1, 4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_1);

  characteristic_modes->get(R2, 4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_2);

  characteristic_modes->get(R1, 5) =
      get(electron_fraction) * characteristic_modes->get(R1, 0);
  characteristic_modes->get(R2, 5) =
      get(electron_fraction) * characteristic_modes->get(R2, 0);

  // R3
  Scalar<DataVector>& common_R3 = get<::Tags::TempScalar<11>>(temp_tensors);
  get(common_R3) =
      get(specific_enthalpy) * get(lorentz_factor) *
      (get(kappa) - get(rest_mass_density) * get(sound_speed_squared));
  for (size_t i = 0; i < 3; ++i) {
    characteristic_modes->get(R3, i + 1) =
        get(common_R3) * spatial_velocity_one_form.get(i);
  }
  characteristic_modes->get(R3, 0) = get(kappa);
  characteristic_modes->get(R3, 4) = get(common_R3) - get(kappa);
  characteristic_modes->get(R3, 5) =
      get(electron_fraction) * characteristic_modes->get(R3, 0);

  // R4
  if (zeta_max_abs < 1e-14) {
    characteristic_modes->get(R4, 5) = 1.0;
  } else {
    for (size_t i = 0; i < 3; ++i) {
      characteristic_modes->get(R4, i + 1) = spatial_velocity_one_form.get(i);
    }
    characteristic_modes->get(R4, 4) = 1.0;
    characteristic_modes->get(R4, 5) =
        -get(kappa) / (get(zeta) * get(lorentz_factor));
  }

  // R±
  Scalar<DataVector>& denom = get<::Tags::TempScalar<12>>(temp_tensors);
  get(denom) =
      get(lorentz_factor) *
      sqrt(1.0 - get(spatial_velocity_squared) * get(sound_speed_squared) -
           get(normal_velocity) * get(normal_velocity) *
               (1.0 - get(sound_speed_squared)));

  Scalar<DataVector>& sound_speed_over_denom =
      get<::Tags::TempScalar<13>>(temp_tensors);
  Scalar<DataVector>& inv_denom = get<::Tags::TempScalar<28>>(temp_tensors);
  get(inv_denom) = 1.0 / get(denom);
  get(sound_speed_over_denom) = get(sound_speed) * get(inv_denom);
  Scalar<DataVector>& sound_speed_normal_velocity_over_denom =
      get<::Tags::TempScalar<29>>(temp_tensors);
  get(sound_speed_normal_velocity_over_denom) =
      get(sound_speed) * get(normal_velocity) * get(inv_denom);

  for (size_t i = 0; i < 3; ++i) {
    characteristic_modes->get(Rplus, i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) +
         get(sound_speed_over_denom) * unit_normal.get(i));

    characteristic_modes->get(Rminus, i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) -
         get(sound_speed_over_denom) * unit_normal.get(i));
  }

  characteristic_modes->get(Rplus, 0) = 1.0;
  characteristic_modes->get(Rminus, 0) = 1.0;

  characteristic_modes->get(Rplus, 4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 + get(sound_speed_normal_velocity_over_denom)) -
      1.0;

  characteristic_modes->get(Rminus, 4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 - get(sound_speed_normal_velocity_over_denom)) -
      1.0;

  characteristic_modes->get(Rplus, 5) = get(electron_fraction);
  characteristic_modes->get(Rminus, 5) = get(electron_fraction);

  // LEFT eigenvectors
  Scalar<DataVector>& prefactor_L12 = get<::Tags::TempScalar<14>>(temp_tensors);
  get(prefactor_L12) =
      1.0 / (get(specific_enthalpy) * get(one_minus_normal_velocity_squared));

  // L1
  characteristic_projectors->get(L1, 0) = -get(v_dot_tangent_1) * get(prefactor_L12);
  characteristic_projectors->get(L1, 4) = -get(v_dot_tangent_1) * get(prefactor_L12);
  for (size_t i = 0; i < 3; ++i) {
    characteristic_projectors->get(L1, i + 1) =
        (get(v_dot_tangent_1) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         get(one_minus_normal_velocity_squared) * tangent_vector_1.get(i)) *
        get(prefactor_L12);
  }

  // L2
  characteristic_projectors->get(L2, 0) = -get(v_dot_tangent_2) * get(prefactor_L12);
  characteristic_projectors->get(L2, 4) = -get(v_dot_tangent_2) * get(prefactor_L12);
  for (size_t i = 0; i < 3; ++i) {
    characteristic_projectors->get(L2, i + 1) =
        (get(v_dot_tangent_2) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         get(one_minus_normal_velocity_squared) * tangent_vector_2.get(i)) *
        get(prefactor_L12);
  }

  // L3
  {
    Scalar<DataVector>& prefactor_L3 =
        get<::Tags::TempScalar<15>>(temp_tensors);
    get(prefactor_L3) = 1.0 / (get(rest_mass_density) * get(specific_enthalpy) *
                               get(sound_speed_squared));
    Scalar<DataVector>& h_minus_one = get<::Tags::TempScalar<16>>(temp_tensors);
    get(h_minus_one) =
        get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
    Scalar<DataVector>& W_minus_one = get<::Tags::TempScalar<17>>(temp_tensors);
    get(W_minus_one) = get(spatial_velocity_squared) * get(W_squared) /
                       (get(lorentz_factor) + 1.0);
    Scalar<DataVector>& h_minus_W = get<::Tags::TempScalar<18>>(temp_tensors);
    get(h_minus_W) = get(h_minus_one) - get(W_minus_one);

    // The zeta/kappa terms are identically zero for every EoS we support
    // (zeta == 0; see above), but must not be evaluated as 0/kappa because
    // kappa -> 0 in cold / atmosphere cells, giving 0/0 = NaN and an FPE.
    // Guard them exactly like the R4 / L4 blocks do.
    characteristic_projectors->get(L3, 0) =
        get(h_minus_W) * get(prefactor_L3);
    if (zeta_max_abs >= 1e-14) {
      characteristic_projectors->get(L3, 0) +=
          get(zeta) * get(electron_fraction) / get(kappa) * get(prefactor_L3);
    }

    for (size_t i = 0; i < 3; ++i) {
      characteristic_projectors->get(L3, i + 1) =
          (get(lorentz_factor) * spatial_velocity.get(i)) * get(prefactor_L3);
    }

    characteristic_projectors->get(L3, 4) =
        (-get(lorentz_factor)) * get(prefactor_L3);
    if (zeta_max_abs >= 1e-14) {
      characteristic_projectors->get(L3, 5) =
          (-get(zeta) / get(kappa)) * get(prefactor_L3);
    }  // else stays 0 (zeroed above)
  }

  // L4
  {
    if (zeta_max_abs < 1e-14) {
      characteristic_projectors->get(L4, 0) = -get(electron_fraction);
      characteristic_projectors->get(L4, 5) = 1.0;
    } else {
      Scalar<DataVector>& prefactor_L4 =
          get<::Tags::TempScalar<19>>(temp_tensors);
      get(prefactor_L4) = get(zeta) * get(lorentz_factor) / get(kappa);
      characteristic_projectors->get(L4, 0) =
          get(prefactor_L4) * get(electron_fraction);
      characteristic_projectors->get(L4, 5) = -get(prefactor_L4);
    }
  }

  // L±
  {
    Scalar<DataVector>& a = get<::Tags::TempScalar<20>>(temp_tensors);
    get(a) = get(W_squared) * get(one_minus_normal_velocity_squared) *
             (get(kappa) + get(rest_mass_density) * get(sound_speed_squared));
    Scalar<DataVector>& c_plus = get<::Tags::TempScalar<21>>(temp_tensors);
    get(c_plus) = get(rest_mass_density) * get(sound_speed) *
                  (get(sound_speed) + get(normal_velocity) * get(denom));
    Scalar<DataVector>& c_minus = get<::Tags::TempScalar<22>>(temp_tensors);
    get(c_minus) = get(rest_mass_density) * get(sound_speed) *
                   (get(sound_speed) - get(normal_velocity) * get(denom));
    Scalar<DataVector>& b_plus = get<::Tags::TempScalar<23>>(temp_tensors);
    get(b_plus) = get(a) - get(c_plus);
    Scalar<DataVector>& b_minus = get<::Tags::TempScalar<24>>(temp_tensors);
    get(b_minus) = get(a) - get(c_minus);
    Scalar<DataVector>& k_term = get<::Tags::TempScalar<25>>(temp_tensors);
    get(k_term) = get(kappa) -
                  get(rest_mass_density) * get(sound_speed_squared) +
                  get(zeta) * get(electron_fraction) / get(specific_enthalpy);
    Scalar<DataVector>& prefactor_Lpm =
        get<::Tags::TempScalar<26>>(temp_tensors);
    get(prefactor_Lpm) =
        1.0 / (2.0 * get(rest_mass_density) * get(specific_enthalpy) *
               get(lorentz_factor) * get(sound_speed_squared) *
               get(one_minus_normal_velocity_squared));

    // S_i
    for (size_t i = 0; i < 3; ++i) {
      characteristic_projectors->get(Lplus, i + 1) =
          (-get(a) * spatial_velocity.get(i) +
           get(rest_mass_density) * get(sound_speed) *
               (get(sound_speed) * get(normal_velocity) + get(denom)) *
               unit_normal_vector.get(i)) *
          get(prefactor_Lpm);

      characteristic_projectors->get(Lminus, i + 1) =
          (-get(a) * spatial_velocity.get(i) +
           get(rest_mass_density) * get(sound_speed) *
               (get(sound_speed) * get(normal_velocity) - get(denom)) *
               unit_normal_vector.get(i)) *
          get(prefactor_Lpm);
    }

    Scalar<DataVector>& hW_k_term_one_minus_normal_velocity_squared =
        get<::Tags::TempScalar<27>>(temp_tensors);
    get(hW_k_term_one_minus_normal_velocity_squared) =
        get(specific_enthalpy) * get(lorentz_factor) * get(k_term) *
        get(one_minus_normal_velocity_squared);

    // D
    characteristic_projectors->get(Lplus, 0) =
        (get(b_plus) - get(hW_k_term_one_minus_normal_velocity_squared)) *
        get(prefactor_Lpm);

    characteristic_projectors->get(Lminus, 0) =
        (get(b_minus) - get(hW_k_term_one_minus_normal_velocity_squared)) *
        get(prefactor_Lpm);

    // tau
    characteristic_projectors->get(Lplus, 4) = get(b_plus) * get(prefactor_Lpm);
    characteristic_projectors->get(Lminus, 4) = get(b_minus) * get(prefactor_Lpm);

    // DYe
    characteristic_projectors->get(Lplus, 5) =
        (get(zeta) * get(lorentz_factor) *
         get(one_minus_normal_velocity_squared)) *
        get(prefactor_Lpm);
    characteristic_projectors->get(Lminus, 5) =
        (get(zeta) * get(lorentz_factor) *
         get(one_minus_normal_velocity_squared)) *
        get(prefactor_Lpm);
  }
}

template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempI<0, 3>, ::Tags::Tempi<0, 3>,
      ::Tags::TempI<1, 3>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  Scalar<DataVector>& pressure =
      get<hydro::Tags::Pressure<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<2>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    // For non-equilibrium 3D EoSs we do not have direct access to kappa and we
    // don't know how to specify zeta, both of which are needed
    // for the expressions here. So, we currently only support equilibrium 3D
    // EoSs, for which we get kappa from the underlying 2D EoS and set zeta to
    // 0.
    if (not equation_of_state.is_equilibrium()) {
      ERROR(
          "flux_jacobian_hydro currently only supports 3D EoSs in "
          "equilibrium.");
    }
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<2>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
    // kappa = (kappa p / rho^2) * rho^2 / p is 0/0 for a cell driven to the
    // atmosphere, where the equation of state returns p -> 0. The division
    // raised a floating-point exception that used to be INVISIBLE: the HLLEM
    // boundary correction disabled FPE trapping around this call, so the
    // resulting inf/NaN was silently masked to "fall back to HLL here" and the
    // run instead died later and elsewhere, in FixConservatives. kappa is
    // finite in that limit and the terms it multiplies vanish with p, so
    // masking the division to zero is safe -- the same idiom used elsewhere in
    // ValenciaDivClean (e.g. FixConservatives' masked divisions).
    // p -> 0 for a cell at the atmosphere floor, where this is 0/0. Floor the
    // denominator against the local energy scale rho*h; kappa is then ~0 there,
    // which is the correct limit (the terms it multiplies vanish with p).
    get(kappa) = get(kappa_times_p_over_rho_squared) *
                 square(get(rest_mass_density)) /
                 floored_denominator(get(pressure), get(rest_mass_density) *
                                                        get(specific_enthalpy));
    // For now, we assume that we are at compositional equilibrium, so we set
    // zeta to zero.
    get(zeta) = 0.0;
  }

  // The expressions in this function have been iteratively optimized by Codex
  // with 3 main goals:
  //   1. Avoid catastrophic cancellations by rewriting terms like W-1 to, e.g.,
  //      v^2 W^2 / (W + 1). Known tricks were listed in a skill used by Codex.
  //   2. Avoid repeated calculations by storing intermediate results in
  //      temporary variables and reusing previous allocations when possible.
  //   3. Try different rearrangements and run benchmark to find the best
  //      optimization.

  // Intermediate variables
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<3>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  tnsr::I<DataVector, 3>& unit_vector = get<::Tags::TempI<0, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(make_not_null(&unit_vector),
                         inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  tnsr::i<DataVector, 3>& spatial_velocity_one_form =
      get<::Tags::Tempi<0, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(
      make_not_null(&spatial_velocity_one_form),
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));

  // Cancellation-safe thermodynamic / kinematic differences
  Scalar<DataVector>& h_minus_1 = get<::Tags::TempScalar<9>>(temp_tensors);
  get(h_minus_1) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  Scalar<DataVector>& velocity_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  tenex::evaluate(make_not_null(&velocity_squared),
                  spatial_velocity(ti::I) * spatial_velocity_one_form(ti::i));
  Scalar<DataVector>& lorentz_factor_squared =
      get<::Tags::TempScalar<11>>(temp_tensors);
  get(lorentz_factor_squared) = square(get(lorentz_factor));
  Scalar<DataVector>& W_minus_1 = get<::Tags::TempScalar<12>>(temp_tensors);
  get(W_minus_1) = get(velocity_squared) * get(lorentz_factor_squared) /
                   (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& h_minus_W = get<::Tags::TempScalar<13>>(temp_tensors);
  get(h_minus_W) = get(h_minus_1) - get(W_minus_1);

  // Derivatives of Z
  Scalar<DataVector>& inv_derivative_denom =
      get<::Tags::TempScalar<7>>(temp_tensors);
  get(inv_derivative_denom) =
      1.0 / ((-get(lorentz_factor_squared) + get(sound_speed_squared) *
                                                 get(velocity_squared) *
                                                 get(lorentz_factor_squared)) *
             get(rest_mass_density));
  Scalar<DataVector>& dzdD = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdD),
      -(lorentz_factor() *
        (-kappa() * h_minus_W() - zeta() * electron_fraction() +
         (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
             rest_mass_density()) *
        inv_derivative_denom()));
  tnsr::I<DataVector, 3>& dzds = get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dzds),
      (spatial_velocity(ti::I) * lorentz_factor_squared() *
       (kappa() + sound_speed_squared() * rest_mass_density()) *
       inv_derivative_denom()));
  Scalar<DataVector>& dzdtau = get<::Tags::TempScalar<5>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdtau),
                  -(lorentz_factor_squared() * (kappa() + rest_mass_density()) *
                    inv_derivative_denom()));
  Scalar<DataVector>& dzdye = get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdye),
                  -(zeta() * lorentz_factor() * inv_derivative_denom()));

  // Common factors used repeatedly in the characteristic matrix entries
  Scalar<DataVector>& D_over_Z = get<::Tags::TempScalar<8>>(temp_tensors);
  get(D_over_Z) = 1.0 / (get(specific_enthalpy) * get(lorentz_factor));
  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      (1.0 - get(D_over_Z) * get(dzdD)) * get(normal_velocity);
  characteristic_matrix->get(0, 1) =
      get(D_over_Z) * (unit_vector.get(0) - dzds.get(0) * get(normal_velocity));
  characteristic_matrix->get(0, 2) =
      get(D_over_Z) * (unit_vector.get(1) - dzds.get(1) * get(normal_velocity));
  characteristic_matrix->get(0, 3) =
      get(D_over_Z) * (unit_vector.get(2) - dzds.get(2) * get(normal_velocity));
  characteristic_matrix->get(4, 1) =
      unit_vector.get(0) - characteristic_matrix->get(0, 1);
  characteristic_matrix->get(4, 2) =
      unit_vector.get(1) - characteristic_matrix->get(0, 2);
  characteristic_matrix->get(4, 3) =
      unit_vector.get(2) - characteristic_matrix->get(0, 3);
  characteristic_matrix->get(5, 1) =
      get(electron_fraction) * characteristic_matrix->get(0, 1);
  characteristic_matrix->get(5, 2) =
      get(electron_fraction) * characteristic_matrix->get(0, 2);
  characteristic_matrix->get(5, 3) =
      get(electron_fraction) * characteristic_matrix->get(0, 3);
  characteristic_matrix->get(0, 4) =
      -(get(D_over_Z) * get(dzdtau) * get(normal_velocity));
  characteristic_matrix->get(0, 5) =
      -(get(D_over_Z) * get(dzdye) * get(normal_velocity));
  characteristic_matrix->get(1, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(0) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(0);
  characteristic_matrix->get(2, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(1) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(1);
  characteristic_matrix->get(3, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(2) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(2);

  characteristic_matrix->get(1, 1) =
      get(normal_velocity) +
      unit_vector.get(0) * spatial_velocity_one_form.get(0) +
      dzds.get(0) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(0) +
      dzds.get(1) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(0) +
      dzds.get(2) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(1) +
      dzds.get(0) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 2) =
      get(normal_velocity) +
      unit_vector.get(1) * spatial_velocity_one_form.get(1) +
      dzds.get(1) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(1) +
      dzds.get(2) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(2) +
      dzds.get(0) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(2) +
      dzds.get(1) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 3) =
      get(normal_velocity) +
      unit_vector.get(2) * spatial_velocity_one_form.get(2) +
      dzds.get(2) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));

  characteristic_matrix->get(1, 4) =
      -unit_normal.get(0) +
      get(dzdtau) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 4) =
      -unit_normal.get(1) +
      get(dzdtau) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 4) =
      -unit_normal.get(2) +
      get(dzdtau) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(1, 5) =
      get(dzdye) * (unit_normal.get(0) -
                    get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 5) =
      get(dzdye) * (unit_normal.get(1) -
                    get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 5) =
      get(dzdye) * (unit_normal.get(2) -
                    get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(4, 0) = -characteristic_matrix->get(0, 0);
  characteristic_matrix->get(4, 4) = -characteristic_matrix->get(0, 4);
  characteristic_matrix->get(4, 5) = -characteristic_matrix->get(0, 5);
  characteristic_matrix->get(5, 0) = -get(electron_fraction) * get(D_over_Z) *
                                     get(dzdD) * get(normal_velocity);
  characteristic_matrix->get(5, 4) =
      get(electron_fraction) * characteristic_matrix->get(0, 4);
  characteristic_matrix->get(5, 5) =
      (1.0 - get(electron_fraction) * get(D_over_Z) * get(dzdye)) *
      get(normal_velocity);
}

template <size_t ThermodynamicDim>
void flux_jacobian_mhd(
    const gsl::not_null<tnsr::iJ<DataVector, 9>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Conserved-variable GLM-Valencia characteristic matrix in the normal
  // direction (lapse = 1, shift = 0), generated symbolically.
  // Variable order: [S_x, S_y, S_z, B^x, B^y, B^z, D, tau, phi].  Mirrors
  // flux_jacobian_hydro (same intermediate-quantity conventions).
  const size_t num_points = get(rest_mass_density).size();

  // Sound speed squared and kappa = dp/d(eps) (same logic as flux_jacobian_hydro).
  DataVector soundSpeedSquared(num_points);
  DataVector kappa(num_points, 0.0);
  Scalar<DataVector> pressure(num_points);
  if constexpr (ThermodynamicDim == 1) {
    soundSpeedSquared =
        (get(equation_of_state.chi_from_density(rest_mass_density)) +
         get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
             rest_mass_density))) /
        get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    const DataVector kappa_times_p_over_rho_squared =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    soundSpeedSquared =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         kappa_times_p_over_rho_squared) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    kappa = kappa_times_p_over_rho_squared / get(pressure) *
            square(get(rest_mass_density));
  } else if constexpr (ThermodynamicDim == 3) {
    if (not equation_of_state.is_equilibrium()) {
      ERROR("flux_jacobian_mhd currently only supports 3D EoSs in equilibrium.");
    }
    const DataVector kappa_times_p_over_rho_squared =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    soundSpeedSquared =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         kappa_times_p_over_rho_squared) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
    kappa = kappa_times_p_over_rho_squared / get(pressure) *
            square(get(rest_mass_density));
  }

  // Named quantities matching matrix.txt.
  const DataVector& restMassDensity = get(rest_mass_density);
  const DataVector& lorentzFactor = get(lorentz_factor);
  const DataVector& specificEnthalpy = get(specific_enthalpy);
  const auto& spatialVelocity = spatial_velocity;
  const auto& magneticField = magnetic_field;
  const auto& unitNormal = unit_normal;

  tnsr::I<DataVector, 3> unitVector(num_points);            // s^i = gamma^{ij} n_j
  tnsr::i<DataVector, 3> spatialVelocityOneForm(num_points);  // v_i
  tnsr::i<DataVector, 3> magneticFieldOneForm(num_points);    // B_i
  for (size_t i = 0; i < 3; ++i) {
    unitVector.get(i) = 0.0;
    spatialVelocityOneForm.get(i) = 0.0;
    magneticFieldOneForm.get(i) = 0.0;
    for (size_t j = 0; j < 3; ++j) {
      unitVector.get(i) += inv_spatial_metric.get(i, j) * unit_normal.get(j);
      spatialVelocityOneForm.get(i) +=
          spatial_metric.get(i, j) * spatial_velocity.get(j);
      magneticFieldOneForm.get(i) +=
          spatial_metric.get(i, j) * magnetic_field.get(j);
    }
  }

  DataVector normalVelocity(num_points, 0.0);
  DataVector normalMagneticField(num_points, 0.0);
  DataVector magneticFieldDotVelocity(num_points, 0.0);
  DataVector magneticFieldSquared(num_points, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    normalVelocity += spatial_velocity.get(i) * unit_normal.get(i);
    normalMagneticField += magnetic_field.get(i) * unit_normal.get(i);
    magneticFieldDotVelocity +=
        magnetic_field.get(i) * spatialVelocityOneForm.get(i);
    magneticFieldSquared += magnetic_field.get(i) * magneticFieldOneForm.get(i);
  }
  const DataVector comovingMagneticFieldSquared =
      magneticFieldSquared / square(lorentzFactor) +
      square(magneticFieldDotVelocity);
  const DataVector Zvar =
      restMassDensity * specificEnthalpy * square(lorentzFactor);
  const DataVector Dvar = restMassDensity * lorentzFactor;

  // --- generated matrix entries (As, from matrix.txt) ---
  const auto x0 = Zvar + magneticFieldSquared;
  const auto x1 = 1.0/x0;
  const auto x2 = magneticFieldDotVelocity*unitVector.get(0);
  const auto x3 = 2*normalMagneticField;
  const auto x4 = soundSpeedSquared - 1;
  const auto x5 = restMassDensity*x4;
  const auto x6 = magneticFieldDotVelocity*x5;
  const auto x7 = kappa + restMassDensity;
  const auto x8 = magneticFieldSquared*x7;
  const auto x9 = restMassDensity*soundSpeedSquared;
  const auto x10 = kappa + x9;
  const auto x11 = Zvar*x10;
  const auto x12 = x11 + x8;
  const auto x13 = magneticField.get(0)*x6 + spatialVelocity.get(0)*x12;
  const auto x14 = square(lorentzFactor);
  const auto x16 = magneticFieldDotVelocity*normalVelocity*x14;
  const auto x17 = 2*x14;
  const auto x18 = normalMagneticField*(2 - x17) + x16;
  const auto x19 = 1.0/restMassDensity;
  const auto x23 = square(magneticFieldDotVelocity);
  const auto x24 = x14*x23;
  const auto x26 = x19/(Zvar*soundSpeedSquared*(x14 - 1) + soundSpeedSquared*x24 - x14*(Zvar + comovingMagneticFieldSquared));
  const auto x27 = x18*x26;
  const auto x29 = unitVector.get(0)*x0;
  const auto x31 = magneticFieldDotVelocity*normalMagneticField;
  const auto x33 = -normalVelocity*x0 + x31;
  const auto x35 = x14*(spatialVelocityOneForm.get(0)*x33 + unitNormal.get(0)*x0);
  const auto x37 = normalVelocity*x0 - x31;
  const auto x38 = unitVector.get(1)*x0;
  const auto x39 = magneticField.get(1)*x6 + spatialVelocity.get(1)*x12;
  const auto x40 = magneticFieldDotVelocity*unitVector.get(1);
  const auto x42 = unitVector.get(2)*x0;
  const auto x43 = magneticField.get(2)*x6 + spatialVelocity.get(2)*x12;
  const auto x44 = magneticFieldDotVelocity*unitVector.get(2);
  const auto x46 = x0 + x24;
  const auto x48 = square(restMassDensity);
  const auto x50 = magneticFieldDotVelocity*spatialVelocity.get(0)*x14*(magneticFieldSquared*x10 + specificEnthalpy*x14*x4*x48 + x11);
  const auto x52 = x26*(magneticField.get(0)*(Zvar*kappa - Zvar*restMassDensity - comovingMagneticFieldSquared*restMassDensity*x17 + restMassDensity*x24*(soundSpeedSquared + 1) + x8) + x50);
  const auto x53 = x17*x31;
  const auto x54 = -magneticFieldDotVelocity*normalVelocity*x17 + normalMagneticField*(4*x14 - 4);
  const auto x55 = Zvar*(kappa + restMassDensity*(-soundSpeedSquared*(x17 - 2) + x17 - 1)) - x24*x5 + x8;
  const auto x57 = x26*x3;
  const auto x58 = x26*(magneticField.get(0)*x55 + x50);
  const auto x59 = normalMagneticField*x17;
  const auto x62 = x1/x14;
  const auto x63 = magneticFieldDotVelocity*spatialVelocity.get(1)*x14*(magneticFieldSquared*x10 + specificEnthalpy*x14*x4*x48 + x11);
  const auto x64 = x26*(magneticField.get(1)*(Zvar*kappa - Zvar*restMassDensity - comovingMagneticFieldSquared*restMassDensity*x17 + restMassDensity*x24*(soundSpeedSquared + 1) + x8) + x63);
  const auto x66 = x26*(magneticField.get(1)*x55 + x63);
  const auto x68 = magneticFieldDotVelocity*spatialVelocity.get(2)*x14*(magneticFieldSquared*x10 + specificEnthalpy*x14*x4*x48 + x11);
  const auto x69 = x26*(magneticField.get(2)*(Zvar*kappa - Zvar*restMassDensity - comovingMagneticFieldSquared*restMassDensity*x17 + restMassDensity*x24*(soundSpeedSquared + 1) + x8) + x68);
  const auto x71 = x26*(magneticField.get(2)*x55 + x68);
  const auto x73 = cube(lorentzFactor);
  const auto x75 = x0*(Zvar*(-kappa + x9) + restMassDensity*x7*x73);
  const auto x81 = -Zvar*lorentzFactor*soundSpeedSquared*(x14 - 1) - soundSpeedSquared*x23*x73 + x73*(Zvar + comovingMagneticFieldSquared);
  const auto x86 = x0*x14*x7;
  const auto x87 = -soundSpeedSquared*x24 + soundSpeedSquared*(-Zvar*x14 + Zvar) + x14*(Zvar + comovingMagneticFieldSquared);
  const auto x92 = x14*(spatialVelocityOneForm.get(1)*x33 + unitNormal.get(1)*x0);
  const auto x97 = x14*(spatialVelocityOneForm.get(2)*x33 + unitNormal.get(2)*x0);
  const auto x100 = x14*x26;
  const auto x101 = x100*x13;
  const auto x104 = x100*x39;
  const auto x105 = normalVelocity*x100;
  const auto x107 = x100*x43;
  const auto x116 = 1/(x48*(Zvar*lorentzFactor*soundSpeedSquared*(x14 - 1) + soundSpeedSquared*x23*x73 - x73*(Zvar + comovingMagneticFieldSquared)));
  const auto x125 = Zvar*normalVelocity;
  const auto x126 = x125 + x31;
  const auto x129 = x1/Zvar;
  const auto x130 = Dvar*x129;
  const auto x141 = 2*x125 + x31;
  characteristic_matrix->get(0, 0) = x1*(magneticFieldOneForm.get(0)*(spatialVelocity.get(0)*x3 + x13*x27 - x2) + spatialVelocityOneForm.get(0)*x29 + x13*x26*x35 + x37);
  characteristic_matrix->get(0, 1) = x1*(magneticFieldOneForm.get(0)*(spatialVelocity.get(1)*x3 + x27*x39 - x40) + spatialVelocityOneForm.get(0)*x38 + x26*x35*x39);
  characteristic_matrix->get(0, 2) = x1*(magneticFieldOneForm.get(0)*(spatialVelocity.get(2)*x3 + x27*x43 - x44) + spatialVelocityOneForm.get(0)*x42 + x26*x35*x43);
  characteristic_matrix->get(0, 3) = x62*(-magneticFieldOneForm.get(0)*(magneticField.get(0)*x54 - spatialVelocity.get(0)*x53 + unitVector.get(0)*x46 - x16*x58 - x57*(magneticField.get(0)*x55 + x50) + x58*x59) - normalMagneticField*x46 + x35*x52);
  characteristic_matrix->get(0, 4) = x62*(-magneticFieldOneForm.get(0)*(magneticField.get(1)*x54 - spatialVelocity.get(1)*x53 + unitVector.get(1)*x46 - x16*x66 - x57*(magneticField.get(1)*x55 + x63) + x59*x66) + x35*x64);
  characteristic_matrix->get(0, 5) = x62*(-magneticFieldOneForm.get(0)*(magneticField.get(2)*x54 - spatialVelocity.get(2)*x53 + unitVector.get(2)*x46 - x16*x71 - x57*(magneticField.get(2)*x55 + x68) + x59*x71) + x35*x69);
  characteristic_matrix->get(0, 6) = x62*(magneticFieldOneForm.get(0)*x18*x75 + x14*(spatialVelocityOneForm.get(0)*x33*x75 - unitNormal.get(0)*x0*(x48*x81 - x75)))/(x48*x81);
  characteristic_matrix->get(0, 7) = x1*x19*(magneticFieldOneForm.get(0)*x0*x18*x7 + spatialVelocityOneForm.get(0)*x33*x86 - unitNormal.get(0)*x0*(restMassDensity*x87 - x86))/x87;
  characteristic_matrix->get(0, 8) = 0;
  characteristic_matrix->get(1, 0) = x1*(magneticFieldOneForm.get(1)*(spatialVelocity.get(0)*x3 + x13*x27 - x2) + spatialVelocityOneForm.get(1)*x29 + x13*x26*x92);
  characteristic_matrix->get(1, 1) = x1*(magneticFieldOneForm.get(1)*(spatialVelocity.get(1)*x3 + x27*x39 - x40) + spatialVelocityOneForm.get(1)*x38 + x26*x39*x92 + x37);
  characteristic_matrix->get(1, 2) = x1*(magneticFieldOneForm.get(1)*(spatialVelocity.get(2)*x3 + x27*x43 - x44) + spatialVelocityOneForm.get(1)*x42 + x26*x43*x92);
  characteristic_matrix->get(1, 3) = x62*(-magneticFieldOneForm.get(1)*(magneticField.get(0)*x54 - spatialVelocity.get(0)*x53 + unitVector.get(0)*x46 - x16*x58 - x57*(magneticField.get(0)*x55 + x50) + x58*x59) + x52*x92);
  characteristic_matrix->get(1, 4) = x62*(-magneticFieldOneForm.get(1)*(magneticField.get(1)*x54 - spatialVelocity.get(1)*x53 + unitVector.get(1)*x46 - x16*x66 - x57*(magneticField.get(1)*x55 + x63) + x59*x66) - normalMagneticField*x46 + x64*x92);
  characteristic_matrix->get(1, 5) = x62*(-magneticFieldOneForm.get(1)*(magneticField.get(2)*x54 - spatialVelocity.get(2)*x53 + unitVector.get(2)*x46 - x16*x71 - x57*(magneticField.get(2)*x55 + x68) + x59*x71) + x69*x92);
  characteristic_matrix->get(1, 6) = x62*(magneticFieldOneForm.get(1)*x18*x75 + x14*(spatialVelocityOneForm.get(1)*x33*x75 - unitNormal.get(1)*x0*(x48*x81 - x75)))/(x48*x81);
  characteristic_matrix->get(1, 7) = x1*x19*(magneticFieldOneForm.get(1)*x0*x18*x7 + spatialVelocityOneForm.get(1)*x33*x86 - unitNormal.get(1)*x0*(restMassDensity*x87 - x86))/x87;
  characteristic_matrix->get(1, 8) = 0;
  characteristic_matrix->get(2, 0) = x1*(magneticFieldOneForm.get(2)*(spatialVelocity.get(0)*x3 + x13*x27 - x2) + spatialVelocityOneForm.get(2)*x29 + x13*x26*x97);
  characteristic_matrix->get(2, 1) = x1*(magneticFieldOneForm.get(2)*(spatialVelocity.get(1)*x3 + x27*x39 - x40) + spatialVelocityOneForm.get(2)*x38 + x26*x39*x97);
  characteristic_matrix->get(2, 2) = x1*(magneticFieldOneForm.get(2)*(spatialVelocity.get(2)*x3 + x27*x43 - x44) + spatialVelocityOneForm.get(2)*x42 + x26*x43*x97 + x37);
  characteristic_matrix->get(2, 3) = x62*(-magneticFieldOneForm.get(2)*(magneticField.get(0)*x54 - spatialVelocity.get(0)*x53 + unitVector.get(0)*x46 - x16*x58 - x57*(magneticField.get(0)*x55 + x50) + x58*x59) + x52*x97);
  characteristic_matrix->get(2, 4) = x62*(-magneticFieldOneForm.get(2)*(magneticField.get(1)*x54 - spatialVelocity.get(1)*x53 + unitVector.get(1)*x46 - x16*x66 - x57*(magneticField.get(1)*x55 + x63) + x59*x66) + x64*x97);
  characteristic_matrix->get(2, 5) = x62*(-magneticFieldOneForm.get(2)*(magneticField.get(2)*x54 - spatialVelocity.get(2)*x53 + unitVector.get(2)*x46 - x16*x71 - x57*(magneticField.get(2)*x55 + x68) + x59*x71) - normalMagneticField*x46 + x69*x97);
  characteristic_matrix->get(2, 6) = x62*(magneticFieldOneForm.get(2)*x18*x75 + x14*(spatialVelocityOneForm.get(2)*x33*x75 - unitNormal.get(2)*x0*(x48*x81 - x75)))/(x48*x81);
  characteristic_matrix->get(2, 7) = x1*x19*(magneticFieldOneForm.get(2)*x0*x18*x7 + spatialVelocityOneForm.get(2)*x33*x86 - unitNormal.get(2)*x0*(restMassDensity*x87 - x86))/x87;
  characteristic_matrix->get(2, 8) = 0;
  characteristic_matrix->get(3, 0) = x1*(magneticFieldOneForm.get(0)*(-normalVelocity*x101 + unitVector.get(0)) + normalMagneticField*(spatialVelocityOneForm.get(0)*x101 - 1));
  characteristic_matrix->get(3, 1) = x1*(magneticFieldOneForm.get(0)*(unitVector.get(1) - x105*x39) + normalMagneticField*spatialVelocityOneForm.get(0)*x104);
  characteristic_matrix->get(3, 2) = x1*(magneticFieldOneForm.get(0)*(unitVector.get(2) - x105*x43) + normalMagneticField*spatialVelocityOneForm.get(0)*x107);
  characteristic_matrix->get(3, 3) = x1*(magneticFieldOneForm.get(0)*(-normalVelocity*x52 + x2) + spatialVelocityOneForm.get(0)*(normalMagneticField*x52 - x29) + x37);
  characteristic_matrix->get(3, 4) = x1*(magneticFieldOneForm.get(0)*(-normalVelocity*x64 + x40) + spatialVelocityOneForm.get(0)*(normalMagneticField*x64 - x38));
  characteristic_matrix->get(3, 5) = x1*(magneticFieldOneForm.get(0)*(-normalVelocity*x69 + x44) + spatialVelocityOneForm.get(0)*(normalMagneticField*x69 - x42));
  characteristic_matrix->get(3, 6) = x116*(Zvar*(-kappa + x9) + restMassDensity*x7*x73)*(magneticFieldOneForm.get(0)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(0));
  characteristic_matrix->get(3, 7) = x100*x7*(magneticFieldOneForm.get(0)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(0));
  characteristic_matrix->get(3, 8) = unitNormal.get(0);
  characteristic_matrix->get(4, 0) = x1*(magneticFieldOneForm.get(1)*(-normalVelocity*x101 + unitVector.get(0)) + normalMagneticField*spatialVelocityOneForm.get(1)*x101);
  characteristic_matrix->get(4, 1) = x1*(magneticFieldOneForm.get(1)*(unitVector.get(1) - x105*x39) + normalMagneticField*(spatialVelocityOneForm.get(1)*x104 - 1));
  characteristic_matrix->get(4, 2) = x1*(magneticFieldOneForm.get(1)*(unitVector.get(2) - x105*x43) + normalMagneticField*spatialVelocityOneForm.get(1)*x107);
  characteristic_matrix->get(4, 3) = x1*(magneticFieldOneForm.get(1)*(-normalVelocity*x52 + x2) + spatialVelocityOneForm.get(1)*(normalMagneticField*x52 - x29));
  characteristic_matrix->get(4, 4) = x1*(magneticFieldOneForm.get(1)*(-normalVelocity*x64 + x40) + spatialVelocityOneForm.get(1)*(normalMagneticField*x64 - x38) + x37);
  characteristic_matrix->get(4, 5) = x1*(magneticFieldOneForm.get(1)*(-normalVelocity*x69 + x44) + spatialVelocityOneForm.get(1)*(normalMagneticField*x69 - x42));
  characteristic_matrix->get(4, 6) = x116*(Zvar*(-kappa + x9) + restMassDensity*x7*x73)*(magneticFieldOneForm.get(1)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(1));
  characteristic_matrix->get(4, 7) = x100*x7*(magneticFieldOneForm.get(1)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(1));
  characteristic_matrix->get(4, 8) = unitNormal.get(1);
  characteristic_matrix->get(5, 0) = x1*(magneticFieldOneForm.get(2)*(-normalVelocity*x101 + unitVector.get(0)) + normalMagneticField*spatialVelocityOneForm.get(2)*x101);
  characteristic_matrix->get(5, 1) = x1*(magneticFieldOneForm.get(2)*(unitVector.get(1) - x105*x39) + normalMagneticField*spatialVelocityOneForm.get(2)*x104);
  characteristic_matrix->get(5, 2) = x1*(magneticFieldOneForm.get(2)*(unitVector.get(2) - x105*x43) + normalMagneticField*(spatialVelocityOneForm.get(2)*x107 - 1));
  characteristic_matrix->get(5, 3) = x1*(magneticFieldOneForm.get(2)*(-normalVelocity*x52 + x2) + spatialVelocityOneForm.get(2)*(normalMagneticField*x52 - x29));
  characteristic_matrix->get(5, 4) = x1*(magneticFieldOneForm.get(2)*(-normalVelocity*x64 + x40) + spatialVelocityOneForm.get(2)*(normalMagneticField*x64 - x38));
  characteristic_matrix->get(5, 5) = x1*(magneticFieldOneForm.get(2)*(-normalVelocity*x69 + x44) + spatialVelocityOneForm.get(2)*(normalMagneticField*x69 - x42) + x37);
  characteristic_matrix->get(5, 6) = x116*(Zvar*(-kappa + x9) + restMassDensity*x7*x73)*(magneticFieldOneForm.get(2)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(2));
  characteristic_matrix->get(5, 7) = x100*x7*(magneticFieldOneForm.get(2)*normalVelocity - normalMagneticField*spatialVelocityOneForm.get(2));
  characteristic_matrix->get(5, 8) = unitNormal.get(2);
  characteristic_matrix->get(6, 0) = x130*(Zvar*unitVector.get(0) + magneticField.get(0)*normalMagneticField - x101*x126);
  characteristic_matrix->get(6, 1) = x130*(Zvar*unitVector.get(1) + magneticField.get(1)*normalMagneticField - x104*x126);
  characteristic_matrix->get(6, 2) = x130*(Zvar*unitVector.get(2) + magneticField.get(2)*normalMagneticField - x107*x126);
  characteristic_matrix->get(6, 3) = x130*(Zvar*normalMagneticField*spatialVelocity.get(0) + Zvar*x2 + magneticFieldSquared*normalMagneticField*spatialVelocity.get(0) - magneticField.get(0)*x141 - x125*x58 - x31*x58);
  characteristic_matrix->get(6, 4) = x130*(Zvar*normalMagneticField*spatialVelocity.get(1) + Zvar*x40 + magneticFieldSquared*normalMagneticField*spatialVelocity.get(1) - magneticField.get(1)*x141 - x125*x66 - x31*x66);
  characteristic_matrix->get(6, 5) = x130*(Zvar*normalMagneticField*spatialVelocity.get(2) + Zvar*x44 + magneticFieldSquared*normalMagneticField*spatialVelocity.get(2) - magneticField.get(2)*x141 - x125*x71 - x31*x71);
  characteristic_matrix->get(6, 6) = x129*(Dvar*x116*x31*x75 + x125*(Dvar*x116*x75 + x0));
  characteristic_matrix->get(6, 7) = Dvar*x126*x19*x7/(Zvar*(Zvar*(-soundSpeedSquared/x14 + x4) - comovingMagneticFieldSquared + soundSpeedSquared*x23));
  characteristic_matrix->get(6, 8) = 0;
  characteristic_matrix->get(7, 0) = x129*(-Dvar*magneticField.get(0)*normalMagneticField + Dvar*x101*x126 + Zvar*unitVector.get(0)*(-Dvar + x0));
  characteristic_matrix->get(7, 1) = x129*(-Dvar*magneticField.get(1)*normalMagneticField + Dvar*x104*x126 + Zvar*unitVector.get(1)*(-Dvar + x0));
  characteristic_matrix->get(7, 2) = x129*(-Dvar*magneticField.get(2)*normalMagneticField + Dvar*x107*x126 + Zvar*unitVector.get(2)*(-Dvar + x0));
  characteristic_matrix->get(7, 3) = x130*(-Zvar*normalMagneticField*spatialVelocity.get(0) - Zvar*x2 - magneticFieldSquared*normalMagneticField*spatialVelocity.get(0) + magneticField.get(0)*x141 + x125*x58 + x31*x58);
  characteristic_matrix->get(7, 4) = x130*(-Zvar*normalMagneticField*spatialVelocity.get(1) - Zvar*x40 - magneticFieldSquared*normalMagneticField*spatialVelocity.get(1) + magneticField.get(1)*x141 + x125*x66 + x31*x66);
  characteristic_matrix->get(7, 5) = x130*(-Zvar*normalMagneticField*spatialVelocity.get(2) - Zvar*x44 - magneticFieldSquared*normalMagneticField*spatialVelocity.get(2) + magneticField.get(2)*x141 + x125*x71 + x31*x71);
  characteristic_matrix->get(7, 6) = -x129*(Dvar*x116*x31*x75 + x125*(Dvar*x116*x75 + x0));
  characteristic_matrix->get(7, 7) = -Dvar*x126*x19*x7/(Zvar*(Zvar*(-soundSpeedSquared/x14 + x4) - comovingMagneticFieldSquared + soundSpeedSquared*x23));
  characteristic_matrix->get(7, 8) = 0;
  characteristic_matrix->get(8, 0) = 0;
  characteristic_matrix->get(8, 1) = 0;
  characteristic_matrix->get(8, 2) = 0;
  characteristic_matrix->get(8, 3) = unitVector.get(0);
  characteristic_matrix->get(8, 4) = unitVector.get(1);
  characteristic_matrix->get(8, 5) = unitVector.get(2);
  characteristic_matrix->get(8, 6) = 0;
  characteristic_matrix->get(8, 7) = 0;
  characteristic_matrix->get(8, 8) = 0;
}

template <size_t MatrixSize, size_t ThermodynamicDim>
void numerical_characteristics(
    const gsl::not_null<tnsr::i<DataVector, MatrixSize>*> characteristic_speeds,
    const gsl::not_null<tnsr::ij<DataVector, MatrixSize>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, MatrixSize>*>
        characteristic_projectors,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  static_assert(MatrixSize == 6 or MatrixSize == 9,
                "numerical_characteristics only supports MatrixSize 6 (hydro) "
                "or 9 (MHD).");
  // Build characteristic matrix.  The system is selected by MatrixSize: 6 uses
  // the hydro+Ye flux Jacobian, 9 uses the full MHD one (which also needs the
  // magnetic field).
  Variables<tmpl::list<::Tags::TempiJ<0, MatrixSize>>> temp_tensors{
      get<0, 0>(spatial_metric).size()};
  tnsr::iJ<DataVector, MatrixSize>& characteristic_matrix =
      get<::Tags::TempiJ<0, MatrixSize>>(temp_tensors);
  if constexpr (MatrixSize == 6) {
    flux_jacobian_hydro(make_not_null(&characteristic_matrix), spatial_velocity,
                        rest_mass_density, specific_internal_energy,
                        electron_fraction,
                        /* other helpful quantities */
                        lorentz_factor, specific_enthalpy, spatial_metric,
                        inv_spatial_metric, unit_normal, equation_of_state);
  } else {
    flux_jacobian_mhd(make_not_null(&characteristic_matrix), spatial_velocity,
                      magnetic_field, rest_mass_density,
                      specific_internal_energy, electron_fraction,
                      /* other helpful quantities */
                      lorentz_factor, specific_enthalpy, spatial_metric,
                      inv_spatial_metric, unit_normal, equation_of_state);
  }

  // Allocate memory to work with blaze::geev (outside of loop to save
  // time/memory)
  constexpr size_t matrix_size = MatrixSize;
  blaze::StaticMatrix<double, matrix_size, matrix_size> blaze_point_matrix{};
  blaze::StaticVector<blaze::complex<double>, matrix_size>
      blaze_complex_eigenvalues{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_L{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_R{};
  std::array<double, matrix_size> blaze_real_eigenvalues{};

  // Loop over each grid point
  const size_t num_points = get<0, 0>(spatial_metric).size();
  for (size_t point = 0; point < num_points; ++point) {
    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        blaze_point_matrix(row, col) =
            characteristic_matrix.get(row, col)[point];
      }
    }

    // Solve eigensystem using blaze:geev
    blaze::geev(blaze_point_matrix, blaze_complex_L, blaze_complex_eigenvalues,
                blaze_complex_R);
    const double tolerance = 1.0e-12;
    for (size_t i = 0; i < matrix_size; ++i) {
      ASSERT(std::abs(blaze_complex_eigenvalues.at(i).imag()) < tolerance,
             "Complex eigenvalue: "
                 << blaze_complex_eigenvalues.at(i).real() << " + "
                 << blaze_complex_eigenvalues.at(i).imag() << " i.");
      gsl::at(blaze_real_eigenvalues, i) =
          blaze_complex_eigenvalues.at(i).real();
    }

// We found cases in which blaze::geev returns a wrong eigenvalue. As a sanity
// check, we use GSL to compute the eigenvalues for the same matrix and check
// that they are the same.
#ifdef SPECTRE_DEBUG
    // Allocate memory to work with GSL:
    // Workspace for computing eigenvalues and eigenvectors
    gsl_eigen_nonsymm_workspace* gsl_workspace =
        gsl_eigen_nonsymm_alloc(matrix_size);
    // Containers for the right eigensystem (A * R = lambda * R)
    gsl_matrix* gsl_point_matrix = gsl_matrix_alloc(matrix_size, matrix_size);
    gsl_vector_complex* gsl_complex_eigenvalues =
        gsl_vector_complex_alloc(matrix_size);
    gsl_matrix_complex* gsl_complex_right_eigenvectors =
        gsl_matrix_complex_alloc(matrix_size, matrix_size);
    std::array<double, matrix_size> gsl_real_eigenvalues{};

    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        const double entry = characteristic_matrix.get(row, col)[point];
        gsl_matrix_set(gsl_point_matrix, row, col, entry);
      }
    }

    // Solve for eigenvalues using GSL
    gsl_eigen_nonsymm(gsl_point_matrix, gsl_complex_eigenvalues, gsl_workspace);
    for (size_t i = 0; i < matrix_size; ++i) {
      gsl::at(gsl_real_eigenvalues, i) =
          GSL_REAL(gsl_vector_complex_get(gsl_complex_eigenvalues, i));
    }

    // Sort both blaze's and GSL's eigenvalues to facilitate comparison
    std::array<double, matrix_size> sorted_blaze_real_eigenvalues =
        blaze_real_eigenvalues;
    std::sort(sorted_blaze_real_eigenvalues.begin(),
              sorted_blaze_real_eigenvalues.end());
    std::sort(gsl_real_eigenvalues.begin(), gsl_real_eigenvalues.end());

    // Compare eigenvalues
    for (size_t i = 0; i < matrix_size; ++i) {
      if (UNLIKELY(std::abs(gsl::at(sorted_blaze_real_eigenvalues, i) -
                            gsl::at(gsl_real_eigenvalues, i)) > 1.e-6)) {
        std::ostringstream matrix_stream;
        matrix_stream << "Point matrix:\n";
        for (size_t row = 0; row < matrix_size; ++row) {
          for (size_t col = 0; col < matrix_size; ++col) {
            matrix_stream << blaze_point_matrix(row, col) << " ";
          }
          matrix_stream << "\n";
        }
        matrix_stream << "Blaze eigenvalue: "
                      << gsl::at(sorted_blaze_real_eigenvalues, i)
                      << ", GSL eigenvalue: "
                      << gsl::at(gsl_real_eigenvalues, i) << "\n";

        ERROR(
            "The eigensolvers from blaze and GSL found different eigenvalues "
            "for the same matrix.\n"
            << matrix_stream.str());
      }
    }

    // Free GSL allocations
    gsl_eigen_nonsymm_free(gsl_workspace);
    gsl_matrix_free(gsl_point_matrix);
    gsl_vector_complex_free(gsl_complex_eigenvalues);
    gsl_matrix_complex_free(gsl_complex_right_eigenvectors);
#endif

    // Check and save results
    // Track which indices have been processed to avoid double-counting partners
    std::array<bool, matrix_size> processed_right_eigenvectors{};
    std::array<bool, matrix_size> processed_left_eigenvectors{};
    // Note: we're looping through the ith eigenvalue/vectors, not the ith row!
    for (size_t i = 0; i < matrix_size; ++i) {
      // Store eigenvalue
      characteristic_speeds->get(i)[point] = gsl::at(blaze_real_eigenvalues, i);

      // For each PAIR of degenerate eigenvalues, it is possible that GSL
      // returns a PAIR of complex eigenvectors that are complex conjugates of
      // each other. Here, we check if the ith right/left eigenvector is
      // complex.
      bool right_eigenvector_is_complex = false;
      bool left_eigenvector_is_complex = false;
      for (size_t k = 0; k < matrix_size; ++k) {
        if (std::abs(blaze_complex_R(k, i).imag()) > tolerance) {
          right_eigenvector_is_complex = true;
        }

        if (std::abs(blaze_complex_L(k, i).imag()) > tolerance) {
          left_eigenvector_is_complex = true;
        }
      }

      // If either eigenvector is complex, then we know that the vectors formed
      // by their real and imaginary parts are also linearly-independent
      // eigenvectors. Here, we use this fact to build real-valued left and
      // right eigenvectors.
      if (not gsl::at(processed_right_eigenvectors, i)) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (right_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_R(k, i);
              const auto conjugate_component = blaze_complex_R(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not right_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex right eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (right_eigenvector_is_complex) {
          ASSERT(
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith right eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_modes->get(i, k)[point] = blaze_complex_R(k, i).real();
          if (right_eigenvector_is_complex) {
            characteristic_modes->get(conjugate_i, k)[point] =
                blaze_complex_R(k, i).imag();
          }
        }
        gsl::at(processed_right_eigenvectors, i) = true;
        if (right_eigenvector_is_complex) {
          gsl::at(processed_right_eigenvectors, conjugate_i) = true;
        }
      }

      // Same as above, but for left eigenvectors
      if (not gsl::at(processed_left_eigenvectors, i)) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (left_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_L(k, i);
              const auto conjugate_component = blaze_complex_L(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not left_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex left eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (left_eigenvector_is_complex) {
          ASSERT(
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith left eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_projectors->get(i, k)[point] =
              blaze_complex_L(k, i).real();
          if (left_eigenvector_is_complex) {
            characteristic_projectors->get(conjugate_i, k)[point] =
                blaze_complex_L(k, i).imag();
          }
        }
        gsl::at(processed_left_eigenvectors, i) = true;
        if (left_eigenvector_is_complex) {
          gsl::at(processed_left_eigenvectors, conjugate_i) = true;
        }
      }
    }
  }
}

namespace Tags {

template <size_t ThermodynamicDim>
void CharacteristicSpeedsCompute::function(
    const gsl::not_null<return_type*> result,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& /* electron_fraction */,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  characteristic_speeds_approximate_mhd<ThermodynamicDim>(
      result, rest_mass_density, /*electron_fraction*/ {},
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FUNCTION_INSTANTIATION(r, data)                                     \
  template void CharacteristicSpeedsCompute::function<DIM(data)>(           \
      const gsl::not_null<return_type*> result,                             \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& electron_fraction,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, DIM(data)>&             \
          equation_of_state);

GENERATE_INSTANTIATIONS(FUNCTION_INSTANTIATION, (1, 2, 3))
#undef DIM
#undef FUNCTION_INSTANTIATION

}  // namespace Tags

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                                 \
  template std::array<DataVector, 9>                                           \
  characteristic_speeds_approximate_mhd<GET_DIM(data)>(                        \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_approximate_mhd<GET_DIM(data)>(          \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,             \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template tnsr::i<DataVector, 3> characteristic_speeds_hydro<GET_DIM(data)>(  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_hydro<GET_DIM(data)>(                    \
      const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_mhd<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 9>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state,                                                   \
      SlowMagnetosonicSpeedMethod slow_speed_method);                          \
  template void characteristic_eigenvectors_mhd<GET_DIM(data)>(                \
      const gsl::not_null<tnsr::ij<DataVector, 9>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 9>*> characteristic_projectors, \
      const tnsr::i<DataVector, 9>& characteristic_speeds,                     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state,                                                   \
      const bool skip_fluid_subspace);                                         \
  template void characteristic_eigenvectors_hydro<GET_DIM(data)>(              \
      const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors, \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void flux_jacobian_hydro<GET_DIM(data)>(                            \
      const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void flux_jacobian_mhd<GET_DIM(data)>(                              \
      const gsl::not_null<tnsr::iJ<DataVector, 9>*> characteristic_matrix,     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION

// numerical_characteristics is instantiated over both the matrix size
// (6 = hydro+Ye, 9 = MHD) and the thermodynamic dimension.
#define GET_SIZE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define NUMERICAL_CHARACTERISTICS_INSTANTIATION(r, data)                       \
  template void numerical_characteristics<GET_SIZE(data), GET_DIM(data)>(      \
      const gsl::not_null<tnsr::i<DataVector, GET_SIZE(data)>*>                 \
          characteristic_speeds,                                               \
      const gsl::not_null<tnsr::ij<DataVector, GET_SIZE(data)>*>               \
          characteristic_modes,                                                \
      const gsl::not_null<tnsr::IJ<DataVector, GET_SIZE(data)>*>               \
          characteristic_projectors,                                           \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);

GENERATE_INSTANTIATIONS(NUMERICAL_CHARACTERISTICS_INSTANTIATION, (6, 9),
                        (1, 2, 3))

#undef GET_DIM
#undef GET_SIZE
#undef NUMERICAL_CHARACTERISTICS_INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
