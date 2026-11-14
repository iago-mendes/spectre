// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindRealEigensystem.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

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

// Compute the sound speed squared from the equation of state.
template <size_t ThermodynamicDim>
void sound_speed_squared_from_eos(
    const gsl::not_null<Scalar<DataVector>*> sound_speed_squared,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  if constexpr (ThermodynamicDim == 1) {
    get(*sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(*sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(*sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(*sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(*sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }
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
  const double noise =
      8.0 * std::numeric_limits<double>::epsilon() *
      std::max({1.0, std::abs(c0), std::abs(c1), std::abs(c2), std::abs(c3)});
  // A genuine multiple root, or a derivative so small that noise/derivative
  // (and the slack summed from two of them) would overflow, e.g. the slow
  // roots of a subnormal B_n: the ordering carries no information at all.
  if (not(derivative > 1.0e-300 * noise)) {
    return std::numeric_limits<double>::infinity();
  }
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
// (papers/2010-anton-et-al-eigenvectors), and is also the order of the
// `MhdSpeed` labels in Characteristics.hpp. NOTE THE ORDER:
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
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

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
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

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
      ::Tags::TempScalar<13>, ::Tags::TempScalar<14>, ::Tags::TempScalar<15>,
      ::Tags::TempI<0, 3>, ::Tags::Tempi<0, 3>, ::Tags::TempI<1, 3>>>
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
    get(pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
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
    // Note: the 2D EoS interface only provides kappa p / rho^2, so we need a
    //       positive pressure to recover kappa.
    if (UNLIKELY(min(get(pressure)) <= 0.0)) {
      ERROR(
          "flux_jacobian_hydro with a 2D equation of state needs a positive "
          "pressure, but the minimum pressure is "
          << min(get(pressure)) << ".");
    }
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    Scalar<DataVector>& temperature = get<::Tags::TempScalar<2>>(temp_tensors);
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(pressure) = get(equation_of_state.pressure_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(kappa) = get(equation_of_state.kappa_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(zeta) = get(equation_of_state.zeta_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
  }

  // The expressions in this function are written to:
  //   1. Avoid catastrophic cancellations by rewriting terms like W - 1 as,
  //      e.g., v^2 W^2 / (W + 1).
  //   2. Avoid repeated calculations by storing intermediate results in
  //      temporary variables and reusing previous allocations when possible.
  //   3. Be fast, as measured by the benchmark in the unit test.

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
  // Note: h - 1 is computed as eps + p / rho instead of from specific_enthalpy,
  //       so the two must be consistent, h = 1 + eps + p / rho.
  Scalar<DataVector>& specific_enthalpy_minus_one =
      get<::Tags::TempScalar<9>>(temp_tensors);
  get(specific_enthalpy_minus_one) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  // Check that the specific enthalpy is consistent with the equation of state.
  constexpr double enthalpy_tolerance = 1.0e-8;
  ASSERT(
      max(abs(get(specific_enthalpy) - 1.0 - get(specific_enthalpy_minus_one)) /
          get(specific_enthalpy)) <= enthalpy_tolerance,
      "The specific enthalpy is inconsistent with the equation of state: "
      "the largest relative difference between h and 1 + eps + p / rho is "
          << max(abs(get(specific_enthalpy) - 1.0 -
                     get(specific_enthalpy_minus_one)) /
                 get(specific_enthalpy))
          << ".");
  Scalar<DataVector>& velocity_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  tenex::evaluate(make_not_null(&velocity_squared),
                  spatial_velocity(ti::I) * spatial_velocity_one_form(ti::i));
  Scalar<DataVector>& lorentz_factor_squared =
      get<::Tags::TempScalar<11>>(temp_tensors);
  get(lorentz_factor_squared) = square(get(lorentz_factor));
  Scalar<DataVector>& lorentz_factor_minus_one =
      get<::Tags::TempScalar<12>>(temp_tensors);
  get(lorentz_factor_minus_one) = get(velocity_squared) *
                                  get(lorentz_factor_squared) /
                                  (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& specific_enthalpy_minus_lorentz_factor =
      get<::Tags::TempScalar<13>>(temp_tensors);
  get(specific_enthalpy_minus_lorentz_factor) =
      get(specific_enthalpy_minus_one) - get(lorentz_factor_minus_one);

  // Derivatives of Z = rho h W^2 with respect to the conserved variables
  // (D, S_i, tau, D Y_e), where S_i = Z v_i
  Scalar<DataVector>& inv_derivative_denom =
      get<::Tags::TempScalar<7>>(temp_tensors);
  get(inv_derivative_denom) =
      1.0 / ((-get(lorentz_factor_squared) + get(sound_speed_squared) *
                                                 get(velocity_squared) *
                                                 get(lorentz_factor_squared)) *
             get(rest_mass_density));
  Scalar<DataVector>& dz_d_density = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dz_d_density),
      -(lorentz_factor() *
        (-kappa() * specific_enthalpy_minus_lorentz_factor() -
         zeta() * electron_fraction() +
         (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
             rest_mass_density()) *
        inv_derivative_denom()));
  tnsr::I<DataVector, 3>& dz_d_momentum =
      get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dz_d_momentum),
      (spatial_velocity(ti::I) * lorentz_factor_squared() *
       (kappa() + sound_speed_squared() * rest_mass_density()) *
       inv_derivative_denom()));
  Scalar<DataVector>& dz_d_energy = get<::Tags::TempScalar<5>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_energy),
                  -(lorentz_factor_squared() * (kappa() + rest_mass_density()) *
                    inv_derivative_denom()));
  // Note: dZ/dtau - 1 is computed directly to avoid cancellation, since
  //       dZ/dtau = 1 at rest when kappa = 0.
  Scalar<DataVector>& dz_d_energy_minus_one =
      get<::Tags::TempScalar<14>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_energy_minus_one),
                  -(lorentz_factor_squared() *
                    (kappa() + sound_speed_squared() * rest_mass_density() *
                                   velocity_squared()) *
                    inv_derivative_denom()));
  Scalar<DataVector>& dz_d_electron_fraction =
      get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_electron_fraction),
                  -(zeta() * lorentz_factor() * inv_derivative_denom()));

  // Common factors used repeatedly in the characteristic matrix entries
  Scalar<DataVector>& inv_enthalpy_times_lorentz_factor =
      get<::Tags::TempScalar<8>>(temp_tensors);
  get(inv_enthalpy_times_lorentz_factor) =
      1.0 / (get(specific_enthalpy) * get(lorentz_factor));
  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      (1.0 - get(inv_enthalpy_times_lorentz_factor) * get(dz_d_density)) *
      get(normal_velocity);
  characteristic_matrix->get(0, 1) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(0) - dz_d_momentum.get(0) * get(normal_velocity));
  characteristic_matrix->get(0, 2) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(1) - dz_d_momentum.get(1) * get(normal_velocity));
  characteristic_matrix->get(0, 3) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(2) - dz_d_momentum.get(2) * get(normal_velocity));
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
  characteristic_matrix->get(0, 4) = -(get(inv_enthalpy_times_lorentz_factor) *
                                       get(dz_d_energy) * get(normal_velocity));
  characteristic_matrix->get(0, 5) =
      -(get(inv_enthalpy_times_lorentz_factor) * get(dz_d_electron_fraction) *
        get(normal_velocity));
  // Momentum rows
  Scalar<DataVector>& normal_minus_velocity_term =
      get<::Tags::TempScalar<15>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    // n_i - v_n v_i, shared by columns 1 to 3 and 5
    get(normal_minus_velocity_term) =
        unit_normal.get(i) -
        get(normal_velocity) * spatial_velocity_one_form.get(i);
    characteristic_matrix->get(i + 1, 0) =
        (-1.0 + get(dz_d_density)) * unit_normal.get(i) -
        get(dz_d_density) * get(normal_velocity) *
            spatial_velocity_one_form.get(i);
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        characteristic_matrix->get(i + 1, j + 1) =
            get(normal_velocity) +
            unit_vector.get(j) * spatial_velocity_one_form.get(i) +
            dz_d_momentum.get(j) * get(normal_minus_velocity_term);
      } else {
        characteristic_matrix->get(i + 1, j + 1) =
            unit_vector.get(j) * spatial_velocity_one_form.get(i) +
            dz_d_momentum.get(j) * get(normal_minus_velocity_term);
      }
    }
    // Note: this entry does not use n_i - v_n v_i, since
    //       -n_i + dZ/dtau (n_i - v_n v_i) would cancel at rest, where
    //       dZ/dtau = 1.
    characteristic_matrix->get(i + 1, 4) =
        get(dz_d_energy_minus_one) * unit_normal.get(i) -
        get(dz_d_energy) * get(normal_velocity) *
            spatial_velocity_one_form.get(i);
    characteristic_matrix->get(i + 1, 5) =
        get(dz_d_electron_fraction) * get(normal_minus_velocity_term);
  }
  characteristic_matrix->get(4, 0) = -characteristic_matrix->get(0, 0);
  characteristic_matrix->get(4, 4) = -characteristic_matrix->get(0, 4);
  characteristic_matrix->get(4, 5) = -characteristic_matrix->get(0, 5);
  characteristic_matrix->get(5, 0) = -get(electron_fraction) *
                                     get(inv_enthalpy_times_lorentz_factor) *
                                     get(dz_d_density) * get(normal_velocity);
  characteristic_matrix->get(5, 4) =
      get(electron_fraction) * characteristic_matrix->get(0, 4);
  characteristic_matrix->get(5, 5) =
      (1.0 - get(electron_fraction) * get(inv_enthalpy_times_lorentz_factor) *
                 get(dz_d_electron_fraction)) *
      get(normal_velocity);
}

template <size_t ThermodynamicDim>
void numerical_characteristics(
    const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,
    const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,
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
  // Build characteristic matrix.
  Variables<tmpl::list<::Tags::TempiJ<0, 6>>> temp_tensors{
      get<0, 0>(spatial_metric).size()};
  tnsr::iJ<DataVector, 6>& characteristic_matrix =
      get<::Tags::TempiJ<0, 6>>(temp_tensors);
  flux_jacobian_hydro(make_not_null(&characteristic_matrix), spatial_velocity,
                      rest_mass_density, specific_internal_energy,
                      electron_fraction,
                      /* other helpful quantities */
                      lorentz_factor, specific_enthalpy, spatial_metric,
                      inv_spatial_metric, unit_normal, equation_of_state);

  // Find the eigensystem of the characteristic matrix.
  find_real_eigensystem(characteristic_speeds, characteristic_modes,
                        characteristic_projectors, characteristic_matrix);
}

void magnetosonic_quartic_coefficients(
    gsl::not_null<tnsr::i<DataVector, 4>*> quartic_coefficients,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& normal_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& normal_magnetic_field,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
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
  const auto achievable_tolerance = [&c0, &c1, &c2, &c3, tolerance](
                                        const double x_point,
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
      << " at x = " << x[worst_point] << " (floor = "
      << achievable_tolerance(x[worst_point], worst_point) << ")");
}

template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    const gsl::not_null<tnsr::i<DataVector, 7>*> characteristic_speeds,

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
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_points) {
    for (size_t i = 0; i < 7; ++i) {
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
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

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
            << cs2 << " at point " << point << ", rest-mass density = "
            << get(rest_mass_density)[point] << ", specific internal energy = "
            << get(specific_internal_energy)[point]
            << ", specific enthalpy = " << get(specific_enthalpy)[point]
            << ". Either the equation of state is acausal here, or the "
               "caller passed a state that is not consistent with it.");
    }
  }

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
  get(comoving_magnetic_field_squared) *= get(inv_rho_h);
  tnsr::i<DataVector, 4>& quartic_coefficients =
      get<::Tags::Tempi<0, 4>>(temp_tensors);
  magnetosonic_quartic_coefficients(
      make_not_null(&quartic_coefficients), sound_speed_squared,
      normal_velocity, lorentz_factor, normal_magnetic_field,
      magnetic_field_dot_spatial_velocity, comoving_magnetic_field_squared);

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
    const double alfven_plus_i = alfven_plus[point];

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
    ASSERT(
        discriminant_i >= -discriminant_tolerance,
        "Failed to compute slow magnetosonic speeds: reduced quadratic "
        "has negative discriminant below tolerance. discriminant = "
            << discriminant_i << ", tolerance = " << discriminant_tolerance
            << ", point = " << point << ". State at this point: cs^2 = "
            << get(sound_speed_squared)[point] << ", v_n = " << vn_i
            << ", W = " << get(lorentz_factor)[point] << ", B_n/sqrt(rho h) = "
            << get(normal_magnetic_field)[point] << ", (B.v)/sqrt(rho h) = "
            << get(magnetic_field_dot_spatial_velocity)[point]
            << ", b^2/(rho h) = " << get(comoving_magnetic_field_squared)[point]
            << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
            << c0[point] << ", c1 = " << c1[point] << ", c2 = " << c2[point]
            << ", c3 = " << c3[point]
            << ". Fast roots returned by Newton: fast_minus = "
            << fast_minus[point] << ", fast_plus = " << fast_plus[point]
            << " (quartic there: " << evaluate_quartic(fast_minus[point], point)
            << ", " << evaluate_quartic(fast_plus[point], point)
            << "); Alfven speeds " << alfven_minus_i << ", " << alfven_plus_i
            << ".");
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
    ASSERT(
        discriminant_i <= 4.0 + discriminant_tolerance,
        "Failed to compute slow magnetosonic speeds: reduced quadratic "
        "has a discriminant above 4, which no valid deflation can "
        "produce. discriminant = "
            << discriminant_i << ", tolerance = " << discriminant_tolerance
            << ", point = " << point << ". State at this point: cs^2 = "
            << get(sound_speed_squared)[point] << ", v_n = " << vn_i
            << ", W = " << get(lorentz_factor)[point] << ", B_n/sqrt(rho h) = "
            << get(normal_magnetic_field)[point] << ", (B.v)/sqrt(rho h) = "
            << get(magnetic_field_dot_spatial_velocity)[point]
            << ", b^2/(rho h) = " << get(comoving_magnetic_field_squared)[point]
            << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
            << c0[point] << ", c1 = " << c1[point] << ", c2 = " << c2[point]
            << ", c3 = " << c3[point]
            << ". Fast roots returned by Newton: fast_minus = "
            << fast_minus[point] << ", fast_plus = " << fast_plus[point]
            << " (quartic there: " << evaluate_quartic(fast_minus[point], point)
            << ", " << evaluate_quartic(fast_plus[point], point)
            << "); Alfven speeds " << alfven_minus_i << ", " << alfven_plus_i
            << ".");
    // Round-off can leave a valid near-degenerate deflation slightly
    // negative (the measured floor over admissible states is -6.2e-7), so
    // floor it at zero before the square root. A value outside the band
    // itself is not repaired: the ASSERTs above report it.
    discriminant_i = std::max(discriminant_i, 0.0);

    // The cancellation-safe root of y^2 + b y + c = 0.
    const double q_i =
        -0.5 * (b_i + (b_i >= 0.0 ? 1.0 : -1.0) * sqrt(discriminant_i));

    slow_plus[point] = q_i;
    slow_minus[point] =
        -c3[point] - fast_minus[point] - fast_plus[point] - slow_plus[point];
    if (slow_minus[point] > slow_plus[point]) {
      std::swap(slow_minus[point], slow_plus[point]);
    }

    // ---- slow pair: re-solve in the comoving variable mu = lambda - v_n ----
    // The deflation above inherits the fast roots' error, amplified by
    // 1/(slow_plus - slow_minus), which is large near the slow/entropy
    // degeneracy (B_n -> 0). Re-solve each slow root on the magnetosonic
    // quartic in mu (Anton et al. 2010, N_4 times c_s^2/(rho h)):
    //   F(mu) = c_s^2 Bc^2 - (c_s^2 + b^2) W^2 mu^2 + (1 - c_s^2) W^4 mu^4/G,
    //   Bc = B_n/W - (B.v) W mu,   G = 1 - (v_n + mu)^2.
    // F has no O(1) cancellation near mu = 0. Seeds: the closed form that
    // zeroes its first two terms (exact as B_n -> 0) and the deflation result.
    // The seed with the smaller Newton step goes first; a root is accepted
    // only if Newton converged inside the slot's interlacing interval, and
    // otherwise the deflation result is kept.
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
      const double t1 = cs2_i * square(bc);
      const double t2 = mu2_coef * mu2;
      const double t3 = mu4_coef * square(mu2) * inv_g;
      const double f = t1 - t2 + t3;
      const double df =
          -2.0 * cs2_i * bc * bv_w - 2.0 * mu2_coef * mu +
          mu4_coef * mu2 * inv_g * (4.0 * mu + 2.0 * mu2 * lam * inv_g);
      // at a root to working precision (e.g. B_n = 0, mu = 0, or next to a
      // slow/Alfven near-double root, where the step test never passes)
      if (std::abs(f) <= 8.0 * std::numeric_limits<double>::epsilon() *
                             (std::abs(t1) + std::abs(t2) + std::abs(t3))) {
        return 0.0;
      }
      // a stationary point off a root gives no usable step (e.g. a
      // deflation that returned slow_minus = slow_plus = v_n with B.v = 0)
      return df == 0.0 ? std::numeric_limits<double>::infinity() : f / df;
    };
    // F is evaluated only inside the light cone: outside it G = 1 - lambda^2
    // <= 0 and the evaluation can divide by zero or overflow (a Newton
    // iterate from a poor deflation seed can leave it).
    const auto inside_light_cone = [&vn_i](const double mu) {
      return std::abs(vn_i + mu) < 1.0;
    };
    const double cs_i = sqrt(cs2_i);
    const double t_i = sqrt(cs2_i + b2_i);
    const double num = cs_i * bn_i / w2_i;
    // cs (B.v) -/+ t vanishes only if t = 0, i.e. c_s^2 = b^2 = 0, where
    // every root is v_n
    const double seed_1 = t_i > 0.0 ? num / (cs_i * bv_i + t_i) : 0.0;
    const double seed_2 = t_i > 0.0 ? num / (cs_i * bv_i - t_i) : 0.0;
    const std::array<double, 2> closed{
        {std::min(seed_1, seed_2), std::max(seed_1, seed_2)}};
    const std::array<double, 2> deflated{
        {slow_minus[point] - vn_i, slow_plus[point] - vn_i}};
    const double slack =
        4.0 * std::numeric_limits<double>::epsilon() *
        (std::abs(alfven_minus_i) + std::abs(alfven_plus_i) + std::abs(vn_i));
    const std::array<double, 2> lower{{alfven_minus_i - vn_i - slack, -slack}};
    const std::array<double, 2> upper{{slack, alfven_plus_i - vn_i + slack}};
    std::array<double, 2> polished{{slow_minus[point], slow_plus[point]}};
    for (size_t j = 0; j < 2; ++j) {
      const std::array<double, 2> seeds{
          {gsl::at(closed, j), gsl::at(deflated, j)}};
      const std::array<double, 2> first_steps{
          {inside_light_cone(seeds[0])
               ? newton_step(seeds[0])
               : std::numeric_limits<double>::infinity(),
           inside_light_cone(seeds[1])
               ? newton_step(seeds[1])
               : std::numeric_limits<double>::infinity()}};
      const std::array<size_t, 2> order =
          std::abs(first_steps[0]) <= std::abs(first_steps[1])
              ? std::array<size_t, 2>{{0, 1}}
              : std::array<size_t, 2>{{1, 0}};
      for (const size_t c : order) {
        double mu = gsl::at(seeds, c);
        double step = gsl::at(first_steps, c);
        // Accept a root only if Newton converged: the interval then
        // identifies it as this slot's slow root. An unconverged iterate
        // inside the interval can still miss the quartic's tolerance.
        bool converged = step == 0.0;
        for (size_t iter = 0; iter < 8 and step != 0.0 and std::isfinite(step);
             ++iter) {
          // never step out of the light cone: halve the step instead
          size_t halvings = 0;
          while (not inside_light_cone(mu - step) and halvings < 60) {
            step *= 0.5;
            ++halvings;
          }
          mu -= step;
          if (not inside_light_cone(mu)) {
            break;
          }
          if (halvings == 0 and
              std::abs(step) <=
                  4.0 * std::numeric_limits<double>::epsilon() * std::abs(mu)) {
            converged = true;
            break;
          }
          step = newton_step(mu);
          converged = step == 0.0;
        }
        if (converged and mu >= gsl::at(lower, j) and mu <= gsl::at(upper, j)) {
          // a subnormal offset (from a subnormal B_n) is v_n itself, as
          // for B_n = 0: keeps a = W (v_n - lambda) off the subnormal range
          gsl::at(polished, j) =
              vn_i +
              (std::abs(mu) < std::numeric_limits<double>::min() ? 0.0 : mu);
          break;
        }
      }
    }
    slow_minus[point] = std::min(polished[0], polished[1]);
    slow_plus[point] = std::max(polished[0], polished[1]);

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
    // papers/2010-anton-et-al-eigenvectors; the order of the `MhdSpeed`
    // labels in Characteristics.hpp). A fast root in a slow
    // slot breaks it by the fast/slow gap, which is many orders of magnitude
    // above the slack.
    //
    // Placed BEFORE the clamps, like its siblings, so a debug build reports
    // the raw values a broken deflation produced rather than repaired ones.
    ASSERT(
        interlacing_violation(fast_minus[point], alfven_minus[point],
                              slow_minus[point], vn_i, slow_plus[point],
                              alfven_plus[point], fast_plus[point]) <=
            interlacing_tolerance(c0[point], c1[point], c2[point], c3[point],
                                  slow_minus[point], slow_plus[point]),
        "Slow magnetosonic speeds solve the quartic but are not the SLOW "
        "roots: the characteristic speeds do not interlace. Required "
        "(Anton et al. 2010, Eq. 42): fast- <= alfven- <= slow- <= "
        "entropy <= slow+ <= alfven+ <= fast+, slack "
            << interlacing_tolerance(c0[point], c1[point], c2[point], c3[point],
                                     slow_minus[point], slow_plus[point])
            << " (four times the quartic's own root-location noise floor, "
               "which is what a cluster of nearly coincident speeds "
               "allows). Got fast- = "
            << fast_minus[point] << ", alfven- = " << alfven_minus[point]
            << ", slow- = " << slow_minus[point] << ", entropy = " << vn_i
            << ", slow+ = " << slow_plus[point]
            << ", alfven+ = " << alfven_plus[point]
            << ", fast+ = " << fast_plus[point] << "; worst inversion = "
            << interlacing_violation(fast_minus[point], alfven_minus[point],
                                     slow_minus[point], vn_i, slow_plus[point],
                                     alfven_plus[point], fast_plus[point])
            << ", point = " << point << ". State at this point: cs^2 = "
            << get(sound_speed_squared)[point] << ", v_n = " << vn_i
            << ", W = " << get(lorentz_factor)[point] << ", B_n/sqrt(rho h) = "
            << get(normal_magnetic_field)[point] << ", (B.v)/sqrt(rho h) = "
            << get(magnetic_field_dot_spatial_velocity)[point]
            << ", b^2/(rho h) = " << get(comoving_magnetic_field_squared)[point]
            << ". Quartic x^4 + c3 x^3 + c2 x^2 + c1 x + c0 with c0 = "
            << c0[point] << ", c1 = " << c1[point] << ", c2 = " << c2[point]
            << ", c3 = " << c3[point]
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
      const gsl::not_null<tnsr::i<DataVector, 7>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
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
  template void numerical_characteristics<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,      \
      const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors, \
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
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
