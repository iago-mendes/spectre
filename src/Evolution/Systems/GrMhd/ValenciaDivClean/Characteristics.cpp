// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <iostream>

#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void compute_characteristic_speeds(
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
}  // namespace

namespace grmhd::ValenciaDivClean {

namespace approx {

template <size_t ThermodynamicDim>
void characteristic_speeds(
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

  compute_characteristic_speeds(char_speeds, lapse, shift, spatial_velocity,
                                spatial_velocity_squared, sound_speed_squared,
                                alfven_speed_squared, unit_normal);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds(
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
  characteristic_speeds(make_not_null(&char_speeds), rest_mass_density,
                        electron_fraction, specific_internal_energy,
                        specific_enthalpy, spatial_velocity, lorentz_factor,
                        magnetic_field, lapse, shift, spatial_metric,
                        unit_normal, equation_of_state);
  return char_speeds;
}

}  // namespace approx

namespace detail {

/**
 * \note In the characteristic matrix $A_c^{\ b}$, the index $b$ labels columns,
 * while the index $c$ labels rows. This was chosen so that the right
 * eigenvectors, $A_b^{\ c} c^b = y x^c$ are column vectors, while the left
 * eigenvectors, $l_c A_b^{\ c} = x l_b$ are row vectors.
 */
void flux_jacobian_hydro(
    gsl::not_null<tnsr::iJ<DataVector, 5>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state,
    const Scalar<DataVector>& specific_enthalpy) {
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  get(sound_speed_squared) =
      (get(equation_of_state.chi_from_density_and_energy(
           rest_mass_density, specific_internal_energy)) +
       get(equation_of_state
               .kappa_times_p_over_rho_squared_from_density_and_energy(
                   rest_mass_density, specific_internal_energy))) /
      get(specific_enthalpy);

  const Scalar<DataVector> kappa_times_p_over_rho_squared =
      equation_of_state.kappa_times_p_over_rho_squared_from_density_and_energy(
          rest_mass_density, specific_internal_energy);
  const Scalar<DataVector> pressure =
      equation_of_state.pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy);
  const Scalar<DataVector> kappa =
      tenex::evaluate(kappa_times_p_over_rho_squared() / pressure() *
                      square(rest_mass_density()));
  //   get(kappa) *= 1.0 / get(pressure);
  //   get(kappa) *= get(square(rest_mass_density));

  // std::cout << "flux_jacobian_hydro 1" << std::endl;

  // Intermediate variables
  const auto Z = tenex::evaluate(rest_mass_density() * specific_enthalpy() *
                                 square(lorentz_factor()));
  const auto D = tenex::evaluate(rest_mass_density() * lorentz_factor());
  const auto normal_velocity =
      tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
  const auto unit_vector = tenex::evaluate<ti::I>(
      inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  const auto spatial_velocity_one_form = tenex::evaluate<ti::i>(
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));
  const auto mixed_spatial_metric = tenex::evaluate<ti::I, ti::j>(
      inv_spatial_metric(ti::I, ti::K) * spatial_metric(ti::k, ti::j));
  //   const auto dZdS = tenex::evaluate<ti::i>(
  //       -(kappa() + rest_mass_density() * sound_speed_squared()) *
  //       square(lorentz_factor()) / rest_mass_density() /
  //       (square(lorentz_factor()) -
  //        sound_speed_squared() * (square(lorentz_factor()) - 1.0)) *
  //       spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));
  //   const auto dZdD = tenex::evaluate(
  //       ((rest_mass_density() + kappa()) * square(lorentz_factor()) +
  //        specific_enthalpy() * lorentz_factor() *
  //            (rest_mass_density() * sound_speed_squared() - kappa())) /
  //       rest_mass_density() /
  //       (square(lorentz_factor()) -
  //        sound_speed_squared() * (square(lorentz_factor()) - 1.0)));
  //   const auto dZdtau = tenex::evaluate(
  //       (rest_mass_density() + kappa()) * square(lorentz_factor()) /
  //       rest_mass_density() /
  //       (square(lorentz_factor()) -
  //        sound_speed_squared() * (square(lorentz_factor()) - 1.0)));

  const auto dzdD = tenex::evaluate(
      (-(sound_speed_squared() * Z() * D()) +
       lorentz_factor() *
           (Z() * kappa() -
            D() * lorentz_factor() * (D() + kappa() * lorentz_factor()))) /
      (square(D()) *
       (-square(lorentz_factor()) +
        sound_speed_squared() * (-1.0 + square(lorentz_factor())))));
  const auto dzds = tenex::evaluate<ti::I>(
      (spatial_velocity(ti::I) * square(lorentz_factor()) *
       (sound_speed_squared() * D() + kappa() * lorentz_factor())) /
      (D() * (-square(lorentz_factor()) +
              sound_speed_squared() * (-1.0 + square(lorentz_factor())))));
  const auto dzdtau = tenex::evaluate(
      -((square(lorentz_factor()) * (D() + kappa() * lorentz_factor())) /
        (D() * (-square(lorentz_factor()) +
                sound_speed_squared() * (-1.0 + square(lorentz_factor()))))));

  // std::cout << "flux_jacobian_hydro 2" << std::endl;

  // characteristic_matrix->get(0, 0) =
  //   (get(Z) - get(D) * get(dZdD)) * get(normal_velocity) / get(Z);
  characteristic_matrix->get(0, 0) =
      ((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z);
  // std::cout << "flux_jacobian_hydro 3" << std::endl;
  // std::cout << "D = " << D << std::endl;
  // std::cout << "unit_normal = " << unit_normal << std::endl;
  // std::cout << "dZdS = " << dZdS << std::endl;
  // std::cout << "normal_velocity = " << normal_velocity << std::endl;
  // std::cout << "get(Z) = " << get(Z) << std::endl;
  for (size_t B = 0; B < 3; ++B) {
    // characteristic_matrix->get(B + 1, 0) =
    //     get(D) * (unit_normal.get(B) - dZdS.get(B) * get(normal_velocity)) /
    //     get(get(Z));
    characteristic_matrix->get(0, B + 1) =
        (get(D) * (unit_vector.get(B) - dzds.get(B) * get(normal_velocity))) /
        get(Z);
    // std::cout << "flux_jacobian_hydro 3." << B << std::endl;
  }
  // std::cout << "flux_jacobian_hydro 4" << std::endl;
  // characteristic_matrix->get(4, 0) =
  //     -get(D) * get(dZdtau) * get(normal_velocity) / get(get(Z));
  characteristic_matrix->get(0, 4) =
      -((get(D) * get(dzdtau) * get(normal_velocity)) / get(Z));
  // std::cout << "flux_jacobian_hydro 5" << std::endl;
  for (size_t c = 0; c < 3; ++c) {
    // characteristic_matrix->get(0, c + 1) =
    //     (-1.0 + get(dZdD)) * unit_vector.get(c) -
    //     get(dZdD) * get(normal_velocity) * spatial_velocity.get(c);
    // [[2,1]]
    characteristic_matrix->get(c + 1, 0) =
        (-1.0 + get(dzdD)) * unit_normal.get(c) -
        get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(c);
    // std::cout << "flux_jacobian_hydro 6" << std::endl;
    for (size_t B = 0; B < 3; ++B) {
      // characteristic_matrix->get(B + 1, c + 1) =
      //     spatial_metric.get(B, c) * get(normal_velocity) +
      //     unit_normal.get(B) * spatial_velocity.get(c) +
      //     dZdS.get(B) * (unit_vector.get(c) -
      //                    get(normal_velocity) * spatial_velocity.get(c));
      // [[2,2]]
      characteristic_matrix->get(c + 1, B + 1) =
          mixed_spatial_metric.get(B, c) * get(normal_velocity) +
          unit_vector.get(B) * spatial_velocity_one_form.get(c) +
          dzds.get(B) *
              (unit_normal.get(c) -
               get(normal_velocity) * spatial_velocity_one_form.get(c));
    }
    // std::cout << "flux_jacobian_hydro 7" << std::endl;
    // characteristic_matrix->get(4, c + 1) =
    //     (-1.0 + get(dZdtau)) * unit_vector.get(c) -
    //     get(dZdtau) * get(normal_velocity) * spatial_velocity.get(c);
    // [[2,3]]
    characteristic_matrix->get(c + 1, 4) =
        (-1.0 + get(dzdtau)) * unit_normal.get(c) -
        get(dzdtau) * get(normal_velocity) * spatial_velocity_one_form.get(c);
  }
  // std::cout << "flux_jacobian_hydro 8" << std::endl;
  // characteristic_matrix->get(0, 4) =
  //     -(get(get(Z)) - get(D) * get(dZdD)) * get(normal_velocity) /
  //     get(get(Z));
  characteristic_matrix->get(4, 0) =
      -(((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z));
  for (size_t B = 0; B < 3; ++B) {
    // std::cout << "flux_jacobian_hydro 9" << std::endl;
    // characteristic_matrix->get(B + 1, 4) =
    //     ((get(get(Z)) - get(D)) * unit_normal.get(B) +
    //      get(D) * dZdS.get(B) * get(normal_velocity)) /
    //     get(get(Z));
    characteristic_matrix->get(4, B + 1) =
        ((get(Z) - get(D)) * unit_vector.get(B) +
         get(D) * dzds.get(B) * get(normal_velocity)) /
        get(Z);
  }
  // std::cout << "flux_jacobian_hydro 10" << std::endl;
  // characteristic_matrix->get(4, 4) =
  //     get(D) * get(dZdtau) * get(normal_velocity) / get(get(Z));
  characteristic_matrix->get(4, 4) =
      (get(D) * get(dzdtau) * get(normal_velocity)) / get(Z);
}

}  // namespace detail

std::pair<std::array<DataVector, 5>,
          std::pair<std::array<tnsr::I<DataVector, 5>, 5>,
                    std::array<tnsr::I<DataVector, 5>, 5>>>
numerical_eigensystem(
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, 2>& equation_of_state,
    const Scalar<DataVector>& specific_enthalpy) {
  std::cout << "numerical_eigensystem:" << std::endl;

  // Define size of variables declared below
  const size_t num_points = get(rest_mass_density).size();
  constexpr size_t matrix_size = 5;

  // Initialize container for all eigenvalues and eigenvectors
  std::array<DataVector, matrix_size> all_eigenvalues;
  std::array<tnsr::I<DataVector, matrix_size>, matrix_size>
      all_right_eigenvectors;
  std::array<tnsr::I<DataVector, matrix_size>, matrix_size>
      all_left_eigenvectors;
  for (size_t i = 0; i < matrix_size; ++i) {
    all_eigenvalues[i].destructive_resize(num_points);
    for (size_t j = 0; j < matrix_size; ++j) {
      all_right_eigenvectors[i].get(j).destructive_resize(num_points);
      all_left_eigenvectors[i].get(j).destructive_resize(num_points);
    }
  }

  // std::cout << "Building characteristic matrix..." << std::endl;

  tnsr::iJ<DataVector, 5> characteristic_matrix =
      make_with_value<tnsr::iJ<DataVector, 5>>(spatial_metric, 0.0);
  detail::flux_jacobian_hydro(
      make_not_null(&characteristic_matrix), spatial_velocity,
      rest_mass_density, specific_internal_energy,
      /* other helpful quantities */
      lorentz_factor, spatial_metric, inv_spatial_metric, unit_normal,
      equation_of_state, specific_enthalpy);

  // std::cout << "... built characteristic matrix!" << std::endl;

  // Allocate memory to work with GSL (outside of loop to save time/memory):
  // Workspace for computing eigenvalues and eigenvectors
  gsl_eigen_nonsymmv_workspace* gsl_workspace =
      gsl_eigen_nonsymmv_alloc(matrix_size);
  // Containers for the right eigensystem (A * R = lambda * R)
  gsl_matrix* point_matrix = gsl_matrix_alloc(matrix_size, matrix_size);
  gsl_vector_complex* complex_right_eigenvalues =
      gsl_vector_complex_alloc(matrix_size);
  gsl_matrix_complex* complex_right_eigenvectors =
      gsl_matrix_complex_alloc(matrix_size, matrix_size);
  // Containers for the left eigensystem (A^T * L = lambda * L)
  gsl_matrix* point_matrix_T = gsl_matrix_alloc(matrix_size, matrix_size);
  gsl_vector_complex* complex_left_eigenvalues =
      gsl_vector_complex_alloc(matrix_size);
  gsl_matrix_complex* complex_left_eigenvectors =
      gsl_matrix_complex_alloc(matrix_size, matrix_size);

  // Allocate memory to with with blaze::geev
  Matrix blaze_point_matrix(matrix_size, matrix_size);
  blaze::DynamicVector<blaze::complex<double>> blaze_complex_eigenvalues(
      matrix_size);
  blaze::DynamicMatrix<blaze::complex<double>> blaze_complex_L(matrix_size,
                                                               matrix_size);
  blaze::DynamicMatrix<blaze::complex<double>> blaze_complex_R(matrix_size,
                                                               matrix_size);
  std::vector<double> blaze_real_eigenvalues(matrix_size);

  // Loop over each grid point
  for (size_t point = 0; point < num_points; ++point) {
    std::cout << "point " << point << ":" << std::endl;

    // Build matrix (and its transpose) at this grid point
    std::cout << "A = {" << std::endl;
    for (size_t row = 0; row < matrix_size; ++row) {
      std::cout << "\t{";
      for (size_t col = 0; col < matrix_size; ++col) {
        double entry = characteristic_matrix.get(row, col)[point];
        gsl_matrix_set(point_matrix, row, col, entry);
        gsl_matrix_set(point_matrix_T, col, row, entry);
        blaze_point_matrix(row, col) = entry;
        std::cout << std::fixed << std::setprecision(16);
        std::cout << entry;
        if (col != matrix_size - 1) {
          std::cout << ",";
        }
      }
      std::cout << "}";
      if (row != matrix_size - 1) {
        std::cout << "," << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "};" << std::endl;

    // Solve right eigensystem (A * R = lambda * R)
    // Note: the right eigenvectors of A are stored as columns of
    // complex_right_eigenvectors
    gsl_eigen_nonsymmv(point_matrix, complex_right_eigenvalues,
                       complex_right_eigenvectors, gsl_workspace);

    // Solve left eigensystem (L * A = lambda * L, or A^T * L = lambda * L)
    // Note: the left eigenvectors of A are also stored as columns of
    // complex_left_eigenvalues
    gsl_eigen_nonsymmv(point_matrix_T, complex_left_eigenvalues,
                       complex_left_eigenvectors, gsl_workspace);

    // Sort eigenvalues/vectors so that the left and right systems match
    gsl_eigen_nonsymmv_sort(complex_right_eigenvalues,
                            complex_right_eigenvectors,
                            GSL_EIGEN_SORT_ABS_DESC);
    gsl_eigen_nonsymmv_sort(complex_left_eigenvalues, complex_left_eigenvectors,
                            GSL_EIGEN_SORT_ABS_DESC);

    std::cout << "Vgsl = {";
    for (size_t i = 0; i < matrix_size; ++i) {
      gsl_complex entry = gsl_vector_complex_get(complex_right_eigenvalues, i);
      std::cout << std::fixed << std::setprecision(16);
      std::cout << GSL_REAL(entry);
      if (i < matrix_size - 1) {
        std::cout << ",";
      }
    }
    std::cout << "};" << std::endl;

    std::cout << "Rgsl = {";
    for (size_t row = 0; row < matrix_size; ++row) {
      std::cout << "\t{";
      for (size_t col = 0; col < matrix_size; ++col) {
        gsl_complex entry =
            gsl_matrix_complex_get(complex_right_eigenvectors, row, col);
        std::cout << std::fixed << std::setprecision(16);
        std::cout << GSL_REAL(entry) << " + " << GSL_IMAG(entry) << " I";
        ;
        if (col != matrix_size - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "}";
      if (row != matrix_size - 1) {
        std::cout << "," << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "};" << std::endl;

    std::cout << "Lgsl = {";
    for (size_t row = 0; row < matrix_size; ++row) {
      std::cout << "\t{";
      for (size_t col = 0; col < matrix_size; ++col) {
        gsl_complex entry =
            gsl_matrix_complex_get(complex_left_eigenvectors, row, col);
        std::cout << std::fixed << std::setprecision(16);
        std::cout << GSL_REAL(entry) << " + " << GSL_IMAG(entry) << " I";
        ;
        if (col != matrix_size - 1) {
          std::cout << ", ";
        }
      }
      std::cout << "}";
      if (row != matrix_size - 1) {
        std::cout << "," << std::endl;
      }
      std::cout << std::endl;
    }
    std::cout << "};" << std::endl;

    // Solve eigensystem using blaze:geev for comparison
    blaze::geev(blaze_point_matrix, blaze_complex_L, blaze_complex_eigenvalues,
                blaze_complex_R);
    for (size_t i = 0; i < matrix_size; ++i) {
      blaze_real_eigenvalues[i] = blaze_complex_eigenvalues[i].real();
    }
    std::sort(blaze_real_eigenvalues.begin(), blaze_real_eigenvalues.end(),
              [](const double& a, const double& b) {
                return std::abs(a) > std::abs(b);
              });

    std::cout << "Vblaze = {";
    for (size_t i = 0; i < matrix_size; ++i) {
      std::cout << std::fixed << std::setprecision(16);
      std::cout << blaze_real_eigenvalues[i];
      if (i < matrix_size - 1) {
        std::cout << ",";
      }
    }
    std::cout << "};" << std::endl;

    std::cout << "Rblaze = {" << std::endl;
    for (size_t i = 0; i < matrix_size; ++i) {
      std::cout << "\t{";
      for (size_t j = 0; j < matrix_size; ++j) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << blaze_complex_R(i, j).real() << "+"
                  << blaze_complex_R(i, j).imag() << "I";
        if (j < matrix_size - 1) {
          std::cout << ",";
        }
      }
      std::cout << "}";
      if (i < matrix_size - 1) {
        std::cout << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "};" << std::endl;

    std::cout << "Lblaze = {" << std::endl;
    for (size_t i = 0; i < matrix_size; ++i) {
      std::cout << "\t{";
      for (size_t j = 0; j < matrix_size; ++j) {
        std::cout << std::fixed << std::setprecision(16);
        std::cout << blaze_complex_L(i, j).real() << "+"
                  << blaze_complex_L(i, j).imag() << "I";
        if (j < matrix_size - 1) {
          std::cout << ",";
        }
      }
      std::cout << "}";
      if (i < matrix_size - 1) {
        std::cout << ",";
      }
      std::cout << std::endl;
    }
    std::cout << "};" << std::endl;

    // Check and save results
    const double tolerance = 1.0e-10;
    // To-do: improve names / handling of complex conjugate pairs
    bool is_right_pair_from_previous = false;
    bool is_left_pair_from_previous = false;
    // Note: we're looping through the ith eigenvalue/vector, not the ith row!
    for (size_t i = 0; i < matrix_size; ++i) {
      // Get eigenvalue from the right eigensystem
      gsl_complex eigenvalue =
          gsl_vector_complex_get(complex_right_eigenvalues, i);
      ASSERT(std::abs(GSL_IMAG(eigenvalue)) < tolerance,
             "Complex eigenvalue: " << GSL_REAL(eigenvalue) << " + "
                                    << GSL_IMAG(eigenvalue) << " i.");

      // Check that the eigenvalue from the left eigensystem is the same as the
      // eigenvalue from the right eigensystem
      gsl_complex left_eigenvalue =
          gsl_vector_complex_get(complex_left_eigenvalues, i);
      ASSERT(std::abs(GSL_REAL(eigenvalue) - GSL_REAL(left_eigenvalue)) <
                 tolerance,
             "Eigenvalues from left/right eigensystems differ by "
                 << std::abs(GSL_REAL(eigenvalue) - GSL_REAL(left_eigenvalue))
                 << ".");

      double diff_with_blaze =
          std::abs(GSL_REAL(eigenvalue) - blaze_real_eigenvalues[i]);
      ASSERT(diff_with_blaze < tolerance,
             "Eigenvalue " << i << " from GSL and blaze::geev differ by "
                           << diff_with_blaze << ".");

      // Store eigenvalue
      all_eigenvalues[i][point] = GSL_REAL(eigenvalue);

      // For each PAIR of degenerate eigenvalues, it is possible that GSL
      // returns a PAIR of complex eigenvectors that are complex conjugates of
      // each other. Here, we check if this is the case for the ith right/left
      // eigenvector.
      bool right_eigenvector_is_complex = false;
      bool left_eigenvector_is_complex = false;
      for (size_t j = 0; j < matrix_size; ++j) {
        if (not is_right_pair_from_previous) {
          gsl_complex right_component =
              gsl_matrix_complex_get(complex_right_eigenvectors, j, i);
          if (std::abs(GSL_IMAG(right_component)) > tolerance) {
            // Check that this isn't the last eigenvector
            ASSERT(
                i + 1 < matrix_size,
                "Complex eigenvector at last index without a conjugate pair.");

            // Check that the next eigenvector is the complex conjugate
            gsl_complex conjugate_component =
                gsl_matrix_complex_get(complex_right_eigenvectors, j, i + 1);
            ASSERT(std::abs(GSL_REAL(right_component) -
                            GSL_REAL(conjugate_component)) < tolerance,
                   "Expected conjugate pair for right eigenvector, but real "
                   "parts differ by "
                       << std::abs(GSL_REAL(right_component) -
                                   GSL_REAL(conjugate_component))
                       << ".");
            ASSERT(std::abs(GSL_IMAG(right_component) +
                            GSL_IMAG(conjugate_component)) < tolerance,
                   "Expected conjugate pair for right eigenvector, but "
                   "imaginary parts add up to "
                       << std::abs(GSL_IMAG(right_component) +
                                   GSL_IMAG(conjugate_component))
                       << ".");

            right_eigenvector_is_complex = true;
          }
        }

        if (not is_left_pair_from_previous) {
          gsl_complex left_component =
              gsl_matrix_complex_get(complex_left_eigenvectors, j, i);
          if (std::abs(GSL_IMAG(left_component)) > tolerance) {
            // Check that this isn't the last eigenvector
            ASSERT(
                i + 1 < matrix_size,
                "Complex eigenvector at last index without a conjugate pair.");

            // Check that the next eigenvector is the complex conjugate
            gsl_complex conjugate_component =
                gsl_matrix_complex_get(complex_left_eigenvectors, j, i + 1);
            ASSERT(std::abs(GSL_REAL(left_component) -
                            GSL_REAL(conjugate_component)) < tolerance,
                   "Expected conjugate pair for right eigenvector, but real "
                   "parts differ by "
                       << std::abs(GSL_REAL(left_component) -
                                   GSL_REAL(conjugate_component))
                       << ".");
            ASSERT(std::abs(GSL_IMAG(left_component) +
                            GSL_IMAG(conjugate_component)) < tolerance,
                   "Expected conjugate pair for right eigenvector, but "
                   "imaginary parts add up to "
                       << std::abs(GSL_IMAG(left_component) +
                                   GSL_IMAG(conjugate_component))
                       << ".");

            left_eigenvector_is_complex = true;
          }
        }
      }

      // If either left or right eigenvector is complex, then their respective
      // eigenvalues must be degenerate
      if (right_eigenvector_is_complex or left_eigenvector_is_complex) {
        gsl_complex next_eigenvalue =
            gsl_vector_complex_get(complex_right_eigenvalues, i + 1);
        ASSERT(GSL_REAL(eigenvalue) - GSL_REAL(next_eigenvalue) < tolerance,
               "Expected degenerate eigenvalues for complex eigenvectors, but "
               "eigenvalues differ by "
                   << std::abs(GSL_REAL(eigenvalue) - GSL_REAL(next_eigenvalue))
                   << ".");
      }

      // If either eigenvector is complex, then we know that the vectors formed
      // by their real and imaginary parts are also linearly-independent
      // eigenvectors. Here, we use this fact to build real-valued left and
      // right eigenvectors.
      for (size_t j = 0; j < matrix_size; ++j) {
        if (is_right_pair_from_previous) {
          is_right_pair_from_previous = false;
        } else {
          gsl_complex right_component =
              gsl_matrix_complex_get(complex_right_eigenvectors, j, i);
          if (right_eigenvector_is_complex) {
            all_right_eigenvectors[i].get(j)[point] = GSL_REAL(right_component);
            all_right_eigenvectors[i + 1].get(j)[point] =
                GSL_IMAG(right_component);
          } else {
            all_right_eigenvectors[i].get(j)[point] = GSL_REAL(right_component);
          }
        }

        if (is_left_pair_from_previous) {
          is_left_pair_from_previous = false;
        } else {
          gsl_complex left_component =
              gsl_matrix_complex_get(complex_left_eigenvectors, j, i);
          if (left_eigenvector_is_complex) {
            all_left_eigenvectors[i].get(j)[point] = GSL_REAL(left_component);
            all_left_eigenvectors[i + 1].get(j)[point] =
                GSL_IMAG(left_component);
          } else {
            all_left_eigenvectors[i].get(j)[point] = GSL_REAL(left_component);
          }
        }
      }

      // If either eigenvector is complex, SKIP the next index because we've
      // already handled the i+1th eigenvalue/vectors.
      // std::cout << "right_eigenvector_is_complex, left_eigenvector_is_complex
      // = " << right_eigenvector_is_complex << ", " <<
      // left_eigenvector_is_complex << std::endl;
      is_right_pair_from_previous = right_eigenvector_is_complex;
      is_left_pair_from_previous = left_eigenvector_is_complex;
      // if (right_eigenvector_is_complex or left_eigenvector_is_complex) {
      //   i += 1;
      //   std::cout << "Skipping eigenvalue/vectors " << i << "..." <<
      //   std::endl;
      // }
    }
  }

  // Free GSL allocations
  gsl_eigen_nonsymmv_free(gsl_workspace);
  gsl_matrix_free(point_matrix);
  gsl_matrix_free(point_matrix_T);
  gsl_vector_complex_free(complex_right_eigenvalues);
  gsl_matrix_complex_free(complex_right_eigenvectors);
  gsl_vector_complex_free(complex_left_eigenvalues);
  gsl_matrix_complex_free(complex_left_eigenvectors);

  return {all_eigenvalues, {all_right_eigenvectors, all_left_eigenvectors}};
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<DataVector, 9>                                        \
  approx::characteristic_speeds<GET_DIM(data)>(                             \
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
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state);                                               \
  template void approx::characteristic_speeds<GET_DIM(data)>(               \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,          \
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
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
