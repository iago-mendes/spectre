// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/AnalyticData/GrMhd/BeckwithStoneKhInstability.hpp"

#include <cmath>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/LorentzFactor.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::AnalyticData {

namespace {
// Horizontal velocity profile v_x(y): a smooth double shear layer, -v_sh in the
// central band |y| < y0 and +v_sh outside.
double bs_kh_velocity_x(const double y, const double shear_velocity,
                        const double strip_half_width,
                        const double transition_thickness) {
  return shear_velocity *
         (std::tanh((y - strip_half_width) / transition_thickness) -
          std::tanh((y + strip_half_width) / transition_thickness) + 1.0);
}
}  // namespace

BeckwithStoneKhInstability::BeckwithStoneKhInstability(
    const double adiabatic_index, const double shear_velocity,
    const double strip_half_width, const double transition_thickness,
    const double upper_density, const double lower_density,
    const double pressure, const double perturbation_amplitude,
    const double perturbation_width,
    const std::array<double, 3>& magnetic_field)
    : adiabatic_index_(adiabatic_index),
      shear_velocity_(shear_velocity),
      strip_half_width_(strip_half_width),
      transition_thickness_(transition_thickness),
      upper_density_(upper_density),
      lower_density_(lower_density),
      pressure_(pressure),
      perturbation_amplitude_(perturbation_amplitude),
      perturbation_width_(perturbation_width),
      magnetic_field_(magnetic_field),
      equation_of_state_(adiabatic_index) {
  ASSERT(upper_density_ > 0.0 and lower_density_ > 0.0,
         "The mass density must be positive everywhere. Upper density: "
             << upper_density_ << ", Lower density: " << lower_density_ << ".");
  ASSERT(pressure_ > 0.0, "The pressure must be positive. The value given was "
                              << pressure_ << ".");
  ASSERT(perturbation_width_ > 0.0,
         "The perturbation width must be positive. The value given was "
             << perturbation_width_ << ".");
  ASSERT(transition_thickness_ > 0.0,
         "The transition thickness must be positive. The value given was "
             << transition_thickness_ << ".");
}

std::unique_ptr<evolution::initial_data::InitialData>
BeckwithStoneKhInstability::get_clone() const {
  return std::make_unique<BeckwithStoneKhInstability>(*this);
}

BeckwithStoneKhInstability::BeckwithStoneKhInstability(CkMigrateMessage* msg)
    : InitialData(msg) {}

void BeckwithStoneKhInstability::pup(PUP::er& p) {
  InitialData::pup(p);
  p | adiabatic_index_;
  p | shear_velocity_;
  p | strip_half_width_;
  p | transition_thickness_;
  p | upper_density_;
  p | lower_density_;
  p | pressure_;
  p | perturbation_amplitude_;
  p | perturbation_width_;
  p | magnetic_field_;
  p | equation_of_state_;
  p | background_spacetime_;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/) const {
  auto result = make_with_value<Scalar<DataType>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  const double mean_density = 0.5 * (lower_density_ + upper_density_);
  const double half_density_jump = 0.5 * (upper_density_ - lower_density_);
  for (size_t s = 0; s < n_pts; ++s) {
    const double y = get_element(get<1>(x), s);
    const double v_x = bs_kh_velocity_x(y, shear_velocity_, strip_half_width_,
                                        transition_thickness_);
    get_element(get(result), s) =
        mean_density + half_density_jump * (v_x / shear_velocity_);
  }
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/) const {
  return {make_with_value<Scalar<DataType>>(x, 0.1)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const {
  return equation_of_state_.specific_internal_energy_from_density_and_pressure(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3, Frame::Inertial>& x,
    tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, pressure_);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/) const {
  auto result = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  const size_t n_pts = get_size(get<0>(x));
  const double one_over_two_sigma_squared =
      0.5 / square(perturbation_width_);
  for (size_t s = 0; s < n_pts; ++s) {
    const double xx = get_element(get<0>(x), s);
    const double y = get_element(get<1>(x), s);
    const double sign_y = (y >= 0.0 ? 1.0 : -1.0);
    get_element(get<0>(result), s) = bs_kh_velocity_x(
        y, shear_velocity_, strip_half_width_, transition_thickness_);
    get_element(get<1>(result), s) =
        sign_y * perturbation_amplitude_ * shear_velocity_ *
        sin(2.0 * M_PI * xx) *
        exp(-one_over_two_sigma_squared *
            square(y - sign_y * strip_half_width_));
  }
  return {std::move(result)};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/) const {
  auto mag_field = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(
      get<0>(x), magnetic_field_[0]);
  get<1>(mag_field) = magnetic_field_[1];
  get<2>(mag_field) = magnetic_field_[2];
  return mag_field;
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const {
  return make_with_value<Scalar<DataType>>(x, 0.0);
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/) const {
  using velocity_tag = hydro::Tags::SpatialVelocity<DataType, 3>;
  const auto velocity =
      get<velocity_tag>(variables(x, tmpl::list<velocity_tag>{}));
  return {hydro::lorentz_factor(dot_product(velocity, velocity))};
}

template <typename DataType>
tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>
BeckwithStoneKhInstability::variables(
    const tnsr::I<DataType, 3>& x,
    tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/) const {
  return hydro::relativistic_specific_enthalpy(
      get<hydro::Tags::RestMassDensity<DataType>>(
          variables(x, tmpl::list<hydro::Tags::RestMassDensity<DataType>>{})),
      get<hydro::Tags::SpecificInternalEnergy<DataType>>(variables(
          x, tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>>{})),
      get<hydro::Tags::Pressure<DataType>>(
          variables(x, tmpl::list<hydro::Tags::Pressure<DataType>>{})));
}

PUP::able::PUP_ID BeckwithStoneKhInstability::my_PUP_ID = 0;

bool operator==(const BeckwithStoneKhInstability& lhs,
                const BeckwithStoneKhInstability& rhs) {
  // No comparison for equation_of_state_. Comparing adiabatic_index_ should
  // suffice.
  return lhs.adiabatic_index_ == rhs.adiabatic_index_ and
         lhs.shear_velocity_ == rhs.shear_velocity_ and
         lhs.strip_half_width_ == rhs.strip_half_width_ and
         lhs.transition_thickness_ == rhs.transition_thickness_ and
         lhs.upper_density_ == rhs.upper_density_ and
         lhs.lower_density_ == rhs.lower_density_ and
         lhs.pressure_ == rhs.pressure_ and
         lhs.perturbation_amplitude_ == rhs.perturbation_amplitude_ and
         lhs.perturbation_width_ == rhs.perturbation_width_ and
         lhs.magnetic_field_ == rhs.magnetic_field_;
}

bool operator!=(const BeckwithStoneKhInstability& lhs,
                const BeckwithStoneKhInstability& rhs) {
  return not(lhs == rhs);
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define TAG(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SCALARS(_, data)                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data)>>     \
      BeckwithStoneKhInstability::variables(                 \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x, \
          tmpl::list<TAG(data) < DTYPE(data)>>) const;

GENERATE_INSTANTIATIONS(
    INSTANTIATE_SCALARS, (double, DataVector),
    (hydro::Tags::RestMassDensity, hydro::Tags::ElectronFraction,
     hydro::Tags::SpecificInternalEnergy, hydro::Tags::Pressure,
     hydro::Tags::DivergenceCleaningField, hydro::Tags::SpecificEnthalpy,
     hydro::Tags::LorentzFactor))

#define INSTANTIATE_VECTORS(_, data)                                         \
  template tuples::TaggedTuple<TAG(data) < DTYPE(data), 3, Frame::Inertial>> \
      BeckwithStoneKhInstability::variables(                                 \
          const tnsr::I<DTYPE(data), 3, Frame::Inertial>& x,                 \
          tmpl::list<TAG(data) < DTYPE(data), 3, Frame::Inertial>>) const;

GENERATE_INSTANTIATIONS(INSTANTIATE_VECTORS, (double, DataVector),
                        (hydro::Tags::SpatialVelocity,
                         hydro::Tags::MagneticField))

#undef INSTANTIATE_VECTORS
#undef INSTANTIATE_SCALARS
#undef TAG
#undef DTYPE
}  // namespace grmhd::AnalyticData
