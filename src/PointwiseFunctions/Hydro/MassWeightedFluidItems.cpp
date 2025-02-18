// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "MassWeightedFluidItems.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace hydro {

std::ostream& operator<<(std::ostream& os, HalfPlaneIntegralMask mask) {
  switch (mask) {
    case HalfPlaneIntegralMask::None:
      return os << "None";
    case HalfPlaneIntegralMask::PositiveXOnly:
      return os << "PositiveXOnly";
    case HalfPlaneIntegralMask::NegativeXOnly:
      return os << "NegativeXOnly";
    default:
      ERROR("Unknown HalfPlaneIntegralMask!");
  }
}

std::string name(const HalfPlaneIntegralMask mask){
  return MakeString{} << mask;
}

template <typename DataType, size_t Dim, typename Frame>
void u_lower_t(const gsl::not_null<Scalar<DataType>*> result,
               const Scalar<DataType>& lorentz_factor,
               const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
               const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
               const Scalar<DataType>& lapse,
               const tnsr::I<DataType, Dim, Frame>& shift) {
  dot_product(result, spatial_velocity, shift, spatial_metric);
  result->get() = get(lorentz_factor) * (get(lapse) * (-1.0) + result->get());
}

template <typename DataType, size_t Dim, typename Frame>
Scalar<DataType> u_lower_t(
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Frame>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Frame>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Frame>& shift) {
  auto result = make_with_value<Scalar<DataType>>(lorentz_factor, 0.0);
  u_lower_t(make_not_null(&result), lorentz_factor, spatial_velocity,
            spatial_metric, lapse, shift);
  return result;
}

template <typename DataType>
void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const Scalar<DataType>& specific_internal_energy) {
  result->get() = get(tilde_d) * get(specific_internal_energy);
}

template <typename DataType>
Scalar<DataType> mass_weighted_internal_energy(
    const Scalar<DataType>& tilde_d,
    const Scalar<DataType>& specific_internal_energy) {
  auto result = make_with_value<Scalar<DataType>>(tilde_d, 0.0);
  mass_weighted_internal_energy(make_not_null(&result), tilde_d,
                                specific_internal_energy);
  return result;
}

template <typename DataType>
void mass_weighted_kinetic_energy(const gsl::not_null<Scalar<DataType>*> result,
                                  const Scalar<DataType>& tilde_d,
                                  const Scalar<DataType>& lorentz_factor) {
  result->get() = get(tilde_d) * (get(lorentz_factor) - 1.0);
}

template <typename DataType>
Scalar<DataType> mass_weighted_kinetic_energy(
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor) {
  auto result = make_with_value<Scalar<DataType>>(tilde_d, 0.0);
  mass_weighted_kinetic_energy(make_not_null(&result), tilde_d, lorentz_factor);
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
void tilde_d_unbound_ut_criterion(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift) {
  u_lower_t(result, lorentz_factor, spatial_velocity, spatial_metric, lapse,
            shift);
  result->get() = get(tilde_d) * step_function(-1.0 - result->get());
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> tilde_d_unbound_ut_criterion(
    const Scalar<DataType>& tilde_d, const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, Dim, Fr>& spatial_velocity,
    const tnsr::ii<DataType, Dim, Fr>& spatial_metric,
    const Scalar<DataType>& lapse, const tnsr::I<DataType, Dim, Fr>& shift) {
  auto result = make_with_value<Scalar<DataType>>(tilde_d, 0.0);
  tilde_d_unbound_ut_criterion(make_not_null(&result), tilde_d, lorentz_factor,
                               spatial_velocity, spatial_metric, lapse, shift);
  return result;
}

template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim>
void tilde_d_in_half_plane(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords) {
  get(*result) = get(tilde_d);
  switch (IntegralMask) {
    case HalfPlaneIntegralMask::PositiveXOnly:
      get(*result) *= step_function(get<0>(grid_coords));
      break;
    case HalfPlaneIntegralMask::NegativeXOnly:
      get(*result) *= step_function(get<0>(grid_coords) * (-1.0));
      break;
    default:
      break;
  }
}

template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim>
Scalar<DataType> tilde_d_in_half_plane(
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords) {
  auto result = make_with_value<Scalar<DataType>>(tilde_d, 0.0);
  tilde_d_in_half_plane<IntegralMask>(make_not_null(&result), tilde_d,
                                      grid_coords);
  return result;
}

template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim,
          typename Fr>
void mass_weighted_coords(
    const gsl::not_null<tnsr::I<DataType, Dim, Fr>*> result,
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords,
    const tnsr::I<DataType, Dim, Fr>& compute_coords) {
  for (size_t i = 0; i < Dim; i++) {
    result->get(i) = get(tilde_d) * (compute_coords.get(i));
    switch (IntegralMask) {
      case HalfPlaneIntegralMask::PositiveXOnly:
        result->get(i) *= step_function(get<0>(grid_coords));
        break;
      case HalfPlaneIntegralMask::NegativeXOnly:
        result->get(i) *= step_function(get<0>(grid_coords) * (-1.0));
        break;
      default:
        break;
    }
  }
}

template <HalfPlaneIntegralMask IntegralMask, typename DataType, size_t Dim,
          typename Fr>
tnsr::I<DataType, Dim, Fr> mass_weighted_coords(
    const Scalar<DataType>& tilde_d,
    const tnsr::I<DataType, Dim, Frame::Grid>& grid_coords,
    const tnsr::I<DataType, Dim, Fr>& compute_coords) {
  auto result = make_with_value<tnsr::I<DataType, Dim, Fr>>(tilde_d, 0.0);
  mass_weighted_coords<IntegralMask>(make_not_null(&result), tilde_d,
                                     grid_coords, compute_coords);
  return result;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void u_lower_t(                                                     \
      const gsl::not_null<Scalar<DataVector>*> result,                         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);           \
  template Scalar<DataVector> u_lower_t(                                       \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);           \
  template void tilde_d_unbound_ut_criterion(                                  \
      const gsl::not_null<Scalar<DataVector>*> result,                         \
      const Scalar<DataVector>& tilde_d,                                       \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);           \
  template Scalar<DataVector> tilde_d_unbound_ut_criterion(                    \
      const Scalar<DataVector>& tilde_d,                                       \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& spatial_velocity, \
      const tnsr::ii<DataVector, DIM(data), Frame::Inertial>& spatial_metric,  \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define HALFPLANE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                                \
  template void mass_weighted_coords<HALFPLANE(data)>(                      \
      const gsl::not_null<tnsr::I<DataVector, DIM(data), Frame::Inertial>*> \
          result,                                                           \
      const Scalar<DataVector>& tilde_d,                                    \
      const tnsr::I<DataVector, DIM(data), Frame::Grid>& dg_grid_coords,    \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& dg_coords);    \
  template tnsr::I<DataVector, DIM(data), Frame::Inertial>                  \
  mass_weighted_coords<HALFPLANE(data)>(                                    \
      const Scalar<DataVector>& tilde_d,                                    \
      const tnsr::I<DataVector, DIM(data), Frame::Grid>& dg_grid_coords,    \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& dg_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (HalfPlaneIntegralMask::None,
                         HalfPlaneIntegralMask::PositiveXOnly,
                         HalfPlaneIntegralMask::NegativeXOnly))

#undef DIM
#undef OBJECT
#undef INSTANTIATE

// For tilde_d_in_half_plane, we require limiting the integrand to a half
// plane -> Do not instantiate the function for None
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define HALFPLANE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template void tilde_d_in_half_plane<HALFPLANE(data)>(                   \
      const gsl::not_null<Scalar<DataVector>*> result,                    \
      const Scalar<DataVector>& tilde_d,                                  \
      const tnsr::I<DataVector, DIM(data), Frame::Grid>& dg_grid_coords); \
  template Scalar<DataVector> tilde_d_in_half_plane<HALFPLANE(data)>(     \
      const Scalar<DataVector>& tilde_d,                                  \
      const tnsr::I<DataVector, DIM(data), Frame::Grid>& dg_grid_coords);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3),
                        (HalfPlaneIntegralMask::PositiveXOnly,
                         HalfPlaneIntegralMask::NegativeXOnly))

#undef DIM
#undef OBJECT
#undef INSTANTIATE

template void mass_weighted_internal_energy(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& specific_internal_energy);
template Scalar<DataVector> mass_weighted_internal_energy(
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& specific_internal_energy);
template void mass_weighted_kinetic_energy(
    const gsl::not_null<Scalar<DataVector>*> result,
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& lorentz_factor);
template Scalar<DataVector> mass_weighted_kinetic_energy(
    const Scalar<DataVector>& tilde_d,
    const Scalar<DataVector>& lorentz_factor);

}  // namespace hydro
