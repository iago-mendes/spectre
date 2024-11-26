// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/GeneralRelativity/Surfaces/InverseSurfaceMetric.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gr::surfaces {

// template <typename Frame>
// void surface_metric(
//     gsl::not_null<tnsr::ii<DataVector, 2, Frame::Spherical<Frame>>*> result,
//     const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
//     const ylm::Tags::aliases::Jacobian<Frame>& tangents,
//     const ylm::Strahlkorper<Frame>& strahlkorper) {
//   *result =
//       make_with_value<tnsr::ii<DataVector, 2, Frame::Spherical<Frame>>>(
//           get<0, 0>(spatial_metric), 0.0);

//   const Scalar<DataVector> sin_theta{
//       sin(strahlkorper.ylm_spherepack().theta_phi_points()[0])};

//   for (size_t i = 0; i < 3; ++i) {
//     for (size_t j = 0; j < 3; ++j) {
//       get<0, 0>(surface_metric) +=
//           spatial_metric.get(i, j) * tangents.get(i, 0) * tangents.get(i, 0);
//       get<1, 1>(surface_metric) +=
//           spatial_metric.get(i, j) * tangents.get(i, 1) * tangents.get(i, 1)
//           * square(get(sin_theta));
//       get<0, 1>(surface_metric) += spatial_metric.get(i, j) *
//                                    tangents.get(i, 0) * tangents.get(j, 1) *
//                                    get(sin_theta);
//     }
//   }
//   return surface_metric;
// }

// template <typename Frame>
// tnsr::ii<DataVector, 2, Frame::Spherical<Frame>> surface_metric(
//     const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
//     const ylm::Tags::aliases::Jacobian<Frame>& tangents,
//     const ylm::Strahlkorper<Frame>& strahlkorper) {
//   tnsr::ii<DataVector, 2, Frame::Spherical<Frame>> result;
//   surface_metric(make_not_null(&result), spatial_metric, tangets,
//   strahlkorper); return result;
// }

template <typename Frame>
void surface_metric_theta_theta(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents) {
  *result = make_with_value<Scalar<DataVector>>(get<0, 0>(spatial_metric), 0.0);

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) +=
          spatial_metric.get(i, j) * tangents.get(i, 0) * tangents.get(i, 0);
    }
  }
}

template <typename Frame>
void surface_metric_phi_phi(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper) {
  *result = make_with_value<Scalar<DataVector>>(get<0, 0>(spatial_metric), 0.0);

  const Scalar<DataVector> sin_theta{
      sin(strahlkorper.ylm_spherepack().theta_phi_points()[0])};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += spatial_metric.get(i, j) * tangents.get(i, 1) *
                      tangents.get(i, 1) * square(get(sin_theta));
    }
  }
}

template <typename Frame>
void surface_metric_theta_phi(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper) {
  *result = make_with_value<Scalar<DataVector>>(get<0, 0>(spatial_metric), 0.0);

  const Scalar<DataVector> sin_theta{
      sin(strahlkorper.ylm_spherepack().theta_phi_points()[0])};

  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(*result) += spatial_metric.get(i, j) * tangents.get(i, 0) *
                      tangents.get(j, 1) * get(sin_theta);
    }
  }
}

}  // namespace gr::surfaces

#define FRAME(data) BOOST_PP_TUPLE_ELEM(0, data)
#define INSTANTIATE(_, data)                                           \
  template void gr::surfaces::surface_metric_theta_theta<FRAME(data)>( \
      const gsl::not_null<Scalar<DataVector>*> result,                 \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,      \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents);      \
  template void gr::surfaces::surface_metric_phi_phi<FRAME(data)>(     \
      const gsl::not_null<Scalar<DataVector>*> result,                 \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,      \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,       \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper);             \
  template void gr::surfaces::surface_metric_theta_phi<FRAME(data)>(   \
      const gsl::not_null<Scalar<DataVector>*> result,                 \
      const tnsr::ii<DataVector, 3, FRAME(data)>& spatial_metric,      \
      const ylm::Tags::aliases::Jacobian<FRAME(data)>& tangents,       \
      const ylm::Strahlkorper<FRAME(data)>& strahlkorper);
GENERATE_INSTANTIATIONS(INSTANTIATE,
                        (Frame::Grid, Frame::Distorted, Frame::Inertial))
#undef INSTANTIATE
#undef FRAME
