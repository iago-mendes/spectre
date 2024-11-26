// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr::surfaces {
/// @{
/// \ingroup SurfacesGroup
/// \brief Computes the 2-metric of a Strahlkorper.
///
// template <typename Frame>
// void surface_metric(
//     gsl::not_null<tnsr::ii<DataVector, 2, Frame::Spherical<Frame>>*> result,
//     const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
//     const ylm::Tags::aliases::Jacobian<Frame>& tangents,
//     const ylm::Strahlkorper<Frame>& strahlkorper);

// /// Return-by-value overload
// template <typename Frame>
// tnsr::ii<DataVector, 2, Frame::Spherical<Frame>> surface_metric(
//     const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
//     const ylm::Tags::aliases::Jacobian<Frame>& tangents,
//     const ylm::Strahlkorper<Frame>& strahlkorper);

template <typename Frame>
void surface_metric_theta_theta(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents);

template <typename Frame>
void surface_metric_phi_phi(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper);

template <typename Frame>
void surface_metric_theta_phi(
    gsl::not_null<Scalar<DataVector>*> result,
    const tnsr::ii<DataVector, 3, Frame>& spatial_metric,
    const ylm::Tags::aliases::Jacobian<Frame>& tangents,
    const ylm::Strahlkorper<Frame>& strahlkorper);
/// @}

}  // namespace gr::surfaces
