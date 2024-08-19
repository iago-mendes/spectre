// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "PointwiseFunctions/Xcts/CenterOfMass.hpp"

namespace Xcts {

void center_of_mass_surface_integrand(
    gsl::not_null<tnsr::II<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal) {
  tenex::evaluate<ti::I, ti::J>(result,
                                3. / (8. * M_PI) * pow<4>(conformal_factor()) *
                                    unit_normal(ti::I) * unit_normal(ti::J));
}

tnsr::II<DataVector, 3> center_of_mass_surface_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal) {
  tnsr::II<DataVector, 3> result;
  center_of_mass_surface_integrand(make_not_null(&result), conformal_factor,
                                   unit_normal);
  return result;
}

void center_of_mass_volume_integrand(
    gsl::not_null<tnsr::I<DataVector, 3>*> result,
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& deriv_conformal_metric) {
  tenex::evaluate<ti::I>(
      result, 4. * pow<3>(conformal_factor()) * deriv_conformal_factor(ti::j) *
                      unit_normal(ti::I) * unit_normal(ti::J) -
                  pow<4>(conformal_factor()) *
                      deriv_conformal_metric(ti::j, ti::k, ti::l) *
                      unit_normal(ti::I) * unit_normal(ti::J) *
                      unit_normal(ti::K) * unit_normal(ti::L));
}

tnsr::I<DataVector, 3> center_of_mass_volume_integrand(
    const Scalar<DataVector>& conformal_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& deriv_conformal_factor,
    const tnsr::I<DataVector, 3>& unit_normal,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& deriv_conformal_metric) {
  tnsr::I<DataVector, 3> result;
  center_of_mass_volume_integrand(make_not_null(&result), conformal_factor,
                                  deriv_conformal_factor, unit_normal,
                                  deriv_conformal_metric);
  return result;
}

}  // namespace Xcts
