// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <tuple>

#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

/*!
 * \brief Computes the next iteration of the acceleration due to scalar self
 * force from the current iteration of the regular field.
 */
struct IterateAccelerationTerms {
  static constexpr size_t Dim = 3;
  using simple_tags = tmpl::list<Tags::AccelerationTerms>;
  using return_tags = tmpl::list<Tags::AccelerationTerms>;
  using argument_tags = tmpl::list<
      Tags::ParticlePositionVelocity<Dim>, Tags::BackgroundQuantities<Dim>,
      Tags::GeodesicAcceleration<Dim>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 0, Dim, Frame::Inertial>,
      Stf::Tags::StfTensor<::Tags::dt<Tags::PsiWorldtube>, 0, Dim,
                           Frame::Inertial>,
      Stf::Tags::StfTensor<Tags::PsiWorldtube, 1, Dim, Frame::Inertial>,
      Tags::Charge, Tags::Mass, ::Tags::Time, Tags::SelfForceTurnOnTime,
      Tags::SelfForceTurnOnInterval>;
  static void apply(
      const gsl::not_null<Scalar<DataVector>*> acceleration_terms,
      const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
      const tuples::TaggedTuple<
          gr::Tags::SpacetimeMetric<double, Dim>,
          gr::Tags::InverseSpacetimeMetric<double, Dim>,
          gr::Tags::SpacetimeChristoffelSecondKind<double, Dim>,
          gr::Tags::TraceSpacetimeChristoffelSecondKind<double, Dim>,
          Tags::TimeDilationFactor>& background,
      const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
      const Scalar<double>& psi_monopole, const Scalar<double>& dt_psi_monopole,
      const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole, double charge,
      std::optional<double> mass, double time,
      std::optional<double> turn_on_time,
      std::optional<double> turn_on_interval);
};

}  // namespace CurvedScalarWave::Worldtube
