// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <memory>
#include <optional>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {

/*!
 * \brief The HLLD Riemann solver of \cite Mignone2009 (MUB2009).
 *
 * The HLLD solver restores the two Alfv&eacute;n (rotational) waves and the
 * contact wave on top of the outer fast magnetosonic waves, giving a five-wave
 * approximation of the Riemann fan. Compared to HLL it resolves contact and
 * rotational discontinuities much more sharply (at the cost of a non-linear
 * solve for the total pressure across the fan).
 *
 * The Riemann problem is solved in the frame normal to the interface. In flat
 * space (lapse \f$\alpha=1\f$, shift \f$\beta^i=0\f$, \f$\sqrt{\gamma}=1\f$)
 * the densitized conserved variables reduce to the special-relativistic
 * conserved variables and this reproduces the standard MUB2009 solver; that is
 * the regime of the Mattia & Mignone (2022) test suite used for validation.
 *
 * If the total-pressure solve fails, or the intermediate states are unphysical,
 * the solver falls back to the HLL flux, so it is at least as robust as HLL.
 *
 * The characteristic/signal speeds are those of
 * `grmhd::ValenciaDivClean::characteristic_speeds()`; the divergence-cleaning
 * field \f$\tilde\Phi\f$ is not part of the HLLD fan and is treated with the
 * HLL flux.
 */
class Hlld final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  // The interface unit normal (interior side) is needed in dg_boundary_terms to
  // solve the 1D Riemann problem in the direction normal to the interface.
  struct InterfaceUnitNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  // Departure of the metric from flat space, |lapse-1| + |shift| + |sqrt(det)-1|.
  // The five-wave fan is reconstructed assuming flat space (the regime of the
  // relativistic M&M tests); where the metric is curved we fall back to HLL.
  struct MetricFlatness : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  struct MagneticFieldMagnitudeForHydro {
    static constexpr Options::String help = {
        "When the magnetic field is below this value we use the hydro "
        "characteristic speeds."};
    using type = double;
  };
  struct LightSpeedDensityCutoff {
    static constexpr Options::String help = {
        "When the density is below this value we just use the light speed for "
        "the characteristic speeds."};
    using type = double;
  };
  using options =
      tmpl::list<MagneticFieldMagnitudeForHydro, LightSpeedDensityCutoff>;
  static constexpr Options::String help = {
      "Computes the HLLD boundary correction term for the GRMHD system."};

  Hlld() = default;
  Hlld(const Hlld&) = default;
  Hlld& operator=(const Hlld&) = default;
  Hlld(Hlld&&) = default;
  Hlld& operator=(Hlld&&) = default;
  ~Hlld() override = default;

  Hlld(double magnetic_field_magnitude_for_hydro,
       double light_speed_density_cutoff);

  /// \cond
  explicit Hlld(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hlld);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  // In addition to what HLL packages (conserved vars, their normal fluxes, and
  // the two extreme char speeds) HLLD needs the primitive state on each side to
  // reconstruct the Alfven/contact fan.
  using dg_package_field_tags = tmpl::list<
      Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
      Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
      Tags::TildePhi, ::Tags::NormalDotFlux<Tags::TildeD>,
      ::Tags::NormalDotFlux<Tags::TildeYe>,
      ::Tags::NormalDotFlux<Tags::TildeTau>,
      ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
      ::Tags::NormalDotFlux<Tags::TildePhi>, LargestOutgoingCharSpeed,
      LargestIngoingCharSpeed, InterfaceUnitNormal, MetricFlatness,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::Pressure<DataVector>, hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpecificInternalEnergy<DataVector>>;
  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>;
  using dg_package_data_primitive_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::LorentzFactor<DataVector>>;
  using dg_package_data_volume_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState>;
  // The equation of state is needed in dg_boundary_terms to build the
  // fast-magnetosonic bounds for the scalar/MHD split (as in Hll and Hllem).
  using dg_boundary_terms_volume_tags =
      tmpl::list<hydro::Tags::GrmhdEquationOfState>;

  double dg_package_data(
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> packaged_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> packaged_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_normal_dot_flux_tilde_b,
      gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<Scalar<DataVector>*> packaged_metric_flatness,
      gsl::not_null<Scalar<DataVector>*> packaged_rest_mass_density,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> packaged_pressure,
      gsl::not_null<Scalar<DataVector>*> packaged_lorentz_factor,
      gsl::not_null<Scalar<DataVector>*> packaged_specific_internal_energy,

      const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
      const Scalar<DataVector>& tilde_tau,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_phi,

      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
      const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
      const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
      const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,

      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& temperature,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& lorentz_factor,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& normal_vector,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state)
      const;

  static void dg_boundary_terms(
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          boundary_correction_tilde_b,
      gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
      const Scalar<DataVector>& tilde_d_int,
      const Scalar<DataVector>& tilde_ye_int,
      const Scalar<DataVector>& tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
      const Scalar<DataVector>& tilde_phi_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_int,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
      const Scalar<DataVector>& largest_outgoing_char_speed_int,
      const Scalar<DataVector>& largest_ingoing_char_speed_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
      const Scalar<DataVector>& metric_flatness_int,
      const Scalar<DataVector>& rest_mass_density_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_int,
      const Scalar<DataVector>& pressure_int,
      const Scalar<DataVector>& lorentz_factor_int,
      const Scalar<DataVector>& specific_internal_energy_int,
      const Scalar<DataVector>& tilde_d_ext,
      const Scalar<DataVector>& tilde_ye_ext,
      const Scalar<DataVector>& tilde_tau_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
      const Scalar<DataVector>& tilde_phi_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_s_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          normal_dot_flux_tilde_b_ext,
      const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
      const Scalar<DataVector>& largest_outgoing_char_speed_ext,
      const Scalar<DataVector>& largest_ingoing_char_speed_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
      const Scalar<DataVector>& metric_flatness_ext,
      const Scalar<DataVector>& rest_mass_density_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity_ext,
      const Scalar<DataVector>& pressure_ext,
      const Scalar<DataVector>& lorentz_factor_ext,
      const Scalar<DataVector>& specific_internal_energy_ext,
      dg::Formulation dg_formulation,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

 private:
  friend bool operator==(const Hlld& lhs, const Hlld& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Hlld& lhs, const Hlld& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
