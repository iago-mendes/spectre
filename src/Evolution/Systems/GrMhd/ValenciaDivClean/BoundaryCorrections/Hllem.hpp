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
namespace Options {
class Option;
template <typename T>
struct create_from_yaml;
}  // namespace Options
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/// Which intermediate waves the HLLEM anti-diffusion restores on top of HLL.
/// This is the central knob of the solver: Mattia & Mignone (2022) show (their
/// Fig 13) that for the Kelvin-Helmholtz instability restoring contact+slow
/// resolves the secondary vortices while contact+Alfven smooths them out.
enum class HllemWaves {
  /// Only the contact/entropy wave (HLLC-like).
  Contact,
  /// Contact + the two Alfven (rotational) waves.
  ContactAlfven,
  /// Contact + the two slow-magnetosonic waves.
  ContactSlow,
  /// All five internal waves (contact + 2 Alfven + 2 slow).
  All
};
std::ostream& operator<<(std::ostream& os, HllemWaves waves);

/*!
 * \brief The HLLEM Riemann solver (Einfeldt-Munz-Roe-Sjogreen 1991; Dumbser &
 * Balsara 2016) for the GRMHD GLM-Valencia system.
 *
 * HLLEM starts from the diffusive two-wave HLL flux and adds an anti-diffusive
 * correction that restores selected intermediate waves via the characteristic
 * decomposition,
 * \f{align*}{
 *   G_\text{HLLEM} = G_\text{HLL}
 *     - \frac{\lambda_+\lambda_-}{\lambda_+-\lambda_-}
 *       \sum_{k\in\text{restored}} \delta_k\,(\ell_k\cdot\Delta U)\,r_k ,
 * \f}
 * where \f$r_k,\ell_k\f$ are the right/left eigenvectors of wave \f$k\f$,
 * \f$\Delta U=U_\text{ext}-U_\text{int}\f$, and \f$\delta_k\f$ is the Einfeldt
 * anti-diffusion coefficient. Unlike HLLC/HLLD (which reconstruct the fan
 * nonlinearly and cannot restore slow waves), HLLEM restores any wave for which
 * an eigenvector is available -- including the slow modes -- which is why it is
 * the natural vehicle for the slow-mode Kelvin-Helmholtz test.
 *
 * This uses the (compact, corrected) GRMHD eigenvectors of
 * `grmhd::ValenciaDivClean::characteristic_eigenvectors_mhd`, so its quality is
 * a direct function of the eigenvector quality -- the point of the comparison
 * against classical (Anile/Komissarov/Anton) HLLEM. The eigensystem is
 * evaluated at the arithmetic-average state so the flux is conservative. Near a
 * degeneracy the collapse-prone eigenvectors are handled by the complementary
 * projection (as in `Marquina`); the fan is reconstructed assuming flat space
 * (the regime of the relativistic M&M tests) with an HLL fallback for curved
 * backgrounds and for non-finite results.
 */
class Hllem final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct InterfaceUnitNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  struct MetricFlatness : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  struct WavesToRestore {
    using type = HllemWaves;
    static constexpr Options::String help = {
        "Which intermediate waves the anti-diffusion restores: Contact, "
        "ContactAlfven, ContactSlow, or All."};
  };
  struct DegeneracyTolerance {
    static constexpr Options::String help = {
        "Speed-gap below which neighbouring waves are treated as degenerate "
        "and "
        "handled by the complementary projection."};
    using type = double;
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
      tmpl::list<WavesToRestore, DegeneracyTolerance,
                 MagneticFieldMagnitudeForHydro, LightSpeedDensityCutoff>;
  static constexpr Options::String help = {
      "Computes the HLLEM boundary correction term for the GRMHD system."};

  Hllem() = default;
  Hllem(const Hllem&) = default;
  Hllem& operator=(const Hllem&) = default;
  Hllem(Hllem&&) = default;
  Hllem& operator=(Hllem&&) = default;
  ~Hllem() override = default;

  Hllem(HllemWaves waves_to_restore, double degeneracy_tolerance,
        double magnetic_field_magnitude_for_hydro,
        double light_speed_density_cutoff);

  /// \cond
  explicit Hllem(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hllem);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

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
  // eigensystem at the averaged interface state.
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

  void dg_boundary_terms(
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
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state)
      const;

 private:
  friend bool operator==(const Hllem& lhs, const Hllem& rhs);

  HllemWaves waves_to_restore_{HllemWaves::All};
  double degeneracy_tolerance_{std::numeric_limits<double>::signaling_NaN()};
  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Hllem& lhs, const Hllem& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

/// \cond
template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves create(
      const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::HllemWaves>::
    create<void>(const Options::Option& options);
/// \endcond
