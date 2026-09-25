// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <optional>
#include <ostream>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
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
/// Which characteristic decomposition the Marquina solver uses.
enum class MarquinaCharacteristicsSystem {
  /// Relativistic hydrodynamics with composition (no magnetic field): the
  /// 6-field hydro+Ye decomposition (an approximation for GRMHD).
  HydroYe,
  /// The full 9-wave GLM-Valencia GRMHD decomposition.
  Mhd
};
std::ostream& operator<<(std::ostream& os, MarquinaCharacteristicsSystem t);

/// How the Marquina solver obtains the characteristic eigenvectors.
enum class MarquinaCharacteristicsMethod {
  /// Always use the analytic closed-form eigenvectors; error if anything goes
  /// wrong (e.g. they are degenerate / not biorthogonal).
  AlwaysAnalytic,
  /// Always use the numeric (per-point eigensolver) eigensystem.
  AlwaysNumeric,
  /// Use the analytic eigenvectors, falling back to the numeric eigensystem
  /// where the analytic ones are too degenerate (not biorthogonal).
  AnalyticWithNumericFallback,
  /// Complementary projection with the SPEED-GAP detector: per point, only the
  /// waves whose speeds actually collapse (gap < DegeneracyTolerance) are
  /// reconstructed by the complement; well-separated waves stay analytic.  So
  /// away from a degeneracy this reduces to AlwaysAnalytic.
  AnalyticWithComplementaryProjection,
  /// Complementary projection applied UNCONDITIONALLY to the collapse-prone
  /// fluid subspace (Alfven-, slow-, entropy, slow+, Alfven+), regardless of
  /// whether those speeds are currently degenerate.  Unlike
  /// AnalyticWithComplementaryProjection this is NOT adaptive: it always lumps
  /// that subspace into one complement vector, so it is more diffusive than the
  /// full analytic decomposition when the modes are well separated, and only
  /// pays off when they collapse.  (The fast and GLM-scalar waves stay analytic.)
  AlwaysComplementaryProjection
};
std::ostream& operator<<(std::ostream& os, MarquinaCharacteristicsMethod t);

/*!
 * \brief The Marquina boundary correction (flux) for the GRMHD GLM-Valencia
 * system, built from a characteristic decomposition.
 *
 * The characteristic decomposition is selected by the `CharacteristicsSystem`
 * (HydroYe or Mhd) and `CharacteristicsMethod` (AlwaysAnalytic, AlwaysNumeric,
 * AnalyticWithNumericFallback, AnalyticWithComplementaryProjection) options.
 */
class Marquina final : public evolution::BoundaryCorrection {
 private:
  // Sized for the larger MHD system (9 waves); the hydro+Ye system uses the
  // leading subset (3 distinct speeds / 6 characteristic fields).
  struct CharacteristicSpeeds : db::SimpleTag {
    using type = tnsr::i<DataVector, 9, Frame::NoFrame>;
  };
  struct LeftCharacteristicFields : db::SimpleTag {
    using type = tnsr::iJ<DataVector, 9, Frame::NoFrame>;
  };
  struct RightCharacteristicFields : db::SimpleTag {
    using type = tnsr::ij<DataVector, 9, Frame::NoFrame>;
  };

 public:
  struct CharacteristicsSystem {
    using type = MarquinaCharacteristicsSystem;
    static constexpr Options::String help = {
        "Which characteristic decomposition to use: HydroYe or Mhd."};
  };
  struct CharacteristicsMethod {
    using type = MarquinaCharacteristicsMethod;
    static constexpr Options::String help = {
        "How to obtain the eigenvectors: AlwaysAnalytic, AlwaysNumeric, "
        "AnalyticWithNumericFallback, or AnalyticWithComplementaryProjection."};
  };
  struct DegeneracyTolerance {
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 1.0; }
    static type suggested_value() { return 1.0e-3; }
    static constexpr Options::String help = {
        "Speed-gap tolerance (on the c_h = 1 speed scale): an Mhd wave is treated "
        "as degenerate -- and reconstructed by the complementary projection "
        "instead of the analytic per-wave decomposition -- when its characteristic "
        "speed lies within this tolerance of another wave's speed (only used with "
        "AnalyticWithComplementaryProjection). The speed gap tracks the degeneracy "
        "directly; a larger tolerance sends more near-degenerate waves to the "
        "robust complement, a smaller one keeps more waves analytic."};
  };
  struct UseModifiedFormula {
    using type = bool;
    static type suggested_value() { return false; }
    static constexpr Options::String help = {
        "Use the MODIFIED Marquina flux formula of Aloy et al. 1999 (ApJS 122, "
        "151) instead of the original Donat-Marquina flux. The original applies "
        "sided upwinding where a wave's characteristic speed has the same sign "
        "on both states, and its Lax-Friedrichs-like viscosity ONLY where the "
        "speed changes sign; the modified formula drops that if-clause and "
        "applies the viscous branch everywhere. It is more dissipative but far "
        "more stable -- our original-form Marquina is exact at t=0.05 on the "
        "|B|x2 stationary contact and then diverges to rho ~ 130 by t=1, while "
        "codes that use the modified formula (Whisky, GENESIS, Ratpenat) are "
        "robust in this regime."};
  };

  using options = tmpl::list<CharacteristicsSystem, CharacteristicsMethod,
                             DegeneracyTolerance, UseModifiedFormula>;
  static constexpr Options::String help = {
      "The Marquina boundary correction for the GRMHD GLM-Valencia system."};

  Marquina() = default;
  Marquina(MarquinaCharacteristicsSystem characteristics_system,
           MarquinaCharacteristicsMethod characteristics_method,
           double degeneracy_tolerance, bool use_modified_formula = false);
  Marquina(MarquinaCharacteristicsSystem characteristics_system,
           MarquinaCharacteristicsMethod characteristics_method);
  Marquina(const Marquina&) = default;
  Marquina& operator=(const Marquina&) = default;
  Marquina(Marquina&&) = default;
  Marquina& operator=(Marquina&&) = default;
  ~Marquina() override = default;

  /// \cond
  explicit Marquina(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Marquina);  // NOLINT
  /// \endcond
  void pup(PUP::er& p) override;  // NOLINT

  std::unique_ptr<BoundaryCorrection> get_clone() const override;

  using dg_package_field_tags =
      tmpl::list<Tags::TildeD, Tags::TildeYe, Tags::TildeTau,
                 Tags::TildeS<Frame::Inertial>, Tags::TildeB<Frame::Inertial>,
                 Tags::TildePhi, ::Tags::NormalDotFlux<Tags::TildeD>,
                 ::Tags::NormalDotFlux<Tags::TildeYe>,
                 ::Tags::NormalDotFlux<Tags::TildeTau>,
                 ::Tags::NormalDotFlux<Tags::TildeS<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildeB<Frame::Inertial>>,
                 ::Tags::NormalDotFlux<Tags::TildePhi>, CharacteristicSpeeds,
                 LeftCharacteristicFields, RightCharacteristicFields>;

  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>;
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
  using dg_boundary_terms_volume_tags = tmpl::list<>;

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
      gsl::not_null<tnsr::i<DataVector, 9, Frame::NoFrame>*>
          packaged_characteristic_speeds,
      gsl::not_null<tnsr::iJ<DataVector, 9, Frame::NoFrame>*>
          packaged_left_characteristic_fields,
      gsl::not_null<tnsr::ij<DataVector, 9, Frame::NoFrame>*>
          packaged_right_characteristic_fields,

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

      const Scalar<DataVector>& /*lapse*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*shift*/,
      const tnsr::i<DataVector, 3,
                    Frame::Inertial>& /*spatial_velocity_one_form*/,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& temperature,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& lorentz_factor,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
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
      const tnsr::i<DataVector, 9, Frame::NoFrame>& characteristic_speeds_int,
      const tnsr::iJ<DataVector, 9, Frame::NoFrame>&
          left_characteristic_fields_int,
      const tnsr::ij<DataVector, 9, Frame::NoFrame>&
          right_characteristic_fields_int,
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
      const tnsr::i<DataVector, 9, Frame::NoFrame>& characteristic_speeds_ext,
      const tnsr::iJ<DataVector, 9, Frame::NoFrame>&
          left_characteristic_fields_ext,
      const tnsr::ij<DataVector, 9, Frame::NoFrame>&
          right_characteristic_fields_ext,
      dg::Formulation dg_formulation) const;

 private:
  friend bool operator==(const Marquina& lhs, const Marquina& rhs);

  MarquinaCharacteristicsSystem characteristics_system_{
      MarquinaCharacteristicsSystem::HydroYe};
  MarquinaCharacteristicsMethod characteristics_method_{
      MarquinaCharacteristicsMethod::AlwaysAnalytic};
  double degeneracy_tolerance_{0.5};
  bool use_modified_formula_{false};
};

bool operator==(const Marquina& lhs, const Marquina& rhs);
bool operator!=(const Marquina& lhs, const Marquina& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

/// \cond
template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsSystem> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::BoundaryCorrections::
      MarquinaCharacteristicsSystem
      create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsSystem
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsSystem>::
    create<void>(const Options::Option& options);

template <>
struct Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsMethod> {
  template <typename Metavariables>
  static grmhd::ValenciaDivClean::BoundaryCorrections::
      MarquinaCharacteristicsMethod
      create(const Options::Option& options) {
    return create<void>(options);
  }
};
template <>
grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsMethod
Options::create_from_yaml<
    grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsMethod>::
    create<void>(const Options::Option& options);
/// \endcond
