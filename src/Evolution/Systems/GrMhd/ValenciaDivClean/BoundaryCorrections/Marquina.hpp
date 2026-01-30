// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <optional>
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
/// \endcond

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
/*!
 * TO-DO
 */
class Marquina final : public evolution::BoundaryCorrection {
 private:
  struct CharacteristicSpeeds : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::NoFrame>;
  };
  struct LeftCharacteristicFields : db::SimpleTag {
    using type = tnsr::iJ<DataVector, 6, Frame::NoFrame>;
  };
  struct RightCharacteristicFields : db::SimpleTag {
    using type = tnsr::ij<DataVector, 6, Frame::NoFrame>;
  };

 public:
  using options = tmpl::list<>;
  static constexpr Options::String help = {"TO-DO"};

  Marquina() = default;
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

  static double dg_package_data(
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
      gsl::not_null<tnsr::i<DataVector, 3, Frame::NoFrame>*>
          packaged_characteristic_speeds,
      gsl::not_null<tnsr::iJ<DataVector, 6, Frame::NoFrame>*>
          packaged_left_characteristic_fields,
      gsl::not_null<tnsr::ij<DataVector, 6, Frame::NoFrame>*>
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
      const Scalar<DataVector>& /*temperature*/,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& specific_internal_energy,
      const Scalar<DataVector>& pressure,
      const Scalar<DataVector>& lorentz_factor,

      const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
      const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
      const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
      /*mesh_velocity*/,
      const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
      const EquationsOfState::EquationOfState<true, 3>& equation_of_state);

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
      const tnsr::i<DataVector, 3, Frame::NoFrame>& characteristic_speeds_int,
      const tnsr::iJ<DataVector, 6, Frame::NoFrame>&
          left_characteristic_fields_int,
      const tnsr::ij<DataVector, 6, Frame::NoFrame>&
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
      const tnsr::i<DataVector, 3, Frame::NoFrame>& characteristic_speeds_ext,
      const tnsr::iJ<DataVector, 6, Frame::NoFrame>&
          left_characteristic_fields_ext,
      const tnsr::ij<DataVector, 6, Frame::NoFrame>&
          right_characteristic_fields_ext,
      dg::Formulation dg_formulation);
};

bool operator==(const Marquina& lhs, const Marquina& rhs);
bool operator!=(const Marquina& lhs, const Marquina& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
