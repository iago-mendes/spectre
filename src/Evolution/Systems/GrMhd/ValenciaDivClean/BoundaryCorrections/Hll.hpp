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
 * \brief An HLL Riemann solver
 *
 * Let \f$U\f$ be the evolved variable, \f$F^i\f$ the flux, and \f$n_i\f$ be the
 * outward directed unit normal to the interface. Denoting \f$F := n_i F^i\f$,
 * the HLL boundary correction is \cite Harten1983
 *
 * \f{align*}
 * G_\text{HLL} = \frac{\lambda_\text{max} F_\text{int} +
 * \lambda_\text{min} F_\text{ext}}{\lambda_\text{max} - \lambda_\text{min}}
 * - \frac{\lambda_\text{min}\lambda_\text{max}}{\lambda_\text{max} -
 *   \lambda_\text{min}} \left(U_\text{int} - U_\text{ext}\right)
 * \f}
 *
 * where "int" and "ext" stand for interior and exterior.
 * \f$\lambda_\text{min}\f$ and \f$\lambda_\text{max}\f$ are defined as
 *
 * \f{align*}
 * \lambda_\text{min} &=
 * \text{min}\left(\lambda^{-}_\text{int},-\lambda^{+}_\text{ext}, 0\right) \\
 * \lambda_\text{max} &=
 * \text{max}\left(\lambda^{+}_\text{int},-\lambda^{-}_\text{ext}, 0\right)
 * \f}
 *
 * where \f$\lambda^{+}\f$ (\f$\lambda^{-}\f$) is the largest characteristic
 * speed in the outgoing (ingoing) direction. Note the minus signs in front of
 * \f$\lambda^{\pm}_\text{ext}\f$, which is because an outgoing speed w.r.t. the
 * neighboring element is an ingoing speed w.r.t. the local element, and vice
 * versa. Similarly, the \f$F_{\text{ext}}\f$ term in \f$G_\text{HLL}\f$ has a
 * positive sign because the outward directed normal of the neighboring element
 * has the opposite sign, i.e. \f$n_i^{\text{ext}}=-n_i^{\text{int}}\f$.
 *
 * The characteristic/signal speeds are given in the documentation for
 * `grmhd::ValenciaDivClean::characteristic_speeds()`. Since the fluid is
 * travelling slower than the speed of light, the speeds we are interested in
 * are
 *
 * \f{align*}{
 *   \lambda^{\pm}&=\pm\alpha-\beta^i n_i,
 * \f}
 *
 * which correspond to the divergence cleaning field.
 *
 * The divergence cleaning pair \f$(\tilde{\Phi}, n_i \tilde{B}^i)\f$
 * decouples from the rest of the system in the Riemann problem at a face, on
 * any background: projected on the unit normal its fluxes are
 * \f$\alpha \tilde{\Phi} - \beta^n n_i \tilde{B}^i\f$ and
 * \f$\alpha n_i \tilde{B}^i - \beta^n \tilde{\Phi}\f$, with
 * \f$\beta^n = \beta^i n_i\f$. It propagates at the speed of light, so only
 * it uses the light-speed bounds above. All other variables use the
 * fast-magnetosonic speeds of the two sides, combined by the same min/max.
 * Each side computes them with its own lapse, shift and spatial metric,
 *
 * \f{align*}{
 *   \lambda^{\pm}_\text{fast} = \frac{\alpha}{1 - v^2 a^2}\left[v_n (1 - a^2)
 *   \pm \sqrt{a^2 (1 - v^2) \left(1 - v^2 a^2 - v_n^2 (1 - a^2)\right)}
 *   \right] - \beta^n,
 * \f}
 *
 * with \f$v_n = v^i n_i\f$, \f$v^2 = v^i v_i\f$ and the estimate
 * \f$a^2 = c_s^2 + v_A^2 (1 - c_s^2)\f$, \f$v_A^2 = b^2 / (b^2 + \rho h)\f$,
 * \f$b^2 = B^2 (1 - v^2) + (B^i v_i)^2\f$ and
 * \f$B^i = \tilde{B}^i / \sqrt{\gamma}\f$. Dropping the dependence on the
 * direction of the field makes this an upper bound on the fast speed. The
 * specific internal energy and the pressure come from the equation of state.
 * Without a magnetic field the fast speeds are the hydro speeds. Each side
 * raises its unit normal with its own spatial metric. The magnetic field is
 * split along the average \f$m\f$ of the two sides' unit normals, with
 * \f$n_i n^j = m_i m^j / (m_k m^k)\f$, so that the correction stays
 * antisymmetric under the exchange of the two sides when their normals
 * differ. The MHD fluxes of the two sides are used as packaged, each with its
 * own
 * \f$n_i \tilde{B}^i\f$: a jump in the normal field is a stationary mode of
 * the MHD subsystem, and the bounds always include zero, so it lies inside the
 * fan. The face value of the divergence cleaning pair is therefore not
 * substituted into the fluxes as in \cite Dedner2002.
 *
 * \note
 * - In the strong form the `dg_boundary_terms` function returns
 *   \f$G - F_\text{int}\f$
 * - For either \f$\lambda_\text{min} = 0\f$ or \f$\lambda_\text{max} = 0\f$
 *   (i.e. all characteristics move in the same direction) the HLL boundary
 *   correction reduces to pure upwinding.
 * - Some references use \f$S\f$ instead of \f$\lambda\f$ for the
 *   signal/characteristic speeds
 */
class Hll final : public evolution::BoundaryCorrection {
 public:
  struct LargestOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LargestIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// @{
  /// The fast-magnetosonic speeds of this side, used for all variables but
  /// the divergence cleaning pair.
  struct FastOutgoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct FastIngoingCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  /// @}
  /// @{
  /// The unit normal covector and vector of this side; `dg_boundary_terms`
  /// splits the magnetic field along the average of the two sides' normals.
  struct InterfaceUnitNormal : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  struct InterfaceUnitNormalVector : db::SimpleTag {
    using type = tnsr::I<DataVector, 3, Frame::Inertial>;
  };
  /// @}

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
      "Computes the HLL boundary correction term for the GRMHD system."};

  Hll() = default;
  Hll(const Hll&) = default;
  Hll& operator=(const Hll&) = default;
  Hll(Hll&&) = default;
  Hll& operator=(Hll&&) = default;
  ~Hll() override = default;

  Hll(double magnetic_field_magnitude_for_hydro,
      double light_speed_density_cutoff);

  /// \cond
  explicit Hll(CkMigrateMessage* /*unused*/);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Hll);  // NOLINT
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
                 ::Tags::NormalDotFlux<Tags::TildePhi>,
                 LargestOutgoingCharSpeed, LargestIngoingCharSpeed,
                 FastOutgoingCharSpeed, FastIngoingCharSpeed,
                 InterfaceUnitNormal, InterfaceUnitNormalVector>;
  using dg_package_data_temporary_tags = tmpl::list<
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      gr::Tags::SpatialMetric<DataVector, 3>>;
  using dg_package_data_primitive_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Temperature<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>>;
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
      gsl::not_null<Scalar<DataVector>*> packaged_largest_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_largest_ingoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_fast_outgoing_char_speed,
      gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          packaged_interface_unit_normal,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          packaged_interface_unit_normal_vector,

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
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& electron_fraction,
      const Scalar<DataVector>& temperature,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,

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
      const Scalar<DataVector>& fast_outgoing_char_speed_int,
      const Scalar<DataVector>& fast_ingoing_char_speed_int,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          interface_unit_normal_vector_int,
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
      const Scalar<DataVector>& fast_outgoing_char_speed_ext,
      const Scalar<DataVector>& fast_ingoing_char_speed_ext,
      const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
      const tnsr::I<DataVector, 3, Frame::Inertial>&
          interface_unit_normal_vector_ext,
      dg::Formulation dg_formulation);

 private:
  friend bool operator==(const Hll& lhs, const Hll& rhs);

  double magnetic_field_magnitude_for_hydro_{
      std::numeric_limits<double>::signaling_NaN()};
  double light_speed_density_cutoff_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const Hll& lhs, const Hll& rhs);
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
