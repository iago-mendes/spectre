// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Smooth (double shear layer) Kelvin-Helmholtz initial data following
 * Beckwith & Stone (2011) and Mattia & Mignone (2022), Section 4.5.
 *
 * In contrast to \ref KhInstability, which uses a discontinuous strip, this
 * class uses a smooth `tanh` shear profile, which avoids the unphysical states a
 * sharp solver can produce at the discontinuity.  Two shear layers sit at
 * \f$y = \pm y_0\f$ (`StripHalfWidth`) of thickness \f$a\f$
 * (`TransitionThickness`) in a domain periodic in all directions.  The
 * horizontal velocity is
 *
 * \f{align*}
 * v_x(y) = v_\text{sh}\left[\tanh\!\left(\frac{y - y_0}{a}\right)
 *        - \tanh\!\left(\frac{y + y_0}{a}\right) + 1\right],
 * \f}
 *
 * so \f$v_x = -v_\text{sh}\f$ in the central band \f$|y| < y_0\f$ and
 * \f$v_x = +v_\text{sh}\f$ outside.  The density is tied to the velocity,
 *
 * \f{align*}
 * \rho(y) = \tfrac{1}{2}(\rho_l + \rho_h)
 *         + \tfrac{1}{2}(\rho_h - \rho_l)\,\frac{v_x}{v_\text{sh}},
 * \f}
 *
 * (so \f$\rho = \rho_l\f$ in the central band and \f$\rho_h\f$ outside).  The
 * instability is seeded with a transverse velocity localized at the shear
 * layers,
 *
 * \f{align*}
 * v_y(x, y) = \mathrm{sign}(y)\,A_0\,v_\text{sh}\,\sin(2\pi x)\,
 *   \exp\!\left[-\frac{(y - \mathrm{sign}(y)\,y_0)^2}{2\sigma^2}\right].
 * \f}
 *
 * The pressure is uniform, the EoS is an ideal fluid, and a uniform magnetic
 * field can be added.
 */
class BeckwithStoneKhInstability
    : public evolution::initial_data::InitialData,
      public MarkAsAnalyticData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<BeckwithStoneKhInstability> {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  /// The adiabatic index of the fluid.
  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the fluid."};
  };

  /// The shear velocity \f$v_\text{sh}\f$ (the flow speed far from the layers).
  struct ShearVelocity {
    using type = double;
    static constexpr Options::String help = {
        "The asymptotic shear velocity along x."};
  };

  /// The half-separation \f$y_0\f$ of the two shear layers.
  struct StripHalfWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Half the separation of the two shear layers (layers at +/- this)."};
  };

  /// The thickness \f$a\f$ of each shear layer.
  struct TransitionThickness {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The thickness of the tanh shear layers."};
  };

  /// The mass density \f$\rho_h\f$ outside the central band.
  struct UpperDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The (higher) mass density outside the central band."};
  };

  /// The mass density \f$\rho_l\f$ in the central band.
  struct LowerDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The (lower) mass density in the central band."};
  };

  /// The initial (constant) pressure of the fluid.
  struct Pressure {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The initial (constant) pressure."};
  };

  /// The amplitude \f$A_0\f$ of the transverse-velocity perturbation.
  struct PerturbAmplitude {
    using type = double;
    static constexpr Options::String help = {
        "The amplitude of the perturbation."};
  };

  /// The width \f$\sigma\f$ of the perturbation envelope.
  struct PerturbWidth {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "The characteristic length for the width of the perturbation."};
  };

  /// The uniform magnetic field.
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {"The uniform magnetic field."};
  };

  using options =
      tmpl::list<AdiabaticIndex, ShearVelocity, StripHalfWidth,
                 TransitionThickness, UpperDensity, LowerDensity, Pressure,
                 PerturbAmplitude, PerturbWidth, MagneticField>;

  static constexpr Options::String help = {
      "Smooth (tanh double shear layer) magnetized KH instability initial data "
      "(Beckwith & Stone 2011; Mattia & Mignone 2022)."};

  BeckwithStoneKhInstability() = default;
  BeckwithStoneKhInstability(const BeckwithStoneKhInstability& /*rhs*/) =
      default;
  BeckwithStoneKhInstability& operator=(
      const BeckwithStoneKhInstability& /*rhs*/) = default;
  BeckwithStoneKhInstability(BeckwithStoneKhInstability&& /*rhs*/) = default;
  BeckwithStoneKhInstability& operator=(BeckwithStoneKhInstability&& /*rhs*/) =
      default;
  ~BeckwithStoneKhInstability() override = default;

  BeckwithStoneKhInstability(double adiabatic_index, double shear_velocity,
                             double strip_half_width, double transition_thickness,
                             double upper_density, double lower_density,
                             double pressure, double perturbation_amplitude,
                             double perturbation_width,
                             const std::array<double, 3>& magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit BeckwithStoneKhInstability(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BeckwithStoneKhInstability);
  /// \endcond

  /// @{
  /// Retrieve the GRMHD variables at a given position.
  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::RestMassDensity<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::RestMassDensity<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::ElectronFraction<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::ElectronFraction<DataType>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::SpecificInternalEnergy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Pressure<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Pressure<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpatialVelocity<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpatialVelocity<DataType, 3>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::MagneticField<DataType, 3>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::MagneticField<DataType, 3>>;

  template <typename DataType>
  auto variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<hydro::Tags::DivergenceCleaningField<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::DivergenceCleaningField<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::LorentzFactor<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::LorentzFactor<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::SpecificEnthalpy<DataType>> /*meta*/)
      const -> tuples::TaggedTuple<hydro::Tags::SpecificEnthalpy<DataType>>;

  template <typename DataType>
  auto variables(const tnsr::I<DataType, 3>& x,
                 tmpl::list<hydro::Tags::Temperature<DataType>> /*meta*/) const
      -> tuples::TaggedTuple<hydro::Tags::Temperature<DataType>> {
    return TemperatureInitialization::variables(
        x, tmpl::list<hydro::Tags::Temperature<DataType>>{});
  }
  /// @}

  /// Retrieve a collection of hydrodynamic variables at position x
  template <typename DataType, typename Tag1, typename Tag2, typename... Tags>
  tuples::TaggedTuple<Tag1, Tag2, Tags...> variables(
      const tnsr::I<DataType, 3>& x,
      tmpl::list<Tag1, Tag2, Tags...> /*meta*/) const {
    return {tuples::get<Tag1>(variables(x, tmpl::list<Tag1>{})),
            tuples::get<Tag2>(variables(x, tmpl::list<Tag2>{})),
            tuples::get<Tags>(variables(x, tmpl::list<Tags>{}))...};
  }

  /// Retrieve the metric variables
  template <typename DataType, typename Tag,
            Requires<tmpl::list_contains_v<
                gr::analytic_solution_tags<3, DataType>, Tag>> = nullptr>
  tuples::TaggedTuple<Tag> variables(const tnsr::I<DataType, 3>& x,
                                     tmpl::list<Tag> /*meta*/) const {
    constexpr double dummy_time = 0.0;
    return background_spacetime_.variables(x, dummy_time, tmpl::list<Tag>{});
  }

  const EquationsOfState::IdealFluid<true>& equation_of_state() const {
    return equation_of_state_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
  double shear_velocity_ = std::numeric_limits<double>::signaling_NaN();
  double strip_half_width_ = std::numeric_limits<double>::signaling_NaN();
  double transition_thickness_ = std::numeric_limits<double>::signaling_NaN();
  double upper_density_ = std::numeric_limits<double>::signaling_NaN();
  double lower_density_ = std::numeric_limits<double>::signaling_NaN();
  double pressure_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_amplitude_ = std::numeric_limits<double>::signaling_NaN();
  double perturbation_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  friend bool operator==(const BeckwithStoneKhInstability& lhs,
                         const BeckwithStoneKhInstability& rhs);

  friend bool operator!=(const BeckwithStoneKhInstability& lhs,
                         const BeckwithStoneKhInstability& rhs);
};
}  // namespace grmhd::AnalyticData
