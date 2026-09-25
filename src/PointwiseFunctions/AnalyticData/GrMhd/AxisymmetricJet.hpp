// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>

#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::AnalyticData {

/*!
 * \brief Analytic initial data for a magnetized axisymmetric relativistic jet.
 *
 * This is the Fig. 6 test problem of Del Zanna, Bucciantini & Londrillo,
 * A&A **400**, 397 (2003) (arXiv:astro-ph/0210618, `tex/delzannal2.tex`
 * lines 1337-1354): a light, highly relativistic beam injected along the
 * symmetry axis into a dense static ambient medium threaded by a uniform
 * axial magnetic field.
 *
 * The parameters used for that figure are:
 *
 * ```yaml
 * AdiabaticIndex: 1.6666666666666667   # 5/3
 * AmbientDensity: 10.
 * AmbientPressure: 0.01
 * JetDensity: 0.1
 * JetPressure: 0.01
 * JetVelocity: [0., 0.99, 0.]
 * InletRadius: 1.
 * NozzleLength: 1.
 * SmoothingWidth: 0.
 * MagneticField: [0., 0.1, 0.]
 * ```
 *
 * ### Coordinate convention
 *
 * This class is written for the `domain::creators::CartoonCylinder` domain,
 * in which
 *
 * - \f$x^0\f$ is the **cylindrical radius** \f$r \geq 0\f$ (the paper's
 *   \f$r\f$),
 * - \f$x^1\f$ is the **symmetry axis** (the paper's \f$z\f$, the direction the
 *   jet propagates along),
 * - \f$x^2\f$ is the collapsed azimuthal direction.
 *
 * That assignment follows from the Killing vector \f$(0, -z, 0, x)\f$ used for
 * `Spectral::Quadrature::AxialSymmetry` (see `Killing_vector_derivatives()` in
 * `NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp`), which
 * generates rotations about the \f$y\f$ axis on the plane \f$z = 0\f$.
 *
 * \warning This is *not* the convention of `grmhd::AnalyticData::SlabJet`,
 * whose inlet is the region \f$x \leq 0\f$, \f$|y| \leq R\f$. On a
 * `CartoonCylinder` domain no grid point satisfies \f$x \leq 0\f$, so
 * `SlabJet` silently returns uniform ambient values there.
 *
 * ### The nozzle
 *
 * The jet region is
 *
 * \f{align}{
 *   x^0 \leq R_\mathrm{inlet} \quad\mathrm{and}\quad
 *   x^1 \leq L_\mathrm{nozzle},
 * \f}
 *
 * i.e. the paper's \f$0 < z < 1\f$, \f$0 < r < 1\f$ nozzle. The condition on
 * \f$x^1\f$ is deliberately one-sided (no lower bound): the ghost zones of a
 * `DirichletAnalytic` boundary condition on the lower-\f$x^1\f$ face lie at
 * \f$x^1 < 0\f$, and they must carry jet values for material to flow in. A
 * two-sided condition would return ambient there and no inflow would happen.
 *
 * The paper keeps these values constant in time inside the nozzle *volume*.
 * SpECTRE has no interior-forcing mechanism, so the intended use is
 * `grmhd::ValenciaDivClean::BoundaryConditions::DirichletAnalytic` with this
 * class as the prescription on the lower-\f$x^1\f$ face, which is constant in
 * time because this is analytic *data* rather than an analytic solution. That
 * is the same substitution PLUTO makes for the same problem family
 * (`Test_Problems/RMHD/Jet/init.c`, `UserDefBoundary` at `X2_BEG`).
 *
 * ### Smoothing
 *
 * The paper says the nozzle values are "initially smoothed" but inherits the
 * profile from its Paper I (Del Zanna & Bucciantini 2002, A&A **390**, 1177),
 * which does not state it either in any form available here. `SmoothingWidth`
 * therefore defaults to **zero**, giving a sharp nozzle -- which is what PLUTO
 * uses for this family (`prof = (fabs(x1[i]) <= 1.0)`). A positive width
 * replaces each step by \f$\tfrac{1}{2}[1 + \tanh(s/w)]\f$ with \f$s\f$ the
 * signed distance inside the nozzle edge, and exists only as a sensitivity
 * knob. There is no oracle for any nonzero value.
 */
class AxisymmetricJet
    : public evolution::initial_data::InitialData,
      public MarkAsAnalyticData,
      public AnalyticDataBase,
      public hydro::TemperatureInitialization<AxisymmetricJet> {
 public:
  using equation_of_state_type = EquationsOfState::IdealFluid<true>;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {
        "The adiabatic index of the ideal fluid"};
    static double lower_bound() { return 1.; }
  };
  struct AmbientDensity {
    using type = double;
    static constexpr Options::String help = {
        "Fluid rest mass density outside the jet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 10.; }
  };
  struct AmbientPressure {
    using type = double;
    static constexpr Options::String help = {"Fluid pressure outside the jet"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.01; }
  };
  struct AmbientElectronFraction {
    using type = double;
    static constexpr Options::String help = {
        "Electron fraction outside the jet"};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
    static double suggested_value() { return 0.; }
  };
  struct JetDensity {
    using type = double;
    static constexpr Options::String help = {
        "Fluid rest mass density inside the nozzle"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.1; }
  };
  struct JetPressure {
    using type = double;
    static constexpr Options::String help = {
        "Fluid pressure inside the nozzle"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.01; }
  };
  struct JetElectronFraction {
    using type = double;
    static constexpr Options::String help = {
        "Electron fraction inside the nozzle"};
    static double lower_bound() { return 0.; }
    static double upper_bound() { return 1.; }
    static double suggested_value() { return 0.; }
  };
  struct JetVelocity {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Fluid spatial velocity inside the nozzle. For the Del Zanna Fig. 6 "
        "setup on a CartoonCylinder domain this points along the symmetry "
        "axis, i.e. [0., v_z, 0.]"};
  };
  struct InletRadius {
    using type = double;
    static constexpr Options::String help = {
        "Nozzle radius, a bound on the cylindrical radius x^0"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 1.; }
  };
  struct NozzleLength {
    using type = double;
    static constexpr Options::String help = {
        "Nozzle extent along the symmetry axis, an upper bound on x^1. The "
        "condition is one-sided so that DirichletAnalytic ghost zones below "
        "the inflow face also carry jet values"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 1.; }
  };
  struct SmoothingWidth {
    using type = double;
    static constexpr Options::String help = {
        "Width of the tanh smoothing of the nozzle edges. Zero (the default) "
        "gives a sharp nozzle, which is what this problem family is normally "
        "run with and the only setting with an oracle"};
    static double lower_bound() { return 0.; }
    static double suggested_value() { return 0.; }
  };
  struct MagneticField {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "Uniform magnetic field, the same inside and outside the nozzle. For "
        "the Del Zanna Fig. 6 setup this is purely axial, [0., B_z, 0.]"};
    static std::array<double, 3> suggested_value() { return {{0., 0.1, 0.}}; }
  };

  using options = tmpl::list<AdiabaticIndex, AmbientDensity, AmbientPressure,
                             AmbientElectronFraction, JetDensity, JetPressure,
                             JetElectronFraction, JetVelocity, InletRadius,
                             NozzleLength, SmoothingWidth, MagneticField>;

  static constexpr Options::String help = {
      "Magnetized axisymmetric relativistic jet (Del Zanna et al. 2003, "
      "Fig. 6). Intended for a CartoonCylinder domain, where x^0 is the "
      "cylindrical radius and x^1 the symmetry axis."};

  AxisymmetricJet() = default;
  AxisymmetricJet(const AxisymmetricJet& /*rhs*/) = default;
  AxisymmetricJet& operator=(const AxisymmetricJet& /*rhs*/) = default;
  AxisymmetricJet(AxisymmetricJet&& /*rhs*/) = default;
  AxisymmetricJet& operator=(AxisymmetricJet&& /*rhs*/) = default;
  ~AxisymmetricJet() = default;

  AxisymmetricJet(double adiabatic_index, double ambient_density,
                  double ambient_pressure, double ambient_electron_fraction,
                  double jet_density, double jet_pressure,
                  double jet_electron_fraction,
                  std::array<double, 3> jet_velocity, double inlet_radius,
                  double nozzle_length, double smoothing_width,
                  std::array<double, 3> magnetic_field);

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override;

  /// \cond
  explicit AxisymmetricJet(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(AxisymmetricJet);
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

  /// The fraction of the jet state present at position x: 1 inside the
  /// nozzle, 0 outside it, and a tanh blend in between when
  /// `SmoothingWidth` is positive. Exposed so that tests can assert the
  /// selector directly.
  template <typename DataType>
  DataType nozzle_profile(const tnsr::I<DataType, 3>& x) const;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) override;

 private:
  EquationsOfState::IdealFluid<true> equation_of_state_{};
  gr::Solutions::Minkowski<3> background_spacetime_{};

  double ambient_density_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double ambient_electron_fraction_ =
      std::numeric_limits<double>::signaling_NaN();
  double jet_density_ = std::numeric_limits<double>::signaling_NaN();
  double jet_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double jet_electron_fraction_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> jet_velocity_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};
  double inlet_radius_ = std::numeric_limits<double>::signaling_NaN();
  double nozzle_length_ = std::numeric_limits<double>::signaling_NaN();
  double smoothing_width_ = std::numeric_limits<double>::signaling_NaN();
  std::array<double, 3> magnetic_field_{
      {std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN(),
       std::numeric_limits<double>::signaling_NaN()}};

  friend bool operator==(const AxisymmetricJet& lhs,
                         const AxisymmetricJet& rhs);

  friend bool operator!=(const AxisymmetricJet& lhs,
                         const AxisymmetricJet& rhs);
};

}  // namespace grmhd::AnalyticData
