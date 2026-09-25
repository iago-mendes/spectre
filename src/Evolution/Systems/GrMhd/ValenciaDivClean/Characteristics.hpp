// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

namespace grmhd {
namespace ValenciaDivClean {
/// Number of times a denominator floor in the characteristic decomposition has
/// actually BOUND since the last reset. A floor that never binds is inert; one
/// that binds means the eigensystem is being evaluated where it is not
/// meaningful (atmosphere cell, or a genuine wave degeneracy).
size_t denominator_floor_count();
/// \see denominator_floor_count
void reset_denominator_floor_count();

/// @{
/*!
 * \brief Compute the characteristic speeds for the Valencia formulation of
 * GRMHD with divergence cleaning.
 *
 * Obtaining the exact form of the characteristic speeds involves the solution
 * of a nontrivial quartic equation for the fast and slow modes. Here we make
 * use of a common approximation in the literature (e.g. \cite Gammie2003)
 * where the resulting characteristic speeds are analogous to those of the
 * Valencia formulation of the 3-D relativistic Euler system
 * (see RelativisticEuler::Valencia::characteristic_speeds),
 *
 * \f{align*}
 * \lambda_2 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{3, 4, 5, 6, 7} &= \alpha v_n - \beta_n,\\
 * \lambda_{8} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * with the substitution
 *
 * \f{align*}
 * c_s^2 \longrightarrow c_s^2 + v_A^2(1 - c_s^2)
 * \f}
 *
 * in the definition of \f$\Lambda^\pm\f$. Here \f$v_A\f$ is the Alfvén
 * speed. In addition, two more speeds corresponding to the divergence cleaning
 * mode and the longitudinal magnetic field are added,
 *
 * \f{align*}
 * \lambda_1 = -\alpha - \beta_n,\\
 * \lambda_9 = \alpha - \beta_n.
 * \f}
 *
 * \note The ordering assumed here is such that, in the Newtonian limit,
 * the exact expressions for \f$\lambda_{2, 8}\f$, \f$\lambda_{3, 7}\f$,
 * and \f$\lambda_{4, 6}\f$ should reduce to the
 * corresponding fast modes, Alfvén modes, and slow modes, respectively.
 * See \cite Dedner2002 for a detailed description of the hyperbolic
 * characterization of Newtonian MHD.  In terms of the primitive variables:
 *
 * \f{align*}
 * v^2 &= \gamma_{mn} v^m v^n \\
 * c_s^2 &= \frac{1}{h} \left[ \left( \frac{\partial p}{\partial \rho}
 * \right)_\epsilon +
 * \frac{p}{\rho^2} \left(\frac{\partial p}{\partial \epsilon}
 * \right)_\rho \right] \\
 * v_A^2 &= \frac{b^2}{b^2 + \rho h} \\
 * b^2 &= \frac{1}{W^2} \gamma_{mn} B^m B^n + \left( \gamma_{mn} B^m v^n
 * \right)^2
 * \f}
 *
 * where \f$\gamma_{mn}\f$ is the spatial metric, \f$\rho\f$ is the rest
 * mass density, \f$W = 1/\sqrt{1-v_i v^i}\f$ is the Lorentz factor, \f$h = 1 +
 * \epsilon + \frac{p}{\rho}\f$ is the specific enthalpy, \f$v^i\f$ is the
 * spatial velocity, \f$\epsilon\f$ is the specific internal energy, \f$p\f$ is
 * the pressure, and \f$B^i\f$ is the spatial magnetic field measured by an
 * Eulerian observer.
 */

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds_approximate_mhd(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <size_t ThermodynamicDim>
void characteristic_speeds_approximate_mhd(
    gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/*!
 * \brief Labels for the characteristic speeds of the relativistic hydrodynamics
 * system.
 *
 * \see `grmhd::ValenciaDivClean::characteristic_speeds_hydro`
 */
enum HydroSpeed : uint32_t {
  NormalDotVelocity = 0,
  LambdaPlus = 1,
  LambdaMinus = 2
};
/// @}

/// @{
/*!
 * \brief Labels for the analytic right (modes) / left (projectors) eigenvectors
 * of the relativistic hydrodynamics system, used by
 * `characteristic_eigenvectors_hydro`.
 */
enum HydroVectorR : uint32_t {
  R1 = 0,
  R2 = 1,
  R3 = 2,
  R4 = 3,
  Rplus = 4,
  Rminus = 5,
};

enum HydroVectorL : uint32_t {
  L1 = 0,
  L2 = 1,
  L3 = 2,
  L4 = 3,
  Lplus = 4,
  Lminus = 5,
};
/// @}

/// @{
/*!
 * \brief Compute the characteristic speeds for the relativistic hydrodynamics
 * system in the Eulerian frame.
 *
 * These are the eigenvalues of the flux Jacobian projected along a spatial
 * direction with unit normal \f$ s_i\f$, measured by an Eulerian observer.
 * They consist of four degenerate modes and two non-degenerate modes:
 *
 * \f{align*}
 * \lambda_\mathrm{deg} &= v_n ,\\
 * \lambda_{\pm} &=
 *   \frac{ (1 - c_s^2)\,v_n \pm c_s\,d / W^2 }
 *        { 1 - v^2 c_s^2 }.
 * \f}
 *
 * where
 *
 * \f{align*}
 * d &\equiv
 *   W\,\sqrt{ 1 - v^2 c_s^2 - v_n^2(1 - c_s^2) } .
 * \f}
 *
 * The variables are defined as:
 *
 * \f{align*}
 * v^2 &= \gamma_{mn} v^m v^n ,\\
 * v_n &= v^a s_a ,\\
 * W &= \frac{1}{\sqrt{1 - v^a v_a}} ,\\
 * h &= 1 + \epsilon + \frac{p}{\rho} ,\\
 * c_s^2 &= \frac{1}{h}\left(\chi + \kappa \frac{p}{\rho^2}\right) ,
 * \f}
 *
 * The returned tensor has index 0..2:
 * - `characteristic_speeds.get(0)`: degenerate speed \f$v_n\f$
 * - `characteristic_speeds.get(1)`: acoustic speed \f$\lambda_+\f$
 * - `characteristic_speeds.get(2)`: acoustic speed \f$\lambda_-\f$
 *
 * \see `grmhd::ValenciaDivClean::numerical_characteristics` for how these
 * speeds are paired with characteristic modes and characteristic projectors.
 */

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);

template <size_t ThermodynamicDim>
tnsr::i<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/*!
 * \brief Compute analytic right (modes) and left (projectors) eigenvectors for
 * the relativistic hydrodynamics (no magnetic field, with composition) system.
 *
 * The output conventions match `characteristic_eigenvectors_mhd` /
 * `numerical_characteristics`:
 * - Right eigenvectors / modes: `characteristic_modes.get(i, n)` is the
 *   \f$i\f$th wave eigenvector component \f$n\f$ (`i` follows `HydroVectorR`).
 * - Left eigenvectors / projectors: `characteristic_projectors.get(i, n)`
 *   (`i` follows `HydroVectorL`).
 *
 * Used by the Marquina boundary correction.
 */
template <size_t ThermodynamicDim>
void characteristic_eigenvectors_hydro(
    gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/*!
 * \brief Labels for the characteristic speeds of the relativistic
 * magnetohydrodynamics system.
 *
 * We order the speeds as follows:
 * \begin{equation}
 *   \lambda_\text{scalar}^-
 *   \leq \lambda_\text{fastB}^-
 *   \leq \lambda_\text{Alfven}^-
 *   \leq \lambda_\text{slowB}^-
 *   \leq \lambda_\text{entropy}
 *   \leq \lambda_\text{slowB}^+
 *   \leq \lambda_\text{Alfven}^+
 *   \leq \lambda_\text{fastB}^+
 *   \leq \lambda_\text{scalar}^+
 * \end{equation}
 *
 * \see `grmhd::ValenciaDivClean::characteristic_speeds_mhd`
 */
enum MhdSpeed : uint32_t {
  ScalarMinus = 0,
  FastMagnetosonicMinus = 1,
  AlfvenMinus = 2,
  SlowMagnetosonicMinus = 3,
  Entropy = 4,
  SlowMagnetosonicPlus = 5,
  AlfvenPlus = 6,
  FastMagnetosonicPlus = 7,
  ScalarPlus = 8
};
/// @}

/*!
 * \brief Choice of algorithm for the slow magnetosonic speeds.
 */
enum class SlowMagnetosonicSpeedMethod {
  Toms748,
  ReducedQuadratic,
  ReducedQuadraticThenNewton
};

/// @{
/*!
 * \brief Coefficients of the quartic polynomial whose roots give the
 * magnetosonic characteristic speeds.
 *
 * These coefficients are calculated by expanding Eq. (2.30) in
 * arXiv:2511.13837v1 in the Eulerian frame and diving by the coefficient of the
 * quartic term.
 *
 * We order the coefficients as follows:
 * \begin{equation}
 *   x^4 + c_3 x^3 + c_2 x^2 + c_1 x + c_0 = 0,
 * \end{equation}
 * where $c_i$ is given by `quartic_coefficients.get(i)`.
 *
 * \see `grmhd::ValenciaDivClean::characteristic_speeds_mhd`
 */
void magnetosonic_quartic_coefficients(
    gsl::not_null<tnsr::i<DataVector, 4>*> quartic_coefficients,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& normal_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& normal_magnetic_field,
    const Scalar<DataVector>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataVector>& magnetic_field_squared,
    const Scalar<DataVector>& comoving_magnetic_field_squared);
/// @}

/// @{
/*!
 * \brief Find a magnetosonic speed by Newton-Raphson rootfinding.
 *
 * \note `magnetosonic_speed` should be initialized to a initial guess close to
 * the desired root. For example, +1.0 for the fast magnetosonic speed in the
 * positive direction and -1.0 for the fast magnetosonic speed in the negative
 * direction.
 *
 * \see `grmhd::ValenciaDivClean::magnetosonic_quartic_coefficients` for how to
 * specify the quartic coefficients.
 */
void find_magnetosonic_speed_from_quartic(
    gsl::not_null<DataVector*> magnetosonic_speed,
    const tnsr::i<DataVector, 4>& quartic_coefficients);
/// @}

/// @{
/*!
 * \brief Compute the characteristic speeds for the relativistic
 * magnetohydrodynamics system in the Eulerian frame.
 *
 * TO-DO
 *
 * \see `grmhd::ValenciaDivClean::numerical_characteristics` for how these
 * speeds are paired with characteristic modes and characteristic projectors.
 */
template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    gsl::not_null<tnsr::i<DataVector, 9>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    SlowMagnetosonicSpeedMethod slow_speed_method =
        SlowMagnetosonicSpeedMethod::ReducedQuadratic);
/// @}

/// @{
/*!
 * \brief Compute analytical right and left eigenvectors for the GRMHD
 * Valencia divergence-cleaning system using precomputed characteristic speeds.
 *
 * The input `characteristic_speeds` follows the `MhdSpeed` ordering.
 *
 * The output conventions match `numerical_characteristics`:
 * - Right eigenvectors / modes: `characteristic_modes.get(i, n)` is the
 *   \f$i\f$th wave eigenvector component \f$n\f$.
 * - Left eigenvectors / projectors:
 *   `characteristic_projectors.get(i, n)`.
 *
 * Component ordering for eigenvector entries \f$n\f$ is
 * \f$[S_x,S_y,S_z,B_x,B_y,B_z,D,\tau,\phi]\f$.
 */
template <size_t ThermodynamicDim>
void characteristic_eigenvectors_mhd(
    gsl::not_null<tnsr::ij<DataVector, 9>*> characteristic_modes,
    gsl::not_null<tnsr::IJ<DataVector, 9>*> characteristic_projectors,
    const tnsr::i<DataVector, 9>& characteristic_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state,
    bool skip_fluid_subspace = false);
/// @}

/**
 * \brief Compute the characteristic matrix for relativistic hydrodynamics +
 * composition ($Y_e$), in a given direction.
 *
 * \see `grmhd::ValenciaDivClean::numerical_characteristics` for the
 * conventions relating this matrix to characteristic speeds, modes, and
 * projectors.
 */
template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/**
 * \brief Compute the conserved-variable GLM-Valencia (divergence-cleaning)
 * characteristic matrix in a given direction (lapse = 1, shift = 0).
 *
 * This is the matrix "As" of the analytic derivation (flux Jacobian dotted with
 * the spatial normal) for variables [S_i, B^i, D, tau, phi].  It is used to
 * verify that the analytic MHD characteristic speeds and eigenvectors solve the
 * eigenproblem A.R = lambda R (and L.A = lambda L).
 */
template <size_t ThermodynamicDim>
void flux_jacobian_mhd(
    gsl::not_null<tnsr::iJ<DataVector, 9>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

/// @{
/**
 * \brief Compute numerical characteristic speeds, characteristic projectors
 * (left eigenvectors), and characteristic modes (right eigenvectors) for
 * a given characterisitc matrix.
 *
 * \note The system is selected by the template parameter `MatrixSize`: a value
 * of 6 builds the eigensystem for relativistic hydrodynamics with composition
 * dependence ($Y_e$, no magnetic field, calling `flux_jacobian_hydro`), while a
 * value of 9 builds it for the full GLM-Valencia GRMHD system (calling
 * `flux_jacobian_mhd`).  The `magnetic_field` argument is ignored for the hydro
 * case.  (This explicit selector is a stopgap; a system trait will replace it.)
 *
 * We label the characteristic matrix in a given direction, $A_{m}{}^{n}$, such
 * that the index $m$ labels rows and the index $n$ labels columns.
 *
 * With this choice of indices, the $i$th left eigensystem is given by
 * \begin{equation}
 *   \sum_m L^{im} A_{m}{}^{n} = \lambda_{(i)} L^{in},
 * \end{equation}
 * where $\lambda_{(i)}$ is the $i$th eigenvalue / characteristic speed.
 * Similarly, the $i$th right eigensystem is given by
 * \begin{equation}
 *   \sum_n R_{in} A_{m}{}^{n} = \lambda_{(i)} R_{im}.
 * \end{equation}
 *
 * We provide the characteristics in the following types:
 * - Eigenvalues $\lambda_{(i)}$: `tnsr::i<DataVector, 6> characteristic_speeds`
 * - Right eigenvectors $R_{in}$: `tnsr::ij<DataVector, 6> characteristic_modes`
 * - Left eigenvectors $L^{im}$: `tnsr::IJ<DataVector, 6>
 *   characteristic_projectors`
 *
 * Note that the indices $i$, $m$, and $n$ are not tensorial. They simply label
 * the eigensystem, rows of the matrix, and columns of the matrix, respectively.
 * We make the conventions (raised or lowered) presented above to clearly
 * distinguish which indices (first or second) of the matrix get contracted
 * with which eigenvectors (left or right).
 *
 * We refer to the left eigenvectors as "projectors" because they are used to
 * project a state into the characteristic basis. Similarly, we refer to the
 * right eigenvectors as "modes" because they are used to reconstruct the state.
 * For example, consider a state vector $U_{m}$ that can be represented in the
 * characteristic basis as
 * \begin{equation}
 *   U_{m} = \sum_i w^{i} R_{im}.
 * \end{equation}
 * Then, the characteristic fields $w^{i}$ can be obtained by projecting the
 * state vector $U_{m}$ into the characteristic basis using the left
 * eigenvectors:
 * \begin{equation}
 *   w^{i} = \sum_m L^{im} U_{m}.
 * \end{equation}
 */
template <size_t MatrixSize, size_t ThermodynamicDim>
void numerical_characteristics(
    gsl::not_null<tnsr::i<DataVector, MatrixSize>*> characteristic_speeds,
    gsl::not_null<tnsr::ij<DataVector, MatrixSize>*> characteristic_modes,
    gsl::not_null<tnsr::IJ<DataVector, MatrixSize>*> characteristic_projectors,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,

    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state);
/// @}

namespace Tags {
/// \brief Compute the characteristic speeds for the Valencia formulation of
/// GRMHD with divergence cleaning.
///
/// \details see grmhd::ValenciaDivClean::characteristic_speeds
struct CharacteristicSpeedsCompute : Tags::CharacteristicSpeeds,
                                     db::ComputeTag {
  using base = Tags::CharacteristicSpeeds;
  using argument_tags =
      tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 hydro::Tags::MagneticField<DataVector, 3>,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<3>>,
                 hydro::Tags::GrmhdEquationOfState>;

  using volume_tags = tmpl::list<hydro::Tags::GrmhdEquationOfState>;

  using return_type = std::array<DataVector, 9>;

  template <size_t ThermodynamicDim>
  void function(gsl::not_null<return_type*> result,
                const Scalar<DataVector>& rest_mass_density,
                const Scalar<DataVector>& /* electron_fraction */,
                const Scalar<DataVector>& specific_internal_energy,
                const Scalar<DataVector>& specific_enthalpy,
                const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
                const Scalar<DataVector>& lorentz_factor,
                const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
                const Scalar<DataVector>& lapse,
                const tnsr::I<DataVector, 3>& shift,
                const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
                const tnsr::i<DataVector, 3>& unit_normal,
                const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
                    equation_of_state);
};

struct LargestCharacteristicSpeed : db::SimpleTag {
  using type = double;
};

struct ComputeLargestCharacteristicSpeed : db::ComputeTag,
                                           LargestCharacteristicSpeed {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(gsl::not_null<double*> speed,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3>& shift,
                       const tnsr::ii<DataVector, 3>& spatial_metric) {
    const auto shift_magnitude = magnitude(shift, spatial_metric);
    *speed = max(get(shift_magnitude) + get(lapse));
  }
};
}  // namespace Tags
}  // namespace ValenciaDivClean
}  // namespace grmhd
