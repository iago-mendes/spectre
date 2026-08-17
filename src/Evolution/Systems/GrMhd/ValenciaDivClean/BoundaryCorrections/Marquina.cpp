// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Marquina.hpp"

#include <algorithm>
#include <cmath>
#include <ostream>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/TensorExpression.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "Options/Options.hpp"
#include "Options/ParseError.hpp"
#include "Options/ParseOptions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/ErrorHandling/FloatingPointExceptions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
std::ostream& operator<<(std::ostream& os,
                         const MarquinaCharacteristicsSystem t) {
  switch (t) {
    case MarquinaCharacteristicsSystem::HydroYe:
      return os << "HydroYe";
    case MarquinaCharacteristicsSystem::Mhd:
      return os << "Mhd";
    default:
      ERROR("Unknown MarquinaCharacteristicsSystem");
  }
}

std::ostream& operator<<(std::ostream& os,
                         const MarquinaCharacteristicsMethod t) {
  switch (t) {
    case MarquinaCharacteristicsMethod::AlwaysAnalytic:
      return os << "AlwaysAnalytic";
    case MarquinaCharacteristicsMethod::AlwaysNumeric:
      return os << "AlwaysNumeric";
    case MarquinaCharacteristicsMethod::AnalyticWithNumericFallback:
      return os << "AnalyticWithNumericFallback";
    case MarquinaCharacteristicsMethod::AnalyticWithComplementaryProjection:
      return os << "AnalyticWithComplementaryProjection";
    case MarquinaCharacteristicsMethod::AlwaysComplementaryProjection:
      return os << "AlwaysComplementaryProjection";
    default:
      ERROR("Unknown MarquinaCharacteristicsMethod");
  }
}

Marquina::Marquina(const MarquinaCharacteristicsSystem characteristics_system,
                   const MarquinaCharacteristicsMethod characteristics_method,
                   const double degeneracy_tolerance, const bool use_modified_formula)
    : characteristics_system_(characteristics_system),
      characteristics_method_(characteristics_method),
      degeneracy_tolerance_(degeneracy_tolerance),
      use_modified_formula_(use_modified_formula) {}

Marquina::Marquina(const MarquinaCharacteristicsSystem characteristics_system,
                   const MarquinaCharacteristicsMethod characteristics_method)
    : Marquina(characteristics_system, characteristics_method, 0.5) {}

Marquina::Marquina(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Marquina::get_clone() const {
  return std::make_unique<Marquina>(*this);
}

void Marquina::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | characteristics_system_;
  p | characteristics_method_;
  p | degeneracy_tolerance_;
  p | use_modified_formula_;
}

double Marquina::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<tnsr::i<DataVector, 9, Frame::NoFrame>*>
        packaged_characteristic_speeds,
    const gsl::not_null<tnsr::iJ<DataVector, 9, Frame::NoFrame>*>
        packaged_left_eigenvectors,
    const gsl::not_null<tnsr::ij<DataVector, 9, Frame::NoFrame>*>
        packaged_right_eigenvectors,

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
    const Scalar<DataVector>& /*pressure*/,
    const Scalar<DataVector>& lorentz_factor,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& /*normal_dot_mesh_velocity*/,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  // Supported:
  //  - AlwaysAnalytic (both systems): closed-form eigenvectors.
  //  - AnalyticWithComplementaryProjection (MHD): handle the degenerate wave
  //    subspace by the complement of the well-conditioned waves (Fedkiw-Merriman-
  //    Osher 1997), avoiding the ill-defined degenerate eigenvectors; see
  //    runs-ai/mhd_marquina/reports/complementary_projection_study.md.
  //  - AlwaysNumeric (MHD): build the decomposition from the per-point numeric
  //    eigensolver (blaze::geev via numerical_characteristics) instead of the
  //    closed-form eigenvectors.  For distinct eigenvalues geev's left/right
  //    eigenvectors are biorthogonal to round-off, so a per-wave rescale
  //    L_i <- L_i/(L_i.R_i) gives L.R = I with no inversion.  This is a PURELY
  //    numeric method: it does NOT fall back to the complementary projection.
  //    At an exact (or double-underflowed) degeneracy geev returns an arbitrary,
  //    non-biorthonormal basis for the repeated eigenspace and L.R != I; in that
  //    case the numeric decomposition is unusable and we ERROR (use
  //    AnalyticWithComplementaryProjection for a degeneracy-robust method).
  //    See runs-ai/mhd_marquina/meetings/2026-07-09/numeric_eigensystem/.
  const bool use_numeric =
      characteristics_method_ == MarquinaCharacteristicsMethod::AlwaysNumeric;
  // AlwaysComplementaryProjection unconditionally complements the collapse-prone
  // fluid subspace (non-adaptive); AnalyticWithComplementaryProjection only
  // complements waves the speed-gap detector flags as degenerate.
  const bool always_complementary_projection =
      characteristics_method_ ==
      MarquinaCharacteristicsMethod::AlwaysComplementaryProjection;
  // Complementary projection (either variant) is used ONLY when its method is
  // explicitly requested; AlwaysAnalytic and AlwaysNumeric never invoke it (they
  // ERROR if their decomposition is unusable at a degeneracy).
  const bool complementary_projection =
      characteristics_method_ ==
          MarquinaCharacteristicsMethod::AnalyticWithComplementaryProjection or
      always_complementary_projection;
  if (characteristics_method_ ==
          MarquinaCharacteristicsMethod::AnalyticWithNumericFallback or
      (use_numeric and
       characteristics_system_ == MarquinaCharacteristicsSystem::HydroYe)) {
    ERROR(
        "Marquina supports CharacteristicsMethod: AlwaysAnalytic and "
        "AnalyticWithComplementaryProjection (both systems), and AlwaysNumeric "
        "(MHD system).  AnalyticWithNumericFallback and numeric HydroYe are not "
        "yet implemented.  Requested "
        << characteristics_method_ << " with system " << characteristics_system_
        << ".");
  }
  const size_t num_points = get(tilde_d).size();
  const Scalar<DataVector> consistent_pressure =
      equation_of_state.pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy, electron_fraction);
  Scalar<DataVector> specific_enthalpy{num_points};
  get(specific_enthalpy) = 1.0 + get(specific_internal_energy) +
                           get(consistent_pressure) / get(rest_mass_density);
  const auto det_and_inv_spatial_metric = determinant_and_inverse(spatial_metric);
  const auto& det_spatial_metric = det_and_inv_spatial_metric.first;
  const auto& inv_spatial_metric = det_and_inv_spatial_metric.second;
  const auto normal_covector_mag =
      magnitude(normal_covector, inv_spatial_metric);
  tnsr::i<DataVector, 3, Frame::Inertial> unit_normal_covector{num_points};
  for (size_t i = 0; i < 3; ++i) {
    unit_normal_covector.get(i) =
        normal_covector.get(i) / get(normal_covector_mag);
  }

  // Zero the (9-wide) packaged characteristic data; the hydro+Ye system fills
  // only the leading 3-speed / 6x6 subset.
  for (size_t i = 0; i < 9; ++i) {
    packaged_characteristic_speeds->get(i) = DataVector(num_points, 0.0);
    for (size_t j = 0; j < 9; ++j) {
      packaged_left_eigenvectors->get(i, j) = DataVector(num_points, 0.0);
      packaged_right_eigenvectors->get(i, j) = DataVector(num_points, 0.0);
    }
  }

  if (characteristics_system_ == MarquinaCharacteristicsSystem::HydroYe) {
    // Analytic hydro+Ye eigensystem: three distinct speeds and 6x6
    // modes/projectors over [tilde_d, tilde_s_x,y,z, tilde_tau, tilde_ye].
    tnsr::i<DataVector, 3> hydro_speeds{num_points, 0.0};
    tnsr::ij<DataVector, 6> hydro_modes{num_points, 0.0};
    tnsr::IJ<DataVector, 6> hydro_projectors{num_points, 0.0};
    characteristic_speeds_hydro(
        make_not_null(&hydro_speeds), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal_covector,
        equation_of_state);
    characteristic_eigenvectors_hydro(
        make_not_null(&hydro_modes), make_not_null(&hydro_projectors),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor,
        unit_normal_covector, spatial_metric, equation_of_state);
    for (size_t i = 0; i < 3; ++i) {
      packaged_characteristic_speeds->get(i) = hydro_speeds.get(i);
    }
    for (size_t i = 0; i < 6; ++i) {
      for (size_t j = 0; j < 6; ++j) {
        packaged_left_eigenvectors->get(i, j) = hydro_projectors.get(i, j);
        packaged_right_eigenvectors->get(i, j) = hydro_modes.get(i, j);
      }
    }
  } else {
    // Analytic MHD eigensystem: nine per-wave speeds and 9x9 modes/projectors
    // over [tilde_s_x,y,z, tilde_b_x,y,z, tilde_d, tilde_tau, tilde_phi]
    // (electron fraction advects separately, handled in dg_boundary_terms).
    // Recover the primitive magnetic field B^i = tilde_b^i / sqrt(gamma).
    const DataVector sqrt_det_spatial_metric = sqrt(get(det_spatial_metric));
    tnsr::I<DataVector, 3, Frame::Inertial> magnetic_field{num_points};
    for (size_t i = 0; i < 3; ++i) {
      magnetic_field.get(i) = tilde_b.get(i) / sqrt_det_spatial_metric;
    }
    tnsr::i<DataVector, 9> mhd_speeds{num_points, 0.0};
    characteristic_speeds_mhd(
        make_not_null(&mhd_speeds), spatial_velocity, magnetic_field,
        rest_mass_density, specific_internal_energy, lorentz_factor,
        specific_enthalpy, spatial_metric, unit_normal_covector,
        equation_of_state);
    tnsr::ij<DataVector, 9> mhd_modes{num_points, 0.0};
    tnsr::IJ<DataVector, 9> mhd_projectors{num_points, 0.0};
    // The analytic eigenvector formulas divide by zero at exact degeneracy
    // (e.g. B_normal = 0 on a shear-aligned face); with complementary
    // projection we let those produce non-finite values here (FP exceptions
    // disabled for this scope) and zero the affected waves below, then
    // reconstruct their subspace by complement in dg_boundary_terms.  For
    // AlwaysAnalytic the scope keeps exceptions enabled (unchanged behaviour).
    // For AlwaysNumeric we also disable exceptions so a degenerate geev block
    // (L.R != I) is caught by an explicit biorthonormality check that ERRORs,
    // rather than tripping an FP trap on 1/(L_i.R_i).
    // Only the NUMERIC (geev) path still needs exceptions disabled, so that a
    // degenerate block is caught by the explicit biorthonormality check below
    // rather than by an FP trap. The analytic path keeps them enabled: its
    // denominators are floored in Characteristics.cpp, so a trap there is a
    // real bug (this is what surfaced the atmosphere division-by-zero).
    const ScopedFpeState fpe_scope(not use_numeric);
    if (use_numeric) {
      // Numeric eigenvectors from blaze::geev.  geev returns the eigenpairs in
      // an arbitrary per-point order, so we reorder them into the canonical
      // MhdSpeed enum order by matching each numeric eigenvalue to the nearest
      // (accurate) analytic speed.  The analytic speeds are used only for the
      // ordering and the Marquina split; the eigenVECTORS come from geev.  The
      // subsequent loop rescales L_i by 1/(L_i.R_i) -- for distinct eigenvalues
      // that yields L.R = I; an exact-degenerate block (non-biorthonormal) is
      // rejected by the check after the loop.
      tnsr::i<DataVector, 9> numeric_speeds{num_points, 0.0};
      tnsr::ij<DataVector, 9> numeric_modes{num_points, 0.0};
      tnsr::IJ<DataVector, 9> numeric_projectors{num_points, 0.0};
      numerical_characteristics<9>(
          make_not_null(&numeric_speeds), make_not_null(&numeric_modes),
          make_not_null(&numeric_projectors), spatial_velocity, magnetic_field,
          rest_mass_density, specific_internal_energy, electron_fraction,
          lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
          unit_normal_covector, equation_of_state);
      for (size_t pt = 0; pt < num_points; ++pt) {
        std::array<bool, 9> used{};
        for (size_t k = 0; k < 9; ++k) {
          size_t best = 9;
          double best_dist = std::numeric_limits<double>::infinity();
          for (size_t g = 0; g < 9; ++g) {
            if (not gsl::at(used, g)) {
              const double dist =
                  std::abs(numeric_speeds.get(g)[pt] - mhd_speeds.get(k)[pt]);
              if (dist < best_dist) {
                best_dist = dist;
                best = g;
              }
            }
          }
          gsl::at(used, best) = true;
          for (size_t n = 0; n < 9; ++n) {
            mhd_modes.get(k, n)[pt] = numeric_modes.get(best, n)[pt];
            mhd_projectors.get(k, n)[pt] = numeric_projectors.get(best, n)[pt];
          }
        }
      }
    } else {
      // skip-degenerate optimization: AlwaysComplementaryProjection zeroes and
      // complements the fluid subspace (waves 2-6) unconditionally, so don't
      // build those eigenvectors at all (bit-identical, less work).
      characteristic_eigenvectors_mhd(
          make_not_null(&mhd_modes), make_not_null(&mhd_projectors), mhd_speeds,
          spatial_velocity, magnetic_field, rest_mass_density,
          specific_internal_energy, lorentz_factor, specific_enthalpy,
          spatial_metric, unit_normal_covector, equation_of_state,
          always_complementary_projection);
    }
    // characteristic_eigenvectors_mhd returns biorthogonal but NOT
    // biorthonormal eigenvectors (L_i . R_i is not 1); the Marquina
    // reconstruction needs L . R = identity, so rescale each left eigenvector
    // by 1 / (L_i . R_i).
    //
    // With complementary projection we additionally guard against degeneracy:
    // where the biorthogonality cosine |L_i.R_i| / (|L_i| |R_i|) underflows the
    // wave's eigenvectors are ill-defined, so we ZERO that wave's packaged
    // left/right rows.  dg_boundary_terms then reconstructs the zeroed
    // (degenerate) subspace by complement (I - P_nondeg) with the degenerate
    // group's speed, never touching the bad eigenvectors.  Away from
    // degeneracy nothing is zeroed and this reduces to AlwaysAnalytic.
    // Degeneracy detector = SPEED GAP (the DegeneracyTolerance option is now a
    // speed-gap tolerance, on the c_h = 1 speed scale): a wave is degenerate when
    // its characteristic speed lies within this tolerance of another wave's speed,
    // i.e. the speeds collapse.  The speed gap tracks the degeneracy directly,
    // unlike the biorthogonality cosine |L_i.R_i|/(|L_i||R_i|), which stays O(1)
    // even where the speeds are numerically identical and the eigenvectors are
    // garbage (see cpm_analysis/FINDINGS.md).  Kept (well-separated) waves are
    // still renormalized by 1/(L_i.R_i) and treated per-wave; the degenerate
    // cluster is reconstructed by the complement in dg_boundary_terms.
    const double gap_tolerance = degeneracy_tolerance_;
    for (size_t i = 0; i < 9; ++i) {
      DataVector diagonal(num_points, 0.0);
      DataVector norm_left(num_points, 0.0);
      DataVector norm_right(num_points, 0.0);
      for (size_t n = 0; n < 9; ++n) {
        diagonal += mhd_projectors.get(i, n) * mhd_modes.get(i, n);
        norm_left += mhd_projectors.get(i, n) * mhd_projectors.get(i, n);
        norm_right += mhd_modes.get(i, n) * mhd_modes.get(i, n);
      }
      packaged_characteristic_speeds->get(i) = mhd_speeds.get(i);
      if (complementary_projection) {
        // A wave is degenerate where its eigenvectors came out non-finite (the
        // analytic formulas divided by zero) or where its biorthogonality cosine
        // |L_i.R_i| / (|L_i| |R_i|) underflows.  Zero BOTH rows there so the
        // complement projection in dg_boundary_terms excludes the wave cleanly
        // (0, not 0*nan); elsewhere normalize so L_i.R_i = 1.
        for (size_t pt = 0; pt < num_points; ++pt) {
          const double scale = sqrt(norm_left[pt] * norm_right[pt]);
          // Minimum gap of wave i's speed to any other wave's speed.
          double min_speed_gap = std::numeric_limits<double>::infinity();
          for (size_t k = 0; k < 9; ++k) {
            if (k != i) {
              min_speed_gap = std::min(
                  min_speed_gap,
                  std::abs(mhd_speeds.get(i)[pt] - mhd_speeds.get(k)[pt]));
            }
          }
          // AlwaysComplementaryProjection: unconditionally complement the
          // collapse-prone fluid subspace (MhdSpeed indices 2,3,4,5,6 =
          // Alfven-, slow-, entropy, slow+, Alfven+), regardless of the gap;
          // the fast (1,7) and GLM-scalar (0,8) waves stay analytic.  Otherwise
          // (adaptive CPM) flag by the speed gap / non-finiteness.
          const bool in_fluid_subspace = (i >= 2 and i <= 6);
          const bool degenerate =
              always_complementary_projection
                  ? in_fluid_subspace
                  : (not std::isfinite(diagonal[pt]) or
                     not std::isfinite(scale) or min_speed_gap <= gap_tolerance);
          for (size_t j = 0; j < 9; ++j) {
            if (degenerate) {
              packaged_left_eigenvectors->get(i, j)[pt] = 0.0;
              packaged_right_eigenvectors->get(i, j)[pt] = 0.0;
            } else {
              packaged_left_eigenvectors->get(i, j)[pt] =
                  mhd_projectors.get(i, j)[pt] / diagonal[pt];
              packaged_right_eigenvectors->get(i, j)[pt] =
                  mhd_modes.get(i, j)[pt];
            }
          }
        }
      } else {
        const DataVector inv_diagonal = 1.0 / diagonal;
        for (size_t j = 0; j < 9; ++j) {
          packaged_left_eigenvectors->get(i, j) =
              mhd_projectors.get(i, j) * inv_diagonal;
          packaged_right_eigenvectors->get(i, j) = mhd_modes.get(i, j);
        }
      }
    }
    if (use_numeric) {
      // AlwaysNumeric is a purely numeric method (no complement fallback).
      // Verify that the rescaled numeric decomposition reproduces the identity,
      // sum_i R_i (x) L_i = I.  For distinct eigenvalues geev's eigenvectors are
      // biorthogonal, so this holds to round-off; at an exact / double-
      // underflowed degeneracy geev returns a non-biorthonormal block basis and
      // this fails -- there the fully numeric characteristics are unusable, so we
      // ERROR (the user should choose AnalyticWithComplementaryProjection for a
      // degeneracy-robust method).
      double max_identity_error = 0.0;
      for (size_t pt = 0; pt < num_points; ++pt) {
        for (size_t m = 0; m < 9; ++m) {
          for (size_t n = 0; n < 9; ++n) {
            double recon = 0.0;
            for (size_t i = 0; i < 9; ++i) {
              recon += packaged_right_eigenvectors->get(i, m)[pt] *
                       packaged_left_eigenvectors->get(i, n)[pt];
            }
            max_identity_error = std::max(
                max_identity_error, std::abs(recon - (m == n ? 1.0 : 0.0)));
          }
        }
      }
      if (not(max_identity_error < 1.0e-6)) {
        ERROR(
            "Marquina AlwaysNumeric: the numeric characteristic decomposition is "
            "not biorthonormal (max |sum_i R_i x L_i - I| = "
            << max_identity_error
            << "), which happens at an exact / underflowed degeneracy where "
               "blaze::geev returns an arbitrary basis for the repeated "
               "eigenspace.  The fully numeric characteristics cannot be used "
               "here; use CharacteristicsMethod: "
               "AnalyticWithComplementaryProjection for a degeneracy-robust "
               "decomposition.");
      }
    }
  }

  // Package conservative variables
  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  // Package conservative fluxes dotted with normal
  normal_dot_flux(packaged_normal_dot_flux_tilde_d, unit_normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, unit_normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, unit_normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, unit_normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, unit_normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, unit_normal_covector,
                  flux_tilde_phi);

  // Return the maximum absolute characteristic speed so that time step doesn't
  // violate CFL condition.
  using std::max;
  double max_abs_char_speed = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    max_abs_char_speed =
        max(max_abs_char_speed, max((*packaged_characteristic_speeds)[i]));
  }
  return max_abs_char_speed;
}

void Marquina::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
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
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const tnsr::i<DataVector, 9, Frame::NoFrame>& characteristic_speeds_ext,
    const tnsr::iJ<DataVector, 9, Frame::NoFrame>&
        left_characteristic_fields_ext,
    const tnsr::ij<DataVector, 9, Frame::NoFrame>&
        right_characteristic_fields_ext,
    dg::Formulation dg_formulation) const {
  if (characteristics_system_ == MarquinaCharacteristicsSystem::Mhd) {
    // 9-wave MHD decomposition.  Conserved-variable order matching the
    // eigenvector components: [S_x,S_y,S_z, B_x,B_y,B_z, D, tau, phi].  The
    // electron fraction advects passively and is handled by a fallback flux.
    const size_t num_points = get(tilde_d_int).size();
    using Mhd = grmhd::ValenciaDivClean::MhdSpeed;

    // Align the exterior decomposition (computed with the opposite normal) to
    // the interior frame: negate every speed and swap the +/- wave pairs
    // (Entropy is self-paired, so its speed is just negated).
    const std::array<std::array<size_t, 2>, 4> plus_minus_pairs{
        {{{Mhd::ScalarMinus, Mhd::ScalarPlus}},
         {{Mhd::FastMagnetosonicMinus, Mhd::FastMagnetosonicPlus}},
         {{Mhd::AlfvenMinus, Mhd::AlfvenPlus}},
         {{Mhd::SlowMagnetosonicMinus, Mhd::SlowMagnetosonicPlus}}}};
    auto aligned_speeds_ext = characteristic_speeds_ext;
    auto aligned_left_ext = left_characteristic_fields_ext;
    auto aligned_right_ext = right_characteristic_fields_ext;
    aligned_speeds_ext.get(Mhd::Entropy) =
        -characteristic_speeds_ext.get(Mhd::Entropy);
    for (const auto& pair : plus_minus_pairs) {
      const size_t m = pair[0];
      const size_t p = pair[1];
      aligned_speeds_ext.get(m) = -characteristic_speeds_ext.get(p);
      aligned_speeds_ext.get(p) = -characteristic_speeds_ext.get(m);
      for (size_t j = 0; j < 9; ++j) {
        aligned_left_ext.get(m, j) = left_characteristic_fields_ext.get(p, j);
        aligned_left_ext.get(p, j) = left_characteristic_fields_ext.get(m, j);
        aligned_right_ext.get(m, j) = right_characteristic_fields_ext.get(p, j);
        aligned_right_ext.get(p, j) = right_characteristic_fields_ext.get(m, j);
      }
    }

    // Local (mutable) copies of the interior eigenvectors so we can restrict to
    // a symmetric well-conditioned set below.
    auto left_int = left_characteristic_fields_int;
    auto right_int = right_characteristic_fields_int;
    const bool complementary_projection =
        characteristics_method_ ==
            MarquinaCharacteristicsMethod::AnalyticWithComplementaryProjection or
        characteristics_method_ ==
            MarquinaCharacteristicsMethod::AlwaysComplementaryProjection;
    if (complementary_projection) {
      // A wave is handled per-wave only if it is well conditioned (non-zeroed by
      // dg_package_data) on BOTH sides of the face; otherwise it joins the
      // complement block.  This keeps the per-wave reconstruction and the
      // complement projector consistent from either element, so the numerical
      // flux stays single-valued.  Zero such waves' interior and aligned-
      // exterior rows here.
      for (size_t point = 0; point < num_points; ++point) {
        for (size_t i = 0; i < 9; ++i) {
          double int_row_abs = 0.0;
          double ext_row_abs = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            int_row_abs += std::abs(left_int.get(i, n)[point]);
            ext_row_abs += std::abs(aligned_left_ext.get(i, n)[point]);
          }
          if (int_row_abs == 0.0 or ext_row_abs == 0.0) {
            for (size_t n = 0; n < 9; ++n) {
              left_int.get(i, n)[point] = 0.0;
              right_int.get(i, n)[point] = 0.0;
              aligned_left_ext.get(i, n)[point] = 0.0;
              aligned_right_ext.get(i, n)[point] = 0.0;
            }
          }
        }
      }
    }

    const std::array<const DataVector*, 9> u_int{
        {&get<0>(tilde_s_int), &get<1>(tilde_s_int), &get<2>(tilde_s_int),
         &get<0>(tilde_b_int), &get<1>(tilde_b_int), &get<2>(tilde_b_int),
         &get(tilde_d_int), &get(tilde_tau_int), &get(tilde_phi_int)}};
    const std::array<const DataVector*, 9> u_ext{
        {&get<0>(tilde_s_ext), &get<1>(tilde_s_ext), &get<2>(tilde_s_ext),
         &get<0>(tilde_b_ext), &get<1>(tilde_b_ext), &get<2>(tilde_b_ext),
         &get(tilde_d_ext), &get(tilde_tau_ext), &get(tilde_phi_ext)}};
    const std::array<const DataVector*, 9> f_int{
        {&get<0>(normal_dot_flux_tilde_s_int),
         &get<1>(normal_dot_flux_tilde_s_int),
         &get<2>(normal_dot_flux_tilde_s_int),
         &get<0>(normal_dot_flux_tilde_b_int),
         &get<1>(normal_dot_flux_tilde_b_int),
         &get<2>(normal_dot_flux_tilde_b_int),
         &get(normal_dot_flux_tilde_d_int), &get(normal_dot_flux_tilde_tau_int),
         &get(normal_dot_flux_tilde_phi_int)}};
    const std::array<const DataVector*, 9> f_ext{
        {&get<0>(normal_dot_flux_tilde_s_ext),
         &get<1>(normal_dot_flux_tilde_s_ext),
         &get<2>(normal_dot_flux_tilde_s_ext),
         &get<0>(normal_dot_flux_tilde_b_ext),
         &get<1>(normal_dot_flux_tilde_b_ext),
         &get<2>(normal_dot_flux_tilde_b_ext),
         &get(normal_dot_flux_tilde_d_ext), &get(normal_dot_flux_tilde_tau_ext),
         &get(normal_dot_flux_tilde_phi_ext)}};
    const std::array<DataVector*, 9> b_out{
        {&get<0>(*boundary_correction_tilde_s),
         &get<1>(*boundary_correction_tilde_s),
         &get<2>(*boundary_correction_tilde_s),
         &get<0>(*boundary_correction_tilde_b),
         &get<1>(*boundary_correction_tilde_b),
         &get<2>(*boundary_correction_tilde_b),
         &get(*boundary_correction_tilde_d), &get(*boundary_correction_tilde_tau),
         &get(*boundary_correction_tilde_phi)}};

    for (size_t n = 0; n < 9; ++n) {
      *gsl::at(b_out, n) = 0.0;
    }
    // Electron fraction: passive scalar, conservative central fallback flux.
    get(*boundary_correction_tilde_ye) =
        0.5 * (get(normal_dot_flux_tilde_ye_int) -
               get(normal_dot_flux_tilde_ye_ext));

    Scalar<DataVector> omega_int{num_points};
    Scalar<DataVector> omega_ext{num_points};
    Scalar<DataVector> phi_int{num_points};
    Scalar<DataVector> phi_ext{num_points};
    // ------------------------------------------------------------------
    // MODIFIED Marquina flux formula (Aloy et al. 1999, ApJS 122, 151), as used
    // by Whisky/GENESIS/Ratpenat for relativistic flows. This is NOT the sided
    // Donat-Marquina flux with its viscous branch forced on -- that mixes the
    // two one-sided eigenbases and is inconsistent with the complementary
    // projection. It is the standard Roe/LLF form written out directly:
    //
    //   F = 1/2 (F_L + F_R) - 1/2 sum_p alpha_p (l_p . du) r_p,
    //   alpha_p = max(|lambda_p^L|, |lambda_p^R|),   du = u_R - u_L
    //
    // with the dissipation applied symmetrically in BOTH one-sided bases (each
    // is a complete decomposition on its own state, so averaging them keeps the
    // scheme symmetric without inventing an averaged state -- which is exactly
    // what Marquina-type schemes exist to avoid). More dissipative than the
    // original, and stable: the original form is exact at t = 0.05 on the
    // |B|x2 stationary contact and then diverges to rho ~ 130 by t = 1.
    if (use_modified_formula_) {
      for (size_t n = 0; n < 9; ++n) {
        *gsl::at(b_out, n) =
            0.5 * (*gsl::at(f_int, n) - *gsl::at(f_ext, n));
      }
      Scalar<DataVector> projected_jump_int{num_points};
      Scalar<DataVector> projected_jump_ext{num_points};
      std::array<DataVector, 9> jump{};
      std::array<DataVector, 9> reconstructed_int{};
      std::array<DataVector, 9> reconstructed_ext{};
      for (size_t n = 0; n < 9; ++n) {
        gsl::at(jump, n) = *gsl::at(u_ext, n) - *gsl::at(u_int, n);
        gsl::at(reconstructed_int, n) = DataVector{num_points, 0.0};
        gsl::at(reconstructed_ext, n) = DataVector{num_points, 0.0};
      }
      for (size_t i = 0; i < 9; ++i) {
        get(projected_jump_int) = 0.0;
        get(projected_jump_ext) = 0.0;
        for (size_t n = 0; n < 9; ++n) {
          get(projected_jump_int) += left_int.get(i, n) * gsl::at(jump, n);
          get(projected_jump_ext) +=
              aligned_left_ext.get(i, n) * gsl::at(jump, n);
        }
        const DataVector& lambda_int = characteristic_speeds_int.get(i);
        const DataVector& lambda_ext = aligned_speeds_ext.get(i);
        DataVector alpha{num_points};
        for (size_t point = 0; point < num_points; ++point) {
          alpha[point] = std::max(std::abs(lambda_int[point]),
                                  std::abs(lambda_ext[point]));
        }
        for (size_t n = 0; n < 9; ++n) {
          const DataVector contribution_int =
              get(projected_jump_int) *
              right_characteristic_fields_int.get(i, n);
          const DataVector contribution_ext =
              get(projected_jump_ext) * aligned_right_ext.get(i, n);
          gsl::at(reconstructed_int, n) += contribution_int;
          gsl::at(reconstructed_ext, n) += contribution_ext;
          *gsl::at(b_out, n) -=
              0.25 * alpha * (contribution_int + contribution_ext);
        }
      }
      // The degenerate rows were ZEROED in dg_package_data, so neither one-sided
      // basis is complete: sum_p r_p l_p != I, and the per-wave dissipation
      // above misses the degenerate subspace entirely. Dissipate what is left
      // over, (I - sum_p r_p l_p) du -- the same role the complementary
      // projection plays for the sided formula. Without it Balsara-1 -- whose
      // tangential field REVERSES through zero, so B_t -> 0 is degenerate right
      // at the flip -- is under-dissipated exactly where it needs damping, and
      // dies in con2prim.
      //
      // Use the speed of the DEGENERATE waves themselves, not the global maximum
      // over all nine. The complement spans exactly the zeroed (degenerate)
      // subspace, so its signal speed is theirs; damping it at the fast speed
      // instead put a resolution-independent floor under the error and stalled
      // convergence (order 0.07 between ref4 and ref5 on Balsara-1, versus 0.43
      // for the original formula).
      DataVector complement_speed{num_points, 0.0};
      for (size_t point = 0; point < num_points; ++point) {
        bool any_degenerate = false;
        for (size_t i = 0; i < 9; ++i) {
          bool left_row_is_zero = true;
          for (size_t n = 0; n < 9; ++n) {
            if (left_int.get(i, n)[point] != 0.0) {
              left_row_is_zero = false;
              break;
            }
          }
          if (left_row_is_zero) {
            any_degenerate = true;
            complement_speed[point] = std::max(
                complement_speed[point],
                std::max(std::abs(characteristic_speeds_int.get(i)[point]),
                         std::abs(aligned_speeds_ext.get(i)[point])));
          }
        }
        // No zeroed rows -> the bases are complete, the complement is zero and
        // this term is inert; keep the speed at zero rather than the fast speed.
        if (not any_degenerate) {
          complement_speed[point] = 0.0;
        }
      }
      // Damp the complement with Lax-Friedrichs at the degenerate waves' own
      // speed. This is the ONLY variant of the three measured that survives
      // Balsara-1 at ref5: halving it (0.02827 -> 0.03144 at ref4, ref5 crash)
      // and upwinding it as the original formula's CPM block does
      // (0.03075 at ref4, ref5 crash) are both less accurate AND less stable.
      // The degenerate subspace simply needs this much dissipation to hold
      // together at high resolution -- see FINDINGS_runs.md sections 35-37 for
      // the measurements, including the convergence plateau it costs.
      for (size_t n = 0; n < 9; ++n) {
        *gsl::at(b_out, n) -=
            0.25 * complement_speed *
            ((gsl::at(jump, n) - gsl::at(reconstructed_int, n)) +
             (gsl::at(jump, n) - gsl::at(reconstructed_ext, n)));
      }

      if (dg_formulation == dg::Formulation::StrongInertial) {
        for (size_t n = 0; n < 9; ++n) {
          *gsl::at(b_out, n) -= *gsl::at(f_int, n);
        }
        get(*boundary_correction_tilde_ye) -= get(normal_dot_flux_tilde_ye_int);
      }
      return;
    }

    Scalar<DataVector> phi_plus{num_points};
    Scalar<DataVector> phi_minus{num_points};
    for (size_t i = 0; i < 9; ++i) {
      get(omega_int) = 0.0;
      get(omega_ext) = 0.0;
      get(phi_int) = 0.0;
      get(phi_ext) = 0.0;
      for (size_t n = 0; n < 9; ++n) {
        get(omega_int) += left_int.get(i, n) * *gsl::at(u_int, n);
        get(omega_ext) += aligned_left_ext.get(i, n) * *gsl::at(u_ext, n);
        get(phi_int) += left_int.get(i, n) * *gsl::at(f_int, n);
        get(phi_ext) += aligned_left_ext.get(i, n) * (-*gsl::at(f_ext, n));
      }
      const DataVector& lambda_int = characteristic_speeds_int.get(i);
      const DataVector& lambda_ext = aligned_speeds_ext.get(i);
      for (size_t point = 0; point < num_points; ++point) {
        if (lambda_int[point] >= 0.0 and lambda_ext[point] >= 0.0) {
          get(phi_plus)[point] = get(phi_int)[point];
          get(phi_minus)[point] = 0.0;
        } else if (lambda_int[point] <= 0.0 and lambda_ext[point] <= 0.0) {
          get(phi_plus)[point] = 0.0;
          get(phi_minus)[point] = get(phi_ext)[point];
        } else {
          const double alpha = std::max(std::abs(lambda_int[point]),
                                        std::abs(lambda_ext[point]));
          get(phi_plus)[point] =
              0.5 * (get(phi_int)[point] + alpha * get(omega_int)[point]);
          get(phi_minus)[point] =
              0.5 * (get(phi_ext)[point] - alpha * get(omega_ext)[point]);
        }
      }
      for (size_t n = 0; n < 9; ++n) {
        *gsl::at(b_out, n) +=
            get(phi_plus) * right_characteristic_fields_int.get(i, n) +
            get(phi_minus) * aligned_right_ext.get(i, n);
      }
    }

    // Complementary projection (Fedkiw-Merriman-Osher 1997): the waves whose
    // eigenvectors dg_package_data zeroed (degenerate; left row == 0) contributed
    // nothing to the per-wave sum above.  Reconstruct them here as ONE block via
    // the complement of the well-conditioned projector, P_block = I - sum_{well}
    // R_i L_i, carried by the degenerate group's (shared) speed.  This never uses
    // the ill-defined degenerate eigenvectors.  NOTE: a single complement only
    // resolves ONE degenerate group (it assumes the zeroed waves share an upwind
    // direction, true for the B_normal=0 group at v_normal); multiple distinct
    // degenerate groups fall back to a dissipative average-speed treatment.
    // Away from degeneracy nothing is zeroed, so P_block = 0 and this is a no-op.
    if (complementary_projection) {
      std::array<DataVector, 9> proj_u_int;
      std::array<DataVector, 9> proj_f_int;
      std::array<DataVector, 9> proj_u_ext;
      std::array<DataVector, 9> proj_f_ext;
      for (size_t n = 0; n < 9; ++n) {
        gsl::at(proj_u_int, n) = DataVector(num_points, 0.0);
        gsl::at(proj_f_int, n) = DataVector(num_points, 0.0);
        gsl::at(proj_u_ext, n) = DataVector(num_points, 0.0);
        gsl::at(proj_f_ext, n) = DataVector(num_points, 0.0);
      }
      // P U = sum_i (L_i . U) R_i, per conserved component (degenerate L_i == 0
      // drop out, leaving the projector onto the well-conditioned subspace).
      for (size_t i = 0; i < 9; ++i) {
        DataVector li_u_int(num_points, 0.0);
        DataVector li_f_int(num_points, 0.0);
        DataVector li_u_ext(num_points, 0.0);
        DataVector li_f_ext(num_points, 0.0);
        for (size_t n = 0; n < 9; ++n) {
          li_u_int += left_int.get(i, n) * *gsl::at(u_int, n);
          li_f_int += left_int.get(i, n) * *gsl::at(f_int, n);
          li_u_ext += aligned_left_ext.get(i, n) * *gsl::at(u_ext, n);
          li_f_ext += aligned_left_ext.get(i, n) * (-*gsl::at(f_ext, n));
        }
        for (size_t n = 0; n < 9; ++n) {
          gsl::at(proj_u_int, n) += li_u_int * right_int.get(i, n);
          gsl::at(proj_f_int, n) += li_f_int * right_int.get(i, n);
          gsl::at(proj_u_ext, n) += li_u_ext * aligned_right_ext.get(i, n);
          gsl::at(proj_f_ext, n) += li_f_ext * aligned_right_ext.get(i, n);
        }
      }
      for (size_t point = 0; point < num_points; ++point) {
        // Reconstruct the whole complementary (degenerate) subspace following
        // Fedkiw-Merriman-Osher: since these waves nearly share a speed (they
        // collapse toward the material speed v_n at the degeneracy), the
        // complement has a common upwind direction whenever their speeds share a
        // sign (paper Remark 6).  Upwind the complement VECTOR by that sign
        // (single-valued: sign-based upwinding gives the same flux from either
        // element); fall back to the symmetric local Lax-Friedrichs flux only
        // where the degenerate speeds change sign (a transonic degeneracy).  The
        // degenerate set is symmetric across the face (left_int is masked to zero
        // a wave degenerate on either side), so the complement is single-valued.
        double alpha = 0.0;
        bool any_degenerate = false;
        double min_deg_speed = std::numeric_limits<double>::infinity();
        double max_deg_speed = -std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < 9; ++i) {
          double row_abs = 0.0;
          for (size_t n = 0; n < 9; ++n) {
            row_abs += std::abs(left_int.get(i, n)[point]);
          }
          if (row_abs == 0.0) {
            any_degenerate = true;
            const double si = characteristic_speeds_int.get(i)[point];
            const double se = aligned_speeds_ext.get(i)[point];
            alpha = std::max({alpha, std::abs(si), std::abs(se)});
            min_deg_speed = std::min({min_deg_speed, si, se});
            max_deg_speed = std::max({max_deg_speed, si, se});
          }
        }
        if (not any_degenerate) {
          continue;
        }
        const bool all_pos = (min_deg_speed >= 0.0);
        const bool all_neg = (max_deg_speed <= 0.0);
        for (size_t n = 0; n < 9; ++n) {
          const double f_bar_int =
              (*gsl::at(f_int, n))[point] - gsl::at(proj_f_int, n)[point];
          const double f_bar_ext =
              -(*gsl::at(f_ext, n))[point] - gsl::at(proj_f_ext, n)[point];
          const double u_bar_int =
              (*gsl::at(u_int, n))[point] - gsl::at(proj_u_int, n)[point];
          const double u_bar_ext =
              (*gsl::at(u_ext, n))[point] - gsl::at(proj_u_ext, n)[point];
          double contrib;
          if (all_pos) {
            contrib = f_bar_int;  // FMO componentwise upwind from interior
          } else if (all_neg) {
            contrib = f_bar_ext;  // FMO componentwise upwind from exterior
          } else {
            contrib = 0.5 * (f_bar_int + f_bar_ext) -
                      0.5 * alpha * (u_bar_ext - u_bar_int);  // LF at sign change
          }
          (*gsl::at(b_out, n))[point] += contrib;
        }
      }
    }

    // ROBUSTNESS FALLBACK. Marquina uses the eigenbasis AS its flux, so an
    // ill-conditioned decomposition does not degrade gracefully the way HLLEM's
    // anti-diffusion does (HLLEM sits on a stable HLL base; a bad eigenvector
    // there costs accuracy, not validity). Flooring the denominators in
    // Characteristics.cpp stops the FP trap but does NOT make the eigenvectors
    // meaningful: with the floors alone Marquina completed the CW |B|x2 test
    // with rho reaching 77 where the exact solution is bounded by 10.
    //
    // So detect an untrustworthy decomposition from what we already have --
    // non-finite entries, or a biorthonormality diagonal l_i.r_i that has
    // collapsed for a wave whose left row is not identically zero (rows that ARE
    // zero were deliberately dropped as degenerate above) -- and fall back to a
    // plain local Lax-Friedrichs flux at that point, which needs no eigenbasis.
    // This is a per-point switch: cells where the decomposition is fine are
    // untouched, so it costs nothing where the physics lives.
    for (size_t point = 0; point < num_points; ++point) {
      bool trustworthy = true;
      for (size_t i = 0; i < 9 and trustworthy; ++i) {
        double diagonal = 0.0;
        bool left_row_is_zero = true;
        for (size_t n = 0; n < 9; ++n) {
          const double left_entry = left_int.get(i, n)[point];
          const double right_entry =
              right_characteristic_fields_int.get(i, n)[point];
          if (not std::isfinite(left_entry) or not std::isfinite(right_entry)) {
            trustworthy = false;
            break;
          }
          if (left_entry != 0.0) {
            left_row_is_zero = false;
          }
          diagonal += left_entry * right_entry;
        }
        if (not left_row_is_zero and std::abs(diagonal) < 1.0e-8) {
          trustworthy = false;
        }
      }
      for (size_t n = 0; n < 9 and trustworthy; ++n) {
        if (not std::isfinite((*gsl::at(b_out, n))[point])) {
          trustworthy = false;
        }
      }
      if (trustworthy) {
        continue;
      }
      double alpha = 0.0;
      for (size_t i = 0; i < 9; ++i) {
        alpha = std::max(
            alpha, std::max(std::abs(characteristic_speeds_int.get(i)[point]),
                            std::abs(aligned_speeds_ext.get(i)[point])));
      }
      for (size_t n = 0; n < 9; ++n) {
        (*gsl::at(b_out, n))[point] =
            0.5 * ((*gsl::at(f_int, n))[point] - (*gsl::at(f_ext, n))[point]) -
            0.5 * alpha *
                ((*gsl::at(u_ext, n))[point] - (*gsl::at(u_int, n))[point]);
      }
    }

    if (dg_formulation == dg::Formulation::StrongInertial) {
      for (size_t n = 0; n < 9; ++n) {
        *gsl::at(b_out, n) -= *gsl::at(f_int, n);
      }
      get(*boundary_correction_tilde_ye) -= get(normal_dot_flux_tilde_ye_int);
    }
    return;
  }

  // Hydro+Ye path.  With AnalyticWithComplementaryProjection (Fedkiw-Merriman-
  // Osher 1997) the 4-fold degenerate contact subspace (all at speed v_n: the
  // shear R1,R2 and contact/entropy R3,R4) is NOT decomposed into its (non-
  // unique) eigenvectors.  Instead only the two acoustic waves (Rplus,Rminus,
  // at v_n +/- c_s, always well-conditioned) are decomposed, and the contact
  // subspace is reconstructed as the complement f_bar = (I - P_acoustic) F and
  // upwinded componentwise in the single v_n direction (Marquina split keyed on
  // v_n, LF term only at a transonic-contact sign change -- no blanket LLF).
  // Because all four contact waves share the speed v_n, this is algebraically
  // identical to the full 6-wave decomposition (paper Remark 1), which the unit
  // test verifies to round-off.  See reports/cpm_paper_study.md.
  const bool complementary_projection =
      characteristics_method_ ==
      MarquinaCharacteristicsMethod::AnalyticWithComplementaryProjection;

  auto aligned_characteristic_speeds_ext = characteristic_speeds_ext;
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity);
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus);
  aligned_characteristic_speeds_ext.get(
      grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus) =
      -characteristic_speeds_ext.get(
          grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus);

  auto aligned_left_characteristic_fields_ext = left_characteristic_fields_ext;
  auto aligned_right_characteristic_fields_ext =
      right_characteristic_fields_ext;
  for (size_t j = 0; j < 6; ++j) {
    aligned_left_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rplus, j) =
        left_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rminus, j);
    aligned_left_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rminus, j) =
        left_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rplus, j);
    aligned_right_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rplus, j) =
        right_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rminus, j);
    aligned_right_characteristic_fields_ext.get(
        grmhd::ValenciaDivClean::HydroVectorR::Rminus, j) =
        right_characteristic_fields_ext.get(
            grmhd::ValenciaDivClean::HydroVectorR::Rplus, j);
  }
  // Initialize boundary corrections to zero, as we'll compute them by adding
  // the contributions from each characteristic field.  (Each assignment zeros
  // the whole DataVector; the previous `for (point ...)` wrapper repeated this
  // num_points times with `point` unused -- an O(N^2) no-op.  Removing it gives
  // a large speedup at realistic face-point counts; found by Emily/Claude.)
  const size_t num_points = get(tilde_d_int).size();
  get(*boundary_correction_tilde_d) = 0.0;
  get<0>(*boundary_correction_tilde_s) = 0.0;
  get<1>(*boundary_correction_tilde_s) = 0.0;
  get<2>(*boundary_correction_tilde_s) = 0.0;
  get(*boundary_correction_tilde_tau) = 0.0;
  get(*boundary_correction_tilde_ye) = 0.0;

  // Not yet implemented for magnetic field and divergence cleaning field, so
  // set to zero
  get<0>(*boundary_correction_tilde_b) = 0.0;
  get<1>(*boundary_correction_tilde_b) = 0.0;
  get<2>(*boundary_correction_tilde_b) = 0.0;
  get(*boundary_correction_tilde_phi) = 0.0;

  // Fallback flux for unmodeled B and Phi fields to satisfy DG contracts
  for (size_t j = 0; j < 3; ++j) {
    boundary_correction_tilde_b->get(j) =
        0.5 * (normal_dot_flux_tilde_b_int.get(j) -
               normal_dot_flux_tilde_b_ext.get(j));
  }
  get(*boundary_correction_tilde_phi) =
      0.5 *
      (get(normal_dot_flux_tilde_phi_int) - get(normal_dot_flux_tilde_phi_ext));

  // Temporary variables
  Scalar<DataVector> omega_i_int{num_points};
  Scalar<DataVector> omega_i_ext{num_points};
  Scalar<DataVector> phi_i_int{num_points};
  Scalar<DataVector> phi_i_ext{num_points};
  Scalar<DataVector> phi_i_plus{num_points};
  Scalar<DataVector> phi_i_minus{num_points};

  // Loop over characteristic fields
  for (size_t i = 0; i < 6; ++i) {
    // With complementary projection the four contact-subspace waves (i < Rplus)
    // are handled together by the complement block after this loop; here we only
    // decompose the two acoustic waves.
    if (complementary_projection and
        i < grmhd::ValenciaDivClean::HydroVectorR::Rplus) {
      continue;
    }
    // Project conservative variables and normal fluxes onto "characteristic
    // basis"
    get(omega_i_int) =
        left_characteristic_fields_int.get(i, 0) * get(tilde_d_int) +
        left_characteristic_fields_int.get(i, 1) * get<0>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 2) * get<1>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 3) * get<2>(tilde_s_int) +
        left_characteristic_fields_int.get(i, 4) * get(tilde_tau_int) +
        left_characteristic_fields_int.get(i, 5) * get(tilde_ye_int);
    get(omega_i_ext) =
        aligned_left_characteristic_fields_ext.get(i, 0) * get(tilde_d_ext) +
        aligned_left_characteristic_fields_ext.get(i, 1) * get<0>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 2) * get<1>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 3) * get<2>(tilde_s_ext) +
        aligned_left_characteristic_fields_ext.get(i, 4) * get(tilde_tau_ext) +
        aligned_left_characteristic_fields_ext.get(i, 5) * get(tilde_ye_ext);
    get(phi_i_int) = left_characteristic_fields_int.get(i, 0) *
                         get(normal_dot_flux_tilde_d_int) +
                     left_characteristic_fields_int.get(i, 1) *
                         get<0>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 2) *
                         get<1>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 3) *
                         get<2>(normal_dot_flux_tilde_s_int) +
                     left_characteristic_fields_int.get(i, 4) *
                         get(normal_dot_flux_tilde_tau_int) +
                     left_characteristic_fields_int.get(i, 5) *
                         get(normal_dot_flux_tilde_ye_int);
    get(phi_i_ext) = aligned_left_characteristic_fields_ext.get(i, 0) *
                         (-get(normal_dot_flux_tilde_d_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 1) *
                         (-get<0>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 2) *
                         (-get<1>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 3) *
                         (-get<2>(normal_dot_flux_tilde_s_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 4) *
                         (-get(normal_dot_flux_tilde_tau_ext)) +
                     aligned_left_characteristic_fields_ext.get(i, 5) *
                         (-get(normal_dot_flux_tilde_ye_ext));

    // TO-DO: improve how we handle the indices of characteristic speeds
    size_t hydro_speed_index;
    switch (i) {
      case grmhd::ValenciaDivClean::HydroVectorR::R1:
      case grmhd::ValenciaDivClean::HydroVectorR::R2:
      case grmhd::ValenciaDivClean::HydroVectorR::R3:
      case grmhd::ValenciaDivClean::HydroVectorR::R4:
        hydro_speed_index =
            grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity;
        break;
      case grmhd::ValenciaDivClean::HydroVectorR::Rplus:
        hydro_speed_index = grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus;
        break;
      case grmhd::ValenciaDivClean::HydroVectorR::Rminus:
        hydro_speed_index = grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus;
        break;
      default:
        ERROR("Unhandled index value in switch statement.");
    }

    // Compute Marquina fluxes in "characteristic basis"
    const DataVector& lambda_i_int =
        characteristic_speeds_int.get(hydro_speed_index);
    const DataVector& lambda_i_ext =
        aligned_characteristic_speeds_ext.get(hydro_speed_index);
    for (size_t point = 0; point < num_points; ++point) {
      if (lambda_i_int[point] >= 0.0 and lambda_i_ext[point] >= 0.0) {
        get(phi_i_plus)[point] = get(phi_i_int)[point];
        get(phi_i_minus)[point] = 0.0;
      } else if (lambda_i_int[point] <= 0.0 and lambda_i_ext[point] <= 0.0) {
        get(phi_i_plus)[point] = 0.0;
        get(phi_i_minus)[point] = get(phi_i_ext)[point];
      } else {
        double alpha = std::max(std::abs(lambda_i_int[point]),
                                std::abs(lambda_i_ext[point]));
        get(phi_i_plus)[point] =
            0.5 * (get(phi_i_int)[point] + alpha * get(omega_i_int)[point]);
        get(phi_i_minus)[point] =
            0.5 * (get(phi_i_ext)[point] - alpha * get(omega_i_ext)[point]);
      }
    }

    // Reconstruct Marquina fluxes in "conserved basis"
    // TO-DO: handle dg_formulation (strong/weak)
    get(*boundary_correction_tilde_d) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 0) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 0);
    get<0>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 1) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 1);
    get<1>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 2) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 2);
    get<2>(*boundary_correction_tilde_s) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 3) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 3);
    get(*boundary_correction_tilde_tau) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 4) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 4);
    get(*boundary_correction_tilde_ye) +=
        get(phi_i_plus) * right_characteristic_fields_int.get(i, 5) +
        get(phi_i_minus) * aligned_right_characteristic_fields_ext.get(i, 5);
  }

  if (complementary_projection) {
    // Contact subspace (all waves at v_n) via the FMO complement, upwinded
    // componentwise in the single v_n direction.  f_bar = (I - P_acoustic) F
    // and u_bar = (I - P_acoustic) U, with P_acoustic = sum over Rplus,Rminus of
    // R_a (L_a . *).  The same Marquina split used per-wave above is applied to
    // the complement VECTOR keyed on v_n (LF term only at a sign change).  This
    // equals the sum over the four contact waves of their per-wave contribution
    // (they share the speed v_n), to round-off, without their eigenvectors.
    using RV = grmhd::ValenciaDivClean::HydroVectorR;
    const size_t vn_index =
        grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity;
    const DataVector& lambda_vn_int = characteristic_speeds_int.get(vn_index);
    const DataVector& lambda_vn_ext =
        aligned_characteristic_speeds_ext.get(vn_index);
    const std::array<const DataVector*, 6> u_int_c{
        {&get(tilde_d_int), &get<0>(tilde_s_int), &get<1>(tilde_s_int),
         &get<2>(tilde_s_int), &get(tilde_tau_int), &get(tilde_ye_int)}};
    const std::array<const DataVector*, 6> u_ext_c{
        {&get(tilde_d_ext), &get<0>(tilde_s_ext), &get<1>(tilde_s_ext),
         &get<2>(tilde_s_ext), &get(tilde_tau_ext), &get(tilde_ye_ext)}};
    const std::array<const DataVector*, 6> f_int_c{
        {&get(normal_dot_flux_tilde_d_int),
         &get<0>(normal_dot_flux_tilde_s_int),
         &get<1>(normal_dot_flux_tilde_s_int),
         &get<2>(normal_dot_flux_tilde_s_int),
         &get(normal_dot_flux_tilde_tau_int),
         &get(normal_dot_flux_tilde_ye_int)}};
    const std::array<const DataVector*, 6> f_ext_c{
        {&get(normal_dot_flux_tilde_d_ext),
         &get<0>(normal_dot_flux_tilde_s_ext),
         &get<1>(normal_dot_flux_tilde_s_ext),
         &get<2>(normal_dot_flux_tilde_s_ext),
         &get(normal_dot_flux_tilde_tau_ext),
         &get(normal_dot_flux_tilde_ye_ext)}};
    const std::array<DataVector*, 6> b_out_c{
        {&get(*boundary_correction_tilde_d),
         &get<0>(*boundary_correction_tilde_s),
         &get<1>(*boundary_correction_tilde_s),
         &get<2>(*boundary_correction_tilde_s),
         &get(*boundary_correction_tilde_tau),
         &get(*boundary_correction_tilde_ye)}};
    const std::array<size_t, 2> acoustic{{RV::Rplus, RV::Rminus}};
    for (size_t point = 0; point < num_points; ++point) {
      // Acoustic scalar projections L_a . (F,U) at this point (interior uses
      // +F, exterior uses -F, matching the loop's phi/omega conventions).
      std::array<double, 2> la_f_int{{0.0, 0.0}};
      std::array<double, 2> la_u_int{{0.0, 0.0}};
      std::array<double, 2> la_f_ext{{0.0, 0.0}};
      std::array<double, 2> la_u_ext{{0.0, 0.0}};
      for (size_t a = 0; a < 2; ++a) {
        const size_t wa = gsl::at(acoustic, a);
        for (size_t n = 0; n < 6; ++n) {
          la_f_int[a] += left_characteristic_fields_int.get(wa, n)[point] *
                         (*gsl::at(f_int_c, n))[point];
          la_u_int[a] += left_characteristic_fields_int.get(wa, n)[point] *
                         (*gsl::at(u_int_c, n))[point];
          la_f_ext[a] +=
              aligned_left_characteristic_fields_ext.get(wa, n)[point] *
              (-(*gsl::at(f_ext_c, n))[point]);
          la_u_ext[a] +=
              aligned_left_characteristic_fields_ext.get(wa, n)[point] *
              (*gsl::at(u_ext_c, n))[point];
        }
      }
      const double lam_i = lambda_vn_int[point];
      const double lam_e = lambda_vn_ext[point];
      const double alpha = std::max(std::abs(lam_i), std::abs(lam_e));
      const bool all_pos = (lam_i >= 0.0 and lam_e >= 0.0);
      const bool all_neg = (lam_i <= 0.0 and lam_e <= 0.0);
      for (size_t n = 0; n < 6; ++n) {
        double pf_int = 0.0;
        double pu_int = 0.0;
        double pf_ext = 0.0;
        double pu_ext = 0.0;
        for (size_t a = 0; a < 2; ++a) {
          const size_t wa = gsl::at(acoustic, a);
          const double r_int =
              right_characteristic_fields_int.get(wa, n)[point];
          const double r_ext =
              aligned_right_characteristic_fields_ext.get(wa, n)[point];
          pf_int += la_f_int[a] * r_int;
          pu_int += la_u_int[a] * r_int;
          pf_ext += la_f_ext[a] * r_ext;
          pu_ext += la_u_ext[a] * r_ext;
        }
        const double fbar_int = (*gsl::at(f_int_c, n))[point] - pf_int;
        const double ubar_int = (*gsl::at(u_int_c, n))[point] - pu_int;
        const double fbar_ext = -(*gsl::at(f_ext_c, n))[point] - pf_ext;
        const double ubar_ext = (*gsl::at(u_ext_c, n))[point] - pu_ext;
        double contact_plus = 0.0;
        double contact_minus = 0.0;
        if (all_pos) {
          contact_plus = fbar_int;
        } else if (all_neg) {
          contact_minus = fbar_ext;
        } else {
          contact_plus = 0.5 * (fbar_int + alpha * ubar_int);
          contact_minus = 0.5 * (fbar_ext - alpha * ubar_ext);
        }
        (*gsl::at(b_out_c, n))[point] += contact_plus + contact_minus;
      }
    }
  }

  if (dg_formulation == dg::Formulation::StrongInertial) {
    get(*boundary_correction_tilde_d) -= get(normal_dot_flux_tilde_d_int);
    get<0>(*boundary_correction_tilde_s) -= get<0>(normal_dot_flux_tilde_s_int);
    get<1>(*boundary_correction_tilde_s) -= get<1>(normal_dot_flux_tilde_s_int);
    get<2>(*boundary_correction_tilde_s) -= get<2>(normal_dot_flux_tilde_s_int);
    get(*boundary_correction_tilde_tau) -= get(normal_dot_flux_tilde_tau_int);
    get(*boundary_correction_tilde_ye) -= get(normal_dot_flux_tilde_ye_int);
    get<0>(*boundary_correction_tilde_b) -= get<0>(normal_dot_flux_tilde_b_int);
    get<1>(*boundary_correction_tilde_b) -= get<1>(normal_dot_flux_tilde_b_int);
    get<2>(*boundary_correction_tilde_b) -= get<2>(normal_dot_flux_tilde_b_int);
    get(*boundary_correction_tilde_phi) -= get(normal_dot_flux_tilde_phi_int);
  }
}

bool operator==(const Marquina& lhs, const Marquina& rhs) {
  return lhs.characteristics_system_ == rhs.characteristics_system_ and
         lhs.characteristics_method_ == rhs.characteristics_method_ and
         lhs.degeneracy_tolerance_ == rhs.degeneracy_tolerance_ and
         lhs.use_modified_formula_ == rhs.use_modified_formula_;
}

bool operator!=(const Marquina& lhs, const Marquina& rhs) {
  return not(lhs == rhs);
}

// NOLINTNEXTLINE
PUP::able::PUP_ID Marquina::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections

template <>
grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsSystem
Options::create_from_yaml<grmhd::ValenciaDivClean::BoundaryCorrections::
                              MarquinaCharacteristicsSystem>::
    create<void>(const Options::Option& options) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "HydroYe") {
    return bc::MarquinaCharacteristicsSystem::HydroYe;
  } else if (type_read == "Mhd") {
    return bc::MarquinaCharacteristicsSystem::Mhd;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to MarquinaCharacteristicsSystem. Must be one of "
                     "HydroYe or Mhd.");
}

template <>
grmhd::ValenciaDivClean::BoundaryCorrections::MarquinaCharacteristicsMethod
Options::create_from_yaml<grmhd::ValenciaDivClean::BoundaryCorrections::
                              MarquinaCharacteristicsMethod>::
    create<void>(const Options::Option& options) {
  namespace bc = grmhd::ValenciaDivClean::BoundaryCorrections;
  const auto type_read = options.parse_as<std::string>();
  if (type_read == "AlwaysAnalytic") {
    return bc::MarquinaCharacteristicsMethod::AlwaysAnalytic;
  } else if (type_read == "AlwaysNumeric") {
    return bc::MarquinaCharacteristicsMethod::AlwaysNumeric;
  } else if (type_read == "AnalyticWithNumericFallback") {
    return bc::MarquinaCharacteristicsMethod::AnalyticWithNumericFallback;
  } else if (type_read == "AnalyticWithComplementaryProjection") {
    return bc::MarquinaCharacteristicsMethod::
        AnalyticWithComplementaryProjection;
  } else if (type_read == "AlwaysComplementaryProjection") {
    return bc::MarquinaCharacteristicsMethod::AlwaysComplementaryProjection;
  }
  PARSE_ERROR(options.context(),
              "Failed to convert \""
                  << type_read
                  << "\" to MarquinaCharacteristicsMethod. Must be one of "
                     "AlwaysAnalytic, AlwaysNumeric, "
                     "AnalyticWithNumericFallback, or "
                     "AnalyticWithComplementaryProjection.");
}
