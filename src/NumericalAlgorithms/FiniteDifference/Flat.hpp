// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "NumericalAlgorithms/FiniteDifference/Reconstruct.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Direction;
template <size_t Dim>
class Index;
/// \endcond

namespace fd::reconstruction {
namespace detail {
struct FlatReconstructor {
  SPECTRE_ALWAYS_INLINE static std::array<double, 2> pointwise(
      const double* const q, const int /*stride*/) {
    // Piecewise-constant (donor-cell / first-order / "flat") reconstruction:
    // both faces of the cell take the cell-centered value (zero slope). The
    // neighbor points in the stencil are ignored.
    return {{q[0], q[0]}};
  }

  // The reconstruction machinery requires an odd stencil width >= 3, so we use
  // 3 and simply ignore the neighbor points.
  SPECTRE_ALWAYS_INLINE static constexpr size_t stencil_width() { return 3; }
};
}  // namespace detail

/*!
 * \ingroup FiniteDifferenceGroup
 * \brief Performs piecewise-constant ("flat", first-order, donor-cell)
 * reconstruction on the `volume_vars` in each direction.
 *
 * On a 1d mesh we denote the solution at the \f$j\f$th point by \f$u_j\f$. The
 * reconstructed solution on both faces of the \f$j\f$th cell equals the
 * cell-centered value,
 *
 * \f{align}
 * u_{j-1/2} = u_{j+1/2} = u_j,
 * \f}
 *
 * i.e. the slope is zero. This mimics PLUTO's flat reconstruction and is used
 * to reproduce first-order shock-tube solver comparisons.
 */
template <size_t Dim>
void flat(const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_upper_side_of_face_vars,
          const gsl::not_null<std::array<gsl::span<double>, Dim>*>
              reconstructed_lower_side_of_face_vars,
          const gsl::span<const double>& volume_vars,
          const DirectionMap<Dim, gsl::span<const double>>& ghost_cell_vars,
          const Index<Dim>& volume_extents, const size_t number_of_variables) {
  detail::reconstruct<detail::FlatReconstructor>(
      reconstructed_upper_side_of_face_vars,
      reconstructed_lower_side_of_face_vars, volume_vars, ghost_cell_vars,
      volume_extents, number_of_variables);
}
}  // namespace fd::reconstruction
