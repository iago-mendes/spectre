// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <limits>

/// DEBUG PROBE (branch `hllem_degeneracy_debug`, not for merging).
///
/// Where a face of the FD subcell grid sits, so that `Hllem` can write the
/// coordinates of the face states it records. The subcell time derivative
/// sets this before it calls the boundary correction and clears it after;
/// nothing reads it unless the environment variable `SPECTRE_HLLEM_PROBE_DIR`
/// is set. It changes no evolved value.
namespace grmhd::ValenciaDivClean::BoundaryCorrections::hllem_probe {
struct FaceContext {
  bool valid = false;
  double time = std::numeric_limits<double>::signaling_NaN();
  size_t dim = 0;
  // block-logical [-1, 1] bounds of the element in each direction
  std::array<double, 3> xi_lower{};
  std::array<double, 3> xi_upper{};
  // extents of the face mesh: (n+1) cells in `dim`, n in the others
  std::array<size_t, 3> face_extents{};
};

inline FaceContext& context() {
  static thread_local FaceContext ctx{};
  return ctx;
}
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections::hllem_probe
