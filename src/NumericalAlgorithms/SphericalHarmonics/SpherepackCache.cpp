// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/SpherepackCache.hpp"

#include <cstddef>

#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "Utilities/StaticCache.hpp"

namespace ylm {
const Spherepack& get_spherepack_cache(const size_t l_max) {
  static const auto spherepack_cache = make_static_cache<CacheRange<2, 151>>(
      [](const size_t l) { return ylm::Spherepack(l, l); });
  return spherepack_cache(l_max);
}
}  // namespace ylm
