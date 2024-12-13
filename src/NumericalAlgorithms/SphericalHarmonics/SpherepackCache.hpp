// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

namespace ylm {
class Spherepack;

/// Retrieves a cached Spherepack object with m_max equal to l_max
const Spherepack& get_spherepack_cache(size_t l_max);
}  // namespace ylm
