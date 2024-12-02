# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def tortoise_radius_from_boyer_lindquist_minus_r_plus(
    r_minus_r_plus, mass, dimensionless_spin
):
    r_plus = mass * (1.0 + np.sqrt(1.0 - dimensionless_spin**2))
    r_minus = mass * (1.0 - np.sqrt(1.0 - dimensionless_spin**2))
    r = r_minus_r_plus + r_plus
    return r + 2 * mass / (r_plus - r_minus) * (
        r_plus * np.log(r_minus_r_plus / (2 * mass))
        - r_minus * np.log((r - r_minus) / (2 * mass))
    )
