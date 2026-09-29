# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import unittest

import numpy as np
import numpy.testing as npt

from spectre.Pipelines.Bbh.ControlId import control_scales
from spectre.support.Logging import configure_logging

# Control parameters of a bound binary, and the index of the first component of
# each control parameter and its free data in the vectors of residuals and free
# data
BOUND_CONTROL_PARAMS = [
    "MassA",
    "MassB",
    "DimensionlessSpinA",
    "DimensionlessSpinB",
    "CenterOfMass",
    "AdmLinearMomentum",
]
BOUND_PARAM_INDEX = {
    "MassA": 0,
    "conformal_mass_a": 0,
    "MassB": 1,
    "conformal_mass_b": 1,
    "DimensionlessSpinA": 2,
    "horizon_rotation_a": 2,
    "DimensionlessSpinB": 5,
    "horizon_rotation_b": 5,
    "CenterOfMass": 8,
    "center_of_mass_offset": 8,
    "AdmLinearMomentum": 11,
    "linear_velocity": 11,
}


class TestControlScales(unittest.TestCase):
    def test_bound_scales(self):
        for q, chi_a, chi_b in [
            (1.0, 0.0, 0.0),
            (1.0e5, 0.0, 0.0),
            (1.0e5, 0.99, 0.9),
        ]:
            target_params = {
                "MassA": q / (1.0 + q),
                "MassB": 1.0 / (1.0 + q),
                "DimensionlessSpinA": [0.0, 0.0, chi_a],
                "DimensionlessSpinB": [0.0, 0.0, chi_b],
                "CenterOfMass": [0.0, 0.0, 0.0],
                "AdmLinearMomentum": [0.0, 0.0, 0.0],
            }
            # Conformal masses that differ from the targets, so the test
            # distinguishes the two
            conformal_mass_a = 0.85 * target_params["MassA"]
            conformal_mass_b = 0.85 * target_params["MassB"]
            u = np.zeros(14)
            u[0] = conformal_mass_a
            u[1] = conformal_mass_b
            u_scale, f_scale = control_scales(
                BOUND_CONTROL_PARAMS, BOUND_PARAM_INDEX, target_params, u, 14
            )
            with self.subTest(q=q, chi_a=chi_a, chi_b=chi_b):
                # Mass controls and residuals in units of the target masses
                self.assertAlmostEqual(u_scale[0], target_params["MassA"])
                self.assertAlmostEqual(u_scale[1], target_params["MassB"])
                self.assertAlmostEqual(f_scale[0], target_params["MassA"])
                self.assertAlmostEqual(f_scale[1], target_params["MassB"])
                # Spin controls as the effective spin 2 rbar Omega, where rbar
                # is the horizon radius of the conformal Kerr solution
                for idx, chi, conformal_mass in [
                    (2, chi_a, conformal_mass_a),
                    (5, chi_b, conformal_mass_b),
                ]:
                    rbar = conformal_mass * (1.0 + np.sqrt(1.0 - chi**2))
                    npt.assert_allclose(
                        u_scale[idx : idx + 3], 1.0 / (2.0 * rbar), rtol=1e-14
                    )
                # Spin residuals are already dimensionless
                npt.assert_equal(f_scale[2:8], 1.0)
                # Center of mass and ADM linear momentum are not scaled
                npt.assert_equal(u_scale[8:], 1.0)
                npt.assert_equal(f_scale[8:], 1.0)

    def test_hyperbolic_scales(self):
        control_params = ["MassA", "MassB", "AdmMass", "AdmAngularMomentumZ"]
        param_index = {
            "MassA": 0,
            "conformal_mass_a": 0,
            "MassB": 1,
            "conformal_mass_b": 1,
            "AdmMass": 2,
            "radial_expansion_velocity": 2,
            "AdmAngularMomentumZ": 3,
            "orbital_angular_velocity": 3,
        }
        target_params = {
            "MassA": 0.5,
            "MassB": 0.5,
            "AdmMass": 1.02,
            "AdmAngularMomentumZ": 12.5,
        }
        u = np.array([0.42, 0.42, -3.0e-3, 2.0e-4])
        u_scale, f_scale = control_scales(
            control_params, param_index, target_params, u, 4
        )
        # Masses in units of their targets
        npt.assert_allclose(u_scale[:2], 0.5)
        npt.assert_allclose(f_scale[:2], 0.5)
        # Free data of hyperbolic encounters in units of their initial values,
        # and the ADM quantities in units of their targets
        npt.assert_allclose(u_scale[2:], [3.0e-3, 2.0e-4])
        npt.assert_allclose(f_scale[2:], [1.02, 12.5])

    def test_scales_are_positive(self):
        # A target of zero falls back to the conformal mass, and vanishing
        # free data falls back to unit scales
        target_params = {
            "MassA": 0.0,
            "MassB": 0.5,
            "DimensionlessSpinA": [0.0, 0.0, 0.0],
            "DimensionlessSpinB": [0.0, 0.0, 0.0],
            "CenterOfMass": [0.0, 0.0, 0.0],
            "AdmLinearMomentum": [0.0, 0.0, 0.0],
        }
        u = np.zeros(14)
        u[0] = 0.4
        u_scale, f_scale = control_scales(
            BOUND_CONTROL_PARAMS, BOUND_PARAM_INDEX, target_params, u, 14
        )
        self.assertAlmostEqual(u_scale[0], 0.4)
        self.assertAlmostEqual(f_scale[0], 0.4)
        self.assertTrue(np.all(u_scale > 0.0))
        self.assertTrue(np.all(f_scale > 0.0))


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
