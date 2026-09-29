# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import unittest

import numpy as np
import numpy.testing as npt

from spectre.Pipelines.Bbh.ControlId import (
    MAX_EFFECTIVE_SPIN,
    control_scales,
    effective_spin,
    spin_step_size_constraint,
)
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


class TestSpinStepSizeConstraint(unittest.TestCase):
    def chi_eff(self, alpha, mass, delta_mass, rotation, delta_rotation, chi):
        return effective_spin(
            mass + alpha * delta_mass,
            np.asarray(rotation) + alpha * np.asarray(delta_rotation),
            chi,
        )

    def test_effective_spin(self):
        # For the Kerr horizon rotation the effective spin is the spin
        chi = [0.0, 0.6, 0.8]
        mass = 0.7
        rotation = -np.asarray(chi) / (2.0 * mass * (1.0 + np.sqrt(1.0 - 1.0)))
        self.assertAlmostEqual(effective_spin(mass, rotation, chi), 1.0)
        chi = [0.0, 0.0, 0.5]
        rotation = [0.0, 0.0, -0.5 / (2.0 * mass * (1.0 + np.sqrt(0.75)))]
        self.assertAlmostEqual(effective_spin(mass, rotation, chi), 0.5)

    def test_unconstrained_step(self):
        self.assertEqual(
            spin_step_size_constraint(
                0.5, 0.01, [0.0, 0.0, -0.2], [0.0, 0.0, -0.01], [0.0, 0.0, 0.3]
            ),
            1.0,
        )

    def test_rotation_step(self):
        # With a fixed mass the limit is reached where
        # |Omega + alpha dOmega| = max_effective_spin / (2 S Mbar)
        chi = [0.0, 0.0, 0.9]
        mass = 0.45
        spin_term = 1.0 + np.sqrt(1.0 - 0.81)
        rotation, delta_rotation = -0.6, -0.5
        alpha = spin_step_size_constraint(
            mass, 0.0, [0.0, 0.0, rotation], [0.0, 0.0, delta_rotation], chi
        )
        expected_alpha = (
            MAX_EFFECTIVE_SPIN / (2.0 * spin_term * mass) + rotation
        ) / -delta_rotation
        self.assertTrue(0.0 < expected_alpha < 1.0)
        self.assertAlmostEqual(alpha, expected_alpha, places=12)

    def test_mass_and_rotation_step(self):
        # The mass decreases while the rotation grows, so the limit must be
        # solved with the mass that the step actually produces. Solving it with
        # the mass at the end of the full step overshoots the limit.
        chi = [0.0, 0.0, 0.99]
        mass, delta_mass = 0.34, -0.03
        spin_term = 1.0 + np.sqrt(1.0 - 0.99**2)
        rotation = [0.0, 0.0, -0.98 / (2.0 * spin_term * mass)]
        delta_rotation = [0.0, 0.0, -0.4]
        args = (mass, delta_mass, rotation, delta_rotation, chi)
        alpha = spin_step_size_constraint(*args)
        self.assertTrue(0.0 < alpha < 1.0)
        self.assertLessEqual(self.chi_eff(alpha, *args), MAX_EFFECTIVE_SPIN)
        self.assertAlmostEqual(
            self.chi_eff(alpha, *args), MAX_EFFECTIVE_SPIN, places=12
        )
        # Holding the mass at 'mass + delta_mass' while solving for the
        # rotation step gives a longer step, which exceeds the limit
        frozen_mass_alpha = (
            MAX_EFFECTIVE_SPIN / (2.0 * spin_term * (mass + delta_mass))
            + rotation[2]
        ) / -delta_rotation[2]
        self.assertGreater(frozen_mass_alpha, alpha)
        self.assertGreater(
            self.chi_eff(frozen_mass_alpha, *args), MAX_EFFECTIVE_SPIN
        )

    def test_already_above_limit(self):
        chi = [0.0, 0.0, 0.5]
        spin_term = 1.0 + np.sqrt(0.75)
        rotation = [0.0, 0.0, -1.001 / (2.0 * spin_term * 0.5)]
        self.assertEqual(
            spin_step_size_constraint(0.5, 0.0, rotation, [0.0, 0.0, 0.1], chi),
            0.0,
        )

    def test_limit_along_the_whole_step(self):
        # The mass vanishes at the end of the step, so the effective spin rises
        # above the limit and falls back to zero. The step must stop before
        # the effective spin first reaches the limit.
        chi = [0.0, 0.0, 0.0]
        args = (1.0, -1.0, [0.0, 0.0, -0.2], [0.0, 0.0, -1.0], chi)
        self.assertEqual(self.chi_eff(1.0, *args), 0.0)
        alpha = spin_step_size_constraint(*args)
        self.assertTrue(0.0 < alpha < 1.0)
        self.assertAlmostEqual(
            self.chi_eff(alpha, *args), MAX_EFFECTIVE_SPIN, places=12
        )
        for alpha_along_step in np.linspace(0.0, alpha, 100):
            self.assertLessEqual(
                self.chi_eff(alpha_along_step, *args), MAX_EFFECTIVE_SPIN
            )

    def test_random_steps(self):
        rng = np.random.default_rng(seed=42)
        num_limited = 0
        for _ in range(1000):
            chi = [0.0, 0.0, rng.uniform(0.0, 0.999)]
            spin_term = 1.0 + np.sqrt(1.0 - chi[2] ** 2)
            mass = rng.uniform(0.1, 1.0)
            rotation = rng.normal(size=3)
            rotation *= rng.uniform(0.0, 0.99) / (
                2.0 * spin_term * mass * np.linalg.norm(rotation)
            )
            delta_mass = rng.uniform(-0.5, 0.5) * mass
            delta_rotation = rng.normal(size=3) * np.linalg.norm(rotation)
            args = (mass, delta_mass, rotation, delta_rotation, chi)
            alpha = spin_step_size_constraint(*args)
            self.assertTrue(0.0 < alpha <= 1.0)
            for alpha_along_step in np.linspace(0.0, alpha, 20):
                self.assertLessEqual(
                    self.chi_eff(alpha_along_step, *args),
                    MAX_EFFECTIVE_SPIN * (1.0 + 1.0e-14),
                )
            if alpha < 1.0:
                num_limited += 1
                self.assertAlmostEqual(
                    self.chi_eff(alpha, *args), MAX_EFFECTIVE_SPIN, places=10
                )
        # Make sure the test covers limited steps
        self.assertGreater(num_limited, 100)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
