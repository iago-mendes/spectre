# Distributed under the MIT License.
# See LICENSE.txt for details.

import json
import logging
import shutil
import subprocess
import sys
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import numpy.testing as npt
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import to_dataframe
from spectre.Pipelines.Bbh.ControlId import (
    BROYDEN_RESIDUAL_NOISE,
    DEFAULT_CONDITION_THRESHOLD,
    DEFAULT_MAX_ITERATIONS,
    FINITE_DIFFERENCE_STEP,
    GLOBALIZATION_LEGEND,
    GLOBALIZATION_MAX_TRIALS,
    GLOBALIZATIONS,
    LEGACY_GLOBALIZATION_COLUMNS,
    MAX_EFFECTIVE_SPIN,
    NEAR_SINGULAR_STEPS,
    UnconvergedSolveError,
    _convergence_errors,
    _convergence_test,
    _optimal_polynomial_order,
    _read_dat,
    _seeded_solve_failure,
    _watch_seeded_solve,
    control_id,
    control_scales,
    effective_spin,
    elliptic_solve_convergence,
    excision_move,
    near_singular_newton_step,
    restart_control_id,
    spin_step_size_constraint,
)
from spectre.Pipelines.Bbh.InitialData import id_parameters
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


# Synthetic physical parameters that converge exponentially with the
# polynomial order P, except for the center of mass, which converges slowly
class TestNearSingularNewtonStep(unittest.TestCase):
    """The safeguard of the Newton step at a nearly singular Jacobian"""

    def jacobian(self, singular_values):
        """J = U diag(singular_values) V^T with fixed rotations U, V"""
        rotation = lambda angle: np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        U, V = rotation(0.3), rotation(-1.1)
        return U @ np.diag(singular_values) @ V.T, U, V

    def test_well_conditioned(self):
        J, _, _ = self.jacobian([3.0, 1.0])
        F = np.array([0.2, -0.7])
        for method in NEAR_SINGULAR_STEPS:
            step, info = near_singular_newton_step(J, F, method, 10.0)
            npt.assert_allclose(step, -np.linalg.solve(J, F), rtol=1e-14)
            self.assertFalse(info["regularized"])
            self.assertAlmostEqual(info["condition"], 3.0)
            self.assertTrue(np.isnan(info["mu"]) and np.isnan(info["rank"]))

    def test_nearly_singular(self):
        # The residual lies mostly along the image of the nearly null
        # direction, as in the control loops that stall at spin 0.999: the
        # Newton step goes almost only along it
        J, U, V = self.jacobian([10.0, 1.0e-3])
        F = 0.1 * U[:, 0] + 1.0 * U[:, 1]
        newton, info = near_singular_newton_step(J, F, "none", 1.0e3)
        self.assertFalse(info["regularized"])
        self.assertAlmostEqual(info["condition"], 1.0e4, delta=1.0e-6)
        self.assertGreater(
            abs(newton @ V[:, 1]), 0.999 * np.linalg.norm(newton)
        )
        gradient = J.T @ F
        # The truncated SVD leaves the nearly null direction out
        step, info = near_singular_newton_step(J, F, "truncated_svd", 1.0e3)
        self.assertTrue(info["regularized"])
        self.assertEqual(info["rank"], 1)
        npt.assert_allclose(step, -0.1 / 10.0 * V[:, 0], atol=1e-15)
        self.assertLess(gradient @ step, 0.0)
        # Levenberg-Marquardt damps it with mu = (sigma_max / threshold)^2
        step, info = near_singular_newton_step(
            J, F, "levenberg_marquardt", 1.0e3
        )
        self.assertTrue(info["regularized"])
        self.assertAlmostEqual(info["mu"], 1.0e-4, delta=1.0e-16)
        npt.assert_allclose(
            step,
            -(10.0 * 0.1 / (100.0 + 1.0e-4)) * V[:, 0]
            - (1.0e-3 * 1.0 / (1.0e-6 + 1.0e-4)) * V[:, 1],
            rtol=1e-9,
        )
        self.assertLess(gradient @ step, 0.0)
        # Below the threshold nothing changes
        for method in NEAR_SINGULAR_STEPS:
            step, info = near_singular_newton_step(J, F, method, 1.0e5)
            self.assertFalse(info["regularized"])
            npt.assert_allclose(step, newton, rtol=1e-12)

    def test_dennis_schnabel_defaults(self):
        # Dennis & Schnabel's threshold macheps^(-1/2) and their mu,
        # (n macheps)^(1/2) |J^T J|_1
        self.assertAlmostEqual(
            DEFAULT_CONDITION_THRESHOLD, np.finfo(float).eps ** -0.5
        )
        J, U, _ = self.jacobian([1.0, 1.0e-9])
        step, info = near_singular_newton_step(
            J, U[:, 1], "levenberg_marquardt"
        )
        self.assertTrue(info["regularized"])
        self.assertAlmostEqual(
            info["mu"],
            np.sqrt(2.0 * np.finfo(float).eps) * np.linalg.norm(J.T @ J, 1),
            delta=1e-22,
        )
        J, _, _ = self.jacobian([1.0, 1.0e-7])
        _, info = near_singular_newton_step(J, U[:, 1], "levenberg_marquardt")
        self.assertFalse(info["regularized"])


def measured_params_at(P: int):
    return {
        "MassA": 0.5 + 1.0e-2**P,
        "DimensionlessSpinA": np.array([0.0, 0.0, 0.1 + 2.0 * 1.0e-2**P]),
        "CenterOfMass": np.array([1.0e-2 ** (P / 4.0), 0.0, 0.0]),
    }


def mock_generate_id(target_params, run_dir, polynomial_order, **kwargs):
    if polynomial_order in mock_generate_id.failing_orders:
        raise RuntimeError(f"Solve at P={polynomial_order} failed")
    if polynomial_order in mock_generate_id.unconverged_orders:
        # What the postprocessing of the solve raises (see 'PostprocessId.py')
        raise UnconvergedSolveError(f"Residual too large in '{run_dir}'.")
    Path(run_dir).mkdir(parents=True)
    (Path(run_dir) / "P.txt").write_text(str(polynomial_order))


def mock_measured_params(run_dir, control_params):
    P = int((Path(run_dir) / "P.txt").read_text())
    return {
        key: value
        for key, value in measured_params_at(P).items()
        if key in control_params
    }


class TestControlId(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/ControlId"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        # The initial-data directory, with the run that the convergence test
        # starts from as its first segment
        self.id_dir = self.test_dir / "ID"
        self.run_dir = self.id_dir / "0000_InitialData"
        self.run_dir.mkdir(parents=True)
        self.output_filename = self.id_dir / "ControlParams.h5"
        mock_generate_id.failing_orders = []
        mock_generate_id.unconverged_orders = []

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_convergence_errors(self):
        measured_params_by_order = {P: measured_params_at(P) for P in [4, 5, 6]}
        errors = _convergence_errors(
            measured_params_by_order, ["MassA", "DimensionlessSpinA"]
        )
        self.assertEqual(list(errors), [4, 5])
        npt.assert_allclose(errors[4], 2.0 * (1.0e-8 - 1.0e-12), atol=1.0e-14)
        npt.assert_allclose(errors[5], 2.0 * (1.0e-10 - 1.0e-12), atol=1.0e-14)
        errors = _convergence_errors(measured_params_by_order, ["MassA"])
        npt.assert_allclose(errors[4], 1.0e-8 - 1.0e-12, atol=1.0e-14)

    def test_optimal_polynomial_order(self):
        errors = {4: 1.0e-3, 5: 1.0e-6, 6: 1.0e-8}
        self.assertEqual(_optimal_polynomial_order(errors, 1.0e-5), 5)
        self.assertEqual(_optimal_polynomial_order(errors, 1.0e-2), 4)
        self.assertIsNone(_optimal_polynomial_order(errors, 1.0e-9))
        self.assertIsNone(_optimal_polynomial_order({}, 1.0e-5))

    def run_convergence_test(self, polynomial_order, mode, **kwargs):
        (self.run_dir / "P.txt").write_text(str(polynomial_order))
        test_dir = self.id_dir / "ConvergenceTest" / mode.capitalize()
        with (
            patch(
                "spectre.Pipelines.Bbh.ControlId.generate_id",
                side_effect=mock_generate_id,
            ) as generate_id,
            patch(
                "spectre.Pipelines.Bbh.ControlId._measured_params",
                side_effect=mock_measured_params,
            ),
        ):
            result = _convergence_test(
                run_dir=self.run_dir,
                test_dir=test_dir,
                polynomial_order=polynomial_order,
                target_params={},
                free_data={"conformal_mass_a": 0.4},
                separation=20.0,
                refinement_level=1,
                negative_expansion_bc=True,
                mode=mode,
                output_filename=self.output_filename,
                **kwargs,
            )
        return result, test_dir, generate_id

    def read_convergence_errors(self, subfile_name):
        with spectre_h5.H5File(str(self.output_filename), "r") as open_file:
            return to_dataframe(open_file.get_dat(f"{subfile_name}.dat"))

    def test_initial_convergence_test(self):
        result, test_dir, generate_id = self.run_convergence_test(
            polynomial_order=3,
            mode="initial",
            control_params=["MassA", "DimensionlessSpinA", "CenterOfMass"],
            tolerance=1.0e-9,
            max_polynomial_order=8,
        )
        # Probes P=2, then climbs until P=5 meets the tolerance relative to
        # P=6. The slowly converging center of mass doesn't enter the selection.
        self.assertEqual(result, 5)
        solved_orders = [
            call.kwargs["polynomial_order"] for call in generate_id.mock_calls
        ]
        self.assertEqual(solved_orders, [2, 4, 5, 6])
        for call in generate_id.mock_calls:
            P = call.kwargs["polynomial_order"]
            self.assertEqual(
                Path(call.kwargs["run_dir"]).resolve(),
                (test_dir / f"P{P:02d}").resolve(),
            )
            self.assertTrue(call.kwargs["redirect_output"])
            self.assertFalse(call.kwargs["control"])
            self.assertIsNone(call.kwargs["scheduler"])
            self.assertEqual(call.kwargs["conformal_mass_a"], 0.4)
        # The directory of the starting polynomial order links to the run
        self.assertTrue((test_dir / "P03").is_symlink())
        self.assertEqual((test_dir / "P03").resolve(), self.run_dir.resolve())
        # Convergence errors of every component, relative to P=6
        errors = self.read_convergence_errors("InitialConvergenceTest")
        self.assertEqual(list(errors["PolynomialOrder"]), [2, 3, 4, 5, 6])
        self.assertEqual(
            list(errors.columns),
            ["PolynomialOrder", "MassA"]
            + [f"DimensionlessSpinA_{xyz}" for xyz in "xyz"]
            + [f"CenterOfMass_{xyz}" for xyz in "xyz"],
        )
        npt.assert_allclose(
            errors["MassA"],
            [1.0e-2**P - 1.0e-12 for P in range(2, 7)],
            atol=1.0e-14,
        )
        npt.assert_allclose(
            errors["DimensionlessSpinA_z"],
            [2.0 * (1.0e-2**P - 1.0e-12) for P in range(2, 7)],
            atol=1.0e-14,
        )
        npt.assert_allclose(errors["DimensionlessSpinA_x"], 0.0)
        self.assertEqual(errors["CenterOfMass_x"].iloc[-1], 0.0)
        self.assertGreater(errors["CenterOfMass_x"].iloc[-2], 1.0e-9)

    def test_initial_convergence_test_reaches_max(self):
        result, _, generate_id = self.run_convergence_test(
            polynomial_order=3,
            mode="initial",
            control_params=["MassA"],
            tolerance=1.0e-30,
            max_polynomial_order=5,
        )
        self.assertEqual(result, 5)
        self.assertEqual(len(generate_id.mock_calls), 3)

    def test_final_convergence_test(self):
        mock_generate_id.failing_orders = [5]
        result, test_dir, generate_id = self.run_convergence_test(
            polynomial_order=7,
            mode="final",
            control_params=["MassA", "CenterOfMass"],
            tolerance=1.0e-3,
        )
        # Solves at every P from 4 to 9, except the control P=7
        solved_orders = [
            call.kwargs["polynomial_order"] for call in generate_id.mock_calls
        ]
        self.assertEqual(solved_orders, [4, 5, 6, 8, 9])
        self.assertEqual((test_dir / "P07").resolve(), self.run_dir.resolve())
        self.assertFalse((test_dir / "P05").exists())
        # The center of mass enters the selection in the final mode:
        # |1e-2^(P/4) - 1e-2^(9/4)| < 1e-3 first at P=6
        self.assertEqual(result, 6)
        errors = self.read_convergence_errors("FinalConvergenceTest")
        self.assertEqual(list(errors["PolynomialOrder"]), [4, 6, 7, 8, 9])
        # Without the option every solve starts from the background
        for call in generate_id.mock_calls:
            self.assertIsNone(call.kwargs["initial_guess_file_glob"])
            self.assertFalse(call.kwargs["numeric_initial_guess_in_tests"])

    def test_unconverged_solve_in_convergence_test(self):
        # An unconverged solve is skipped like a failed one, and reported as
        # unconverged
        mock_generate_id.unconverged_orders = [5]
        with self.assertLogs(
            "spectre.Pipelines.Bbh.ControlId", level="WARNING"
        ) as logs:
            result, test_dir, generate_id = self.run_convergence_test(
                polynomial_order=7,
                mode="final",
                control_params=["MassA", "CenterOfMass"],
                tolerance=1.0e-3,
            )
        self.assertEqual(result, 6)
        errors = self.read_convergence_errors("FinalConvergenceTest")
        self.assertEqual(list(errors["PolynomialOrder"]), [4, 6, 7, 8, 9])
        unconverged = [record for record in logs.output if "P=5" in record]
        self.assertEqual(len(unconverged), 1)
        self.assertIn(
            "solve at P=5 did not converge, skipping it: Residual too large",
            unconverged[0],
        )

    def test_numeric_initial_guess(self):
        mock_generate_id.failing_orders = [5]
        result, test_dir, generate_id = self.run_convergence_test(
            polynomial_order=6,
            mode="final",
            control_params=["MassA"],
            tolerance=1.0e-3,
            numeric_initial_guess=True,
        )
        # Every solve starts from the nearest P that is already solved, the
        # higher one on a tie. The failed solve at P=5 is never a seed.
        seed_dirs = {
            call.kwargs["polynomial_order"]: (
                Path(call.kwargs["initial_guess_file_glob"]).parent.resolve()
            )
            for call in generate_id.mock_calls
        }
        self.assertEqual(
            seed_dirs,
            {
                4: self.run_dir.resolve(),
                5: self.run_dir.resolve(),
                7: self.run_dir.resolve(),
                8: (test_dir / "P07").resolve(),
            },
        )
        for call in generate_id.mock_calls:
            self.assertEqual(
                Path(call.kwargs["initial_guess_file_glob"]).name,
                "BbhVolume*.h5",
            )
            self.assertTrue(call.kwargs["numeric_initial_guess_in_tests"])

    def solved_orders(self, generate_id):
        return [
            call.kwargs["polynomial_order"] for call in generate_id.mock_calls
        ]

    def test_final_convergence_test_no_climb_by_default(self):
        result, _, generate_id = self.run_convergence_test(
            polynomial_order=6,
            mode="final",
            control_params=["CenterOfMass"],
            tolerance=1.0e-4,
        )
        # |1e-2^(P/4) - 1e-2^(8/4)| >= 1e-4 for every P < 8
        self.assertIsNone(result)
        self.assertEqual(self.solved_orders(generate_id), [4, 5, 7, 8])

    def test_final_convergence_test_climbs(self):
        result, _, generate_id = self.run_convergence_test(
            polynomial_order=6,
            mode="final",
            control_params=["CenterOfMass"],
            tolerance=1.0e-4,
            max_polynomial_order=12,
        )
        # No P meets the tolerance relative to P=8, so the test climbs to P=9,
        # where |1e-2^(8/4) - 1e-2^(9/4)| < 1e-4, and stops there
        self.assertEqual(result, 8)
        self.assertEqual(self.solved_orders(generate_id), [4, 5, 7, 8, 9])
        errors = self.read_convergence_errors("FinalConvergenceTest")
        self.assertEqual(list(errors["PolynomialOrder"]), [4, 5, 6, 7, 8, 9])

    def test_final_convergence_test_climbs_to_max(self):
        result, _, generate_id = self.run_convergence_test(
            polynomial_order=6,
            mode="final",
            control_params=["CenterOfMass"],
            tolerance=1.0e-30,
            max_polynomial_order=10,
        )
        self.assertIsNone(result)
        self.assertEqual(self.solved_orders(generate_id), [4, 5, 7, 8, 9, 10])

    def test_final_convergence_test_full_ladder_before_climb(self):
        # A P below the control's meets the tolerance only relative to P=8, so
        # the whole ladder up to max(8, P + 2) is solved and nothing above it
        result, _, generate_id = self.run_convergence_test(
            polynomial_order=6,
            mode="final",
            control_params=["MassA"],
            tolerance=1.0e-9,
            max_polynomial_order=12,
        )
        self.assertEqual(result, 5)
        self.assertEqual(self.solved_orders(generate_id), [4, 5, 7, 8])

    def test_elliptic_absolute_residual(self):
        # Every solve of the test gets the elliptic solver's tolerance, and
        # none gets it by default
        for kwargs, expected in (
            ({}, None),
            (dict(elliptic_absolute_residual=1.0e-10), 1.0e-10),
        ):
            _, _, generate_id = self.run_convergence_test(
                polynomial_order=6,
                mode="final",
                control_params=["MassA"],
                tolerance=1.0e-4,
                **kwargs,
            )
            self.assertEqual(self.solved_orders(generate_id), [4, 5, 7, 8])
            for call in generate_id.mock_calls:
                self.assertEqual(
                    call.kwargs["elliptic_absolute_residual"], expected
                )


# Solver output, one entry per line
def newton_log(*lines):
    return "\n".join(lines) + "\n"


ITERATION = "NewtonRaphson({}) iteration complete ({} globalization steps,"
OVERSHOOT = (
    "NewtonRaphson({}): Step with length {} didn't sufficiently decrease the"
    " residual (possible overshoot). Residual: {}. Next step length: {}."
)
# A solve whose guess is far from the solution (eiger 8211478, control solve
# 1): the full step of the second Newton iteration raises the residual by 2e5
STALLED_SOLVE_OUTPUT = newton_log(
    "NewtonRaphson initialized with residual: 3.005263e+01",
    ITERATION.format(1, 0) + " step length 1). Remaining residual: 6.709558",
    OVERSHOOT.format(2, 1, "5.893451e+06", 0.1),
    OVERSHOOT.format(2, 0.1, "6.962544e+00", 0.05),
    ITERATION.format(2, 2) + " step length 0.05). Remaining residual: 6.6195",
)
# A solve from the background (eiger 8211781, control solve 0, iterations 1
# and 7) whose first full step raises the residual by a factor 1.2
GOOD_SOLVE_OUTPUT = newton_log(
    "NewtonRaphson initialized with residual: 1.619000e+01",
    OVERSHOOT.format(1, 1, "1.967734e+01", 0.403682),
    ITERATION.format(1, 1) + " step length 0.403682). Remaining residual: 13.0",
    ITERATION.format(2, 0) + " step length 1). Remaining residual: 9.87e-12",
    "NewtonRaphson has converged in 2 iterations: AbsoluteResidual - The"
    " residual magnitude has decreased to 1e-10 or below (9.87491e-12).",
)
UNCONVERGED_SOLVE_OUTPUT = newton_log(
    "NewtonRaphson initialized with residual: 1.0e+00",
    ITERATION.format(1, 0) + " step length 1). Remaining residual: 5.0e-01",
    # The solver's words when it stops at its maximum number of iterations
    "NewtonRaphson has converged in 1 iterations: Reached the maximum number"
    " of iterations (1).",
)

# A control loop of the conformal masses of a non-spinning binary, where the
# measured Christodoulou mass is the nonlinear function 'mock_mass' of the
# conformal mass. Starting from INITIAL_CONFORMAL_MASS, the first steps of the
# free data change the conformal masses by tens of percent and the later ones
# by less.
SEEDING_CONTROL_PARAMS = [
    "MassA",
    "MassB",
    "DimensionlessSpinA",
    "DimensionlessSpinB",
]
INITIAL_CONFORMAL_MASS = 0.15


def mock_mass(conformal_mass):
    return conformal_mass + 2.0 * conformal_mass**2


def seeding_input_file():
    kerr = lambda mass: {
        "KerrSchild": {
            "Mass": mass,
            "Spin": [0.0, 0.0, 0.0],
            "Center": [0.0, 0.0, 0.0],
        }
    }
    horizon = {
        "Interior": {
            "ExciseWithBoundaryCondition": {
                "ApparentHorizon": {"Rotation": [0.0, 0.0, 0.0]}
            }
        }
    }
    metadata = {
        "TargetParams": {
            "MassA": 0.5,
            "MassB": 0.5,
            "MassRatio": 1.0,
            "DimensionlessSpinA": [0.0, 0.0, 0.0],
            "DimensionlessSpinB": [0.0, 0.0, 0.0],
        }
    }
    body = {
        "Background": {
            "Binary": {
                "XCoords": [-5.0, 5.0],
                "CenterOfMassOffset": [0.0, 0.0],
                "ObjectLeft": kerr(INITIAL_CONFORMAL_MASS),
                "ObjectRight": kerr(INITIAL_CONFORMAL_MASS),
                "AngularVelocity": 0.01,
                "Expansion": 0.0,
                "LinearVelocity": [0.0, 0.0, 0.0],
            }
        },
        "DomainCreator": {
            "BinaryCompactObject": {"ObjectA": horizon, "ObjectB": horizon}
        },
    }
    return metadata, body


def write_seeding_segment(run_dir, free_data, output):
    run_dir.mkdir(parents=True)
    (run_dir / "free_data.json").write_text(json.dumps(free_data))
    if output is not None:
        (run_dir / "spectre.out").write_text(output)


def mock_control_generate_id(
    target_params, segments_dir, initial_guess_file_glob, **kwargs
):
    """Runs the next segment. A solve from a previous solution 'fails' in the
    ways listed in 'mock_control_generate_id.seeded_failures', one per seeded
    solve, in order ('None' for a good solve). The solves from the background
    whose indices (counting from 0, over the solves from the background in
    this control loop) are in 'mock_control_generate_id.unconverged_background'
    don't converge."""
    num_segments = len(list(Path(segments_dir).glob("[0-9]*_InitialData")))
    run_dir = Path(segments_dir) / f"{num_segments:04d}_InitialData"
    free_data = {
        key: kwargs[key] for key in ["conformal_mass_a", "conformal_mass_b"]
    }
    mock_control_generate_id.calls.append(
        dict(run_dir=run_dir, glob=initial_guess_file_glob, **free_data)
    )
    failure = None
    if initial_guess_file_glob is not None and (
        mock_control_generate_id.seeded_failures
    ):
        failure = mock_control_generate_id.seeded_failures.pop(0)
    if initial_guess_file_glob is None:
        background_index = sum(
            call["glob"] is None for call in mock_control_generate_id.calls[:-1]
        )
        if background_index in mock_control_generate_id.unconverged_background:
            failure = "unconverged"
    if failure == "overshoot":
        write_seeding_segment(run_dir, free_data, STALLED_SOLVE_OUTPUT)
        raise subprocess.CalledProcessError(returncode=143, cmd="SolveXcts")
    write_seeding_segment(
        run_dir,
        free_data,
        (
            UNCONVERGED_SOLVE_OUTPUT
            if failure == "unconverged"
            else GOOD_SOLVE_OUTPUT
        ),
    )
    if failure == "unconverged":
        # What the postprocessing of the solve raises (see 'PostprocessId.py')
        raise UnconvergedSolveError(
            f"The elliptic solve in '{run_dir}' did not converge."
        )


def mock_control_measured_params(run_dir, control_params):
    free_data = json.loads((Path(run_dir) / "free_data.json").read_text())
    return {
        "MassA": mock_mass(free_data["conformal_mass_a"]),
        "MassB": mock_mass(free_data["conformal_mass_b"]),
        "DimensionlessSpinA": np.zeros(3),
        "DimensionlessSpinB": np.zeros(3),
    }


class TestNumericInitialGuessInControl(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/ControlIdSeeding"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.id_dir = self.test_dir / "ID"
        self.run_dir = self.id_dir / "0000_InitialData"
        write_seeding_segment(
            self.run_dir,
            {
                "conformal_mass_a": INITIAL_CONFORMAL_MASS,
                "conformal_mass_b": INITIAL_CONFORMAL_MASS,
            },
            GOOD_SOLVE_OUTPUT,
        )
        self.input_file = self.run_dir / "InitialData.yaml"
        with open(self.input_file, "w") as open_input_file:
            yaml.safe_dump_all(seeding_input_file(), open_input_file)
        mock_control_generate_id.calls = []
        mock_control_generate_id.seeded_failures = []
        mock_control_generate_id.unconverged_background = []

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_control(self, **kwargs):
        with (
            patch(
                "spectre.Pipelines.Bbh.ControlId.generate_id",
                side_effect=mock_control_generate_id,
            ),
            patch(
                "spectre.Pipelines.Bbh.ControlId._measured_params",
                side_effect=mock_control_measured_params,
            ),
        ):
            last_run_dir = control_id(
                id_input_file_path=self.input_file,
                residual_tolerance=1.0e-8,
                **{
                    "control_params": SEEDING_CONTROL_PARAMS,
                    "numeric_initial_guess": True,
                    **kwargs,
                },
            )
        return Path(last_run_dir), mock_control_generate_id.calls

    def test_subset_of_horizon_params(self):
        # A spin or a mass that is not controlled has no row or column in the
        # initial Jacobian, and its free data keeps its initial value
        for control_params in [
            ["MassA", "MassB", "DimensionlessSpinA"],
            ["MassA", "DimensionlessSpinB"],
            ["DimensionlessSpinA"],
        ]:
            with self.subTest(control_params=control_params):
                self.tearDown()
                self.setUp()
                _, calls = self.run_control(control_params=control_params)
                for mass_key, free_data_key in [
                    ("MassA", "conformal_mass_a"),
                    ("MassB", "conformal_mass_b"),
                ]:
                    if mass_key in control_params:
                        self.assertAlmostEqual(
                            mock_mass(calls[-1][free_data_key]), 0.5, places=7
                        )
                    else:
                        for call in calls:
                            self.assertEqual(
                                call[free_data_key], INITIAL_CONFORMAL_MASS
                            )

    def test_excision_move(self):
        params = lambda x_right, y, radius_right, radius_left: {
            "XRight": x_right,
            "XLeft": x_right - 10.0,
            "CenterOfMassOffset_y": y,
            "CenterOfMassOffset_z": 0.0,
            "ExcisionRadiusRight": radius_right,
            "ExcisionRadiusLeft": radius_left,
        }
        # The right excision shrinks from 0.8 to 0.6
        self.assertAlmostEqual(
            excision_move(
                params(1.0, 0.0, 0.8, 0.1), params(1.0, 0.0, 0.6, 0.1)
            ),
            1.0 / 3.0,
        )
        # The right excision grows: the new domain reaches into nothing
        self.assertEqual(
            excision_move(
                params(1.0, 0.0, 0.6, 0.1), params(1.0, 0.0, 0.8, 0.1)
            ),
            0.0,
        )
        # Both centers move by 0.05, half the left (smaller) radius
        self.assertAlmostEqual(
            excision_move(
                params(1.0, 0.0, 0.8, 0.1), params(1.03, 0.04, 0.8, 0.1)
            ),
            0.5,
        )

    def test_seeded_solve_failure(self):
        self.assertIn(
            "Newton iteration 2", _seeded_solve_failure(STALLED_SOLVE_OUTPUT)
        )
        self.assertIsNone(_seeded_solve_failure(GOOD_SOLVE_OUTPUT))
        # Not an overshoot. Whether a finished solve converged is decided from
        # its residuals (see 'TestEllipticSolveConvergence').
        self.assertIsNone(_seeded_solve_failure(UNCONVERGED_SOLVE_OUTPUT))
        self.assertIsNone(_seeded_solve_failure(""))

    @unittest.skipUnless(sys.platform.startswith("linux"), "needs /proc")
    def test_watch_seeded_solve(self):
        # A stand-in for the executable, running in a new segment, whose output
        # shows an overshooting Newton step: the watcher stops it
        segment = self.id_dir / "0001_InitialData"
        segment.mkdir()
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "SolveXcts"],
            cwd=segment,
        )
        (segment / "spectre.out").write_text(STALLED_SOLVE_OUTPUT)
        stop, failure = threading.Event(), []
        _watch_seeded_solve(
            self.id_dir, self.run_dir, stop, failure, poll_interval=0.1
        )
        self.assertEqual(process.wait(timeout=10), -15)
        self.assertEqual(len(failure), 1)
        self.assertIn("Newton iteration 2", failure[0])

    def test_seeding_without_threshold(self):
        # Every solve after the first starts from the previous one
        last_run_dir, calls = self.run_control()
        self.assertGreater(len(calls), 3)
        for i, call in enumerate(calls):
            self.assertEqual(
                call["glob"],
                str(self.id_dir / f"{i:04d}_InitialData" / "BbhVolume*.h5"),
            )
        self.assertEqual(last_run_dir, calls[-1]["run_dir"])

    def test_seeding_threshold(self):
        last_run_dir, calls = self.run_control(
            numeric_initial_guess_max_excision_move=0.1
        )
        previous = dict(
            conformal_mass_a=INITIAL_CONFORMAL_MASS,
            conformal_mass_b=INITIAL_CONFORMAL_MASS,
        )
        steps = []
        for i, call in enumerate(calls):
            # The centers don't move (the center of mass isn't controlled), so
            # the excision move is the relative shrinking of the radii, which
            # are proportional to the conformal masses
            step = max(
                max(0.0, previous[key] - call[key]) / call[key]
                for key in ["conformal_mass_a", "conformal_mass_b"]
            )
            steps.append(step)
            if step < 0.1:
                self.assertEqual(
                    call["glob"],
                    str(self.id_dir / f"{i:04d}_InitialData" / "BbhVolume*.h5"),
                )
            else:
                self.assertIsNone(call["glob"])
            previous = call
        # Some steps reach deep and the last ones don't, so the test covers
        # both branches
        self.assertGreater(max(steps), 0.1)
        self.assertLess(steps[-1], 0.1)
        # Same control as without the threshold: the guess doesn't change the
        # (mock) solution
        _, calls_without = (self.tearDown(), self.setUp(), self.run_control())[
            2
        ]
        self.assertEqual(len(calls), len(calls_without))
        for call, call_without in zip(calls, calls_without):
            self.assertEqual(
                call["conformal_mass_a"], call_without["conformal_mass_a"]
            )

    def test_seeding_control_residual_threshold(self):
        threshold = 1.0e-3
        with self.assertLogs(
            "spectre.Pipelines.Bbh.ControlId", level="INFO"
        ) as logs:
            last_run_dir, calls = self.run_control(
                numeric_initial_guess_max_control_residual=threshold
            )
        # The control residual of each solve: the mock masses of segment i,
        # where segment 0 is the initial solve and call i runs segment i + 1
        control_residual = lambda free_data: max(
            abs(mock_mass(free_data[key]) - 0.5)
            for key in ["conformal_mass_a", "conformal_mass_b"]
        )
        previous_residual = control_residual(
            dict(
                conformal_mass_a=INITIAL_CONFORMAL_MASS,
                conformal_mass_b=INITIAL_CONFORMAL_MASS,
            )
        )
        previous_residuals = []
        for i, call in enumerate(calls):
            previous_residuals.append(previous_residual)
            if previous_residual < threshold:
                self.assertEqual(
                    call["glob"],
                    str(self.id_dir / f"{i:04d}_InitialData" / "BbhVolume*.h5"),
                )
            else:
                self.assertIsNone(call["glob"])
            previous_residual = control_residual(call)
        # Some solves follow a large control residual and the last ones don't,
        # so the test covers both branches
        self.assertGreaterEqual(max(previous_residuals), threshold)
        self.assertGreaterEqual(previous_residuals[1], threshold)
        self.assertLess(previous_residuals[-1], threshold)
        # Every solve after the first logs its decision and why
        decisions = [
            record
            for record in logs.output
            if "Numeric initial guess: previous control residual =" in record
        ]
        self.assertEqual(len(decisions), len(calls))
        for decision, call in zip(decisions, calls):
            self.assertIn(
                (
                    "start from the background."
                    if call["glob"] is None
                    else "start from the previous solution."
                ),
                decision,
            )
        # Same control as without the threshold: the guess doesn't change the
        # (mock) solution
        _, calls_without = (self.tearDown(), self.setUp(), self.run_control())[
            2
        ]
        self.assertEqual(len(calls), len(calls_without))
        for call, call_without in zip(calls, calls_without):
            self.assertEqual(
                call["conformal_mass_a"], call_without["conformal_mass_a"]
            )

    def test_seeding_both_thresholds(self):
        # A solve starts from the previous solution only if both conditions
        # allow it: with an excision-move threshold that allows every step,
        # the control-residual threshold decides alone, and vice versa
        _, calls_residual = self.run_control(
            numeric_initial_guess_max_control_residual=1.0e-3
        )
        self.tearDown()
        self.setUp()
        _, calls_both = self.run_control(
            numeric_initial_guess_max_control_residual=1.0e-3,
            numeric_initial_guess_max_excision_move=1.0e6,
        )
        self.assertEqual(
            [call["glob"] for call in calls_both],
            [call["glob"] for call in calls_residual],
        )
        self.tearDown()
        self.setUp()
        _, calls_move = self.run_control(
            numeric_initial_guess_max_excision_move=0.1
        )
        self.tearDown()
        self.setUp()
        _, calls_both = self.run_control(
            numeric_initial_guess_max_control_residual=1.0e6,
            numeric_initial_guess_max_excision_move=0.1,
        )
        self.assertEqual(
            [call["glob"] for call in calls_both],
            [call["glob"] for call in calls_move],
        )

    def test_seeding_fallback(self):
        mock_control_generate_id.seeded_failures = [
            "overshoot",
            "unconverged",
        ]
        last_run_dir, calls = self.run_control(
            numeric_initial_guess_fallback=True
        )
        # Solve 1 from solve 0 overshoots and is redone from the background
        # (segment 2); solve 2 from segment 2 doesn't converge and is redone
        # from the background (segment 4); later solves start from the
        # previous accepted segment
        self.assertIsNotNone(calls[0]["glob"])
        self.assertIsNone(calls[1]["glob"])
        self.assertEqual(
            calls[1]["conformal_mass_a"], calls[0]["conformal_mass_a"]
        )
        self.assertEqual(
            calls[2]["glob"],
            str(self.id_dir / "0002_InitialData" / "BbhVolume*.h5"),
        )
        self.assertIsNone(calls[3]["glob"])
        self.assertEqual(
            calls[3]["conformal_mass_a"], calls[2]["conformal_mass_a"]
        )
        self.assertEqual(
            calls[4]["glob"],
            str(self.id_dir / "0004_InitialData" / "BbhVolume*.h5"),
        )
        self.assertEqual(last_run_dir, calls[-1]["run_dir"])
        # Without the fallback, the failed solve stops the control loop
        self.tearDown()
        self.setUp()
        mock_control_generate_id.seeded_failures = ["overshoot"]
        with self.assertRaises(subprocess.CalledProcessError):
            self.run_control()

    def test_unconverged_control_solve(self):
        # An unconverged solve stops the control loop with an error that names
        # the control iteration, instead of measuring its parameters
        mock_control_generate_id.unconverged_background = [1]
        with self.assertRaisesRegex(
            UnconvergedSolveError,
            r"Control iteration 2: The elliptic solve in .*0002_InitialData.*"
            r" did not converge",
        ):
            self.run_control(numeric_initial_guess=False)
        self.assertEqual(len(mock_control_generate_id.calls), 2)

    def test_unconverged_control_solve_with_fallback(self):
        # A seeded solve that doesn't converge is redone from the background
        mock_control_generate_id.seeded_failures = ["unconverged"]
        last_run_dir, calls = self.run_control(
            numeric_initial_guess_fallback=True
        )
        self.assertIsNotNone(calls[0]["glob"])
        self.assertIsNone(calls[1]["glob"])
        self.assertEqual(
            calls[1]["conformal_mass_a"], calls[0]["conformal_mass_a"]
        )
        self.assertEqual(last_run_dir, calls[-1]["run_dir"])
        # If the redone solve doesn't converge either, the control loop stops
        self.tearDown()
        self.setUp()
        mock_control_generate_id.seeded_failures = ["unconverged"]
        mock_control_generate_id.unconverged_background = [0]
        with self.assertRaisesRegex(
            UnconvergedSolveError, r"Control iteration 1: .*0002_InitialData"
        ):
            self.run_control(numeric_initial_guess_fallback=True)
        self.assertEqual(len(mock_control_generate_id.calls), 2)
        # Without the fallback, the unconverged seeded solve stops it
        self.tearDown()
        self.setUp()
        mock_control_generate_id.seeded_failures = ["unconverged"]
        with self.assertRaisesRegex(
            UnconvergedSolveError, r"Control iteration 1: .*0001_InitialData"
        ):
            self.run_control()


NEWTON_LEGEND = [
    "Iteration",
    "Walltime",
    "GlobalizationStep",
    "Residual",
    "StepLength",
]


def newton_input_file(absolute_residual=1.0e-10, max_iterations=10):
    return {
        "Observers": {"ReductionFileName": "BbhReductions"},
        "NonlinearSolver": {
            "NewtonRaphson": {
                "ConvergenceCriteria": {
                    "MaxIterations": max_iterations,
                    "RelativeResidual": 0.0,
                    "AbsoluteResidual": absolute_residual,
                }
            }
        },
    }


def write_newton_residuals(run_dir, rows):
    """Write rows (iteration, globalization step, residual) like the solver"""
    with spectre_h5.H5File(str(run_dir / "BbhReductions.h5"), "a") as h5file:
        dat_file = h5file.try_insert_dat(
            "NewtonRaphsonResiduals", NEWTON_LEGEND, 0
        )
        for i, (iteration, globalization_step, residual) in enumerate(rows):
            dat_file.append(
                [
                    float(iteration),
                    float(i),
                    float(globalization_step),
                    residual,
                    1.0,
                ]
            )


# Residuals of a solve that converged in 5 iterations (jarrus, 'q1' P = 4,
# solve 0, Newton 1e-10)
CONVERGED_ROWS = [
    (0, 0, 7.74728203),
    (1, 0, 1.20553895),
    (2, 0, 1.48862250e-03),
    (3, 0, 5.04113365e-05),
    (4, 0, 3.46045792e-08),
    (5, 0, 2.67020329e-11),
]
# Residuals of a solve that reached MaxIterations = 10 (jarrus,
# 'q20_spinsAB' replay k09): every full step overshoots and a step of length
# 0.1 is taken. The solver reports "has converged in 10 iterations: Reached
# the maximum number of iterations (10)".
CAPPED_ROWS = [(0, 0, 5.29922198)] + [
    row
    for iteration, (trial, accepted) in enumerate(
        [
            (1.60340010e07, 5.11592927),
            (1.80658351e02, 4.63544583),
            (1.09794959e01, 3.95295999),
            (3.36460248e02, 3.56500966),
            (3.97017391e04, 3.23090059),
            (2.77856159e07, 2.99454568),
            (2.52493392e06, 2.75328482),
            (2.03989875e04, 2.48924314),
            (1.51345665e03, 2.25158835),
            (4.63290711e03, 2.02844549),
        ],
        start=1,
    )
    for row in [(iteration, 0, trial), (iteration, 1, accepted)]
]


class TestBroydenUpdate(unittest.TestCase):
    """The 'paper' and 'secant' updates of the Jacobian on the mocked control
    loop of the conformal masses (see 'TestNumericInitialGuessInControl')."""

    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/ControlIdBroyden"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_control(self, name, mass=mock_mass, noise=0.0, **kwargs):
        """Runs the control loop in its own directory, with the measured mass
        'mass(conformal_mass)' plus normal noise of the given size. Returns
        the tables of 'ControlParams.h5'."""
        id_dir = self.test_dir / name / "ID"
        run_dir = id_dir / "0000_InitialData"
        write_seeding_segment(
            run_dir,
            {
                "conformal_mass_a": INITIAL_CONFORMAL_MASS,
                "conformal_mass_b": INITIAL_CONFORMAL_MASS,
            },
            GOOD_SOLVE_OUTPUT,
        )
        with open(run_dir / "InitialData.yaml", "w") as open_input_file:
            yaml.safe_dump_all(seeding_input_file(), open_input_file)
        mock_control_generate_id.calls = []
        mock_control_generate_id.seeded_failures = []
        mock_control_generate_id.unconverged_background = []
        rng = np.random.default_rng(42)

        def measured_params(run_dir, control_params):
            free_data = json.loads(
                (Path(run_dir) / "free_data.json").read_text()
            )
            return {
                "MassA": (
                    mass(free_data["conformal_mass_a"])
                    + noise * rng.standard_normal()
                ),
                "MassB": (
                    mass(free_data["conformal_mass_b"])
                    + noise * rng.standard_normal()
                ),
                "DimensionlessSpinA": np.zeros(3),
                "DimensionlessSpinB": np.zeros(3),
            }

        try:
            with (
                patch(
                    "spectre.Pipelines.Bbh.ControlId.generate_id",
                    side_effect=mock_control_generate_id,
                ),
                patch(
                    "spectre.Pipelines.Bbh.ControlId._measured_params",
                    side_effect=measured_params,
                ),
            ):
                control_id(
                    id_input_file_path=run_dir / "InitialData.yaml",
                    control_params=SEEDING_CONTROL_PARAMS,
                    residual_tolerance=1.0e-8,
                    **kwargs,
                )
        finally:
            self.tables = {
                table: _read_dat(id_dir / "ControlParams.h5", table)
                for table in [
                    "ControlScales",
                    "FreeData",
                    "Residuals",
                    "Jacobian",
                    "StepSizeConstraints",
                ]
            }
        return self.tables

    def secant_residuals(self, tables, jacobians):
        """|J_k du_k - (F_k - F_{k-1})| of each update, in scaled variables"""
        n = tables["FreeData"].shape[1]
        u_scale, f_scale = np.split(tables["ControlScales"][0], [n])
        u = tables["FreeData"] / u_scale
        F = tables["Residuals"] / f_scale
        return [
            np.max(
                np.abs(
                    jacobians[k].reshape(n, n) @ (u[k] - u[k - 1])
                    - (F[k] - F[k - 1])
                )
            )
            for k in range(1, len(jacobians))
        ]

    def test_full_steps_reproduce_paper(self):
        # Without step-size constraints that bind, every step is the full
        # Newton step, and the two updates are the same to the last bit
        paper = self.run_control("paper", broyden_update="paper")
        secant = self.run_control("secant", broyden_update="secant")
        self.assertTrue(np.all(paper["StepSizeConstraints"][:, 3] == 1.0))
        self.assertGreater(len(paper["Jacobian"]), 3)
        self.assertTrue(np.array_equal(paper["Jacobian"], secant["Jacobian"]))
        self.assertTrue(np.array_equal(paper["FreeData"], secant["FreeData"]))

    def test_shortened_steps_satisfy_secant_condition(self):
        tables = self.run_control(
            "secant", broyden_update="secant", step_size_constraints=["mass"]
        )
        alphas = tables["StepSizeConstraints"][:, 3]
        self.assertLess(alphas[0], 0.5)
        self.assertLess(np.max(np.abs(tables["Residuals"][-1])), 1.0e-8)
        # J_k du_k = F_k - F_{k-1}, except in components that changed as
        # predicted within the noise
        for residual in self.secant_residuals(tables, tables["Jacobian"]):
            self.assertLess(residual, BROYDEN_RESIDUAL_NOISE)
        # The paper's update misses it after a shortened step
        n = tables["FreeData"].shape[1]
        u_scale, f_scale = np.split(tables["ControlScales"][0], [n])
        du = (tables["FreeData"][1] - tables["FreeData"][0]) / u_scale
        paper_J1 = tables["Jacobian"][0].reshape(n, n) + np.outer(
            tables["Residuals"][1] / f_scale, du
        ) / np.dot(du, du)
        self.assertGreater(
            self.secant_residuals(tables, [None, paper_J1])[0], 1.0e-2
        )

    def test_noise_guard(self):
        # The measured mass is linear with the slope of the initial Jacobian,
        # plus noise below BROYDEN_RESIDUAL_NOISE. Every shortened step then
        # changes the residuals as predicted, within the noise, so the
        # 'secant' update leaves the Jacobian alone. The 'paper' update
        # doesn't.
        linear = lambda conformal_mass: conformal_mass + 0.1
        kwargs = dict(
            mass=linear, noise=1.0e-10, step_size_constraints=["mass"]
        )
        secant = self.run_control("secant", broyden_update="secant", **kwargs)
        paper = self.run_control("paper", broyden_update="paper", **kwargs)
        self.assertLess(secant["StepSizeConstraints"][0, 3], 0.5)
        self.assertGreater(len(secant["Jacobian"]), 3)
        for J in secant["Jacobian"]:
            self.assertTrue(np.array_equal(J, secant["Jacobian"][0]))
        self.assertGreater(
            np.max(np.abs(paper["Jacobian"][1] - paper["Jacobian"][0])), 0.1
        )

    def test_stop_on_unmeasurable_step(self):
        # A step-size constraint that allows only steps of 1e-12: the 'secant'
        # update stops before the solve. The 'paper' update divides the
        # residual by the step and blows up the Jacobian.
        with patch(
            "spectre.Pipelines.Bbh.ControlId.MAX_RELATIVE_MASS_STEP", 1.0e-12
        ):
            with self.assertRaisesRegex(RuntimeError, "cannot make progress"):
                self.run_control(
                    "secant",
                    broyden_update="secant",
                    step_size_constraints=["mass"],
                )
            self.assertEqual(mock_control_generate_id.calls, [])
            self.assertEqual(len(self.tables["Jacobian"]), 1)
            paper = self.run_control(
                "paper",
                broyden_update="paper",
                step_size_constraints=["mass"],
                max_iterations=1,
            )
        self.assertGreater(np.max(np.abs(paper["Jacobian"][1])), 1.0e10)


# A measured mass that saturates: the response of the mass to the conformal
# mass falls off like the response of a spin near extremality to the horizon
# rotation. From beyond the solution (SATURATING_SOLUTION, about 1.17), where
# the response is weak, Broyden's method steps far past the solution and the
# control residual jumps.
SATURATING_MASS_SCALE = 0.52
SATURATING_INITIAL_CONFORMAL_MASS = 1.5


def saturating_mass(conformal_mass):
    return SATURATING_MASS_SCALE * np.tanh(
        conformal_mass / SATURATING_MASS_SCALE
    )


def column(table, name):
    return table[:, GLOBALIZATION_LEGEND.index(name)]


def jumps(residuals, factor=3.0):
    """Solves whose max |F| exceeds 'factor' times the best one before"""
    max_residuals = np.max(np.abs(residuals), axis=1)
    best = np.minimum.accumulate(max_residuals)
    return [
        solve
        for solve in range(1, len(max_residuals))
        if max_residuals[solve] > factor * best[solve - 1]
    ]


class TestGlobalization(unittest.TestCase):
    """The line search and the trust region on the mocked control loop of the
    conformal masses (see 'TestNumericInitialGuessInControl')."""

    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(),
            "support/Pipelines/Bbh/ControlIdGlobalization",
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def run_control(
        self,
        name,
        mass=saturating_mass,
        initial_conformal_mass=SATURATING_INITIAL_CONFORMAL_MASS,
        measured_masses=None,
        **kwargs,
    ):
        """Runs the control loop in its own directory, with the measured mass
        'mass(conformal_mass)' of each black hole, or the measured masses
        'measured_masses(conformal_mass_a, conformal_mass_b)'. Returns the
        tables of 'ControlParams.h5'."""
        if measured_masses is None:
            measured_masses = lambda mass_a, mass_b: (
                mass(mass_a),
                mass(mass_b),
            )
        id_dir = self.test_dir / name / "ID"
        run_dir = id_dir / "0000_InitialData"
        write_seeding_segment(
            run_dir,
            {
                "conformal_mass_a": initial_conformal_mass,
                "conformal_mass_b": initial_conformal_mass,
            },
            GOOD_SOLVE_OUTPUT,
        )
        metadata, body = seeding_input_file()
        for side in ["ObjectLeft", "ObjectRight"]:
            body["Background"]["Binary"][side]["KerrSchild"][
                "Mass"
            ] = initial_conformal_mass
        with open(run_dir / "InitialData.yaml", "w") as open_input_file:
            yaml.safe_dump_all([metadata, body], open_input_file)
        mock_control_generate_id.calls = []
        mock_control_generate_id.seeded_failures = []
        mock_control_generate_id.unconverged_background = []

        def measured_params(run_dir, control_params):
            free_data = json.loads(
                (Path(run_dir) / "free_data.json").read_text()
            )
            mass_a, mass_b = measured_masses(
                free_data["conformal_mass_a"], free_data["conformal_mass_b"]
            )
            return {
                "MassA": mass_a,
                "MassB": mass_b,
                "DimensionlessSpinA": np.zeros(3),
                "DimensionlessSpinB": np.zeros(3),
            }

        try:
            with (
                patch(
                    "spectre.Pipelines.Bbh.ControlId.generate_id",
                    side_effect=mock_control_generate_id,
                ),
                patch(
                    "spectre.Pipelines.Bbh.ControlId._measured_params",
                    side_effect=measured_params,
                ),
            ):
                self.last_run_dir = control_id(
                    id_input_file_path=run_dir / "InitialData.yaml",
                    control_params=SEEDING_CONTROL_PARAMS,
                    residual_tolerance=1.0e-8,
                    **kwargs,
                )
        finally:
            self.tables = {
                table: _read_dat(id_dir / "ControlParams.h5", table)
                for table in [
                    "ControlScales",
                    "FreeData",
                    "Residuals",
                    "Jacobian",
                    "StepSizeConstraints",
                    "Globalization",
                ]
            }
        return self.tables

    def test_no_globalization_by_default(self):
        tables = self.run_control("none")
        self.assertEqual(tables["Globalization"].size, 0)
        self.assertEqual(Path(self.last_run_dir).name, "0013_InitialData")

    def test_globalizations_stop_the_jumps(self):
        # Without a globalization, the control residual jumps (as in the
        # survey's loops at spins near 1)
        none = self.run_control("none")
        self.assertGreater(len(jumps(none["Residuals"])), 0)
        self.assertLess(np.max(np.abs(none["Residuals"][-1])), 1.0e-8)
        for globalization in ["line_search", "trust_region"]:
            tables = self.run_control(
                globalization, globalization=globalization
            )
            residuals, rows = tables["Residuals"], tables["Globalization"]
            num_solves = len(residuals) - 1
            # Every solve after the first has a row, in order
            npt.assert_array_equal(
                column(rows, "Solve"), range(1, len(rows) + 1)
            )
            self.assertEqual(len(rows), num_solves)
            self.assertTrue(
                np.all(
                    column(rows, "Globalization")
                    == GLOBALIZATIONS.index(globalization)
                )
            )
            # The loop converges in at most as many solves as without
            self.assertLess(np.max(np.abs(residuals[-1])), 1.0e-8)
            self.assertLessEqual(num_solves, len(none["Residuals"]) - 1)
            self.assertEqual(
                Path(self.last_run_dir).name, f"{num_solves:04d}_InitialData"
            )
            # The accepted solves decrease f, and their control residuals
            # never jump
            accepted = column(rows, "Accepted") == 1.0
            npt.assert_array_less(
                column(rows, "FAfter")[accepted],
                column(rows, "FBefore")[accepted],
            )
            accepted_residuals = residuals[
                np.concatenate([[0], column(rows, "Solve")[accepted]]).astype(
                    int
                )
            ]
            self.assertEqual(jumps(accepted_residuals), [])
            # The tables have one row per solve. A rejected solve leaves the
            # free data, and the next one steps from it again.
            self.assertEqual(len(tables["FreeData"]), num_solves + 1)
            self.assertEqual(len(tables["StepSizeConstraints"]), num_solves)
            self.assertEqual(len(tables["Jacobian"]), num_solves)
            iterations = column(rows, "Iteration")
            for row in np.flatnonzero(~accepted):
                solve = int(column(rows, "Solve")[row])
                npt.assert_array_equal(
                    tables["Jacobian"][solve], tables["Jacobian"][solve - 1]
                )
                self.assertEqual(iterations[row + 1], iterations[row])
                self.assertEqual(
                    column(rows, "Trial")[row + 1],
                    column(rows, "Trial")[row] + 1,
                )
            if globalization == "line_search":
                # Each iteration's first trial is the whole step, and the
                # backtracks are shorter
                rejected = np.flatnonzero(~accepted)
                self.assertGreater(len(rejected), 0)
                alphas = column(rows, "LineSearchAlpha")
                npt.assert_array_equal(alphas[column(rows, "Trial") == 0], 1.0)
                npt.assert_array_less(alphas[rejected + 1], alphas[rejected])
            else:
                # The trust region never lets the step grow past it
                npt.assert_array_less(
                    column(rows, "StepLength"),
                    column(rows, "TrustRadius") * (1.0 + 1.0e-12),
                )

    def test_failure_resets_jacobian_then_stops(self):
        # The measured masses don't respond to the free data, so no step can
        # decrease f: each globalization rejects its trials, resets the
        # Jacobian, rejects its trials again and stops
        constant_mass = lambda conformal_mass: 0.4
        for globalization in ["line_search", "trust_region"]:
            with self.assertRaisesRegex(
                RuntimeError, "found no step that decreases"
            ):
                self.run_control(
                    globalization,
                    mass=constant_mass,
                    initial_conformal_mass=0.3,
                    globalization=globalization,
                )
            rows = self.tables["Globalization"]
            self.assertEqual(len(rows), 2 * GLOBALIZATION_MAX_TRIALS)
            npt.assert_array_equal(column(rows, "Accepted"), 0.0)
            npt.assert_array_equal(
                column(rows, "JacobianReset"),
                [0.0] * GLOBALIZATION_MAX_TRIALS
                + [1.0]
                + [0.0] * (GLOBALIZATION_MAX_TRIALS - 1),
            )
            # The reset Jacobian is the initial one: this map has no secant
            # information to add
            jacobians = self.tables["Jacobian"]
            npt.assert_array_equal(
                jacobians[GLOBALIZATION_MAX_TRIALS], jacobians[0]
            )
            # The free data never moved
            npt.assert_array_equal(
                self.tables["Residuals"],
                np.repeat(self.tables["Residuals"][:1], len(rows) + 1, axis=0),
            )

    def test_stopped_loop_runs_final_convergence_test(self):
        # A loop that a globalization stops runs the convergence test after
        # the control loop at its last accepted solve, then stops
        constant_mass = lambda conformal_mass: 0.4
        tests = []

        def convergence_test(**kwargs):
            tests.append(kwargs)
            return kwargs["polynomial_order"]

        with patch(
            "spectre.Pipelines.Bbh.ControlId._convergence_test",
            side_effect=convergence_test,
        ):
            with self.assertRaisesRegex(
                RuntimeError, "found no step that decreases"
            ):
                self.run_control(
                    "stopped",
                    mass=constant_mass,
                    initial_conformal_mass=0.3,
                    globalization="trust_region",
                    run_convergence_tests=True,
                )
        self.assertEqual([test["mode"] for test in tests], ["initial", "final"])
        # No step was accepted, so the test runs at the first solve
        self.assertEqual(Path(tests[1]["run_dir"]).name, "0000_InitialData")
        self.assertEqual(tests[1]["free_data"]["conformal_mass_a"], 0.3)
        self.assertEqual(tests[1]["free_data"]["conformal_mass_b"], 0.3)

    def test_finite_difference_fallback(self):
        # The initial Jacobian (the identity in the masses) has the wrong sign
        # in the row of MassA, so the Newton steps go uphill. The line search
        # with the reset to the initial Jacobian stops; with finite-difference
        # columns it converges.
        jacobian = np.array([[-1.0, -0.5], [-0.8, 1.0]])
        offset = np.array([1.2, 0.3])
        linear_masses = lambda mass_a, mass_b: tuple(
            offset + jacobian @ np.array([mass_a, mass_b])
        )
        kwargs = dict(
            measured_masses=linear_masses,
            initial_conformal_mass=0.3,
            globalization="line_search",
        )
        with self.assertRaisesRegex(
            RuntimeError, "found no step that decreases"
        ):
            self.run_control("reset", **kwargs)
        reset_solves = len(self.tables["Residuals"]) - 1
        tables = self.run_control(
            "finite_difference",
            globalization_fallback="finite_difference",
            **kwargs,
        )
        self.assertLess(np.max(np.abs(tables["Residuals"][-1])), 1.0e-8)
        self.assertLessEqual(len(tables["Residuals"]) - 1, reset_solves)
        rows = tables["Globalization"]
        kind = column(rows, "Kind")
        # The finite-difference solves follow the rejected solves of the
        # first direction, one per mass, and are not steps
        probe_rows = np.flatnonzero(kind == 1.0)
        npt.assert_array_equal(probe_rows, [4, 5])
        npt.assert_array_equal(column(rows, "Accepted")[:6], 0.0)
        npt.assert_allclose(
            column(rows, "StepLength")[probe_rows], FINITE_DIFFERENCE_STEP
        )
        npt.assert_array_equal(column(rows, "Fallback"), 1.0)
        npt.assert_array_equal(column(rows, "JacobianReset")[6], 1.0)
        # The new columns are the true Jacobian of the scaled masses (the
        # scales of both are the target mass), and the next step converges
        num_free_data = tables["FreeData"].shape[1]
        refreshed_jacobian = tables["Jacobian"][6].reshape(
            num_free_data, num_free_data
        )
        npt.assert_allclose(refreshed_jacobian[:2, :2], jacobian, atol=1.0e-9)
        self.assertEqual(column(rows, "Accepted")[6], 1.0)
        # With a trust region too
        tables = self.run_control(
            "trust_region",
            globalization_fallback="finite_difference",
            **{**kwargs, "globalization": "trust_region"},
        )
        self.assertLess(np.max(np.abs(tables["Residuals"][-1])), 1.0e-8)
        self.assertGreater(
            np.sum(column(tables["Globalization"], "Kind") == 1.0), 0
        )

    def test_uphill_detection(self):
        # The Newton steps of the wrong-sign Jacobian go uphill, and f rises
        # in proportion to the step: the fallback starts after two rejected
        # solves instead of four
        linear_masses = lambda mass_a, mass_b: (1.0 - mass_a, mass_b + 0.1)
        for globalization in ["line_search", "trust_region"]:
            kwargs = dict(
                measured_masses=linear_masses,
                initial_conformal_mass=0.3,
                globalization=globalization,
                globalization_fallback="finite_difference",
            )
            rows = self.run_control(f"without_{globalization}", **kwargs)[
                "Globalization"
            ]
            npt.assert_array_equal(
                column(rows, "Kind")[: GLOBALIZATION_MAX_TRIALS + 1],
                [0.0] * GLOBALIZATION_MAX_TRIALS + [1.0],
            )
            npt.assert_array_equal(column(rows, "Uphill"), 0.0)
            tables = self.run_control(
                f"with_{globalization}", uphill_detection=True, **kwargs
            )
            rows = tables["Globalization"]
            npt.assert_array_equal(column(rows, "Kind")[:3], [0.0, 0.0, 1.0])
            npt.assert_array_equal(column(rows, "Uphill")[:2], [0.0, 1.0])
            npt.assert_array_equal(column(rows, "UphillDetection"), 1.0)
            self.assertLess(np.max(np.abs(tables["Residuals"][-1])), 1.0e-8)
        # An overshoot is not uphill: on the saturating map f rises more than
        # linearly with the step, and the line search backtracks as without
        # the detection
        without = self.run_control("overshoot", globalization="line_search")
        with_detection = self.run_control(
            "overshoot_detection",
            globalization="line_search",
            uphill_detection=True,
        )
        npt.assert_array_equal(
            without["Residuals"], with_detection["Residuals"]
        )
        npt.assert_array_equal(
            column(with_detection["Globalization"], "Uphill"), 0.0
        )

    def test_near_singular_step(self):
        # Measured masses that depend almost only on their sum: the true
        # Jacobian is nearly singular (condition number 4e3). The initial
        # Jacobian has the wrong sign, so the trust region falls back to the
        # finite-difference columns, i.e., the true Jacobian.
        jacobian = np.array([[-1.0, -1.0], [-1.0, -1.0 + 1.0e-3]])
        offset = np.array([1.2, 0.3])
        linear_masses = lambda mass_a, mass_b: tuple(
            offset + jacobian @ np.array([mass_a, mass_b])
        )
        kwargs = dict(
            measured_masses=linear_masses,
            initial_conformal_mass=0.3,
            globalization="trust_region",
            globalization_fallback="finite_difference",
        )
        # Without the safeguard, the Newton step of the true Jacobian goes
        # along the nearly null direction, so far (scaled length 70) that the
        # spin step-size constraint shortens it to 9 %, and from there the
        # loop finds no step. The table records the condition numbers.
        with self.assertRaisesRegex(RuntimeError, "found no step"):
            self.run_control("none", **kwargs)
        tables = self.tables
        lowest_f_newton = np.min(column(tables["Globalization"], "FAfter"))
        rows = tables["Globalization"]
        npt.assert_array_equal(column(rows, "NearSingularStep"), 0.0)
        npt.assert_array_equal(column(rows, "Regularized"), 0.0)
        npt.assert_array_equal(
            column(rows, "ConditionThreshold"), DEFAULT_CONDITION_THRESHOLD
        )
        trials = column(rows, "Kind") == 0.0
        npt.assert_allclose(
            column(rows, "Condition")[trials],
            [
                np.linalg.cond(tables["Jacobian"][int(solve) - 1].reshape(8, 8))
                for solve in column(rows, "Solve")[trials]
            ],
            rtol=1e-10,
        )
        self.assertTrue(np.all(np.isnan(column(rows, "Condition")[~trials])))
        # The explicit default changes nothing
        with self.assertRaisesRegex(RuntimeError, "found no step"):
            self.run_control(
                "explicit_none", near_singular_step="none", **kwargs
            )
        default = self.tables
        for name, table in tables.items():
            npt.assert_array_equal(default[name], table, name)
        # The truncated SVD leaves the nearly null direction out of the step
        # after the finite-difference columns: f drops by the residual's part
        # along the other direction, lower than the Newton step got, and the
        # part along the nearly null one stays, so the loop stops
        with self.assertRaisesRegex(RuntimeError, "found no step"):
            self.run_control(
                "truncated_svd",
                near_singular_step="truncated_svd",
                condition_threshold=100.0,
                **kwargs,
            )
        tables = self.tables
        rows = tables["Globalization"]
        npt.assert_array_equal(
            column(rows, "NearSingularStep"),
            NEAR_SINGULAR_STEPS.index("truncated_svd"),
        )
        npt.assert_array_equal(column(rows, "ConditionThreshold"), 100.0)
        regularized = np.flatnonzero(column(rows, "Regularized") == 1.0)
        self.assertGreater(len(regularized), 0)
        first = regularized[0]
        solve = int(column(rows, "Solve")[first])
        self.assertGreater(column(rows, "Condition")[first], 100.0)
        self.assertEqual(column(rows, "Rank")[first], 7.0)
        self.assertEqual(column(rows, "Accepted")[first], 1.0)
        self.assertLess(column(rows, "FAfter")[first], 0.7 * lowest_f_newton)
        J = tables["Jacobian"][solve - 1].reshape(8, 8)
        npt.assert_allclose(J[:2, :2], jacobian, atol=1.0e-9)
        u_scale = tables["ControlScales"][0][:8]
        scaled_step = (
            tables["FreeData"][solve] - tables["FreeData"][solve - 1]
        ) / u_scale
        # Only the steepest-descent part of the dogleg step has a component
        # along the nearly null direction, of order sigma_min / sigma_max
        null_direction = np.linalg.svd(J)[2][-1]
        self.assertLess(
            abs(scaled_step @ null_direction),
            1.0e-3 * np.linalg.norm(scaled_step),
        )
        # Levenberg-Marquardt damps that direction instead: f falls slowly.
        # It records its mu.
        self.run_control(
            "levenberg_marquardt",
            near_singular_step="levenberg_marquardt",
            condition_threshold=100.0,
            max_iterations=10,
            **kwargs,
        )
        rows = self.tables["Globalization"]
        npt.assert_array_less(
            np.diff(np.max(np.abs(self.tables["Residuals"][6:]), axis=1)), 0.0
        )
        regularized = np.flatnonzero(column(rows, "Regularized") == 1.0)
        self.assertGreater(len(regularized), 0)
        for row in regularized:
            J = self.tables["Jacobian"][
                int(column(rows, "Solve")[row]) - 1
            ].reshape(8, 8)
            singular_values = np.linalg.svd(J, compute_uv=False)
            self.assertAlmostEqual(
                column(rows, "Mu")[row],
                max(
                    np.sqrt(8.0 * np.finfo(float).eps)
                    * np.linalg.norm(J.T @ J, 1),
                    (singular_values[0] / 100.0) ** 2,
                ),
                delta=1e-14,
            )
        # The safeguard needs a globalization
        # (checked before the control loop reads anything)
        with self.assertRaisesRegex(AssertionError, "needs a 'globalization'"):
            control_id(
                id_input_file_path=self.test_dir / "unused.yaml",
                control_params=SEEDING_CONTROL_PARAMS,
                near_singular_step="truncated_svd",
            )

    def test_max_step(self):
        # The line search shortens the Newton step to the 'max_step'
        max_step = 0.05
        tables = self.run_control(
            "line_search",
            globalization="line_search",
            max_step=max_step,
            max_iterations=4,
        )
        rows = tables["Globalization"]
        npt.assert_array_less(
            column(rows, "StepLength"), max_step * (1.0 + 1.0e-12)
        )
        npt.assert_array_equal(column(rows, "MaxStep"), max_step)


class TestEllipticSolveConvergence(unittest.TestCase):
    def setUp(self):
        self.run_dir = Path(
            unit_test_build_path(),
            "support/Pipelines/Bbh/EllipticSolveConvergence",
        )
        shutil.rmtree(self.run_dir, ignore_errors=True)
        self.run_dir.mkdir(parents=True)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_converged(self):
        write_newton_residuals(self.run_dir, CONVERGED_ROWS)
        convergence = elliptic_solve_convergence(
            newton_input_file(), self.run_dir
        )
        self.assertTrue(convergence.converged)
        self.assertEqual(convergence.iterations, 5)
        self.assertEqual(convergence.residual, 2.67020329e-11)
        self.assertEqual(convergence.initial_residual, 7.74728203)
        self.assertIn("converged", str(convergence))
        # The same residuals miss a smaller tolerance
        self.assertFalse(
            elliptic_solve_convergence(
                newton_input_file(absolute_residual=1.0e-12), self.run_dir
            ).converged
        )

    def test_max_iterations_is_not_convergence(self):
        write_newton_residuals(self.run_dir, CAPPED_ROWS)
        convergence = elliptic_solve_convergence(
            newton_input_file(), self.run_dir
        )
        self.assertFalse(convergence.converged)
        self.assertEqual(convergence.iterations, 10)
        # The accepted step of the last iteration, not its overshooting trial
        self.assertEqual(convergence.residual, 2.02844549)
        self.assertIn("did not converge", str(convergence))
        self.assertIn("2.028445e+00", str(convergence))
        self.assertIn("MaxIterations = 10", str(convergence))
        self.assertIn("stopped at MaxIterations", str(convergence))

    def test_stopped_early(self):
        # E.g., a globalization failure in iteration 3
        write_newton_residuals(
            self.run_dir, [(0, 0, 10.0), (1, 0, 5.0), (2, 0, 4.0), (3, 0, 4.5)]
        )
        convergence = elliptic_solve_convergence(
            newton_input_file(), self.run_dir
        )
        self.assertFalse(convergence.converged)
        self.assertEqual(convergence.iterations, 3)
        self.assertIn("before MaxIterations", str(convergence))

    def test_nan(self):
        write_newton_residuals(
            self.run_dir, [(0, 0, 10.0), (1, 0, float("nan"))]
        )
        self.assertFalse(
            elliptic_solve_convergence(
                newton_input_file(), self.run_dir
            ).converged
        )

    def test_relative_residual(self):
        write_newton_residuals(self.run_dir, [(0, 0, 10.0), (1, 0, 1.0e-3)])
        input_file = newton_input_file()
        criteria = input_file["NonlinearSolver"]["NewtonRaphson"][
            "ConvergenceCriteria"
        ]
        self.assertFalse(
            elliptic_solve_convergence(input_file, self.run_dir).converged
        )
        criteria["RelativeResidual"] = 1.0e-4
        self.assertTrue(
            elliptic_solve_convergence(input_file, self.run_dir).converged
        )

    def test_last_solve(self):
        # Several solves in one run (e.g., AMR iterations): the last one counts
        write_newton_residuals(self.run_dir, CAPPED_ROWS + CONVERGED_ROWS)
        convergence = elliptic_solve_convergence(
            newton_input_file(), self.run_dir
        )
        self.assertTrue(convergence.converged)
        self.assertEqual(convergence.initial_residual, 7.74728203)
        shutil.rmtree(self.run_dir)
        self.run_dir.mkdir()
        write_newton_residuals(self.run_dir, CONVERGED_ROWS + CAPPED_ROWS)
        self.assertFalse(
            elliptic_solve_convergence(
                newton_input_file(), self.run_dir
            ).converged
        )

    def test_missing_residuals(self):
        with self.assertRaisesRegex(
            UnconvergedSolveError, "did not run or did not finish"
        ):
            elliptic_solve_convergence(newton_input_file(), self.run_dir)


# A control loop of the masses, spins and center of mass of an unequal-mass
# binary with mock solves, to restart from its iterations. The mock solves
# write the input file of every run like 'generate_id' renders it, and the
# measured parameters are coupled nonlinear functions of all free data.
RESTART_CONTROL_PARAMS = [
    "MassA",
    "MassB",
    "DimensionlessSpinA",
    "DimensionlessSpinB",
    "CenterOfMass",
]
RESTART_TARGET_PARAMS = {
    "MassA": 0.5,
    "MassB": 0.4,
    "MassRatio": 1.25,
    "DimensionlessSpinA": [0.0, 0.0, 0.2],
    "DimensionlessSpinB": [0.0, 0.0, 0.0],
    "CenterOfMass": [0.0, 0.0, 0.0],
}
RESTART_SEPARATION = 10.0
# Options of the control loop, as the first run's 'Next' section passes them on
RESTART_NEXT_WITH = {
    "control": True,
    "control_params": RESTART_CONTROL_PARAMS,
    "control_residual_tolerance": 1.0e-8,
    "control_polynomial_order": 6,
    "control_refinement_level": 1,
    "negative_expansion_bc": True,
    "numeric_initial_guess": True,
    "numeric_initial_guess_max_control_residual": 1.0e-3,
    "evolve": False,
    "pipeline_dir": "None",
}


class SimulatedCrash(BaseException):
    """A job that dies (e.g., a node failure or a time limit)"""


def write_restart_run(
    run_dir, free_data, polynomial_order, next_with, complete=True
):
    """A run as 'generate_id' renders it. The mock solve 'completes' if the
    run has its free data in 'free_data.json'."""
    params = id_parameters(
        **free_data,
        separation=RESTART_SEPARATION,
        refinement_level=1,
        polynomial_order=polynomial_order,
        negative_expansion_bc=True,
        target_params=RESTART_TARGET_PARAMS,
    )
    kerr = lambda mass: {"KerrSchild": {"Mass": mass}}
    horizon = lambda side: {
        "Interior": {
            "ExciseWithBoundaryCondition": {
                "ApparentHorizon": {
                    "Rotation": [
                        params[f"HorizonRotation{side}_{xyz}"] for xyz in "xyz"
                    ]
                }
            }
        }
    }
    metadata = {
        "TargetParams": RESTART_TARGET_PARAMS,
        "Next": {
            "With": {**next_with, "control_polynomial_order": polynomial_order}
        },
    }
    body = {
        "Background": {
            "Binary": {
                "XCoords": [params["XLeft"], params["XRight"]],
                "CenterOfMassOffset": [
                    params["CenterOfMassOffset_y"],
                    params["CenterOfMassOffset_z"],
                ],
                "ObjectLeft": kerr(params["ConformalMassLeft"]),
                "ObjectRight": kerr(params["ConformalMassRight"]),
                "AngularVelocity": params["OrbitalAngularVelocity"],
                "Expansion": params["RadialExpansionVelocity"],
                "LinearVelocity": [
                    params[f"LinearVelocity_{xyz}"] for xyz in "xyz"
                ],
            }
        },
        "DomainCreator": {
            "BinaryCompactObject": {
                "ObjectA": horizon("Right"),
                "ObjectB": horizon("Left"),
            }
        },
    }
    run_dir.mkdir(parents=True)
    with open(run_dir / "InitialData.yaml", "w") as open_input_file:
        yaml.safe_dump_all([metadata, body], open_input_file)
    if complete:
        to_json = lambda value: np.asarray(value, dtype=float).tolist()
        (run_dir / "free_data.json").write_text(
            json.dumps(
                {key: to_json(value) for key, value in free_data.items()}
            )
        )


def mock_restart_generate_id(
    target_params, polynomial_order, initial_guess_file_glob, **kwargs
):
    """Runs the next segment of the control loop, or a solve of a
    convergence test. Raises a 'SimulatedCrash' in the solve whose index (over
    all solves of this mock) is 'mock_restart_generate_id.crash_at'."""
    free_data = {
        key: kwargs[key]
        for key in [
            "conformal_mass_a",
            "conformal_mass_b",
            "horizon_rotation_a",
            "horizon_rotation_b",
            "center_of_mass_offset",
            "linear_velocity",
            "radial_expansion_velocity",
            "orbital_angular_velocity",
        ]
    }
    if kwargs.get("segments_dir") is not None:
        segments_dir = Path(kwargs["segments_dir"])
        num_segments = len(list(segments_dir.glob("[0-9]*_InitialData")))
        run_dir = segments_dir / f"{num_segments:04d}_InitialData"
    else:
        run_dir = Path(kwargs["run_dir"])
    crash = len(mock_restart_generate_id.calls) == (
        mock_restart_generate_id.crash_at
    )
    # A dry run renders the input file, but doesn't solve
    dry_run = kwargs.get("submit") is False
    mock_restart_generate_id.templates.append(
        kwargs.get("id_input_file_template")
    )
    mock_restart_generate_id.calls.append(
        dict(
            name=run_dir.name,
            glob=initial_guess_file_glob,
            # The run that the solve starts from
            seed=(
                None
                if initial_guess_file_glob is None
                else Path(initial_guess_file_glob).parent.name
            ),
            P=polynomial_order,
            globalization=kwargs.get("control_globalization", "none"),
            **{
                key: np.asarray(value, dtype=float).tolist()
                for key, value in free_data.items()
            },
        )
    )
    write_restart_run(
        run_dir,
        free_data,
        polynomial_order,
        {**RESTART_NEXT_WITH, "control": False},
        complete=not (crash or dry_run or x_right(free_data) <= 0.0),
    )
    if crash:
        raise SimulatedCrash(f"Crashed in '{run_dir}'.")
    # What the domain creator says when the input file is checked
    if x_right(free_data) <= 0.0:
        raise RuntimeError(
            "InvalidInputFileError: The x-coordinate of ObjectA's center is"
            " expected to be positive."
        )


def x_right(free_data):
    return id_parameters(
        **free_data,
        separation=RESTART_SEPARATION,
        refinement_level=1,
        polynomial_order=6,
        negative_expansion_bc=True,
        target_params=RESTART_TARGET_PARAMS,
    )["XRight"]


def mock_restart_measured_params(run_dir, control_params):
    free_data = json.loads((Path(run_dir) / "free_data.json").read_text())
    mass_a, mass_b = (
        free_data["conformal_mass_a"],
        free_data["conformal_mass_b"],
    )
    rotation_a = np.array(free_data["horizon_rotation_a"])
    rotation_b = np.array(free_data["horizon_rotation_b"])
    offset = np.array(free_data["center_of_mass_offset"])
    # A truncation error that falls with the polynomial order, so that the
    # convergence test before the control loop needs several solves
    truncation_error = 0.0
    if mock_restart_measured_params.order_error != 0.0:
        with open(Path(run_dir) / "InitialData.yaml", "r") as open_input_file:
            metadata = next(yaml.safe_load_all(open_input_file))
        polynomial_order = metadata["Next"]["With"]["control_polynomial_order"]
        truncation_error = (
            mock_restart_measured_params.order_error * 10.0**-polynomial_order
        )
    mass = mock_restart_measured_params.mass
    return {
        "MassA": mass(mass_a) + 0.01 * mass_b + truncation_error,
        "MassB": mass(mass_b) + 0.01 * mass_a,
        "DimensionlessSpinA": (
            -4.0 * mass_a * rotation_a + 0.01 * mass_b * rotation_b
        ),
        "DimensionlessSpinB": -4.0 * mass_b * rotation_b,
        "CenterOfMass": (
            offset
            + np.array([0.3 * (mass_a - mass_b), 0.0, 0.0])
            + 0.1 * offset**2
            # Moves the center of mass that the control aims for
            + np.array([mock_restart_measured_params.com_shift, 0.0, 0.0])
        ),
    }


def read_control_params(id_dir):
    with h5py.File(Path(id_dir) / "ControlParams.h5", "r") as open_h5_file:
        return {
            name: open_h5_file[name][()]
            for name in open_h5_file
            if name.endswith(".dat")
        }


def free_data_of(call):
    return {
        key: call[key]
        for key in [
            "conformal_mass_a",
            "conformal_mass_b",
            "horizon_rotation_a",
            "horizon_rotation_b",
            "center_of_mass_offset",
            "linear_velocity",
            "radial_expansion_velocity",
            "orbital_angular_velocity",
        ]
    }


def without_globs(calls):
    """The solves without the paths of their initial guesses, which name the
    directory of the control loop"""
    return [{**call, "glob": None} for call in calls]


def list_files(directory):
    return sorted(
        (str(path.relative_to(directory)), path.stat().st_size)
        for path in Path(directory).rglob("*")
        if path.is_file()
    )


class TestRestartControlId(unittest.TestCase):
    def setUp(self):
        mock_restart_measured_params.com_shift = 0.0
        mock_restart_measured_params.order_error = 0.0
        mock_restart_measured_params.mass = mock_mass
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/ControlIdRestart"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def mocks(self, crash_at=None):
        mock_restart_generate_id.calls = []
        mock_restart_generate_id.templates = []
        mock_restart_generate_id.crash_at = crash_at
        return (
            patch(
                "spectre.Pipelines.Bbh.ControlId.generate_id",
                side_effect=mock_restart_generate_id,
            ),
            patch(
                "spectre.Pipelines.Bbh.ControlId._measured_params",
                side_effect=mock_restart_measured_params,
            ),
        )

    def run_control(
        self,
        name,
        crash_at=None,
        initial_conformal_mass=0.35,
        allow_failure=False,
        **next_with,
    ):
        """Runs a control loop like 'postprocess_id' does after the first
        solve. Returns its directory, its solves, and its last run (or
        'None' if it crashed, or failed and 'allow_failure')."""
        id_dir = self.test_dir / name
        next_with = {**RESTART_NEXT_WITH, **next_with}
        write_restart_run(
            id_dir / "0000_InitialData",
            dict(
                conformal_mass_a=initial_conformal_mass,
                conformal_mass_b=initial_conformal_mass,
                horizon_rotation_a=[0.0, 0.0, -0.1],
                horizon_rotation_b=[0.0, 0.0, 0.0],
                center_of_mass_offset=[0.0, 0.0, 0.0],
                linear_velocity=[0.0, 0.0, 0.0],
                radial_expansion_velocity=0.0,
                orbital_angular_velocity=0.01,
            ),
            next_with["control_polynomial_order"],
            next_with,
        )
        generate_id, measured_params = self.mocks(crash_at)
        last_run_dir = None
        with generate_id, measured_params:
            try:
                last_run_dir = control_id(
                    id_input_file_path=id_dir
                    / "0000_InitialData"
                    / "InitialData.yaml",
                    control_params=RESTART_CONTROL_PARAMS,
                    residual_tolerance=next_with["control_residual_tolerance"],
                    polynomial_order=next_with["control_polynomial_order"],
                    numeric_initial_guess=True,
                    numeric_initial_guess_max_control_residual=next_with[
                        "numeric_initial_guess_max_control_residual"
                    ],
                    run_convergence_tests=next_with.get(
                        "run_convergence_tests", False
                    ),
                    globalization=next_with.get(
                        "control_globalization", "none"
                    ),
                    globalization_fallback=next_with.get(
                        "control_globalization_fallback", "reset"
                    ),
                    uphill_detection=next_with.get(
                        "control_uphill_detection", False
                    ),
                    near_singular_step=next_with.get(
                        "control_near_singular_step", "none"
                    ),
                    condition_threshold=next_with.get(
                        "control_condition_threshold"
                    ),
                    max_iterations=next_with.get(
                        "control_max_iterations", DEFAULT_MAX_ITERATIONS
                    ),
                )
            except SimulatedCrash:
                pass
            except RuntimeError:
                if not allow_failure:
                    raise
        return id_dir, mock_restart_generate_id.calls, last_run_dir

    def restart(self, restart_from, name, **kwargs):
        generate_id, measured_params = self.mocks()
        with generate_id, measured_params:
            last_run_dir = restart_control_id(
                restart_from, id_dir=self.test_dir / name, **kwargs
            )
        return (
            self.test_dir / name,
            mock_restart_generate_id.calls,
            last_run_dir,
        )

    def assertSameControlParams(self, id_dir, other_id_dir):
        control_params = read_control_params(id_dir)
        other_control_params = read_control_params(other_id_dir)
        self.assertEqual(sorted(control_params), sorted(other_control_params))
        for name, data in control_params.items():
            npt.assert_array_equal(data, other_control_params[name], name)

    def test_restart_from_every_iteration(self):
        id_dir, calls, last_run_dir = self.run_control("Uninterrupted")
        num_iterations = len(calls) + 1
        # The loop is long enough to cover the delayed control of the center of
        # mass, and solves from the background and from previous solutions
        self.assertGreater(num_iterations, 6)
        self.assertTrue(any(call["glob"] is None for call in calls))
        self.assertTrue(any(call["glob"] is not None for call in calls))
        free_data = read_control_params(id_dir)["FreeData.dat"]
        self.assertEqual(len(free_data), num_iterations)
        self.assertNotEqual(free_data[-1, -3], 0.0)  # Center of mass offset
        files = list_files(id_dir)
        for restart_iteration in range(num_iterations):
            restart_dir, restart_calls, restart_last_run_dir = self.restart(
                id_dir,
                f"Restart{restart_iteration}",
                restart_iteration=restart_iteration,
            )
            # The restart solves the iterations after the restart iteration
            # with the same free data and initial guesses, i.e., the same
            # input files
            self.assertEqual(
                without_globs(restart_calls),
                without_globs(calls[restart_iteration:]),
            )
            # The first solve starts from the same run of the previous loop
            if restart_calls:
                self.assertEqual(
                    restart_calls[0]["glob"], calls[restart_iteration]["glob"]
                )
            self.assertSameControlParams(restart_dir, id_dir)
            self.assertEqual(
                Path(restart_last_run_dir).name, Path(last_run_dir).name
            )
            for i in range(restart_iteration + 1):
                self.assertTrue(
                    (restart_dir / f"{i:04d}_InitialData").is_symlink()
                )
        # The previous loop is unchanged
        self.assertEqual(list_files(id_dir), files)

    def test_restart_after_crash(self):
        reference_dir, calls, _ = self.run_control("Reference")
        for crash_at in [0, 3, len(calls) - 1]:
            crashed_dir, crashed_calls, last_run_dir = self.run_control(
                f"Crashed{crash_at}", crash_at=crash_at
            )
            self.assertIsNone(last_run_dir)
            self.assertEqual(len(crashed_calls), crash_at + 1)
            # The restart continues from the last completed iteration, which
            # ran the solve before the crash
            restart_dir, restart_calls, _ = self.restart(
                crashed_dir, f"RestartAfterCrash{crash_at}"
            )
            self.assertEqual(
                without_globs(crashed_calls[:-1] + restart_calls),
                without_globs(calls),
            )
            self.assertEqual(
                len(crashed_calls) - 1 + len(restart_calls), len(calls)
            )
            # The first solve after the restart starts from the same run as
            # the one that crashed
            self.assertEqual(
                restart_calls[0]["glob"], crashed_calls[-1]["glob"]
            )
            self.assertSameControlParams(restart_dir, reference_dir)
            # The crashed solve is not in the new directory
            self.assertFalse(
                (restart_dir / crashed_calls[-1]["name"]).is_symlink()
            )

    def test_restart_after_crash_in_final_convergence_test(self):
        reference_dir, calls, _ = self.run_control(
            "Reference", run_convergence_tests=True
        )
        # The control loop runs at the polynomial order that the test before
        # the control loop selected, which is not the default
        control_orders = {
            call["P"] for call in calls if call["name"].endswith("_InitialData")
        }
        self.assertEqual(control_orders, {5})
        final_test_calls = [
            call for call in calls if call["name"].startswith("P")
        ][1:]
        self.assertEqual([call["P"] for call in final_test_calls], [4, 6, 7, 8])
        crash_at = calls.index(final_test_calls[2])  # At P=7
        crashed_dir, crashed_calls, _ = self.run_control(
            "Crashed", crash_at=crash_at, run_convergence_tests=True
        )
        restart_dir, restart_calls, last_run_dir = self.restart(
            crashed_dir, "Restart"
        )
        # The restart skips the control loop and the solves of the test that
        # finished
        self.assertEqual(restart_calls, calls[crash_at:])
        self.assertEqual([call["P"] for call in restart_calls], [7, 8])
        self.assertSameControlParams(restart_dir, reference_dir)

    def test_restart_final_test_only(self):
        reference_dir, calls, _ = self.run_control(
            "Reference", run_convergence_tests=True
        )
        tables = read_control_params(reference_dir)
        restart_iteration = 3
        self.assertGreater(len(tables["FreeData.dat"]), restart_iteration + 1)
        restart_dir, restart_calls, last_run_dir = self.restart(
            reference_dir,
            "FinalTestOnly",
            restart_iteration=restart_iteration,
            final_test_only=True,
        )
        # Only the convergence test after the control loop solves, at the free
        # data of the restart iteration
        self.assertGreater(len(restart_calls), 0)
        self.assertTrue(
            all(call["name"].startswith("P") for call in restart_calls)
        )
        self.assertEqual(
            Path(last_run_dir).name, f"{restart_iteration:04d}_InitialData"
        )
        restart_tables = read_control_params(restart_dir)
        npt.assert_array_equal(
            restart_tables["FreeData.dat"],
            tables["FreeData.dat"][: restart_iteration + 1],
        )
        free_data = restart_tables["FreeData.dat"][restart_iteration]
        for call in restart_calls:
            npt.assert_allclose(
                [call["conformal_mass_a"], call["conformal_mass_b"]],
                free_data[:2],
                rtol=1.0e-14,
            )
        self.assertTrue((restart_dir / "ConvergenceTest" / "Final").is_dir())
        record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
        self.assertTrue(record["final_test_only"])
        # A loop without convergence tests has no final test to run
        no_tests_dir, _, _ = self.run_control("NoTests")
        with self.assertRaisesRegex(ValueError, "without convergence tests"):
            self.restart(
                no_tests_dir,
                "NoTestsFinalTestOnly",
                restart_iteration=restart_iteration,
                final_test_only=True,
            )

    def test_restart_after_crash_in_initial_convergence_test(self):
        mock_restart_measured_params.order_error = 10.0
        reference_dir, calls, reference_last_run_dir = self.run_control(
            "Reference", run_convergence_tests=True
        )
        initial_test_calls = calls[
            : next(
                i
                for i, call in enumerate(calls)
                if call["name"].endswith("_InitialData")
            )
        ]
        self.assertEqual([call["P"] for call in initial_test_calls], [5, 7])
        # Crash in the second solve of the test, after the first finished
        crash_at = 1
        crashed_dir, crashed_calls, _ = self.run_control(
            "Crashed", crash_at=crash_at, run_convergence_tests=True
        )
        self.assertEqual(len(crashed_calls), crash_at + 1)
        # A dry run renders the input file of the solve that crashed, and
        # solves nothing
        check_dir, check_calls, check_run_dir = self.restart(
            crashed_dir, "Check", dry_run=True
        )
        self.assertEqual(check_calls, calls[crash_at : crash_at + 1])
        self.assertEqual(
            Path(check_run_dir),
            check_dir / "ConvergenceTest" / "Initial" / calls[crash_at]["name"],
        )
        # The restart reuses the solve of the test that finished
        restart_dir, restart_calls, last_run_dir = self.restart(
            crashed_dir, "Restart"
        )
        self.assertTrue(
            (
                restart_dir / "ConvergenceTest" / "Initial" / calls[0]["name"]
            ).is_symlink()
        )
        self.assertEqual(
            without_globs(restart_calls), without_globs(calls[crash_at:])
        )
        self.assertSameControlParams(restart_dir, reference_dir)
        self.assertEqual(
            Path(last_run_dir).name, Path(reference_last_run_dir).name
        )

    def test_restart_with_constraint_that_did_not_bind(self):
        # The center of mass that the control aims for needs the larger black
        # hole at negative x, so the step that starts the control of the
        # center of mass crosses x_A = 0 and the loop dies, as in the survey
        mock_restart_measured_params.com_shift = 6.0
        id_dir, calls, last_run_dir = self.run_control(
            "Uninterrupted", allow_failure=True
        )
        self.assertIsNone(last_run_dir)
        self.assertLessEqual(x_right(free_data_of(calls[-1])), 0.0)
        steps = read_control_params(id_dir)["StepSizeConstraints.dat"]
        com_alpha, applied_alpha = 1, 3
        # The 'com' constraint (not enforced) would have shortened only the
        # fatal step
        npt.assert_array_equal(steps[:-1, com_alpha], 1.0)
        self.assertLess(steps[-1, com_alpha], 1.0)
        self.assertEqual(steps[-1, applied_alpha], 1.0)
        for dry_run in [True, False]:
            restart_dir, restart_calls, run_dir = self.restart(
                id_dir,
                f"Restart{dry_run}",
                step_size_constraints=["spin", "com"],
                dry_run=dry_run,
                # One more iteration than the previous loop completed
                max_iterations=len(calls),
            )
            # The restart replays the loop and takes the shortened step
            new_steps = read_control_params(restart_dir)[
                "StepSizeConstraints.dat"
            ]
            npt.assert_array_equal(new_steps[:-1], steps[:-1])
            npt.assert_array_equal(new_steps[-1, :3], steps[-1, :3])
            self.assertEqual(new_steps[-1, applied_alpha], steps[-1, com_alpha])
            self.assertGreater(x_right(free_data_of(restart_calls[0])), 0.0)
            self.assertEqual(restart_calls[0]["name"], calls[-1]["name"])
            # The new runs record the changed constraints
            template = mock_restart_generate_id.templates[0].read_text()
            self.assertIn(
                "    control_step_size_constraints: [spin, com]\n", template
            )
            record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
            self.assertEqual(
                record["changed_options"],
                {
                    "control_step_size_constraints": {
                        "old": ["spin"],
                        "new": ["spin", "com"],
                    },
                    "control_max_iterations": {"old": 30, "new": len(calls)},
                },
            )
            if dry_run:
                # The dry run renders the input file and stops
                self.assertEqual(len(restart_calls), 1)
                self.assertEqual(Path(run_dir).name, calls[-1]["name"])
                self.assertFalse((Path(run_dir) / "free_data.json").exists())
            else:
                # The restarted loop solves the shortened step and stops at
                # its maximum number of iterations
                self.assertEqual(len(restart_calls), 1)
                self.assertEqual(Path(run_dir).name, calls[-1]["name"])
                self.assertTrue((Path(run_dir) / "free_data.json").exists())

    def test_restart_refuses_constraint_that_binds(self):
        # From a small initial conformal mass, the first steps change the
        # masses by more than the 'mass' constraint allows (not enforced)
        id_dir, calls, _ = self.run_control(
            "Uninterrupted", initial_conformal_mass=0.15
        )
        steps = read_control_params(id_dir)["StepSizeConstraints.dat"]
        mass_alpha, applied_alpha = 0, 3
        binding_iteration = 1 + int(np.flatnonzero(steps[:, mass_alpha] < 1)[0])
        self.assertLess(binding_iteration, len(calls))
        with self.assertRaisesRegex(
            RuntimeError,
            "change the replayed step of control iteration"
            f" {binding_iteration}:",
        ):
            self.restart(
                id_dir,
                "Refused",
                restart_iteration=binding_iteration,
                step_size_constraints=["spin", "mass"],
                dry_run=True,
            )
        # From the iteration before, the binding step is the first new one
        restart_dir, _, _ = self.restart(
            id_dir,
            "Accepted",
            restart_iteration=binding_iteration - 1,
            step_size_constraints=["spin", "mass"],
            dry_run=True,
        )
        new_steps = read_control_params(restart_dir)["StepSizeConstraints.dat"]
        self.assertEqual(len(new_steps), binding_iteration)
        self.assertEqual(
            new_steps[-1, applied_alpha], steps[binding_iteration - 1, 0]
        )

    def test_restart_options(self):
        id_dir, calls, _ = self.run_control("Uninterrupted")
        # Options that would change the replayed iterations can't change
        with self.assertRaisesRegex(ValueError, "can't change the options"):
            self.restart(id_dir, "Tolerance", residual_tolerance=1.0e-3)
        with self.assertRaisesRegex(ValueError, "maximum number of iterations"):
            self.restart(
                id_dir, "MaxIterations", restart_iteration=3, max_iterations=3
            )
        # Options that don't change anything are recorded as unchanged
        restart_dir, restart_calls, _ = self.restart(
            id_dir, "Same", restart_iteration=2, step_size_constraints=["spin"]
        )
        self.assertEqual(without_globs(restart_calls), without_globs(calls[2:]))
        self.assertEqual(set(mock_restart_generate_id.templates), {None})
        record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
        self.assertEqual(record["changed_options"], {})

    def test_scheduled_restart(self):
        id_dir, calls, last_run_dir = self.run_control("Uninterrupted")
        submit_script_template = self.test_dir / "Template.sh"
        submit_script_template.write_text(
            "#!/bin/bash -\n"
            "#SBATCH -J {{ job_name }}\n"
            "#SBATCH -t {{ time_limit }}\n"
            "#SBATCH -A {{ account }}\n"
            "export RUN_DIR={{ run_dir }}\n"
            "export SPECTRE_CLI={{ spectre_cli }}\n"
            "cd ${RUN_DIR}\n"
            "{% block run_command %}\n"
            "{{ executable }} --input-file {{ input_file }}\n"
            "{% endblock %}\n"
            "${SPECTRE_CLI} run-next {{ input_file }} -i .\n"
        )
        restart_dir = self.test_dir / "Scheduled"
        generate_id, measured_params = self.mocks()
        with generate_id, measured_params:
            submit_script = restart_control_id(
                id_dir,
                restart_dir,
                restart_iteration=2,
                step_size_constraints=["spin", "com"],
                scheduler="sbatch",
                submit_script_template=submit_script_template,
                job_name="restart-test",
                time_limit="1:00:00",
                extra_params=dict(account="uzh66"),
                submit=False,
            )
        # The restart is checked with a dry run, which renders one solve
        self.assertEqual(len(mock_restart_generate_id.calls), 1)
        check = yaml.safe_load(
            (restart_dir / "RestartCheck" / "Restart.yaml").read_text()
        )
        self.assertTrue(check["dry_run"])
        # The submit script runs the restart in place of the executable
        text = submit_script.read_text()
        self.assertIn("#SBATCH -J restart-test\n", text)
        self.assertIn("#SBATCH -A uzh66\n", text)
        self.assertIn(f"export RUN_DIR={restart_dir}\n", text)
        self.assertIn(
            f"${{SPECTRE_CLI}} bbh restart-id {id_dir.resolve()} -o"
            f" {restart_dir} -i 2 --step-size-constraint=spin"
            " --step-size-constraint=com --no-schedule\nexit $?\n",
            text,
        )
        self.assertNotIn("--input-file", text)
        # The job then runs the restart in the directory, which holds no
        # control loop yet. The 'com' constraint doesn't bind in this loop,
        # so the restart reproduces it.
        self.assertFalse((restart_dir / "ControlParams.h5").exists())
        _, restart_calls, restart_last_run_dir = self.restart(
            id_dir,
            "Scheduled",
            restart_iteration=2,
            step_size_constraints=["spin", "com"],
        )
        self.assertEqual(without_globs(restart_calls), without_globs(calls[2:]))
        self.assertSameControlParams(restart_dir, id_dir)
        self.assertEqual(
            Path(restart_last_run_dir).name, Path(last_run_dir).name
        )

    def test_restart_with_globalization(self):
        # A loop without a globalization whose control residual jumps
        mock_restart_measured_params.mass = saturating_mass
        id_dir, calls, _ = self.run_control(
            "Uninterrupted",
            initial_conformal_mass=SATURATING_INITIAL_CONFORMAL_MASS,
        )
        tables = read_control_params(id_dir)
        self.assertNotIn("Globalization.dat", tables)
        first_jump = jumps(tables["Residuals.dat"])[0]
        restart_iteration = first_jump - 1
        for globalization in ["line_search", "trust_region"]:
            restart_dir, restart_calls, last_run_dir = self.restart(
                id_dir,
                f"Restart_{globalization}",
                restart_iteration=restart_iteration,
                globalization=globalization,
            )
            restart_tables = read_control_params(restart_dir)
            # The replayed iterations are the previous loop's
            for name, num_rows in [
                ("Residuals.dat", restart_iteration + 1),
                ("FreeData.dat", restart_iteration + 1),
                ("Jacobian.dat", restart_iteration + 1),
                ("StepSizeConstraints.dat", restart_iteration),
            ]:
                npt.assert_array_equal(
                    restart_tables[name][:num_rows],
                    tables[name][:num_rows],
                    name,
                )
            # The globalization acts on the new solves only, and they record
            # it in their input files
            rows = restart_tables["Globalization.dat"]
            npt.assert_array_equal(
                column(rows, "Solve"),
                range(restart_iteration + 1, restart_iteration + 1 + len(rows)),
            )
            self.assertEqual(
                {call["globalization"] for call in restart_calls},
                {globalization},
            )
            self.assertEqual(set(mock_restart_generate_id.templates), {None})
            record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
            self.assertEqual(
                record["changed_options"],
                {
                    "control_globalization": {
                        "old": "none",
                        "new": globalization,
                    }
                },
            )
            # The first new step comes from the replayed state, with the
            # new method
            self.assertEqual(rows[0, 0], restart_iteration + 1)
            self.assertEqual(
                column(rows, "Globalization")[0],
                GLOBALIZATIONS.index(globalization),
            )
            self.assertGreater(len(restart_calls), 1)

    def test_restart_globalized_loop(self):
        # Loops with a globalization, restarted from accepted and rejected
        # solves, reproduce themselves
        mock_restart_measured_params.mass = saturating_mass
        for globalization in ["line_search", "trust_region"]:
            id_dir, calls, last_run_dir = self.run_control(
                f"Uninterrupted_{globalization}",
                initial_conformal_mass=SATURATING_INITIAL_CONFORMAL_MASS,
                control_globalization=globalization,
            )
            rows = read_control_params(id_dir)["Globalization.dat"]
            self.assertEqual(len(rows), len(calls))
            # Restart from the first solve, from the first rejected one and
            # the one after it, and from two later ones
            rejected_solves = column(rows, "Solve")[
                column(rows, "Accepted") == 0.0
            ].astype(int)
            self.assertGreater(len(rejected_solves), 0)
            first_rejected = int(rejected_solves[0])
            for restart_iteration in sorted(
                {
                    0,
                    first_rejected,
                    first_rejected + 1,
                    len(calls) // 2,
                    len(calls) - 1,
                }
            ):
                restart_dir, restart_calls, restart_last_run_dir = self.restart(
                    id_dir,
                    f"Restart_{globalization}_{restart_iteration}",
                    restart_iteration=restart_iteration,
                )
                self.assertEqual(
                    without_globs(restart_calls),
                    without_globs(calls[restart_iteration:]),
                )
                self.assertSameControlParams(restart_dir, id_dir)
                self.assertEqual(
                    Path(restart_last_run_dir).name, Path(last_run_dir).name
                )
        # The replay runs with the methods the solves ran with, also when the
        # restart switches the globalization off. (Without a globalization,
        # this loop later steps beyond the spin limit and stops.)
        with self.assertRaisesRegex(RuntimeError, "allow no step"):
            self.restart(
                id_dir,
                "RestartWithout",
                restart_iteration=3,
                globalization="none",
            )
        restart_tables = read_control_params(self.test_dir / "RestartWithout")
        npt.assert_array_equal(restart_tables["Globalization.dat"], rows[:3])
        for name in ["Residuals.dat", "FreeData.dat", "Jacobian.dat"]:
            npt.assert_array_equal(
                restart_tables[name][:4], read_control_params(id_dir)[name][:4]
            )

    def test_restart_with_finite_differences(self):
        # Measured masses that fall with the conformal masses: the trust
        # region with the reset to the initial Jacobian (the wrong sign)
        # stops, as some loops on eiger did
        mock_restart_measured_params.mass = lambda mass: 0.9 - mass
        id_dir, calls, last_run_dir = self.run_control(
            "Reset",
            initial_conformal_mass=0.35,
            control_globalization="trust_region",
            allow_failure=True,
        )
        self.assertIsNone(last_run_dir)
        tables = read_control_params(id_dir)
        rows = tables["Globalization.dat"]
        accepted = np.flatnonzero(column(rows, "Accepted") == 1.0)
        last_accepted = int(column(rows, "Solve")[accepted[-1]])
        # Restarted from its last accepted solve with the finite-difference
        # fallback and the uphill detection, it converges
        restart_dir, restart_calls, _ = self.restart(
            id_dir,
            "Restart",
            restart_iteration=last_accepted,
            globalization_fallback="finite_difference",
            uphill_detection=True,
        )
        restart_tables = read_control_params(restart_dir)
        self.assertLess(
            np.max(np.abs(restart_tables["Residuals.dat"][-1])), 1.0e-8
        )
        restart_rows = restart_tables["Globalization.dat"]
        npt.assert_array_equal(
            restart_rows[:last_accepted], rows[:last_accepted]
        )
        npt.assert_array_equal(
            column(restart_rows, "Fallback")[last_accepted:], 1.0
        )
        self.assertGreater(np.sum(column(restart_rows, "Kind") == 1.0), 0)
        record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
        self.assertEqual(
            sorted(record["changed_options"]),
            ["control_globalization_fallback", "control_uphill_detection"],
        )
        # A loop with the fallback from the start, restarted from before,
        # at and after its finite-difference solves, reproduces itself
        id_dir, calls, last_run_dir = self.run_control(
            "FiniteDifference",
            initial_conformal_mass=0.35,
            control_globalization="trust_region",
            control_globalization_fallback="finite_difference",
            control_uphill_detection=True,
        )
        rows = read_control_params(id_dir)["Globalization.dat"]
        probe_solves = column(rows, "Solve")[column(rows, "Kind") == 1.0]
        self.assertGreater(len(probe_solves), 1)
        for restart_iteration in [
            int(probe_solves[0]) - 1,
            int(probe_solves[0]),
            int(probe_solves[-1]),
            int(probe_solves[-1]) + 1,
        ]:
            restart_dir, restart_calls, restart_last_run_dir = self.restart(
                id_dir,
                f"Restart{restart_iteration}",
                restart_iteration=restart_iteration,
            )
            self.assertEqual(
                without_globs(restart_calls),
                without_globs(calls[restart_iteration:]),
            )
            self.assertSameControlParams(restart_dir, id_dir)

    def test_restart_loop_before_finite_differences(self):
        # Loops that ran before the finite-difference fallback have the first
        # columns of 'Globalization.dat' only, and replay with the defaults of
        # the later ones
        mock_restart_measured_params.mass = saturating_mass
        id_dir, calls, _ = self.run_control(
            "Uninterrupted",
            initial_conformal_mass=SATURATING_INITIAL_CONFORMAL_MASS,
            control_globalization="line_search",
        )
        rows = read_control_params(id_dir)["Globalization.dat"]
        with h5py.File(id_dir / "ControlParams.h5", "a") as open_h5_file:
            attrs = dict(open_h5_file["Globalization.dat"].attrs)
            del open_h5_file["Globalization.dat"]
            dataset = open_h5_file.create_dataset(
                "Globalization.dat",
                data=rows[:, :LEGACY_GLOBALIZATION_COLUMNS],
                maxshape=(None, LEGACY_GLOBALIZATION_COLUMNS),
            )
            for key, value in attrs.items():
                dataset.attrs[key] = (
                    value[:LEGACY_GLOBALIZATION_COLUMNS]
                    if key == "Legend"
                    else value
                )
        restart_dir, restart_calls, _ = self.restart(
            id_dir, "Restart", restart_iteration=len(calls) // 2
        )
        self.assertEqual(
            without_globs(restart_calls),
            without_globs(calls[len(calls) // 2 :]),
        )
        npt.assert_array_equal(
            read_control_params(restart_dir)["Globalization.dat"], rows
        )

    def test_restart_with_near_singular_step(self):
        # A loop whose Levenberg-Marquardt safeguard acts at almost every
        # step (a threshold of 2 on the condition number), restarted from
        # before, at and after a safeguarded step, reproduces itself
        next_with = dict(
            control_globalization="trust_region",
            control_near_singular_step="levenberg_marquardt",
            control_condition_threshold=2.0,
            control_max_iterations=8,
        )
        id_dir, calls, _ = self.run_control("Safeguarded", **next_with)
        rows = read_control_params(id_dir)["Globalization.dat"]
        regularized = column(rows, "Solve")[column(rows, "Regularized") == 1.0]
        self.assertGreater(len(regularized), 2)
        for restart_iteration in [
            int(regularized[0]) - 1,
            int(regularized[1]),
            int(regularized[-1]) - 1,
        ]:
            restart_dir, restart_calls, _ = self.restart(
                id_dir,
                f"Restart{restart_iteration}",
                restart_iteration=restart_iteration,
            )
            self.assertEqual(
                without_globs(restart_calls),
                without_globs(calls[restart_iteration:]),
            )
            self.assertSameControlParams(restart_dir, id_dir)
        # A loop without it, restarted with it: the new solves record the
        # option in their input files and their rows
        id_dir, _, _ = self.run_control(
            "Plain", control_globalization="trust_region"
        )
        restart_dir, _, _ = self.restart(
            id_dir,
            "RestartSafeguarded",
            restart_iteration=2,
            near_singular_step="truncated_svd",
            condition_threshold=50.0,
        )
        record = yaml.safe_load((restart_dir / "Restart.yaml").read_text())
        self.assertEqual(
            record["changed_options"],
            {
                "control_near_singular_step": {
                    "old": "none",
                    "new": "truncated_svd",
                },
                "control_condition_threshold": {"old": None, "new": 50.0},
            },
        )
        rows = read_control_params(restart_dir)["Globalization.dat"]
        new_rows = column(rows, "Solve") > 2
        npt.assert_array_equal(
            column(rows, "NearSingularStep")[new_rows],
            NEAR_SINGULAR_STEPS.index("truncated_svd"),
        )
        npt.assert_array_equal(
            column(rows, "ConditionThreshold")[new_rows], 50.0
        )
        npt.assert_array_equal(column(rows, "NearSingularStep")[~new_rows], 0.0)

    def test_replay_mismatch(self):
        id_dir, calls, _ = self.run_control("Uninterrupted")
        # A run whose free data the replay doesn't reproduce, e.g., because
        # the previous loop ran other code
        input_file = id_dir / "0002_InitialData" / "InitialData.yaml"
        metadata, body = yaml.safe_load_all(input_file.read_text())
        body["Background"]["Binary"]["ObjectRight"]["KerrSchild"]["Mass"] *= (
            1.0 + 1.0e-8
        )
        input_file.write_text(yaml.safe_dump_all([metadata, body]))
        with self.assertRaisesRegex(
            RuntimeError, "doesn't reproduce the free data"
        ):
            self.restart(id_dir, "Restart", restart_iteration=3)
        # The restart from an earlier iteration doesn't replay that run
        self.restart(id_dir, "Restart1", restart_iteration=1)


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
