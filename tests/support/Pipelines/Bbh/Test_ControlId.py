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

import numpy as np
import numpy.testing as npt
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import to_dataframe
from spectre.Pipelines.Bbh.ControlId import (
    MAX_EFFECTIVE_SPIN,
    _convergence_errors,
    _convergence_test,
    _optimal_polynomial_order,
    _seeded_solve_failure,
    _watch_seeded_solve,
    conformal_mass_step,
    control_id,
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


# Synthetic physical parameters that converge exponentially with the
# polynomial order P, except for the center of mass, which converges slowly
def measured_params_at(P: int):
    return {
        "MassA": 0.5 + 1.0e-2**P,
        "DimensionlessSpinA": np.array([0.0, 0.0, 0.1 + 2.0 * 1.0e-2**P]),
        "CenterOfMass": np.array([1.0e-2 ** (P / 4.0), 0.0, 0.0]),
    }


def mock_generate_id(target_params, run_dir, polynomial_order, **kwargs):
    if polynomial_order in mock_generate_id.failing_orders:
        raise RuntimeError(f"Solve at P={polynomial_order} failed")
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
    "NewtonRaphson has reached its maximum number of iterations (1).",
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
    solve, in order ('None' for a good solve)."""
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
                control_params=SEEDING_CONTROL_PARAMS,
                residual_tolerance=1.0e-8,
                numeric_initial_guess=True,
                **kwargs,
            )
        return Path(last_run_dir), mock_control_generate_id.calls

    def test_conformal_mass_step(self):
        self.assertAlmostEqual(
            conformal_mass_step(
                dict(conformal_mass_a=0.8, conformal_mass_b=0.04),
                dict(conformal_mass_a=0.6, conformal_mass_b=0.05),
            ),
            1.0 / 3.0,
        )

    def test_seeded_solve_failure(self):
        self.assertIn(
            "Newton iteration 2",
            _seeded_solve_failure(STALLED_SOLVE_OUTPUT, finished=False),
        )
        self.assertIsNone(_seeded_solve_failure(GOOD_SOLVE_OUTPUT, True))
        self.assertIsNone(_seeded_solve_failure(GOOD_SOLVE_OUTPUT, False))
        self.assertIsNone(
            _seeded_solve_failure(UNCONVERGED_SOLVE_OUTPUT, finished=False)
        )
        self.assertIn(
            "did not converge",
            _seeded_solve_failure(UNCONVERGED_SOLVE_OUTPUT, finished=True),
        )

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
            numeric_initial_guess_max_mass_step=0.1
        )
        previous = dict(
            conformal_mass_a=INITIAL_CONFORMAL_MASS,
            conformal_mass_b=INITIAL_CONFORMAL_MASS,
        )
        steps = []
        for i, call in enumerate(calls):
            step = conformal_mass_step(previous, call)
            steps.append(step)
            if step < 0.1:
                self.assertEqual(
                    call["glob"],
                    str(self.id_dir / f"{i:04d}_InitialData" / "BbhVolume*.h5"),
                )
            else:
                self.assertIsNone(call["glob"])
            previous = call
        # The first step is large and the last ones are small, so the test
        # covers both branches
        self.assertGreater(steps[0], 0.1)
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


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
