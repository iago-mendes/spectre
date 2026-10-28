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
    MAX_EFFECTIVE_SPIN,
    UnconvergedSolveError,
    _convergence_errors,
    _convergence_test,
    _optimal_polynomial_order,
    _seeded_solve_failure,
    _watch_seeded_solve,
    control_id,
    control_scales,
    effective_spin,
    elliptic_solve_convergence,
    excision_move,
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
                control_params=SEEDING_CONTROL_PARAMS,
                residual_tolerance=1.0e-8,
                **{"numeric_initial_guess": True, **kwargs},
            )
        return Path(last_run_dir), mock_control_generate_id.calls

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
    return {
        "MassA": mock_mass(mass_a) + 0.01 * mass_b,
        "MassB": mock_mass(mass_b) + 0.01 * mass_a,
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
