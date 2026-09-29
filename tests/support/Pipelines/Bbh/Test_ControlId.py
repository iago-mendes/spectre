# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt

import spectre.IO.H5 as spectre_h5
from spectre.Informer import unit_test_build_path
from spectre.IO.H5 import to_dataframe
from spectre.Pipelines.Bbh.ControlId import (
    _convergence_errors,
    _convergence_test,
    _optimal_polynomial_order,
)
from spectre.support.Logging import configure_logging


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


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
