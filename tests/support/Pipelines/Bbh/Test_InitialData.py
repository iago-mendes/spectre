# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

import numpy.testing as npt
import yaml
from click.testing import CliRunner

from spectre.Informer import unit_test_build_path
from spectre.Pipelines.Bbh.InitialData import (
    MIN_SHELL_THICKNESS,
    generate_id_command,
    id_parameters,
)
from spectre.support.Logging import configure_logging


class TestInitialData(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/InitialData"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_generate_id(self):
        params = id_parameters(
            conformal_mass_a=0.6,
            conformal_mass_b=0.4,
            horizon_rotation_a=[-0.04, -0.08, -0.1],
            horizon_rotation_b=[-0.3, -0.4, -0.4],
            center_of_mass_offset=[0.1, 0.2, 0.3],
            linear_velocity=[0.1, 0.2, 0.3],
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            negative_expansion_bc=True,
            target_params={
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.1, 0.2, 0.3],
                "DimensionlessSpinB": [0.4, 0.5, 0.6],
            },
        )
        self.assertEqual(params["ConformalMassRight"], 0.6)
        self.assertEqual(params["ConformalMassLeft"], 0.4)
        self.assertEqual(params["XRight"], 8.0 + 0.1)
        self.assertEqual(params["XLeft"], -12.0 + 0.1)
        self.assertEqual(
            [params[f"CenterOfMassOffset_{yz}"] for yz in "yz"],
            [0.2, 0.3],
        )
        self.assertEqual(
            [params[f"LinearVelocity_{xyz}"] for xyz in "xyz"],
            [0.1, 0.2, 0.3],
        )
        self.assertAlmostEqual(params["ExcisionRadiusRight"], 1.07546791205)
        self.assertAlmostEqual(params["ExcisionRadiusLeft"], 0.5504049327)
        self.assertEqual(params["OrbitalAngularVelocity"], 0.01)
        self.assertEqual(params["RadialExpansionVelocity"], -1.0e-5)
        self.assertEqual(
            [params[f"ConformalSpinRight_{xyz}"] for xyz in "xyz"],
            [0.1, 0.2, 0.3],
        )
        self.assertEqual(
            [params[f"ConformalSpinLeft_{xyz}"] for xyz in "xyz"],
            [0.4, 0.5, 0.6],
        )
        npt.assert_allclose(
            [params[f"HorizonRotationRight_{xyz}"] for xyz in "xyz"],
            [-0.04, -0.08, -0.1 + 0.01],
        )
        npt.assert_allclose(
            [params[f"HorizonRotationLeft_{xyz}"] for xyz in "xyz"],
            [-0.3, -0.4, -0.4 + 0.01],
        )
        # The shells are thick enough at this separation, so the outer radii
        # are the historical fractions of the separation
        self.assertAlmostEqual(params["ObjectAOuterRadius"], 20.0 / 3.75)
        self.assertAlmostEqual(params["ObjectBOuterRadius"], 20.0 / 3.75 / 1.5)
        self.assertAlmostEqual(params["FalloffWidthRight"], 6.479672589667676)
        self.assertAlmostEqual(params["FalloffWidthLeft"], 5.520327410332324)
        self.assertEqual(params["L"], 1)
        self.assertEqual(params["P"], 5)
        # Newtonian center of mass (without offset) is zero
        self.assertAlmostEqual(
            params["ConformalMassRight"] * (params["XRight"] - 0.1)
            + params["ConformalMassLeft"] * (params["XLeft"] - 0.1),
            0.0,
        )

    def test_small_separation(self):
        mass_ratio = 1.0e4
        mass_a = mass_ratio / (1.0 + mass_ratio)
        mass_b = 1.0 / (1.0 + mass_ratio)

        def params_at(separation):
            return id_parameters(
                conformal_mass_a=0.82 * mass_a,
                conformal_mass_b=0.82 * mass_b,
                horizon_rotation_a=[0.0, 0.0, 0.0],
                horizon_rotation_b=[0.0, 0.0, 0.0],
                center_of_mass_offset=[0.0, 0.0, 0.0],
                linear_velocity=[0.0, 0.0, 0.0],
                separation=separation,
                orbital_angular_velocity=0.01,
                radial_expansion_velocity=0.0,
                refinement_level=1,
                polynomial_order=5,
                negative_expansion_bc=True,
                target_params={
                    "MassA": mass_a,
                    "MassB": mass_b,
                    "DimensionlessSpinA": [0.0, 0.0, 0.0],
                    "DimensionlessSpinB": [0.0, 0.0, 0.0],
                },
            )

        # The shells are floored at MIN_SHELL_THICKNESS excision radii
        params = params_at(5.5)
        self.assertAlmostEqual(
            params["ObjectAOuterRadius"],
            MIN_SHELL_THICKNESS * params["ExcisionRadiusRight"],
        )
        self.assertAlmostEqual(
            params["ObjectBOuterRadius"],
            MIN_SHELL_THICKNESS * params["ExcisionRadiusLeft"],
        )
        # At even smaller separations the shell around object A is clipped to
        # stay inside its cube
        with self.assertLogs(level="WARNING"):
            params = params_at(5.0)
        self.assertAlmostEqual(params["ObjectAOuterRadius"], 0.9 * 5.0 / 2.0)
        self.assertAlmostEqual(
            params["ObjectBOuterRadius"],
            MIN_SHELL_THICKNESS * params["ExcisionRadiusLeft"],
        )

    def test_extra_radial_points_clip(self):
        mass_ratio = 1.0e6
        mass_a = mass_ratio / (1.0 + mass_ratio)
        mass_b = 1.0 / (1.0 + mass_ratio)

        def params_at(polynomial_order):
            return id_parameters(
                conformal_mass_a=0.82 * mass_a,
                conformal_mass_b=0.82 * mass_b,
                horizon_rotation_a=[0.0, 0.0, 0.0],
                horizon_rotation_b=[0.0, 0.0, 0.0],
                center_of_mass_offset=[0.0, 0.0, 0.0],
                linear_velocity=[0.0, 0.0, 0.0],
                separation=20.0,
                orbital_angular_velocity=0.01,
                radial_expansion_velocity=0.0,
                refinement_level=1,
                polynomial_order=polynomial_order,
                negative_expansion_bc=True,
                target_params={
                    "MassA": mass_a,
                    "MassB": mass_b,
                    "DimensionlessSpinA": [0.0, 0.0, 0.0],
                    "DimensionlessSpinB": [0.0, 0.0, 0.0],
                },
            )

        # The extra radial points round(0.9 * ln(q)) = 12 fit below the
        # maximum of 20 points at low polynomial order
        self.assertEqual(params_at(5)["ExtraRadPoints"], 12)
        # At higher polynomial order they are clipped instead of failing
        with self.assertLogs(level="WARNING") as logs:
            params = params_at(8)
        self.assertTrue(
            any(
                "Clipping extra radial refinement p" in line
                for line in logs.output
            )
        )
        self.assertEqual(params["ExtraRadPoints"], 20 - 8 - 2)

    def test_cli(self):
        common_args = [
            "--mass-ratio",
            "1.5",
            "--chi-A",
            "0.1",
            "0.2",
            "0.3",
            "--chi-B",
            "0.4",
            "0.5",
            "0.6",
            "--separation",
            "20",
            "--orbital-angular-velocity",
            "0.01",
            "--radial-expansion-velocity",
            "-1.0e-5",
            "--refinement-level",
            "1",
            "--polynomial-order",
            "5",
            "-E",
            str(self.bin_dir / "SolveXcts"),
            "--no-schedule",
        ]
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        try:
            generate_id_command(
                common_args
                + [
                    "-o",
                    str(self.test_dir),
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        self.assertTrue(
            (self.test_dir / "0000_InitialData/InitialData.yaml").exists()
        )
        # Test with pipeline directory
        try:
            generate_id_command(
                common_args
                + [
                    "-d",
                    str(self.test_dir / "Pipeline"),
                    "--evolve",
                    "--eccentricity-control",
                    "--no-submit",
                ]
            )
        except SystemExit as e:
            self.assertEqual(e.code, 0)
        with open(
            self.test_dir
            / "Pipeline/Ecc0/ID/0000_InitialData/InitialData.yaml",
            "r",
        ) as open_input_file:
            metadata = next(yaml.safe_load_all(open_input_file))
        self.assertEqual(
            metadata["TargetParams"],
            {
                "MassRatio": 1.5,
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.1, 0.2, 0.3],
                "DimensionlessSpinB": [0.4, 0.5, 0.6],
                "CenterOfMass": [0.0, 0.0, 0.0],
                "AdmLinearMomentum": [0.0, 0.0, 0.0],
                "Eccentricity": 0.0,
                "EccentricityAbsoluteTolerance": 1e-3,
                "MeanAnomalyFraction": None,
                "NumOrbits": None,
                "TimeToMerger": None,
                "EvolutionLev": 1,
            },
        )
        self.assertEqual(
            metadata["Next"],
            {
                "Run": "spectre.Pipelines.Bbh.PostprocessId:postprocess_id",
                "With": {
                    "id_input_file_path": "__file__",
                    "id_run_dir": "./",
                    "pipeline_dir": str(self.test_dir.resolve() / "Pipeline"),
                    "horizon_l_max": 20,
                    "control": True,
                    "control_delay": 2,
                    "control_refinement_level": 1,
                    "control_polynomial_order": 5,
                    "control_params": [
                        "MassA",
                        "MassB",
                        "DimensionlessSpinA",
                        "DimensionlessSpinB",
                        "CenterOfMass",
                        "AdmLinearMomentum",
                    ],
                    "evolve": True,
                    "eccentricity_control": True,
                    "negative_expansion_bc": True,
                    "scheduler": "None",
                    "copy_executable": "None",
                    "submit_script_template": "None",
                    "submit": True,
                },
            },
        )


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
