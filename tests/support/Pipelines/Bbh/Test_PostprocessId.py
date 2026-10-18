# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import numpy.testing as npt
import yaml
from click.testing import CliRunner

import spectre.IO.H5 as spectre_h5
from spectre.Domain import ElementId, ElementMap, serialize_domain
from spectre.Domain.Creators import Sphere
from spectre.Informer import unit_test_build_path
from spectre.IO.H5.IterElements import Element
from spectre.NumericalAlgorithms.LinearOperators import partial_derivative
from spectre.Pipelines.Bbh.ControlId import UnconvergedSolveError
from spectre.Pipelines.Bbh.InitialData import generate_id
from spectre.Pipelines.Bbh.PostprocessId import (
    postprocess_id,
    postprocess_id_command,
)
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.PointwiseFunctions.GeneralRelativity import ricci_tensor
from spectre.Spectral import Basis, Mesh, Quadrature
from spectre.support.Logging import configure_logging


class TestPostprocessId(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(
            unit_test_build_path(), "support/Pipelines/Bbh/PostprocessId"
        )
        shutil.rmtree(self.test_dir, ignore_errors=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.bin_dir = Path(unit_test_build_path(), "../../bin").resolve()
        generate_id(
            {
                "MassRatio": 1.5,
                "MassA": 0.6,
                "MassB": 0.4,
                "DimensionlessSpinA": [0.0, 0.0, 0.0],
                "DimensionlessSpinB": [0.0, 0.0, 0.0],
            },
            separation=20.0,
            orbital_angular_velocity=0.01,
            radial_expansion_velocity=-1.0e-5,
            refinement_level=1,
            polynomial_order=5,
            run_dir=self.test_dir / "ID",
            scheduler=None,
            submit=False,
            executable=str(self.bin_dir / "SolveXcts"),
        )
        self.id_run_dir = self.test_dir / "ID"

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_cli(self):
        # Not using `CliRunner.invoke()` because it runs in an isolated
        # environment and doesn't work with MPI in the container.
        # Can only test that the command runs until these errors until we have
        # some BBH initial data to postprocess. The ID was not solved, so there
        # are no residuals of the elliptic solve.
        with self.assertRaisesRegex(
            UnconvergedSolveError, "did not run or did not finish"
        ):
            postprocess_id_command(
                [
                    str(self.id_run_dir / "InitialData.yaml"),
                ]
            )
        with self.assertRaisesRegex(ValueError, "Number of observations"):
            postprocess_id_command(
                [
                    str(self.id_run_dir / "InitialData.yaml"),
                    "--allow-unconverged-solve",
                ]
            )

    def write_schwarzschild_run(self, run_dir, obs_time, newton_residuals):
        """Write a run whose solution is Schwarzschild in Kerr-Schild
        coordinates (horizon at coordinate radius 2), observed at 'obs_time',
        with the Newton-Raphson residuals 'newton_residuals' (one per
        iteration) and an AbsoluteResidual of 1e-10. Returns the path of the
        input file."""
        run_dir.mkdir()
        domain = Sphere(
            inner_radius=1.5,
            outer_radius=3.0,
            excise=True,
            initial_refinement=0,
            initial_number_of_grid_points=8,
            use_equiangular_map=True,
        ).create_domain()
        solution = KerrSchild(mass=1.0, dimensionless_spin=[0.0, 0.0, 0.0])
        element_volume_data = []
        for block_id in range(6):
            element_id = ElementId[3](block_id)
            element = Element(
                id=element_id,
                mesh=Mesh[3](10, Basis.Legendre, Quadrature.GaussLobatto),
                map=ElementMap(element_id, domain),
            )
            tensors = solution.variables(
                element.inertial_coordinates,
                [
                    "SpatialMetric",
                    "InverseSpatialMetric",
                    "ExtrinsicCurvature",
                    "SpatialChristoffelSecondKind",
                ],
            )
            christoffels = tensors["SpatialChristoffelSecondKind"]
            tensors["SpatialRicci"] = ricci_tensor(
                christoffels,
                partial_derivative(
                    christoffels, element.mesh, element.inv_jacobian
                ),
            )
            element_volume_data.append(
                spectre_h5.ElementVolumeData(
                    element.id,
                    [
                        spectre_h5.TensorComponent(
                            name + tensor.component_suffix(i), tensor[i]
                        )
                        for name, tensor in tensors.items()
                        for i in range(len(tensor))
                    ],
                    element.mesh,
                )
            )
        with spectre_h5.H5File(str(run_dir / "VolumeData0.h5"), "w") as h5file:
            volfile = h5file.insert_vol("VolumeData", version=0)
            volfile.write_volume_data(
                observation_id=0,
                observation_value=obs_time,
                elements=element_volume_data,
                serialized_domain=serialize_domain(domain),
            )
        # The residuals that the elliptic solver observes
        with spectre_h5.H5File(
            str(run_dir / "BbhReductions.h5"), "w"
        ) as h5file:
            dat_file = h5file.insert_dat(
                "NewtonRaphsonResiduals",
                [
                    "Iteration",
                    "Walltime",
                    "GlobalizationStep",
                    "Residual",
                    "StepLength",
                ],
                0,
            )
            for iteration, residual in enumerate(newton_residuals):
                dat_file.append([iteration, iteration, 0.0, residual, 1.0])
        # Write the parts of an ID input file that the postprocessing reads.
        # Both horizons are placed at the origin.
        id_input_file_path = run_dir / "InitialData.yaml"
        with open(id_input_file_path, "w") as open_input_file:
            yaml.safe_dump_all(
                [
                    {"TargetParams": {}},
                    {
                        "Background": {
                            "Binary": {
                                "XCoords": [0.0, 0.0],
                                "CenterOfMassOffset": [0.0, 0.0],
                            }
                        },
                        "DomainCreator": {
                            "BinaryCompactObject": {
                                "ObjectA": {"InnerRadius": 1.5},
                                "ObjectB": {"InnerRadius": 1.5},
                            }
                        },
                        "Observers": {
                            "VolumeFileName": "VolumeData",
                            "ReductionFileName": "BbhReductions",
                        },
                        "NonlinearSolver": {
                            "NewtonRaphson": {
                                "ConvergenceCriteria": {
                                    "MaxIterations": 3,
                                    "RelativeResidual": 0.0,
                                    "AbsoluteResidual": 1.0e-10,
                                }
                            }
                        },
                        "EventsAndTriggersAtIterations": [
                            {
                                "Trigger": "Always",
                                "Events": [
                                    {
                                        "ObserveFields": {
                                            "SubfileName": "VolumeData"
                                        }
                                    }
                                ],
                            }
                        ],
                    },
                ],
                open_input_file,
            )
        return id_input_file_path

    def test_horizons_at_observation_time(self):
        # The observation time is not zero, like the iteration count of an
        # elliptic solve
        run_dir = self.test_dir / "Horizons"
        obs_time = 3.0
        id_input_file_path = self.write_schwarzschild_run(
            run_dir, obs_time, newton_residuals=[1.0, 1.0e-4, 1.0e-11]
        )

        postprocess_id(id_input_file_path, horizon_l_max=12, control=False)

        # The horizon data must be labeled with the time of the observation
        with spectre_h5.H5File(str(run_dir / "Horizons.h5"), "r") as h5file:
            for object_label in ["AhA", "AhB"]:
                quantities = h5file.get_dat(f"{object_label}.dat")
                legend = quantities.get_legend()
                npt.assert_allclose(
                    quantities.get_data()[0, legend.index("ChristodoulouMass")],
                    1.0,
                    atol=1e-3,
                )
                h5file.close_current_object()
                coefs = h5file.get_dat(f"{object_label}/Coefficients.dat")
                self.assertEqual(coefs.get_legend()[0], "Time")
                self.assertEqual(coefs.get_data()[0, 0], obs_time)
                h5file.close_current_object()
                coords = h5file.get_vol(f"{object_label}/Coordinates")
                self.assertEqual(
                    coords.get_observation_value(
                        coords.list_observation_ids()[0]
                    ),
                    obs_time,
                )
                h5file.close_current_object()

    def test_unconverged_solve(self):
        # The solve reached MaxIterations = 3 above the AbsoluteResidual 1e-10:
        # the postprocessing stops before it finds horizons or controls
        run_dir = self.test_dir / "Unconverged"
        id_input_file_path = self.write_schwarzschild_run(
            run_dir, 3.0, newton_residuals=[1.0, 0.5, 0.2, 0.1]
        )
        with patch(
            "spectre.Pipelines.Bbh.PostprocessId.control_id"
        ) as control_id:
            with self.assertRaisesRegex(
                UnconvergedSolveError,
                r"did not converge: the Newton-Raphson residual is 1.0+e-01"
                r" after 3 iterations .*AbsoluteResidual = 1.0+e-10.*"
                r"MaxIterations = 3",
            ):
                postprocess_id(id_input_file_path, horizon_l_max=12)
            control_id.assert_not_called()
        self.assertFalse((run_dir / "Horizons.h5").exists())
        # Allowed explicitly, the postprocessing finds the horizons
        postprocess_id(
            id_input_file_path,
            horizon_l_max=12,
            control=False,
            require_converged_solve=False,
        )
        self.assertTrue((run_dir / "Horizons.h5").exists())


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
