# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import shutil
import unittest
from pathlib import Path

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
        # Can only test that the command runs until this error until we have
        # some BBH initial data to postprocess.
        with self.assertRaisesRegex(ValueError, "Number of observations"):
            postprocess_id_command(
                [
                    str(self.id_run_dir / "InitialData.yaml"),
                ]
            )

    def test_horizons_at_observation_time(self):
        # Write a Schwarzschild solution in Kerr-Schild coordinates (horizon at
        # coordinate radius 2) to a volume file, at an observation time that is
        # not zero, like the iteration count of an elliptic solve
        run_dir = self.test_dir / "Horizons"
        run_dir.mkdir()
        obs_time = 3.0
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
                        "Observers": {"VolumeFileName": "VolumeData"},
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


if __name__ == "__main__":
    configure_logging(log_level=logging.DEBUG)
    unittest.main(verbosity=2)
