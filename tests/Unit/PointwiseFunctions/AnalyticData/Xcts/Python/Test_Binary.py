# Distributed under the MIT License.
# See LICENSE.txt for details.

import unittest

import numpy as np
import numpy.testing as npt

from spectre.DataStructures import DataVector
from spectre.DataStructures.Tensor import InverseJacobian, tnsr
from spectre.PointwiseFunctions.AnalyticData.Xcts import Binary
from spectre.PointwiseFunctions.AnalyticSolutions.GeneralRelativity import (
    KerrSchild,
)
from spectre.Spectral import Basis, Mesh, Quadrature, logical_coordinates

PARAMS = dict(
    xcoords=[-5.0, 6.0],
    center_of_mass_offset=[0.1, -0.2],
    mass_left=0.4,
    dimensionless_spin_left=[0.1, 0.2, 0.3],
    mass_right=0.7,
    dimensionless_spin_right=[0.0, 0.0, -0.4],
    angular_velocity=0.02,
    expansion=-1.0e-3,
    linear_velocity=[1.0e-3, -2.0e-3, 3.0e-3],
    falloff_widths=[4.0, 5.0],
)


def to_numpy(tensor):
    """Dense numpy array of a spectre tensor, shape (3,) * rank + (n,)."""
    rank = tensor.rank
    if rank == 0:
        return np.asarray(tensor.get())
    n = len(tensor.get(*([0] * rank)))
    out = np.empty((3,) * rank + (n,))
    for idx in np.ndindex(*(3,) * rank):
        out[idx] = np.asarray(tensor.get(*idx))
    return out


def affine_element(num_points, lower, upper):
    """Points, mesh and inverse Jacobian of the box [lower, upper] mapped
    affinely from the logical cube."""
    mesh = Mesh[3](num_points, Basis.Legendre, Quadrature.GaussLobatto)
    xi = np.asarray(logical_coordinates(mesh))
    lower, upper = np.asarray(lower), np.asarray(upper)
    x = 0.5 * (upper + lower)[:, None] + 0.5 * (upper - lower)[:, None] * xi
    n = x.shape[1]
    inv_jac = InverseJacobian[DataVector, 3](num_points=n, fill=0.0)
    for d in range(3):
        inv_jac[inv_jac.get_storage_index(d, d)] = DataVector(
            n, 2.0 / (upper[d] - lower[d])
        )
    return tnsr.I[DataVector, 3](x), x, mesh, inv_jac


def isolated(x, center, mass, spin, quantities):
    """The isolated Kerr-Schild hole at `center`, from the GR binding."""
    shifted = x - np.asarray(center)[:, None]
    ks = KerrSchild(mass=mass, dimensionless_spin=spin)
    return [
        to_numpy(v)
        for v in ks.variables(
            tnsr.I[DataVector, 3](shifted), quantities
        ).values()
    ]


class TestBinary(unittest.TestCase):
    def setUp(self):
        self.binary = Binary(**PARAMS)
        self.centers = [
            [PARAMS["xcoords"][i]] + PARAMS["center_of_mass_offset"]
            for i in range(2)
        ]
        self.x_t, self.x, self.mesh, self.inv_jac = affine_element(
            8, [-2.0, -1.5, -1.0], [2.0, 1.5, 1.0]
        )

    def test_superposition(self):
        # Conformal metric and K against the isolated holes of the GR
        # Kerr-Schild binding, superposed with the Gaussian windows by hand
        g, K = self.binary.variables(
            self.x_t, ["Conformal(SpatialMetric)", "TraceExtrinsicCurvature"]
        ).values()
        g, K = to_numpy(g), to_numpy(K)
        g_expected = np.zeros_like(g)
        K_expected = np.zeros_like(K)
        flat = np.eye(3)[:, :, None]
        for i, (mass, spin) in enumerate(
            [
                (PARAMS["mass_left"], PARAMS["dimensionless_spin_left"]),
                (PARAMS["mass_right"], PARAMS["dimensionless_spin_right"]),
            ]
        ):
            gamma, trK = isolated(
                self.x,
                self.centers[i],
                mass,
                spin,
                ["SpatialMetric", "TraceExtrinsicCurvature"],
            )
            r2 = np.sum(
                (self.x - np.asarray(self.centers[i])[:, None]) ** 2, axis=0
            )
            window = np.exp(-r2 / PARAMS["falloff_widths"][i] ** 2)
            g_expected += window * (gamma - flat)
            K_expected += window * trK
        g_expected += flat
        npt.assert_allclose(g, g_expected, rtol=0.0, atol=1e-14)
        npt.assert_allclose(K, K_expected, rtol=0.0, atol=1e-14)

    def test_shift_background(self):
        (beta,) = self.binary.variables(self.x_t, ["ShiftBackground"]).values()
        x, y, z = self.x
        omega, adot = PARAMS["angular_velocity"], PARAMS["expansion"]
        v = PARAMS["linear_velocity"]
        expected = np.array(
            [
                -omega * y + adot * x + v[0],
                omega * x + adot * y + v[1],
                adot * z + v[2],
            ]
        )
        npt.assert_allclose(to_numpy(beta), expected, rtol=0.0, atol=1e-15)

    def test_metric_derivative(self):
        # Analytic derivative (including the window-function term) against
        # second-order central differences
        (dg,) = self.binary.variables(
            self.x_t, ["deriv(Conformal(SpatialMetric))"]
        ).values()
        dg = to_numpy(dg)
        h = 1.0e-5
        for k in range(3):
            step = np.zeros((3, 1))
            step[k] = h
            plus, minus = [
                to_numpy(
                    self.binary.variables(
                        tnsr.I[DataVector, 3](self.x + s),
                        ["Conformal(SpatialMetric)"],
                    )["Conformal(SpatialMetric)"]
                )
                for s in (step, -step)
            ]
            npt.assert_allclose(
                dg[k], (plus - minus) / (2.0 * h), rtol=0.0, atol=1e-9
            )

    def test_mesh_and_no_mesh_agree(self):
        names = [
            "Conformal(SpatialMetric)",
            "Conformal(InverseSpatialMetric)",
            "ConformalChristoffelSecondKind",
            "LongitudinalShiftBackgroundMinusDtConformalMetric",
        ]
        without = self.binary.variables(self.x_t, names)
        with_mesh = self.binary.variables(
            self.x_t, self.mesh, self.inv_jac, names
        )
        for name in names:
            npt.assert_array_equal(
                to_numpy(without[name]), to_numpy(with_mesh[name])
            )

    def test_numeric_ricci_converges_to_vacuum_value(self):
        # One hole: no falloff, and a companion of negligible mass far away.
        # The conformal metric is then the Kerr-Schild spatial metric of the
        # left hole (to ~1e-14), whose Ricci scalar is K_ij K^ij - K^2 by the
        # vacuum Hamiltonian constraint. The numeric Ricci scalar must
        # converge to it with the number of points.
        single = Binary(
            **dict(
                PARAMS,
                mass_right=1.0e-14,
                falloff_widths=None,
            )
        )
        errors = []
        for num_points in (8, 12, 16):
            x_t, x, mesh, inv_jac = affine_element(
                num_points, [-3.0, -1.0, -1.0], [-1.0, 1.0, 1.0]
            )
            (ricci,) = single.variables(
                x_t, mesh, inv_jac, ["ConformalRicciScalar"]
            ).values()
            inv_g, K_ij, trK = isolated(
                x,
                self.centers[0],
                PARAMS["mass_left"],
                PARAMS["dimensionless_spin_left"],
                [
                    "InverseSpatialMetric",
                    "ExtrinsicCurvature",
                    "TraceExtrinsicCurvature",
                ],
            )
            exact = (
                np.einsum(
                    "ik...,jl...,ij...,kl...->...", inv_g, inv_g, K_ij, K_ij
                )
                - trK**2
            )
            errors.append(np.max(np.abs(to_numpy(ricci) - exact)))
        # Measured 2026-10-06: 3.9e-4, 3.5e-6, 2.4e-8 (max |R| = 0.041)
        self.assertLess(errors[1], errors[0] / 30.0)
        self.assertLess(errors[2], errors[1] / 30.0)
        self.assertLess(errors[2], 1e-7)

    def test_flat_region(self):
        # Where both windows vanish the conformal metric is flat. The
        # background shift (rotation + expansion + boost) is a conformal
        # Killing vector of flat space, so its longitudinal operator and that
        # operator's divergence vanish; so do the Ricci scalar and dK.
        flat = Binary(**dict(PARAMS, falloff_widths=[1.0e-3, 1.0e-3]))
        values = flat.variables(
            self.x_t,
            self.mesh,
            self.inv_jac,
            [
                "Conformal(SpatialMetric)",
                "LongitudinalShiftBackgroundMinusDtConformalMetric",
                "div(LongitudinalShiftBackgroundMinusDtConformalMetric)",
                "ConformalRicciScalar",
                "deriv(TraceExtrinsicCurvature)",
            ],
        )
        npt.assert_array_equal(
            to_numpy(values["Conformal(SpatialMetric)"]),
            np.broadcast_to(np.eye(3)[:, :, None], (3, 3, self.x.shape[1])),
        )
        for name in list(values)[1:]:
            npt.assert_allclose(
                to_numpy(values[name]), 0.0, rtol=0.0, atol=1e-14
            )

    def test_errors(self):
        with self.assertRaisesRegex(ValueError, "without a mesh"):
            self.binary.variables(self.x_t, ["ConformalRicciScalar"])
        with self.assertRaisesRegex(ValueError, "not available"):
            self.binary.variables(
                self.x_t, self.mesh, self.inv_jac, ["Nonexistent"]
            )
        with self.assertRaisesRegex(ValueError, "ascending"):
            Binary(**dict(PARAMS, xcoords=[1.0, -1.0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
