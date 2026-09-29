# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
import os
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import h5py
import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5 import to_dataframe
from spectre.Pipelines.Bbh.InitialData import TargetParams, generate_id
from spectre.support.DirectoryStructure import Segment

logger = logging.getLogger(__name__)

# The default tolerance is just below the effect of junk radiation on the
# controlled paramaters, which  is about 1.0e-3.
DEFAULT_RESIDUAL_TOLERANCE = 1.0e-4
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_CONTROL_DELAY = 2
# Control of the asymptotic quantities (ADM mass, ADM momenta, and center of
# mass) starts only once the horizon masses and spins are within this residual
# of their targets
HORIZON_RESIDUAL_FOR_ASYMPTOTIC_CONTROL = 5.0e-3

# Step-size constraints of the control loop. Each constraint limits the step of
# the free data to a fraction alpha in [0, 1] of the quasi-Newton step:
# - mass: the step of each conformal mass is at most MAX_RELATIVE_MASS_STEP
#   times its current value.
# - com: the x-coordinate of the larger black hole stays above MIN_X_A, so the
#   black hole does not cross the origin.
# - spin: the effective spin of each black hole stays below MAX_EFFECTIVE_SPIN
#   (see 'spin_step_size_constraint').
StepSizeConstraint = Literal["mass", "com", "spin"]
STEP_SIZE_CONSTRAINTS: List[StepSizeConstraint] = ["mass", "com", "spin"]
DEFAULT_STEP_SIZE_CONSTRAINTS: Sequence[StepSizeConstraint] = ("spin",)
MAX_RELATIVE_MASS_STEP = 0.2
MIN_X_A = 1.0e-10
MAX_EFFECTIVE_SPIN = 1.0 - 1.0e-4
DEFAULT_CONVERGENCE_TEST_TOLERANCE = 1.0e-5
CONVERGENCE_TEST_MAX_POLYNOMIAL_ORDER = 12

# Free data choices associated with each physical parameter
# Note 1: the values below need to match the argument names of `generate_id`.
# Note 2: conformal_mass_a/b and conformal_spin_a/b refer to the Kerr masses and
#         spins used in the background.
FreeDataFromParams: Dict[TargetParams, str] = {
    "MassA": "conformal_mass_a",
    "MassB": "conformal_mass_b",
    "DimensionlessSpinA": "horizon_rotation_a",
    "DimensionlessSpinB": "horizon_rotation_b",
    "CenterOfMass": "center_of_mass_offset",
    "AdmLinearMomentum": "linear_velocity",
    "AdmMass": "radial_expansion_velocity",
    "AdmAngularMomentumZ": "orbital_angular_velocity",
}

# Quantites (free data or parameters) that are scalars
# Note: this is useful for switching between dictionaries and arrays below.
ScalarQuantities = [
    "MassA",
    "MassB",
    "conformal_mass_a",
    "conformal_mass_b",
    "AdmMass",
    "AdmAngularMomentumZ",
    "radial_expansion_velocity",
    "orbital_angular_velocity",
]


def control_scales(
    control_params: List[TargetParams],
    param_index_map: Dict[str, int],
    target_params: Dict[TargetParams, Union[float, Sequence[float]]],
    u: np.ndarray,
    num_params: int,
):
    """Natural magnitudes of the free data and of the residuals.

    The control problem is posed in raw free data, whose components differ in
    magnitude by powers of the mass ratio: conformal_mass_b ~ 1/q, while
    horizon_rotation_b ~ chi_B / (2 rbar_B) ~ q. The Jacobian inherits this, so
    d(chi_B)/d(Omega_B) = 2 rbar_B ~ 1/q and d(chi_B)/d(conformal_mass_b) ~ q,
    and its condition number grows like q (chi_B = 0) or q^3 (chi_B > 0).

    This hurts Broyden's method: its update 'J += outer(F, du) / (du . du)'
    writes its correction into the columns of J in proportion to 'du', so a
    step dominated by Omega_B pollutes the Omega_B column of every row,
    including rows that should not couple to it at all. Inverting the polluted
    Jacobian then leaks the chi_B residual into the conformal-mass step.

    Dividing each free-data component by the scale below makes the spin control
    the effective spin 'chi_eff_a = 2 rbar_a Omega_a', where
    'rbar_a = Mbar_a (1 + sqrt(1 - chi_a^2))' is the horizon radius of the
    conformal Kerr solution, and each mass control a fraction of its target.
    The mass residuals are measured relative to their targets as well, and the
    spin residuals are already dimensionless. The scaled Jacobian is then O(1)
    in every entry.

    The same rule extends to the controls of hyperbolic encounters: the ADM
    mass and the ADM angular momentum are divided by their targets, and the
    free data that control them (the radial expansion velocity and the orbital
    angular velocity) by their own initial magnitudes. Without this, the
    largest Jacobian entry is d(J_ADM^z)/d(Omega_0) = eta D^2, so the condition
    number grows as the square of the separation D.

    The scales are computed once from the initial free data and held fixed, so
    that the Jacobian accumulated by Broyden's method keeps a single consistent
    meaning across iterations.

    Arguments:
      control_params: List of parameters to control.
      param_index_map: Index of each control parameter, and of its free data,
        in the vectors of residuals and free data.
      target_params: Target values of the control parameters.
      u: Initial free data.
      num_params: Length of the vectors of residuals and free data.

    Returns: (u_scale, f_scale), both of length 'num_params', such that the
      dimensionless free data is 'u / u_scale' and the dimensionless residual
      is 'F / f_scale'. Components without a natural scale are set to 1.
    """
    u_scale = np.ones(num_params)
    f_scale = np.ones(num_params)

    def set_scale(param_key, free_data_key):
        # Only the controlled quantities are in the 'param_index_map'
        if (
            param_key not in control_params
            or param_key not in param_index_map
            or free_data_key not in param_index_map
        ):
            return
        idx = param_index_map[param_key]
        u_value = u[param_index_map[free_data_key]]
        f_value = target_params.get(param_key)
        if u_value and np.isfinite(u_value):
            u_scale[idx] = abs(u_value)
        if f_value and np.isfinite(f_value):
            f_scale[idx] = abs(f_value)

    # Controls of hyperbolic encounters
    set_scale("AdmMass", "radial_expansion_velocity")
    set_scale("AdmAngularMomentumZ", "orbital_angular_velocity")

    # Horizon masses and spins
    for mass_key, spin_key in zip(
        ["MassA", "MassB"], ["DimensionlessSpinA", "DimensionlessSpinB"]
    ):
        if mass_key in control_params:
            idx = param_index_map[mass_key]
            # Fall back to the conformal mass if the target is unusable, so the
            # scale is never zero
            target_mass = abs(target_params.get(mass_key, 0.0))
            if target_mass == 0.0:
                target_mass = abs(u[idx])
            if target_mass > 0.0:
                u_scale[idx] = target_mass
                f_scale[idx] = target_mass
        if spin_key in control_params and mass_key in control_params:
            conformal_spin = target_params[spin_key]
            spin_term = 1.0 + np.sqrt(
                1.0 - np.dot(conformal_spin, conformal_spin)
            )
            rbar = abs(u[param_index_map[mass_key]]) * spin_term
            if rbar > 0.0:
                idx = param_index_map[spin_key]
                # Omega -> 2 rbar Omega, i.e. the effective spin
                u_scale[idx : idx + 3] = 1.0 / (2.0 * rbar)
    return u_scale, f_scale


def effective_spin(
    conformal_mass: float,
    horizon_rotation: Sequence[float],
    conformal_spin: Sequence[float],
) -> float:
    """Effective spin 'chi_eff = 2 rbar |Omega|' of a black hole.

    Here 'rbar = Mbar (1 + sqrt(1 - chi^2))' is the horizon radius of the
    conformal Kerr solution with mass Mbar and dimensionless spin chi, and Omega
    is the horizon rotation. The apparent-horizon boundary condition has no
    solution for chi_eff >= 1.
    """
    spin_term = 1.0 + np.sqrt(1.0 - np.dot(conformal_spin, conformal_spin))
    return (
        2.0
        * spin_term
        * abs(conformal_mass)
        * np.linalg.norm(np.asarray(horizon_rotation))
    )


def spin_step_size_constraint(
    conformal_mass: float,
    delta_conformal_mass: float,
    horizon_rotation: Sequence[float],
    delta_horizon_rotation: Sequence[float],
    conformal_spin: Sequence[float],
    max_effective_spin: float = MAX_EFFECTIVE_SPIN,
) -> float:
    """Largest fraction of a step that keeps the effective spin below a limit.

    The step changes the conformal mass Mbar by 'delta_conformal_mass' and the
    horizon rotation Omega by 'delta_horizon_rotation'. Both are scaled by the
    same alpha, so the effective spin along the step (see 'effective_spin') is

        chi_eff(alpha) = 2 S |Mbar + alpha dMbar| |Omega + alpha dOmega|,

    with S = 1 + sqrt(1 - chi^2). This function returns the largest alpha in
    [0, 1] such that chi_eff stays below 'max_effective_spin' along the whole
    step from 0 to alpha. Note that the change of the mass must be taken into
    account: solving for alpha with the mass held at another value can
    overshoot the limit.

    The condition chi_eff(alpha)^2 = max_effective_spin^2 is a quartic
    polynomial equation in alpha. Its real roots in [0, 1] bracket the
    admissible interval, and the crossing is refined by bisection so that the
    returned alpha is guaranteed to satisfy the limit.

    Returns 0 if the effective spin already exceeds the limit at alpha = 0.
    """
    spin_term = 1.0 + np.sqrt(1.0 - np.dot(conformal_spin, conformal_spin))
    rotation = np.asarray(horizon_rotation, dtype=float)
    delta_rotation = np.asarray(delta_horizon_rotation, dtype=float)
    mass_poly = np.polynomial.Polynomial([conformal_mass, delta_conformal_mass])
    rotation_sq_poly = np.polynomial.Polynomial(
        [
            np.dot(rotation, rotation),
            2.0 * np.dot(rotation, delta_rotation),
            np.dot(delta_rotation, delta_rotation),
        ]
    )
    # Positive where the effective spin exceeds the limit
    excess = (
        4.0 * spin_term**2 * mass_poly**2 * rotation_sq_poly
        - max_effective_spin**2
    ).trim()

    if excess(0.0) > 0.0:
        return 0.0

    def first_crossing(safe_alpha, unsafe_alpha):
        # Bisect between an alpha that satisfies the limit and one that
        # doesn't, keeping the one that does
        for _ in range(200):
            mid = 0.5 * (safe_alpha + unsafe_alpha)
            if mid <= safe_alpha or mid >= unsafe_alpha:
                break
            if excess(mid) > 0.0:
                unsafe_alpha = mid
            else:
                safe_alpha = mid
        return safe_alpha

    # The sign of the excess can only change at its real roots, so check it
    # once between consecutive roots
    roots = excess.roots() if excess.degree() > 0 else []
    crossings = sorted(
        root.real
        for root in roots
        if abs(root.imag) <= 1.0e-10 * max(1.0, abs(root))
        and 0.0 < root.real < 1.0
    )
    safe_alpha = 0.0
    lower = 0.0
    for upper in crossings + [1.0]:
        mid = 0.5 * (lower + upper)
        if excess(mid) > 0.0:
            return first_crossing(safe_alpha, mid)
        safe_alpha = mid
        lower = upper
    if excess(1.0) > 0.0:
        return first_crossing(safe_alpha, 1.0)
    return 1.0


def _measured_params(
    run_dir: Union[str, Path], control_params: List[TargetParams]
) -> Dict[TargetParams, Union[float, np.ndarray]]:
    """Physical parameters measured in an initial data run

    Reads the horizon quantities from 'Horizons.h5' and the ADM integrals from
    'BbhReductions.h5' in the 'run_dir'. Returns the measured value (scalar or
    3-vector) of each parameter in 'control_params'.
    """
    measured_params: Dict[TargetParams, Union[float, np.ndarray]] = {}

    # Get black hole physical parameters
    with spectre_h5.H5File(f"{run_dir}/Horizons.h5", "r") as horizons_file:
        AhA_quantities = to_dataframe(horizons_file.get_dat("AhA.dat")).iloc[-1]

        if "MassA" in control_params:
            measured_params["MassA"] = AhA_quantities["ChristodoulouMass"]
        if "DimensionlessSpinA" in control_params:
            measured_params["DimensionlessSpinA"] = np.array(
                [
                    AhA_quantities["DimensionlessSpinVector_x"],
                    AhA_quantities["DimensionlessSpinVector_y"],
                    AhA_quantities["DimensionlessSpinVector_z"],
                ]
            )

        horizons_file.close_current_object()
        AhB_quantities = to_dataframe(horizons_file.get_dat("AhB.dat")).iloc[-1]

        if "MassB" in control_params:
            measured_params["MassB"] = AhB_quantities["ChristodoulouMass"]
        if "DimensionlessSpinB" in control_params:
            measured_params["DimensionlessSpinB"] = np.array(
                [
                    AhB_quantities["DimensionlessSpinVector_x"],
                    AhB_quantities["DimensionlessSpinVector_y"],
                    AhB_quantities["DimensionlessSpinVector_z"],
                ]
            )

    # Get ADM integrals
    with spectre_h5.H5File(
        f"{run_dir}/BbhReductions.h5", "r"
    ) as reductions_file:
        adm_integrals = to_dataframe(
            reductions_file.get_dat("AdmIntegrals.dat")
        ).iloc[-1]

        if "CenterOfMass" in control_params:
            measured_params["CenterOfMass"] = np.array(
                [
                    adm_integrals["CenterOfMass_x"],
                    adm_integrals["CenterOfMass_y"],
                    adm_integrals["CenterOfMass_z"],
                ]
            )
        if "AdmLinearMomentum" in control_params:
            measured_params["AdmLinearMomentum"] = np.array(
                [
                    adm_integrals["AdmLinearMomentum_x"],
                    adm_integrals["AdmLinearMomentum_y"],
                    adm_integrals["AdmLinearMomentum_z"],
                ]
            )
        if "AdmMass" in control_params:
            measured_params["AdmMass"] = adm_integrals["AdmMass"]
        if "AdmAngularMomentumZ" in control_params:
            measured_params["AdmAngularMomentumZ"] = adm_integrals[
                "AdmAngularMomentum_z"
            ]

    return measured_params


def _convergence_errors(
    measured_params_by_order: Dict[
        int, Dict[TargetParams, Union[float, np.ndarray]]
    ],
    params: List[TargetParams],
) -> Dict[int, float]:
    """Convergence error of every polynomial order below the highest one

    The convergence error at polynomial order P is

        error(P) = max | measured_param(P) - measured_param(P_ref) |,

    where P_ref is the highest measured polynomial order and the maximum is
    taken over all components of all 'params'.
    """
    P_ref = max(measured_params_by_order)
    ref = measured_params_by_order[P_ref]
    return {
        P: max(
            float(
                np.max(
                    np.abs(
                        np.atleast_1d(measured_params[key])
                        - np.atleast_1d(ref[key])
                    )
                )
            )
            for key in params
        )
        for P, measured_params in sorted(measured_params_by_order.items())
        if P != P_ref
    }


def _optimal_polynomial_order(
    errors: Dict[int, float], tolerance: float
) -> Optional[int]:
    """Smallest polynomial order with a convergence error below 'tolerance'

    Returns 'None' if there is no such polynomial order.
    """
    for P in sorted(errors):
        if errors[P] < tolerance:
            return P
    return None


def _write_convergence_test(
    output_filename: Union[str, Path],
    subfile_name: str,
    measured_params_by_order: Dict[
        int, Dict[TargetParams, Union[float, np.ndarray]]
    ],
    control_params: List[TargetParams],
):
    """Write the convergence errors of every parameter component to a dat file

    Writes one row per measured polynomial order P with the absolute
    differences | measured_param(P) - measured_param(P_ref) | of every component
    of the 'control_params', where P_ref is the highest measured polynomial
    order. The columns are named like the ones of the 'Residuals' subfile.
    """
    P_ref = max(measured_params_by_order)
    ref = measured_params_by_order[P_ref]
    legend = ["PolynomialOrder"]
    for key in control_params:
        if key in ScalarQuantities:
            legend.append(key)
        else:
            legend.extend([f"{key}_{xyz}" for xyz in "xyz"])
    with spectre_h5.H5File(str(output_filename), "a") as output_file:
        dat_file = output_file.try_insert_dat(subfile_name, legend, 0)
        for P, measured_params in sorted(measured_params_by_order.items()):
            row = [float(P)]
            for key in control_params:
                row.extend(
                    np.abs(
                        np.atleast_1d(measured_params[key])
                        - np.atleast_1d(ref[key])
                    )
                )
            dat_file.append(row)


def _convergence_test(
    run_dir: Union[str, Path],
    test_dir: Union[str, Path],
    polynomial_order: int,
    target_params: Dict,
    free_data: Dict,
    separation: float,
    refinement_level: int,
    negative_expansion_bc: bool,
    control_params: List[TargetParams],
    tolerance: float,
    mode: Literal["initial", "final"],
    output_filename: Union[str, Path],
    max_polynomial_order: int = CONVERGENCE_TEST_MAX_POLYNOMIAL_ORDER,
) -> Optional[int]:
    """Test the convergence of the physical parameters with polynomial order

    Solves the initial data problem with the same 'free_data' at several
    polynomial orders P and compares the measured physical parameters to those
    at the highest measured P (see '_convergence_errors'). The optimal P is the
    smallest P whose convergence error is below the 'tolerance'.

    The 'run_dir' holds the solve at 'polynomial_order' that this test starts
    from. The solves at the other polynomial orders run in the 'test_dir', in
    directories named 'P{P:02d}'. The 'P' directory of the 'polynomial_order'
    is a link to the 'run_dir', so the 'test_dir' holds all solves of the test.
    Each solve writes its output to the 'spectre.out' file in its directory.
    The convergence error of every parameter component is written to the
    'output_filename' in a subfile named 'InitialConvergenceTest' or
    'FinalConvergenceTest', depending on the 'mode'.

    Modes:

    - "initial": selects the polynomial order for the control loop. Probes one
      P below the 'polynomial_order' and then climbs up one P at a time until
      an optimal P is found or the 'max_polynomial_order' is reached. The
      center of mass is recorded but doesn't enter the selection, because it
      converges slowly with P when it is far from zero (as before control).
      Returns the optimal P, or the 'max_polynomial_order' if none was found.

    - "final": checks the polynomial order that the control loop used. Solves
      at every P from 4 to max(8, polynomial_order + 2) and uses all
      'control_params' for the selection. The result is informational. Returns
      the optimal P, or 'None' if none was found.
    """
    is_initial = mode == "initial"
    label = "Initial" if is_initial else "Final"
    test_dir = Path(test_dir).resolve()
    test_dir.mkdir(parents=True, exist_ok=True)
    baseline_link = test_dir / f"P{polynomial_order:02d}"
    if not baseline_link.exists():
        baseline_link.symlink_to(
            os.path.relpath(Path(run_dir).resolve(), test_dir)
        )

    # The center of mass is excluded from the selection in the initial mode
    # (see docstring)
    selection_params = [
        key
        for key in control_params
        if not (is_initial and key == "CenterOfMass")
    ] or list(control_params)

    # Polynomial orders to solve at in addition to the 'polynomial_order'
    if is_initial:
        logger.info(
            f"{label} convergence test: start at P={polynomial_order}, up to"
            f" P={max_polynomial_order}, tolerance={tolerance:.2e}."
        )
        if polynomial_order > max_polynomial_order:
            logger.warning(
                f"{label} convergence test: P={polynomial_order} exceeds the"
                f" maximum P={max_polynomial_order}. Skipping the test."
            )
            return polynomial_order
        candidate_orders = (
            [polynomial_order - 1] if polynomial_order > 1 else []
        )
        candidate_orders += list(
            range(polynomial_order + 1, max_polynomial_order + 1)
        )
    else:
        min_order, max_order = 4, max(8, polynomial_order + 2)
        logger.info(
            f"{label} convergence test: control used P={polynomial_order},"
            f" solve at P={min_order}..{max_order},"
            f" tolerance={tolerance:.2e}."
        )
        candidate_orders = [
            P for P in range(min_order, max_order + 1) if P != polynomial_order
        ]

    try:
        measured_params_by_order = {
            polynomial_order: _measured_params(run_dir, control_params)
        }
    except Exception as e:
        logger.warning(
            f"{label} convergence test: could not read the parameters at"
            f" P={polynomial_order}: {e}"
        )
        return polynomial_order if is_initial else None

    optimal_order = None
    for P in candidate_orders:
        P_run_dir = test_dir / f"P{P:02d}"
        try:
            generate_id(
                target_params,
                **free_data,
                separation=separation,
                run_dir=P_run_dir,
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=P,
                negative_expansion_bc=negative_expansion_bc,
                redirect_output=True,
            )
            measured_params_by_order[P] = _measured_params(
                P_run_dir, control_params
            )
        except Exception as e:
            logger.warning(
                f"{label} convergence test: solve at P={P} failed: {e}"
            )
            continue
        errors = _convergence_errors(measured_params_by_order, selection_params)
        logger.info(
            f"{label} convergence test: solved at P={P}. Errors relative to"
            f" P={max(measured_params_by_order)}: "
            + ", ".join(
                f"P={P_i}: {error:.2e}" for P_i, error in errors.items()
            )
        )
        optimal_order = _optimal_polynomial_order(errors, tolerance)
        if is_initial and optimal_order is not None:
            break

    if len(measured_params_by_order) > 1:
        _write_convergence_test(
            output_filename,
            subfile_name=f"{label}ConvergenceTest",
            measured_params_by_order=measured_params_by_order,
            control_params=control_params,
        )

    measured_orders = sorted(measured_params_by_order)
    if is_initial:
        if optimal_order is not None:
            logger.info(
                f"{label} convergence test: selected P={optimal_order}"
                f" (solved at P={measured_orders})."
            )
            return optimal_order
        logger.warning(
            f"{label} convergence test: no P up to {max_polynomial_order} meets"
            f" the tolerance {tolerance:.2e}. Using P={max_polynomial_order}."
        )
        return max_polynomial_order

    if len(measured_params_by_order) < 2:
        logger.warning(
            f"{label} convergence test: not enough solves succeeded."
        )
    elif optimal_order is None:
        logger.warning(
            f"{label} convergence test: no P in {measured_orders} meets the"
            f" tolerance {tolerance:.2e}."
        )
    elif optimal_order == polynomial_order:
        logger.info(
            f"{label} convergence test: P={polynomial_order} is optimal."
        )
    else:
        logger.warning(
            f"{label} convergence test: control used P={polynomial_order}, but"
            f" P={optimal_order} is optimal."
        )
    return optimal_order


def control_id(
    id_input_file_path: Union[str, Path],
    control_params: List[TargetParams],
    id_run_dir: Optional[Union[str, Path]] = None,
    residual_tolerance: float = DEFAULT_RESIDUAL_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    control_delay: int = DEFAULT_CONTROL_DELAY,
    refinement_level: int = 1,
    polynomial_order: int = 6,
    negative_expansion_bc: bool = True,
    step_size_constraints: Sequence[
        StepSizeConstraint
    ] = DEFAULT_STEP_SIZE_CONSTRAINTS,
    run_convergence_tests: bool = False,
    convergence_test_tolerance: float = DEFAULT_CONVERGENCE_TEST_TOLERANCE,
):
    """Control BBH physical parameters.

    This function is called after initial data has been generated and horizons
    have been found in 'PostprocessId.py'. It uses an iterative scheme to drive
    the black hole physical parameters (e.g., masses and spins) closer to the
    desired values.

    For each iteration, this function does the following:

    - Determine new guesses for ID input parameters.

    - Generate initial data using these guesses and post-process it.

    - Compute the difference between the measured physical parameters and their
      desired values.

    Supported control parameters:
      MassA: Mass of the larger black hole.
      MassB: Mass of the smaller black hole.
      DimensionlessSpinA: Dimensionless spin of the larger black hole.
      DimensionlessSpinB: Dimensionless spin of the smaller black hole.
      CenterOfMass: Center of mass integral in general relativity.
      AdmLinearMomentum: ADM linear momentum.
      AdmMass: ADM mass / energy (useful for hyperbolic encounters).
      AdmAngularMomentumZ: ADM angular momentum along the z-axis (useful for
        hyperbolic encounters).

    A subset of these parameters can be chosen as the 'control_params'. The
    input file metadata must contain a 'TargetParams' dictionary with the
    corresponding target values.
    Example of control_params for an equal-mass non-spinning run with minimal
    drift of the center of mass:
    ```yaml
    TargetParams:
        MassA: 0.5
        MassB: 0.5
        DimensionlessSpinA: [0., 0., 0.]
        DimensionlessSpinB: [0., 0., 0.]
        CenterOfMass: [0., 0., 0.]
        AdmLinearMomentum: [0., 0., 0.]
    ```

    The free data is updated with Broyden's method, a quasi-Newton method,
    which is posed in non-dimensional variables (see 'control_scales').
    Diagnostic data is written to 'ControlParams.h5' in the initial-data
    directory: the residuals of the control parameters in 'Residuals.dat', the
    non-dimensional Jacobian in 'Jacobian.dat' (one row per iteration), and the
    scales of the free data and of the residuals, in this order, in
    'ControlScales.dat'. A Jacobian entry in raw units is the non-dimensional
    entry times the residual scale divided by the free-data scale.

    Each step of the free data can be shortened by step-size constraints, which
    scale the whole step by a factor alpha in [0, 1] so that its direction is
    preserved:

    - 'mass': the step of each conformal mass is at most 20% of its value.
    - 'com': the larger black hole stays at positive x, i.e., it does not cross
      the origin.
    - 'spin': the effective spin 2 rbar |Omega| of each black hole stays below
      1 - 1e-4, where the apparent-horizon boundary condition has solutions
      (see 'spin_step_size_constraint').

    The alpha of every constraint is computed, logged and written to
    'StepSizeConstraints.dat' in 'ControlParams.h5' in every iteration, along
    with the alpha that was applied and the effective spins after the step.
    Only the constraints listed in 'step_size_constraints' limit the step. If
    they allow no step at all (alpha = 0) the control loop stops with an error.

    Arguments:
      control_params: List of parameters to control.
      id_input_file_path: Path to the input file of the first initial data run.
      id_run_dir: Directory of the first initial data run. If not provided, the
        directory of the input file is used.
      residual_tolerance: Residual tolerance used for termination condition.
        (Default: 1.e-4)
      max_iterations: Maximum of iterations allowed. Note: each iteration is
        very expensive as it needs to solve an entire initial data problem.
        (Default: 30)
      control_delay: Minimum number of iterations before control of delayed
        parameters starts. We have found that delaying the control of
        asymptotic quantities (ADM mass, ADM momenta, and center of mass) helps
        convergence. In addition, their control starts only once the maximum
        residual of the horizon masses and spins is below
        'HORIZON_RESIDUAL_FOR_ASYMPTOTIC_CONTROL' (5e-3). (Default: 2)
      refinement_level: h-refinement used in control loop.
      polynomial_order: p-refinement used in control loop.
      negative_expansion_bc: Place the excisions inside of apparent horizons.
      step_size_constraints: Step-size constraints to enforce, any of 'mass',
        'com' and 'spin' (see above). All of them are computed and reported
        regardless. (Default: ['spin'])
      run_convergence_tests: Run resolution convergence tests. A test before
        the control loop selects the polynomial order for the control loop,
        and a test after the control loop checks that it was appropriate. The
        solves of the tests run in the 'ConvergenceTest/Initial' and
        'ConvergenceTest/Final' directories next to the control-loop runs, and
        the convergence errors are written to 'ControlParams.h5'. See
        '_convergence_test' for details. (Default: False)
      convergence_test_tolerance: Tolerance of the convergence tests.
        (Default: 1e-5)
    """

    assert (
        len(control_params) > 0
    ), "At least one control parameter must be specified."
    for constraint in step_size_constraints:
        assert constraint in STEP_SIZE_CONSTRAINTS, (
            f"Unknown step-size constraint '{constraint}'. Choose from"
            f" {STEP_SIZE_CONSTRAINTS}."
        )

    # Read input file
    if id_run_dir is None:
        id_run_dir = Path(id_input_file_path).resolve().parent
    with open(id_input_file_path, "r") as open_input_file:
        id_metadata, id_input_file = yaml.safe_load_all(open_input_file)
    target_params = id_metadata["TargetParams"]
    binary_data = id_input_file["Background"]["Binary"]
    domain_data = id_input_file["DomainCreator"]["BinaryCompactObject"]

    # Get initial xyz offset
    # Note: CenterOfMassOffset contains only the yz offsets, so we need to get
    # the x offset from XCoords
    x_B, x_A = binary_data["XCoords"]
    separation = x_A - x_B
    Newtonian_x_A = (
        target_params["MassB"]
        / (target_params["MassA"] + target_params["MassB"])
        * separation
    )
    x_offset = x_A - Newtonian_x_A
    y_offset, z_offset = binary_data["CenterOfMassOffset"]

    # Get initial horizon rotations
    orbital_angular_velocity = binary_data["AngularVelocity"]
    horizon_rotation_a = domain_data["ObjectA"]["Interior"][
        "ExciseWithBoundaryCondition"
    ]["ApparentHorizon"]["Rotation"]
    horizon_rotation_a[2] -= orbital_angular_velocity
    horizon_rotation_b = domain_data["ObjectB"]["Interior"][
        "ExciseWithBoundaryCondition"
    ]["ApparentHorizon"]["Rotation"]
    horizon_rotation_b[2] -= orbital_angular_velocity

    # Combine initial choices of free data in a dictionary
    initial_free_data = dict(
        conformal_mass_a=binary_data["ObjectRight"]["KerrSchild"]["Mass"],
        conformal_mass_b=binary_data["ObjectLeft"]["KerrSchild"]["Mass"],
        horizon_rotation_a=horizon_rotation_a,
        horizon_rotation_b=horizon_rotation_b,
        center_of_mass_offset=[x_offset, y_offset, z_offset],
        linear_velocity=binary_data["LinearVelocity"],
        radial_expansion_velocity=binary_data["Expansion"],
        orbital_angular_velocity=orbital_angular_velocity,
    )

    # Prepare file and legends to output diagnostic data. Every run of the
    # control loop is a segment in the initial-data directory, so the diagnostic
    # data goes next to them.
    id_dir = Path(id_run_dir).resolve().parent
    output_filename = str(id_dir / "ControlParams.h5")
    residual_legend = []
    free_data_legend = []
    jacobian_legend = []
    step_size_legend = [
        "MassAlpha",
        "CenterOfMassAlpha",
        "SpinAlpha",
        "AppliedAlpha",
        "EffectiveSpinA",
        "EffectiveSpinB",
    ]
    for param in control_params:
        free_data = FreeDataFromParams[param]
        if param in ScalarQuantities:
            residual_legend.append(param)
            free_data_legend.append(free_data)
        else:
            residual_legend.extend([f"{param}_{xyz}" for xyz in "xyz"])
            free_data_legend.extend([f"{free_data}_{xyz}" for xyz in "xyz"])
    for residual in residual_legend:
        for free_data in [
            FreeDataFromParams[param] for param in control_params
        ]:
            if free_data in ScalarQuantities:
                jacobian_legend.append(f"d{residual} / d{free_data}")
            else:
                jacobian_legend.extend(
                    [f"d{residual} / d{free_data}_{xyz}" for xyz in "xyz"]
                )

    iteration = 0
    control_run_dir = id_run_dir

    # Function to be minimized
    def Residual(u):
        nonlocal iteration
        nonlocal control_run_dir

        if iteration > 0:
            logger.info(
                "\n"
                "=========================================="
                f" Control of BBH Parameters ({iteration}) "
                "=========================================="
            )
            # Start with initial free data choices and update the ones being
            # controlled in `control_params` with the numeric value from `u`
            free_data = initial_free_data.copy()
            u_iterator = iter(u)
            for key in [FreeDataFromParams[param] for param in control_params]:
                if key in ScalarQuantities:
                    free_data[key] = next(u_iterator)
                else:
                    free_data[key] = [next(u_iterator) for _ in range(3)]

            # Run ID and find horizons. The run continues the sequence of
            # initial-data runs in the 'id_dir', like any other run.
            generate_id(
                target_params,
                **free_data,
                separation=separation,
                segments_dir=id_dir,
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=polynomial_order,
                negative_expansion_bc=negative_expansion_bc,
            )
            control_run_dir = str(Segment.last(id_dir).path)

        # Get measured physical parameters
        measured_params = _measured_params(control_run_dir, control_params)

        # Compute residual of physical parameters
        residual = np.array([])
        for key in control_params:
            target = target_params[key]
            assert target is not None, (
                f"Attempting to control parameter '{key}' but no target value"
                " is provided."
            )
            if key in ScalarQuantities:
                residual = np.append(residual, [measured_params[key] - target])
            else:
                residual = np.append(residual, measured_params[key] - target)
        logger.info(f"Control Residual = {np.max(np.abs(residual)):e}")
        with spectre_h5.H5File(output_filename, "a") as output_file:
            dat_file = output_file.try_insert_dat(
                "Residuals", residual_legend, 0
            )
            dat_file.append(residual)

        return residual

    # Initial guess for free data
    u = np.array([])
    for key in [FreeDataFromParams[param] for param in control_params]:
        if key in ScalarQuantities:
            u = np.append(u, [initial_free_data[key]])
        else:
            u = np.append(u, initial_free_data[key])

    # Initial residual
    F = Residual(u)

    # Select the polynomial order for the control loop
    if run_convergence_tests:
        polynomial_order = _convergence_test(
            run_dir=control_run_dir,
            test_dir=id_dir / "ConvergenceTest" / "Initial",
            polynomial_order=polynomial_order,
            target_params=target_params,
            free_data=initial_free_data.copy(),
            separation=separation,
            refinement_level=refinement_level,
            negative_expansion_bc=negative_expansion_bc,
            control_params=control_params,
            tolerance=convergence_test_tolerance,
            mode="initial",
            output_filename=output_filename,
        )

    # Initialize Jacobian as an identity matrix
    J = np.identity(len(u))

    # Prepare map between parameters and their indices so that we can specify
    # Jacobian terms below
    param_index_map = dict()
    param_index = 0
    for param in control_params:
        param_index_map[param] = param_index
        param_index_map[FreeDataFromParams[param]] = param_index
        param_index += 1 if param in ScalarQuantities else 3

    # Adjust non-unity components of the Jacobian
    #
    # The expressions below come from differentiating the Kerr expressions
    # chi = - 2 r Omega and r = M (1 + sqrt(1 - chi^2)), where chi is the
    # dimensionless spin, r is the horizon radius, Omega is the horizon
    # rotation, and M is the mass.
    for spin_key, mass_key, horizon_rotation in zip(
        ["DimensionlessSpinA", "DimensionlessSpinB"],
        ["conformal_mass_a", "conformal_mass_b"],
        [horizon_rotation_a, horizon_rotation_b],
    ):
        conformal_mass = u[param_index_map[mass_key]]
        conformal_spin = target_params[spin_key]
        spin_term = 1.0 + np.sqrt(1 - np.dot(conformal_spin, conformal_spin))
        for i in range(3):
            J[
                param_index_map[spin_key] + i,
                param_index_map[mass_key],
            ] = (
                -2.0 * horizon_rotation[i] * spin_term
            )
            J[
                param_index_map[spin_key] + i,
                param_index_map[spin_key] + i,
            ] = (
                -2.0 * conformal_mass * spin_term
            )
    # The expression below is the reduced mass of the system, which shows up in
    # the Newtonian expressions further below.
    q = target_params["MassRatio"]
    eta = q / (q + 1) ** 2
    # The expressions below come from differentiating the Newtonian
    # approximation E_ADM ~ M + 1/2 eta adot0^2 D0^2, where adot0 is the
    # initial radial expansion velocity and D0 is the initial
    # separation. They are evaluated under the conditions M=1 and q=MA/MB, which
    # holds (at least approximately) for the initial guess of the free data.
    if "AdmMass" in control_params:
        adot0 = initial_free_data["radial_expansion_velocity"]
        J[
            param_index_map["AdmMass"],
            param_index_map["radial_expansion_velocity"],
        ] = (
            eta * adot0 * separation**2
        )
        if "MassA" in control_params:
            J[
                param_index_map["AdmMass"], param_index_map["conformal_mass_a"]
            ] = (1.0 + 0.5 * eta / q * adot0**2 * separation**2)
        if "MassB" in control_params:
            J[
                param_index_map["AdmMass"], param_index_map["conformal_mass_b"]
            ] = (1.0 + 0.5 * q * eta * adot0**2 * separation**2)
    # The expressions below come from differentiating the Newtonian
    # approximation J_ADM ~ eta D0^2 Omega0, where D0 is the initial
    # separation and Omega0 is the initial angular orbital velocity.
    if "AdmAngularMomentumZ" in control_params:
        OmegaZ0 = initial_free_data["orbital_angular_velocity"]
        J[
            param_index_map["AdmAngularMomentumZ"],
            param_index_map["orbital_angular_velocity"],
        ] = (
            eta * separation**2
        )
        if "MassA" in control_params:
            J[
                param_index_map["AdmAngularMomentumZ"],
                param_index_map["conformal_mass_a"],
            ] = (
                eta / q * separation**2 * OmegaZ0
            )
        if "MassB" in control_params:
            J[
                param_index_map["AdmAngularMomentumZ"],
                param_index_map["conformal_mass_b"],
            ] = (
                q * eta * separation**2 * OmegaZ0
            )

    # Non-dimensionalize the Newton system. From here on J is the Jacobian of
    # the scaled residual F / f_scale with respect to the scaled free data
    # u / u_scale, so that all of its entries are O(1); see 'control_scales'.
    # Only the linear algebra (the Newton step and the Broyden update) works in
    # scaled variables. Everything else keeps working in raw units.
    u_scale, f_scale = control_scales(
        control_params, param_index_map, target_params, u, len(u)
    )
    J *= u_scale[np.newaxis, :] / f_scale[:, np.newaxis]
    with spectre_h5.H5File(output_filename, "a") as output_file:
        dat_file = output_file.try_insert_dat(
            "ControlScales", free_data_legend + residual_legend, 0
        )
        dat_file.append(np.concatenate([u_scale, f_scale]))
        output_file.close_current_object()
        dat_file = output_file.try_insert_dat("Jacobian", jacobian_legend, 0)
        dat_file.append(J.flatten())

    # Indices of parameters for which the control is delayed in the first
    # iterations, until the horizon parameters have settled, to avoid going
    # off-bounds
    #
    # Note: We have experimented with other modifications to Broyden's
    # method, including damping the initial updates of the free data / Jacobian
    # and enforcing a diagonal Jacobian. None of them converged as fast as the
    # delay approach used here. When doing a more complete study in parameter
    # space, we should try to find a more robust approach that works for
    # multiple configurations.
    delayed_indices = np.array([], dtype=bool)
    delayed_params = [
        "CenterOfMass",
        "AdmLinearMomentum",
        "AdmMass",
        "AdmAngularMomentumZ",
    ]
    for key in control_params:
        if key in ScalarQuantities:
            delayed_indices = np.append(
                delayed_indices, [key in delayed_params]
            )
        else:
            delayed_indices = np.append(
                delayed_indices, [key in delayed_params] * 3
            )
    # Indices of the horizon parameters, whose residual decides when the control
    # of the delayed parameters starts
    horizon_indices = np.array([], dtype=bool)
    for key in control_params:
        is_horizon_param = key in [
            "MassA",
            "MassB",
            "DimensionlessSpinA",
            "DimensionlessSpinB",
        ]
        if key in ScalarQuantities:
            horizon_indices = np.append(horizon_indices, [is_horizon_param])
        else:
            horizon_indices = np.append(horizon_indices, [is_horizon_param] * 3)
    delay_control = np.any(delayed_indices)

    def horizon_free_data(u, Delta_u):
        """Conformal mass, horizon rotation, and their steps for each black
        hole, along with its conformal spin. Free data that is not controlled
        keeps its initial value."""
        result = []
        for mass_key, spin_key in zip(
            ["MassA", "MassB"], ["DimensionlessSpinA", "DimensionlessSpinB"]
        ):
            if mass_key in control_params:
                idx = param_index_map[mass_key]
                mass, delta_mass = u[idx], Delta_u[idx]
            else:
                mass = initial_free_data[FreeDataFromParams[mass_key]]
                delta_mass = 0.0
            if spin_key in control_params:
                idx = param_index_map[spin_key]
                rotation = u[idx : idx + 3]
                delta_rotation = Delta_u[idx : idx + 3]
            else:
                rotation = initial_free_data[FreeDataFromParams[spin_key]]
                delta_rotation = np.zeros(3)
            result.append(
                (
                    mass,
                    delta_mass,
                    rotation,
                    delta_rotation,
                    target_params[spin_key],
                )
            )
        return result

    def step_size_alphas(u, Delta_u):
        """The alpha of every step-size constraint, whether it is enforced or
        not, for the step 'Delta_u' from the free data 'u'."""
        alphas = {constraint: 1.0 for constraint in STEP_SIZE_CONSTRAINTS}
        # Step of each conformal mass is at most MAX_RELATIVE_MASS_STEP times
        # its current value
        for mass_key in ["MassA", "MassB"]:
            if mass_key not in control_params:
                continue
            idx = param_index_map[mass_key]
            max_delta = MAX_RELATIVE_MASS_STEP * abs(u[idx])
            if abs(Delta_u[idx]) > max_delta:
                alphas["mass"] = min(
                    alphas["mass"], max_delta / abs(Delta_u[idx])
                )
        # The larger black hole stays at x_A > MIN_X_A
        if "CenterOfMass" in control_params:
            idx = param_index_map["center_of_mass_offset"]
            x_A = Newtonian_x_A + u[idx]
            if x_A + Delta_u[idx] < MIN_X_A:
                alphas["com"] = float(
                    np.clip((MIN_X_A - x_A) / Delta_u[idx], 0.0, 1.0)
                )
        # Effective spin of each black hole stays below MAX_EFFECTIVE_SPIN
        for (
            mass,
            delta_mass,
            rotation,
            delta_rotation,
            conformal_spin,
        ) in horizon_free_data(u, Delta_u):
            if delta_mass == 0.0 and not np.any(delta_rotation):
                continue
            alphas["spin"] = min(
                alphas["spin"],
                spin_step_size_constraint(
                    mass, delta_mass, rotation, delta_rotation, conformal_spin
                ),
            )
        return alphas

    while iteration < max_iterations:
        iteration += 1

        # Start the control of the delayed parameters once a minimum number of
        # iterations has passed and the horizon parameters have settled.
        # Controlling the asymptotic quantities while the horizon masses and
        # spins are still far from their targets can drive the free data
        # off-bounds.
        if delay_control:
            max_horizon_residual = (
                np.max(np.abs(F[horizon_indices]))
                if np.any(horizon_indices)
                else 0.0
            )
            if (
                iteration >= control_delay
                and max_horizon_residual
                < HORIZON_RESIDUAL_FOR_ASYMPTOTIC_CONTROL
            ):
                delay_control = False
            logger.info(
                "Max residual of horizon parameters ="
                f" {max_horizon_residual:e}."
                f" {'Delaying' if delay_control else 'Starting'} control of"
                " asymptotic parameters."
            )

        # Update the free parameters using a quasi-Newton-Raphson method
        Delta_u = -np.dot(np.linalg.inv(J), F / f_scale) * u_scale
        if delay_control:
            Delta_u[delayed_indices] = 0.0

        # Shorten the step to satisfy the enforced step-size constraints. The
        # whole step is scaled by the same alpha to preserve its direction.
        alphas = step_size_alphas(u, Delta_u)
        alpha = min(
            [1.0] + [alphas[constraint] for constraint in step_size_constraints]
        )
        binding = [
            constraint
            for constraint in step_size_constraints
            if alphas[constraint] <= alpha < 1.0
        ]
        logger.info(
            "Step-size constraints: "
            + ", ".join(
                f"{constraint} alpha = {alphas[constraint]:g}"
                + (" (enforced)" if constraint in step_size_constraints else "")
                for constraint in STEP_SIZE_CONSTRAINTS
            )
            + f". Applied alpha = {alpha:g}."
        )
        Delta_u *= alpha
        effective_spins_after_step = [
            effective_spin(mass, rotation, conformal_spin)
            for mass, _, rotation, _, conformal_spin in horizon_free_data(
                u + Delta_u, Delta_u
            )
        ]
        if 0.0 < alpha < 1.0:
            logger.warning(
                f"Step of the free data limited by {binding} to alpha ="
                f" {alpha:g}. Effective spins after the step:"
                f" {effective_spins_after_step}."
            )
        with spectre_h5.H5File(output_filename, "a") as output_file:
            dat_file = output_file.try_insert_dat(
                "StepSizeConstraints", step_size_legend, 0
            )
            dat_file.append(
                [alphas[constraint] for constraint in STEP_SIZE_CONSTRAINTS]
                + [alpha]
                + effective_spins_after_step
            )
        if alpha <= 0.0:
            raise RuntimeError(
                f"The step-size constraints {binding} allow no step of the"
                " free data (alpha = 0), so the control loop cannot make"
                f" progress. Step-size constraints: {alphas}."
            )

        u += Delta_u

        # Compute residual and check stopping condition
        F = Residual(u)
        if np.max(np.abs(F)) < residual_tolerance:
            break
        if delay_control:
            F[delayed_indices] = 0.0

        # Update the Jacobian using Broyden's method, in scaled variables so
        # that no single control can dominate the update
        scaled_Delta_u = Delta_u / u_scale
        J += np.outer(F / f_scale, scaled_Delta_u) / np.dot(
            scaled_Delta_u, scaled_Delta_u
        )
        with spectre_h5.H5File(output_filename, "a") as output_file:
            dat_file = output_file.try_insert_dat(
                "Jacobian", jacobian_legend, 0
            )
            dat_file.append(J.flatten())

    # Check the polynomial order that the control loop used
    if run_convergence_tests:
        final_free_data = initial_free_data.copy()
        u_iterator = iter(u)
        for key in [FreeDataFromParams[param] for param in control_params]:
            if key in ScalarQuantities:
                final_free_data[key] = next(u_iterator)
            else:
                final_free_data[key] = [next(u_iterator) for _ in range(3)]
        _convergence_test(
            run_dir=control_run_dir,
            test_dir=id_dir / "ConvergenceTest" / "Final",
            polynomial_order=polynomial_order,
            target_params=target_params,
            free_data=final_free_data,
            separation=separation,
            refinement_level=refinement_level,
            negative_expansion_bc=negative_expansion_bc,
            control_params=control_params,
            tolerance=convergence_test_tolerance,
            mode="final",
            output_filename=output_filename,
        )

    return control_run_dir
