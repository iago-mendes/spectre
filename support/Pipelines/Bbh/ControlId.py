# Distributed under the MIT License.
# See LICENSE.txt for details.

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import yaml

import spectre.IO.H5 as spectre_h5
from spectre.IO.H5 import to_dataframe
from spectre.Pipelines.Bbh.InitialData import TargetParams, generate_id

logger = logging.getLogger(__name__)

# The default tolerance is just below the effect of junk radiation on the
# controlled paramaters, which  is about 1.0e-3.
DEFAULT_RESIDUAL_TOLERANCE = 1.0e-4
DEFAULT_MAX_ITERATIONS = 30
DEFAULT_CONTROL_DELAY = 2
DEFAULT_CONVERGENCE_TEST_TOLERANCE = 1.0e-5
P_MAX_CONVERGENCE = 12

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


def _measured_params(
    run_dir: Union[str, Path],
    control_params: List[TargetParams],
) -> Dict[TargetParams, Union[float, np.ndarray]]:
    """Measured physical parameters from a completed ID run as a dict.

    Reads Horizons.h5 and BbhReductions.h5 from `run_dir` and returns a
    dict keyed by control parameter name, containing the measured value
    (scalar or length-3 array) for each entry in `control_params`.
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


def _convergence_test(
    run_dir: Union[str, Path],
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
    max_polynomial_order: int = P_MAX_CONVERGENCE,
) -> Optional[int]:
    """Run a polynomial-order convergence test.

    For each evaluated polynomial order P, the convergence error of a
    parameter is

        error_param(P) = max | measured_param(P) - measured_param(P_ref) |

    where P_ref is the highest evaluated order (which therefore has zero
    error). The optimal P is the smallest P < P_ref whose error is below
    `tolerance` for every parameter included in the selection criterion
    (see below).

    Per-component convergence errors at every evaluated P are appended to
    a dat subfile in `output_filename` named "InitialConvergenceTest" or
    "FinalConvergenceTest" depending on `mode`. The legend mirrors the one
    used for the control Residuals.dat: one column per scalar parameter
    and three columns (`<param>_x`, `<param>_y`, `<param>_z`) per vector
    parameter, plus a leading "PolynomialOrder" column.

    Modes:

    - "initial": Pre-control test that selects the polynomial order to use
      in the control loop. Reads the baseline params from `run_dir`
      (assumed to contain an iter-0 solve at `polynomial_order`), probes
      one P below, and climbs upward one P at a time until an optimal P
      is found or `max_polynomial_order` is reached. CenterOfMass is
      excluded from the selection criterion here: its P-convergence is
      slow when the CoM is far from zero (as at iter-0 before control),
      so including it tends to push the test all the way to P_max. The
      CoM error is still recorded in the diagnostics. Returns the
      selected P.

    - "final": Post-control test that validates the polynomial order used
      by the control loop. Sweeps P from 4 to max(8, polynomial_order+2),
      reusing the supplied free data, and includes all controlled
      parameters in the selection criterion. Results are informational
      only. Returns the optimal P if found, else None.
    """
    conv_dir = Path(run_dir) / "ConvergenceTest"
    conv_dir.mkdir(exist_ok=True)
    P_baseline_dir = conv_dir / f"P{polynomial_order:02d}"
    if not P_baseline_dir.exists():
        P_baseline_dir.symlink_to("..")

    is_initial = mode == "initial"
    label = "Initial" if is_initial else "Final"

    # Parameters that gate the choice of optimal P. CenterOfMass is
    # excluded in initial mode (see docstring); we still record its
    # convergence error in the diagnostics.
    selection_params = [
        k for k in control_params if not (is_initial and k == "CenterOfMass")
    ]
    if not selection_params:
        logger.warning(
            f"{label} convergence test: no control params remain in the"
            " selection criterion; falling back to all control_params."
        )
        selection_params = list(control_params)

    # Polynomial orders to probe beyond the baseline. The initial mode
    # probes P-1 first and then climbs upward one P at a time, breaking
    # out as soon as an optimal P is found. The final mode sweeps a fixed
    # range around the control P.
    if is_initial:
        logger.info(
            f"{label} convergence test: baseline P={polynomial_order},"
            f" P_max={max_polynomial_order}, tolerance={tolerance:.2e}"
        )
        if polynomial_order > max_polynomial_order:
            logger.warning(
                f"{label} convergence test: polynomial_order={polynomial_order}"
                f" exceeds max_polynomial_order={max_polynomial_order}."
                f" Skipping test, returning P={polynomial_order}."
            )
            return polynomial_order
        P_candidates: List[int] = []
        if polynomial_order >= 2:
            P_candidates.append(polynomial_order - 1)
        P_candidates.extend(
            range(polynomial_order + 1, max_polynomial_order + 1)
        )
    else:
        P_min = 4
        P_max = max(8, polynomial_order + 2)
        logger.info(
            f"{label} convergence test: P_control={polynomial_order},"
            f" sweep P={P_min}..{P_max}, tolerance={tolerance:.2e}"
        )
        P_candidates = [
            p for p in range(P_min, P_max + 1) if p != polynomial_order
        ]

    # Seed with the baseline solve already present in run_dir.
    try:
        measured_params_by_order: Dict[
            int, Dict[TargetParams, Union[float, np.ndarray]]
        ] = {polynomial_order: _measured_params(run_dir, control_params)}
    except Exception as e:
        logger.warning(
            f"{label} convergence test: could not read baseline params at"
            f" P={polynomial_order}: {e}."
        )
        return polynomial_order if is_initial else None

    # Run additional ID solves at each candidate P. After every successful
    # solve, recompute the optimal P as the smallest P below the current
    # reference order whose error in each selection parameter is below
    # tolerance. The initial mode breaks out as soon as that P is found
    # (so the climb stops at the first acceptable P); the final mode runs
    # all P in the sweep before settling on the optimal.
    P_optimal: Optional[int] = None
    for p in P_candidates:
        P_run_dir = conv_dir / f"P{p:02d}"
        P_run_dir.mkdir(exist_ok=True)
        try:
            # Each sub-solve writes its own spectre.out under P_run_dir so
            # the main log stays focused on the per-P summary lines below.
            generate_id(
                target_params,
                **free_data,
                separation=separation,
                run_dir=str(P_run_dir),
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=p,
                negative_expansion_bc=negative_expansion_bc,
                redirect_output=True,
            )
            measured_params_by_order[p] = _measured_params(
                str(P_run_dir), control_params
            )
        except Exception as e:
            logger.warning(
                f"{label} convergence test solve at P={p} failed: {e}"
            )
            continue

        P_ref = max(measured_params_by_order)
        ref = measured_params_by_order[P_ref]

        # Selection errors of every measured P (below P_ref) against the
        # current reference. Reused for both logging and the optimal-P
        # check below.
        errors_vs_ref: Dict[int, float] = {}
        for p_other in sorted(measured_params_by_order):
            if p_other == P_ref:
                continue
            errors_vs_ref[p_other] = max(
                float(
                    np.max(
                        np.abs(
                            np.atleast_1d(measured_params_by_order[p_other][k])
                            - np.atleast_1d(ref[k])
                        )
                    )
                )
                for k in selection_params
            )

        if p == P_ref:
            prev_errors = ", ".join(
                f"P={p_prev}: {errors_vs_ref[p_prev]:.2e}"
                for p_prev in sorted(errors_vs_ref)
            )
            logger.info(
                f"{label} convergence test: solved P={p} (new reference)."
                f" Selection errors vs new ref: {prev_errors}."
            )
        else:
            logger.info(
                f"{label} convergence test: solved P={p}, selection error"
                f" vs ref P={P_ref}: {errors_vs_ref[p]:.2e}."
            )

        P_optimal = None
        for p_candidate in sorted(errors_vs_ref):
            if errors_vs_ref[p_candidate] < tolerance:
                P_optimal = p_candidate
                break
        if is_initial and P_optimal is not None:
            break

    # Write per-parameter convergence errors to the diagnostics file.
    if len(measured_params_by_order) >= 2:
        P_ref = max(measured_params_by_order)
        ref = measured_params_by_order[P_ref]
        subfile_name = (
            "InitialConvergenceTest" if is_initial else "FinalConvergenceTest"
        )
        legend = ["PolynomialOrder"]
        for k in control_params:
            if k in ScalarQuantities:
                legend.append(k)
            else:
                legend.extend(f"{k}_{xyz}" for xyz in "xyz")
        with spectre_h5.H5File(str(output_filename), "a") as output_file:
            dat_file = output_file.try_insert_dat(subfile_name, legend, 0)
            for p in sorted(measured_params_by_order):
                row = [float(p)]
                for k in control_params:
                    diff = np.atleast_1d(
                        measured_params_by_order[p][k]
                    ) - np.atleast_1d(ref[k])
                    row.extend(float(np.abs(d)) for d in diff)
                dat_file.append(row)

    measured_orders = sorted(measured_params_by_order)
    if is_initial:
        if P_optimal is not None:
            logger.info(
                f"{label} convergence test: selected P={P_optimal}"
                f" (measurements at {measured_orders})."
            )
            return P_optimal
        logger.warning(
            f"{label} convergence test: reached P_max={max_polynomial_order}"
            f" without finding P meeting tolerance={tolerance:.2e}."
            f" Using P_max={max_polynomial_order}."
        )
        return max_polynomial_order

    # Final mode: log whether the chosen polynomial_order was optimal.
    if len(measured_params_by_order) < 2:
        logger.warning(
            f"{label} convergence test: not enough solves succeeded."
        )
        return None
    if P_optimal is None:
        logger.warning(
            f"{label} convergence test: no P in {measured_orders}"
            f" meets tolerance={tolerance:.2e}."
        )
    elif P_optimal == polynomial_order:
        logger.info(
            f"{label} convergence test: P={polynomial_order} confirmed optimal."
        )
    else:
        logger.warning(
            f"{label} convergence test: control ran at P={polynomial_order},"
            f" optimal would have been P={P_optimal}."
        )
    return P_optimal


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
      control_delay: Number of iterations before control of delayed parameters
        starts. We have found that delaying the control of asymptotic quantities
        (ADM mass, ADM momenta, and center of mass) helps convergence.
        (Default: 2)
      refinement_level: h-refinement used in control loop.
      polynomial_order: p-refinement used in control loop.
      negative_expansion_bc: Place the excisions inside of apparent horizons.
      run_convergence_tests: Run resolution convergence tests. An iter-0 test
        selects the polynomial order for the control loop, and a post-control
        test checks that the selected resolution was appropriate. (Default:
        False)
      convergence_test_tolerance: Tolerance for convergence tests. (Default:
        1e-5)
    """

    assert (
        len(control_params) > 0
    ), "At least one control parameter must be specified."

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

    # Prepare file and legends to output diagnostic data
    output_filename = f"{id_run_dir}/../ControlParams.h5"
    residual_legend = []
    jacobian_legend = []
    for param in control_params:
        if param in ScalarQuantities:
            residual_legend.append(param)
        else:
            residual_legend.extend([f"{param}_{xyz}" for xyz in "xyz"])
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
            control_run_dir = f"{id_run_dir}/../ControlParams_{iteration:03}"

            # Start with initial free data choices and update the ones being
            # controlled in `control_params` with the numeric value from `u`
            free_data = initial_free_data.copy()
            u_iterator = iter(u)
            for key in [FreeDataFromParams[param] for param in control_params]:
                if key in ScalarQuantities:
                    free_data[key] = next(u_iterator)
                else:
                    free_data[key] = [next(u_iterator) for _ in range(3)]

            # Run ID and find horizons
            generate_id(
                target_params,
                **free_data,
                separation=separation,
                run_dir=control_run_dir,
                control=False,
                evolve=False,
                scheduler=None,
                refinement_level=refinement_level,
                polynomial_order=polynomial_order,
                negative_expansion_bc=negative_expansion_bc,
            )

        # Get measured physical parameters and compute residual
        measured_params = _measured_params(control_run_dir, control_params)
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
                residual = np.append(
                    residual,
                    np.asarray(measured_params[key]) - np.asarray(target),
                )
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
    best_residual = np.max(np.abs(F))
    best_iteration = 0

    # Select polynomial order via iter-0 convergence test
    if run_convergence_tests:
        polynomial_order = _convergence_test(
            run_dir=control_run_dir,
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
    with spectre_h5.H5File(output_filename, "a") as output_file:
        dat_file = output_file.try_insert_dat("Jacobian", jacobian_legend, 0)
        dat_file.append(J.flatten())

    # Indices of parameters for which the control is delayed until the horizon
    # parameters have converged, to avoid going off-bounds
    #
    # Note: We have experimented with other modifications to Broyden's
    # method, including damping the initial updates of the free data / Jacobian
    # and enforcing a diagonal Jacobian. None of them converged as fast as the
    # delay approach used here. When doing a more complete study in parameter
    # space, we should try to find a more robust approach that works for
    # multiple configurations.
    delay_asymptotic_control = True
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

    while iteration < max_iterations:
        iteration += 1

        max_horizon_residual = np.max(
            np.abs(
                np.array(
                    [
                        F[param_index_map["MassA"]],
                        F[param_index_map["MassB"]],
                        F[param_index_map["DimensionlessSpinA"] + 0],
                        F[param_index_map["DimensionlessSpinA"] + 1],
                        F[param_index_map["DimensionlessSpinA"] + 2],
                        F[param_index_map["DimensionlessSpinB"] + 0],
                        F[param_index_map["DimensionlessSpinB"] + 1],
                        F[param_index_map["DimensionlessSpinB"] + 2],
                    ]
                )
            )
        )
        if max_horizon_residual < 5.0e-3:
            delay_asymptotic_control = False
        logger.info(
            f"Max residual of horizon parameters = {max_horizon_residual:e}."
            f" {'Delaying' if delay_asymptotic_control else 'Not delaying'}"
            " control of asymptotic parameters."
        )

        # Update the free parameters using a quasi-Newton-Raphson method
        Delta_u = -np.dot(np.linalg.inv(J), F)
        if delay_asymptotic_control:
            Delta_u[delayed_indices] = 0.0

        # Compute the largest uniform scale factor alpha in (0,1] such that
        # all constraints are satisfied, then apply it to the full step. This
        # preserves the Broyden-learned coupling direction; only the magnitude
        # is reduced. Each constraint provides an independent analytical alpha;
        # the minimum is taken.
        alpha = 1.0
        limiting_constraints = []

        # Constraint 1: mass step <= max_rel_step * current_mass (linear)
        max_rel_step = 0.2
        for mass_key in ["MassA", "MassB"]:
            if mass_key not in control_params:
                continue
            idx = param_index_map[mass_key]
            if abs(Delta_u[idx]) > max_rel_step * abs(u[idx]):
                alpha_mass = max_rel_step * abs(u[idx]) / abs(Delta_u[idx])
                if alpha_mass < alpha:
                    alpha = alpha_mass
                    limiting_constraints = [mass_key]
                elif alpha_mass == alpha:
                    limiting_constraints.append(mass_key)

        # Constraint 2: xA = Newtonian_x_A + center_of_mass_offset > 0 (linear)
        if "CenterOfMass" in control_params:
            prev_xA = (
                Newtonian_x_A + u[param_index_map["center_of_mass_offset"]]
            )
            delta_xA = Delta_u[param_index_map["center_of_mass_offset"]]
            eps_xa = 1.0e-10
            if prev_xA + delta_xA < eps_xa:
                alpha_xa = (eps_xa - prev_xA) / delta_xA
                if alpha_xa < alpha:
                    alpha = alpha_xa
                    limiting_constraints = ["CenterOfMass"]
                elif alpha_xa == alpha:
                    limiting_constraints.append("CenterOfMass")

        # Constraint 3: effective spin < 1 (quadratic, per BH)
        # ||eff_spin|| = 2 * S * M * ||omega|| where S = 1 + sqrt(1 - chi^2).
        # With the mass scaled by the alpha already computed above, solve for
        # the largest alpha_rot on the rotation such that the product stays
        # below 1 - eps_spin. This is a quadratic in alpha_rot.
        eps_spin = 1.0e-4
        for mass_key, spin_key in zip(
            ["MassA", "MassB"],
            ["DimensionlessSpinA", "DimensionlessSpinB"],
        ):
            if mass_key not in control_params or spin_key not in control_params:
                continue
            proposed_mass = (
                u[param_index_map[mass_key]]
                + alpha * Delta_u[param_index_map[mass_key]]
            )
            conformal_spin = target_params[spin_key]
            spin_term = 1.0 + np.sqrt(
                1.0 - np.dot(conformal_spin, conformal_spin)
            )
            omega_old = u[
                param_index_map[spin_key] : param_index_map[spin_key] + 3
            ]
            d_omega = Delta_u[
                param_index_map[spin_key] : param_index_map[spin_key] + 3
            ]
            limit = (1.0 - eps_spin) / (2.0 * spin_term * proposed_mass)
            eff_spin_proposed = np.linalg.norm(omega_old + d_omega)
            if eff_spin_proposed <= limit:
                continue
            # Solve a*x^2 + b*x + c = 0 for the critical rotation scale
            a = np.dot(d_omega, d_omega)
            b = 2.0 * np.dot(omega_old, d_omega)
            c = np.dot(omega_old, omega_old) - limit**2
            discriminant = b**2 - 4.0 * a * c
            if discriminant < 0:
                logger.warning(
                    f"Effective spin for {spin_key} exceeds 1 and no valid"
                    " rotation update exists. Setting rotation step to zero."
                )
                alpha_spin = 0.0
            else:
                # Numerically stable root selection (avoid cancellation)
                if b < 0:
                    alpha_spin = (-b + np.sqrt(discriminant)) / (2.0 * a)
                else:
                    alpha_spin = (2.0 * c) / (-b - np.sqrt(discriminant))
            alpha_spin = max(0.0, min(alpha_spin, 1.0))
            if alpha_spin < alpha:
                alpha = alpha_spin
                limiting_constraints = [spin_key]
            elif alpha_spin == alpha:
                limiting_constraints.append(spin_key)

        if alpha < 1.0:
            eff_spin_after = {
                sk: (
                    2.0
                    * (
                        1.0
                        + np.sqrt(
                            1.0 - np.dot(target_params[sk], target_params[sk])
                        )
                    )
                    * (
                        u[param_index_map[mk]]
                        + alpha * Delta_u[param_index_map[mk]]
                    )
                    * np.linalg.norm(
                        u[param_index_map[sk] : param_index_map[sk] + 3]
                        + alpha
                        * Delta_u[param_index_map[sk] : param_index_map[sk] + 3]
                    )
                )
                for mk, sk in zip(
                    ["MassA", "MassB"],
                    ["DimensionlessSpinA", "DimensionlessSpinB"],
                )
                if mk in control_params and sk in control_params
            }
            logger.warning(
                f"Backtracking alpha = {alpha:.6f} (limited by"
                f" {limiting_constraints}). Preserving Broyden direction."
                + (
                    f" Effective spins after scaling: {eff_spin_after}."
                    if eff_spin_after
                    else ""
                )
            )
            Delta_u = alpha * Delta_u

        u += Delta_u

        # Compute residual and check stopping condition
        F = Residual(u)
        current_residual = np.max(np.abs(F))
        if current_residual < best_residual:
            best_residual = current_residual
            best_iteration = iteration
        if current_residual < residual_tolerance:
            break

        # Update the Jacobian using Broyden's method
        J += np.outer(F, Delta_u) / np.dot(Delta_u, Delta_u)
        with spectre_h5.H5File(output_filename, "a") as output_file:
            dat_file = output_file.try_insert_dat(
                "Jacobian", jacobian_legend, 0
            )
            dat_file.append(J.flatten())

    converged = np.max(np.abs(F)) < residual_tolerance
    if not converged:
        logger.error(
            f"Control loop failed to converge after {max_iterations}"
            f" iterations. Best residual: {best_residual:.2e} at"
            f" iteration {best_iteration}."
        )
    if run_convergence_tests:
        # Reconstruct free data from final u for the post-control test
        final_free_data = initial_free_data.copy()
        u_iterator = iter(u)
        for key in [FreeDataFromParams[param] for param in control_params]:
            if key in ScalarQuantities:
                final_free_data[key] = next(u_iterator)
            else:
                final_free_data[key] = [next(u_iterator) for _ in range(3)]
        _convergence_test(
            run_dir=control_run_dir,
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
