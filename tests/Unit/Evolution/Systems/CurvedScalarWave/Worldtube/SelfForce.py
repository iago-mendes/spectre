# Distributed under the MIT License.
# See LICENSE.txt for details.

import numpy as np


def self_force_acceleration(
    dt_psi_monopole, psi_dipole, vel, charge, mass, inverse_metric, dilation
):
    # Prepend extra value so dimensions work out for einsum.
    # The 0th component does not affect the final result
    four_vel = np.concatenate((np.empty(1), vel), axis=0)
    d_psi = np.concatenate(([dt_psi_monopole], psi_dipole), axis=0)
    self_force_acc = np.einsum("ij,j", inverse_metric, d_psi)
    self_force_acc -= np.einsum("i,j,j", four_vel, inverse_metric[0], d_psi)
    self_force_acc *= charge / mass / dilation**2
    return self_force_acc[1:]


def self_force_per_mass(d_psi, four_velocity, charge, mass, inverse_metric):
    self_force_per_mass = np.einsum("ij,j", inverse_metric, d_psi)
    self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, d_psi
    )
    return charge / mass * self_force_per_mass


def dt_self_force_per_mass(
    d_psi,
    dt_d_psi,
    four_velocity,
    dt_four_velocity,
    charge,
    mass,
    inverse_metric,
    dt_inverse_metric,
):
    dt_self_force_per_mass = np.einsum("ij,j", dt_inverse_metric, d_psi)
    dt_self_force_per_mass += np.einsum("ij,j", inverse_metric, dt_d_psi)

    dt_self_force_per_mass += np.einsum(
        "i,j,j", dt_four_velocity, four_velocity, d_psi
    )
    dt_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, dt_four_velocity, d_psi
    )
    dt_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, dt_d_psi
    )
    return charge / mass * dt_self_force_per_mass


def dt2_self_force_per_mass(
    d_psi,
    dt_d_psi,
    dt2_d_psi,
    four_velocity,
    dt_four_velocity,
    dt2_four_velocity,
    charge,
    mass,
    inverse_metric,
    dt_inverse_metric,
    dt2_inverse_metric,
):
    dt2_self_force_per_mass = np.einsum("ij,j", dt2_inverse_metric, d_psi)
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "ij,j", dt_inverse_metric, dt_d_psi
    )
    dt2_self_force_per_mass += np.einsum("ij,j", inverse_metric, dt2_d_psi)

    dt2_self_force_per_mass += np.einsum(
        "i,j,j", dt2_four_velocity, four_velocity, d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", dt_four_velocity, dt_four_velocity, d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", dt_four_velocity, four_velocity, dt_d_psi
    )
    dt2_self_force_per_mass += 2.0 * np.einsum(
        "i,j,j", four_velocity, dt_four_velocity, dt_d_psi
    )
    dt2_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, dt2_four_velocity, d_psi
    )
    dt2_self_force_per_mass += np.einsum(
        "i,j,j", four_velocity, four_velocity, dt2_d_psi
    )
    return charge / mass * dt2_self_force_per_mass


def Du_self_force_per_mass(
    self_force, dt_self_force, four_velocity, christoffel
):
    Du_self_force_per_mass = four_velocity[0] * dt_self_force
    Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, four_velocity, self_force
    )
    return Du_self_force_per_mass


def dt_Du_self_force_per_mass(
    self_force,
    dt_self_force,
    dt2_self_force,
    four_velocity,
    dt_four_velocity,
    christoffel,
    dt_christoffel,
):
    dt_Du_self_force_per_mass = (
        dt_four_velocity[0] * dt_self_force + dt2_self_force * four_velocity[0]
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", dt_christoffel, four_velocity, self_force
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, dt_four_velocity, self_force
    )
    dt_Du_self_force_per_mass += np.einsum(
        "ijk,j,k", christoffel, four_velocity, dt_self_force
    )
    return dt_Du_self_force_per_mass


def turn_on_function(time, timescale):
    return 1.0 - np.exp(-((time / timescale) ** 4))


def iterate_acceleration_terms(
    vel,
    psi_monopole,
    dt_psi_monopole,
    psi_dipole,
    dt_psi_dipole,
    roll_on,
    dt_roll_on,
    dt2_roll_on,
    imetric,
    christoffel,
    trace_christoffel,
    di_imetric,
    dij_imetric,
    di_trace_christoffel,
    di_christoffel,
    u0,
    geodesic_acc,
    charge,
    mass,
    current_iteration,
):
    res = np.zeros(15)

    evolved_mass = (
        mass if current_iteration == 1 else mass - charge * psi_monopole
    )
    dt_evolved_mass = dt_psi_monopole + np.einsum("i,i", vel, psi_dipole)
    dt_evolved_mass = 0 if current_iteration == 1 else -charge * dt_evolved_mass
    self_force_acc = self_force_acceleration(
        dt_psi_monopole, psi_dipole, vel, charge, evolved_mass, imetric, u0
    )
    acc = geodesic_acc + roll_on * self_force_acc
    res[:3] = acc

    d_psi = np.concatenate(([dt_psi_monopole], psi_dipole), axis=0)
    u = np.concatenate(([u0], u0 * vel))
    f = self_force_per_mass(d_psi, u, charge, evolved_mass, imetric)
    f_rollon = f * roll_on
    res[3:6] = f_rollon[:3]

    dt_u = (
        charge / evolved_mass * np.einsum("ab,b", imetric, d_psi)
        - np.einsum("abc,b,c", christoffel, u, u)
    ) / u0
    dt_u_roll_on = dt_u * roll_on
    dt_imetric = np.einsum("abc,a", di_imetric, vel)
    dt2_psi = (
        -2.0 * np.einsum("i,i", imetric[0, 1:], dt_psi_dipole)
        + trace_christoffel[0] * dt_psi_monopole
        + np.einsum("i,i", trace_christoffel[1:], psi_dipole)
    ) / imetric[0, 0]
    dt_d_psi = np.concatenate(([dt2_psi], dt_psi_dipole))
    dt_d_psi[0] += np.einsum("i,i", vel, dt_psi_dipole)
    dt_f = (
        charge
        / evolved_mass
        * (
            np.einsum("ab,b", dt_imetric, d_psi)
            + np.einsum("a,b,b", dt_u_roll_on, u, d_psi)
            + np.einsum("a,b,b", u, dt_u_roll_on, d_psi)
            + np.einsum("ab,b", imetric, dt_d_psi)
            + np.einsum("a,b,b", u, u, dt_d_psi)
        )
    )
    dt_f -= dt_evolved_mass / evolved_mass * f
    dt_f_rollon = dt_f * roll_on + f * dt_roll_on
    res[6:9] = dt_f_rollon[:3]

    cov_f = Du_self_force_per_mass(f_rollon, dt_f_rollon, u, christoffel)
    res[9:12] = cov_f[:3]
    dt2_di_psi = (
        -2.0 * np.einsum("ij,j", di_imetric[:, 1:, 0], dt_psi_dipole)
        + di_trace_christoffel[:, 0] * dt_psi_monopole
        + trace_christoffel[0] * dt_psi_dipole
        + np.einsum("ij,j", di_trace_christoffel[:, 1:], psi_dipole)
        - di_imetric[:, 0, 0] * dt2_psi
    ) / imetric[0, 0]
    dt3_psi = (
        -2.0 * np.einsum("i,i", imetric[0, 1:], dt2_di_psi)
        + trace_christoffel[0] * dt2_psi
        + np.einsum("i,i", trace_christoffel[1:], dt_psi_dipole)
    ) / imetric[0, 0]
    dt2_d_psi = np.concatenate(([dt3_psi], dt2_di_psi))
    dt2_d_psi[0] += 2.0 * np.einsum("i,i", vel, dt2_di_psi) + np.einsum(
        "i,i", acc, dt_psi_dipole
    )
    dt_christoffel = np.einsum("iabc,i", di_christoffel, vel)
    dt2_imetric = np.einsum("ijab,i,j", dij_imetric, vel, vel) + np.einsum(
        "iab,i", di_imetric, acc
    )
    dt2_u = (
        charge
        / evolved_mass
        * (
            np.einsum("ab,b", dt_imetric, d_psi)
            + np.einsum("ab,b", imetric, dt_d_psi)
            - dt_evolved_mass / evolved_mass * np.einsum("ab,b", imetric, d_psi)
        )
        - np.einsum("abc,b,c", dt_christoffel, u, u)
        - 2.0 * np.einsum("abc,b,c", christoffel, dt_u, u)
        - dt_u * dt_u[0]
    ) / u[0]
    dt2_u_roll_on = dt_roll_on * dt_u + roll_on * dt2_u
    dt2_f = dt2_self_force_per_mass(
        d_psi,
        dt_d_psi,
        dt2_d_psi,
        u,
        dt_u_roll_on,
        dt2_u_roll_on,
        charge,
        evolved_mass,
        imetric,
        dt_imetric,
        dt2_imetric,
    )
    dt2_evolved_mass = (
        dt2_psi
        + 2.0 * np.einsum("i,i", vel, dt_psi_dipole)
        + np.einsum("i,i", acc, psi_dipole)
    )
    dt2_evolved_mass = (
        0.0 if current_iteration == 1 else -charge * dt2_evolved_mass
    )

    dt2_f += (
        -2.0 * dt_evolved_mass / evolved_mass * dt_f
        + f
        * (2.0 * dt_evolved_mass**2 - dt2_evolved_mass * evolved_mass)
        / evolved_mass**2
    )

    dt2_f_roll_on = dt2_roll_on * f + 2.0 * dt_roll_on * dt_f + roll_on * dt2_f
    dt_cov_f = dt_Du_self_force_per_mass(
        f_rollon,
        dt_f_rollon,
        dt2_f_roll_on,
        u,
        dt_u_roll_on,
        christoffel,
        dt_christoffel,
    )
    res[12:15] = dt_cov_f[:3]

    return res
