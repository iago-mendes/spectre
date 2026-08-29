/* C interface to PLUTO's HLLD Riemann solver, for use from SpECTRE.
 *
 * Everything PLUTO-side is reached through this one entry point; the C++ side
 * never sees a PLUTO type. See README.md for the list of local modifications
 * made to the vendored PLUTO sources. */
#ifndef PLUTO_HLLD_SHIM_H
#define PLUTO_HLLD_SHIM_H
#ifdef __cplusplus
extern "C" {
#endif

/* Solve `npts` Riemann problems with PLUTO's HLLD_Solver.
 *
 *  vL, vR     [in]  npts*8, PLUTO RMHD PRIMITIVE order and units:
 *                   {RHO, VX1, VX2, VX3, BX1, BX2, BX3, PRS}
 *                   ALREADY ROTATED so component 1 (VX1) and 4 (BX1) are the
 *                   interface-normal velocity and magnetic field.
 *  gamma      [in]  ideal-gas adiabatic index (PLUTO's EOS is IDEAL only).
 *  flux_out   [out] npts*8, PLUTO CONSERVED order
 *                   {RHO, MX1, MX2, MX3, BX1, BX2, BX3, ENG}, where ENG is the
 *                   REDUCED energy E - D (RMHD_REDUCED_ENERGY == YES), which is
 *                   what SpECTRE's TildeTau uses.
 *  press_out  [out] npts, the normal total-pressure term. PLUTO carries this
 *                   OUTSIDE the momentum flux, so the caller must add it to the
 *                   normal momentum component itself.
 *
 * Returns 0 on success, non-zero if the solve could not be set up.
 * Thread-safe: all PLUTO state it touches is _Thread_local, and each calling
 * thread lazily builds its own Sweep. */
int pluto_hlld_flux(int npts, const double* vL, const double* vR, double gamma,
                    double* flux_out, double* press_out);

#ifdef __cplusplus
}
#endif
#endif
