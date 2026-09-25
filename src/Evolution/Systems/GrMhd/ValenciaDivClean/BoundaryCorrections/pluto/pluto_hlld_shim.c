#include "pluto.h"
#include "globals.h"   /* this TU owns PLUTO's globals (normally main.c's job) */
#include "pluto_hlld_shim.h"
#include <string.h>

#define PLUTO_SHIM_CAPACITY 1024   /* interfaces per PLUTO call; larger faces chunk */

static _Thread_local int   shim_ready = 0;
static _Thread_local Sweep shim_sweep;
static _Thread_local double shim_cmax[PLUTO_SHIM_CAPACITY];

static void shim_init(void) {
  if (shim_ready) return;
  /* NMAX_POINT sizes every array MakeState allocates, and HLLD_Solver's own
     lazily-allocated scratch. It must be set BEFORE MakeState. */
  /* Headroom matters: MakeState allocates the RIGHT state with a +1 pointer
     offset (tools.c, "in order to access stateR->v[i-1]"), so stateR's valid
     index range is -1 .. NMAX_POINT-2. Sizing NMAX_POINT exactly at the chunk
     size walks one past the end on the last interface of a full chunk. */
  NMAX_POINT = PLUTO_SHIM_CAPACITY + 2;
  /* MXn/BXn/VXn are globals; zero without this, which silently makes PLUTO
     treat the density slot as momentum. We always hand it normal-aligned
     states, so IDIR is always the correct choice. */
  SetVectorIndices(IDIR);
  g_stepNumber = 0;
  g_maxRiemannIter = 0;
  MakeState(&shim_sweep);
  shim_ready = 1;
}

int pluto_hlld_flux(int npts, const double* vL, const double* vR, double gamma,
                    double* flux_out, double* press_out) {
  if (npts <= 0) return 0;
  shim_init();
  g_gamma = gamma;

  for (int base = 0; base < npts; base += PLUTO_SHIM_CAPACITY) {
    const int n = (npts - base < PLUTO_SHIM_CAPACITY) ? (npts - base)
                                                      : PLUTO_SHIM_CAPACITY;
    for (int i = 0; i < n; ++i) {
      const double* sL = vL + (size_t)(base + i) * 8;
      const double* sR = vR + (size_t)(base + i) * 8;
      double* dL = shim_sweep.stateL.v[i];
      double* dR = shim_sweep.stateR.v[i];
      for (int nv = 0; nv < NVAR; ++nv) { dL[nv] = 0.0; dR[nv] = 0.0; }
      dL[RHO]=sL[0]; dL[VX1]=sL[1]; dL[VX2]=sL[2]; dL[VX3]=sL[3];
      dL[BX1]=sL[4]; dL[BX2]=sL[5]; dL[BX3]=sL[6]; dL[PRS]=sL[7];
      dR[RHO]=sR[0]; dR[VX1]=sR[1]; dR[VX2]=sR[2]; dR[VX3]=sR[3];
      dR[BX1]=sR[4]; dR[BX2]=sR[5]; dR[BX3]=sR[6]; dR[PRS]=sR[7];
      shim_sweep.flag[i] = 0;
    }
    /* HLLD_Solver computes a2 (SoundSpeed2), flux/prs (Flux) and SL/SR
       (HLL_Speed) itself; it needs u and h supplied. grid may be NULL --
       it is only dereferenced inside the HLLD_DEBUG block, which is off. */
    PrimToCons(shim_sweep.stateL.v, shim_sweep.stateL.u, 0, n - 1);
    PrimToCons(shim_sweep.stateR.v, shim_sweep.stateR.u, 0, n - 1);
    Enthalpy  (shim_sweep.stateL.v, shim_sweep.stateL.h, 0, n - 1);
    Enthalpy  (shim_sweep.stateR.v, shim_sweep.stateR.h, 0, n - 1);

    HLLD_Solver(&shim_sweep, 0, n - 1, shim_cmax, NULL);

    for (int i = 0; i < n; ++i) {
      double* out = flux_out + (size_t)(base + i) * 8;
      for (int nv = 0; nv < 8; ++nv) out[nv] = shim_sweep.flux[i][nv];
      if (press_out) press_out[base + i] = shim_sweep.press[i];
    }
  }
  return 0;
}
