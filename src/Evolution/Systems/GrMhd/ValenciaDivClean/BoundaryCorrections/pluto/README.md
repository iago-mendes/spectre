# Vendored PLUTO sources — HLLD Riemann solver

This directory contains a **minimal subset of PLUTO v4.4** (Mignone et al.),
vendored so that SpECTRE's `PlutoHlld` boundary correction calls PLUTO's *own*
compiled HLLD rather than a re-implementation of it.

Why vendored rather than re-implemented: we ported `hlld.c` to C++ by hand and
spent several sessions chasing a high-order-reconstruction instability that we
could not attribute. Calling the reference implementation directly removes that
entire class of porting bug. See `meetings/2026-09-03/plan.md` in the runs repo.

## Provenance

Upstream: PLUTO v4.4, `Src/` tree. Copied from our working tree at
`runs-ai/mhd_marquina/PLUTO/`.

**That tree was NOT pristine** — it carried our own experiment code. Everything
we removed or changed is listed below. Nothing else was touched.

## Source set (16 PLUTO files + 1 shim)

Determined from a link map, not by guessing: these are exactly the objects the
linker pulls when resolving `HLLD_Solver` and its transitive dependencies.

```
arrays.c  debug_tools.c  eigenv.c  eos.c (EOS/Ideal)  fluxes.c  hll_speed.c
hlld.c  mappers.c  math_misc.c  math_qr_decomp.c  math_root_finders.c
output_log.c  rmhd_energy_solve.c  rmhd_pressure_fix.c  set_indexes.c  tools.c
```
`pluto_hlld_shim.{c,h}` is ours: the only entry point the C++ side uses.

`rmhd_entropy_solve.c` is not needed (entropy switch off) and
`riemann_check.c` is not needed once `COUNT_FAILURES` is off.

## Modifications to upstream — the complete list

1. **Removed our "EXPERIMENT 1b" instrumentation from `hlld.c`** (33 lines,
   was lines 321-353): an admissibility-gate probe with its own `static double
   gate_tot, gate_bad` and a `printf`. `hlld.c` went 759 -> 726 lines.
   *Verified*: the ST1 reference flux is bit-identical before and after.

2. **`_Thread_local` on all mutable state written during a solve.** SpECTRE
   runs Charm++ in SMP mode (many worker threads per process); PLUTO was written
   for one thread per process and uses file-scope and function-static state.
   - `hlld.c`: `static double Sc, Bx;` and `static double **Uhll, **Fhll, **Vhll;`
   - `globals.h` + `pluto.h`: `VXn/VXt/VXb`, `MXn/MXt/MXb`, `BXn/BXt/BXb`,
     `g_maxRiemannIter`, `NMAX_POINT`, `g_gamma`
   Each is marked `/* SPECTRE-MOD */` at the definition site.
   *Verified*: 8 threads solving concurrently all reproduce the reference flux.

3. **`definitions.h`**: `COUNT_FAILURES` and `ENABLE_HLLEM` removed (both were
   our additions). `COUNT_FAILURES` made `RiemannCheck` write `riemann_check.dat`
   from every process every step, which is unacceptable inside SpECTRE.

## Configuration

`PHYSICS=RMHD`, `EOS=IDEAL`, `DIVB_CONTROL=NO`, `NTRACER=0`, no GLM, no CT.
Hence `NVAR = NFLX = 8` with
`RHO=0, MX1=1, MX2=2, MX3=3, BX1=4, BX2=5, BX3=6, ENG=PRS=7`.

`RMHD_REDUCED_ENERGY == YES`, i.e. PLUTO's `ENG` is the **reduced** energy
`E - D`. This matches SpECTRE's `TildeTau`. Verified numerically: the ST1 left
state gives `u[ENG] = 1.625 = (rho h W^2 - p + B^2/2) - D = 2.625 - 1`.

`DIVB_CONTROL=NO` means PLUTO has no divergence-cleaning field. The GLM scalar
`TildePhi` and the normal magnetic-field component are therefore handled on the
SpECTRE side by the same scalar/MHD split `Hllem.cpp` uses; PLUTO supplies only
the 8 MHD fluxes.

## Gotchas found the hard way (do not "simplify" these away)

- **`SetVectorIndices(IDIR)` is mandatory.** `MXn/BXn/VXn` are globals; left at
  zero, PLUTO treats the density slot as momentum and silently produces garbage
  that only shows up as a segfault deep inside `ConsToPrim -> Where()`.
- **`MakeState` offsets the RIGHT state by +1** (`stateR->v = ARRAY_2D(...)+1`),
  so its valid index range is `-1 .. NMAX_POINT-2`. The shim sets
  `NMAX_POINT = capacity + 2` for that reason; sizing it exactly at the chunk
  size overruns on the last interface of a full chunk.
- **`Grid*` may be NULL** for our purposes (only dereferenced in the disabled
  `HLLD_DEBUG` block), but `Where()` will crash if any PLUTO error path is hit,
  since there is no grid to report a location in.
- **PLUTO must be called with FP-exception trapping disabled.** Its allocator
  (`Array2D`, reached from `MakeState`) and its root find raise FPE, so under
  SpECTRE's trapping (on by default in unit tests) the process dies with
  SIGFPE inside PLUTO's memory setup. `PlutoHlld.cpp` wraps its call in
  `ScopedFpeState(false)`; any direct caller of the shim must do the same.
- PLUTO's `Flux()` needs `state->h` (enthalpy) prefilled; `HLLD_Solver` computes
  `a2`, `flux`, `prs`, `SL`, `SR` itself.
