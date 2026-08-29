#
# Distributed under the MIT License.
# See LICENSE.txt for details.

spectre_target_sources(
  ${LIBRARY}
  PRIVATE
  Hll.cpp
  Hlld.cpp
  Hllem.cpp
  Marquina.cpp
  PlutoHlld.cpp
  Rusanov.cpp
  )

spectre_target_headers(
  ${LIBRARY}
  INCLUDE_DIRECTORY ${CMAKE_SOURCE_DIR}/src
  HEADERS
  Factory.hpp
  Hll.hpp
  Hlld.hpp
  HlldImpl.hpp
  Hllem.hpp
  Marquina.hpp
  PlutoHlld.hpp
  NamespaceDocs.hpp
  Rusanov.hpp
  )

# --- Vendored PLUTO v4.4 HLLD (see pluto/README.md) -------------------------
# PlutoHlld calls PLUTO's own compiled HLLD rather than a re-implementation.
#
# These are C sources, and they must live in their OWN target rather than being
# added to ${LIBRARY}: SpECTRE precompiles a C++ header for that library, and
# CMake insists on a per-language PCH for every language a target uses, so
# adding C files makes it demand a C PCH ("cmake_pch.h") that does not exist.
# Per-source SKIP_PRECOMPILE_HEADERS does not avoid this. A separate static
# library with PCH disabled keeps the C and C++ toolchains cleanly apart.
set(PLUTO_HLLD_SOURCES
  pluto/arrays.c pluto/debug_tools.c pluto/eigenv.c pluto/eos.c
  pluto/fluxes.c pluto/hll_speed.c pluto/hlld.c pluto/mappers.c
  pluto/math_misc.c pluto/math_qr_decomp.c pluto/math_root_finders.c
  pluto/output_log.c pluto/rmhd_energy_solve.c pluto/rmhd_pressure_fix.c
  pluto/set_indexes.c pluto/tools.c pluto/pluto_hlld_shim.c
  )

add_library(PlutoHlldC STATIC ${PLUTO_HLLD_SOURCES})

set_target_properties(
  PlutoHlldC PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  DISABLE_PRECOMPILE_HEADERS ON
  LINKER_LANGUAGE C
  )

# PLUTO resolves its headers relative to its own directory ("pluto.h",
# "States/plm_coeffs.h", ...). PUBLIC so PlutoHlld.cpp picks up the shim header.
target_include_directories(
  PlutoHlldC PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/pluto
  )

# Upstream PLUTO is not warning-clean under SpECTRE's flags. We do not patch
# third-party code to silence warnings; our own shim is small and reviewed.
target_compile_options(PlutoHlldC PRIVATE -w)

target_link_libraries(${LIBRARY} PRIVATE PlutoHlldC)
