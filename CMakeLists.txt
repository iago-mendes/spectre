# Distributed under the MIT License.
# See LICENSE.txt for details.

cmake_minimum_required(VERSION 3.18.0)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

# Create a new (or overwrite existing) info file at start of configuration
file(WRITE "${CMAKE_BINARY_DIR}/BuildInfo.txt" "")

# Determine the project version and other metadata
include(SpectreLoadMetadata)

# Set up the project. Notes:
# - Fortran is needed for SPHEREPACK
project(${SPECTRE_NAME} VERSION ${SPECTRE_VERSION} LANGUAGES CXX C Fortran)

# Unset the CMake-defined version variable because it strips zeros from the
# version components, e.g. 2020.12.07 becomes 2020.12.7
unset(${SPECTRE_NAME}_VERSION)
# Also unset the version-component variables because they have no meaning in
# our versioning scheme
unset(${SPECTRE_NAME}_VERSION_MAJOR)
unset(${SPECTRE_NAME}_VERSION_MINOR)
unset(${SPECTRE_NAME}_VERSION_PATCH)

# Policies
# The `cmake_minimum_required` above sets policies to `NEW` that are compatible
# with the given minimum cmake version. Here we overwrite policies that we
# have back-ported in our cmake code.
# - We use test names with '.' characters in them
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.19)
  cmake_policy(SET CMP0110 NEW)
endif ()
# - We allow both uppercase and case-preserved hints in `find_package` calls, so
#   `PACKAGENAME_ROOT` and `PackageName_ROOT` are both searched in
#   `FindPackageName.cmake` (NEW behavior). The OLD behavior was to search only
#   `PackageName_ROOT` (before CMake v3.27). We have backported the NEW behavior
#   by adding the uppercase name explicitly to the list of hints.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.27)
  cmake_policy(SET CMP0144 NEW)
endif()
# - The `NEW` behavior of `DOWNLOAD_EXTRACT_TIMESTAMP` is recommended.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.24)
  cmake_policy(SET CMP0135 NEW)
endif()
# - We allow CMake 3.30 and above to load Boost's CMake config file that is
#   installed alongside Boost. Before, CMake provided its own FindBoost module.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.30)
  cmake_policy(SET CMP0167 NEW)
endif ()

# FetchContent can be used to download and build some dependencies. We can
# consider turning this on by default later.
option(SPECTRE_FETCH_MISSING_DEPS "Download missing dependencies" OFF)
# - Mark FetchContent as 'SYSTEM' to suppress warnings.
#   Supported since CMake version 3.25.
if (CMAKE_VERSION VERSION_GREATER_EQUAL 3.25)
  set(SPECTRE_FETCHCONTENT_BASE_ARGS SYSTEM)
else()
  set(SPECTRE_FETCHCONTENT_BASE_ARGS "")
endif()

# Define standard installation directories
include(GNUInstallDirs)
# Disable `make install` depending on `make all` since we want to control what
# we install more closely. With this setting, and targets marked as `OPTIONAL`,
# only targets that were built will be installed.
set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

set(CMAKE_VERBOSE_MAKEFILE OFF)

include(SpectreGetGitHash)
include(SpectreSetSiteName)

# We need Python in the build system and throughout the code. Setting it up
# early so we can rely on a consistent Python version in the build system.
find_package(Python 3.8 REQUIRED)

# Define the location of Python code in the build directory. Unit tests etc. can
# add this location to their `PYTHONPATH`.
set(SPECTRE_PYTHON_PREFIX_PARENT "${CMAKE_BINARY_DIR}/bin/python")
get_filename_component(
  SPECTRE_PYTHON_PREFIX_PARENT "${SPECTRE_PYTHON_PREFIX_PARENT}" ABSOLUTE)
set(SPECTRE_PYTHON_SITE_PACKAGES "${CMAKE_BINARY_DIR}/lib/\
python${Python_VERSION_MAJOR}.${Python_VERSION_MINOR}/site-packages")
set(PYTHONPATH "${SPECTRE_PYTHON_PREFIX_PARENT}:\
${SPECTRE_PYTHON_SITE_PACKAGES}")
if(DEFINED ENV{PYTHONPATH})
  set(PYTHONPATH "${PYTHONPATH}:$ENV{PYTHONPATH}")
endif()
message(STATUS "PYTHONPATH: ${PYTHONPATH}")
include(BootstrapPyDeps)

option(BUILD_DOCS "Enable building documentation" ON)
option(
  DOCS_ONLY
  "Skip all initialization not required for rendering documentation"
  OFF
)
if (BUILD_DOCS AND DOCS_ONLY)
  include(SetupDoxygen)
  return()
endif()

option(USE_PCH "Use precompiled headers" ON)

include(SpectreInitializeVariables)

include(CheckCompilerVersion)
include(ProhibitInSourceBuild)
include(SpectreSetupFlagsTarget)
include(SetupFortran)
include(SetupNinjaColors)
include(SetOutputDirectory)
include(SpectreAddInterfaceLibraryHeaders)
include(SpectreTargetHeaders)
include(SpectreTargetSources)
include(SetupFormaline)
include(SetupGitHooks)
include(SetupLicenseInfo)
include(SetBuildType)
include(SetupPic)
include(SetupLinkTimeOptimization)
include(SetCxxStandard)
include(StripSymbols)
# We need Boost for InfoAtLink
include(SetupBoost)
include(SetupInformer)
include(SetupCCache)
include(SetupCharm)
include(EnableWarnings)
include(SetupGoldOrLldLinker)

# In order to use certain code analysis tools like clang-tidy and cppcheck the
# compile commands need to be accessible. CMake can write these to a
# "compile_commands.json" file.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(SetupLIBCXX)
include(SetupSpectreInlining)
include(SetupCxxFlags)
include(SetupProfiling)
include(SetupSanitizers)
include(SetupListTargets)
include(AddSpectreExecutable)
include(CheckBrokenArray0)

# Blaze depends on both Blas and Lapack, since LIBXSMM is a Blas
# alternative we set it up early too. Finally, Blaze uses the GSL
# blas headers, so we need to set that up before Blaze as well.
include(SetupBlas)
include(SetupLapack)
include(SetupLIBXSMM)
include(SetupGsl)

include(SetupBlaze)
include(SetupFuka)
include(SetupGoogleBenchmark)
include(SetupHdf5)
include(SetupAllocator)
include(SetupPapi)
include(SetupPybind11)
include(SetupSpec)
include(SetupStl)
include(SetupXsimd)
include(SetupYamlCpp)
include(SetupOpenMP)
include(SetupParaView)

add_subdirectory(external)

include(SetupKokkos)

include(SetupLIBCXXCharm)
# The precompiled header must be setup after all libraries have been found
if (USE_PCH)
  include(SetupPch)
endif()

# All special targets and configs that need to be applied to *all*
# executables must be added at once in the 'UpdateAddExecutables' file.
# This is because of what is likely a bug in CMake where if a function is
# overridden multiple times (using the _function_name(...) method) then some
# versions of CMake (at least 3.13.2) segfault.
include(UpdateAddExecutables)

# The ClangFormat, clang-tidy, Doxygen, and CodeCov are intentionally
# after the PCH setup because that way they are able to change their
# dependencies on the PCH if necessary.
include(SetupClangFormat)
include(SetupClangTidy)
if(BUILD_DOCS)
  include(SetupDoxygen)
  include(SetupSphinx)
endif()
include(CodeCoverageDetection)
include(SpectreAddLibraries)

include(SpectreSetupTesting)
if(BUILD_TESTING)
  include(SetupCatch)
  include(SetupPypp)
  include(SpectreAddTestLibs)
  include(SpectreAddCatchTests)
  include(AddInputFileTests)
  include(AddStandaloneTests)
endif()

include(SpectreSetupPythonPackage)

# Set global include directory for all targets
include_directories(${CMAKE_SOURCE_DIR}/src)
# Charm++ generated headers are created in the build directory
# (by `add_charm_module`)
include_directories(SYSTEM ${CMAKE_BINARY_DIR}/src)

add_subdirectory(src)
add_subdirectory(support)
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
add_subdirectory(tools)

include(PrintUsefulCMakeInfo)

include(SpectreCheckDependencies)
