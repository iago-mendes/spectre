# Distributed under the MIT License.
# See LICENSE.txt for details.

# Dump tool for the Del Zanna et al. (2003) Fig. 6 jet setup. Evaluates
# grmhd::AnalyticData::AxisymmetricJet (and, for comparison, SlabJet) on a real
# domain::creators::CartoonCylinder domain and writes CSV, so that figures and
# derived-quantity tables come from the C++ classes through SpECTRE's own
# domain machinery rather than from a reimplementation.
#
# EXCLUDE_FROM_ALL: this is an analysis tool, not part of a normal build.

set(EXECUTABLE ExportAxisymmetricJetData)

add_spectre_executable(
  ${EXECUTABLE}
  EXCLUDE_FROM_ALL
  ExportAxisymmetricJetData.cpp
  )

target_link_libraries(
  ${EXECUTABLE}
  PRIVATE
  DataStructures
  Domain
  DomainBoundaryConditions
  DomainCreators
  DomainStructure
  GeneralRelativitySolutions
  GrMhdAnalyticData
  Hydro
  Options
  Spectral
  Utilities
  ValenciaDivClean
  )
