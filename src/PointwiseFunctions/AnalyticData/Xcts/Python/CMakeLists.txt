# Distributed under the MIT License.
# See LICENSE.txt for details.

set(LIBRARY "PyXctsAnalyticData")

spectre_python_add_module(
  Xcts
  LIBRARY_NAME ${LIBRARY}
  MODULE_PATH "PointwiseFunctions/AnalyticData"
  SOURCES
  Bindings.cpp
  PYTHON_FILES
  __init__.py
)

spectre_python_link_libraries(
  ${LIBRARY}
  PRIVATE
  DataStructures
  GeneralRelativitySolutions
  pybind11::module
  Spectral
  Utilities
  XctsAnalyticData
  XctsSolutions
)

spectre_python_add_dependencies(
  ${LIBRARY}
  PyDataStructures
  PySpectral
  PyTensor
)
