// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Python/Spherepack.hpp"

#include <array>
#include <cstddef>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/ModalVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/AngularOrdering.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/FillYlmLegendAndData.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/ReadSurfaceYlm.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/IO/StrahlkorperCoordsToTextFile.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Spherepack.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
void bind_spherepack(pybind11::module& m) {  // NOLINT
  py::class_<ylm::Spherepack>(m, "Spherepack")
      .def("theta_phi_points", &ylm::Spherepack::theta_phi_points)
      .def(
          "interpolate",
          [](const Spherepack& self, const DataVector& collocation_values,
             const std::array<DataVector, 2>& target_points) {
            return self.interpolate(collocation_values, target_points);
          },
          py::arg("collocation_values"), py::arg("target_points"));
}
}  // namespace ylm::py_bindings
