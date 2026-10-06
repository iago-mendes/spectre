// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/Binary.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"
#include "Utilities/TMPL.hpp"

namespace py = pybind11;

namespace Xcts::AnalyticData::py_bindings {

namespace {
using IsolatedObject = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;
using KerrSchildBinary = Binary<elliptic::analytic_data::AnalyticSolution,
                                tmpl::list<IsolatedObject>>;

using all_tags = KerrSchildBinary::tags<DataVector>;
// Tags that `CommonVariables` computes by numerical differentiation on a mesh
using numeric_tags = tmpl::list<
    ::Tags::deriv<
        Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>,
        tmpl::size_t<3>, Frame::Inertial>,
    Tags::ConformalRicciTensor<DataVector, 3, Frame::Inertial>,
    Tags::ConformalRicciScalar<DataVector>,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    ::Tags::div<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
        DataVector, 3, Frame::Inertial>>>;
using analytic_tags = tmpl::list_difference<all_tags, numeric_tags>;

template <typename TagsList>
std::string tag_names() {
  std::string names{};
  tmpl::for_each<TagsList>([&names](const auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    names += "  " + db::tag_name<tag>() + "\n";
  });
  return names;
}

// Copies the requested quantities out of `vars` into a dict, and throws if a
// name is not in `TagsList`.
template <typename TagsList>
py::dict select(const tuples::tagged_tuple_from_typelist<TagsList>& vars,
                const std::vector<std::string>& requested_quantities,
                const bool have_mesh) {
  py::dict result{};
  for (const auto& requested_quantity : requested_quantities) {
    bool found = false;
    tmpl::for_each<TagsList>(
        [&requested_quantity, &vars, &result, &found](const auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          if (not found and requested_quantity == db::tag_name<tag>()) {
            result[requested_quantity.c_str()] = get<tag>(vars);
            found = true;
          }
        });
    if (not found) {
      throw std::invalid_argument(
          "Requested quantity '" + requested_quantity + "' is not available" +
          (have_mesh ? std::string{}
                     : std::string{" without a mesh and inverse Jacobian "
                                   "(numerical derivatives need them)"}) +
          ". Available quantities are:\n" + tag_names<TagsList>());
    }
  }
  return result;
}
}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");

  py::class_<KerrSchildBinary>(
      m, "Binary",
      "Binary background of the XCTS equations (Xcts::AnalyticData::Binary), "
      "superposing two Kerr-Schild black holes ('KerrSchild' isolated "
      "objects, as in the BBH initial-data input files). The arguments are "
      "the input-file options: 'XCoords', 'CenterOfMassOffset', the "
      "'ObjectLeft'/'ObjectRight' KerrSchild 'Mass', 'Spin', 'Center' and "
      "'Velocity', 'AngularVelocity', 'Expansion', 'LinearVelocity' and "
      "'FalloffWidths' (None disables the Gaussian falloff).")
      .def(py::init(
               [](const std::array<double, 2>& xcoords,
                  const std::array<double, 2>& center_of_mass_offset,
                  const double mass_left,
                  const std::array<double, 3>& dimensionless_spin_left,
                  const double mass_right,
                  const std::array<double, 3>& dimensionless_spin_right,
                  const double angular_velocity, const double expansion,
                  const std::array<double, 3>& linear_velocity,
                  const std::optional<std::array<double, 2>>& falloff_widths,
                  const std::array<double, 3>& center_left,
                  const std::array<double, 3>& velocity_left,
                  const std::array<double, 3>& center_right,
                  const std::array<double, 3>& velocity_right) {
                 if (xcoords[0] >= xcoords[1]) {
                   throw std::invalid_argument(
                       "Specify 'xcoords' ascending from left to right.");
                 }
                 return std::make_unique<KerrSchildBinary>(
                     xcoords, center_of_mass_offset,
                     std::make_unique<IsolatedObject>(
                         mass_left, dimensionless_spin_left, center_left,
                         velocity_left),
                     std::make_unique<IsolatedObject>(
                         mass_right, dimensionless_spin_right, center_right,
                         velocity_right),
                     angular_velocity, expansion, linear_velocity,
                     falloff_widths);
               }),
           py::arg("xcoords"), py::arg("center_of_mass_offset"),
           py::arg("mass_left"), py::arg("dimensionless_spin_left"),
           py::arg("mass_right"), py::arg("dimensionless_spin_right"),
           py::arg("angular_velocity"), py::arg("expansion"),
           py::arg("linear_velocity"), py::arg("falloff_widths"),
           py::arg("center_left") = std::array<double, 3>{{0., 0., 0.}},
           py::arg("velocity_left") = std::array<double, 3>{{0., 0., 0.}},
           py::arg("center_right") = std::array<double, 3>{{0., 0., 0.}},
           py::arg("velocity_right") = std::array<double, 3>{{0., 0., 0.}})
      .def_property_readonly("x_coords", &KerrSchildBinary::x_coords)
      .def_property_readonly("y_offset", &KerrSchildBinary::y_offset)
      .def_property_readonly("z_offset", &KerrSchildBinary::z_offset)
      .def_property_readonly("angular_velocity",
                             &KerrSchildBinary::angular_velocity)
      .def_property_readonly("expansion", &KerrSchildBinary::expansion)
      .def_property_readonly("linear_velocity",
                             &KerrSchildBinary::linear_velocity)
      .def_property_readonly("falloff_widths",
                             &KerrSchildBinary::falloff_widths)
      .def(
          "variables",
          [](const KerrSchildBinary& binary, const tnsr::I<DataVector, 3>& x,
             const std::vector<std::string>& requested_quantities) {
            return select<analytic_tags>(binary.variables(x, analytic_tags{}),
                                         requested_quantities, false);
          },
          py::arg("x"), py::arg("requested_quantities"),
          "Background quantities at the points 'x' that need no numerical "
          "derivative.")
      .def(
          "variables",
          [](const KerrSchildBinary& binary, const tnsr::I<DataVector, 3>& x,
             const Mesh<3>& mesh,
             const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                                   Frame::Inertial>& inv_jacobian,
             const std::vector<std::string>& requested_quantities) {
            return select<all_tags>(
                binary.variables(x, mesh, inv_jacobian, all_tags{}),
                requested_quantities, true);
          },
          py::arg("x"), py::arg("mesh"), py::arg("inv_jacobian"),
          py::arg("requested_quantities"),
          "Background quantities at the collocation points 'x' of 'mesh' in "
          "one element, whose map has the inverse Jacobian 'inv_jacobian'. "
          "The conformal Ricci tensor and scalar, the derivative of the "
          "Christoffel symbols and of the extrinsic-curvature trace, and the "
          "divergence of the longitudinal shift background are differentiated "
          "numerically on 'mesh', as SolveXcts does on its own mesh.");
}

}  // namespace Xcts::AnalyticData::py_bindings
