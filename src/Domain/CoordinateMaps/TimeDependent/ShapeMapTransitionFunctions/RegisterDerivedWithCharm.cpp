// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/RegisterDerivedWithCharm.hpp"

#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/SphereTransition.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ShapeMapTransitionFunctions/Wedge.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain::CoordinateMaps::ShapeMapTransitionFunctions {
void register_derived_with_charm() {
  register_classes_with_charm<SphereTransition, Wedge>();
}
}  // namespace domain::CoordinateMaps::ShapeMapTransitionFunctions
