// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/BlockLogicalCoordinates.hpp"
#include "Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/TimeDependence/UniformTranslation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/BoundaryConditions/BoundaryCondition.hpp"
#include "Helpers/Domain/Creators/TestHelpers.hpp"
#include "Utilities/CartesianProduct.hpp"
#include "Utilities/MakeVector.hpp"

namespace domain {
namespace {
void test_interval_construction(const creators::Interval& domain_creator,
                                const bool expect_boundary_conditions,
                                const bool is_periodic,
                                const std::array<double, 1>& lower_bound,
                                const std::array<double, 1>& upper_bound,
                                const std::vector<double>& times = {0.},
                                const std::array<double, 1>& velocity = {
                                    {0.}}) {
  // Generic tests
  const auto domain = TestHelpers::domain::creators::test_domain_creator(
      domain_creator, expect_boundary_conditions, is_periodic, times);
  TestHelpers::domain::BoundaryConditions::test_boundary_conditions(
      domain_creator.external_boundary_conditions());
  CHECK(domain_creator.grid_anchors().empty());

  // Interval-specific tests
  CHECK(domain.block_groups().empty());
  const auto& blocks = domain.blocks();
  CHECK(blocks.size() == 1);
  CHECK(domain.excision_spheres().empty());

  const auto& block = blocks[0];
  CHECK(block.name() == "Interval");
  const auto& external_boundaries = block.external_boundaries();
  CHECK(external_boundaries.size() == (is_periodic ? 0 : 2));

  // Check that the block boundaries have the expected coordinates
  tnsr::I<DataVector, 1> expected_block_boundaries{2_st};
  const auto functions_of_time = domain_creator.functions_of_time();
  for (const double time : times) {
    CAPTURE(time);
    get<0>(expected_block_boundaries)[0] = lower_bound[0] + time * velocity[0];
    get<0>(expected_block_boundaries)[1] = upper_bound[0] + time * velocity[0];
    const auto block_logical_coords = block_logical_coordinates(
        domain, expected_block_boundaries, time, functions_of_time);
    CHECK(block_logical_coords[0]->id.get_index() == 0);
    CHECK(get<0>(block_logical_coords[0]->data) == -1.);
    CHECK(block_logical_coords[1]->id.get_index() == 0);
    CHECK(get<0>(block_logical_coords[1]->data) == 1.);
  }
}

std::string option_string(const CoordinateMaps::Distribution distribution,
                          const std::optional<double>& singularity,
                          const bool is_periodic, const bool time_dependent,
                          const bool with_boundary_conditions) {
  const std::string time_dep_options = time_dependent
                                           ? "  TimeDependence:\n"
                                             "    UniformTranslation:\n"
                                             "      InitialTime: 0.\n"
                                             "      Velocity: [2.3]\n"
                                           : "  TimeDependence: None\n";
  const std::string bc_options =
      with_boundary_conditions
          ? ("  BoundaryConditions:\n"
             "    - Lower:\n" +
             std::string{is_periodic ? "        Periodic\n"
                                     : "        TestBoundaryCondition:\n"
                                       "          Direction: lower-xi\n"
                                       "          BlockId: 0\n"} +
             "      Upper:\n" +
             std::string{is_periodic ? "        Periodic\n"
                                     : "        TestBoundaryCondition:\n"
                                       "          Direction: upper-xi\n"
                                       "          BlockId: 0\n"})
          : ("  IsPeriodicIn: [" + std::string{is_periodic ? "True" : "False"} +
             "]\n");
  return "Interval:\n"
         "  LowerBound: [-1.2]\n"
         "  UpperBound: [0.8]\n" +
         (singularity.has_value()
              ? ("  Distribution:\n"
                 "    - " +
                 get_output(distribution) +
                 ":\n"
                 "        SingularityPosition: " +
                 std::to_string(*singularity))
              : ("  Distribution: [" + get_output(distribution) + "]")) +
         "\n"
         "  InitialGridPoints: [4]\n"
         "  InitialRefinement: [3]\n" +
         time_dep_options + bc_options;
}

void test_interval() {
  INFO("Interval");
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}};
  const std::vector<std::array<size_t, 1>> refinement_level{{{3}}};
  const std::array<double, 1> lower_bound{{-1.2}};
  const std::array<double, 1> upper_bound{{0.8}};

  const auto lower_boundary_condition = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::lower_xi(), 0);
  const auto upper_boundary_condition = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::upper_xi(), 0);
  const auto periodic_boundary_condition =
      std::make_unique<TestHelpers::domain::BoundaryConditions::
                           TestPeriodicBoundaryCondition<1>>();

  const auto distributions =
      make_array(CoordinateMaps::Distribution::Linear,
                 CoordinateMaps::Distribution::Logarithmic);
  const std::array<std::optional<double>, 2> singularities{{std::nullopt, -2.}};

  const std::array<double, 1> velocity{{2.5}};
  const auto time_dependence = std::make_unique<
      domain::creators::time_dependence::UniformTranslation<1>>(0., velocity);
  const std::vector<double> times{0., 1.5};

  for (const auto& [is_periodic, distribution_index, time_dependent] :
       cartesian_product(make_array(true, false), make_array(0, 1),
                         make_array(true, false))) {
    CAPTURE(is_periodic);
    const auto& distribution = gsl::at(distributions, distribution_index);
    const auto& singularity = gsl::at(singularities, distribution_index);
    CAPTURE(distribution);
    CAPTURE(singularity);
    CAPTURE(time_dependent);

    const creators::Interval interval{
        lower_bound,
        upper_bound,
        refinement_level[0],
        grid_points[0],
        {{is_periodic}},
        {{{distribution, singularity}}},
        time_dependent ? time_dependence->get_clone() : nullptr};
    test_interval_construction(
        interval, false, is_periodic, lower_bound, upper_bound, times,
        time_dependent ? velocity : std::array<double, 1>{{0.}});
    TestHelpers::domain::creators::test_creation(
        option_string(distribution, singularity, is_periodic, time_dependent,
                      false),
        interval, false);

    const creators::Interval interval_with_bc{
        lower_bound,
        upper_bound,
        refinement_level[0],
        grid_points[0],
        {{{{is_periodic ? periodic_boundary_condition->get_clone()
                        : lower_boundary_condition->get_clone(),
            is_periodic ? periodic_boundary_condition->get_clone()
                        : upper_boundary_condition->get_clone()}}}},
        {{{distribution, singularity}}},
        time_dependent ? time_dependence->get_clone() : nullptr};
    test_interval_construction(
        interval_with_bc, true, is_periodic, lower_bound, upper_bound, times,
        time_dependent ? velocity : std::array<double, 1>{{0.}});
    TestHelpers::domain::creators::test_creation(
        option_string(distribution, singularity, is_periodic, time_dependent,
                      true),
        interval_with_bc, true);
  }
}

void test_parse_errors() {
  INFO("Test parse errors");

  const std::array<double, 1> lower_bound{{-1.2}};
  const std::array<double, 1> upper_bound{{0.8}};
  const std::vector<std::array<size_t, 1>> refinement_level{{{3}}};
  const std::vector<std::array<size_t, 1>> grid_points{{{4}}};

  const auto lower_boundary_condition = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::lower_xi(), 0);
  const auto upper_boundary_condition = std::make_unique<
      TestHelpers::domain::BoundaryConditions::TestBoundaryCondition<1>>(
      Direction<1>::upper_xi(), 0);
  const auto periodic_boundary_condition =
      std::make_unique<TestHelpers::domain::BoundaryConditions::
                           TestPeriodicBoundaryCondition<1>>();

  CHECK_THROWS_WITH(
      creators::Interval(
          lower_bound, upper_bound, refinement_level[0], grid_points[0],
          {{{{lower_boundary_condition->get_clone(),
              periodic_boundary_condition->get_clone()}}}},
          {{{CoordinateMaps::Distribution::Linear, std::nullopt}}}, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Periodic boundary conditions must be applied for both upper and "
          "lower direction"));
  CHECK_THROWS_WITH(
      creators::Interval(
          lower_bound, upper_bound, refinement_level[0], grid_points[0],
          {{{{periodic_boundary_condition->get_clone(),
              lower_boundary_condition->get_clone()}}}},
          {{{CoordinateMaps::Distribution::Linear, std::nullopt}}}, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "Periodic boundary conditions must be applied for both upper and "
          "lower direction"));
  CHECK_THROWS_WITH(
      creators::Interval(
          lower_bound, upper_bound, refinement_level[0], grid_points[0],
          {{{{lower_boundary_condition->get_clone(),
              std::make_unique<TestHelpers::domain::BoundaryConditions::
                                   TestNoneBoundaryCondition<3>>()}}}},
          {{{CoordinateMaps::Distribution::Linear, std::nullopt}}}, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
  CHECK_THROWS_WITH(
      creators::Interval(
          lower_bound, upper_bound, refinement_level[0], grid_points[0],
          {{{{std::make_unique<TestHelpers::domain::BoundaryConditions::
                                   TestNoneBoundaryCondition<3>>(),
              lower_boundary_condition->get_clone()}}}},
          {{{CoordinateMaps::Distribution::Linear, std::nullopt}}}, nullptr,
          Options::Context{false, {}, 1, 1}),
      Catch::Matchers::ContainsSubstring(
          "None boundary condition is not supported. If you would like an "
          "outflow-type boundary condition, you must use that."));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Creators.Interval", "[Domain][Unit]") {
  test_interval();
  test_parse_errors();
}
}  // namespace domain
