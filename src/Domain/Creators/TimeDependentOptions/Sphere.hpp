// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotScaleTrans.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Shape.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/TimeDependentOptions/ShapeMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Grid;
struct Distorted;
struct Inertial;
}  // namespace Frame
/// \endcond

namespace domain::creators::sphere {
/*!
 * \brief This holds all options related to the time dependent maps of the
 * domain::creators::Sphere domain creator.
 */
struct TimeDependentMapOptions {
 private:
  template <typename SourceFrame, typename TargetFrame>
  using MapType =
      std::unique_ptr<domain::CoordinateMapBase<SourceFrame, TargetFrame, 3>>;
  using IdentityMap = domain::CoordinateMaps::Identity<3>;
  // Time-dependent maps
  using ShapeMap = domain::CoordinateMaps::TimeDependent::Shape;
  using RotScaleTransMap =
      domain::CoordinateMaps::TimeDependent::RotScaleTrans<3>;

  template <typename SourceFrame, typename TargetFrame>
  using IdentityForComposition =
      domain::CoordinateMap<SourceFrame, TargetFrame, IdentityMap>;
  using GridToDistortedComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Distorted, ShapeMap>;
  using GridToInertialComposition =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap,
                            RotScaleTransMap>;
  using GridToInertialSimple =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotScaleTransMap>;
  using DistortedToInertialComposition =
      domain::CoordinateMap<Frame::Distorted, Frame::Inertial,
                            RotScaleTransMap>;
  using GridToInertialShapeMap =
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, ShapeMap>;

 public:
  using maps_list =
      tmpl::list<IdentityForComposition<Frame::Grid, Frame::Inertial>,
                 IdentityForComposition<Frame::Grid, Frame::Distorted>,
                 IdentityForComposition<Frame::Distorted, Frame::Inertial>,
                 GridToDistortedComposition, GridToInertialShapeMap,
                 GridToInertialSimple, GridToInertialComposition,
                 DistortedToInertialComposition>;

  /// \brief The initial time of the functions of time.
  struct InitialTime {
    using type = double;
    static constexpr Options::String help = {
        "The initial time of the functions of time"};
  };

  using ShapeMapOptions =
      time_dependent_options::ShapeMapOptions<false, domain::ObjectLabel::None>;

  struct RotationMapOptions {
    using type = Options::Auto<RotationMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "RotationMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent rotation map about an arbitrary axis. "
        "Specify 'None' to not use this map."};

    struct InitialValues {
      using type = std::array<std::array<double, 4>, 3>;
      static constexpr Options::String help = {
          "The initial values for the quaternion function and its first two "
          "derivatives."};
    };

    struct DecayTimescaleRotation {
      using type = double;
      static constexpr Options::String help = {
          "The timescale for how fast the rotation approaches its asymptotic "
          "value."};
    };

    using options = tmpl::list<InitialValues, DecayTimescaleRotation>;
    RotationMapOptions();
    RotationMapOptions(std::array<std::array<double, 4>, 3> initial_values_in,
                       double decay_timescale_in);

    std::array<std::array<double, 4>, 3> initial_values{};
    double decay_timescale{std::numeric_limits<double>::signaling_NaN()};
  };

  struct ExpansionMapOptions {
    using type = Options::Auto<ExpansionMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "ExpansionMap"; }
    static constexpr Options::String help = {
        "Options for the expansion map. Specify 'None' to not use this map."};

    struct InitialValues {
      using type = std::array<double, 3>;
      static constexpr Options::String help = {
          "Initial value and first two derivatives of expansion."};
    };

    struct DecayTimescaleExpansion {
      using type = double;
      static constexpr Options::String help = {
          "The timescale for how fast the expansion approaches "
          "its asymptotic value."};
    };

    struct InitialValuesOuterBoundary {
      using type = std::array<double, 3>;
      static constexpr Options::String help = {
          "Initial value and first two derivatives of expansion outside the "
          "transition region."};
    };

    struct DecayTimescaleExpansionOuterBoundary {
      using type = double;
      static constexpr Options::String help = {
          "The timescale for how fast the expansion approaches "
          "its asymptotic value outside the transition region."};
    };

    using options = tmpl::list<InitialValues, DecayTimescaleExpansion,
                               InitialValuesOuterBoundary,
                               DecayTimescaleExpansionOuterBoundary>;
    ExpansionMapOptions();
    ExpansionMapOptions(std::array<double, 3> initial_values_in,
                        double decay_timescale_in,
                        std::array<double, 3> initial_values_outer_boundary_in,
                        double decay_timescale_outer_boundary_in);

    std::array<double, 3> initial_values{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double decay_timescale{std::numeric_limits<double>::signaling_NaN()};
    std::array<double, 3> initial_values_outer_boundary{
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN(),
        std::numeric_limits<double>::signaling_NaN()};
    double decay_timescale_outer_boundary{
        std::numeric_limits<double>::signaling_NaN()};
  };

  struct TranslationMapOptions {
    using type = Options::Auto<TranslationMapOptions, Options::AutoLabel::None>;
    static std::string name() { return "TranslationMap"; }
    static constexpr Options::String help = {
        "Options for a time-dependent translation map in that keeps the "
        "outer boundary fixed. Specify 'None' to not use this map."};

    struct InitialValues {
      using type = std::array<std::array<double, 3>, 3>;
      static constexpr Options::String help = {
          "Initial values for the translation map, its velocity and "
          "acceleration."};
    };

    using options = tmpl::list<InitialValues>;
    TranslationMapOptions();
    explicit TranslationMapOptions(
        std::array<std::array<double, 3>, 3> initial_values_in);

    std::array<std::array<double, 3>, 3> initial_values{};
  };

  struct TransitionRotScaleTrans {
    using type = bool;
    static constexpr Options::String help = {
        "Transition rotation, expansion, and translation to zero in the outer "
        "shell"};
  };

  using options = tmpl::list<InitialTime, ShapeMapOptions, RotationMapOptions,
                             ExpansionMapOptions, TranslationMapOptions,
                             TransitionRotScaleTrans>;
  static constexpr Options::String help{
      "The options for all the hard-coded time dependent maps in the "
      "Sphere domain."};

  TimeDependentMapOptions() = default;

  TimeDependentMapOptions(
      double initial_time, std::optional<ShapeMapOptions> shape_map_options,
      std::optional<RotationMapOptions> rotation_map_options,
      std::optional<ExpansionMapOptions> expansion_map_options,
      std::optional<TranslationMapOptions> translation_map_options,
      bool transition_rot_scale_trans);

  /*!
   * \brief Create the function of time map using the options that were
   * provided to this class.
   *
   * Currently, this will add:
   *
   * - Size: `PiecewisePolynomial<3>`
   * - Shape: `PiecewisePolynomial<2>`
   * - Rotation: `SettleToConstantQuaternion`
   * - Expansion: `SettleToConstant`
   * - ExpansionOuterBoundary: `PiecewisePolynomial<2>`
   * - Translation: `PiecewisePolynomial<2>`
   */
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
  create_functions_of_time(const std::unordered_map<std::string, double>&
                               initial_expiration_times) const;

  /*!
   * \brief Construct the actual maps that will be used.
   *
   * Currently, this constructs a:
   *
   * - Shape: `Shape` (with a size function of time)
   * - Rotation: `Rotation`
   * - Expansion: `Expansion`
   * - Expansion outside the transition region: `ExpansionOuterBoundary`
   * - Translation: `Translation`
   */
  void build_maps(const std::array<double, 3>& center, bool is_filled,
                  double inner_radius,
                  const std::vector<double>& radial_partitions,
                  double outer_radius);

  /*!
   * \brief This will construct the map from `Frame::Distorted` to
   * `Frame::Inertial`.
   *
   * For blocks with a shape map, this will be a RotScaleTrans map. For other
   * blocks, this returns `nullptr`.
   */
  MapType<Frame::Distorted, Frame::Inertial> distorted_to_inertial_map(
      size_t block_number, bool is_inner_cube) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to
   * `Frame::Distorted`.
   *
   * For blocks with a shape map, this will return the `Shape` map (with a size
   * function of time). For other blocks, this returns `nullptr`.
   */
  MapType<Frame::Grid, Frame::Distorted> grid_to_distorted_map(
      size_t block_number, bool is_inner_cube) const;

  /*!
   * \brief This will construct the map from `Frame::Grid` to `Frame::Inertial`.
   *
   * For blocks with a shape map, this will return the `Shape` and
   * `RotScaleTrans` composition. For other blocks, this returns just the
   * `RotScaleTrans` map. In the outer shell, the `RotScaleTrans` map will
   * transition to zero.
   */
  MapType<Frame::Grid, Frame::Inertial> grid_to_inertial_map(
      size_t block_number, bool is_outer_shell, bool is_inner_cube) const;

  /*!
   * \brief Whether or not the distorted frame is being used. I.e. whether or
   * not shape map options were specified.
   */
  bool using_distorted_frame() const;

  inline static const std::string size_name{"Size"};
  inline static const std::string shape_name{"Shape"};
  inline static const std::string rotation_name{"Rotation"};
  inline static const std::string expansion_name{"Expansion"};
  inline static const std::string expansion_outer_boundary_name{
      "ExpansionOuterBoundary"};
  inline static const std::string translation_name{"Translation"};

 private:
  double initial_time_{std::numeric_limits<double>::signaling_NaN()};
  bool filled_{false};
  double deformed_radius_{std::numeric_limits<double>::signaling_NaN()};
  std::array<ShapeMap, 12> shape_maps_{};
  RotScaleTransMap inner_rot_scale_trans_map_{};
  RotScaleTransMap transition_rot_scale_trans_map_{};

  std::optional<ShapeMapOptions> shape_map_options_{};
  std::optional<RotationMapOptions> rotation_map_options_{};
  std::optional<ExpansionMapOptions> expansion_map_options_{};
  std::optional<TranslationMapOptions> translation_map_options_{};
  bool transition_rot_scale_trans_{false};
};
}  // namespace domain::creators::sphere
