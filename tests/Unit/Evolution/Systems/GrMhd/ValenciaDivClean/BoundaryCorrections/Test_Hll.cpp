// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <optional>
#include <random>
#include <string>
#include <tuple>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/System.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/BoundaryCorrections.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace {
namespace helpers = TestHelpers::evolution::dg;

struct ConvertPolytropic {
  using unpacked_container = bool;
  using packed_container = EquationsOfState::EquationOfState<true, 3>;
  using packed_type = bool;

  static inline unpacked_container unpack(const packed_container& /*packed*/,
                                          const size_t /*grid_point_index*/) {
    return true;
  }

  [[noreturn]] static inline void pack(
      const gsl::not_null<packed_container*> /*packed*/,
      const unpacked_container& /*unpacked*/,
      const size_t /*grid_point_index*/) {
    ERROR("Should not be converting an EOS from an unpacked to a packed type");
  }

  static inline size_t get_size(const packed_container& /*packed*/) {
    return 1;
  }
};

// With B = 0 the packaged fast speeds must BE the hydro speeds.
//
// `dg_package_data`'s per-side fast-magnetosonic block is otherwise untested:
// the pypp reference below runs only on curved backgrounds, where the fast
// speeds fall back to the light speed and the block never executes at all.
//
// With no magnetic field the Alfven speed vanishes, so
// a^2 = c_s^2 + v_A^2 (1 - c_s^2) collapses to c_s^2 and the fast speeds must
// come out equal to the sound-speed branch that the SAME call computes for
// the Largest speeds through entirely separate code. That pins the two speed
// indices (7 outgoing, 1 ingoing), the sign convention and the handling of
// the interface normal -- get any of them wrong and these disagree.
void test_fast_speeds_reduce_to_hydro() {
  INFO("HLL: with B = 0 the packaged fast speeds are the hydro speeds");
  const size_t num_points = 4;
  const DataVector used_for_size{num_points};
  const auto eos =
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos();

  const Scalar<DataVector> rest_mass_density{DataVector{1.0, 0.5, 2.0, 0.8}};
  const Scalar<DataVector> electron_fraction{DataVector{num_points, 0.1}};
  const Scalar<DataVector> temperature{DataVector{num_points, 0.0}};
  const Scalar<DataVector> specific_internal_energy =
      eos->specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const Scalar<DataVector> pressure = eos->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);

  tnsr::I<DataVector, 3, Frame::Inertial> spatial_velocity{num_points, 0.0};
  get<0>(spatial_velocity) = DataVector{0.10, -0.25, 0.05, 0.30};
  get<1>(spatial_velocity) = DataVector{-0.20, 0.15, 0.00, 0.10};
  get<2>(spatial_velocity) = DataVector{0.05, 0.05, -0.10, 0.00};
  // Flat space, so the one-form has the same components as the vector.
  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{num_points,
                                                                    0.0};
  DataVector v_squared{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_velocity_one_form.get(i) = spatial_velocity.get(i);
    v_squared += spatial_velocity.get(i) * spatial_velocity.get(i);
  }
  const Scalar<DataVector> lorentz_factor{1.0 / sqrt(1.0 - v_squared)};

  // Minkowski, so the fast-speed block is live.
  const Scalar<DataVector> lapse{DataVector{num_points, 1.0}};
  const tnsr::I<DataVector, 3, Frame::Inertial> shift{num_points, 0.0};
  tnsr::i<DataVector, 3, Frame::Inertial> normal_covector{num_points, 0.0};
  get<0>(normal_covector) = DataVector{num_points, 0.6};
  get<1>(normal_covector) = DataVector{num_points, 0.8};
  const tnsr::I<DataVector, 3, Frame::Inertial> normal_vector{num_points, 0.0};

  // No magnetic field: this is what makes the two branches comparable.
  const Scalar<DataVector> zero_scalar{DataVector{num_points, 0.0}};
  const tnsr::i<DataVector, 3, Frame::Inertial> zero_lower{num_points, 0.0};
  const tnsr::I<DataVector, 3, Frame::Inertial> zero_upper{num_points, 0.0};
  const tnsr::Ij<DataVector, 3, Frame::Inertial> zero_mixed{num_points, 0.0};
  const tnsr::IJ<DataVector, 3, Frame::Inertial> zero_upper2{num_points, 0.0};

  Scalar<DataVector> p_tilde_d{}, p_tilde_ye{}, p_tilde_tau{}, p_tilde_phi{};
  Scalar<DataVector> p_nf_d{}, p_nf_ye{}, p_nf_tau{}, p_nf_phi{};
  tnsr::i<DataVector, 3, Frame::Inertial> p_tilde_s{}, p_nf_s{}, p_normal{};
  tnsr::I<DataVector, 3, Frame::Inertial> p_tilde_b{}, p_nf_b{};
  Scalar<DataVector> p_largest_out{}, p_largest_in{}, p_fast_out{}, p_fast_in{},
      p_flatness{};
  for (auto* scalar : {&p_tilde_d, &p_tilde_ye, &p_tilde_tau, &p_tilde_phi,
                       &p_nf_d, &p_nf_ye, &p_nf_tau, &p_nf_phi, &p_largest_out,
                       &p_largest_in, &p_fast_out, &p_fast_in, &p_flatness}) {
    *scalar = Scalar<DataVector>{DataVector{num_points, 0.0}};
  }
  p_tilde_s = zero_lower;
  p_nf_s = zero_lower;
  p_normal = zero_lower;
  p_tilde_b = zero_upper;
  p_nf_b = zero_upper;

  grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8}
      .dg_package_data(
          make_not_null(&p_tilde_d), make_not_null(&p_tilde_ye),
          make_not_null(&p_tilde_tau), make_not_null(&p_tilde_s),
          make_not_null(&p_tilde_b), make_not_null(&p_tilde_phi),
          make_not_null(&p_nf_d), make_not_null(&p_nf_ye),
          make_not_null(&p_nf_tau), make_not_null(&p_nf_s),
          make_not_null(&p_nf_b), make_not_null(&p_nf_phi),
          make_not_null(&p_largest_out), make_not_null(&p_largest_in),
          make_not_null(&p_fast_out), make_not_null(&p_fast_in),
          make_not_null(&p_normal), make_not_null(&p_flatness),

          zero_scalar, zero_scalar, zero_scalar, zero_lower, zero_upper,
          zero_scalar,

          zero_upper, zero_upper, zero_upper, zero_mixed, zero_upper2,
          zero_upper,

          lapse, shift, spatial_velocity_one_form,

          rest_mass_density, electron_fraction, temperature, spatial_velocity,
          specific_internal_energy, pressure, lorentz_factor,

          normal_covector, normal_vector, std::nullopt, std::nullopt, *eos);

  Approx speed_approx = Approx::custom().epsilon(1.0e-13).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(get(p_fast_out), get(p_largest_out),
                               speed_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(get(p_fast_in), get(p_largest_in), speed_approx);
  // And they are genuine sound speeds, not the light speed that the curved
  // fallback would have left behind.
  CHECK(max(abs(get(p_fast_out))) < 1.0);
  CHECK(max(abs(get(p_fast_in))) < 1.0);
  CHECK(max(get(p_flatness)) == approx(0.0));
}

// Two-sided signal speeds: the int<->ext swap must flip the sign of the flux.
//
// The conservation harness below checks F*(int, ext) == -F*(ext, int) too, but
// only on random states whose background is CURVED -- its lapse range is
// [0.3, 1.0] and its shift [0.01, 0.02], so `metric_flatness` never drops to
// 1e-12. Where the background is curved the fast-magnetosonic bounds sit at
// the light speed and the magnetic field takes its unsplit fallback, so
// neither the distinct fast bounds nor the normal/tangential split is ever
// evaluated there. This checks the same property on a FLAT interface with
// fast speeds strictly inside the light speeds, which is the configuration
// every shock tube, Kelvin-Helmholtz and jet run actually uses.
//
// It is the standing guard against the bounds being rebuilt one-sidedly.
// Shown discriminating on 2026-09-11: replacing
//   fast_max = max(0, fast_out_int, -fast_in_ext)
// by the one-sided
//   fast_max = max(0, fast_out_int)
// (and likewise for fast_min) makes this check fail.
void test_int_ext_swap_symmetry() {
  INFO("HLL: two-sided fast bounds, antisymmetric under the int<->ext swap");
  MAKE_GENERATOR(gen);
  const DataVector used_for_size{5};
  std::uniform_real_distribution<> dist(0.5, 1.5);
  std::uniform_real_distribution<> speed_dist(0.2, 0.9);

  const auto scalar = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<Scalar<DataVector>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto covector = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<tnsr::i<DataVector, 3, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto vector = [&gen, &dist, &used_for_size]() {
    return make_with_random_values<tnsr::I<DataVector, 3, Frame::Inertial>>(
        make_not_null(&gen), make_not_null(&dist), used_for_size);
  };
  const auto speed = [&gen, &speed_dist, &used_for_size](const double sign) {
    auto result = make_with_random_values<Scalar<DataVector>>(
        make_not_null(&gen), make_not_null(&speed_dist), used_for_size);
    get(result) *= sign;
    return result;
  };

  struct Packet {
    Scalar<DataVector> tilde_d;
    Scalar<DataVector> tilde_ye;
    Scalar<DataVector> tilde_tau;
    tnsr::i<DataVector, 3, Frame::Inertial> tilde_s;
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_b;
    Scalar<DataVector> tilde_phi;
    Scalar<DataVector> nf_d;
    Scalar<DataVector> nf_ye;
    Scalar<DataVector> nf_tau;
    tnsr::i<DataVector, 3, Frame::Inertial> nf_s;
    tnsr::I<DataVector, 3, Frame::Inertial> nf_b;
    Scalar<DataVector> nf_phi;
    Scalar<DataVector> largest_out;
    Scalar<DataVector> largest_in;
    Scalar<DataVector> fast_out;
    Scalar<DataVector> fast_in;
    tnsr::i<DataVector, 3, Frame::Inertial> normal;
    Scalar<DataVector> flatness;
  };

  const auto make_packet = [&scalar, &covector, &vector, &speed,
                            &used_for_size](const double normal_sign) {
    Packet p{};
    p.tilde_d = scalar();
    p.tilde_ye = scalar();
    p.tilde_tau = scalar();
    p.tilde_s = covector();
    p.tilde_b = vector();
    p.tilde_phi = scalar();
    p.nf_d = scalar();
    p.nf_ye = scalar();
    p.nf_tau = scalar();
    p.nf_s = covector();
    p.nf_b = vector();
    p.nf_phi = scalar();
    // Flat background: light speed is +/-1 against each side's own outward
    // normal, and the fast speeds are strictly inside it.
    p.largest_out = Scalar<DataVector>{DataVector{used_for_size.size(), 1.0}};
    p.largest_in = Scalar<DataVector>{DataVector{used_for_size.size(), -1.0}};
    p.fast_out = speed(1.0);
    p.fast_in = speed(-1.0);
    // A unit covector; the neighbour's points the other way.
    p.normal =
        tnsr::i<DataVector, 3, Frame::Inertial>{used_for_size.size(), 0.0};
    get<0>(p.normal) = normal_sign * 0.6;
    get<1>(p.normal) = normal_sign * 0.8;
    p.flatness = Scalar<DataVector>{DataVector{used_for_size.size(), 0.0}};
    return p;
  };

  const Packet interior = make_packet(1.0);
  const Packet exterior = make_packet(-1.0);

  const auto boundary_terms = [](const Packet& in, const Packet& ex) {
    std::array<Scalar<DataVector>, 4> scalars{};
    tnsr::i<DataVector, 3, Frame::Inertial> s{get(in.tilde_d).size(), 0.0};
    tnsr::I<DataVector, 3, Frame::Inertial> b{get(in.tilde_d).size(), 0.0};
    for (auto& entry : scalars) {
      entry = Scalar<DataVector>{DataVector{get(in.tilde_d).size(), 0.0}};
    }
    grmhd::ValenciaDivClean::BoundaryCorrections::Hll::dg_boundary_terms(
        make_not_null(&scalars[0]), make_not_null(&scalars[1]),
        make_not_null(&scalars[2]), make_not_null(&s), make_not_null(&b),
        make_not_null(&scalars[3]),

        in.tilde_d, in.tilde_ye, in.tilde_tau, in.tilde_s, in.tilde_b,
        in.tilde_phi, in.nf_d, in.nf_ye, in.nf_tau, in.nf_s, in.nf_b, in.nf_phi,
        in.largest_out, in.largest_in, in.fast_out, in.fast_in, in.normal,
        in.flatness,

        ex.tilde_d, ex.tilde_ye, ex.tilde_tau, ex.tilde_s, ex.tilde_b,
        ex.tilde_phi, ex.nf_d, ex.nf_ye, ex.nf_tau, ex.nf_s, ex.nf_b, ex.nf_phi,
        ex.largest_out, ex.largest_in, ex.fast_out, ex.fast_in, ex.normal,
        ex.flatness,

        dg::Formulation::WeakInertial);
    return std::make_tuple(scalars, s, b);
  };

  const auto forward = boundary_terms(interior, exterior);
  const auto swapped = boundary_terms(exterior, interior);

  Approx swap_approx = Approx::custom().epsilon(1.0e-14).scale(1.0);
  for (size_t i = 0; i < 4; ++i) {
    CAPTURE(i);
    const DataVector negated = -get(std::get<0>(swapped)[i]);
    CHECK_ITERABLE_CUSTOM_APPROX(get(std::get<0>(forward)[i]), negated,
                                 swap_approx);
  }
  for (size_t i = 0; i < 3; ++i) {
    CAPTURE(i);
    const DataVector negated_s = -std::get<1>(swapped).get(i);
    CHECK_ITERABLE_CUSTOM_APPROX(std::get<1>(forward).get(i), negated_s,
                                 swap_approx);
    const DataVector negated_b = -std::get<2>(swapped).get(i);
    CHECK_ITERABLE_CUSTOM_APPROX(std::get<2>(forward).get(i), negated_b,
                                 swap_approx);
  }
}

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.BoundaryCorrections.Hll",
                  "[Unit][GrMhd]") {
  PUPable_reg(grmhd::ValenciaDivClean::BoundaryCorrections::Hll);
  test_fast_speeds_reduce_to_hydro();
  test_int_ext_swap_symmetry();
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections"};
  MAKE_GENERATOR(gen);

  using system = grmhd::ValenciaDivClean::System;

  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges{std::array{0.3, 1.0}, std::array{0.01, 0.02}};
  const tuples::TaggedTuple<hydro::Tags::GrmhdEquationOfState> volume_data{
      EquationsOfState::PolytropicFluid<true>{100.0, 2.0}.promote_to_3d_eos()};

  TestHelpers::evolution::dg::test_boundary_correction_conservation<system>(
      make_not_null(&gen),
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  // Test hydro
  const tuples::TaggedTuple<
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>,
      helpers::Tags::Range<grmhd::ValenciaDivClean::Tags::TildeB<>>>
      ranges_hydro{std::array{0.3, 1.0}, std::array{0.01, 0.02},
                   std::array{1.0e-25, 1.0e-20}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_hydro);

  // Test light speed density cutoff
  const tuples::TaggedTuple<
      helpers::Tags::Range<hydro::Tags::RestMassDensity<DataVector>>,
      helpers::Tags::Range<gr::Tags::Lapse<DataVector>>,
      helpers::Tags::Range<gr::Tags::Shift<DataVector, 3>>>
      ranges_atmo{std::array{1.0e-10, 1.0e-9}, std::array{0.3, 1.0},
                  std::array{0.01, 0.02}};
  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8},
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges_atmo);

  const auto hll = TestHelpers::test_factory_creation<
      evolution::BoundaryCorrection,
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll>(
      "Hll:\n"
      "  MagneticFieldMagnitudeForHydro: 1.0e-30\n"
      "  LightSpeedDensityCutoff: 1.0e-8\n");

  TestHelpers::evolution::dg::test_boundary_correction_with_python<
      system, tmpl::list<ConvertPolytropic>>(
      make_not_null(&gen), "Hll", "dg_package_data", "dg_boundary_terms",
      dynamic_cast<const grmhd::ValenciaDivClean::BoundaryCorrections::Hll&>(
          *hll),
      Mesh<2>{5, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      volume_data, ranges);

  CHECK_FALSE(
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
      grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{2.0e-30, 1.0e-8});
  CHECK(grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 1.0e-8} !=
        grmhd::ValenciaDivClean::BoundaryCorrections::Hll{1.0e-30, 2.0e-8});
}
}  // namespace
