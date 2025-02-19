// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "DataStructures/VariablesTag.hpp"
#include "Evolution/Actions/RunEventsAndTriggers.hpp"
#include "Evolution/Systems/Cce/Actions/BoundaryComputeAndSendToEvolution.hpp"
#include "Evolution/Systems/Cce/Actions/CalculateScriInputs.hpp"
#include "Evolution/Systems/Cce/Actions/CharacteristicEvolutionBondiCalculations.hpp"
#include "Evolution/Systems/Cce/Actions/FilterSwshVolumeQuantity.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionScri.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionTime.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeCharacteristicEvolutionVariables.hpp"
#include "Evolution/Systems/Cce/Actions/InitializeFirstHypersurface.hpp"
#include "Evolution/Systems/Cce/Actions/InsertInterpolationScriData.hpp"
#include "Evolution/Systems/Cce/Actions/Psi0Matching.hpp"
#include "Evolution/Systems/Cce/Actions/RequestBoundaryData.hpp"
#include "Evolution/Systems/Cce/Actions/ScriObserveInterpolated.hpp"
#include "Evolution/Systems/Cce/Actions/TimeManagement.hpp"
#include "Evolution/Systems/Cce/Actions/UpdateGauge.hpp"
#include "Evolution/Systems/Cce/LinearSolve.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/PrecomputeCceDependencies.hpp"
#include "Evolution/Systems/Cce/ScriPlusValues.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/System.hpp"
#include "IO/Observer/ObserverComponent.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/Goto.hpp"
#include "ParallelAlgorithms/Actions/MutateApply.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Time/Actions/AdvanceTime.hpp"
#include "Time/Actions/ChangeStepSize.hpp"
#include "Time/Actions/CleanHistory.hpp"
#include "Time/Actions/RecordTimeStepperData.hpp"
#include "Time/Actions/SelfStartActions.hpp"
#include "Time/Actions/UpdateU.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {

struct CceEvolutionLabelTag {};

/*!
 * \brief The component for handling the CCE evolution and waveform output.
 *
 * \details The \ref DataBoxGroup associated with the CharacteristicEvolution
 * will contain many spin-weighted volume tags associated with the ongoing CCE
 * computation, as well as storage for the boundary values and quantities
 * related to managing the evolution.
 *
 * Metavariables requirements:
 * - Phases:
 *  - `Initialization`
 *  - `Evolve`
 * - Type aliases:
 *  - `evolved_coordinates_variables_tag`: A `Tags::Variables` with real-valued
 * tensors associated with coordinates that must be evolved.
 *  - `evolved_swsh_tags`: The spin-weighted quantities to be evolved (typically
 * `BondiJ`).
 *  - `evolved_swsh_dt_tags`: The spin-weighed quantities associated that are to
 * act as the time derivative to evolve `evolved_swsh_tags` (typically
 * `BondiH`).
 *  - `cce_boundary_communication_tags`: A typelist of tags that will be
 * communicated between the worldtube boundary component and the extraction
 * component (typically
 * `Cce::Tags::characteristic_worldtube_boundary_tags<Tags::BoundaryValue>`)
 *  - `cce_gauge_boundary_tags`: A typelist of tags that will be derived via
 * `GaugeAdjustedBoundaryValue` and corresponding gauge utilities
 *  - `cce_integrand_tags`: A typelist of tags needed as inputs to the
 * linear solve. Obtainable from the metafunction
 * `Cce::integrand_terms_to_compute_for_bondi_variable`.
 *  - `cce_integration_independent_tags`: A typelist of tags that are to be
 * computed and stored for each hypersurface iteration, but have no dependencies
 * on the intermediate hypersurface steps (typically
 * `Cce::pre_computation_steps`).
 *  - `cce_temporary_equations_tags`: A typelist of temporary buffers maintained
 * for intermediate steps in the integrand equations. Obtainable from the
 * metafunction `Cce::integrand_terms_to_compute_for_bondi_variable`.
 *  - `cce_pre_swsh_derivatives_tags`: A typelist of inputs to spin-weighted
 * derivative calculations to compute and cache for intermediate steps of the
 * CCE calculation. (typically `Cce::all_pre_swsh_derivative_tags`)
 *  - `cce_swsh_derivative_tags`: A typelist of spin-weighted derivatives to
 * compute and cache for intermediate steps of the CCE calculation. (typically
 * `Cce::all_swsh_derivative_tags`)
 *  - `cce_transform_buffer_tags`: A typelist of spin-weighted spherical
 * harmonic transform modes used to compute the spin-weighted derivatives in the
 * modal representation. (typically `Cce::all_transform_buffer_tags`).
 *  - `cce_angular_coordinate_tags`: A typelist of real-valued angular
 * coordinates that are not evolved.
 *  - `cce_scri_tags`: the tags of quantities to compute at scri+
 *  - `cce_hypersurface_initialization`: a mutator (for use with
 * `::Actions::MutateApply`) that is used to compute the initial hypersurface
 * data from the boundary data.
 */
template <class Metavariables>
struct CharacteristicEvolution {
  using chare_type = Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  static constexpr bool evolve_ccm = Metavariables::evolve_ccm;
  using cce_system = Cce::System<evolve_ccm>;

  using initialize_action_list = tmpl::list<
      Actions::InitializeCharacteristicEvolutionVariables<Metavariables>,
      Actions::InitializeCharacteristicEvolutionTime<
          typename Metavariables::evolved_coordinates_variables_tag,
          typename Metavariables::evolved_swsh_tags,
          Metavariables::local_time_stepping>,
      Actions::InitializeCharacteristicEvolutionScri<
          typename Metavariables::scri_values_to_observe,
          typename Metavariables::cce_boundary_component>,
      Parallel::Actions::TerminatePhase>;

  using simple_tags_from_options =
      Parallel::get_simple_tags_from_options<initialize_action_list>;

  // the list of actions that occur for each of the hypersurface-integrated
  // Bondi tags
  template <typename BondiTag>
  using hypersurface_computation = tmpl::list<
      ::Actions::MutateApply<GaugeAdjustedBoundaryValue<BondiTag>>,
      Actions::CalculateIntegrandInputsForTag<BondiTag>,
      tmpl::transform<integrand_terms_to_compute_for_bondi_variable<BondiTag>,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<ComputeBondiIntegrand, tmpl::_1>>>,
      ::Actions::MutateApply<
          RadialIntegrateBondi<Tags::EvolutionGaugeBoundaryValue, BondiTag>>,
      // Once we finish the U computation, we need to update all the quantities
      // that depend on the time derivative of the gauge
      tmpl::conditional_t<
          std::is_same_v<BondiTag, Tags::BondiU>,
          tmpl::list<
              ::Actions::MutateApply<GaugeUpdateTimeDerivatives>,
              tmpl::conditional_t<
                  tt::is_a_v<AnalyticWorldtubeBoundary,
                             typename Metavariables::cce_boundary_component>,
                  tmpl::list<>,
                  tmpl::conditional_t<evolve_ccm,
                                      ::Actions::MutateApply<
                                          GaugeUpdateInertialTimeDerivatives>,
                                      tmpl::list<>>>,
              ::Actions::MutateApply<
                  GaugeAdjustedBoundaryValue<Tags::DuRDividedByR>>,
              ::Actions::MutateApply<PrecomputeCceDependencies<
                  Tags::EvolutionGaugeBoundaryValue, Tags::DuRDividedByR>>>,
          tmpl::list<>>>;

  using compute_scri_quantities_and_observe = tmpl::list<
      ::Actions::MutateApply<
          CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>>,
      Actions::CalculateScriInputs,
      tmpl::transform<typename metavariables::cce_scri_tags,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<CalculateScriPlusValue, tmpl::_1>>>,
      tmpl::transform<
          typename metavariables::scri_values_to_observe,
          tmpl::bind<
              Actions::InsertInterpolationScriData, tmpl::_1,
              tmpl::pin<typename Metavariables::cce_boundary_component>>>,
      Actions::ScriObserveInterpolated<
          observers::ObserverWriter<Metavariables>,
          typename Metavariables::cce_boundary_component>>;

  using self_start_extract_action_list = tmpl::list<
      Actions::RequestBoundaryData<
          typename Metavariables::cce_boundary_component,
          CharacteristicEvolution<Metavariables>>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::cce_boundary_communication_tags>,
      // note that the initialization will only actually happen on the
      // iterations immediately following restarts
      Actions::InitializeFirstHypersurface<
          evolve_ccm, typename Metavariables::cce_boundary_component>,
      tmpl::conditional_t<
          tt::is_a_v<AnalyticWorldtubeBoundary,
                     typename Metavariables::cce_boundary_component>,
          Actions::UpdateGauge<false>, Actions::UpdateGauge<evolve_ccm>>,
      Actions::PrecomputeGlobalCceDependencies,
      tmpl::conditional_t<evolve_ccm,
                          Actions::CalculatePsi0AndDerivAtInnerBoundary,
                          tmpl::list<>>,
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      Actions::FilterSwshVolumeQuantity<Tags::BondiH>,
      ::Actions::MutateApply<
          CalculateScriPlusValue<::Tags::dt<Tags::InertialRetardedTime>>>,
      Actions::CalculateScriInputs,
      tmpl::transform<typename metavariables::cce_scri_tags,
                      tmpl::bind<::Actions::MutateApply,
                                 tmpl::bind<CalculateScriPlusValue, tmpl::_1>>>,
      ::Actions::RecordTimeStepperData<cce_system>,
      ::Actions::UpdateU<cce_system>,
      ::Actions::CleanHistory<cce_system, false>>;

  using extract_action_list = tmpl::list<
      Actions::RequestBoundaryData<
          typename Metavariables::cce_boundary_component,
          CharacteristicEvolution<Metavariables>>,
      ::Actions::Label<CceEvolutionLabelTag>,
      tmpl::conditional_t<evolve_ccm, tmpl::list<>,
                          evolution::Actions::RunEventsAndTriggers<
                              Metavariables::local_time_stepping>>,
      Actions::ReceiveWorldtubeData<
          Metavariables,
          typename Metavariables::cce_boundary_communication_tags>,
      Actions::InitializeFirstHypersurface<
          evolve_ccm, typename Metavariables::cce_boundary_component>,
      tmpl::conditional_t<
          tt::is_a_v<AnalyticWorldtubeBoundary,
                     typename Metavariables::cce_boundary_component>,
          Actions::UpdateGauge<false>, Actions::UpdateGauge<evolve_ccm>>,
      Actions::PrecomputeGlobalCceDependencies,
      tmpl::conditional_t<evolve_ccm,
                          Actions::CalculatePsi0AndDerivAtInnerBoundary,
                          tmpl::list<>>,
      tmpl::transform<bondi_hypersurface_step_tags,
                      tmpl::bind<hypersurface_computation, tmpl::_1>>,
      Actions::FilterSwshVolumeQuantity<Tags::BondiH>,
      compute_scri_quantities_and_observe,
      ::Actions::RecordTimeStepperData<cce_system>,
      ::Actions::UpdateU<cce_system>,
      ::Actions::ChangeStepSize<typename Metavariables::cce_step_choosers>,
      ::Actions::CleanHistory<cce_system, false>,
      // We cannot know our next step for certain until after we've performed
      // step size selection, as we may need to reject a step.
      Actions::RequestNextBoundaryData<
          typename Metavariables::cce_boundary_component,
          CharacteristicEvolution<Metavariables>>,
      ::Actions::AdvanceTime, Actions::ExitIfEndTimeReached,
      ::Actions::Goto<CceEvolutionLabelTag>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             initialize_action_list>,
      Parallel::PhaseActions<Parallel::Phase::InitializeTimeStepperHistory,
                             SelfStart::self_start_procedure<
                                 self_start_extract_action_list, cce_system>>,
      Parallel::PhaseActions<Parallel::Phase::Evolve, extract_action_list>>;

  static void initialize(
      Parallel::CProxy_GlobalCache<Metavariables>& /*global_cache*/) {}

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<CharacteristicEvolution<Metavariables>>(
        local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace Cce
