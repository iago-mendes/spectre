// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Inboxes.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/ReceiveElementData.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Time/TimeStepId.hpp"

namespace CurvedScalarWave::Worldtube::Actions {

/*!
 * \brief Sends the acceleration terms to worldtube neighbors
 */
template <typename Metavariables>
struct SendAccelerationTerms {
  static constexpr size_t Dim = Metavariables::volume_dim;
  using simple_tags = tmpl::list<Tags::AccelerationTerms>;
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& faces_grid_coords =
        get<Tags::ElementFacesGridCoordinates<Dim>>(box);
    auto& element_proxies = Parallel::get_parallel_component<
        typename Metavariables::dg_element_array>(cache);
    for (const auto& [element_id, _] : faces_grid_coords) {
      auto data_to_send = db::get<Tags::AccelerationTerms>(box);
      Parallel::receive_data<Tags::SelfForceInbox<Dim>>(
          element_proxies[element_id], db::get<::Tags::TimeStepId>(box),
          std::move(data_to_send));
    }
    return {Parallel::AlgorithmExecution::Continue,
            tmpl::index_of<ActionList, ReceiveElementData>::value};
  }
};
}  // namespace CurvedScalarWave::Worldtube::Actions
