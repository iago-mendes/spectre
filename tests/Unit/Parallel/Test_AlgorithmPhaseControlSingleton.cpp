// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Parallel/Test_AlgorithmPhaseControl.hpp"

#include "Parallel/Algorithms/AlgorithmSingleton.hpp"

extern "C" void CkRegisterMainModule() {
  Parallel::charmxx::register_main_module<
      TestMetavariables<Parallel::Algorithms::Singleton>>();
  Parallel::charmxx::register_init_node_and_proc(
      {&register_factory_classes_with_charm<
          TestMetavariables<Parallel::Algorithms::Singleton>>},
      {});
}
