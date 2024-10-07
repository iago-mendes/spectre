## Distributed under the MIT License.
# See LICENSE.txt for details.

# These tests should get moved out of Unit eventually.

set(_COMPOSE_OUTPUT ${CMAKE_BINARY_DIR}/Test_ConvertComposeTableOutput.h5)

add_test(
  NAME "Unit.Executables.ConvertComposeTable"
  COMMAND
  ${SHELL_EXECUTABLE}
  -c
  "rm -f ${_COMPOSE_OUTPUT} && \
   ${CMAKE_BINARY_DIR}/bin/ConvertComposeTable \
   --compose-directory ${CMAKE_SOURCE_DIR}/tests/Unit/IO/ \
   --output ${_COMPOSE_OUTPUT} \
   --eos-subfile \"/EosTable\" \
   && rm ${_COMPOSE_OUTPUT}"
  )
set_standalone_test_properties("Unit.Executables.ConvertComposeTable")
set_tests_properties(
    "Unit.Executables.ConvertComposeTable"
    PROPERTIES
    LABELS "standalone;unit")
add_dependencies(
  unit-tests
  ConvertComposeTable
  )
add_dependencies(
  test-executables
  ConvertComposeTable
  )

add_standalone_test(
  "Unit.Executables.TimeStepperSummary"
  EXECUTABLE "TimeStepperSummary"
  REGEX_TO_MATCH "AdamsBashforth\\[2\\][^\\n]*N *Y"
  )

set(
  TEST_REDUCECCEWORLDTUBE_LIBRARIES
  Boost::boost
  Cce
  CceHelpers
  DataStructures
  GeneralRelativitySolutions
  Spectral
  SpinWeightedSphericalHarmonics
  )

set(EXECUTABLE "Test_ReduceCceWorldtube")

add_standalone_test_executable(${EXECUTABLE})

target_compile_definitions(
  ${EXECUTABLE}
  PUBLIC
  BINDIR="${CMAKE_BINARY_DIR}"
  )

target_link_libraries(
  "${EXECUTABLE}"
  PRIVATE
  "${TEST_REDUCECCEWORLDTUBE_LIBRARIES}")

add_test(
  NAME "Unit.Executables.ReduceCceWorldtube"
  COMMAND
  ${SHELL_EXECUTABLE}
  -c
  "rm -rf ReduceCceWorldtube &&
   mkdir -p ReduceCceWorldtube && \
   cd ReduceCceWorldtube && \
   ${CMAKE_BINARY_DIR}/bin/${EXECUTABLE}"
  )
