# Distributed under the MIT License.
# See LICENSE.txt for details.

option(MACHINE "Select a machine that we know how to run on, such as a \
particular supercomputer" OFF)

spectre_python_add_module(
  support
  PYTHON_FILES
  CliExceptions.py
  DirectoryStructure.py
  Logging.py
  Machines.py
  Resubmit.py
  RunNext.py
  Schedule.py
)

if(MACHINE)
  message(STATUS "Selected machine: ${MACHINE}")
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/Machines/${MACHINE}.yaml
    ${SPECTRE_PYTHON_PREFIX}/support/Machine.yaml
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/SubmitTemplateBase.sh
    ${SPECTRE_PYTHON_PREFIX}/support/SubmitTemplateBase.sh
    @ONLY
    )
  configure_file(
    ${CMAKE_SOURCE_DIR}/support/SubmitScripts/${MACHINE}.sh
    ${SPECTRE_PYTHON_PREFIX}/support/SubmitTemplate.sh
    @ONLY
    )
endif()

# Generate shell completion scripts. These don't usually change, so they are
# committed to the source code and configured to the build dir. See the
# [click docs](https://click.palletsprojects.com/shell-completion/)
# for instructions how to regenerate the shell completion scripts.
configure_file(
  shell-completion.bash
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.bash @ONLY)
configure_file(
  shell-completion.fish
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.fish @ONLY)
configure_file(
  shell-completion.zsh
  ${SPECTRE_PYTHON_PREFIX_PARENT}/shell-completion.zsh @ONLY)
