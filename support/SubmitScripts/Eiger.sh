{% extends "SubmitTemplateBase.sh" %}

# Distributed under the MIT License.
# See LICENSE.txt for details.

# CPU partition of the CSCS Alps system. Each node has two AMD EPYC 7742
# sockets with 64 cores each and 2 hardware threads per core.
# More information:
# https://docs.cscs.ch/clusters/eiger/
#
# SpECTRE runs in the sxscollaboration/spectre container with Apptainer (the
# CSCS Container Engine injects host libraries that conflict with the image).
# Pass the image with `-p spectre_image=/path/to/spectre.sif` and, if Apptainer
# is not on the PATH, `-p apptainer=/path/to/apptainer`. The batch script runs
# on the host, so the executable and the `spectre` CLI calls after it (e.g.
# run-next) are wrapped in `apptainer exec`. There is no SLURM inside the image,
# so resubmitting a segmented run from the job does not work.
#
# The Charm++ in the image is the multicore build: one process per node, no
# communication thread. So there is one task with one PE per physical core.

{% block head %}
{{ super() -}}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 128
#SBATCH --hint=nomultithread
{% if account is defined %}#SBATCH -A {{ account }}
{% endif -%}
#SBATCH -p {{ queue | default("normal") }}
#SBATCH -t {{ time_limit | default("1-00:00:00") }}
{% endblock %}

{% block charm_ppn %}
# Multicore Charm++: no communication thread, so one PE per CPU
CHARM_PPN=${SLURM_CPUS_PER_TASK}
{% endblock %}

{% block list_modules %}
SPECTRE_BUILD_BIN="$(dirname "${SPECTRE_CLI}")"
SPECTRE_CONTAINER="{{ apptainer | default("apptainer") }} exec \
  --cleanenv \
  --env PREPEND_PATH=${SPECTRE_BUILD_BIN} \
  --bind {{ apptainer_binds | default("/capstor,/ritom") }} \
  {{ spectre_image }}"
export SPECTRE_CLI="${SPECTRE_CONTAINER} ${SPECTRE_CLI}"
echo "Container: ${SPECTRE_CONTAINER}"
{% endblock %}

{% block run_command %}
${SPECTRE_CONTAINER} \
  ${SPECTRE_PROFILING_PREFIX} \
  ${SPECTRE_EXECUTABLE} --input-file ${SPECTRE_INPUT_FILE} \
  +p${CHARM_PPN} +setcpuaffinity \
  ${SPECTRE_CHECKPOINT:+ +restart "${SPECTRE_CHECKPOINT}"}
{% endblock %}
