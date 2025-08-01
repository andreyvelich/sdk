# Copyright 2024 The Kubeflow Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

# How long to wait in seconds for requests to the Kubernetes API Server.
DEFAULT_TIMEOUT = 120

# Common constants.
GROUP = "trainer.kubeflow.org"
VERSION = "v1alpha1"
API_VERSION = f"{GROUP}/{VERSION}"

# The default Kubernetes namespace.
DEFAULT_NAMESPACE = "default"

# The Kind name for the ClusterTrainingRuntime.
CLUSTER_TRAINING_RUNTIME_KIND = "ClusterTrainingRuntime"

# The plural for the ClusterTrainingRuntime.
CLUSTER_TRAINING_RUNTIME_PLURAL = "clustertrainingruntimes"

# The Kind name for the TrainJob.
TRAINJOB_KIND = "TrainJob"

# The plural for the TrainJob.
TRAINJOB_PLURAL = "trainjobs"

# The default status for the TrainJob once users create it.
TRAINJOB_CREATED = "Created"

# The running status of the TrainJob, defined when all training node (e.g. Pods) are running.
TRAINJOB_RUNNING = "Running"

# The complete status of the TrainJob, defined when TrainJob CR has complete condition.
TRAINJOB_COMPLETE = "Complete"

# The failed status of the TrainJob, defined when TrainJob CR has failed condition.
TRAINJOB_FAILED = "Failed"

# The label key to identify the relationship between TrainJob and Pod template in the runtime.
# For example, what PodTemplate must be overridden by TrainJob's .spec.trainer APIs.
TRAINJOB_ANCESTOR_LABEL = "trainer.kubeflow.org/trainjob-ancestor-step"

# The name of the ReplicatedJob and container of the dataset initializer.
# Also, it represents the `trainjob-ancestor-step` label value for the dataset initializer step.
DATASET_INITIALIZER = "dataset-initializer"

# The name of the ReplicatedJob and container of the model initializer.
# Also, it represents the `trainjob-ancestor-step` label value for the model initializer step.
MODEL_INITIALIZER = "model-initializer"

# The env name for the access token of dataset/model initializer.
INITIALIZER_ENV_ACCESS_TOKEN = "ACCESS_TOKEN"

# The default path to the users' workspace.
# TODO (andreyvelich): Discuss how to keep this path is sync with pkg.initializers.constants
WORKSPACE_PATH = "/workspace"

# The path where initializer downloads dataset.
DATASET_PATH = os.path.join(WORKSPACE_PATH, "dataset")

# The path where initializer downloads model.
MODEL_PATH = os.path.join(WORKSPACE_PATH, "model")

# The name of the ReplicatedJob to launch mpirun.
LAUNCHER = "launcher"

# The name of the ReplicatedJob and container of the node. The node usually represents
# single VM where distributed training code is executed.
NODE = "node"

# Unknown indicates that the value can't be identified.
UNKNOWN = "Unknown"

# The label for cpu in the container resources.
CPU_LABEL = "cpu"

# The label for NVIDIA GPU in the container resources.
GPU_LABEL = "nvidia.com/gpu"

# The label for TPU in the container resources.
TPU_LABEL = "google.com/tpu"

# The label key to identify the JobSet name of the Pod.
JOBSET_NAME_LABEL = "jobset.sigs.k8s.io/jobset-name"

# The label key to identify the JobSet's ReplicatedJob of the Pod.
JOBSET_RJOB_NAME_LABEL = "jobset.sigs.k8s.io/replicatedjob-name"

# The label key to identify the Job completion index of the Pod.
JOB_INDEX_LABEL = "batch.kubernetes.io/job-completion-index"

# The Pod pending phase indicates that Pod has been accepted by the Kubernetes cluster,
# but one or more of the containers has not been made ready to run.
POD_PENDING = "Pending"

# The label selector for Pods created by the TrainJob.
# It checks the following rJob.name: dataset-initializer, model-initializer, launcher, node.
POD_LABEL_SELECTOR = ("{}={{trainjob_name}},{} in ({}, {}, {}, {})").format(
    JOBSET_NAME_LABEL,
    JOBSET_RJOB_NAME_LABEL,
    DATASET_INITIALIZER,
    MODEL_INITIALIZER,
    LAUNCHER,
    NODE,
)

# The default PIP index URL to download Python packages.
DEFAULT_PIP_INDEX_URL = os.getenv("DEFAULT_PIP_INDEX_URL", "https://pypi.org/simple")

# The default command for the Custom Trainer.
DEFAULT_CUSTOM_COMMAND = ["bash", "-c"]

# The default command for the TorchTune Trainer.
DEFAULT_TORCHTUNE_COMMAND = ["tune", "run"]

# The default entrypoint for torchrun.
TORCH_ENTRYPOINT = "torchrun"

# The Torch env name for the number of procs per node (e.g. number of GPUs per Pod).
TORCH_ENV_NUM_PROC_PER_NODE = "PET_NPROC_PER_NODE"

# The default home directory for the MPI user.
DEFAULT_MPI_USER_HOME = os.getenv("DEFAULT_MPI_USER_HOME", "/home/mpiuser")

# The default location for MPI hostfile.
# TODO (andreyvelich): We should get this info from Runtime CRD.
MPI_HOSTFILE = "/etc/mpi/hostfile"

# The default entrypoint for mpirun.
MPI_ENTRYPOINT = "mpirun"

# The Instruct Datasets class in torchtune
TORCHTUNE_INSTRUCT_DATASET = "torchtune.datasets.instruct_dataset"
