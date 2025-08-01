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

import logging
import multiprocessing
import queue
import random
import string
import uuid
from typing import Dict, List, Optional, Union, Set

from kubeflow.trainer.constants import constants
from kubeflow.trainer.types import types
from kubeflow.trainer.utils import utils
from kubeflow_trainer_api import models
from kubernetes import client, config, watch

logger = logging.getLogger(__name__)


class TrainerClient:
    def __init__(
        self,
        config_file: Optional[str] = None,
        context: Optional[str] = None,
        client_configuration: Optional[client.Configuration] = None,
        namespace: Optional[str] = None,
    ):
        """TrainerClient constructor. Configure logging in your application
            as follows to see detailed information from the TrainerClient APIs:
            .. code-block:: python
                import logging
                logging.basicConfig()
                log = logging.getLogger("kubeflow.trainer.api.trainer_client")
                log.setLevel(logging.DEBUG)

        Args:
            config_file: Path to the kube-config file. Defaults to ~/.kube/config.
            context: Set the active context. Defaults to current_context from the kube-config.
            client_configuration: Client configuration for cluster authentication.
                You have to provide valid configuration with Bearer token or
                with username and password. You can find an example here:
                https://github.com/kubernetes-client/python/blob/67f9c7a97081b4526470cad53576bc3b71fa6fcc/examples/remote_cluster.py#L31
            namespace: Target Kubernetes namespace. If SDK runs outside of Kubernetes cluster it
                takes the namespace from the kube-config context. If SDK runs inside
                the Kubernetes cluster it takes namespace from the
                `/var/run/secrets/kubernetes.io/serviceaccount/namespace` file. By default it
                uses the `default` namespace.
        """

        if namespace is None:
            namespace = utils.get_default_target_namespace(context)

        # If client configuration is not set, use kube-config to access Kubernetes APIs.
        if client_configuration is None:
            # Load kube-config or in-cluster config.
            if config_file or not utils.is_running_in_k8s():
                config.load_kube_config(config_file=config_file, context=context)
            else:
                config.load_incluster_config()

        k8s_client = client.ApiClient(client_configuration)
        self.custom_api = client.CustomObjectsApi(k8s_client)
        self.core_api = client.CoreV1Api(k8s_client)

        self.namespace = namespace

    def list_runtimes(self) -> List[types.Runtime]:
        """List of the available Runtimes.

        Returns:
            List[Runtime]: List of available training runtimes.
                If no runtimes exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list Runtimes.
            RuntimeError: Failed to list Runtimes.
        """

        result = []
        try:
            thread = self.custom_api.list_cluster_custom_object(
                constants.GROUP,
                constants.VERSION,
                constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
                async_req=True,
            )

            runtime_list = models.TrainerV1alpha1ClusterTrainingRuntimeList.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)
            )

            if not runtime_list:
                return result

            for runtime in runtime_list.items:
                result.append(self.__get_runtime_from_crd(runtime))

        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to list {constants.CLUSTER_TRAINING_RUNTIME_KIND}s "
                f"in namespace: {self.namespace}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to list {constants.CLUSTER_TRAINING_RUNTIME_KIND}s "
                f"in namespace: {self.namespace}"
            )

        return result

    def get_runtime(self, name: str) -> types.Runtime:
        """Get the the Runtime object"""

        try:
            thread = self.custom_api.get_cluster_custom_object(
                constants.GROUP,
                constants.VERSION,
                constants.CLUSTER_TRAINING_RUNTIME_PLURAL,
                name,
                async_req=True,
            )

            runtime = models.TrainerV1alpha1ClusterTrainingRuntime.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)  # type: ignore
            )

        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to get {constants.CLUSTER_TRAINING_RUNTIME_PLURAL}: "
                f"{self.namespace}/{name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to get {constants.CLUSTER_TRAINING_RUNTIME_PLURAL}: "
                f"{self.namespace}/{name}"
            )

        return self.__get_runtime_from_crd(runtime)  # type: ignore

    def train(
        self,
        runtime: types.Runtime = types.DEFAULT_RUNTIME,
        initializer: Optional[types.Initializer] = None,
        trainer: Optional[Union[types.CustomTrainer, types.BuiltinTrainer]] = None,
    ) -> str:
        """
        Create the TrainJob. You can configure these types of training task:

        - Custom Training Task: Training with a self-contained function that encapsulates
            the entire model training process, e.g. `CustomTrainer`.
        - Config-driven Task with Existing Trainer: Training with a trainer that already includes
            the post-training logic, requiring only parameter adjustments, e.g. `BuiltinTrainer`.

        Args:
            runtime (`types.Runtime`): Reference to one of existing Runtimes.
            initializer (`Optional[types.Initializer]`):
                Configuration for the dataset and model initializers.
            trainer (`Optional[types.CustomTrainer, types.BuiltinTrainer]`):
                Configuration for Custom Training Task or Config-driven Task with Builtin Trainer.

        Returns:
            str: The unique name of the TrainJob that has been generated.

        Raises:
            ValueError: Input arguments are invalid.
            TimeoutError: Timeout to create TrainJobs.
            RuntimeError: Failed to create TrainJobs.
        """

        # Generate unique name for the TrainJob.
        # TODO (andreyvelich): Discuss this TrainJob name generation.
        train_job_name = random.choice(string.ascii_lowercase) + uuid.uuid4().hex[:11]

        # Build the Trainer.
        trainer_crd = models.TrainerV1alpha1Trainer()

        if trainer:
            # If users choose to use a custom training function.
            if isinstance(trainer, types.CustomTrainer):
                trainer_crd = utils.get_trainer_crd_from_custom_trainer(
                    trainer, runtime
                )

            # If users choose to use a builtin trainer for post-training.
            elif isinstance(trainer, types.BuiltinTrainer):
                trainer_crd = utils.get_trainer_crd_from_builtin_trainer(
                    trainer, initializer
                )

            else:
                raise ValueError(
                    f"The trainer type {type(trainer)} is not supported. "
                    "Please use CustomTrainer or BuiltinTrainer."
                )

        train_job = models.TrainerV1alpha1TrainJob(
            apiVersion=constants.API_VERSION,
            kind=constants.TRAINJOB_KIND,
            metadata=models.IoK8sApimachineryPkgApisMetaV1ObjectMeta(
                name=train_job_name
            ),
            spec=models.TrainerV1alpha1TrainJobSpec(
                runtimeRef=models.TrainerV1alpha1RuntimeRef(name=runtime.name),
                trainer=(
                    trainer_crd
                    if trainer_crd != models.TrainerV1alpha1Trainer()
                    else None
                ),
                initializer=(
                    models.TrainerV1alpha1Initializer(
                        dataset=utils.get_dataset_initializer(initializer.dataset),
                        model=utils.get_model_initializer(initializer.model),
                    )
                    if isinstance(initializer, types.Initializer)
                    else None
                ),
            ),
        )

        # Create the TrainJob.
        try:
            self.custom_api.create_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                train_job.to_dict(),
            )
        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to create {constants.TRAINJOB_KIND}: {self.namespace}/{train_job_name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to create {constants.TRAINJOB_KIND}: {self.namespace}/{train_job_name}"
            )

        logger.debug(
            f"{constants.TRAINJOB_KIND} {self.namespace}/{train_job_name} has been created"
        )

        return train_job_name

    def list_jobs(
        self, runtime: Optional[types.Runtime] = None
    ) -> List[types.TrainJob]:
        """List of all TrainJobs.

        Returns:
            List[TrainerV1alpha1TrainJob]: List of created TrainJobs.
                If no TrainJob exist, an empty list is returned.

        Raises:
            TimeoutError: Timeout to list TrainJobs.
            RuntimeError: Failed to list TrainJobs.
        """

        result = []
        try:
            thread = self.custom_api.list_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                async_req=True,
            )

            trainjob_list = models.TrainerV1alpha1TrainJobList.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)
            )

            if not trainjob_list:
                return result

            for trainjob in trainjob_list.items:
                # If runtime object is set, we check the TrainJob's runtime reference.
                if (
                    runtime is not None
                    and trainjob.spec
                    and trainjob.spec.runtime_ref
                    and trainjob.spec.runtime_ref.name != runtime.name
                ):
                    continue

                result.append(self.__get_trainjob_from_crd(trainjob))

        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to list {constants.TRAINJOB_KIND}s in namespace: {self.namespace}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to list {constants.TRAINJOB_KIND}s in namespace: {self.namespace}"
            )

        return result

    def get_job(self, name: str) -> types.TrainJob:
        """Get the TrainJob object"""

        try:
            thread = self.custom_api.get_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                name,
                async_req=True,
            )

            trainjob = models.TrainerV1alpha1TrainJob.from_dict(
                thread.get(constants.DEFAULT_TIMEOUT)  # type: ignore
            )

        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to get {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            )

        return self.__get_trainjob_from_crd(trainjob)  # type: ignore

    def get_job_logs(
        self,
        name: str,
        follow: Optional[bool] = False,
        step: str = constants.NODE,
        node_rank: int = 0,
    ) -> Dict[str, str]:
        """Get the logs from TrainJob"""

        # Get the TrainJob Pod name.
        pod_name = None
        for c in self.get_job(name).steps:
            if c.status != constants.POD_PENDING:
                if c.name == step or c.name == f"{step}-{node_rank}":
                    pod_name = c.pod_name
        if pod_name is None:
            return {}

        # Dict where key is the Pod type and value is the Pod logs.
        logs_dict = {}

        # TODO (andreyvelich): Potentially, refactor this.
        # Support logging of multiple Pods.
        # TODO (andreyvelich): Currently, follow is supported only for node container.
        if follow and step == constants.NODE:
            log_streams = []
            log_streams.append(
                watch.Watch().stream(
                    self.core_api.read_namespaced_pod_log,
                    name=pod_name,
                    namespace=self.namespace,
                    container=constants.NODE,
                )
            )
            finished = [False] * len(log_streams)

            # Create thread and queue per stream, for non-blocking iteration.
            log_queue_pool = utils.get_log_queue_pool(log_streams)

            # Iterate over every watching pods' log queue
            while True:
                for index, log_queue in enumerate(log_queue_pool):
                    if all(finished):
                        break
                    if finished[index]:
                        continue
                    # grouping the every 50 log lines of the same pod.
                    for _ in range(50):
                        try:
                            logline = log_queue.get(timeout=1)
                            if logline is None:
                                finished[index] = True
                                break
                            # Print logs to the StdOut and update results dict.
                            print(f"[{step}-{node_rank}]: {logline}")
                            logs_dict[f"{step}-{node_rank}"] = (
                                logs_dict.get(f"{step}-{node_rank}", "")
                                + logline
                                + "\n"
                            )
                        except queue.Empty:
                            break
                if all(finished):
                    return logs_dict

        try:
            if step == constants.DATASET_INITIALIZER:
                logs_dict[constants.DATASET_INITIALIZER] = (
                    self.core_api.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        container=constants.DATASET_INITIALIZER,
                    )
                )
            elif step == constants.MODEL_INITIALIZER:
                logs_dict[constants.MODEL_INITIALIZER] = (
                    self.core_api.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        container=constants.MODEL_INITIALIZER,
                    )
                )
            else:
                logs_dict[f"{step}-{node_rank}"] = (
                    self.core_api.read_namespaced_pod_log(
                        name=pod_name,
                        namespace=self.namespace,
                        container=constants.NODE,
                    )
                )

        except Exception:
            raise RuntimeError(
                f"Failed to read logs for the pod {self.namespace}/{pod_name}"
            )

        return logs_dict

    def wait_for_job_status(
        self,
        name: str,
        status: Set[str] = {constants.TRAINJOB_COMPLETE},
        timeout: int = 600,
    ) -> types.TrainJob:
        """Wait for TrainJob to reach the desired status

        Args:
            name: Name of the TrainJob.
            status: Set of expected statuses. It must be subset of Created, Running, Complete, and
                Failed statuses.
            timeout: How many seconds to wait until TrainJob reaches one of the expected conditions.

        Returns:
            TrainJob: The training job that reaches the desired status.

        Raises:
            ValueError: The input values are incorrect.
            RuntimeError: Failed to get TrainJob or TrainJob reaches unexpected Failed status.
            TimeoutError: Timeout to wait for TrainJob status.
        """

        job_statuses = {
            constants.TRAINJOB_CREATED,
            constants.TRAINJOB_RUNNING,
            constants.TRAINJOB_COMPLETE,
            constants.TRAINJOB_FAILED,
        }
        if not status.issubset(job_statuses):
            raise ValueError(
                f"Expected status {status} must be a subset of {job_statuses}"
            )

        # Use Kubernetes watch API to monitor the TrainJob's Pods.
        w = watch.Watch()
        try:
            for event in w.stream(
                self.core_api.list_namespaced_pod,
                self.namespace,
                label_selector=constants.POD_LABEL_SELECTOR.format(trainjob_name=name),
                timeout_seconds=timeout,
            ):
                # Check the status after event is generated for the TrainJob's Pods.
                trainjob = self.get_job(name)
                logger.debug(f"TrainJob {name}, status {trainjob.status}")

                # Raise an error if TrainJob is Failed and it is not the expected status.
                if (
                    constants.TRAINJOB_FAILED not in status
                    and trainjob.status == constants.TRAINJOB_FAILED
                ):
                    raise RuntimeError(f"TrainJob {name} is Failed")

                # Return the TrainJob if it reaches the expected status.
                if trainjob.status in status:
                    return trainjob

        except TimeoutError:
            raise TimeoutError(f"Timeout to get the TrainJob {name}")
        except Exception:
            raise RuntimeError(f"Failed to watch Pods for TrainJob {name}")
        finally:
            w.stop()

        raise TimeoutError(
            f"Timeout waiting for TrainJob {name} to reach status: {status} status"
        )

    def delete_job(self, name: str):
        """Delete the TrainJob.

        Args:
            name: Name of the TrainJob.

        Raises:
            TimeoutError: Timeout to delete TrainJob.
            RuntimeError: Failed to delete TrainJob.
        """

        try:
            self.custom_api.delete_namespaced_custom_object(
                constants.GROUP,
                constants.VERSION,
                self.namespace,
                constants.TRAINJOB_PLURAL,
                name=name,
            )
        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to delete {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to delete {constants.TRAINJOB_KIND}: {self.namespace}/{name}"
            )

        logger.debug(
            f"{constants.TRAINJOB_KIND} {self.namespace}/{name} has been deleted"
        )

    def __get_runtime_from_crd(
        self,
        runtime_crd: models.TrainerV1alpha1ClusterTrainingRuntime,
    ) -> types.Runtime:

        if not (
            runtime_crd.metadata
            and runtime_crd.metadata.name
            and runtime_crd.spec
            and runtime_crd.spec.ml_policy
            and runtime_crd.spec.template.spec
            and runtime_crd.spec.template.spec.replicated_jobs
        ):
            raise Exception(f"ClusterTrainingRuntime CRD is invalid: {runtime_crd}")

        return types.Runtime(
            name=runtime_crd.metadata.name,
            trainer=utils.get_runtime_trainer(
                runtime_crd.spec.template.spec.replicated_jobs,
                runtime_crd.spec.ml_policy,
            ),
        )

    def __get_trainjob_from_crd(
        self,
        trainjob_crd: models.TrainerV1alpha1TrainJob,
    ) -> types.TrainJob:

        if not (
            trainjob_crd.metadata
            and trainjob_crd.metadata.name
            and trainjob_crd.metadata.namespace
            and trainjob_crd.spec
            and trainjob_crd.metadata.creation_timestamp
        ):
            raise Exception(f"TrainJob CRD is invalid: {trainjob_crd}")

        name = trainjob_crd.metadata.name
        namespace = trainjob_crd.metadata.namespace

        runtime = self.get_runtime(trainjob_crd.spec.runtime_ref.name)

        # Construct the TrainJob from the CRD.
        trainjob = types.TrainJob(
            name=name,
            creation_timestamp=trainjob_crd.metadata.creation_timestamp,
            runtime=runtime,
            steps=[],
            # Number of nodes is taken from TrainJob or TrainingRuntime
            num_nodes=(
                trainjob_crd.spec.trainer.num_nodes
                if trainjob_crd.spec.trainer and trainjob_crd.spec.trainer.num_nodes
                else runtime.trainer.num_nodes
            ),
            status=constants.TRAINJOB_CREATED,  # The default TrainJob status.
        )

        # Add the TrainJob components, e.g. trainer nodes and initializer.
        try:
            response = self.core_api.list_namespaced_pod(
                namespace,
                label_selector=constants.POD_LABEL_SELECTOR.format(trainjob_name=name),
                async_req=True,
            ).get(constants.DEFAULT_TIMEOUT)

            # Convert Pod to the correct format.
            pod_list = models.IoK8sApiCoreV1PodList.from_dict(response.to_dict())
            if not pod_list:
                return trainjob

            for pod in pod_list.items:
                # Pod must have labels to detect the TrainJob step.
                # Every Pod always has a single TrainJob step.
                if not (
                    pod.metadata
                    and pod.metadata.name
                    and pod.metadata.labels
                    and pod.spec
                ):
                    raise Exception(f"TrainJob Pod is invalid: {pod}")

                # Get the Initializer step.
                if pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL] in {
                    constants.DATASET_INITIALIZER,
                    constants.MODEL_INITIALIZER,
                }:
                    trainjob.steps.append(
                        utils.get_trainjob_initializer_step(
                            pod.metadata.name,
                            pod.spec,
                            pod.status,
                        )
                    )
                # Get the Node step.
                elif pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL] in {
                    constants.LAUNCHER,
                    constants.NODE,
                }:
                    trainjob.steps.append(
                        utils.get_trainjob_node_step(
                            pod.metadata.name,
                            pod.spec,
                            pod.status,
                            trainjob.runtime,
                            pod.metadata.labels[constants.JOBSET_RJOB_NAME_LABEL],
                            int(pod.metadata.labels[constants.JOB_INDEX_LABEL]),
                        )
                    )
        except multiprocessing.TimeoutError:
            raise TimeoutError(
                f"Timeout to list {constants.TRAINJOB_KIND}'s steps: {namespace}/{name}"
            )
        except Exception:
            raise RuntimeError(
                f"Failed to list {constants.TRAINJOB_KIND}'s steps: {namespace}/{name}"
            )

        # Update the TrainJob status from its conditions.
        if trainjob_crd.status and trainjob_crd.status.conditions:
            for c in trainjob_crd.status.conditions:
                if c.type == constants.TRAINJOB_COMPLETE and c.status == "True":
                    trainjob.status = c.type
                elif c.type == constants.TRAINJOB_FAILED and c.status == "True":
                    trainjob.status = c.type
        else:
            # The TrainJob running status is defined when all training node (e.g. Pods) are running.
            num_running_nodes = sum(
                1
                for step in trainjob.steps
                if step.name.startswith(constants.NODE)
                and step.status == constants.TRAINJOB_RUNNING
            )

            if trainjob.num_nodes == num_running_nodes:
                trainjob.status = constants.TRAINJOB_RUNNING

        return trainjob
