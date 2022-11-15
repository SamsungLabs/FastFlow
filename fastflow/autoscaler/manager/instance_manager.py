# Copyright 2022 Samsung Electronics Co., Ltd. All Rights Reserved
import threading
import time
import traceback

from fastflow.autoscaler.client.fastflow_client import AutoScalerServerClient
from fastflow.autoscaler.client.fastflow_client import DispatcherClient
from fastflow.autoscaler.framework.launcher import LauncherFactory
from fastflow.autoscaler.server.app import create_server
from fastflow.autoscaler.server.handler import FastFlowHTTPRequestHandler
from fastflow.autoscaler.utils.common_util import get_hostname
from fastflow.autoscaler.utils.manager_util import bind_port_safety, threaded
from fastflow.autoscaler.utils.manager_util import retry_until_function_ready
from fastflow.core.exceptions import DispatcherNotReadyError
from fastflow.core.exceptions import WorkersNotReadyError
from fastflow.platform.logger import get_logger

logger = get_logger()


@threaded
def _instance_watcher_thread(instance_manager):
    instance_manager.watch()


@threaded
def _server_runner_thread(server):
    server.serve_forever()


class _ServerHelper:
    def __init__(self,
                 server_port):
        self._uuid_token = FastFlowHTTPRequestHandler.uuid_token
        server_port = bind_port_safety(server_port)  # TODO: user option
        self._server_address = get_hostname() + ":" + str(server_port)
        self._server = create_server(server_port)
        self._server_thread = _server_runner_thread(self._server)
        self._server_thread.setName("server")
        self._server_thread.start()

    def is_alive(self):
        return self._server_thread.is_alive()

    def shutdown(self):
        self._server.shutdown()

    def get_uuid_token(self):
        return self._uuid_token

    def get_server_address(self):
        return self._server_address


# pylint: disable=unused-argument
class DummyDispatcher:
    """
    Use Dummy Dispatcher object
    when autoscale_enabled is False.
    """

    def __init__(self,
                 dispatcher_address):
        self._dispatcher_address = dispatcher_address

    def get_dispatcher_address_or_wait_until_ready(self,
                                                   *args,
                                                   **kwargs):
        return self._dispatcher_address


class DummyInstanceManager:
    """
    Use Dummy Instance Manager object
    when autoscale_enabled is False.
    then, autoscaler never work.
    """

    def __init__(self,
                 dispatcher_addr,
                 dispatcher_port,
                 *args,
                 **kwargs):
        self._dispatcher_address = dispatcher_addr + ":" \
                                   + str(dispatcher_port)

    def shutdown_all(self):
        pass

    def wait_until_workers_and_local_worker_ready(self,
                                                  *args,
                                                  **kwargs):
        pass

    def wait_until_workers_ready(self,
                                 *args,
                                 **kwargs):
        pass

    def watch(self):
        pass

    def launch_and_get_dispatcher(self,
                                  *args,
                                  **kwargs):
        return DummyDispatcher(self._dispatcher_address)

    def launch_workers(self,
                       *args,
                       **kwargs):
        pass


# pylint: enable=unused-argument


def _remove_local_worker_from_worker_list(current_workers, local_worker):
    # remove local worker address
    current_workers = current_workers.copy()
    try:
        # pylint: disable=protected-access
        current_workers.remove(local_worker._address)
        # pylint: enable=protected-access
    except KeyError:
        logger.warning("local_worker does not exist.")
        raise WorkersNotReadyError
    return current_workers


def _worker_ready_and_integrity_check(dispatcher_client,
                                      worker_instances,
                                      local_worker):
    """
    dispatcher_client: client of tf_data_service dispatcher
    worker_instances: worker instances that instance manager has launched
    local_worker: local tf_data_service worker


    detail: check workers are ready,
            and check the integrity of all workers.
            it means detect unauthorized alteration.
    """
    resp = dispatcher_client.get_workers()

    if not resp:
        raise ConnectionError("connecting dispatcher failed")
    expected_workers = set(
        map(
            lambda w_i: w_i.get_info().get("address"),
            filter(
                lambda w_i: w_i.get_info(),
                worker_instances.values()
            )
        )
    )

    if len(expected_workers) != len(worker_instances):
        raise WorkersNotReadyError

    current_workers = set(
        map(
            lambda worker: worker.address,
            resp.workers
        )
    )

    # if local_worker is not None, then remove worker from current_workers
    if local_worker:
        current_workers = _remove_local_worker_from_worker_list(
            current_workers,
            local_worker
        )

    if not expected_workers.issubset(current_workers):
        raise WorkersNotReadyError

    # Integrity Check
    if expected_workers != current_workers:
        raise RuntimeError("Integrity Error, Worker")

    return resp.workers


# TODO: Use Condition Variable and Add async logic for performance
class Dispatcher:
    def __init__(self,
                 instance,
                 server_helper,
                 instance_manager_thread):
        """
        instance: dispatcher instance
        server_helper: Required to detect if the server is dead.
        instance_manager_thread: Required to detect
                                 if the instance_manager is dead.
        """
        self._instance = instance
        self._server_helper = server_helper
        self._instance_manager_thread = instance_manager_thread

    def _get_dispatcher_info(self):
        if not self._instance.get_info():
            raise DispatcherNotReadyError
        return self._instance.get_info()

    def get_dispatcher_address_or_wait_until_ready(self, timeout=3600):
        dispatcher_info = retry_until_function_ready(
            self._instance_manager_thread,
            self._server_helper,
            self._get_dispatcher_info,
            timeout,
            "timeout because dispatcher not ready"
        )
        return dispatcher_info.get("address")

    def get_id(self):
        return self._instance.get_id()


class InstanceManager:
    """
    Manage One Dispatcher Instance and Many Worker Instances
    in fastflow.autoscaler.framework.instance.

    Instance can have two roles, "dispatcher" or "worker".
    Multi-Dispatcher does not support in v0.0.1.
    """

    def __init__(self,
                 instance_option,
                 server_port=1515):
        """
        server_port:
            binding port(default: 1515)
        self_hosted_cluster_terminator:
            python_function
        """
        self._server_helper = _ServerHelper(server_port)
        self._instance_manager_thread = \
            _instance_watcher_thread(self)
        self._instance_manager_thread.setName("instance_manager")
        self._instances = {}
        self._is_shutdown = threading.Event()
        self._shutdown_request = False
        self._lock = threading.Lock()
        self._instance_manager_thread.start()
        self._launcher = LauncherFactory.get_launcher(instance_option)
        # Need Multi-Dispatchers?
        self._dispatcher = None
        self._communication_data = {}
        self._communication_data["server_address"] = \
            self._server_helper.get_server_address()
        self._communication_data["uuid_token"] = \
            self._server_helper.get_uuid_token()

    def _shutdown_watch(self):
        """Stops the watch loop.

        Blocks until the loop has finished. This must be called while
        watch() is running in another thread, or it will
        deadlock.
        """
        if self._instance_manager_thread.is_alive():
            self._shutdown_request = True
            self._is_shutdown.wait()

    def _shutdown_server(self):
        """Stops the serve_forever loop.

        Blocks until the loop has finished. This must be called while
        serve_forever() is running in another thread, or it will
        deadlock.
        """
        if self._server_helper.is_alive():
            self._server_helper.shutdown()

    def _shutdown_instances(self):
        """
        terminate all instances.
        """
        # TODO: report log signal and waitting
        for _, instance in self._instances.items():
            try:
                instance.terminate()
            except Exception as e:
                logger.error(str(e))
                traceback.print_exc()

    def shutdown_all(self):
        """Stops the watch loop.
        And stops the serve_forever loop.
        Also, terminate all instances and server.

        Blocks until the loop has finished. This must be called while
        watch() and server_forever() are running in another thread, or it will
        deadlock.
        """
        self._shutdown_watch()
        self._shutdown_instances()
        self._shutdown_server()

    def _get_instances_all(self):
        with self._lock:
            return self._instances.copy()

    def _set_instance(self, key, val):
        with self._lock:
            self._instances[key] = val

    # TODO: abstract another class, and rename
    def _integrity_check(self, instance_infos):
        """
        instance_infos: All instance's informations
                        from API "GET /api/instance_infos"


        detail: Check the integrity of all TFDataServiceManager.
                it means detect unauthorized alteration.
        """
        instance_id_set = \
            set(self._get_instances_all().keys())
        mapping_instance_id = \
            set(instance_infos.keys())
        if not mapping_instance_id \
                .issubset(instance_id_set):
            raise RuntimeError("Integrity Error, "
                               "Some Instance Info does not match.")

    # TODO: abstract another class, and rename
    def _integrity_check_and_update(self, client):
        resp = client.get_instance_infos()
        if not resp:
            raise ConnectionError("Server Error")
        instance_infos = resp.json()

        self._integrity_check(instance_infos)

        with self._lock:
            for ins_id, info in instance_infos.items():
                self._instances[ins_id].set_info(info)

    def watch(self):
        self._is_shutdown.clear()
        try:
            client = AutoScalerServerClient(
                self._server_helper.get_server_address()
            )
            while not self._shutdown_request:
                if not threading.main_thread().is_alive():
                    raise RuntimeError("Main thread is dead.")
                logger.debug("Integrity check and update dictionary")
                self._integrity_check_and_update(client)
                # TODO: manage timeout instance
                time.sleep(1)
        except ConnectionError as ce:
            # Server Error
            logger.warning("fastflow will not terminate instance, "
                           "check your cluster")
            self._shutdown_server()
            raise ce
        except RuntimeError as re:
            # integrity check fail or main thread is dead.
            logger.info("shutdown starting.. please wait..")
            self._shutdown_instances()
            self._shutdown_server()
            raise re
        finally:
            self._shutdown_request = False
            self._is_shutdown.set()
            logger.info("autoscaler close")

    def _get_dispatcher_address(self):
        if not self._dispatcher:
            raise RuntimeError("dispatcher was not launched. "
                               "launch dispatcher first.")
        return (
            self
                ._dispatcher
                .get_dispatcher_address_or_wait_until_ready()
        )

    # Need Multi-Dispatchers?
    def launch_and_get_dispatcher(self):
        communication_data = self._communication_data.copy()
        if self._dispatcher:
            raise RuntimeError("Dispatcher already created")
        instance = self._launcher.execute_dispatcher(communication_data)
        self._dispatcher = Dispatcher(instance,
                                      self._server_helper,
                                      self._instance_manager_thread)
        self._set_instance(instance.get_id(), instance)
        return self._dispatcher

    # TODO: Use Condition Variable and Add async logic for performance
    def launch_workers(self,
                       n_workers=0):
        communication_data = self._communication_data.copy()
        if n_workers <= 0:
            return
        # Need selection logic in Multi-Dispatchers?
        communication_data["dispatcher_address"] = \
            self._get_dispatcher_address()
        instances = self._launcher.execute_remote_workers(
            n_workers,
            communication_data
        )
        for instance in instances:
            self._set_instance(instance.get_id(), instance)

    def wait_until_workers_and_local_worker_ready(self,
                                                  local_worker,
                                                  timeout=3600):
        dispatcher_client = DispatcherClient(self._get_dispatcher_address())
        instances = self._get_instances_all()

        # Need Multi-Dispatcher handling?
        # Remove Dispatcher instance from instances
        instances.pop(self._dispatcher.get_id())

        workers = retry_until_function_ready(
            self._instance_manager_thread,
            self._server_helper,
            _worker_ready_and_integrity_check,
            timeout,
            "timeout because workers not ready",
            dispatcher_client,
            instances,
            local_worker
        )
        return workers

    def wait_until_workers_ready(self, timeout=3600):
        return self.wait_until_workers_and_local_worker_ready(
            local_worker=None,
            timeout=timeout
        )


def create_instance_manager(auto_offload_conf):
    if auto_offload_conf.autoscale_enabled:
        return InstanceManager(auto_offload_conf.instance_option,
                               auto_offload_conf.server_port)
    return DummyInstanceManager(auto_offload_conf.dispatcher_addr,
                                auto_offload_conf.dispatcher_port)
