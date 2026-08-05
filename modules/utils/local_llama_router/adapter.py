"""Docker and HTTP adapter for the managed local llama.cpp Router."""

from __future__ import annotations

import copy
import errno
import json
import os
import shlex
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.parse import quote

import requests

from modules.utils.exceptions import OperationCancelledError
from modules.utils.llama_cpp_runtime import (
    DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC,
    inspect_llama_cpp_version_from_image,
    resolve_docker_compose_command,
    run_docker_command,
)

from .contracts import (
    ROUTER_OWNER_LABEL,
    ROUTER_OWNER_VALUE,
    ROUTER_PAIR_LABEL,
    ROUTER_PROJECT_NAME,
    ROUTER_PROJECT_LABEL,
    ROUTER_SERVICE_NAME,
    RouterPair,
    RouterPairKind,
    RouterRuntimeContract,
    RouterRuntimeSpec,
    build_router_contract,
    canonical_sha256,
    expected_router_server_args,
    router_environment,
)


_ROUTER_PAIR_LABEL_VALUES = frozenset(kind.value for kind in RouterPairKind)


class RouterAdapterError(RuntimeError):
    """The local Docker/Router runtime cannot satisfy its product contract."""


class RouterAdapterOwnershipError(RouterAdapterError):
    """A port or container is not proven to belong to this Router."""


@dataclass(frozen=True)
class RouterContainerInspection:
    name: str
    exists: bool
    running: bool
    image: str
    image_id: str
    labels: Mapping[str, str]
    command: tuple[str, ...]
    entrypoint: tuple[str, ...]
    ports: Mapping[str, Any]
    mounts: tuple[Mapping[str, Any], ...]
    device_requests: tuple[Mapping[str, Any], ...]
    pid: int | None

    def owned_by_router(self) -> bool:
        return bool(
            str(self.labels.get(ROUTER_OWNER_LABEL, "")) == ROUTER_OWNER_VALUE
            and str(self.labels.get(ROUTER_PROJECT_LABEL, ""))
            == ROUTER_PROJECT_NAME
            # Derive the accepted pair labels from the pair enum so a new pair
            # is never misread as a foreign container.
            and str(self.labels.get(ROUTER_PAIR_LABEL, "")) in _ROUTER_PAIR_LABEL_VALUES
        )

    def owned_by(self, contract: RouterRuntimeContract) -> bool:
        expected = contract.ownership_labels
        return all(
            str(self.labels.get(key, "")) == str(value)
            for key, value in expected.items()
        )


@dataclass(frozen=True)
class RouterModelSnapshot:
    loaded_models: tuple[str, ...]
    loaded_count: int
    slots_idle: bool
    slots: tuple[Mapping[str, Any], ...]
    raw_models: Mapping[str, Any]
    raw_slots: Any
    transitional_models: tuple[str, ...] = ()


class RouterCommandAdapter(Protocol):
    """Narrow I/O boundary consumed by the typed Router coordinator."""

    def build_contract(
        self,
        spec: RouterRuntimeSpec,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterRuntimeContract: ...

    def prepare(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterContainerInspection: ...

    def model_snapshot(
        self,
        pair: RouterPair,
        *,
        timeout_sec: float = 5.0,
        include_slots: bool = True,
    ) -> RouterModelSnapshot: ...

    def load_model(
        self,
        pair: RouterPair,
        model_alias: str,
        *,
        timeout_sec: float = 180.0,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None: ...

    def unload_model(
        self,
        pair: RouterPair,
        model_alias: str,
        *,
        timeout_sec: float = 60.0,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None: ...

    def assert_implicit_autoload_rejected(
        self,
        pair: RouterPair,
        model_alias: str,
    ) -> None: ...

    def stop_pair(
        self,
        contract: RouterRuntimeContract,
        *,
        allow_stale_owned_fingerprint: bool = False,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None: ...

    def stop_owned_port_occupants(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None: ...

    def stop_owned_pair_ports(
        self,
        pair: RouterPair,
        *,
        cancel_checker: Callable[[], bool] | None = None,
        reject_foreign: bool = True,
        require_ports_free: bool = True,
    ) -> tuple[str, ...]: ...

    def owned_gpu_process_ids(
        self,
        contract: RouterRuntimeContract,
    ) -> frozenset[int]: ...


class DockerRouterCommandAdapter:
    """Fail-closed Router adapter backed by Docker Compose and Router APIs."""

    _HEALTH_POLL_SEC = 0.25
    _PORT_RELEASE_TIMEOUT_SEC = 3.0

    def __init__(
        self,
        *,
        request_session: requests.Session | None = None,
    ) -> None:
        self._session = request_session or requests.Session()
        self._compose_command: tuple[str, ...] | None = None

    def build_contract(
        self,
        spec: RouterRuntimeSpec,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterRuntimeContract:
        pair = spec.pair
        if not pair.compose_file.is_file():
            raise RouterAdapterError(
                f"Router compose file is missing: {pair.compose_file}"
            )
        if not pair.preset_file.is_file():
            raise RouterAdapterError(
                f"Router models preset is missing: {pair.preset_file}"
            )
        self._raise_if_cancelled(cancel_checker, "before Router contract inspection")
        self._ensure_image(spec.image_ref, cancel_checker=cancel_checker)
        image = self._inspect_image(spec.image_ref, cancel_checker=cancel_checker)
        image_id = str(image.get("Id") or "").strip()
        if not image_id:
            raise RouterAdapterError(
                f"Docker did not report an image ID for Router image {spec.image_ref}."
            )
        repo_digest = self._select_repo_digest(image)
        if not repo_digest:
            raise RouterAdapterError(
                f"Docker did not report a repository digest for Router image {spec.image_ref}."
            )
        entrypoint = self._string_tuple((image.get("Config") or {}).get("Entrypoint"))
        if not entrypoint:
            raise RouterAdapterError("Router image has no inspectable entrypoint.")
        binary_version = inspect_llama_cpp_version_from_image(
            spec.image_ref,
            cancel_checker=cancel_checker,
        )
        if not binary_version:
            raise RouterAdapterError(
                "Router llama.cpp binary version could not be inspected."
            )

        preset_sha256 = self._file_sha256(pair.preset_file)
        provisional_env = self._compose_environment_for_spec(
            spec,
            fingerprint="<computed>",
            preset_sha256=preset_sha256,
        )
        provisional_config = self._resolved_compose_config(
            pair,
            environment=provisional_env,
            cancel_checker=cancel_checker,
        )
        normalized_config = self._normalize_dynamic_values(
            provisional_config,
            dynamic_values={"<computed>", "unprepared"},
        )
        service = self._service_config(normalized_config)
        self._assert_resolved_compose_contract(spec, normalized_config)
        effective_environment = self._service_environment(service)
        server_args = self._string_tuple(service.get("command"))
        expected_args = expected_router_server_args(pair)
        if server_args != expected_args:
            raise RouterAdapterError(
                "Router compose command differs from the explicit no-autoload contract."
            )
        contract = build_router_contract(
            spec=spec,
            image_id=image_id,
            repo_digest=repo_digest,
            entrypoint=entrypoint,
            binary_version=binary_version,
            resolved_compose_config=normalized_config,
            effective_environment=effective_environment,
            port_mapping=self._service_ports(service),
            volume_mapping=self._service_volumes(service),
            device_mapping=self._service_devices(service),
            server_args=server_args,
            preset_sha256=preset_sha256,
        )
        effective_env = router_environment(contract)
        resolved_actual = self._resolved_compose_config(
            pair,
            environment=effective_env,
            cancel_checker=cancel_checker,
        )
        normalized_actual = self._normalize_dynamic_values(
            resolved_actual,
            dynamic_values={contract.fingerprint, "unprepared", "<computed>"},
        )
        if normalized_actual != normalized_config:
            raise RouterAdapterError(
                "Router resolved Compose config changed after the fingerprint was applied."
            )
        return contract

    def prepare(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> RouterContainerInspection:
        try:
            self.stop_owned_port_occupants(contract, cancel_checker=cancel_checker)
            environment = router_environment(contract)
            self._run_compose(
                contract.pair,
                ("pull", "--policy", "missing"),
                environment=environment,
                cancel_checker=cancel_checker,
            )
            self._run_compose(
                contract.pair,
                ("up", "-d", "--force-recreate"),
                environment=environment,
                cancel_checker=cancel_checker,
            )
            inspection = self._inspect_container(contract.pair.container_name)
            self._assert_container_contract(inspection, contract)
            self._wait_for_health(contract.pair, cancel_checker=cancel_checker)
            snapshot = self.model_snapshot(contract.pair)
            if snapshot.loaded_count != 0 or not snapshot.slots_idle:
                raise RouterAdapterError(
                    "Router did not start with zero loaded models and idle slots."
                )
            return inspection
        except Exception:
            # The only cleanup allowed after a failed prepare is the exact
            # just-requested, product-owned container. A foreign process at a
            # matching port remains untouched and the original ownership error
            # is preserved for the caller.
            inspection = self._inspect_container(contract.pair.container_name)
            if (
                inspection.exists
                and inspection.running
                and inspection.owned_by(contract)
            ):
                try:
                    self._stop_inspection(inspection, cancel_checker=None)
                except Exception as cleanup_error:
                    raise RouterAdapterError(
                        "Router prepare failed and its owned partial container could not be stopped."
                    ) from cleanup_error
            raise

    def model_snapshot(
        self,
        pair: RouterPair,
        *,
        timeout_sec: float = 5.0,
        include_slots: bool = True,
    ) -> RouterModelSnapshot:
        models_payload = self._request_json(
            "GET",
            f"{pair.router_base_url}/models",
            timeout_sec=timeout_sec,
        )
        entries = models_payload.get("data")
        if not isinstance(entries, list):
            entries = models_payload.get("models")
        if not isinstance(entries, list):
            raise RouterAdapterError("Router model API did not return a model list.")
        loaded: list[str] = []
        transitional: list[str] = []
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise RouterAdapterError("Router model API returned a non-object model entry.")
            raw_status = entry.get("status")
            if isinstance(raw_status, Mapping):
                status = str(raw_status.get("value") or "").strip().lower()
            else:
                status = str(raw_status or "").strip().lower()
            model_id = str(entry.get("id") or entry.get("name") or "").strip()
            if not status:
                raise RouterAdapterError("Router model API omitted a model status.")
            if status == "loaded":
                if not model_id:
                    raise RouterAdapterError("Router reported a loaded model without an identity.")
                loaded.append(model_id)
            elif status in {"loading", "unloading", "downloading"}:
                transitional.append(model_id or "<unnamed>")
            elif status not in {"unloaded", "sleeping"}:
                raise RouterAdapterError(f"Router returned an unknown model status: {status}")
        # Router-mode GET routes require a URL-encoded model query. Query
        # only loaded instances: when the model API proves zero loaded models,
        # there is no process with slots to drain and an unloaded-model GET
        # must not be allowed to autoload one just to inspect it.
        slot_payloads: dict[str, Any] = {}
        slots: list[Mapping[str, Any]] = []
        if include_slots:
            for model_alias in sorted(loaded):
                payload = self._request_payload(
                    "GET",
                    (
                        f"{pair.router_base_url}/slots?model="
                        f"{quote(model_alias, safe='')}&autoload=false"
                    ),
                    timeout_sec=timeout_sec,
                )
                slot_payloads[model_alias] = payload
                slots.extend(self._slots_list(payload))
        slots_idle = bool(
            not transitional
            and (
                not include_slots
                or all(not bool(slot.get("is_processing", False)) for slot in slots)
            )
        )
        return RouterModelSnapshot(
            loaded_models=tuple(sorted(loaded)),
            loaded_count=len(loaded),
            slots_idle=slots_idle,
            slots=tuple(copy.deepcopy(slot) for slot in slots),
            raw_models=copy.deepcopy(models_payload),
            raw_slots=copy.deepcopy(slot_payloads),
            transitional_models=tuple(sorted(transitional)),
        )

    def load_model(
        self,
        pair: RouterPair,
        model_alias: str,
        *,
        timeout_sec: float = 180.0,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        self._raise_if_cancelled(cancel_checker, "before Router model load")
        self._request_json(
            "POST",
            f"{pair.router_base_url}/models/load",
            json_payload={"model": model_alias},
            timeout_sec=timeout_sec,
        )
        self._raise_if_cancelled(cancel_checker, "after Router model load")

    def unload_model(
        self,
        pair: RouterPair,
        model_alias: str,
        *,
        timeout_sec: float = 60.0,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        self._raise_if_cancelled(cancel_checker, "before Router model unload")
        self._request_json(
            "POST",
            f"{pair.router_base_url}/models/unload",
            json_payload={"model": model_alias},
            timeout_sec=timeout_sec,
        )
        self._raise_if_cancelled(cancel_checker, "after Router model unload")

    def assert_implicit_autoload_rejected(
        self,
        pair: RouterPair,
        model_alias: str,
    ) -> None:
        """Prove a request cannot silently re-load an unloaded model."""

        try:
            response = self._session.post(
                f"{pair.router_base_url}/v1/chat/completions",
                json={
                    "model": model_alias,
                    "messages": [{"role": "user", "content": "router-contract"}],
                    "max_tokens": 1,
                    "stream": False,
                },
                timeout=15.0,
            )
        except requests.RequestException as exc:
            raise RouterAdapterError(
                "Router implicit-autoload contract probe could not reach the Router."
            ) from exc
        snapshot = self.model_snapshot(pair)
        if response.status_code < 400 or snapshot.loaded_count != 0:
            raise RouterAdapterError(
                "Router accepted an unloaded-model request or implicitly autoloaded it."
            )

    def stop_pair(
        self,
        contract: RouterRuntimeContract,
        *,
        allow_stale_owned_fingerprint: bool = False,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        inspection = self._inspect_container(contract.pair.container_name)
        if not inspection.exists or not inspection.running:
            return
        if not inspection.owned_by_router():
            raise RouterAdapterOwnershipError(
                f"Refusing to stop foreign container {contract.pair.container_name}."
            )
        if (
            not allow_stale_owned_fingerprint
            and not inspection.owned_by(contract)
        ):
            raise RouterAdapterOwnershipError(
                f"Refusing to stop Router container with a different ownership fingerprint: "
                f"{contract.pair.container_name}."
            )
        self._stop_inspection(inspection, cancel_checker=cancel_checker)

    def _stop_inspection(
        self,
        inspection: RouterContainerInspection,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        self._raise_if_cancelled(cancel_checker, "before Router container stop")
        completed = run_docker_command(
            [
                "docker",
                "stop",
                "--time",
                str(DEFAULT_MANAGED_RUNTIME_STOP_TIMEOUT_SEC),
                inspection.name,
            ],
            check=False,
            cancel_checker=cancel_checker,
        )
        if completed.returncode != 0:
            detail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip()
            raise RouterAdapterError(
                f"Unable to stop owned Router container {inspection.name}: {detail}"
            )
        after = self._inspect_container(inspection.name)
        if after.exists and after.running:
            raise RouterAdapterError(
                f"Docker reported Router container still running: {inspection.name}"
            )

    def stop_owned_port_occupants(
        self,
        contract: RouterRuntimeContract,
        *,
        cancel_checker: Callable[[], bool] | None = None,
    ) -> None:
        self.stop_owned_pair_ports(contract.pair, cancel_checker=cancel_checker)

    def stop_owned_pair_ports(
        self,
        pair: RouterPair,
        *,
        cancel_checker: Callable[[], bool] | None = None,
        reject_foreign: bool = True,
        require_ports_free: bool = True,
    ) -> tuple[str, ...]:
        """Free a Router pair's host ports, stopping only Router-owned containers.

        This works from the pair alone so the separate-server path can reclaim a
        Router container left behind by an earlier process, where no prepared
        contract exists yet.

        With ``reject_foreign`` the Router route refuses to bind a port held by
        anything it does not own.  The separate-server route clears
        ``reject_foreign`` and ``require_ports_free`` because the container it is
        about to reuse legitimately holds that port; only a Router leftover has
        to be released there.
        """

        occupants: dict[str, RouterContainerInspection] = {}
        released: list[str] = []
        for port in (pair.ocr_port, pair.gemma_port):
            for inspection in self._containers_publishing_port(port):
                occupants[inspection.name] = inspection
        for inspection in occupants.values():
            if not inspection.running:
                # An exited container does not own a host port. Compose can
                # recreate our own stopped container without mutating it here.
                continue
            if not inspection.owned_by_router():
                if reject_foreign:
                    raise RouterAdapterOwnershipError(
                        "Router port is held by a foreign container; it will not be stopped: "
                        f"{inspection.name}"
                    )
                continue
            self._stop_inspection(inspection, cancel_checker=cancel_checker)
            released.append(inspection.name)
        if require_ports_free:
            for port in (pair.ocr_port, pair.gemma_port):
                self._assert_host_port_available(
                    port,
                    cancel_checker=cancel_checker,
                )
        return tuple(released)

    def _assert_host_port_available(
        self,
        port: int,
        *,
        cancel_checker: Callable[[], bool] | None,
        timeout_sec: float | None = None,
    ) -> None:
        """Reject a direct or unowned listener without attempting to stop it.

        Docker's publish filter only identifies Docker containers. A raw
        llama-server or another host process using the same loopback port must
        be surfaced as an ownership error before Compose is allowed to issue a
        generic bind failure. The bind probe is read-only with respect to the
        existing listener: it never connects to or terminates that process.
        """

        deadline = time.monotonic() + max(
            0.0,
            float(
                self._PORT_RELEASE_TIMEOUT_SEC
                if timeout_sec is None
                else timeout_sec
            ),
        )
        last_error: OSError | None = None
        while True:
            self._raise_if_cancelled(
                cancel_checker,
                f"while checking Router port {port} ownership",
            )
            probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                probe.bind(("127.0.0.1", int(port)))
                return
            except OSError as exc:
                last_error = exc
                if exc.errno not in {errno.EADDRINUSE, errno.EACCES}:
                    raise RouterAdapterError(
                        f"Unable to verify Router port {port} availability: {exc}"
                    ) from exc
            finally:
                probe.close()
            if time.monotonic() >= deadline:
                raise RouterAdapterOwnershipError(
                    "Router port is held by a direct or unowned server; it will not "
                    f"be stopped: 127.0.0.1:{port} ({last_error})"
                )
            time.sleep(self._HEALTH_POLL_SEC)

    def owned_gpu_process_ids(
        self,
        contract: RouterRuntimeContract,
    ) -> frozenset[int]:
        inspection = self._inspect_container(contract.pair.container_name)
        if (
            not inspection.exists
            or not inspection.running
            or not inspection.owned_by(contract)
        ):
            return frozenset()
        host_completed = run_docker_command(
            ["docker", "top", inspection.name, "-eo", "pid"],
            check=False,
        )
        namespace_completed = run_docker_command(
            ["docker", "exec", inspection.name, "ps", "-eo", "pid"],
            check=False,
        )

        # Native Docker reports host PIDs through nvidia-smi, whereas Docker
        # Desktop / WSL reports a process PID from the container namespace.
        # Keep both namespaces, but only from the exact product-owned
        # container checked above. A driver PID must still be present in this
        # union before the handoff gate accepts its disappearance.
        pids: set[int] = set()
        for completed in (host_completed, namespace_completed):
            if completed.returncode != 0:
                continue
            for line in (completed.stdout or "").splitlines():
                value = line.strip()
                if not value or not value.isdigit():
                    continue
                pid = int(value)
                if pid > 0:
                    pids.add(pid)
        return frozenset(pids)

    def gpu_snapshot(self, contract: RouterRuntimeContract) -> dict[str, Any]:
        """Sample GPU totals and PIDs from the exact owned container namespace.

        On Docker Desktop/WSL, the host driver can expose a different PID
        namespace or no active-compute PID at all. When NVML has no PID, the
        verified Router container is inspected for an exact model child with
        an open ``/dev/dxg`` handle instead. This method never runs against a
        foreign or fingerprint-mismatched container.
        """

        inspection = self._inspect_container(contract.pair.container_name)
        if (
            not inspection.exists
            or not inspection.running
            or not inspection.owned_by(contract)
        ):
            return self._unavailable_gpu_snapshot("owned-router-container-unavailable")

        gpu = run_docker_command(
            [
                "docker",
                "exec",
                inspection.name,
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )
        processes = run_docker_command(
            [
                "docker",
                "exec",
                inspection.name,
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
        )
        if gpu.returncode != 0:
            return self._unavailable_gpu_snapshot("owned-router-nvidia-smi-failed")
        rows = self._parse_nvidia_gpu_rows(gpu.stdout or "")
        process_rows = (
            self._parse_nvidia_compute_rows(processes.stdout or "")
            if processes.returncode == 0
            else []
        )
        worker_processes = (
            self._router_worker_processes(
                inspection,
                contract,
                gpu_rows=rows,
            )
            if not process_rows
            else {
                "query_available": False,
                "rows": [],
                "source": "owned-router-container-proc",
                "reason": "nvml-active-process-visible",
            }
        )
        return {
            "sampled_at": time.time(),
            "process": {"available": False, "reason": "router-container-driver-view"},
            "driver": {
                "available": bool(rows),
                "gpu_count": len(rows),
                "gpus": rows,
                "primary": rows[0] if rows else None,
                "sampled_at": time.time(),
                "source": "owned-router-container",
            },
            "driver_process": {
                "query_available": False,
                "available": False,
                "rows": [],
                "selected": None,
                "reason": "router-container-driver-view",
            },
            "driver_processes": {
                "query_available": processes.returncode == 0,
                "rows": process_rows,
                "gpu_uuids": sorted(
                    {
                        str(row.get("gpu_uuid") or "").strip()
                        for row in process_rows
                    }
                )
                if processes.returncode == 0
                else [],
                "source": "owned-router-container",
            },
            "router_worker_processes": worker_processes,
        }

    @staticmethod
    def _unavailable_gpu_snapshot(reason: str) -> dict[str, Any]:
        return {
            "sampled_at": time.time(),
            "process": {"available": False, "reason": reason},
            "driver": {
                "available": False,
                "gpu_count": 0,
                "gpus": [],
                "primary": None,
                "source": "owned-router-container",
            },
            "driver_process": {
                "query_available": False,
                "available": False,
                "rows": [],
                "selected": None,
                "reason": reason,
            },
            "driver_processes": {
                "query_available": False,
                "rows": [],
                "gpu_uuids": [],
                "source": "owned-router-container",
            },
            "router_worker_processes": {
                "query_available": False,
                "rows": [],
                "source": "owned-router-container-proc",
                "reason": reason,
            },
        }

    def _router_worker_processes(
        self,
        inspection: RouterContainerInspection,
        contract: RouterRuntimeContract,
        *,
        gpu_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Return exact model workers with an open WSL GPU device handle.

        WSL's NVML compatibility layer can expose total GPU memory while
        returning no active-compute PIDs.  Never infer an empty PID set from
        that limitation: inspect only the already verified owned container,
        require a single visible GPU, an exact configured alias/model command,
        and an open ``/dev/dxg`` descriptor on the child process.
        """

        if len(gpu_rows) != 1:
            return {
                "query_available": False,
                "rows": [],
                "source": "owned-router-container-proc",
                "reason": "router-worker-fallback-requires-one-visible-gpu",
            }
        gpu_uuid = str(gpu_rows[0].get("uuid") or "").strip()
        if not gpu_uuid:
            return {
                "query_available": False,
                "rows": [],
                "source": "owned-router-container-proc",
                "reason": "router-worker-fallback-missing-gpu-uuid",
            }
        listing = run_docker_command(
            [
                "docker",
                "exec",
                inspection.name,
                "ps",
                "-eo",
                "pid=,args=",
            ],
            check=False,
        )
        if listing.returncode != 0:
            return {
                "query_available": False,
                "rows": [],
                "source": "owned-router-container-proc",
                "reason": "router-worker-process-list-failed",
            }

        expected_models = {
            contract.ocr_model.alias: contract.ocr_model,
            contract.gemma_model.alias: contract.gemma_model,
        }
        workers: list[dict[str, Any]] = []
        for raw_line in (listing.stdout or "").splitlines():
            parts = raw_line.strip().split(maxsplit=1)
            if len(parts) != 2 or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if pid <= 1:
                continue
            command = parts[1]
            try:
                arguments = shlex.split(command)
            except ValueError:
                continue
            alias = self._command_option(arguments, "--alias")
            model_path = self._command_option(arguments, "--model")
            material = expected_models.get(alias or "")
            if material is None or not model_path:
                continue
            if Path(model_path).name != Path(material.model_file).name:
                continue
            device_nodes = self._process_gpu_device_nodes(inspection.name, pid)
            if device_nodes is None:
                return {
                    "query_available": False,
                    "rows": [],
                    "source": "owned-router-container-proc",
                    "reason": "router-worker-device-fd-inspection-failed",
                }
            workers.append(
                {
                    "pid": pid,
                    "gpu_uuid": gpu_uuid,
                    "model_alias": alias,
                    "model_file": Path(material.model_file).name,
                    "gpu_device_attached": "/dev/dxg" in device_nodes,
                    "device_nodes": sorted(device_nodes),
                }
            )
        return {
            "query_available": True,
            "rows": workers,
            "source": "owned-router-container-proc",
            "reason": "",
        }

    @staticmethod
    def _command_option(arguments: Sequence[str], option: str) -> str:
        try:
            index = list(arguments).index(option)
        except ValueError:
            return ""
        if index + 1 >= len(arguments):
            return ""
        return str(arguments[index + 1] or "").strip()

    @staticmethod
    def _process_gpu_device_nodes(
        container_name: str,
        pid: int,
    ) -> set[str] | None:
        completed = run_docker_command(
            [
                "docker",
                "exec",
                container_name,
                "sh",
                "-lc",
                f"for fd in /proc/{int(pid)}/fd/*; do readlink \"$fd\" 2>/dev/null || true; done",
            ],
            check=False,
        )
        if completed.returncode != 0:
            return None
        return {
            line.strip()
            for line in (completed.stdout or "").splitlines()
            if line.strip().startswith("/dev/")
        }

    @staticmethod
    def _parse_nvidia_gpu_rows(output: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_line in output.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 7:
                continue
            try:
                rows.append(
                    {
                        "index": int(parts[0]),
                        "uuid": parts[1],
                        "name": parts[2],
                        "memory_total_mb": int(parts[3]),
                        "memory_used_mb": int(parts[4]),
                        "memory_free_mb": int(parts[5]),
                        "gpu_util_percent": int(parts[6]),
                        "memory_util_percent": int(parts[7]) if len(parts) > 7 else None,
                    }
                )
            except ValueError:
                continue
        return rows

    @staticmethod
    def _parse_nvidia_compute_rows(output: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for raw_line in output.splitlines():
            parts = [part.strip() for part in raw_line.split(",")]
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[0])
            except ValueError:
                continue
            try:
                memory_used: float | None = float(parts[-1])
            except ValueError:
                memory_used = None
            rows.append(
                {
                    "pid": pid,
                    "gpu_uuid": parts[1],
                    "process_name": ",".join(parts[2:-1]).strip(),
                    "memory_used_mb": memory_used,
                    "memory_reported": memory_used is not None,
                }
            )
        return rows

    def _compose_environment_for_spec(
        self,
        spec: RouterRuntimeSpec,
        *,
        fingerprint: str,
        preset_sha256: str,
    ) -> dict[str, str]:
        return {
            "LLAMA_ROUTER_IMAGE": spec.image_ref,
            "LLAMA_ROUTER_FINGERPRINT": fingerprint,
            "LLAMA_ROUTER_PRESET_SHA256": preset_sha256,
            "PADDLEOCR_ROUTER_MODEL_VOLUME": spec.ocr_model.volume_name,
            "GEMMA_ROUTER_MODEL_VOLUME": spec.gemma_model.volume_name,
            "PADDLEOCR_ROUTER_READY_MANIFEST_SHA256": (
                spec.ocr_model.ready_manifest_sha256
            ),
            "GEMMA_ROUTER_READY_MANIFEST_SHA256": (
                spec.gemma_model.ready_manifest_sha256
            ),
            "PADDLEOCR_ROUTER_MODEL_SHA256": spec.ocr_model.model_sha256,
            "PADDLEOCR_ROUTER_MMPROJ_SHA256": spec.ocr_model.mmproj_sha256,
            "GEMMA_ROUTER_MODEL_SHA256": spec.gemma_model.model_sha256,
        }

    def _ensure_image(
        self,
        image_ref: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        inspection = run_docker_command(
            ["docker", "image", "inspect", image_ref],
            check=False,
            cancel_checker=cancel_checker,
        )
        if inspection.returncode == 0:
            return
        pull = run_docker_command(
            ["docker", "pull", image_ref],
            check=False,
            cancel_checker=cancel_checker,
        )
        if pull.returncode != 0:
            detail = ((pull.stderr or "") + "\n" + (pull.stdout or "")).strip()
            raise RouterAdapterError(
                f"Unable to pull pinned Router image {image_ref}: {detail}"
            )

    def _inspect_image(
        self,
        image_ref: str,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> Mapping[str, Any]:
        completed = run_docker_command(
            ["docker", "image", "inspect", image_ref],
            check=False,
            cancel_checker=cancel_checker,
        )
        if completed.returncode != 0:
            raise RouterAdapterError(f"Unable to inspect Router image {image_ref}.")
        try:
            payload = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RouterAdapterError("Docker returned invalid Router image inspection JSON.") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise RouterAdapterError("Docker returned no Router image inspection object.")
        return payload[0]

    def _resolved_compose_config(
        self,
        pair: RouterPair,
        *,
        environment: Mapping[str, str],
        cancel_checker: Callable[[], bool] | None,
    ) -> Mapping[str, Any]:
        command = [
            *self._resolve_compose_command(cancel_checker=cancel_checker),
            "--project-name",
            ROUTER_PROJECT_NAME,
            "-f",
            str(pair.compose_file),
            "config",
            "--format",
            "json",
        ]
        completed = run_docker_command(
            command,
            cwd=pair.compose_file.parent,
            env=self._merged_environment(environment),
            check=False,
            cancel_checker=cancel_checker,
        )
        if completed.returncode != 0:
            detail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip()
            raise RouterAdapterError(f"Router Compose config failed: {detail}")
        try:
            payload = json.loads(completed.stdout or "{}")
        except json.JSONDecodeError as exc:
            raise RouterAdapterError("Docker Compose did not return JSON configuration.") from exc
        if not isinstance(payload, Mapping):
            raise RouterAdapterError("Docker Compose returned an invalid configuration object.")
        return copy.deepcopy(payload)

    def _run_compose(
        self,
        pair: RouterPair,
        args: Sequence[str],
        *,
        environment: Mapping[str, str],
        cancel_checker: Callable[[], bool] | None,
    ) -> None:
        command = [
            *self._resolve_compose_command(cancel_checker=cancel_checker),
            "--project-name",
            ROUTER_PROJECT_NAME,
            "-f",
            str(pair.compose_file),
            *args,
        ]
        completed = run_docker_command(
            command,
            cwd=pair.compose_file.parent,
            env=self._merged_environment(environment),
            check=False,
            cancel_checker=cancel_checker,
        )
        if completed.returncode != 0:
            detail = ((completed.stderr or "") + "\n" + (completed.stdout or "")).strip()
            raise RouterAdapterError(f"Router Docker Compose {' '.join(args)} failed: {detail}")

    def _resolve_compose_command(
        self,
        *,
        cancel_checker: Callable[[], bool] | None,
    ) -> tuple[str, ...]:
        if self._compose_command is None:
            self._compose_command = resolve_docker_compose_command(
                cancel_checker=cancel_checker,
            )
        return self._compose_command

    @staticmethod
    def _merged_environment(values: Mapping[str, str]) -> dict[str, str]:
        merged = dict(os.environ)
        merged.update({str(key): str(value) for key, value in values.items()})
        return merged

    @staticmethod
    def _select_repo_digest(image: Mapping[str, Any]) -> str:
        digests = image.get("RepoDigests")
        if not isinstance(digests, list):
            return ""
        for value in digests:
            text = str(value or "").strip()
            if text.startswith("ghcr.io/ggml-org/llama.cpp@sha256:"):
                return text
        for value in digests:
            text = str(value or "").strip()
            if "@sha256:" in text:
                return text
        return ""

    @staticmethod
    def _file_sha256(path: Path) -> str:
        return __import__("hashlib").sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _normalize_dynamic_values(value: Any, *, dynamic_values: set[str]) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): DockerRouterCommandAdapter._normalize_dynamic_values(
                    item,
                    dynamic_values=dynamic_values,
                )
                for key, item in sorted(value.items(), key=lambda item: str(item[0]))
            }
        if isinstance(value, list):
            return [
                DockerRouterCommandAdapter._normalize_dynamic_values(
                    item,
                    dynamic_values=dynamic_values,
                )
                for item in value
            ]
        if isinstance(value, tuple):
            return tuple(
                DockerRouterCommandAdapter._normalize_dynamic_values(
                    item,
                    dynamic_values=dynamic_values,
                )
                for item in value
            )
        text = str(value) if value is not None else value
        if isinstance(text, str) and text in dynamic_values:
            return "<router-dynamic>"
        return value

    @staticmethod
    def _service_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
        services = config.get("services")
        if not isinstance(services, Mapping):
            raise RouterAdapterError("Router Compose config omitted services.")
        service = services.get(ROUTER_SERVICE_NAME)
        if not isinstance(service, Mapping):
            raise RouterAdapterError(
                f"Router Compose config omitted {ROUTER_SERVICE_NAME!r} service."
            )
        return service

    @classmethod
    def _assert_resolved_compose_contract(
        cls,
        spec: RouterRuntimeSpec,
        config: Mapping[str, Any],
    ) -> None:
        """Reject a resolved Compose file that weakens the Router contract.

        The full resolved configuration is fingerprinted, but a fingerprint
        alone would merely record an unsafe edit such as a writable model
        mount.  These assertions make the product's two-port, read-only,
        one-GPU deployment a fail-closed precondition.
        """

        pair = spec.pair
        service = cls._service_config(config)
        labels = service.get("labels")
        if not isinstance(labels, Mapping) or (
            str(labels.get(ROUTER_OWNER_LABEL, "")) != ROUTER_OWNER_VALUE
            or str(labels.get(ROUTER_PAIR_LABEL, "")) != pair.kind.value
        ):
            raise RouterAdapterError(
                "Router Compose ownership labels do not match the selected pair."
            )

        ports = service.get("ports")
        if not isinstance(ports, list):
            raise RouterAdapterError("Router Compose ports are missing.")
        expected_ports = {
            ("127.0.0.1", str(pair.ocr_port), "8080", "tcp"),
            ("127.0.0.1", str(pair.gemma_port), "8080", "tcp"),
        }
        actual_ports: set[tuple[str, str, str, str]] = set()
        for entry in ports:
            if not isinstance(entry, Mapping):
                raise RouterAdapterError("Router Compose ports contain an invalid entry.")
            actual_ports.add(
                (
                    str(entry.get("host_ip") or ""),
                    str(entry.get("published") or ""),
                    str(entry.get("target") or ""),
                    str(entry.get("protocol") or "tcp").lower(),
                )
            )
        if actual_ports != expected_ports:
            raise RouterAdapterError(
                "Router Compose must publish only the exact localhost OCR and Gemma ports."
            )

        volumes = service.get("volumes")
        if not isinstance(volumes, list):
            raise RouterAdapterError("Router Compose model mounts are missing.")
        expected_mounts = {
            "/models/ocr": "paddleocr-router-models",
            "/models/gemma": "gemma-router-models",
            "/router/models.ini": None,
        }
        actual_mounts: dict[str, Mapping[str, Any]] = {}
        for entry in volumes:
            if not isinstance(entry, Mapping):
                raise RouterAdapterError("Router Compose mounts contain an invalid entry.")
            target = str(entry.get("target") or "")
            if not target or target in actual_mounts:
                raise RouterAdapterError("Router Compose mounts have an invalid target.")
            actual_mounts[target] = entry
        if set(actual_mounts) != set(expected_mounts):
            raise RouterAdapterError(
                "Router Compose must mount only OCR, Gemma, and preset files."
            )
        for target, expected_source in expected_mounts.items():
            mount = actual_mounts[target]
            if not bool(mount.get("read_only")):
                raise RouterAdapterError(
                    f"Router Compose mount must be read-only: {target}"
                )
            if expected_source is not None and str(mount.get("source") or "") != expected_source:
                raise RouterAdapterError(
                    f"Router Compose model mount source changed: {target}"
                )

        declared_volumes = config.get("volumes")
        if not isinstance(declared_volumes, Mapping):
            raise RouterAdapterError("Router Compose external volumes are missing.")
        for logical_name, expected_name in (
            ("paddleocr-router-models", spec.ocr_model.volume_name),
            ("gemma-router-models", spec.gemma_model.volume_name),
        ):
            declaration = declared_volumes.get(logical_name)
            if not isinstance(declaration, Mapping) or not bool(
                declaration.get("external")
            ) or str(declaration.get("name") or "") != expected_name:
                raise RouterAdapterError(
                    f"Router Compose external volume contract changed: {logical_name}"
                )

        environment = cls._service_environment(service)
        capabilities = {
            value.strip()
            for value in str(environment.get("NVIDIA_DRIVER_CAPABILITIES") or "").split(",")
            if value.strip()
        }
        if (
            str(environment.get("NVIDIA_VISIBLE_DEVICES") or "") != "all"
            or not {"compute", "utility"}.issubset(capabilities)
            or not service.get("gpus")
        ):
            raise RouterAdapterError(
                "Router Compose GPU device mapping does not satisfy the product contract."
            )

    @staticmethod
    def _service_environment(service: Mapping[str, Any]) -> dict[str, str]:
        environment = service.get("environment")
        if isinstance(environment, Mapping):
            return {
                str(key): str(value)
                for key, value in sorted(environment.items(), key=lambda item: str(item[0]))
            }
        if isinstance(environment, list):
            values: dict[str, str] = {}
            for entry in environment:
                key, separator, value = str(entry).partition("=")
                if key and separator:
                    values[key] = value
            return dict(sorted(values.items()))
        return {}

    @staticmethod
    def _service_ports(service: Mapping[str, Any]) -> Mapping[str, Any]:
        return copy.deepcopy(service.get("ports") or [])

    @staticmethod
    def _service_volumes(service: Mapping[str, Any]) -> Mapping[str, Any]:
        return copy.deepcopy(service.get("volumes") or [])

    @staticmethod
    def _service_devices(service: Mapping[str, Any]) -> Mapping[str, Any]:
        return {
            "gpus": copy.deepcopy(service.get("gpus")),
            "deploy": copy.deepcopy(service.get("deploy") or {}),
            "runtime": copy.deepcopy(service.get("runtime")),
        }

    @staticmethod
    def _string_tuple(value: Any) -> tuple[str, ...]:
        if isinstance(value, str):
            return (value,)
        if not isinstance(value, (list, tuple)):
            return ()
        return tuple(str(item) for item in value)

    def _inspect_container(self, name: str) -> RouterContainerInspection:
        completed = run_docker_command(
            ["docker", "inspect", name],
            check=False,
        )
        if completed.returncode != 0:
            return RouterContainerInspection(
                name=name,
                exists=False,
                running=False,
                image="",
                image_id="",
                labels={},
                command=(),
                entrypoint=(),
                ports={},
                mounts=(),
                device_requests=(),
                pid=None,
            )
        try:
            payload = json.loads(completed.stdout or "[]")
        except json.JSONDecodeError as exc:
            raise RouterAdapterError("Docker returned invalid Router container inspection JSON.") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise RouterAdapterError(f"Docker returned no container inspection for {name}.")
        item = payload[0]
        config = item.get("Config") if isinstance(item.get("Config"), Mapping) else {}
        state = item.get("State") if isinstance(item.get("State"), Mapping) else {}
        host_config = (
            item.get("HostConfig") if isinstance(item.get("HostConfig"), Mapping) else {}
        )
        labels = config.get("Labels") if isinstance(config.get("Labels"), Mapping) else {}
        network = (
            item.get("NetworkSettings")
            if isinstance(item.get("NetworkSettings"), Mapping)
            else {}
        )
        mounts = item.get("Mounts") if isinstance(item.get("Mounts"), list) else []
        requests_ = (
            host_config.get("DeviceRequests")
            if isinstance(host_config.get("DeviceRequests"), list)
            else []
        )
        raw_pid = state.get("Pid")
        try:
            pid = int(raw_pid) if int(raw_pid) > 0 else None
        except (TypeError, ValueError):
            pid = None
        return RouterContainerInspection(
            name=str(item.get("Name") or name).lstrip("/"),
            exists=True,
            running=bool(state.get("Running", False)),
            image=str(config.get("Image") or ""),
            image_id=str(item.get("Image") or ""),
            labels={str(key): str(value) for key, value in labels.items()},
            command=self._string_tuple(config.get("Cmd")),
            entrypoint=self._string_tuple(config.get("Entrypoint")),
            ports=copy.deepcopy(network.get("Ports") or {}),
            mounts=tuple(
                copy.deepcopy(value) for value in mounts if isinstance(value, Mapping)
            ),
            device_requests=tuple(
                copy.deepcopy(value) for value in requests_ if isinstance(value, Mapping)
            ),
            pid=pid,
        )

    def _assert_container_contract(
        self,
        inspection: RouterContainerInspection,
        contract: RouterRuntimeContract,
    ) -> None:
        if not inspection.exists or not inspection.running:
            raise RouterAdapterError(
                f"Router container did not start: {contract.pair.container_name}"
            )
        if not inspection.owned_by(contract):
            raise RouterAdapterOwnershipError(
                f"Router container ownership labels do not match: {contract.pair.container_name}"
            )
        if canonical_sha256(inspection.command) != contract.command_sha256:
            raise RouterAdapterError(
                "Router container command SHA does not match the fingerprinted command."
            )
        if inspection.image_id and inspection.image_id != contract.image_id:
            raise RouterAdapterError(
                "Router container image ID does not match the fingerprinted image."
            )
        if tuple(inspection.entrypoint) != tuple(contract.entrypoint):
            raise RouterAdapterError(
                "Router container entrypoint does not match the fingerprinted image."
            )

        expected_ports = {
            ("8080/tcp", "127.0.0.1", str(contract.pair.ocr_port)),
            ("8080/tcp", "127.0.0.1", str(contract.pair.gemma_port)),
        }
        actual_ports: set[tuple[str, str, str]] = set()
        for container_port, bindings in inspection.ports.items():
            if not isinstance(bindings, list):
                continue
            for binding in bindings:
                if not isinstance(binding, Mapping):
                    continue
                actual_ports.add(
                    (
                        str(container_port),
                        str(binding.get("HostIp") or ""),
                        str(binding.get("HostPort") or ""),
                    )
                )
        if actual_ports != expected_ports:
            raise RouterAdapterError(
                "Router container port bindings do not match the exact localhost contract."
            )

        expected_volume_names = {
            "/models/ocr": contract.ocr_model.volume_name,
            "/models/gemma": contract.gemma_model.volume_name,
        }
        expected_targets = set(expected_volume_names) | {"/router/models.ini"}
        actual_mounts: dict[str, Mapping[str, Any]] = {}
        for mount in inspection.mounts:
            target = str(mount.get("Destination") or "")
            if not target or target in actual_mounts:
                raise RouterAdapterError("Router container mount inspection is ambiguous.")
            actual_mounts[target] = mount
        if set(actual_mounts) != expected_targets:
            raise RouterAdapterError(
                "Router container mounts do not match the read-only model contract."
            )
        for target, volume_name in expected_volume_names.items():
            mount = actual_mounts[target]
            if (
                str(mount.get("Type") or "") != "volume"
                or str(mount.get("Name") or "") != volume_name
                or bool(mount.get("RW", True))
            ):
                raise RouterAdapterError(
                    f"Router model mount does not match the read-only contract: {target}"
                )
        preset_mount = actual_mounts["/router/models.ini"]
        if (
            str(preset_mount.get("Type") or "") != "bind"
            or Path(str(preset_mount.get("Source") or "")).name
            != contract.pair.preset_file.name
            or bool(preset_mount.get("RW", True))
        ):
            raise RouterAdapterError(
                "Router preset mount does not match the read-only contract."
            )

        has_gpu_request = False
        for request in inspection.device_requests:
            capabilities = request.get("Capabilities")
            if not isinstance(capabilities, list):
                continue
            if any(
                isinstance(group, list)
                and "gpu" in {str(value).lower() for value in group}
                for group in capabilities
            ):
                has_gpu_request = True
                break
        if not has_gpu_request:
            raise RouterAdapterError(
                "Router container does not expose the required GPU device request."
            )

    def _containers_publishing_port(self, port: int) -> tuple[RouterContainerInspection, ...]:
        completed = run_docker_command(
            [
                "docker",
                "ps",
                "--all",
                "--filter",
                f"publish={int(port)}",
                "--format",
                "{{.Names}}",
            ],
            check=False,
        )
        if completed.returncode != 0:
            raise RouterAdapterError(f"Unable to inspect Docker port {port} ownership.")
        names = [line.strip() for line in (completed.stdout or "").splitlines() if line.strip()]
        return tuple(self._inspect_container(name) for name in names)

    def _wait_for_health(
        self,
        pair: RouterPair,
        *,
        cancel_checker: Callable[[], bool] | None,
        timeout_sec: float = 120.0,
    ) -> None:
        deadline = time.monotonic() + max(1.0, float(timeout_sec))
        last_error = ""
        while time.monotonic() < deadline:
            self._raise_if_cancelled(cancel_checker, "while waiting for Router health")
            try:
                response = self._session.get(
                    f"{pair.router_base_url}/health",
                    timeout=5.0,
                )
                if response.status_code == 200:
                    return
                last_error = f"HTTP {response.status_code}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(self._HEALTH_POLL_SEC)
        raise RouterAdapterError(
            f"Router health did not become ready at {pair.router_base_url}/health: {last_error}"
        )

    def _request_payload(
        self,
        method: str,
        url: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        timeout_sec: float,
    ) -> Any:
        normalized_method = str(method or "").upper()
        attempts = 2 if normalized_method == "GET" else 1
        response: requests.Response | None = None
        last_error: requests.RequestException | None = None
        for attempt_index in range(attempts):
            try:
                response = self._session.request(
                    normalized_method,
                    url,
                    json=dict(json_payload) if json_payload is not None else None,
                    timeout=max(0.1, float(timeout_sec)),
                )
                break
            except requests.RequestException as exc:
                last_error = exc
                if attempt_index + 1 >= attempts:
                    raise RouterAdapterError(
                        f"Router request failed: {normalized_method} {url}"
                    ) from exc
                # llama.cpp may close an idle keep-alive socket as a child
                # model exits. Retrying a read on a fresh connection is safe;
                # POST load/unload commands deliberately never get retried.
                try:
                    self._session.close()
                except Exception:
                    pass
        if response is None:
            raise RouterAdapterError(
                f"Router request failed: {normalized_method} {url}"
            ) from last_error
        if response.status_code < 200 or response.status_code >= 300:
            detail = response.text.strip()
            raise RouterAdapterError(
                f"Router request failed: {normalized_method} {url} HTTP {response.status_code}: {detail}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise RouterAdapterError(
                f"Router returned invalid JSON: {normalized_method} {url}"
            ) from exc

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        json_payload: Mapping[str, Any] | None = None,
        timeout_sec: float,
    ) -> dict[str, Any]:
        payload = self._request_payload(
            method,
            url,
            json_payload=json_payload,
            timeout_sec=timeout_sec,
        )
        if not isinstance(payload, dict):
            raise RouterAdapterError(f"Router returned a non-object JSON payload: {method} {url}")
        return payload

    @staticmethod
    def _slots_list(payload: Any) -> list[Mapping[str, Any]]:
        if isinstance(payload, Mapping):
            candidate: Any = payload.get("slots")
            if candidate is None:
                candidate = payload.get("data")
        else:
            candidate = payload
        if not isinstance(candidate, list):
            raise RouterAdapterError("Router slots API did not return a slot list.")
        slots: list[Mapping[str, Any]] = []
        for entry in candidate:
            if not isinstance(entry, Mapping):
                raise RouterAdapterError("Router slots API returned a non-object slot entry.")
            slots.append(entry)
        return slots

    @staticmethod
    def _raise_if_cancelled(
        cancel_checker: Callable[[], bool] | None,
        when: str,
    ) -> None:
        if callable(cancel_checker):
            try:
                cancelled = bool(cancel_checker())
            except Exception:
                cancelled = True
            if cancelled:
                raise OperationCancelledError(f"Router operation cancelled {when}.")
