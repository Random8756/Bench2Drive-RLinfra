"""CUDA to Vulkan adapter mapping helpers for distributed rl_finetune."""

from __future__ import annotations

import csv
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


NVIDIA_ICD_BASENAME = "nvidia_icd.json"


@dataclass(frozen=True)
class GpuMappingResult:
    cuda_to_vulkan: Dict[int, int]
    mapping_source: str
    cuda_devices: List[Dict[str, Any]]
    vulkan_devices: List[Dict[str, Any]]
    vulkaninfo_path: str
    vk_icd_filenames: str
    identity_fallback_used: bool = False
    identity_fallback_reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cuda_to_vulkan": {int(k): int(v) for k, v in self.cuda_to_vulkan.items()},
            "mapping_source": self.mapping_source,
            "cuda_devices": self.cuda_devices,
            "vulkan_devices": self.vulkan_devices,
            "vulkaninfo_path": self.vulkaninfo_path,
            "vk_icd_filenames": self.vk_icd_filenames,
            "identity_fallback_used": bool(self.identity_fallback_used),
            "identity_fallback_reason": self.identity_fallback_reason,
            "warnings": list(self.warnings),
        }


def detect_cuda_to_vulkan_mapping(
    *,
    project_root: Path,
    env: Optional[Mapping[str, str]] = None,
    env_type: str = "carla",
    allow_identity_fallback: bool = True,
    command_timeout: float = 20.0,
) -> GpuMappingResult:
    """Resolve CUDA logical indices to UE4/Vulkan graphicsadapter indices."""

    env_map = dict(os.environ if env is None else env)
    vk_icd_filenames = env_map.get("VK_ICD_FILENAMES", "")
    if str(env_type).lower() == "fake":
        return GpuMappingResult(
            cuda_to_vulkan={idx: idx for idx in range(32)},
            mapping_source="fake",
            cuda_devices=[],
            vulkan_devices=[],
            vulkaninfo_path="",
            vk_icd_filenames=vk_icd_filenames,
        )

    try:
        cuda_query = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,pci.bus_id,name", "--format=csv,noheader"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=float(command_timeout),
            env=_subprocess_env(env_map),
        )
    except Exception as exc:
        raise RuntimeError(f"nvidia-smi query failed; cannot resolve CUDA to Vulkan mapping: {exc}") from exc
    cuda_devices = parse_nvidia_smi_csv(cuda_query.stdout)
    if not cuda_devices:
        raise RuntimeError("nvidia-smi returned no CUDA devices")

    warnings: List[str] = []
    vulkaninfo_path, bundled = find_vulkaninfo(project_root, env_map)
    vulkan_devices: List[Dict[str, Any]] = []
    if vulkaninfo_path:
        vk_env = _subprocess_env(env_map)
        if bundled:
            tool_dir = str(Path(vulkaninfo_path).resolve().parent)
            existing = vk_env.get("LD_LIBRARY_PATH", "")
            vk_env["LD_LIBRARY_PATH"] = f"{tool_dir}{os.pathsep}{existing}" if existing else tool_dir
        try:
            vk = subprocess.run(
                [str(vulkaninfo_path), "--summary"],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=float(command_timeout),
                env=vk_env,
            )
            vulkan_devices = parse_vulkaninfo_summary(vk.stdout)
        except Exception as exc:
            warnings.append(f"vulkaninfo_failed: {exc}")
            vulkan_devices = []
    else:
        warnings.append("vulkaninfo_not_found")

    return resolve_cuda_to_vulkan(
        cuda_devices,
        vulkan_devices,
        vk_icd_filenames=vk_icd_filenames,
        cuda_visible_devices=env_map.get("CUDA_VISIBLE_DEVICES", ""),
        allow_identity_fallback=allow_identity_fallback,
        vulkaninfo_path=str(vulkaninfo_path or ""),
        warnings=warnings,
    )


def find_vulkaninfo(project_root: Path, env: Optional[Mapping[str, str]] = None) -> Tuple[str, bool]:
    env_map = dict(os.environ if env is None else env)
    system_path = shutil.which("vulkaninfo", path=env_map.get("PATH"))
    if system_path:
        return system_path, False
    bundled = Path(project_root) / "third_party" / "vulkan-tools" / "vulkaninfo"
    if bundled.exists():
        return str(bundled), True
    return "", False


def parse_nvidia_smi_csv(text: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    for row in csv.reader(text.splitlines(), skipinitialspace=True):
        if not row or not any(cell.strip() for cell in row):
            continue
        if row[0].strip().lower() == "index":
            continue
        if len(row) < 4:
            raise ValueError(f"Malformed nvidia-smi row: {row!r}")
        try:
            index = int(row[0].strip())
        except ValueError as exc:
            raise ValueError(f"Malformed nvidia-smi GPU index: {row[0]!r}") from exc
        name = ",".join(row[3:]).strip()
        devices.append(
            {
                "index": index,
                "uuid": row[1].strip(),
                "uuid_normalized": _normalize_uuid(row[1]),
                "bus_id": _normalize_pci_bus_id(row[2]),
                "name": name,
                "is_nvidia": "nvidia" in name.lower() or row[1].strip().lower().startswith("gpu-"),
            }
        )
    return devices


def parse_vulkaninfo_summary(text: str) -> List[Dict[str, Any]]:
    devices: List[Dict[str, Any]] = []
    current: Optional[Dict[str, Any]] = None

    def flush() -> None:
        nonlocal current
        if current is None:
            return
        _finalize_vulkan_device(current)
        devices.append(current)
        current = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"^GPU(?P<idx>\d+)\s*:\s*(?P<name>.*)$", line)
        if match is None:
            match = re.match(r"^GPU\s+id\s*(?:=|:)\s*(?P<idx>\d+)(?:\s*\((?P<name>[^)]*)\))?", line, re.I)
        if match is not None:
            flush()
            current = {
                "index": int(match.group("idx")),
                "name": (match.group("name") or "").strip(),
                "uuid": "",
                "uuid_normalized": "",
                "bus_id": "",
                "vendor_id": "",
                "device_type": "",
                "driver_name": "",
            }
            continue
        if current is None or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key_norm = re.sub(r"[^a-z0-9]", "", key.strip().lower())
        value = value.strip()
        if key_norm == "devicename":
            current["name"] = value
        elif key_norm == "deviceuuid":
            current["uuid"] = value
            current["uuid_normalized"] = _normalize_uuid(value)
        elif key_norm == "vendorid":
            current["vendor_id"] = value.lower()
        elif key_norm == "devicetype":
            current["device_type"] = value
        elif key_norm in {"drivername", "driverid"}:
            current["driver_name"] = value
        elif key_norm in {"pcibusid", "pcibusinfo"}:
            current["bus_id"] = _normalize_pci_bus_id(value)
        elif key_norm == "pcidomain":
            current["_pci_domain"] = _parse_int_auto(value)
        elif key_norm == "pcibus":
            current["_pci_bus"] = _parse_int_auto(value)
        elif key_norm == "pcidevice":
            current["_pci_device"] = _parse_int_auto(value)
        elif key_norm == "pcifunction":
            current["_pci_function"] = _parse_int_auto(value)
    flush()
    if not devices:
        raise ValueError("Could not parse any Vulkan GPU adapters from vulkaninfo --summary")
    return devices


def resolve_cuda_to_vulkan(
    cuda_devices: Sequence[Mapping[str, Any]],
    vulkan_devices: Sequence[Mapping[str, Any]],
    *,
    vk_icd_filenames: str,
    cuda_visible_devices: str,
    allow_identity_fallback: bool,
    vulkaninfo_path: str = "",
    warnings: Optional[Sequence[str]] = None,
) -> GpuMappingResult:
    cuda_list = _logical_cuda_devices(cuda_devices, cuda_visible_devices)
    vulkan_list = [dict(device) for device in vulkan_devices]
    warning_list = list(warnings or [])

    if vulkan_list:
        physical = [
            device
            for device in vulkan_list
            if not bool(device.get("is_software", False)) and bool(device.get("is_nvidia", False))
        ]
        uuid_to_index = {
            str(device.get("uuid_normalized", "")): int(device["index"])
            for device in physical
            if device.get("uuid_normalized")
        }
        if cuda_list and all(str(device.get("uuid_normalized", "")) in uuid_to_index for device in cuda_list):
            return GpuMappingResult(
                cuda_to_vulkan={
                    int(device["index"]): int(uuid_to_index[str(device.get("uuid_normalized", ""))])
                    for device in cuda_list
                },
                mapping_source="vulkan_uuid",
                cuda_devices=cuda_list,
                vulkan_devices=vulkan_list,
                vulkaninfo_path=vulkaninfo_path,
                vk_icd_filenames=vk_icd_filenames,
                warnings=warning_list,
            )

        pci_to_index = {
            str(device.get("bus_id", "")): int(device["index"])
            for device in physical
            if device.get("bus_id")
        }
        if cuda_list and all(str(device.get("bus_id", "")) in pci_to_index for device in cuda_list):
            return GpuMappingResult(
                cuda_to_vulkan={int(device["index"]): int(pci_to_index[str(device.get("bus_id", ""))]) for device in cuda_list},
                mapping_source="vulkan_pci",
                cuda_devices=cuda_list,
                vulkan_devices=vulkan_list,
                vulkaninfo_path=vulkaninfo_path,
                vk_icd_filenames=vk_icd_filenames,
                warnings=warning_list,
            )

        raise RuntimeError(
            "Could not align CUDA devices with Vulkan adapters by UUID or PCI bus. "
            "Set VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json, provide a usable "
            "vulkaninfo with UUID/PCI fields, or configure env.carla.gpu_id explicitly."
        )

    mapping, reason = _identity_fallback_mapping(
        cuda_list,
        vk_icd_filenames=vk_icd_filenames,
        cuda_visible_devices=cuda_visible_devices,
        allow_identity_fallback=allow_identity_fallback,
    )
    return GpuMappingResult(
        cuda_to_vulkan=mapping,
        mapping_source="identity_fallback",
        cuda_devices=cuda_list,
        vulkan_devices=vulkan_list,
        vulkaninfo_path=vulkaninfo_path,
        vk_icd_filenames=vk_icd_filenames,
        identity_fallback_used=True,
        identity_fallback_reason=reason,
        warnings=warning_list,
    )


def _identity_fallback_mapping(
    cuda_devices: Sequence[Mapping[str, Any]],
    *,
    vk_icd_filenames: str,
    cuda_visible_devices: str,
    allow_identity_fallback: bool,
) -> Tuple[Dict[int, int], str]:
    failures: List[str] = []
    if not allow_identity_fallback:
        failures.append("distributed.cuda_vulkan_identity_fallback=false")
    if not _uses_only_nvidia_icd(vk_icd_filenames):
        failures.append("VK_ICD_FILENAMES is not restricted to nvidia_icd.json")
    indices = [int(device["index"]) for device in cuda_devices]
    expected = list(range(len(indices)))
    if indices != expected:
        failures.append(f"CUDA indices are not contiguous from 0: {indices}")
    if not _cuda_visible_devices_is_contiguous(cuda_visible_devices, expected):
        failures.append(f"CUDA_VISIBLE_DEVICES is not contiguous from 0: {cuda_visible_devices!r}")
    names = [str(device.get("name", "")).strip() for device in cuda_devices]
    lowered = [name.lower() for name in names if name]
    if not lowered or any("nvidia" not in name for name in lowered):
        failures.append("nvidia-smi devices are not all NVIDIA GPUs")
    if len(set(lowered)) > 1:
        failures.append(f"nvidia-smi devices are not homogeneous: {names}")
    if failures:
        raise RuntimeError(
            "Cannot resolve CUDA to Vulkan mapping without usable vulkaninfo; identity fallback refused: "
            + "; ".join(failures)
            + ". Set VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json, provide vulkaninfo, "
            + "or configure env.carla.gpu_id explicitly."
        )
    return {index: index for index in indices}, "strict_nvidia_icd_homogeneous_contiguous_cuda"


def _logical_cuda_devices(
    cuda_devices: Sequence[Mapping[str, Any]],
    cuda_visible_devices: str,
) -> List[Dict[str, Any]]:
    devices = [dict(device) for device in cuda_devices]
    by_index = {int(device["index"]): device for device in devices}
    by_uuid = {
        str(device.get("uuid_normalized", "")): device
        for device in devices
        if device.get("uuid_normalized")
    }
    text = str(cuda_visible_devices or "").strip()
    if not text:
        result = []
        for device in sorted(devices, key=lambda item: int(item["index"])):
            remapped = dict(device)
            remapped["physical_index"] = int(device["index"])
            remapped["index"] = int(device["index"])
            result.append(remapped)
        return result

    result: List[Dict[str, Any]] = []
    for logical_index, token in enumerate(item.strip() for item in text.split(",") if item.strip()):
        device: Optional[Mapping[str, Any]] = None
        try:
            device = by_index.get(int(token))
        except ValueError:
            device = by_uuid.get(_normalize_uuid(token))
        if device is None:
            raise RuntimeError(
                f"CUDA_VISIBLE_DEVICES entry {token!r} was not found in nvidia-smi output; "
                "cannot resolve CUDA to Vulkan mapping"
            )
        remapped = dict(device)
        remapped["physical_index"] = int(device["index"])
        remapped["index"] = int(logical_index)
        remapped["cuda_visible_token"] = token
        result.append(remapped)
    return result


def _finalize_vulkan_device(device: Dict[str, Any]) -> None:
    if not device.get("bus_id") and all(
        key in device for key in ("_pci_domain", "_pci_bus", "_pci_device", "_pci_function")
    ):
        device["bus_id"] = (
            f"{int(device['_pci_domain']):08X}:"
            f"{int(device['_pci_bus']):02X}:"
            f"{int(device['_pci_device']):02X}."
            f"{int(device['_pci_function'])}"
        )
    for key in ("_pci_domain", "_pci_bus", "_pci_device", "_pci_function"):
        device.pop(key, None)
    name = str(device.get("name", ""))
    driver_name = str(device.get("driver_name", ""))
    vendor_id = str(device.get("vendor_id", "")).lower()
    device_type = str(device.get("device_type", "")).lower()
    lower_name = name.lower()
    device["is_software"] = any(token in lower_name for token in ("llvmpipe", "software")) or "cpu" in device_type
    device["is_nvidia"] = vendor_id == "0x10de" or "nvidia" in lower_name or "nvidia" in driver_name.lower()


def _normalize_uuid(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("gpu-"):
        text = text[4:]
    return text


def _normalize_pci_bus_id(value: Any) -> str:
    text = str(value or "").strip().upper()
    match = re.match(r"^([0-9A-F]{4,8}):([0-9A-F]{2}):([0-9A-F]{2})\.([0-7])$", text)
    if not match:
        return text
    domain, bus, device, function = match.groups()
    return f"{int(domain, 16):08X}:{bus}:{device}.{function}"


def _parse_int_auto(value: str) -> int:
    token = str(value).strip().split()[0]
    return int(token, 0)


def _uses_only_nvidia_icd(vk_icd_filenames: str) -> bool:
    text = str(vk_icd_filenames or "").strip()
    if not text:
        return False
    parts = [part for part in re.split(r"[:;]", text) if part]
    return bool(parts) and all(Path(part).name == NVIDIA_ICD_BASENAME for part in parts)


def _cuda_visible_devices_is_contiguous(value: str, expected: Sequence[int]) -> bool:
    text = str(value or "").strip()
    if not text:
        return True
    try:
        values = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError:
        return False
    return values == list(expected)


def _subprocess_env(env: Mapping[str, str]) -> Dict[str, str]:
    return {str(key): str(value) for key, value in env.items() if value is not None}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Resolve CUDA logical GPU ids to Vulkan graphicsadapter ids.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root used to find optional third_party/vulkan-tools/vulkaninfo.",
    )
    parser.add_argument("--env-type", default="carla", help="Use 'fake' to emit an identity fake-env mapping.")
    parser.add_argument(
        "--no-identity-fallback",
        action="store_true",
        help="Disable strict cuda:N -> graphicsadapter:N fallback when Vulkan enumeration is unavailable.",
    )
    parser.add_argument("--timeout", type=float, default=20.0, help="Subprocess timeout in seconds.")
    parser.add_argument("--json", action="store_true", help="Print JSON output.")
    args = parser.parse_args(argv)

    try:
        result = detect_cuda_to_vulkan_mapping(
            project_root=args.project_root,
            env=os.environ,
            env_type=args.env_type,
            allow_identity_fallback=not args.no_identity_fallback,
            command_timeout=args.timeout,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    data = result.to_dict()
    if args.json:
        print(json.dumps(data, indent=2, sort_keys=True))
    else:
        print(f"mapping_source: {data['mapping_source']}")
        print(f"cuda_to_vulkan: {data['cuda_to_vulkan']}")
        print(f"vulkaninfo_path: {data['vulkaninfo_path'] or '<none>'}")
        print(f"VK_ICD_FILENAMES: {data['vk_icd_filenames'] or '<unset>'}")
        if data.get("identity_fallback_used"):
            print(f"identity_fallback_reason: {data.get('identity_fallback_reason', '')}")
        for warning in data.get("warnings", []):
            print(f"warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
