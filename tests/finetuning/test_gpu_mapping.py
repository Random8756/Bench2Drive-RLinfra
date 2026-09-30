from __future__ import annotations

import unittest

from b2d_rlinfra.finetuning.gpu_mapping import (
    parse_nvidia_smi_csv,
    parse_vulkaninfo_summary,
    resolve_cuda_to_vulkan,
)


NVIDIA_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"


def _cuda_csv(count: int = 8, *, mixed: bool = False) -> str:
    rows = []
    for idx in range(count):
        name = "NVIDIA A100-SXM4-80GB"
        if mixed and idx == count - 1:
            name = "NVIDIA L40S"
        rows.append(
            f"{idx}, GPU-{idx:08x}-aaaa-bbbb-cccc-{idx:012x}, "
            f"00000000:{0x27 + idx:02X}:00.0, {name}"
        )
    return "\n".join(rows)


def _vulkan_nvidia_summary(count: int = 8) -> str:
    blocks = []
    for idx in range(count):
        blocks.append(
            "\n".join(
                [
                    f"GPU{idx}:",
                    "    deviceName = NVIDIA A100-SXM4-80GB",
                    "    vendorID = 0x10de",
                    "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                    f"    deviceUUID = {idx:08x}-aaaa-bbbb-cccc-{idx:012x}",
                    "    VkPhysicalDevicePCIBusInfoPropertiesEXT:",
                    "    pciDomain = 0",
                    f"    pciBus = {0x27 + idx}",
                    "    pciDevice = 0",
                    "    pciFunction = 0",
                ]
            )
        )
    return "\n".join(blocks)


class GpuMappingTest(unittest.TestCase):
    def test_uuid_mapping_skips_llvmpipe(self) -> None:
        cuda_devices = parse_nvidia_smi_csv(_cuda_csv(2))
        vulkan_devices = parse_vulkaninfo_summary(
            "\n".join(
                [
                    "GPU0:",
                    "    deviceName = NVIDIA A100-SXM4-80GB",
                    "    vendorID = 0x10de",
                    "    deviceUUID = 00000000-aaaa-bbbb-cccc-000000000000",
                    "GPU1:",
                    "    deviceName = llvmpipe (LLVM 15.0.7, 256 bits)",
                    "    vendorID = 0x10005",
                    "    deviceType = PHYSICAL_DEVICE_TYPE_CPU",
                    "GPU2:",
                    "    deviceName = NVIDIA A100-SXM4-80GB",
                    "    vendorID = 0x10de",
                    "    deviceUUID = 00000001-aaaa-bbbb-cccc-000000000001",
                ]
            )
        )
        result = resolve_cuda_to_vulkan(
            cuda_devices,
            vulkan_devices,
            vk_icd_filenames=NVIDIA_ICD,
            cuda_visible_devices="0,1",
            allow_identity_fallback=True,
            vulkaninfo_path="/usr/bin/vulkaninfo",
        )
        self.assertEqual(result.mapping_source, "vulkan_uuid")
        self.assertEqual(result.cuda_to_vulkan, {0: 0, 1: 2})

    def test_pci_mapping_when_uuid_is_missing(self) -> None:
        cuda_devices = parse_nvidia_smi_csv(_cuda_csv(2))
        vulkan_devices = parse_vulkaninfo_summary(
            "\n".join(
                [
                    "GPU0:",
                    "    deviceName = NVIDIA A100-SXM4-80GB",
                    "    vendorID = 0x10de",
                    "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                    "    pciDomain = 0",
                    "    pciBus = 40",
                    "    pciDevice = 0",
                    "    pciFunction = 0",
                    "GPU1:",
                    "    deviceName = NVIDIA A100-SXM4-80GB",
                    "    vendorID = 0x10de",
                    "    deviceType = PHYSICAL_DEVICE_TYPE_DISCRETE_GPU",
                    "    pciDomain = 0",
                    "    pciBus = 39",
                    "    pciDevice = 0",
                    "    pciFunction = 0",
                ]
            )
        )
        result = resolve_cuda_to_vulkan(
            cuda_devices,
            vulkan_devices,
            vk_icd_filenames=NVIDIA_ICD,
            cuda_visible_devices="0,1",
            allow_identity_fallback=True,
            vulkaninfo_path="/usr/bin/vulkaninfo",
        )
        self.assertEqual(result.mapping_source, "vulkan_pci")
        self.assertEqual(result.cuda_to_vulkan, {0: 1, 1: 0})

    def test_forced_nvidia_icd_uuid_mapping_can_be_identity(self) -> None:
        cuda_devices = parse_nvidia_smi_csv(_cuda_csv(8))
        vulkan_devices = parse_vulkaninfo_summary(_vulkan_nvidia_summary(8))
        result = resolve_cuda_to_vulkan(
            cuda_devices,
            vulkan_devices,
            vk_icd_filenames=NVIDIA_ICD,
            cuda_visible_devices="0,1,2,3,4,5,6,7",
            allow_identity_fallback=True,
            vulkaninfo_path="/usr/bin/vulkaninfo",
        )
        self.assertEqual(result.mapping_source, "vulkan_uuid")
        self.assertEqual(result.cuda_to_vulkan, {idx: idx for idx in range(8)})

    def test_cuda_visible_devices_reindexes_cuda_logical_ids(self) -> None:
        cuda_devices = parse_nvidia_smi_csv(_cuda_csv(8))
        vulkan_devices = parse_vulkaninfo_summary(_vulkan_nvidia_summary(8))
        result = resolve_cuda_to_vulkan(
            cuda_devices,
            vulkan_devices,
            vk_icd_filenames=NVIDIA_ICD,
            cuda_visible_devices="5,2",
            allow_identity_fallback=True,
            vulkaninfo_path="/usr/bin/vulkaninfo",
        )
        self.assertEqual(result.mapping_source, "vulkan_uuid")
        self.assertEqual(result.cuda_to_vulkan, {0: 5, 1: 2})
        self.assertEqual(result.cuda_devices[0]["physical_index"], 5)
        self.assertEqual(result.cuda_devices[1]["physical_index"], 2)

    def test_no_vulkaninfo_allows_strict_identity_fallback(self) -> None:
        cuda_devices = parse_nvidia_smi_csv(_cuda_csv(8))
        result = resolve_cuda_to_vulkan(
            cuda_devices,
            [],
            vk_icd_filenames=NVIDIA_ICD,
            cuda_visible_devices="0,1,2,3,4,5,6,7",
            allow_identity_fallback=True,
            warnings=["vulkaninfo_not_found"],
        )
        self.assertEqual(result.mapping_source, "identity_fallback")
        self.assertTrue(result.identity_fallback_used)
        self.assertEqual(result.cuda_to_vulkan, {idx: idx for idx in range(8)})
        self.assertIn("strict_nvidia_icd", result.identity_fallback_reason)

    def test_identity_fallback_refuses_unsafe_inputs(self) -> None:
        cases = [
            {
                "allow_identity_fallback": False,
                "vk_icd_filenames": NVIDIA_ICD,
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "cuda_devices": parse_nvidia_smi_csv(_cuda_csv(8)),
            },
            {
                "allow_identity_fallback": True,
                "vk_icd_filenames": NVIDIA_ICD,
                "cuda_visible_devices": "2,5",
                "cuda_devices": parse_nvidia_smi_csv(_cuda_csv(8)),
            },
            {
                "allow_identity_fallback": True,
                "vk_icd_filenames": NVIDIA_ICD,
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "cuda_devices": parse_nvidia_smi_csv(_cuda_csv(8, mixed=True)),
            },
            {
                "allow_identity_fallback": True,
                "vk_icd_filenames": "",
                "cuda_visible_devices": "0,1,2,3,4,5,6,7",
                "cuda_devices": parse_nvidia_smi_csv(_cuda_csv(8)),
            },
        ]
        for case in cases:
            with self.subTest(case=case):
                with self.assertRaisesRegex(RuntimeError, "identity fallback refused"):
                    resolve_cuda_to_vulkan(
                        case["cuda_devices"],
                        [],
                        vk_icd_filenames=case["vk_icd_filenames"],
                        cuda_visible_devices=case["cuda_visible_devices"],
                        allow_identity_fallback=case["allow_identity_fallback"],
                    )


if __name__ == "__main__":
    unittest.main()
