import importlib
import unittest

from examples.flash_attention import b200, fa4_forward
from simplesim import CycleSimulator


class FlashAttentionRegressionTests(unittest.TestCase):
    def test_table1_forward_regressions(self) -> None:
        sim = CycleSimulator(b200)
        cases = [
            ((128, 128, 128), {"tensor_core": 1024, "shared_memory": 768, "sfu": 1024}),
            ((256, 128, 128), {"tensor_core": 2048, "shared_memory": 1536, "sfu": 2048}),
        ]

        for (m, n, d), expected in cases:
            with self.subTest(m=m, n=n, d=d):
                result = sim.simulate_tiled(fa4_forward(m, n, d, dtype_bytes=2))
                self.assertEqual(result.compute_results["tensor_core"].cycles, expected["tensor_core"])
                self.assertEqual(result.memory_results["shared_memory"].cycles, expected["shared_memory"])
                self.assertEqual(result.compute_results["sfu"].cycles, expected["sfu"])
                self.assertEqual(
                    sorted(result.bottleneck_units),
                    sorted(["tensor_core", "sfu"]),
                )


class ExampleImportTests(unittest.TestCase):
    def test_example_modules_import_without_running(self) -> None:
        modules = [
            "examples.flash_attention",
            "examples.fa4_pipeline",
            "examples.fa4_tiled_pipeline",
            "examples.fa4_fine_grained_pipeline",
            "examples.fa4_resource_scheduled",
            "examples.fa4_forward_breakdown",
            "examples.mla_decode",
        ]

        for module_name in modules:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertIsNotNone(module)


if __name__ == "__main__":
    unittest.main()
