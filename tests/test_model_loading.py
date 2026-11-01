from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TORCH_AVAILABLE = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("torchvision") is not None
)


@unittest.skipUnless(TORCH_AVAILABLE, "torch and torchvision are not installed")
class ModelLoadingTests(unittest.TestCase):
    def test_all_supported_architectures_have_five_outputs(self) -> None:
        from constants import LABELS, SUPPORTED_MODEL_NAMES
        from models import create_model

        for model_name in SUPPORTED_MODEL_NAMES:
            with self.subTest(model_name=model_name):
                model = create_model(model_name, len(LABELS))
                if model_name == "resnet18":
                    output_count = model.fc.out_features
                else:
                    output_count = model.classifier[-1].out_features
                self.assertEqual(output_count, len(LABELS))

    def test_efficientnet_checkpoint_loads_strictly(self) -> None:
        import torch

        from constants import LABELS
        from models import DeviceInfo, load_model

        path = PROJECT_ROOT / "model" / "efficientnet_b0.pt"
        device_info = DeviceInfo(torch.device("cpu"), "cpu", "CPU", None)
        model = load_model(path, "efficientnet_b0", len(LABELS), device_info)
        self.assertEqual(model.classifier[-1].out_features, len(LABELS))


if __name__ == "__main__":
    unittest.main()
