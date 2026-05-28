import importlib
import sys
import types
import unittest
from argparse import Namespace
from unittest.mock import patch


class MergeLoraWeightsTest(unittest.TestCase):
    def import_with_stubs(self, loader_cls):
        fake_loaders = types.ModuleType("loaders")
        fake_loaders.LOADERS = {"test-family": loader_cls}

        fake_supported_models = types.ModuleType("supported_models")
        fake_supported_models.MODEL_HF_PATH = {"test-model": "hf/model"}
        fake_supported_models.MODEL_FAMILIES = {"test-model": "test-family"}

        fake_torch = types.ModuleType("torch")
        fake_torch.bfloat16 = "bfloat16"

        fake_transformers = types.ModuleType("transformers")

        fake_peft = types.ModuleType("peft")
        fake_peft.PeftModel = object

        stubs = {
            "loaders": fake_loaders,
            "supported_models": fake_supported_models,
            "torch": fake_torch,
            "transformers": fake_transformers,
            "peft": fake_peft,
        }

        sys.modules.pop("merge_lora_weights", None)
        with patch.dict(sys.modules, stubs):
            return importlib.import_module("merge_lora_weights")

    def test_uses_model_local_path_when_provided(self):
        captured = {}

        class FakeLoader:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def load(self, load_model):
                return None, FakeSaveable(), FakeSaveable(), object()

        module = self.import_with_stubs(FakeLoader)
        module.merge_lora(
            Namespace(
                model_id="test-model",
                model_local_path="/models/local",
                model_path="/adapters/lora",
                model_save_path="",
                load_model=False,
                load_4bit=False,
            )
        )

        self.assertEqual("/models/local", captured["model_local_path"])

    def test_falls_back_to_hf_path_without_local_path(self):
        captured = {}

        class FakeLoader:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def load(self, load_model):
                return None, FakeSaveable(), FakeSaveable(), object()

        module = self.import_with_stubs(FakeLoader)
        module.merge_lora(
            Namespace(
                model_id="test-model",
                model_local_path="",
                model_path="/adapters/lora",
                model_save_path="",
                load_model=False,
                load_4bit=False,
            )
        )

        self.assertEqual("hf/model", captured["model_local_path"])


class FakeSaveable:
    def save_pretrained(self, path):
        self.path = path


if __name__ == "__main__":
    unittest.main()
