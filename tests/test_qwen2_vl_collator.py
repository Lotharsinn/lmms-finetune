import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class Qwen2VLCollatorTest(unittest.TestCase):
    def load_module_with_stubs(self):
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(Image=object)

        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = object

        fake_torch = types.ModuleType("torch")
        fake_torch.Tensor = object
        fake_transformers = types.ModuleType("transformers")
        fake_transformers.PreTrainedTokenizer = object
        fake_transformers.AutoProcessor = object
        fake_transformers.AutoConfig = object

        fake_collators = types.ModuleType("collators")
        fake_collators.__path__ = []

        def register_collator(name):
            def decorator(cls):
                return cls

            return decorator

        fake_collators.register_collator = register_collator

        fake_base = types.ModuleType("collators.base")

        class BaseDataCollator:
            pass

        fake_base.BaseDataCollator = BaseDataCollator

        stubs = {
            "PIL": fake_pil,
            "numpy": fake_numpy,
            "torch": fake_torch,
            "transformers": fake_transformers,
            "collators": fake_collators,
            "collators.base": fake_base,
        }

        module_path = Path(__file__).resolve().parents[1] / "collators" / "qwen2_vl.py"
        spec = importlib.util.spec_from_file_location("collators.qwen2_vl", module_path)
        module = importlib.util.module_from_spec(spec)

        with patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)

        return module

    def test_detects_video_from_non_empty_video_list(self):
        module = self.load_module_with_stubs()

        self.assertTrue(
            module.is_video_batch(
                [
                    {
                        "images": [],
                        "videos": [object()],
                    }
                ]
            )
        )

    def test_text_only_batch_uses_image_path_for_existing_behavior(self):
        module = self.load_module_with_stubs()

        self.assertFalse(
            module.is_video_batch(
                [
                    {
                        "images": [],
                        "videos": [],
                    }
                ]
            )
        )

    def test_mixed_image_and_video_batch_fails_instead_of_ignoring_videos(self):
        module = self.load_module_with_stubs()

        with self.assertRaises(ValueError):
            module.is_video_batch(
                [
                    {
                        "images": [object()],
                        "videos": [],
                    },
                    {
                        "images": [],
                        "videos": [object()],
                    },
                ]
            )


if __name__ == "__main__":
    unittest.main()
