import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class FakeTensor:
    def __init__(self, shape, name="tensor"):
        self.shape = tuple(shape)
        self.name = name
        self.dtype = "long"
        self.device = "cpu"

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, item):
        if isinstance(item, slice):
            return FakeTensor((len(range(*item.indices(self.shape[0]))),), self.name)
        if isinstance(item, int) and len(self.shape) > 1:
            return [0] * self.shape[1]
        return 0

    def squeeze(self, dim=0):
        shape = list(self.shape)
        if dim < len(shape) and shape[dim] == 1:
            shape.pop(dim)
        return FakeTensor(shape or (1,), self.name)

    def unsqueeze(self, dim):
        shape = list(self.shape)
        shape.insert(dim, 1)
        return FakeTensor(shape, self.name)

    def to(self, dtype):
        return self

    def ne(self, value):
        return FakeTensor(self.shape, "attention_mask")

    def clone(self):
        return FakeTensor(self.shape, self.name)


class FakeTorch(types.ModuleType):
    Tensor = FakeTensor
    long = "long"

    def full_like(self, tensor, fill_value):
        return FakeTensor(tensor.shape, "full_like")

    def tensor(self, values):
        return FakeTensor((len(values),), "tensor")

    def full(self, shape, fill_value, dtype=None, device=None):
        return FakeTensor(shape, "full")

    def cat(self, tensors, dim=0):
        tensors = list(tensors)
        if dim == 0:
            if len(tensors[0].shape) == 1:
                return FakeTensor((sum(t.shape[0] for t in tensors),), "cat")
            return FakeTensor((sum(t.shape[0] for t in tensors), *tensors[0].shape[1:]), "cat")
        if dim == 1:
            return FakeTensor((tensors[0].shape[0], sum(t.shape[1] for t in tensors)), "cat")
        raise NotImplementedError(dim)


class Qwen2VLCollatorTest(unittest.TestCase):
    def load_module_with_stubs(self):
        fake_pil = types.ModuleType("PIL")
        fake_pil.Image = types.SimpleNamespace(Image=object)

        fake_numpy = types.ModuleType("numpy")
        fake_numpy.ndarray = object

        fake_torch = FakeTorch("torch")
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
            @property
            def IGNORE_TOKEN_ID(self):
                return -100

            @property
            def PAD_TOKEN_ID(self):
                return self.tokenizer.pad_token_id

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

    def test_replaces_image_and_video_tokens_when_video_prompt_has_both(self):
        module = self.load_module_with_stubs()

        result = module.replace_image_tokens("<image>\n<video>Describe both.", is_video=True)

        self.assertIn("<|image_pad|>", result)
        self.assertIn("<|video_pad|>", result)

    def test_collates_mixed_image_and_video_batch(self):
        module = self.load_module_with_stubs()
        processor = FakeProcessor()
        collator = module.Qwen2VLDataCollator()
        collator.processor = processor
        collator.tokenizer = types.SimpleNamespace(model_max_length=20, pad_token_id=0)
        collator.mask_question_tokens = True

        batch = collator(
            [
                {
                    "images": [object()],
                    "videos": [],
                    "system_prompt": None,
                    "conversations": ["<image>What is shown?", "An image."],
                },
                {
                    "images": [],
                    "videos": [object()],
                    "system_prompt": None,
                    "conversations": ["<video>What is happening?", "A video."],
                },
            ]
        )

        self.assertIn("pixel_values", batch)
        self.assertIn("image_grid_thw", batch)
        self.assertIn("pixel_values_videos", batch)
        self.assertIn("video_grid_thw", batch)
        self.assertEqual([True, False], [call["images"] is not None for call in processor.calls])
        self.assertEqual([False, True], [call["videos"] is not None for call in processor.calls])


class FakeProcessor:
    def __init__(self):
        self.calls = []
        self.tokenizer = self.tokenize

    def tokenize(self, text, add_special_tokens=False, padding=False, return_tensors=None):
        return {"input_ids": FakeTensor((1, 2), "input_ids")}

    def __call__(self, text, images=None, videos=None, padding=False, return_tensors=None):
        self.calls.append({"images": images, "videos": videos})
        result = {"input_ids": FakeTensor((1, 2), "prompt_input_ids")}
        if images is not None:
            result["pixel_values"] = FakeTensor((len(images), 3), "pixel_values")
            result["image_grid_thw"] = FakeTensor((len(images), 3), "image_grid_thw")
        if videos is not None:
            result["pixel_values_videos"] = FakeTensor((len(videos), 3), "pixel_values_videos")
            result["video_grid_thw"] = FakeTensor((len(videos), 3), "video_grid_thw")
        return result


if __name__ == "__main__":
    unittest.main()
