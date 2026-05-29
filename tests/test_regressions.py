import importlib
import sys
import types
import unittest
from argparse import Namespace


class Saveable:
    def __init__(self):
        self.saved_paths = []

    def save_pretrained(self, path):
        self.saved_paths.append(path)


class MergeLoraWeightsTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {}

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        sys.modules.pop("merge_lora_weights", None)

    def _remember_module(self, name):
        if name not in self._saved_modules:
            self._saved_modules[name] = sys.modules.get(name)

    def _install_merge_lora_stubs(self, loader_cls):
        for name in ("torch", "transformers", "peft", "loaders", "supported_models"):
            self._remember_module(name)

        torch = types.ModuleType("torch")
        torch.bfloat16 = object()
        sys.modules["torch"] = torch

        transformers = types.ModuleType("transformers")
        sys.modules["transformers"] = transformers

        peft = types.ModuleType("peft")
        peft.PeftModel = types.SimpleNamespace(from_pretrained=lambda model, path: model)
        sys.modules["peft"] = peft

        loaders = types.ModuleType("loaders")
        loaders.LOADERS = {"demo-family": loader_cls}
        sys.modules["loaders"] = loaders

        supported_models = types.ModuleType("supported_models")
        supported_models.MODEL_HF_PATH = {"demo-model": "hf/demo-model"}
        supported_models.MODEL_FAMILIES = {"demo-model": "demo-family"}
        sys.modules["supported_models"] = supported_models

    def test_model_local_path_is_passed_to_loader(self):
        class RecordingLoader:
            calls = []

            def __init__(self, **kwargs):
                self.calls.append(kwargs)

            def load(self, load_model):
                return None, Saveable(), Saveable(), object()

        self._install_merge_lora_stubs(RecordingLoader)
        merge_lora_weights = importlib.import_module("merge_lora_weights")

        merge_lora_weights.merge_lora(
            Namespace(
                model_id="demo-model",
                model_local_path="/local/base-model",
                model_path="/adapter",
                model_save_path="/merged",
                load_model=False,
                load_4bit=False,
            )
        )

        self.assertEqual(RecordingLoader.calls[0]["model_local_path"], "/local/base-model")

    def test_merge_handles_model_families_without_processors(self):
        tokenizer = Saveable()

        class NoProcessorLoader:
            def __init__(self, **kwargs):
                pass

            def load(self, load_model):
                return None, tokenizer, None, object()

        self._install_merge_lora_stubs(NoProcessorLoader)
        merge_lora_weights = importlib.import_module("merge_lora_weights")

        merge_lora_weights.merge_lora(
            Namespace(
                model_id="demo-model",
                model_local_path="",
                model_path="/adapter",
                model_save_path="/merged",
                load_model=False,
                load_4bit=False,
            )
        )

        self.assertEqual(tokenizer.saved_paths, ["/merged"])


class Qwen2LoaderTests(unittest.TestCase):
    def setUp(self):
        self._saved_modules = {}

    def tearDown(self):
        for name, module in self._saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        for name in list(sys.modules):
            if name == "loaders" or name.startswith("loaders."):
                sys.modules.pop(name, None)

    def _remember_module(self, name):
        if name not in self._saved_modules:
            self._saved_modules[name] = sys.modules.get(name)

    def _install_transformers_stubs(self):
        for name in ("torch", "transformers"):
            self._remember_module(name)

        torch = types.ModuleType("torch")
        torch.dtype = object
        sys.modules["torch"] = torch

        transformers = types.ModuleType("transformers")

        class RecordingQwen2Model:
            calls = []

            @classmethod
            def from_pretrained(cls, path, **kwargs):
                cls.calls.append((path, kwargs))
                return types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=4096))

        class GenericModel:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return types.SimpleNamespace(config=types.SimpleNamespace(hidden_size=4096))

        class GenericProcessor:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return types.SimpleNamespace(tokenizer=object())

        class GenericTokenizer:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return object()

        class GenericConfig:
            @classmethod
            def from_pretrained(cls, path, **kwargs):
                return object()

        transformers.AutoProcessor = GenericProcessor
        transformers.AutoTokenizer = GenericTokenizer
        transformers.AutoConfig = GenericConfig
        transformers.BitsAndBytesConfig = object
        transformers.PreTrainedModel = object
        transformers.PreTrainedTokenizer = object
        transformers.AutoModelForCausalLM = GenericModel
        transformers.LlavaForConditionalGeneration = GenericModel
        transformers.LlavaNextForConditionalGeneration = GenericModel
        transformers.LlavaNextVideoForConditionalGeneration = GenericModel
        transformers.LlavaNextVideoProcessor = GenericProcessor
        transformers.LlavaOnevisionForConditionalGeneration = GenericModel
        transformers.MllamaForConditionalGeneration = GenericModel
        transformers.Qwen2VLForConditionalGeneration = RecordingQwen2Model
        sys.modules["transformers"] = transformers
        return RecordingQwen2Model

    def test_qwen2_loader_uses_local_path_for_model_weights(self):
        recording_model = self._install_transformers_stubs()

        from loaders.qwen2_vl import Qwen2VLModelLoader

        loader = Qwen2VLModelLoader(
            model_hf_path="hf/qwen2-vl",
            model_local_path="/local/qwen2-checkpoint",
            compute_dtype=object(),
        )
        loader.load(load_model=True)

        self.assertEqual(recording_model.calls[0][0], "/local/qwen2-checkpoint")


if __name__ == "__main__":
    unittest.main()
