import hindi_slm


def test_package_metadata():
    assert hindi_slm.__version__
    assert hindi_slm.BASE_MODEL.startswith("Qwen/")


def test_qwen3_5_text_model_is_available():
    # The base model is loaded text-only through this class; fail early if transformers lacks it.
    from transformers import Qwen3_5ForCausalLM  # noqa: F401
