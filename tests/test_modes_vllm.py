from model_manager.modes import get_vllm_mode_config, get_vllm_runtime_mode


def test_ocr_defaults_to_deepseek_ocr2() -> None:
    config = get_vllm_mode_config("ocr", None)

    assert config.model == "deepseek-ai/DeepSeek-OCR-2"
    assert config.max_model_len == 8192
    assert "--trust-remote-code" in config.extra_args


def test_ocr_lighton_override_profile() -> None:
    config = get_vllm_mode_config("ocr", "lightonai/LightOnOCR-2-1B")

    assert config.model == "lightonai/LightOnOCR-2-1B"
    assert config.max_model_len == 8192
    assert "--mm-processor-cache-gb 0" in config.extra_args


def test_vllm_runtime_mode_selection() -> None:
    assert get_vllm_runtime_mode("ocr") == "max_performance"
    assert get_vllm_runtime_mode("perf") == "max_performance"
    assert get_vllm_runtime_mode("chat") == "multi_model"
    assert get_vllm_runtime_mode("embed") == "multi_model"
