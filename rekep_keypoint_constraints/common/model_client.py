"""Đầu nối tới các model Ollama đã cài, dùng lại client của pipeline hiện tại."""

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parents[1]
PIPELINE = HERE.parent / "preprocessing"


def call_role(role, prompt, *, images=None, schema=None, think=None, seed=0):
    """Gọi model cho một vai trò; trả về kết quả của llm_client.call().

    role: ``vision``, ``reasoning`` hoặc ``code``. Ảnh chỉ dùng với ``vision``.
    Hàm không thực thi mã do model tạo ra.
    """
    with (HERE / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    if role not in config["models"]:
        raise ValueError(f"Vai trò model không hợp lệ: {role!r}")
    if images and role != "vision":
        raise ValueError("Ảnh chỉ được gửi cho model vision")
    if str(PIPELINE) not in sys.path:
        sys.path.insert(0, str(PIPELINE))
    from constraint_inference import llm_client

    model = config["models"][role]
    llm_client.check_ready(model, base_url=config["ollama_base_url"])
    return llm_client.call(
        model,
        prompt,
        images=images,
        schema=schema,
        think=(role == "reasoning") if think is None else think,
        seed=seed,
        base_url=config["ollama_base_url"],
        cache_dir=str(HERE / "artifacts" / "cache"),
    )
