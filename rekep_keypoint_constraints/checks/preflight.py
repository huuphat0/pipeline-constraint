"""Kiểm tra model và dữ liệu cục bộ; không tải hoặc chạy model."""

import argparse
import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


HERE = Path(__file__).resolve().parents[1]


def check_files(config):
    failures = []
    for name, relative in config["inputs"].items():
        path = (HERE / relative).resolve()
        exists = path.is_file()
        print(f"{'OK' if exists else 'THIẾU'} input {name}: {path}")
        if not exists:
            failures.append(str(path))
    for relative in config["source_modules"]:
        path = (HERE / relative).resolve()
        exists = path.is_file()
        print(f"{'OK' if exists else 'THIẾU'} source: {path}")
        if not exists:
            failures.append(str(path))
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for model in config["hf_models"]:
        path = hub / ("models--" + model.replace("/", "--"))
        exists = path.is_dir() and any((path / "snapshots").glob("*"))
        print(f"{'OK' if exists else 'THIẾU'} HF cache {model}: {path}")
        if not exists:
            failures.append(model)
    return failures


def check_ollama(config):
    url = config["ollama_base_url"].rstrip("/") + "/api/tags"
    try:
        with urlopen(url, timeout=5) as response:
            available = {item["name"] for item in json.load(response)["models"]}
    except (HTTPError, URLError, TimeoutError, OSError, KeyError, ValueError) as exc:
        print(f"KHÔNG KẾT NỐI Ollama {url}: {exc}")
        return [url]
    missing = []
    for role, model in config["models"].items():
        exists = model in available
        print(f"{'OK' if exists else 'THIẾU'} Ollama {role}: {model}")
        if not exists:
            missing.append(model)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="bỏ qua kết nối tới Ollama")
    args = parser.parse_args()
    with (HERE / "config.json").open(encoding="utf-8") as file:
        config = json.load(file)
    failures = check_files(config)
    if not args.offline:
        failures.extend(check_ollama(config))
    print("\nSẵn sàng." if not failures else f"\nCần xử lý {len(failures)} mục trước khi chạy.")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
