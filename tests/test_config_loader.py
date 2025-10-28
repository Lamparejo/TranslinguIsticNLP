from pathlib import Path

from src.utils.config import load_config


def test_load_config(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text("key: value\n", encoding="utf-8")
    config = load_config(config_file)
    assert config["key"] == "value"
