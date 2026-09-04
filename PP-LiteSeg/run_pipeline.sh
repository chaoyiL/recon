#!/usr/bin/env bash

set -Eeuo pipefail

if (( $# != 0 )); then
    echo "错误：run_pipeline.sh 不接受命令行参数；请修改仓库根目录 config.yaml 的 pp_liteseg 段。" >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "$SCRIPT_DIR/.." && pwd)"

if ! command -v uv >/dev/null 2>&1; then
    echo "错误：找不到 uv，请先安装 uv。" >&2
    exit 2
fi

cd -- "$REPO_DIR"
uv run python - <<'PY'
from importlib.util import find_spec

required = ("torch", "transformers", "paddle", "paddleseg")
missing = [name for name in required if find_spec(name) is None]
if missing:
    names = ", ".join(missing)
    raise SystemExit(
        f"错误：uv 环境缺少依赖：{names}。请按 PP-LiteSeg/README.md 安装。")
PY

started_at=$SECONDS
DATASET_INFO="$(uv run python - "$SCRIPT_DIR" <<'PY'
from pathlib import Path
import sys

tool_dir = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(tool_dir))

from common import (
    load_settings,
    read_manifest,
    require_mapping,
    resolve_config_path,
    validate_dataset_samples,
)

settings_path, settings = load_settings(tool_dir.parent / "config.yaml")
labeling = require_mapping(settings.get("labeling"), "labeling")
dataset_dir = resolve_config_path(
    labeling.get("dataset_dir"),
    base=settings_path.parent,
    name="labeling.dataset_dir",
)
samples = read_manifest(dataset_dir)
if samples:
    validate_dataset_samples(dataset_dir, samples)
print(f"{len(samples)}\t{dataset_dir}")
PY
)"
IFS=$'\t' read -r SAMPLE_COUNT DATASET_DIR <<< "$DATASET_INFO"

if (( SAMPLE_COUNT > 0 )); then
    echo "[1/2] 跳过 SAM2：$DATASET_DIR 已有 $SAMPLE_COUNT 条有效样本"
else
    echo "[1/2] 使用 SAM2 生成监督掩膜"
    PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/auto_label.py"
fi

echo "[2/2] 划分数据并训练、导出 PP-LiteSeg-T1"
PYTHONUNBUFFERED=1 uv run python "$SCRIPT_DIR/train.py"

elapsed=$((SECONDS - started_at))
echo "流水线完成，用时 ${elapsed}s"
