#!/usr/bin/env python3
"""划分 SAM2 伪标签数据并训练、评估、导出官方 PP-LiteSeg-T/STDC1。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from types import SimpleNamespace
import sys
from typing import Any, Mapping, Sequence

import yaml

from common import (
    DEFAULT_SETTINGS_PATH,
    SettingsError,
    atomic_write_text,
    load_settings,
    positive_number,
    read_manifest,
    require_mapping,
    resolve_config_path,
    split_samples_by_video,
    validate_dataset_samples,
    write_paddleseg_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="划分二分类数据并训练 PP-LiteSeg-T1（STDC1）")
    parser.add_argument("--settings", default=DEFAULT_SETTINGS_PATH)
    parser.add_argument("--dataset-dir", help="覆盖 labeling.dataset_dir")
    parser.add_argument("--output-dir", help="覆盖 training.output_dir")
    parser.add_argument("--val-ratio", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--save-interval", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--device", choices=("cpu", "gpu"), help="覆盖 training.device")
    parser.add_argument("--resume-model", help="PaddleSeg checkpoint 目录")
    parser.add_argument("--use-vdl", action="store_true")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="只校验、划分并生成 PaddleSeg 配置，不导入 Paddle",
    )
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def configured_value(
    cli_value: Any,
    section: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    return cli_value if cli_value is not None else section.get(key, default)


def positive_integer(value: Any, name: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "非负" if allow_zero else "正"
        raise SettingsError(f"{name} 必须是{qualifier}整数")
    return value


def build_paddleseg_config(
    *,
    dataset_dir: Path,
    height: int,
    width: int,
    iterations: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, Any]:
    """生成与 PaddleSeg release/2.10 官方 T1 配置同构的二分类配置。"""
    minimum_kept = max(1000, batch_size * height * width // 16)
    def normalize() -> dict[str, Any]:
        return {
            "type": "Normalize",
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        }
    ohem = {"type": "OhemCrossEntropyLoss", "min_kept": minimum_kept}
    return {
        "batch_size": batch_size,
        "iters": iterations,
        "train_dataset": {
            "type": "Dataset",
            "dataset_root": str(dataset_dir),
            "train_path": str(dataset_dir / "train.txt"),
            "num_classes": 2,
            "mode": "train",
            "transforms": [
                {
                    "type": "RandomDistort",
                    "brightness_range": 0.25,
                    "contrast_range": 0.25,
                    "saturation_range": 0.25,
                },
                normalize(),
            ],
        },
        "val_dataset": {
            "type": "Dataset",
            "dataset_root": str(dataset_dir),
            "val_path": str(dataset_dir / "val.txt"),
            "num_classes": 2,
            "mode": "val",
            "transforms": [normalize()],
        },
        "optimizer": {
            "type": "SGD",
            "momentum": 0.9,
            "weight_decay": 5.0e-4,
        },
        "lr_scheduler": {
            "type": "PolynomialDecay",
            "learning_rate": learning_rate,
            "end_lr": 0.0,
            "power": 0.9,
            "warmup_iters": min(500, max(1, iterations // 20)),
            "warmup_start_lr": min(1.0e-5, learning_rate),
        },
        "loss": {"types": [dict(ohem), dict(ohem), dict(ohem)], "coef": [1, 1, 1]},
        "model": {
            "type": "PPLiteSeg",
            "num_classes": 2,
            "backbone": {
                "type": "STDC1",
                "pretrained": "https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet1.tar.gz",
            },
            "arm_out_chs": [32, 64, 128],
            "seg_head_inter_chs": [32, 64, 64],
        },
        "test_config": {"aug_eval": False},
    }


def dump_yaml(value: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(value),
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )


def checkpoint_iteration(path: Path) -> int:
    match = re.fullmatch(r"iter_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def find_export_weights(output_dir: Path) -> Path:
    best = output_dir / "best_model" / "model.pdparams"
    if best.is_file():
        return best
    candidates = sorted(
        (
            path / "model.pdparams"
            for path in output_dir.glob("iter_*")
            if checkpoint_iteration(path) >= 0
        ),
        key=lambda path: checkpoint_iteration(path.parent),
    )
    if not candidates:
        raise RuntimeError(f"训练结束后未找到 model.pdparams: {output_dir}")
    return candidates[-1]


def repair_deploy_model_name(export_dir: Path) -> None:
    """兼容 Paddle 3.x 将静态图拓扑保存为 model.json。"""
    deploy_path = export_dir / "deploy.yaml"
    if not deploy_path.is_file():
        raise RuntimeError(f"导出缺少 deploy.yaml: {deploy_path}")
    with deploy_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("Deploy"), Mapping):
        raise RuntimeError(f"deploy.yaml 缺少 Deploy 字典: {deploy_path}")
    deploy = dict(payload["Deploy"])
    candidates = [export_dir / "model.json", export_dir / "model.pdmodel"]
    model_file = next((path for path in candidates if path.is_file()), None)
    if model_file is None:
        raise RuntimeError(f"导出目录缺少 model.json/model.pdmodel: {export_dir}")
    deploy["model"] = model_file.name
    updated = dict(payload)
    updated["Deploy"] = deploy
    atomic_write_text(deploy_path, dump_yaml(updated))


def run_training(
    *,
    config_path: Path,
    output_dir: Path,
    device: str,
    seed: int,
    iterations: int,
    batch_size: int,
    save_interval: int,
    num_workers: int,
    precision: str,
    resume_model: str | None,
    use_vdl: bool,
) -> None:
    try:
        import paddle
        from paddleseg.core import train as paddleseg_train
        from paddleseg.cvlibs import Config, SegBuilder
        from paddleseg.utils import utils as paddleseg_utils
    except ImportError as error:
        raise RuntimeError(
            "训练需要独立安装 PaddlePaddle GPU 和 paddleseg==2.10.0；"
            "详见 PP-LiteSeg/README.md") from error

    config = Config(str(config_path))
    builder = SegBuilder(config)
    paddleseg_utils.show_env_info()
    paddleseg_utils.show_cfg_info(config)
    paddleseg_utils.set_seed(seed)
    paddleseg_utils.set_device(device)
    paddleseg_utils.set_cv2_num_threads(num_workers)
    model = paddleseg_utils.convert_sync_batchnorm(builder.model, device)
    print(f"Paddle {paddle.__version__} 开始训练，输出目录: {output_dir}")
    # PaddleSeg 2.10 在训练结束后仅为打印信息调用 paddle.flops；该旧调用在
    # Paddle 3.3 + NumPy 2.x 会因参数 Tensor 转 int 失败。FLOPs 不参与训练，
    # 包装器在训练期间临时跳过它，避免模型已经保存却以异常退出。
    original_flops = getattr(paddle, "flops", None)
    if original_flops is not None:
        paddle.flops = lambda *unused_args, **unused_kwargs: None
    try:
        paddleseg_train(
            model,
            builder.train_dataset,
            val_dataset=builder.val_dataset,
            optimizer=builder.optimizer,
            save_dir=str(output_dir),
            iters=iterations,
            batch_size=batch_size,
            resume_model=resume_model,
            save_interval=save_interval,
            log_iters=10,
            num_workers=num_workers,
            use_vdl=use_vdl,
            losses=builder.loss,
            keep_checkpoint_max=5,
            test_config=config.test_config,
            precision=precision,
            amp_level="O1",
            to_static_training=False,
            print_mem_info=False,
            shuffle=True,
        )
    finally:
        if original_flops is not None:
            paddle.flops = original_flops


def export_inference_model(
    *,
    config_path: Path,
    weights_path: Path,
    export_dir: Path,
    height: int,
    width: int,
) -> None:
    try:
        from paddleseg.core.export import export
    except ImportError as error:
        raise RuntimeError("导出需要安装 paddleseg==2.10.0") from error
    args = SimpleNamespace(
        config=str(config_path),
        model_path=str(weights_path),
        save_dir=str(export_dir),
        input_shape=[1, 3, height, width],
        output_op="argmax",
        for_fd=False,
    )
    export(args)
    repair_deploy_model_name(export_dir)
    metadata = {
        "architecture": "PP-LiteSeg-T1",
        "backbone": "STDC1",
        "class_names": ["background", "surface"],
        "input_shape": [1, 3, height, width],
        "foreground_class": 1,
        "weights": str(weights_path),
    }
    atomic_write_text(
        export_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    print(f"固定形状推理模型已导出: {export_dir}")


def main() -> None:
    args = parse_args()
    try:
        settings_path, settings = load_settings(args.settings)
        base = settings_path.parent
        labeling = require_mapping(settings.get("labeling"), "labeling")
        training = require_mapping(settings.get("training"), "training")
        dataset_dir = resolve_config_path(
            args.dataset_dir or labeling.get("dataset_dir"),
            base=Path.cwd() if args.dataset_dir else base,
            name="dataset_dir",
        )
        output_dir = resolve_config_path(
            args.output_dir or training.get("output_dir"),
            base=Path.cwd() if args.output_dir else base,
            name="output_dir",
        )
        val_ratio = positive_number(
            configured_value(args.val_ratio, training, "val_ratio", 0.2),
            "training.val_ratio",
        )
        if not 0 < val_ratio < 1:
            raise SettingsError("training.val_ratio 必须在 0..1 之间")
        seed = positive_integer(
            configured_value(args.seed, training, "seed", 42), "training.seed", allow_zero=True)
        iterations = positive_integer(
            configured_value(args.iterations, training, "iterations", 10000),
            "training.iterations",
        )
        batch_size = positive_integer(
            configured_value(args.batch_size, training, "batch_size", 4),
            "training.batch_size",
        )
        learning_rate = positive_number(
            configured_value(args.learning_rate, training, "learning_rate", 0.005),
            "training.learning_rate",
        )
        save_interval = positive_integer(
            configured_value(args.save_interval, training, "save_interval", 500),
            "training.save_interval",
        )
        num_workers = positive_integer(
            configured_value(args.num_workers, training, "num_workers", 2),
            "training.num_workers",
            allow_zero=True,
        )
        precision = str(configured_value(args.precision, training, "precision", "fp32"))
        if precision not in {"fp32", "fp16"}:
            raise SettingsError("training.precision 必须是 fp32 或 fp16")
        device = configured_value(args.device, training, "device", "gpu")
        if device not in {"cpu", "gpu"}:
            raise SettingsError("training.device 必须是 cpu 或 gpu")

        samples = read_manifest(dataset_dir)
        if not samples:
            raise SettingsError(
                f"{dataset_dir}/manifest.jsonl 不存在或为空，请先运行 auto_label.py")
        height, width = validate_dataset_samples(dataset_dir, samples)
        train_samples, val_samples, split_mode = split_samples_by_video(
            samples, val_ratio=val_ratio, seed=seed)
        if len(train_samples) < batch_size:
            raise SettingsError(
                f"训练样本 {len(train_samples)} 少于 batch_size={batch_size}")
        write_paddleseg_split(
            dataset_dir,
            train_samples,
            val_samples,
            mode=split_mode,
            seed=seed,
        )
        if split_mode == "temporal_tail":
            print(
                "警告：数据集只有一个 source_video，验证集只能取视频末段；"
                "该指标不能代表跨会话泛化能力。",
                file=sys.stderr,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        config = build_paddleseg_config(
            dataset_dir=dataset_dir,
            height=height,
            width=width,
            iterations=iterations,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        config_path = output_dir / "paddleseg_pp_liteseg_t1.yml"
        atomic_write_text(config_path, dump_yaml(config))
        print(
            f"数据集: train={len(train_samples)}, val={len(val_samples)}, "
            f"split={split_mode}, shape={width}x{height}")
        print(f"PaddleSeg 配置: {config_path}")
        if args.prepare_only:
            return

        existing_weights = list(output_dir.glob("**/model.pdparams"))
        if existing_weights and not args.resume_model:
            raise SettingsError(
                f"{output_dir} 已含训练权重；请换 --output-dir 或使用 --resume-model")
        run_training(
            config_path=config_path,
            output_dir=output_dir,
            device=str(device),
            seed=seed,
            iterations=iterations,
            batch_size=batch_size,
            save_interval=min(save_interval, iterations),
            num_workers=num_workers,
            precision=precision,
            resume_model=args.resume_model,
            use_vdl=args.use_vdl,
        )
        if not args.skip_export:
            weights = find_export_weights(output_dir)
            export_inference_model(
                config_path=config_path,
                weights_path=weights,
                export_dir=output_dir / "inference_model",
                height=height,
                width=width,
            )
    except (SettingsError, RuntimeError, ValueError) as error:
        print(f"错误: {error}", file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
