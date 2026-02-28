#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import magic  # python-magic


WORKSPACE_AI = Path(os.environ.get("WORKSPACE_AI", "/workspace-ai"))
LOOK_IMGS_JSON = WORKSPACE_AI / "look_imgs.json"


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def clear_look_imgs_json() -> None:
    # 幂等：失败时不留下任何内容
    WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
    atomic_write_text(LOOK_IMGS_JSON, "")


def detect_mime(path: Path) -> str:
    # mime=True 会返回类似 image/png, text/plain
    return magic.from_file(str(path), mime=True) or ""


def is_image_mime(mime: str) -> bool:
    return mime.startswith("image/")


def build_payload(paths: List[Path], max_bytes: int, max_size_mb: int) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []

    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"文件不存在：{p}")
        if not p.is_file():
            raise ValueError(f"不是普通文件：{p}")

        size = p.stat().st_size
        if size > max_bytes:
            raise ValueError(
                f"图片过大：{p} size={size} bytes > --max-size={max_size_mb}MB（默认 --max-size=3MB）。\n"
                f"建议：先压缩或缩放图片后再调用 look_imgs，避免网络传输失败或超过 OpenAI API/模型供应商限制。"
            )

        mime = detect_mime(p)
        if not is_image_mime(mime):
            raise ValueError(f"不是图片：{p}\n检测到类型：{mime or '(unknown)'}")

        raw = p.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")

        payload.append(
            {
                "path": str(p),
                "mime": mime,
                "b64": b64,
                "bytes": size,
            }
        )

    return payload


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="look_imgs",
        description=(
            "读取多张图片并写入 /workspace-ai/look_imgs.json（base64）。\n"
            "注意：base64 很大很长，请不要直接 cat look_imgs.json。"
        ),
    )
    ap.add_argument(
        "images",
        nargs="+",
        help="图片路径（可相对/绝对路径）",
    )
    ap.add_argument(
        "--max-size",
        type=int,
        default=3,
        help="单张图片最大体积（MB，整数，默认 3）",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    # 失败时确保 json 为空
    clear_look_imgs_json()

    max_size_mb: int = int(args.max_size)
    if max_size_mb <= 0:
        print("--max-size 必须为正整数 MB", file=sys.stderr)
        return 2

    max_bytes = max_size_mb * 1024 * 1024

    paths = [Path(x).expanduser() for x in args.images]

    try:
        payload = build_payload(paths, max_bytes=max_bytes, max_size_mb=max_size_mb)
        WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
        atomic_write_text(LOOK_IMGS_JSON, json.dumps(payload, ensure_ascii=False))
        return 0
    except Exception as e:
        # 再清一次，确保失败无内容
        clear_look_imgs_json()
        print(str(e), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())