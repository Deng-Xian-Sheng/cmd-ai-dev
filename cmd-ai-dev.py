#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rich.markdown import Markdown

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, RichLog, Static, TextArea

from openai import OpenAI
import subprocess
import sys

WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
WORKSPACE_AI = Path(os.environ.get("WORKSPACE_AI", "/workspace-ai"))
SESSION_PATH = WORKSPACE_AI / "session.json"

DEFAULT_TIMEOUT_SEC = 60
CMDOUT_TRUNCATE_CHARS = 20000

VENV_DIR = Path(os.environ.get("VENV_DIR", "/opt/venv"))
VENV_BIN = VENV_DIR / "bin"

TRANSCRIPT_PATH = WORKSPACE_AI / "transcript.log"


def now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def load_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def append_transcript(text: str) -> None:
    WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
    with TRANSCRIPT_PATH.open("a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


@dataclass
class Session:
    messages: List[Dict[str, Any]]

    @staticmethod
    def default_system_prompt() -> str:
        return (
            "你是一个运行在 Docker 容器里的命令行 AI 编程助手（cmd-ai-dev）。\n"
            "\n"
            "环境与目录：\n"
            f"- 用户项目目录：{WORKSPACE}（映射到宿主机，修改会真实影响项目）\n"
            f"- AI 工作目录：{WORKSPACE_AI}（默认不映射，用于脚本/笔记/中间产物，避免污染仓库）\n"
            f"- 会话上下文文件：{SESSION_PATH}\n"
            "\n"
            "执行命令协议：\n"
            "- 需要工具执行命令时，在回复中输出一个 <cmd>...</cmd> 块。\n"
            "- <cmd> 内可选写 <time_out>秒</time_out>（整数秒），不写默认 60 秒。\n"
            "- <cmd> 内其余内容视为要执行的 bash 命令文本，可多行。\n"
            "- 不要把 <cmd> / <cmdout> 放进 Markdown 代码块（不要用 ``` 包裹）。\n"
            "- 每次回复最多包含一个 <cmd>...</cmd>；如需多步，请分多轮：先执行一段命令，收到 <cmdout> 后再输出下一段 <cmd>。\n"
            "\n"
            "工具回给你的格式：\n"
            "- 工具会把命令输出包装为：\n"
            "  <cmdout>\n"
            "  [exit=退出码 timeout=0/1 interrupted=0/1]\n"
            "  ...命令输出（可能截断）...\n"
            "  </cmdout>\n"
            "- 命令输出会同时落盘到 /workspace-ai 供用户复制查看。\n"
            "\n"
            "建议工作方式：\n"
            "- 每次尽量只请求执行一段命令，收到 <cmdout> 后再决定下一步。\n"
            "- 临时脚本/笔记优先写到 /workspace-ai。\n"
            "- 如用户触发 STOP，工具会停止命令链并给你 system_note；此时请等待用户新指令。\n"
            "\n"
            f"提示：容器 venv 的 bin 目录通常为 {VENV_BIN}，应优先使用该环境的 python/pip。\n"
        )

    @classmethod
    def load_or_create(cls) -> "Session":
        WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
        data = load_json(SESSION_PATH)
        if data and isinstance(data.get("messages"), list):
            return cls(messages=data["messages"])
        s = cls(messages=[{"role": "system", "content": cls.default_system_prompt()}])
        s.save()
        return s

    def save(self) -> None:
        payload = {"messages": self.messages, "saved_at": time.time()}
        atomic_write_text(SESSION_PATH, json.dumps(payload, ensure_ascii=False, indent=2))

    def add(self, role: str, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        self.save()


CMD_BLOCK_RE = re.compile(r"<cmd>\s*(.*?)\s*</cmd>", re.DOTALL | re.IGNORECASE)
CMDOUT_BLOCK_RE = re.compile(r"<cmdout>\s*(.*?)\s*</cmdout>", re.DOTALL | re.IGNORECASE)
TIMEOUT_RE = re.compile(r"<time_out>\s*(\d+)\s*</time_out>", re.IGNORECASE | re.DOTALL)


@dataclass
class ParsedAssistant:
    raw: str
    cmd: Optional[str]
    timeout_sec: int
    answer_without_cmd: str


def parse_assistant(text: str) -> ParsedAssistant:
    cmd = None
    timeout = DEFAULT_TIMEOUT_SEC
    m = CMD_BLOCK_RE.search(text)
    answer_wo = text
    if m:
        inner = m.group(1)

        tm = TIMEOUT_RE.search(inner)
        if tm:
            try:
                timeout = int(tm.group(1))
            except Exception:
                timeout = DEFAULT_TIMEOUT_SEC
            inner = TIMEOUT_RE.sub("", inner)

        cmd = inner.strip()
        answer_wo = (text[: m.start()] + text[m.end() :]).strip()

    return ParsedAssistant(raw=text, cmd=cmd, timeout_sec=timeout, answer_without_cmd=answer_wo)


def format_cmdout(exit_code: Optional[int], timed_out: bool, interrupted: bool, output: str, extra_note: Optional[str] = None) -> str:
    header = f"[exit={exit_code} timeout={int(timed_out)} interrupted={int(interrupted)}]"
    body = output
    if extra_note:
        body = (body + "\n\n" + extra_note).strip()
    return f"<cmdout>\n{header}\n{body}\n</cmdout>"


@dataclass
class CmdResult:
    exit_code: Optional[int]
    timed_out: bool
    interrupted: bool
    output: str
    log_path: Path
    truncated: bool


class CommandRunner:
    def __init__(self) -> None:
        self._proc: Optional[asyncio.subprocess.Process] = None

    def _build_env(self) -> Dict[str, str]:
        env = dict(os.environ)

        # 强制把 venv/bin 前置到 PATH（防止登录 shell / profile 重置）
        p = env.get("PATH", "")
        venv_bin = str(VENV_BIN)
        if not p.startswith(venv_bin):
            env["PATH"] = f"{venv_bin}:{p}"

        env["VIRTUAL_ENV"] = str(VENV_DIR)
        env.setdefault("PYTHONUNBUFFERED", "1")
        return env

    async def run(self, cmd: str, timeout_sec: int, cwd: Path) -> CmdResult:
        WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
        log_path = WORKSPACE_AI / f"cmdout_{now_ts()}.log"

        # 关键修复：输出落盘，避免 nohup/& 后台继承 PIPE 导致 communicate() 卡死
        with log_path.open("wb") as logf:
            self._proc = await asyncio.create_subprocess_exec(
                "bash",
                "-c",
                cmd,
                stdout=logf,
                stderr=logf,
                cwd=str(cwd),
                env=self._build_env(),
                start_new_session=True,
            )

            timed_out = False
            interrupted = False

            try:
                await asyncio.wait_for(self._proc.wait(), timeout=timeout_sec)
            except asyncio.TimeoutError:
                timed_out = True
                await self._terminate_group(sig=signal.SIGKILL)
                await self._proc.wait()
            except asyncio.CancelledError:
                interrupted = True
                await self._terminate_group(sig=signal.SIGTERM)
                await self._proc.wait()
            finally:
                rc = self._proc.returncode if self._proc else None
                self._proc = None

        # 读出当前日志内容（后台进程可能还会继续写，但不影响本次返回）
        raw = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
        truncated = False
        output = raw
        if len(output) > CMDOUT_TRUNCATE_CHARS:
            output = output[:CMDOUT_TRUNCATE_CHARS] + "\n...<truncated>..."
            truncated = True

        return CmdResult(
            exit_code=rc,
            timed_out=timed_out,
            interrupted=interrupted,
            output=output,
            log_path=log_path,
            truncated=truncated,
        )

    async def interrupt(self) -> None:
        if self._proc is None:
            return
        await self._terminate_group(sig=signal.SIGTERM)

    async def _terminate_group(self, sig: int) -> None:
        if self._proc is None or self._proc.pid is None:
            return
        try:
            os.killpg(self._proc.pid, sig)
        except ProcessLookupError:
            pass


class LLMClient:
    async def stream_chat(self, messages: List[Dict[str, Any]]) -> str:
        raise NotImplementedError


class OpenAISDKClient(LLMClient):
    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise RuntimeError("缺少环境变量 OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def stream_chat(self, messages: List[Dict[str, Any]]) -> str:
        def _call() -> str:
            acc: List[str] = []
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=float(os.environ.get("OPENAI_TEMPERATURE", "0.2")),
                stream=True,
            )
            for chunk in stream:
                try:
                    delta = chunk.choices[0].delta.content
                except Exception:
                    delta = None
                if delta:
                    acc.append(delta)
            return "".join(acc)

        return await asyncio.to_thread(_call)


class VDivider(Static):
    def on_mount(self) -> None:
        self.refresh_divider()

    def on_resize(self) -> None:
        self.refresh_divider()

    def refresh_divider(self) -> None:
        h = max(1, self.size.height)
        self.update(("│\n" * (h - 1)) + "│")


class CmdAIDevApp(App):
    # 尽量让终端可选中/可复制：禁用 alternate screen（终端兼容性仍各不相同）
    USE_ALTERNATE_SCREEN = False

    BINDINGS = [
        ("ctrl+s", "send", "发送"),
        ("f2", "send", "发送(F2)"),
        ("ctrl+a", "select_all", "全选"),
        ("ctrl+t", "stop", "停止"),
        ("ctrl+r", "reset", "重置"),
        ("ctrl+q", "quit", "退出"),
        ("f3", "stop", "停止(F3)"),
        ("f4", "reset", "重置(F4)"),
    ]

    CSS = """
    #left { width: 1fr; padding: 0 1; }
    #divider { width: 1; color: $text-muted; }
    #right_panel { width: 1fr; border-left: solid $panel; }
    #buttons { height: 3; padding: 0 1; }
    Button { height: 3; padding: 0 1; content-align: center middle; }
    #right { height: 1fr; padding: 0 1; }
    """

    def __init__(self) -> None:
        super().__init__()
        self.session = Session.load_or_create()
        self.llm: LLMClient = OpenAISDKClient()
        self.runner = CommandRunner()

        self.busy: bool = False
        self.stop_requested: bool = False

        self.pending_user_buffer: List[str] = []
        self.queue: asyncio.Queue[Tuple[str, str]] = asyncio.Queue()

    def compose(self) -> ComposeResult:
        with Horizontal():
            yield RichLog(id="left", wrap=True, highlight=True, markup=True)
            yield VDivider("", id="divider")
            with Vertical(id="right_panel"):
                with Horizontal(id="buttons"):
                    yield Button("发送", id="btn_send", variant="success")
                    yield Button("停止", id="btn_stop", variant="warning")
                    yield Button("重置", id="btn_reset")
                    yield Button("退出", id="btn_quit", variant="error")
                yield TextArea(id="right")
        yield Footer()

    def _left(self) -> RichLog:
        return self.query_one("#left", RichLog)

    def write_left_plain(self, text: str) -> None:
        self._left().write(text)
        append_transcript(str(text))

    def write_left_markdown(self, md_text: str) -> None:
        # Rich Markdown 渲染；transcript 仍写纯文本
        self._left().write(Markdown(md_text))
        append_transcript(md_text)
    
    def render_assistant_content(self, assistant_text: str) -> None:
        parsed = parse_assistant(assistant_text)

        narrative = parsed.answer_without_cmd.strip()
        if narrative:
            self.write_left_markdown(narrative)
            self.write_left_plain("")

        if parsed.cmd:
            timeout_sec = parsed.timeout_sec if parsed.timeout_sec > 0 else DEFAULT_TIMEOUT_SEC
            self.write_left_plain("[b magenta]<cmd>[/b magenta]")
            self.write_left_plain(parsed.cmd)
            self.write_left_plain("[b magenta]</cmd>[/b magenta]")
            self.write_left_plain(f"[dim]timeout={timeout_sec}s[/dim]\n")

    def render_user_content(self, user_text: str) -> None:
        # 如果是 <cmdout>，就用工具样式展示；其余内容原样展示
        m = CMDOUT_BLOCK_RE.search(user_text or "")
        if m:
            inner = m.group(1).strip()
            self.write_left_plain("[b green]<cmdout>[/b green]")
            self.write_left_plain(inner if inner else "[i](no output)[/i]")
            self.write_left_plain("[b green]</cmdout>[/b green]\n")

            rest = (user_text[:m.start()] + user_text[m.end():]).strip()
            if rest:
                self.write_left_plain(rest + "\n")
        else:
            # 用户文本一般不需要 markdown 渲染，保持最原始最稳
            self.write_left_plain((user_text or "").strip() + "\n")

    def replay_history_to_left(self) -> None:
        # 默认只重放最近 N 条，防止 session 太大刷屏
        max_n = int(os.environ.get("CMD_AI_DEV_REPLAY", "80"))
        history = self.session.messages[1:]  # skip system
        if max_n > 0 and len(history) > max_n:
            history = history[-max_n:]

        if not history:
            return

        self.write_left_plain(f"[dim]--- 已恢复历史消息（最近 {len(history)} 条，可用 CMD_AI_DEV_REPLAY 调整） ---[/dim]\n")

        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                self.write_left_plain("[b blue]user:[/b blue]")
                self.render_user_content(content)
            elif role == "assistant":
                self.write_left_plain("[b cyan]assistant:[/b cyan]")
                self.render_assistant_content(content)
            else:
                # 其它 role（如 system/tool）默认不展示，避免噪音
                continue

        self.write_left_plain("[dim]--- 历史结束 ---[/dim]\n")

    def on_mount(self) -> None:
        WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
        if not TRANSCRIPT_PATH.exists():
            TRANSCRIPT_PATH.write_text("", encoding="utf-8")

        self.write_left_plain("[b]cmd-ai-dev[/b]  Ctrl+S发送 | Ctrl+T停止 | Ctrl+R重置 | Ctrl+Q退出 | F2发送")
        self.write_left_plain(f"WORKSPACE={WORKSPACE}")
        self.write_left_plain(f"WORKSPACE_AI={WORKSPACE_AI}")
        self.write_left_plain(f"SESSION={SESSION_PATH}")
        self.write_left_plain(f"VENV_BIN={VENV_BIN}")
        self.write_left_plain(f"TRANSCRIPT={TRANSCRIPT_PATH}")
        self.write_left_plain("")

        if len(self.session.messages) > 1:
            self.write_left_plain("[i]已加载现有会话：/workspace-ai/session.json[/i]\n")
            self.replay_history_to_left()

        self.run_worker(self.agent_loop(), exclusive=True, name="agent_loop")

    async def shutdown(self) -> None:
        self.stop_requested = True
        await self.runner.interrupt()
        try:
            self.session.save()
        except Exception:
            pass
        self.exit()

    async def action_quit(self) -> None:
        await self.shutdown()

    async def action_send(self) -> None:
        await self.handle_send()

    async def action_select_all(self) -> None:
        focused = self.focused
        if isinstance(focused, TextArea):
            # 不同 Textual 版本 API 可能略有差异，这里做个兼容
            if hasattr(focused, "action_select_all"):
                focused.action_select_all()  # type: ignore[attr-defined]
            elif hasattr(focused, "select_all"):
                focused.select_all()  # type: ignore[attr-defined]

    async def action_stop(self) -> None:
        await self.handle_stop()

    async def action_reset(self) -> None:
        await self.handle_reset()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn_send":
            await self.handle_send()
        elif bid == "btn_stop":
            await self.handle_stop()
        elif bid == "btn_reset":
            await self.handle_reset()
        elif bid == "btn_quit":
            await self.shutdown()

    async def agent_loop(self) -> None:
        while True:
            role, content = await self.queue.get()
            self.session.add(role, content)
            self.busy = True

            self.write_left_plain("[b cyan]assistant:[/b cyan] (generating...)")
            try:
                assistant_text = await self.llm.stream_chat(self.session.messages)
            except Exception as e:
                self.write_left_plain(f"[b red]LLM error:[/b red] {e}\n")
                self.busy = False
                continue

            # 存 raw（含 <cmd>）到会话
            self.session.add("assistant", assistant_text)

            parsed = parse_assistant(assistant_text)

            # 展示：叙述部分走 markdown；cmd 单独展示，避免重复
            narrative = parsed.answer_without_cmd.strip()
            if narrative:
                self.write_left_markdown(narrative)
                self.write_left_plain("")

            if self.stop_requested:
                note = (
                    "<system_note>\n"
                    "用户触发 STOP：已停止命令链；任何待执行命令已取消；如需继续，请等待用户新指令。\n"
                    "</system_note>"
                )
                self.session.add("user", note)
                self.write_left_plain("[yellow]STOP 已触发：不会执行模型给出的命令。[/yellow]\n")
                self.busy = False
                continue

            if not parsed.cmd:
                self.busy = False
                continue

            cmd_to_run = parsed.cmd
            timeout_sec = parsed.timeout_sec if parsed.timeout_sec > 0 else DEFAULT_TIMEOUT_SEC

            self.write_left_plain("[b magenta]<cmd>[/b magenta]")
            self.write_left_plain(cmd_to_run)
            self.write_left_plain("[b magenta]</cmd>[/b magenta]")
            self.write_left_plain(f"[dim]timeout={timeout_sec}s[/dim]\n")

            result = await self.runner.run(cmd_to_run, timeout_sec=timeout_sec, cwd=WORKSPACE)

            if self.stop_requested or result.interrupted:
                note = (
                    "<system_note>\n"
                    "用户触发 STOP：命令执行已中断/结果丢弃；请等待用户新指令。\n"
                    "</system_note>"
                )
                self.session.add("user", note)
                self.write_left_plain("[yellow]STOP：命令结果未发送给模型。[/yellow]\n")
                self.busy = False
                continue

            extra_note_parts = [f"(命令输出日志：{result.log_path})"]
            if result.truncated:
                extra_note_parts.append("（输出过长已截断，完整输出见日志文件）")
            extra_note = "\n".join(extra_note_parts)

            self.write_left_plain("[b green]<cmdout>[/b green]")
            self.write_left_plain(result.output if result.output.strip() else "[i](no output)[/i]")
            self.write_left_plain("[b green]</cmdout>[/b green]")
            self.write_left_plain(f"[dim]{extra_note}[/dim]\n")

            user_extra = ""
            if self.pending_user_buffer:
                user_extra = "\n" + "\n".join(self.pending_user_buffer).strip()
                self.pending_user_buffer.clear()

            cmdout_msg = format_cmdout(
                exit_code=result.exit_code,
                timed_out=result.timed_out,
                interrupted=result.interrupted,
                output=result.output,
                extra_note=extra_note,
            )

            self.queue.put_nowait(("user", (cmdout_msg + user_extra).strip()))

    async def handle_send(self) -> None:
        ta = self.query_one("#right", TextArea)
        text = ta.text.strip()
        if not text:
            return

        ta.text = ""

        # 新一轮对话开始，清掉 stop 标记
        if not self.busy:
            self.stop_requested = False

        if text in ("/help", ":help"):
            self.write_left_plain("命令：/help  /reset  /exit\n快捷键：Ctrl+S发送 Ctrl+T停止 Ctrl+R重置 Ctrl+Q退出 F2发送\n")
            return
        if text in ("/exit", ":q", ":quit"):
            await self.shutdown()
            return
        if text in ("/reset", ":reset"):
            await self.handle_reset()
            return

        self.write_left_plain("[b blue]user:[/b blue]")
        self.write_left_plain(text + "\n")

        if self.busy:
            self.pending_user_buffer.append(text)
            return

        self.queue.put_nowait(("user", text))

    async def handle_stop(self) -> None:
        self.stop_requested = True
        await self.runner.interrupt()
        self.write_left_plain("[yellow]STOP 请求已发送：将中断命令链（命令会被终止/丢弃）。[/yellow]\n")

    async def handle_reset(self) -> None:
        if SESSION_PATH.exists():
            SESSION_PATH.unlink()
        self.session = Session.load_or_create()
        self.pending_user_buffer.clear()
        self.stop_requested = False
        self.busy = False
        self._left().clear()
        self.write_left_plain("[b]会话已重置[/b]")
        self.write_left_plain(f"SESSION={SESSION_PATH}\n")


def main() -> None:
    WORKSPACE_AI.mkdir(parents=True, exist_ok=True)

    stty_state: Optional[str] = None

    # 在 Textual 接管终端之前保存 stty，并关闭 XON/XOFF（否则 Ctrl+S 可能冻住终端）
    if sys.stdin.isatty():
        try:
            stty_state = subprocess.check_output(
                ["stty", "-g"],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            subprocess.run(["stty", "-ixon"], stderr=subprocess.DEVNULL, check=False)
        except Exception:
            stty_state = None

    try:
        CmdAIDevApp().run()
    finally:
        # 无论如何恢复 stty，避免影响用户 shell（Tab/补全等）
        if stty_state and sys.stdin.isatty():
            try:
                subprocess.run(["stty", stty_state], stderr=subprocess.DEVNULL, check=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()