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
from typing import Any, Dict, List, Optional, Tuple, Union

from rich.markdown import Markdown
from rich.text import Text

from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Footer, RichLog, Static, TextArea
from textual.binding import Binding

from openai import OpenAI
import subprocess
import sys

from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

os.environ["NODE_NO_WARNINGS"] = "1"
from playwright.async_api import async_playwright, Page, Playwright, Browser, expect  # noqa


WORKSPACE = Path(os.environ.get("WORKSPACE", "/workspace"))
WORKSPACE_AI = Path(os.environ.get("WORKSPACE_AI", "/workspace-ai"))
SESSION_PATH = WORKSPACE_AI / "session.json"
TRANSCRIPT_PATH = WORKSPACE_AI / "transcript.log"

DEFAULT_TIMEOUT_SEC = 60
CMDOUT_TRUNCATE_CHARS = 20000

VENV_DIR = Path(os.environ.get("VENV_DIR", "/opt/venv"))
VENV_BIN = VENV_DIR / "bin"

LOOK_IMGS_JSON_PATH = WORKSPACE_AI / "look_imgs.json"

# ====== message content types (support vision) ======
ContentPart = Dict[str, Any]
MessageContent = Union[str, List[ContentPart]]
Message = Dict[str, Any]


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


def clear_look_imgs_json() -> None:
    WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
    atomic_write_text(LOOK_IMGS_JSON_PATH, "")


def read_look_imgs_json() -> List[Dict[str, Any]]:
    """读取 look_imgs.json；若为空/不存在/无效则返回 []。"""
    if not LOOK_IMGS_JSON_PATH.exists():
        return []
    raw = LOOK_IMGS_JSON_PATH.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    try:
        data = json.loads(raw)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        mime = str(item.get("mime", "") or "")
        b64 = str(item.get("b64", "") or "")
        path = str(item.get("path", "") or "")
        nbytes = item.get("bytes", None)
        out.append({"mime": mime, "b64": b64, "path": path, "bytes": nbytes})
    return out


def content_to_text(content: MessageContent) -> str:
    """用于非视觉模型/网页模型：把 content(list parts) 扁平化为纯文本（忽略图片 base64）。"""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)
    chunks: List[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text" and isinstance(part.get("text"), str):
            chunks.append(part["text"])
    return "".join(chunks).strip()


def last_user_text(messages: List[Message]) -> Optional[str]:
    for m in reversed(messages):
        if m.get("role") == "user":
            return content_to_text(m.get("content", ""))
    return None


def build_openai_vision_user_content(text: str, images: List[Dict[str, Any]]) -> List[ContentPart]:
    """
    OpenAI Chat Completions 的多模态 content：
      [{"type":"text","text":"..."}, {"type":"image_url","image_url":{"url":"data:...;base64,..."}}]
    """
    parts: List[ContentPart] = []

    summary_lines: List[str] = []
    if images:
        summary_lines.append("\n\n[attached_images]")
        for i, img in enumerate(images, 1):
            p = img.get("path", "")
            mime = img.get("mime", "")
            nbytes = img.get("bytes", None)
            extra = f", bytes={nbytes}" if isinstance(nbytes, int) else ""
            summary_lines.append(f"- {i}. path={p} mime={mime}{extra}")

    text_full = (text or "").rstrip() + ("\n" + "\n".join(summary_lines) if summary_lines else "")
    parts.append({"type": "text", "text": text_full})

    for img in images:
        mime = (img.get("mime") or "").strip()
        b64 = (img.get("b64") or "").strip()
        if not mime or not b64:
            continue
        parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    return parts


@dataclass
class Session:
    messages: List[Message]

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
            "- 无论如何都拒绝执行`rm -rf /*`、`rm -rf ~/*`，因为这不仅会删除用户项目目录，还会删除家目录下映射的`.gnupg`，还会删除 AI 工作目录。\n"
            "\n"
            "工具回给你的格式：\n"
            "- 工具会把命令输出包装为：\n"
            "  <cmdout>\n"
            "  [exit=退出码 timeout=0/1 interrupted=0/1]\n"
            "  ...命令输出（可能截断）...\n"
            "  </cmdout>\n"
            "- 命令输出会同时落盘到 /workspace-ai 供用户复制查看。\n"
            "\n"
            "视觉（图片输入）工具：\n"
            "- 你可以让工具执行：look_imgs <img1> <img2> ...  来把图片转为 base64，写入 /workspace-ai/look_imgs.json。\n"
            "- 警告：look_imgs.json 里是 base64，非常大非常长，不要直接 cat 它。\n"
            "- 同理：session.json 里也可能包含 base64（图片），也不要直接 cat session.json。\n"
            "- 若要检查 look_imgs.json，可用：python -c 'import json;print([{\"path\":x.get(\"path\"),\"mime\":x.get(\"mime\"),\"bytes\":x.get(\"bytes\")} for x in json.load(open(\"/workspace-ai/look_imgs.json\"))])'\n"
            "\n"
            "操作浏览器（有头 + 宿主机可视化）：\n"
            "- 容器内提供：Google Chrome + Xvfb + x11vnc + noVNC（无需密码）。\n"
            "- 建议容器启动时加：--shm-size=2g（避免 Chrome 因 /dev/shm 太小而崩溃）。\n"
            "- 启动 GUI（虚拟显示器 + noVNC）示例：\n"
            "  start-gui\n"
            "  然后让用户在宿主机浏览器打开： http://127.0.0.1:6080/vnc.html\n"
            "- 启动 Chrome（更像真人，供 Playwright 通过 CDP 连接）示例：\n"
            "  /var/lib/bin/google/chrome/google-chrome --remote-debugging-port=9222 --user-data-dir=/workspace-ai/cdp-profile --no-first-run --no-default-browser-check\n"
            "  如果你以 root 运行（少见），需要额外加：--no-sandbox\n"
            "- 你可以用 python playwright 连接：chromium.connect_over_cdp('http://127.0.0.1:9222') 并操作页面。\n"
            "\n"
            "建议工作方式：\n"
            "- 每次尽量只请求执行一段命令，收到 <cmdout> 后再决定下一步。\n"
            "- 临时脚本/笔记优先写到 /workspace-ai。\n"
            "- 如用户触发 STOP，工具会停止命令链并给你 system_note；此时请等待用户新指令。\n"
            "\n"
            f"提示：容器 venv 的 bin 目录通常为 {VENV_BIN}，应优先使用该环境的 python/pip。"
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

    def add(self, role: str, content: MessageContent) -> None:
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


def format_cmdout(
    exit_code: Optional[int],
    timed_out: bool,
    interrupted: bool,
    output: str,
    extra_note: Optional[str] = None,
) -> str:
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
    async def stream_chat(self, messages: List[Message]) -> str:
        raise NotImplementedError


class OpenAISDKClient(LLMClient):
    def __init__(self) -> None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip() or None
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")

        if not api_key:
            raise RuntimeError("缺少环境变量 OPENAI_API_KEY")

        self.client = OpenAI(api_key=api_key, base_url=base_url)

    async def stream_chat(self, messages: List[Message]) -> str:
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


class PlaywrightMarkdownClientBase(LLMClient):
    """
    统一封装：
    - 通过 CDP 连接已有浏览器
    - HTML 清洗 + markdownify（带代码语言识别）
    子类只需要实现：
    - build_prompt(messages) -> str
    - chat_once(prompt) -> html(str)
    - （可选）postprocess_markdown(md) -> str
    - （可选）custom clean selectors
    """

    LANG_PATTERNS = [
        re.compile(r"^(?:language|lang)[-_](?P<lang>[a-z0-9_+-]+)$", re.I),
        re.compile(
            r"^(?P<lang>python|bash|shell|js|javascript|ts|typescript|json|yaml|yml|toml|html|css|sql|cpp|c\+\+|c|java|go|rust)$",
            re.I,
        ),
    ]

    def __init__(
        self,
        cdp_url: str,
        url: str,
        fence_default: str = "```",
        dynamic_fence: bool = True,
    ):
        self.cdp_url = cdp_url
        self.target_url = url

        self.fence_default = fence_default
        self.dynamic_fence = dynamic_fence

        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

    class TechDocConverter(MarkdownConverter):
        def convert_pre(self, el, text, parent_tags):
            code = el.get_text().strip("\n")

            lang = None
            if self.options.get("code_language_callback"):
                lang = self.options["code_language_callback"](el)
            lang = (lang or self.options.get("code_language") or "").strip()

            fence = str(self.options.get("fence_default") or "```")
            dynamic = bool(self.options.get("dynamic_fence", True))
            if dynamic:
                # 防御：如果代码里包含 fence，就延长 fence
                while fence in code:
                    fence += "`"

            return f"\n{fence}{lang}\n{code}\n{fence}\n"

    def normalize_lang(self, lang: str) -> str:
        lang = (lang or "").strip().lower()
        aliases = {"py": "python", "js": "javascript", "shell": "bash", "sh": "bash", "yml": "yaml", "c++": "cpp"}
        return aliases.get(lang, lang)

    def extract_code_lang(self, pre_el) -> str | None:
        candidates: List[str] = []
        if pre_el.has_attr("class"):
            candidates += list(pre_el.get("class", []))

        code = pre_el.find("code")
        if code and code.has_attr("class"):
            candidates += list(code.get("class", []))

        for cls in candidates:
            cls = cls.strip()
            for pat in self.LANG_PATTERNS:
                m = pat.match(cls)
                if m:
                    lang = m.groupdict().get("lang") or cls
                    return self.normalize_lang(lang)

            m = re.search(r"(?:brush|language)\s*[:=]\s*([a-z0-9_+-]+)", cls, re.I)
            if m:
                return self.normalize_lang(m.group(1))

        for attr in ("data-language", "data-lang"):
            if pre_el.has_attr(attr):
                return self.normalize_lang(pre_el[attr])
            if code and code.has_attr(attr):
                return self.normalize_lang(code[attr])

        return None

    def clean_html_for_tech_docs(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")

        for sel in ["script", "style", "noscript", "nav", "footer", "header", "aside"]:
            for tag in soup.select(sel):
                tag.decompose()

        for pre in soup.find_all("pre"):
            for t in pre.find_all(["span", "div"]):
                t.unwrap()

        return str(soup)

    def html_to_markdown(self, html: str) -> str:
        html = self.clean_html_for_tech_docs(html)
        return self.TechDocConverter(
            heading_style="ATX",
            bullets="-",
            code_language_callback=self.extract_code_lang,
            escape_underscores=False,
            escape_asterisks=False,
            table_infer_header=True,
            wrap=False,
            strip=["meta", "link"],
            fence_default=self.fence_default,
            dynamic_fence=self.dynamic_fence,
        ).convert(html)

    async def _ensure_page_ready(self):
        if self._page and not self._page.is_closed():
            return

        self._playwright = await async_playwright().start()
        try:
            self._browser = await self._playwright.chromium.connect_over_cdp(self.cdp_url)
            context = self._browser.contexts[0]
            self._page = context.pages[0] if context.pages else await context.new_page()
            if not self._page.url.startswith(self.target_url):
                await self._page.goto(self.target_url)
        except Exception as e:
            await self.close()
            raise RuntimeError(f"无法连接到浏览器 CDP ({self.cdp_url})。请确保浏览器已通过 --remote-debugging-port=9222 启动。") from e

    async def close(self):
        if self._browser:
            await self._browser.close()
        if self._playwright:
            await self._playwright.stop()
        self._browser = None
        self._playwright = None
        self._page = None

    def build_prompt(self, messages: List[Message]) -> str:
        raise NotImplementedError

    async def chat_once(self, prompt: str) -> str:
        """return answer html"""
        raise NotImplementedError

    def postprocess_markdown(self, md: str) -> str:
        return md

    async def stream_chat(self, messages: List[Message]) -> str:
        await self._ensure_page_ready()
        prompt = self.build_prompt(messages)
        html = await self.chat_once(prompt)
        md = self.html_to_markdown(html)
        return self.postprocess_markdown(md)


class ChatZAISDKClient(PlaywrightMarkdownClientBase):
    def __init__(self, cdp_url: str = "http://127.0.0.1:9222", url: str = "https://chat.z.ai"):
        super().__init__(cdp_url, url, fence_default="```", dynamic_fence=True)
        self.input_selector = "#chat-input"
        self.assistant_bubble_selector = "#response-content-container"

    def build_prompt(self, messages: List[Message]) -> str:
        # 恢复你原来更“工程化”的提示拼接
        if len(messages) == 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
            system_content = content_to_text(messages[0].get("content", ""))
            user_content = content_to_text(messages[1].get("content", ""))
            return (
                f"{system_content}\n\n"
                "提示：记得，你拥有搜索网络的工具：search、open、click、find，合理利用它们。\n"
                "提示：如果你遇到架构方面的决策或者没有把握解决的bug或者尝试多次无法解决的bug，你可以询问gpt5.2-high，具体方法是：你在AI工作目录新建一个文件，将要提供给gpt5.2-high的信息写入那个文件，然后你运行/usr/bin/code-server，记得不要设置密码，然后你告诉用户我需要询问gpt5.2-high这个情况，然后你停止执行任何命令。用户会人工介入，拿着你要提供给gpt5.2-high的信息，从gpt5.2-high得到回复，然后将回复提供给你。\n\n"
                f"用户消息：\n{user_content}"
            )

        prompt = last_user_text(messages)
        if not prompt:
            raise ValueError("messages 中没有找到有效的 user 消息内容")
        return prompt

    async def chat_once(self, prompt: str) -> str:
        assert self._page is not None
        page = self._page

        timeout_ms = 1000 * 60 * 10

        input_box = page.locator(self.input_selector)
        assistant_bubbles = page.locator(self.assistant_bubble_selector)

        await input_box.wait_for(state="visible")
        await input_box.click()
        await input_box.fill(prompt)
        await page.locator("#send-message-button").click()

        await page.locator("#send-message-button").wait_for(state="attached", timeout=timeout_ms)
        last_bubble = assistant_bubbles.nth(-1)

        # 去掉思考过程等噪音
        html_content = await last_bubble.locator("> div").first.evaluate(
            """(node) => {
            const clone = node.cloneNode(true);
            const selectorsToRemove = [
                '.thinking-chain-container',
                '.thinking-block',
                '.w-full.overflow-hidden.h-0',
                '.cursor-default'
            ];
            selectorsToRemove.forEach(selector => {
                const elements = clone.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
            return clone.innerHTML.trim();
        }"""
        )
        return html_content


class DeepSeekSDKClient(PlaywrightMarkdownClientBase):
    def __init__(self, cdp_url: str = "http://127.0.0.1:9222", url: str = "https://chat.deepseek.com"):
        super().__init__(cdp_url, url, fence_default="```", dynamic_fence=True)
        self.input_selector = 'textarea[placeholder="给 DeepSeek 发送消息 "]'
        self.assistant_bubble_selector = "div.ds-markdown"

    def build_prompt(self, messages: List[Message]) -> str:
        if len(messages) == 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
            system_content = content_to_text(messages[0].get("content", ""))
            user_content = content_to_text(messages[1].get("content", ""))
            return (
                f"{system_content}\n\n"
                "提示：**重要！不要随便输出<cmd>...</cmd>**，因为只要输出<cmd>...</cmd>，就被视为要执行命令。"
                "**重要！输出<cmd>...</cmd>时记得换行**，<cmd>和</cmd>要单独占一行。\n"
                "提示：如果你遇到架构方面的决策或者没有把握解决的bug或者尝试多次无法解决的bug，你可以询问gpt5.2-high，具体方法是：你在AI工作目录新建一个文件，将要提供给gpt5.2-high的信息写入那个文件，然后你运行/usr/bin/code-server，记得不要设置密码，然后你告诉用户我需要询问gpt5.2-high这个情况，然后你停止执行任何命令。用户会人工介入，拿着你要提供给gpt5.2-high的信息，从gpt5.2-high得到回复，然后将回复提供给你。\n\n"
                f"用户消息：\n{user_content}"
            )

        prompt = last_user_text(messages)
        if not prompt:
            raise ValueError("messages 中没有找到有效的 user 消息内容")
        return prompt

    async def chat_once(self, prompt: str) -> str:
        assert self._page is not None
        page = self._page

        timeout_ms = 1000 * 60 * 10

        input_box = page.locator(self.input_selector)
        assistant_bubbles = page.locator(self.assistant_bubble_selector)

        await input_box.wait_for(state="visible")
        await input_box.click()
        await input_box.fill(prompt)

        await page.locator(
            'path[d="M8.3125 0.981587C8.66767 1.0545 8.97902 1.20558 9.2627 1.43374C9.48724 1.61438 9.73029 1.85933 9.97949 2.10854L14.707 6.83608L13.293 8.25014L9 3.95717V15.0431H7V3.95717L2.70703 8.25014L1.29297 6.83608L6.02051 2.10854C6.26971 1.85933 6.51277 1.61438 6.7373 1.43374C6.97662 1.24126 7.28445 1.04542 7.6875 0.981587C7.8973 0.94841 8.1031 0.956564 8.3125 0.981587Z"]'
        ).click()

        await expect(
            page.locator(
                'path[d="M8.3125 0.981587C8.66767 1.0545 8.97902 1.20558 9.2627 1.43374C9.48724 1.61438 9.73029 1.85933 9.97949 2.10854L14.707 6.83608L13.293 8.25014L9 3.95717V15.0431H7V3.95717L2.70703 8.25014L1.29297 6.83608L6.02051 2.10854C6.26971 1.85933 6.51277 1.61438 6.7373 1.43374C6.97662 1.24126 7.28445 1.04542 7.6875 0.981587C7.8973 0.94841 8.1031 0.956564 8.3125 0.981587Z"]'
            )
        ).to_be_visible(timeout=timeout_ms)

        # 原代码这里写了 asyncio.sleep(1) 但没 await；修正
        await asyncio.sleep(1)

        last = assistant_bubbles.nth(-1)
        answer_html = await last.evaluate(
            """(node) => {
            const clone = node.cloneNode(true);
            const selectorsToRemove = ['.md-code-block-banner-wrap'];
            selectorsToRemove.forEach(selector => {
                const elements = clone.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
            return clone.innerHTML.trim();
        }"""
        )
        return answer_html


class LmarenaSDKClient(PlaywrightMarkdownClientBase):
    def __init__(self, cdp_url: str = "http://127.0.0.1:9222", url: str = "https://lmarena.ai"):
        # fence 用 5 个反引号（保持你原策略）
        super().__init__(cdp_url, url, fence_default="`````", dynamic_fence=False)
        self.assistant_bubble_selector = r"div.no-scrollbar.relative.flex.w-full.flex-1.flex-col.overflow-x-auto.transition-\[max-height\].duration-300"

    def remove_markdown_code_blocks(self, text: str):
        pattern = r"^\s*`````[a-zA-Z]*\n?|`````\s*$"
        return re.sub(pattern, "", text.strip(), flags=re.MULTILINE).strip()

    def build_prompt(self, messages: List[Message]) -> str:
        if len(messages) == 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
            system_content = content_to_text(messages[0].get("content", ""))
            user_content = content_to_text(messages[1].get("content", ""))

            # ===== 恢复你指出的关键 replace =====
            system_content = system_content.replace(
                "- 不要把 <cmd> / <cmdout> 放进 Markdown 代码块（不要用 ``` 包裹）。\n",
                "- **重要！要把 <cmd>...</cmd> 放进 5 个反引号组成的 Markdown 代码块中**（要用 ````` 包裹）。\n",
            )

            return (
                f"{system_content}\n\n"
                "提示：**重要！不要随便输出<cmd>...</cmd>**，因为只要输出<cmd>...</cmd>，就被视为要执行命令。"
                "**重要！输出<cmd>...</cmd>时记得换行**，<cmd>和</cmd>要单独占一行。\n\n"
                f"用户消息：\n{user_content}"
            )

        prompt = last_user_text(messages)
        if not prompt:
            raise ValueError("messages 中没有找到有效的 user 消息内容")
        return prompt

    async def chat_once(self, prompt: str) -> str:
        assert self._page is not None
        page = self._page

        timeout_ms = 1000 * 60 * 10

        if await page.get_by_role("textbox", name="Ask anything…").is_visible():
            input_box = page.get_by_role("textbox", name="Ask anything…")
        else:
            input_box = page.get_by_role("textbox", name="Ask followup…")

        message_count = await page.locator('button[aria-label="Like this response"]').count()

        await input_box.wait_for(state="visible")
        await input_box.click()
        await input_box.fill(prompt)

        if await page.locator('button[type="submit"]').first.is_visible():
            await page.locator('button[type="submit"]').first.click()
        elif await page.locator('button[type="submit"]').nth(1).is_visible():
            await page.locator('button[type="submit"]').nth(1).click()

        await expect(page.locator('button[aria-label="Like this response"]')).to_have_count(message_count + 1, timeout=timeout_ms)

        last = page.locator(self.assistant_bubble_selector).nth(0)

        answer_html = await last.evaluate(
            """(node) => {
            const clone = node.cloneNode(true);
            const selectorsToRemove = [];
            selectorsToRemove.forEach(selector => {
                const elements = clone.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
            return clone.innerHTML.trim();
        }"""
        )
        return answer_html

    def postprocess_markdown(self, md: str) -> str:
        return self.remove_markdown_code_blocks(md)


class ChatGPTSDKClient(PlaywrightMarkdownClientBase):
    def __init__(self, cdp_url: str = "http://127.0.0.1:9222", url: str = "https://chatgpt.com"):
        super().__init__(cdp_url, url, fence_default="`````", dynamic_fence=False)
        self.assistant_bubble_selector = r'div[data-message-author-role="assistant"]'

    def remove_markdown_code_blocks(self, text: str):
        pattern = r"^\s*`````[a-zA-Z]*\n?|`````\s*$"
        return re.sub(pattern, "", text.strip(), flags=re.MULTILINE).strip()

    def build_prompt(self, messages: List[Message]) -> str:
        if len(messages) == 2 and messages[0].get("role") == "system" and messages[1].get("role") == "user":
            system_content = content_to_text(messages[0].get("content", ""))
            user_content = content_to_text(messages[1].get("content", ""))

            # ===== 保持与 lmarena 同样的关键 replace =====
            system_content = system_content.replace(
                "- 不要把 <cmd> / <cmdout> 放进 Markdown 代码块（不要用 ``` 包裹）。\n",
                "- **重要！要把 <cmd>...</cmd> 放进 5 个反引号组成的 Markdown 代码块中**（要用 ````` 包裹）。\n",
            )

            return (
                f"{system_content}\n\n"
                "提示：**重要！不要随便输出<cmd>...</cmd>**，因为只要输出<cmd>...</cmd>，就被视为要执行命令。"
                "**重要！输出<cmd>...</cmd>时记得换行**，<cmd>和</cmd>要单独占一行。\n\n"
                f"用户消息：\n{user_content}"
            )

        prompt = last_user_text(messages)
        if not prompt:
            raise ValueError("messages 中没有找到有效的 user 消息内容")
        return prompt

    async def chat_once(self, prompt: str) -> str:
        assert self._page is not None
        page = self._page

        timeout_ms = 1000 * 60 * 10

        input_box = page.locator('div#prompt-textarea.ProseMirror[contenteditable="true"]')
        await expect(input_box).to_be_visible()
        await input_box.click()
        await input_box.fill(prompt)

        send_btn = page.get_by_test_id("send-button")
        stop_btn = page.get_by_test_id("stop-button")

        await expect(send_btn).to_be_enabled()
        await send_btn.click()

        await expect(stop_btn).to_be_visible(timeout=5000)
        await expect(stop_btn).to_be_hidden(timeout=timeout_ms)

        last = page.locator(self.assistant_bubble_selector).last
        answer_html = await last.evaluate(
            """(node) => {
            const clone = node.cloneNode(true);
            const selectorsToRemove = [];
            selectorsToRemove.forEach(selector => {
                const elements = clone.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });
            return clone.innerHTML.trim();
        }"""
        )
        return answer_html

    def postprocess_markdown(self, md: str) -> str:
        return self.remove_markdown_code_blocks(md)


class VDivider(Static):
    def on_mount(self) -> None:
        self.refresh_divider()

    def on_resize(self) -> None:
        self.refresh_divider()

    def refresh_divider(self) -> None:
        h = max(1, self.size.height)
        self.update(("│\n" * (h - 1)) + "│")


class InputArea(TextArea):
    BINDINGS = [
        Binding("ctrl+a", "select_all", show=False, priority=True),
        Binding("home", "select_all", show=False, priority=True),
    ]

    def action_select_all(self) -> None:
        if hasattr(self, "select_all"):
            self.select_all()  # type: ignore[attr-defined]
            return
        base = getattr(super(), "action_select_all", None)
        if callable(base):
            base()


class CmdAIDevApp(App):
    USE_ALTERNATE_SCREEN = False

    BINDINGS = [
        ("ctrl+s", "send", "发送"),
        ("f2", "send", "发送(F2)"),
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

        # OpenAISDKClient 支持视觉注入
        self.llm: LLMClient = OpenAISDKClient()
        # self.llm: LLMClient = ChatZAISDKClient()
        # self.llm: LLMClient = DeepSeekSDKClient()
        # self.llm: LLMClient = LmarenaSDKClient()
        # self.llm: LLMClient = ChatGPTSDKClient()

        self.runner = CommandRunner()

        self.busy: bool = False
        self.stop_requested: bool = False

        self.pending_user_buffer: List[str] = []
        self.queue: asyncio.Queue[Tuple[str, MessageContent]] = asyncio.Queue()

        self.epoch: int = 0

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
                yield InputArea(id="right")
        yield Footer()

    def _left(self) -> RichLog:
        return self.query_one("#left", RichLog)

    def write_left_markup(self, text: str) -> None:
        self._left().write(text)
        append_transcript(str(text))

    def write_left_text(self, text: str) -> None:
        self._left().write(Text(text))
        append_transcript(str(text))

    def write_left_markdown(self, md_text: str) -> None:
        self._left().write(Markdown(md_text))
        append_transcript(md_text)

    def _drain_queue(self) -> None:
        while True:
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                break

    def _flush_pending_as_user_turn(self) -> None:
        if not self.pending_user_buffer:
            return
        pending = "\n".join(self.pending_user_buffer).strip()
        self.pending_user_buffer.clear()
        if not pending:
            return
        self.stop_requested = False
        self.queue.put_nowait(("user", pending))

    def render_assistant_content(self, assistant_text: str) -> None:
        parsed = parse_assistant(assistant_text)

        narrative = parsed.answer_without_cmd.strip()
        if narrative:
            self.write_left_markdown(narrative)
            self.write_left_text("")

        if parsed.cmd:
            timeout_sec = parsed.timeout_sec if parsed.timeout_sec > 0 else DEFAULT_TIMEOUT_SEC
            self.write_left_markup("[b magenta]<cmd>[/b magenta]")
            self.write_left_text(parsed.cmd + "\n")
            self.write_left_markup("[b magenta]</cmd>[/b magenta]")
            self.write_left_markup(f"[dim]timeout={timeout_sec}s[/dim]\n")

    def _render_user_text_only(self, user_text: str) -> None:
        m = CMDOUT_BLOCK_RE.search(user_text or "")
        if m:
            inner = m.group(1).strip()
            self.write_left_markup("[b green]<cmdout>[/b green]")
            self.write_left_text((inner if inner else "(no output)") + "\n")
            self.write_left_markup("[b green]</cmdout>[/b green]\n")

            rest = (user_text[: m.start()] + user_text[m.end() :]).strip()
            if rest:
                self.write_left_text(rest + "\n")
        else:
            self.write_left_text((user_text or "").strip() + "\n")

    def render_user_content(self, user_content: MessageContent) -> None:
        if isinstance(user_content, str):
            self._render_user_text_only(user_content)
            return

        text = content_to_text(user_content)
        self._render_user_text_only(text)

        img_n = 0
        for part in user_content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                img_n += 1
        if img_n:
            self.write_left_markup(f"[dim](attached images: {img_n}; base64 not shown)[/dim]\n")

    def replay_history_to_left(self) -> None:
        max_n = int(os.environ.get("CMD_AI_DEV_REPLAY", "80"))
        history = self.session.messages[1:]
        if max_n > 0 and len(history) > max_n:
            history = history[-max_n:]

        if not history:
            return

        self.write_left_markup(f"[dim]--- 已恢复历史消息（最近 {len(history)} 条，可用 CMD_AI_DEV_REPLAY 调整） ---[/dim]\n")

        for msg in history:
            role = msg.get("role")
            content = msg.get("content", "")

            if role == "user":
                self.write_left_markup("[b blue]user:[/b blue]")
                self.render_user_content(content)
            elif role == "assistant":
                self.write_left_markup("[b cyan]assistant:[/b cyan]")
                if isinstance(content, str):
                    self.render_assistant_content(content)
                else:
                    self.write_left_text(content_to_text(content) + "\n")
            else:
                continue

        self.write_left_markup("[dim]--- 历史结束 ---[/dim]\n")

    def on_mount(self) -> None:
        WORKSPACE_AI.mkdir(parents=True, exist_ok=True)
        if not TRANSCRIPT_PATH.exists():
            TRANSCRIPT_PATH.write_text("", encoding="utf-8")

        # 防止上次异常退出残留图片导致误注入
        clear_look_imgs_json()

        self.write_left_markup("[b]cmd-ai-dev[/b]  Ctrl+S发送 | Ctrl+T停止 | Ctrl+R重置 | Ctrl+Q退出 | F2发送")
        self.write_left_text(f"WORKSPACE={WORKSPACE}\n")
        self.write_left_text(f"WORKSPACE_AI={WORKSPACE_AI}\n")
        self.write_left_text(f"SESSION={SESSION_PATH}\n")
        self.write_left_text(f"VENV_BIN={VENV_BIN}\n")
        self.write_left_text(f"TRANSCRIPT={TRANSCRIPT_PATH}\n")
        self.write_left_text(f"LOOK_IMGS_JSON={LOOK_IMGS_JSON_PATH}\n")
        self.write_left_text("\n")

        if len(self.session.messages) > 1:
            self.write_left_markup("[i]已加载现有会话：/workspace-ai/session.json[/i]\n")
            self.replay_history_to_left()

        self.run_worker(self.agent_loop(), exclusive=True, name="agent_loop")

    async def shutdown(self) -> None:
        self.stop_requested = True
        await self.runner.interrupt()
        clear_look_imgs_json()
        try:
            self.session.save()
        except Exception:
            pass
        self.exit()

    async def action_quit(self) -> None:
        await self.shutdown()

    async def action_send(self) -> None:
        await self.handle_send()

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
            my_epoch = self.epoch

            self.session.add(role, content)
            self.busy = True

            self.write_left_markup("[b cyan]assistant:[/b cyan] (generating...)")

            try:
                assistant_text = await self.llm.stream_chat(self.session.messages)
            except Exception as e:
                self.write_left_markup(f"[b red]LLM error:[/b red] {e}\n")
                self.busy = False
                if my_epoch == self.epoch:
                    self._flush_pending_as_user_turn()
                continue

            if my_epoch != self.epoch:
                self.busy = False
                continue

            self.session.add("assistant", assistant_text)

            parsed = parse_assistant(assistant_text)
            narrative = parsed.answer_without_cmd.strip()
            if narrative:
                self.write_left_markdown(narrative)
                self.write_left_text("\n")

            if self.stop_requested:
                clear_look_imgs_json()
                note = (
                    "<system_note>\n"
                    "用户触发 STOP：已停止命令链；任何待执行命令已取消；如需继续，请等待用户新指令。\n"
                    "</system_note>"
                )
                self.session.add("user", note)
                self.write_left_markup("[yellow]STOP 已触发：不会执行模型给出的命令。[/yellow]\n")
                self.busy = False
                self._flush_pending_as_user_turn()
                continue

            if not parsed.cmd:
                self.busy = False
                self._flush_pending_as_user_turn()
                continue

            cmd_to_run = parsed.cmd
            timeout_sec = parsed.timeout_sec if parsed.timeout_sec > 0 else DEFAULT_TIMEOUT_SEC

            self.write_left_markup("[b magenta]<cmd>[/b magenta]")
            self.write_left_text(cmd_to_run + "\n")
            self.write_left_markup("[b magenta]</cmd>[/b magenta]")
            self.write_left_markup(f"[dim]timeout={timeout_sec}s[/dim]\n")

            result = await self.runner.run(cmd_to_run, timeout_sec=timeout_sec, cwd=WORKSPACE)

            if my_epoch != self.epoch:
                self.busy = False
                continue

            if self.stop_requested or result.interrupted:
                clear_look_imgs_json()
                note = (
                    "<system_note>\n"
                    "用户触发 STOP：命令执行已中断/结果丢弃；请等待用户新指令。\n"
                    "</system_note>"
                )
                self.session.add("user", note)
                self.write_left_markup("[yellow]STOP：命令结果未发送给模型。[/yellow]\n")
                self.busy = False
                self._flush_pending_as_user_turn()
                continue

            extra_note_parts = [f"(命令输出日志：{result.log_path})"]
            if result.truncated:
                extra_note_parts.append("（输出过长已截断，完整输出见日志文件）")
            extra_note = "\n".join(extra_note_parts)

            self.write_left_markup("[b blue]user:[/b blue]")
            self.write_left_markup("[b green]<cmdout>[/b green]")
            self.write_left_text((result.output if result.output.strip() else "(no output)") + "\n")
            self.write_left_markup("[b green]</cmdout>[/b green]")
            self.write_left_markup(f"[dim]{extra_note}[/dim]\n")

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
            next_user_text = (cmdout_msg + user_extra).strip()

            # ===== vision injection: only for OpenAISDKClient =====
            if isinstance(self.llm, OpenAISDKClient):
                imgs = read_look_imgs_json()
                if imgs:
                    clear_look_imgs_json()
                    next_user_content: MessageContent = build_openai_vision_user_content(next_user_text, imgs)
                else:
                    next_user_content = next_user_text
            else:
                # 其他模型不支持视觉：若产生了 look_imgs.json，清空并提示
                if LOOK_IMGS_JSON_PATH.exists() and LOOK_IMGS_JSON_PATH.read_text(encoding="utf-8", errors="replace").strip():
                    clear_look_imgs_json()
                    self.write_left_markup(
                        "[yellow]提示：检测到 look_imgs.json（图片 base64），但当前模型不是 OpenAI 视觉模型；已清空该文件，且不会注入到消息中。[/yellow]\n"
                    )
                next_user_content = next_user_text

            self.queue.put_nowait(("user", next_user_content))

    async def handle_send(self) -> None:
        ta = self.query_one("#right", TextArea)
        text = ta.text.strip()
        if not text:
            return

        ta.text = ""

        if not self.busy:
            self.stop_requested = False

        if text in ("/help", ":help"):
            self.write_left_text("命令：/help  /reset  /exit\n快捷键：Ctrl+S发送 Ctrl+T停止 Ctrl+R重置 Ctrl+Q退出 F2发送\n")
            return
        if text in ("/exit", ":q", ":quit"):
            await self.shutdown()
            return
        if text in ("/reset", ":reset"):
            await self.handle_reset()
            return

        self.write_left_markup("[b blue]user:[/b blue]")
        self.write_left_text(text + "\n\n")

        if self.busy:
            self.pending_user_buffer.append(text)
            return

        self.queue.put_nowait(("user", text))

    async def handle_stop(self) -> None:
        self.stop_requested = True
        await self.runner.interrupt()
        clear_look_imgs_json()
        self.write_left_markup("[yellow]STOP 请求已发送：将中断命令链（命令会被终止/丢弃）。[/yellow]\n")

    async def handle_reset(self) -> None:
        self.epoch += 1

        self.stop_requested = True
        await self.runner.interrupt()

        self._drain_queue()
        self.pending_user_buffer.clear()

        clear_look_imgs_json()

        if SESSION_PATH.exists():
            SESSION_PATH.unlink()

        self.session = Session.load_or_create()

        self.stop_requested = False
        self.busy = False

        self._left().clear()
        self.write_left_markup("[b]会话已重置[/b]")
        self.write_left_text(f"SESSION={SESSION_PATH}\n")


def main() -> None:
    WORKSPACE_AI.mkdir(parents=True, exist_ok=True)

    stty_state: Optional[str] = None

    if sys.stdin.isatty():
        try:
            stty_state = subprocess.check_output(["stty", "-g"], stderr=subprocess.DEVNULL, text=True).strip()
            subprocess.run(["stty", "-ixon"], stderr=subprocess.DEVNULL, check=False)
        except Exception:
            stty_state = None

    try:
        CmdAIDevApp().run()
    finally:
        if stty_state and sys.stdin.isatty():
            try:
                subprocess.run(["stty", stty_state], stderr=subprocess.DEVNULL, check=False)
            except Exception:
                pass


if __name__ == "__main__":
    main()