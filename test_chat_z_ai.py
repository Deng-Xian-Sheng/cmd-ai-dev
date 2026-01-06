import asyncio
import os
os.environ["NODE_NO_WARNINGS"] = "1"
from playwright.async_api import async_playwright, Locator, Page, TimeoutError as PlaywrightTimeoutError
import re
# 需要安装 lxml
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

class ChatBot:
    def __init__(self, page: Page, *,
                 input_selector: str,
                 assistant_bubble_selector: str):
        self.page = page
        self.input_selector = input_selector
        self.assistant_bubble_selector = assistant_bubble_selector
        self.LANG_PATTERNS = [
            # 常见：language-python / lang-python / python
            re.compile(r"^(?:language|lang)[-_](?P<lang>[a-z0-9_+-]+)$", re.I),
            # 有些站：sourceCode python / highlight-source-python 之类（按需扩展）
            re.compile(r"^(?P<lang>python|bash|shell|js|javascript|ts|typescript|json|yaml|yml|toml|html|css|sql|cpp|c\+\+|c|java|go|rust)$", re.I),
        ]

    class TechDocConverter(MarkdownConverter):
        def convert_pre(self, el, text, parent_tags):
            # 拿到纯文本代码（忽略内部高亮标签）
            code = el.get_text()

            # 去掉首尾多余空行（你也可以更激进）
            code = code.strip("\n")

            lang = None
            if self.options.get("code_language_callback"):
                lang = self.options["code_language_callback"](el)

            lang = (lang or self.options.get("code_language") or "").strip()
            fence = "```"

            # 防御：如果代码里本身包含 ```，就用更长的 fence
            if "```" in code:
                fence = "````"

            return f"\n{fence}{lang}\n{code}\n{fence}\n"


    def extract_code_lang(self, pre_el) -> str | None:
        """
        给 markdownify 的 code_language_callback 用：
        - 参数是 <pre> 的 BeautifulSoup Tag
        - 返回语言字符串（如 'python'），或 None
        """
        # 1) 优先从 <pre> 或其内部 <code> 的 class 里找
        candidates = []

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

            # 处理类似 "brush: python" / "language:python"
            m = re.search(r"(?:brush|language)\s*[:=]\s*([a-z0-9_+-]+)", cls, re.I)
            if m:
                return self.normalize_lang(m.group(1))

        # 2) 一些站点会放 data-language / data-lang
        for attr in ("data-language", "data-lang"):
            if pre_el.has_attr(attr):
                return self.normalize_lang(pre_el[attr])

            if code and code.has_attr(attr):
                return self.normalize_lang(code[attr])

        return None


    def normalize_lang(self, lang: str) -> str:
        lang = (lang or "").strip().lower()
        # 常见同义归一化
        aliases = {
            "py": "python",
            "js": "javascript",
            "shell": "bash",
            "sh": "bash",
            "yml": "yaml",
            "c++": "cpp",
        }
        return aliases.get(lang, lang)


    def clean_html_for_tech_docs(self, html: str) -> str:
        """
        预清洗：
        - 去掉脚本/样式/导航等
        - 对代码块内的高亮 <span> 做 unwrap，避免碎片化
        """
        soup = BeautifulSoup(html, "lxml")

        # 删噪音（按需增减）
        for sel in ["script", "style", "noscript", "nav", "footer", "header", "aside"]:
            for tag in soup.select(sel):
                tag.decompose()

        # 代码块内：拆掉多余的 span/div，保留纯文本
        for pre in soup.find_all("pre"):
            # 常见高亮器会把代码拆成 span
            for t in pre.find_all(["span", "div"]):
                t.unwrap()

        return str(soup)


    def html_to_markdown(self, html: str) -> str:
        html = self.clean_html_for_tech_docs(html)

        return self.TechDocConverter(
            heading_style="ATX",                 # ### 标题更像技术文档
            bullets="-",                         # 列表风格统一
            code_language_callback=self.extract_code_lang,
            # 技术文档里下划线/星号很常见（layer_norm, a*b），避免过度转义影响可读性
            escape_underscores=False,
            escape_asterisks=False,
            # 如果你经常碰到表格没有 thead/th，可以打开
            table_infer_header=True,
            # 一般技术博客不需要自动换行重排
            wrap=False,
            # 过滤某些标签（可选）
            strip=["meta", "link"],
        ).convert(html)

    async def ask(self, prompt: str, timeout_ms: int = 1000 * 60 * 5) -> str:
        page = self.page

        input_box = page.locator(self.input_selector)
        assistant_bubbles = page.locator(self.assistant_bubble_selector)

        # 2) 填入并回车发送
        await input_box.click()
        await input_box.fill(prompt)
        await page.locator("#send-message-button").click()

        await page.locator('#send-message-button').wait_for(state="attached", timeout=timeout_ms)

        last = assistant_bubbles.nth(-1)

        answer = await last.locator("> div").first.evaluate("""(node) => {
            // 深拷贝节点，防止影响页面实际显示
            const clone = node.cloneNode(true);
            
            // 定义需要移除的“思考过程”相关的的选择器
            // .thinking-chain-container -> 思考过程的标题栏
            // .thinking-block -> 思考过程的具体内容区域
            // .w-full.overflow-hidden.h-0 -> 有时思考内容被包裹在这个隐藏容器里
            const selectorsToRemove = [
                '.thinking-chain-container', 
                '.thinking-block',
                '.w-full.overflow-hidden.h-0', // 针对你代码片段中折叠区域的容器
                '.cursor-default'
            ];

            selectorsToRemove.forEach(selector => {
                const elements = clone.querySelectorAll(selector);
                elements.forEach(el => el.remove());
            });

            // 返回清洗后的文本 (innerText) 或 HTML (innerHTML)
            return clone.innerHTML.trim();
        }""")
        
        answer = self.html_to_markdown(answer)

        return answer

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        context = browser.contexts[0]
        page = context.pages[0]

        if not page.url.startswith("https://chat.z.ai"):
            await page.goto("https://chat.z.ai")

        # 你需要把下面两个 selector 换成真实的：
        # - 输入框（textarea / contenteditable div）
        # - 对方气泡（只匹配“对方/assistant”的消息，不要把自己发的也算进去）
        bot = ChatBot(
            page,
            input_selector="#chat-input",
            assistant_bubble_selector="#response-content-container"
        )

        a1 = await bot.ask("使用python打印“你好世界”，将代码放在代码块中。我在测试。")
        print("A1:", a1)

        # a1 = await bot.ask("你好，介绍一下你自己，用markdown格式，标题、无序符号等。")
        # print("A1:", a1)

        # a2 = await bot.ask("用三句话总结刚才内容。")
        # print("A2:", a2)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())