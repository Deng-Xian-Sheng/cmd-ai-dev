import asyncio
import io
import os
os.environ["NODE_NO_WARNINGS"] = "1"
from playwright.async_api import async_playwright, Locator, Page, TimeoutError as PlaywrightTimeoutError
from markitdown import MarkItDown

class ChatBot:
    def __init__(self, page: Page, *,
                 input_selector: str,
                 assistant_bubble_selector: str):
        self.page = page
        self.input_selector = input_selector
        self.assistant_bubble_selector = assistant_bubble_selector
        self.md = MarkItDown(enable_plugins=False) # Set to True to enable plugins

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
            return clone.innerHTML;
        }""")
        answer = self.md.convert(io.BytesIO(f"<!doctype html><html><body>{answer}</body></html>".encode())).text_content.replace("<time\\_out>", "<time_out>").replace("</time\\_out>", "</time_out>")
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

        a1 = await bot.ask("你好，介绍一下你自己，用markdown格式，标题、无序符号等。")
        print("A1:", a1)

        a2 = await bot.ask("用三句话总结刚才内容。")
        print("A2:", a2)

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())