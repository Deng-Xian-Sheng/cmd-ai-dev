from playwright.async_api import async_playwright, expect, TimeoutError, Page, Locator
import asyncio

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp("http://127.0.0.1:9222")
        context = browser.contexts[0]
        page:Page = context.pages[0]

        await page.locator("#chat-input").fill("你好？")
        await page.locator("#chat-input").press('Enter')

        await expect(page.locator('div[aria-label="复制"]').nth(3)).to_be_visible(timeout=1000 * 60 * 5)

        print(await page.locator("#response-content-container").nth(3).inner_text())

if __name__ == "__main__":
    asyncio.run(main())