import asyncio
import logging
import os
from typing import List, Dict, Any
from playwright.async_api import async_playwright
from .base_collector import BaseCollector


class PlaywrightScraper(BaseCollector):
    def __init__(self, output_dir: str, timeout: int = 15000):
        super().__init__(output_dir)
        self.timeout = timeout  # in ms

    async def _scrape_url(self, url: str) -> Dict[str, Any]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            try:
                await page.goto(url, timeout=self.timeout)
                content = await page.content()
                text = await page.inner_text("body")
                metadata = {
                    "source": url,
                    "timestamp": asyncio.get_event_loop().time(),
                    "status": "success",
                }
                await browser.close()
                return {"text": text, "metadata": metadata}
            except Exception as e:
                await browser.close()
                self.logger.error(f"Error scraping {url} with Playwright: {e}")
                return {
                    "text": "",
                    "metadata": {"source": url, "error": str(e), "status": "failed"},
                }

    def collect(self, urls: List[str]) -> None:
        asyncio.run(self._collect_async(urls))

    async def _collect_async(self, urls: List[str]) -> None:
        for url in urls:
            self.logger.info(f"Playwright scraping: {url}")
            result = await self._scrape_url(url)
            filename = self._sanitize_filename(url) + ".json"
            self.save_data(result["text"], result["metadata"], filename)

    def _sanitize_filename(self, url: str) -> str:
        import re

        return re.sub(r"[^a-zA-Z0-9]", "_", url)

    def resume(self, urls: List[str]) -> None:
        scraped = set()
        for fname in os.listdir(self.output_dir):
            if fname.endswith(".json"):
                scraped.add(fname.replace(".json", ""))
        to_scrape = [url for url in urls if self._sanitize_filename(url) not in scraped]
        self.logger.info(f"Resuming Playwright scraping. {len(to_scrape)} URLs left.")
        self.collect(to_scrape)
