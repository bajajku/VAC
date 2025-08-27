import requests
from bs4 import BeautifulSoup
from typing import List, Dict
import json
from datetime import datetime
import os
import logging
import time
import io
from PyPDF2 import PdfReader


class WebScraper:
    def __init__(self, urls: List[str], delay: float = 0.5):
        """
        Initialize the web scraper.
        
        Args:
            urls (List[str]): List of URLs to scrape
            delay (float): Delay between requests in seconds
        """
        self.urls = urls
        self.delay = delay
        self.scraped_data: Dict[str, Dict] = {}

        # Create output directory
        self.output_dir = "scrape_results"
        os.makedirs(self.output_dir, exist_ok=True)

        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def get_page_content(self, url: str) -> tuple:
        """Fetch and return the content and metadata of a webpage or PDF"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.content, {
                "status_code": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "last_modified": response.headers.get("last-modified", ""),
                "scrape_time": datetime.now().isoformat()
            }
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {e}")
            return b"", {}

    def extract_page_info(self, content: bytes, url: str, content_type: str) -> Dict:
        """Extract useful information from the page or PDF"""

        # Handle PDFs separately
        if "pdf" in content_type or url.lower().endswith(".pdf"):
            try:
                pdf_text = []
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    pdf_text.append(page.extract_text() or "")
                text_content = "\n".join(pdf_text)
            except Exception as e:
                self.logger.error(f"Error parsing PDF {url}: {e}")
                text_content = ""

            return {
                "title": os.path.basename(url),
                "description": "PDF document",
                "text_content": text_content,
                "images": [],
                "url": url,
                "type": "pdf"
            }

        # Otherwise, treat as HTML
        soup = BeautifulSoup(content, "html.parser")

        title = soup.title.string.strip() if soup.title else ""
        meta_desc = soup.find("meta", attrs={"name": "description"})
        description = meta_desc["content"].strip() if meta_desc else ""
        text_content = " ".join(soup.stripped_strings)
        images = [img.get("src", "") for img in soup.find_all("img")]

        return {
            "title": title,
            "description": description,
            "text_content": text_content,
            "images": images,
            "url": url,
            "type": "html"
        }

    def save_results(self):
        """Save the scraped data to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.output_dir, f"scrape_results_{timestamp}.json")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)

        self.logger.info(f"Results saved to {file_path}")

    def scrape(self):
        """Run scraper on all given URLs"""
        for url in self.urls:
            self.logger.info(f"Scraping: {url}")
            content, metadata = self.get_page_content(url)
            if not content:
                continue

            page_info = self.extract_page_info(content, url, metadata.get("content_type", ""))
            self.scraped_data[url] = {**metadata, **page_info}

            time.sleep(self.delay)  # polite delay

        self.save_results()
        self.logger.info(f"Scraping completed. Scraped {len(self.scraped_data)} pages.")


def main():
    urls = [
        "https://www.canada.ca/en/department-national-defence/services/benefits-military/health-support/sexual-misconduct-response/peer-support-program/about-peer-support-program.html",
        "https://www.canada.ca/en/department-national-defence/corporate/reports-publications/mst-report.html",
        "https://www.veterans.gc.ca/en/about-vac/public-engagement/equity-deserving-groups/women-veterans/military-sexual-trauma",
        "https://atlasveterans.ca/documents/mst/sept16-2021/sept16-greenhorn-tuka-en.pdf",
        "https://atlasveterans.ca",
        "https://www.victimservicestoronto.com",
        "http://trccmwar.ca",
        "http://sherbourne.on.ca/counselling-services/",
        "https://togetherall.com/en-ca/join/togetherall-faqs-for-the-canadian-mst-community/"
    ]

    scraper = WebScraper(urls, delay=1.0)  # 1 sec delay
    scraper.scrape()


if __name__ == "__main__":
    main()