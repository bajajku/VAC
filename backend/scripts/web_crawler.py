import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import time
from typing import Set, List, Dict
import logging
import json
from datetime import datetime
import os

class WebCrawler:
    def __init__(self, start_url: str, max_pages: int = 10, delay: float = 0.5):
        """
        Initialize the web crawler
        
        Args:
            start_url (str): The starting URL to crawl
            max_pages (int): Maximum number of pages to crawl
            delay (float): Delay between requests in seconds
        """
        self.start_url = start_url
        self.max_pages = max_pages
        self.delay = delay
        self.visited_urls: Set[str] = set()
        self.to_visit: List[str] = [start_url]
        self.base_domain = urlparse(start_url).netloc
        self.scraped_data: Dict[str, Dict] = {}
        
        # Create output directory
        self.output_dir = "crawl_results"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def is_valid_url(self, url: str) -> bool:
        """Check if the URL is valid and belongs to the same domain"""
        try:
            parsed = urlparse(url)
            return bool(parsed.netloc) and parsed.netloc == self.base_domain
        except:
            return False

    def get_page_content(self, url: str) -> tuple:
        """Fetch and return the content and metadata of a webpage"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response.text, {
                'status_code': response.status_code,
                'content_type': response.headers.get('content-type', ''),
                'last_modified': response.headers.get('last-modified', ''),
                'crawl_time': datetime.now().isoformat()
            }
        except requests.RequestException as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return "", {}

    def extract_links(self, html_content: str, base_url: str) -> Set[str]:
        """Extract all links from the HTML content"""
        soup = BeautifulSoup(html_content, 'html.parser')
        links = set()
        
        # Find the main tag
        main_tag = soup.find('main')
        if main_tag:
            # Only extract links from within the main tag
            for anchor in main_tag.find_all('a', href=True):
                href = anchor['href']
                absolute_url = urljoin(base_url, href)
                if self.is_valid_url(absolute_url):
                    links.add(absolute_url)
        
        return links

    def extract_page_info(self, html_content: str, url: str) -> Dict:
        """Extract useful information from the page"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Extract title
        title = soup.title.string if soup.title else ''
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc['content'] if meta_desc else ''
        
        # Extract all text content
        text_content = ' '.join(soup.stripped_strings)
        
        # Extract all images
        images = [img.get('src', '') for img in soup.find_all('img')]
        
        return {
            'title': title,
            'description': description,
            'text_content': text_content,
            'images': images,
            'url': url
        }

    def save_results(self):
        """Save the scraped data to JSON files"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save all scraped data
        all_data_file = os.path.join(self.output_dir, f'crawl_results_{timestamp}.json')
        with open(all_data_file, 'w', encoding='utf-8') as f:
            json.dump(self.scraped_data, f, indent=2, ensure_ascii=False)
        
        # Save just the links
        links_file = os.path.join(self.output_dir, f'links_{timestamp}.txt')
        with open(links_file, 'w', encoding='utf-8') as f:
            for url in sorted(self.visited_urls):
                f.write(f"{url}\n")
        
        self.logger.info(f"Results saved to {self.output_dir}")

    def crawl(self):
        """Start the crawling process"""
        self.logger.info(f"Starting crawl from {self.start_url}")
        
        while self.to_visit and len(self.visited_urls) < self.max_pages:
            current_url = self.to_visit.pop(0)
            
            if current_url in self.visited_urls:
                continue
                
            self.logger.info(f"Crawling: {current_url}")
            
            # Get page content and metadata
            content, metadata = self.get_page_content(current_url)
            if not content:
                continue
                
            # Extract page information
            page_info = self.extract_page_info(content, current_url)
            
            # Combine all information
            self.scraped_data[current_url] = {
                **metadata,
                **page_info
            }
            
            # Mark URL as visited
            self.visited_urls.add(current_url)
            
            # Extract and add new links
            new_links = self.extract_links(content, current_url)
            self.to_visit.extend([link for link in new_links if link not in self.visited_urls])
            
            # Respect the delay between requests
            time.sleep(self.delay)
        
        # Save results after crawling is complete
        self.save_results()
        self.logger.info(f"Crawling completed. Visited {len(self.visited_urls)} pages.")

def main():
    # Example usage
    start_url = "https://www.canada.ca/en/department-national-defence/services/benefits-military/health-support/sexual-misconduct-response.html"  # Replace with your target website
    crawler = WebCrawler(start_url, max_pages=100, delay=1.0)
    crawler.crawl()

if __name__ == "__main__":
    main() 