import scrapy
from collections import deque, defaultdict
from scrapy.crawler import CrawlerProcess
from scrapy.utils.log import configure_logging
import logging

SUPPORTED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]
BLACKLIST = ["example.com", "bannedsite.com"]
IGNORE_EXT = [
    ".css", ".js", ".ico", ".svg", ".woff", ".woff2", ".ttf", ".pdf", 
    ".zip", ".exe", ".mp4", ".mp3", ".avi", ".mov"
]

class FastSpider(scrapy.Spider):
    name = "fast_spider"
    max_pages_per_site = 50
    concurrent_requests = 200
    batch_size = 100

    custom_settings = {
        "DOWNLOAD_TIMEOUT": 15,
        "RETRY_TIMES": 2,
        "ROBOTSTXT_OBEY": False,
        "LOG_LEVEL": "INFO",
        "CONCURRENT_REQUESTS": concurrent_requests,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 20,
        "HTTPERROR_ALLOWED_CODES": [403, 404, 500, 503],
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.2,
        "AUTOTHROTTLE_MAX_DELAY": 3.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 150,
        "DOWNLOAD_DELAY": 0,
        "COOKIES_ENABLED": False,
        "REDIRECT_ENABLED": True,
        "REDIRECT_MAX_TIMES": 3,
        # Memory optimization
        "MEMUSAGE_ENABLED": True,
        "MEMUSAGE_LIMIT_MB": 2048,
        # Disable unused middlewares
        "DOWNLOADER_MIDDLEWARES": {
            'scrapy.downloadermiddlewares.useragent.UserAgentMiddleware': None,
        }
    }

    def __init__(self, to_crawl=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visited_links = set()
        self.visited_domains = defaultdict(int)
        self.to_crawl = to_crawl if to_crawl else deque()
        self.batch = []
        self.pages_crawled = 0

    def closed(self, reason):
        # Submit remaining batch
        if self.batch:
            self._submit_batch()
        self.logger.info(f"Crawler finished. Total pages: {self.pages_crawled}")

    def start_requests(self):
        for url in self.to_crawl:
            try:
                domain = url.split("/")[2]
                if url not in self.visited_links and domain not in BLACKLIST:
                    self.visited_links.add(url)
                    yield scrapy.Request(
                        url, 
                        callback=self.parse, 
                        errback=self.handle_error, 
                        dont_filter=True,
                        priority=1  # Higher priority for seed URLs
                    )
            except Exception as e:
                self.logger.warning(f"Invalid URL: {url} - {e}")

    def handle_error(self, failure):
        self.logger.debug(f"Error on {failure.request.url}: {failure.value}")

    def parse(self, response):
        try:
            url = response.url
            domain = url.split("/")[2]

            # Check content type
            content_type = response.headers.get("Content-Type", b"").decode("utf-8", errors="ignore")
            if "text/html" not in content_type.lower():
                return

            # Check domain limit
            if self.visited_domains[domain] >= self.max_pages_per_site:
                return

            self.visited_domains[domain] += 1
            self.pages_crawled += 1

            # Extract data
            title = response.css("title::text").get() or ""
            title = title.strip()[:500]  # Limit title length
            
            description = response.css("meta[name='description']::attr(content)").get() or \
                         response.css("meta[property='og:description']::attr(content)").get() or ""
            description = description.strip()[:1000]  # Limit description length

            # Extract images
            raw_images = response.css("img::attr(src)").getall()
            image_links = []
            for img in raw_images[:10]:  # Limit to 10 images per page
                if any(img.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    full_url = response.urljoin(img)
                    if full_url.startswith('http'):
                        image_links.append(full_url)

            # Add to batch
            self.batch.append((url, title, description, image_links))

            # Submit batch if full
            if len(self.batch) >= self.batch_size:
                self._submit_batch()

            # Follow links
            for link in response.css("a::attr(href)").getall():
                try:
                    link = response.urljoin(link)
                    
                    if not link.startswith("http"):
                        continue
                    
                    # Skip files
                    if any(link.lower().endswith(ext) for ext in IGNORE_EXT):
                        continue
                    
                    # Skip anchors and queries for same page
                    if link.split('#')[0].split('?')[0] == url.split('#')[0].split('?')[0]:
                        continue
                    
                    link_domain = link.split("/")[2]
                    
                    # Check if should follow
                    if (link not in self.visited_links and 
                        link_domain not in BLACKLIST and 
                        self.visited_domains[link_domain] < self.max_pages_per_site):
                        
                        self.visited_links.add(link)
                        yield scrapy.Request(
                            link, 
                            callback=self.parse, 
                            errback=self.handle_error, 
                            dont_filter=True,
                            priority=0
                        )
                except Exception as e:
                    self.logger.debug(f"Error processing link {link}: {e}")

        except Exception as e:
            self.logger.error(f"Error parsing {response.url}: {e}")

    def _submit_batch(self):
        """Submit batch to indexer"""
        try:
            from indexer_optimized import submit_batch_to_indexer
            submit_batch_to_indexer(self.batch)
            self.logger.info(f"Submitted batch of {len(self.batch)} pages")
            self.batch = []
        except Exception as e:
            self.logger.error(f"Error submitting batch: {e}")
            self.batch = []

def run_crawler(url_list, log_level="INFO"):
    """Run the crawler with given URLs"""
    configure_logging(install_root_handler=False)
    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=log_level
    )
    
    process = CrawlerProcess()
    process.crawl(FastSpider, to_crawl=deque(url_list))
    process.start()

if __name__ == "__main__":
    # Test with a few sites
    test_urls = [
        "https://www.wikipedia.org",
        "https://www.github.com",
        "https://www.stackoverflow.com"
    ]
    run_crawler(test_urls)