import logging
import requests
from bs4 import BeautifulSoup
import time
import random
from typing import List, Dict, Any, Optional, Set
import pandas as pd
from urllib.parse import urljoin, urlparse
import re
from pathlib import Path
import json
import hashlib
from datetime import datetime, timezone


# Import error handling utilities
from src.utils.error_handling import (
    retry_with_fallback,
    circuit_breaker,
    robust_web_request,
)
from src.utils.error_classification import (
    classify_and_handle_error,
    get_error_recovery_strategy,
    should_retry_operation,
    get_retry_delay_for_error,
)

# Import configuration manager
from src.utils.config_manager import get_config

# Import memory management
from src.utils.memory_manager import memory_manager, memory_safe, chunked_processing

# Import metadata handler
from .metadata_handler import MetadataHandler, Metadata

# Import Selenium for dynamic site handling
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    from selenium.webdriver.common.action_chains import ActionChains
    from selenium.common.exceptions import (
        TimeoutException,
        NoSuchElementException,
        WebDriverException,
    )

    SELENIUM_AVAILABLE = True
    WebDriverType = webdriver.Remote
except ImportError:
    SELENIUM_AVAILABLE = False
    WebDriverType = Any  # Type alias for when Selenium is not available
    logging.warning("Selenium not available. Install with: pip install selenium")


class DynamicSiteHandler:
    """Handles dynamic site interactions including AJAX, infinite scroll, and cookie consent."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Dynamic site handling settings
        self.wait_timeout = config.get("wait_timeout", 10)
        self.scroll_pause_time = config.get("scroll_pause_time", 2)
        self.max_scroll_attempts = config.get("max_scroll_attempts", 10)
        self.scroll_height_threshold = config.get("scroll_height_threshold", 100)

        # Cookie consent selectors
        self.cookie_selectors = config.get(
            "cookie_selectors",
            [
                "button[contains(text(), 'Accept')]",
                "button[contains(text(), 'Accept All')]",
                "button[contains(text(), 'Allow')]",
                "button[contains(text(), 'OK')]",
                "button[contains(text(), 'Got it')]",
                "button[contains(text(), 'I agree')]",
                "button[contains(text(), 'Continue')]",
                ".cookie-accept",
                ".cookie-consent button",
                "#cookie-accept",
                "#cookie-consent button",
                "[data-testid='cookie-accept']",
                "[data-testid='cookie-consent'] button",
            ],
        )

        # Loading indicators
        self.loading_selectors = config.get(
            "loading_selectors",
            [
                ".loading",
                ".spinner",
                ".loader",
                "[data-loading='true']",
                ".ajax-loading",
                ".content-loading",
            ],
        )

        # Infinite scroll selectors
        self.scroll_selectors = config.get(
            "scroll_selectors",
            [
                ".load-more",
                ".show-more",
                ".pagination",
                ".infinite-scroll",
                "[data-load-more]",
                ".next-page",
            ],
        )

    def setup_driver(
        self, headless: bool = True, browser: str = "chrome"
    ) -> Optional[WebDriverType]:
        """
        Set up Selenium WebDriver with appropriate options.

        Args:
            headless: Whether to run in headless mode
            browser: Browser type ('chrome' or 'firefox')

        Returns:
            Configured WebDriver instance or None if Selenium is not available
        """
        if not SELENIUM_AVAILABLE:
            self.logger.warning("Selenium not available, cannot set up WebDriver")
            return None

        try:
            if browser.lower() == "chrome":
                options = ChromeOptions()
                if headless:
                    options.add_argument("--headless")
                options.add_argument("--no-sandbox")
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--disable-gpu")
                options.add_argument("--window-size=1920,1080")
                options.add_argument(
                    "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                )

                driver = webdriver.Chrome(options=options)

            elif browser.lower() == "firefox":
                options = FirefoxOptions()
                if headless:
                    options.add_argument("--headless")
                options.add_argument("--width=1920")
                options.add_argument("--height=1080")

                driver = webdriver.Firefox(options=options)

            else:
                self.logger.error(f"Unsupported browser: {browser}")
                return None

            # Set page load timeout
            driver.set_page_load_timeout(30)
            driver.implicitly_wait(5)

            self.logger.info(f"WebDriver initialized for {browser}")
            return driver

        except Exception as e:
            self.logger.error(f"Error setting up WebDriver: {e}")
            return None

    def handle_cookie_consent(self, driver: WebDriverType) -> bool:
        """
        Handle cookie consent popups and banners.

        Args:
            driver: WebDriver instance

        Returns:
            True if cookie consent was handled, False otherwise
        """
        try:
            # Wait for page to load
            WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            # Try different cookie consent selectors
            for selector in self.cookie_selectors:
                try:
                    # Wait for cookie consent element to be present
                    cookie_element = WebDriverWait(driver, 3).until(
                        EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
                    )

                    # Click the cookie consent button
                    cookie_element.click()
                    self.logger.info(f"Clicked cookie consent: {selector}")

                    # Wait a moment for the popup to disappear
                    time.sleep(1)
                    return True

                except TimeoutException:
                    continue
                except Exception as e:
                    self.logger.debug(f"Error with cookie selector {selector}: {e}")
                    continue

            self.logger.debug("No cookie consent found or already accepted")
            return False

        except Exception as e:
            self.logger.error(f"Error handling cookie consent: {e}")
            return False

    def wait_for_ajax(
        self, driver: WebDriverType, timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for AJAX requests to complete.

        Args:
            driver: WebDriver instance
            timeout: Timeout in seconds (uses default if None)

        Returns:
            True if AJAX completed, False if timeout
        """
        if timeout is None:
            timeout = self.wait_timeout

        try:
            # Wait for jQuery to complete (if present)
            jquery_complete = driver.execute_script(
                "return typeof jQuery === 'undefined' || jQuery.active === 0"
            )

            # Wait for document ready state
            WebDriverWait(driver, timeout).until(
                lambda d: d.execute_script("return document.readyState") == "complete"
            )

            # Wait for loading indicators to disappear
            for selector in self.loading_selectors:
                try:
                    WebDriverWait(driver, 2).until_not(
                        EC.presence_of_element_located((By.CSS_SELECTOR, selector))
                    )
                except TimeoutException:
                    pass  # Loading indicator might not be present

            self.logger.debug("AJAX requests completed")
            return True

        except TimeoutException:
            self.logger.warning(f"AJAX wait timeout after {timeout} seconds")
            return False
        except Exception as e:
            self.logger.error(f"Error waiting for AJAX: {e}")
            return False

    def handle_infinite_scroll(
        self, driver: WebDriverType, max_attempts: Optional[int] = None
    ) -> int:
        """
        Handle infinite scroll by scrolling to the bottom until no new content loads.

        Args:
            driver: WebDriver instance
            max_attempts: Maximum scroll attempts (uses default if None)

        Returns:
            Number of scroll attempts made
        """
        if max_attempts is None:
            max_attempts = self.max_scroll_attempts

        scroll_attempts = 0
        last_height = driver.execute_script("return document.body.scrollHeight")

        try:
            while scroll_attempts < max_attempts:
                # Scroll to bottom
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

                # Wait for content to load
                time.sleep(self.scroll_pause_time)

                # Wait for AJAX to complete
                self.wait_for_ajax(driver, timeout=5)

                # Calculate new height
                new_height = driver.execute_script("return document.body.scrollHeight")

                # Check if page extended
                if new_height == last_height:
                    # Try clicking "Load More" buttons if present
                    if not self._click_load_more_buttons(driver):
                        break
                    else:
                        time.sleep(self.scroll_pause_time)
                        new_height = driver.execute_script(
                            "return document.body.scrollHeight"
                        )
                        if new_height == last_height:
                            break

                last_height = new_height
                scroll_attempts += 1

                self.logger.debug(
                    f"Scroll attempt {scroll_attempts}: height {new_height}"
                )

            self.logger.info(
                f"Infinite scroll completed after {scroll_attempts} attempts"
            )
            return scroll_attempts

        except Exception as e:
            self.logger.error(f"Error during infinite scroll: {e}")
            return scroll_attempts

    def _click_load_more_buttons(self, driver: WebDriverType) -> bool:
        """
        Click "Load More" buttons if present.

        Args:
            driver: WebDriver instance

        Returns:
            True if a button was clicked, False otherwise
        """
        try:
            for selector in self.scroll_selectors:
                try:
                    # Look for clickable load more buttons
                    buttons = driver.find_elements(By.CSS_SELECTOR, selector)

                    for button in buttons:
                        if button.is_displayed() and button.is_enabled():
                            # Scroll to button
                            driver.execute_script(
                                "arguments[0].scrollIntoView();", button
                            )
                            time.sleep(0.5)

                            # Click button
                            button.click()
                            self.logger.debug(f"Clicked load more button: {selector}")
                            return True

                except Exception as e:
                    self.logger.debug(f"Error with load more selector {selector}: {e}")
                    continue

            return False

        except Exception as e:
            self.logger.error(f"Error clicking load more buttons: {e}")
            return False

    def wait_for_element(
        self, driver: WebDriverType, selector: str, timeout: Optional[int] = None
    ) -> bool:
        """
        Wait for a specific element to be present and visible.

        Args:
            driver: WebDriver instance
            selector: CSS selector for the element
            timeout: Timeout in seconds (uses default if None)

        Returns:
            True if element found, False if timeout
        """
        if timeout is None:
            timeout = self.wait_timeout

        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            return True

        except TimeoutException:
            self.logger.warning(f"Element not found: {selector}")
            return False
        except Exception as e:
            self.logger.error(f"Error waiting for element {selector}: {e}")
            return False

    def extract_dynamic_content(
        self, driver: WebDriverType, url: str
    ) -> Dict[str, Any]:
        """
        Extract content from a dynamic page using Selenium.

        Args:
            driver: WebDriver instance
            url: URL to scrape

        Returns:
            Dictionary containing extracted content
        """
        try:
            # Navigate to URL
            driver.get(url)

            # Handle cookie consent
            self.handle_cookie_consent(driver)

            # Wait for main content to load
            main_selectors = [
                "main",
                "article",
                ".content",
                "#content",
                ".main-content",
            ]
            content_loaded = False

            for selector in main_selectors:
                if self.wait_for_element(driver, selector, timeout=5):
                    content_loaded = True
                    break

            if not content_loaded:
                # Wait for any content to load
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )

            # Wait for AJAX to complete
            self.wait_for_ajax(driver)

            # Handle infinite scroll
            scroll_attempts = self.handle_infinite_scroll(driver)

            # Extract page content
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

            # Get page metadata
            title = driver.title
            current_url = driver.current_url

            # Extract dynamic content
            content = {
                "url": current_url,
                "title": title,
                "content": self._extract_text_content(soup),
                "links": self._extract_links(soup, current_url),
                "images": self._extract_images(soup, current_url),
                "metadata": {
                    "scroll_attempts": scroll_attempts,
                    "page_height": driver.execute_script(
                        "return document.body.scrollHeight"
                    ),
                    "viewport_height": driver.execute_script(
                        "return window.innerHeight"
                    ),
                    "dynamic_content_loaded": True,
                },
            }

            return content

        except Exception as e:
            self.logger.error(f"Error extracting dynamic content from {url}: {e}")
            return {"url": url, "error": str(e), "dynamic_content_loaded": False}

    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup object."""
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()

        # Get text content
        text = soup.get_text()

        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = " ".join(chunk for chunk in chunks if chunk)

        return text

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from BeautifulSoup object."""
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)
        return links

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from BeautifulSoup object."""
        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            absolute_url = urljoin(base_url, src)
            images.append(absolute_url)
        return images


class WebScraper:
    def __init__(
        self,
        user_agent: str = "energyAI-bot",
        base_url: Optional[str] = None,
        max_pages: Optional[int] = None,
    ):
        """
        Initialize web scraper with metadata integration and source-specific extractors

        Args:
            user_agent: User agent string for requests
            base_url: Base URL for scraping (if None, uses config)
            max_pages: Maximum number of pages to scrape (if None, uses config)
        """
        self.logger = logging.getLogger(self.__class__.__name__)

        # Load configuration
        self.config = get_config()
        scraper_config = self.config.get_web_scraping_config()

        # Set parameters from config or arguments
        self.user_agent = user_agent
        self.base_url = base_url or scraper_config.get("base_url", "")
        self.max_pages = max_pages or scraper_config.get("max_pages", 100)

        # Load scraping settings
        self.delay_range = scraper_config.get("delay_range", [1, 3])
        self.timeout = scraper_config.get("timeout", 30)
        self.max_retries = scraper_config.get("max_retries", 3)
        self.user_agents = scraper_config.get(
            "user_agents",
            [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            ],
        )

        # Load content extraction settings
        self.content_config = scraper_config.get("content_extraction", {})
        self.extract_title = self.content_config.get("extract_title", True)
        self.extract_meta = self.content_config.get("extract_meta", True)
        self.extract_links = self.content_config.get("extract_links", True)
        self.extract_images = self.content_config.get("extract_images", False)
        self.clean_html = self.content_config.get("clean_html", True)
        self.min_content_length = self.content_config.get("min_content_length", 100)

        # Load memory management settings
        self.memory_config = scraper_config.get("memory_management", {})
        self.chunk_size = self.memory_config.get("chunk_size", 50)
        self.save_intermediate = self.memory_config.get("save_intermediate", True)
        self.intermediate_dir = Path(
            self.memory_config.get("intermediate_dir", "temp_scraped_data")
        )

        # Load URL filtering settings
        self.url_config = scraper_config.get("url_filtering", {})
        self.allowed_domains = set(self.url_config.get("allowed_domains", []))
        self.excluded_patterns = self.url_config.get("excluded_patterns", [])
        self.required_patterns = self.url_config.get("required_patterns", [])

        # Initialize session with retry mechanism
        self.session = self._create_session()

        # Track scraped URLs to avoid duplicates
        self.scraped_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()

        # Initialize metadata handler
        self.metadata_handler = MetadataHandler()

        # Initialize source-specific extractors
        self.extractors = {
            "generic": GenericWebExtractor(),
            "news": NewsWebExtractor(),
            "blog": BlogWebExtractor(),
            "documentation": DocumentationWebExtractor(),
            "ecommerce": EcommerceWebExtractor(),
            "government": GovernmentWebExtractor(),
            "academic": AcademicWebExtractor(),
        }

        # Initialize dynamic site handler
        dynamic_config = scraper_config.get("dynamic_site_handling", {})
        self.dynamic_handler = DynamicSiteHandler(dynamic_config)

        # Create intermediate directory
        if self.save_intermediate:
            self.intermediate_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(
            f"Web scraper initialized with user_agent: {user_agent}, base_url: {self.base_url}"
        )

    def _create_session(self) -> requests.Session:
        """Create requests session with proper headers and retry mechanism"""
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
        )
        return session

    @memory_safe
    def scrape(
        self, url: str, extractor_type: str = "generic", use_dynamic: bool = None
    ) -> Dict[str, Any]:
        """
        Scrape a single URL with intelligent dynamic/static detection, comprehensive error handling,
        and intelligent fallback strategies.

        Args:
            url: URL to scrape
            extractor_type: Type of extractor to use ('generic', 'news', 'blog', etc.)
            use_dynamic: Force dynamic scraping if True, static if False, auto-detect if None

        Returns:
            Dictionary containing scraped data with metadata and error information
        """
        if url in self.scraped_urls:
            self.logger.debug(f"URL already scraped: {url}")
            return None

        if url in self.failed_urls:
            self.logger.debug(f"URL previously failed: {url}")
            return None

        # Track retry attempts and errors
        retry_count = 0
        max_retries = self.max_retries
        last_error_info = None

        try:
            # Check memory before scraping
            if memory_manager.is_memory_critical():
                self.logger.warning(
                    f"Memory critical before scraping {url}, optimizing..."
                )
                memory_manager.optimize_memory()

            # Determine scraping method
            if use_dynamic is None:
                # Auto-detect: try static first, then dynamic if needed
                scraped_data = self._scrape_static(url, extractor_type)
                if (
                    scraped_data
                    and len(scraped_data.get("content", "")) > self.min_content_length
                ):
                    return scraped_data
                else:
                    # Static scraping didn't get enough content, try dynamic
                    self.logger.info(
                        f"Static scraping insufficient for {url}, trying dynamic"
                    )
                    return self._scrape_dynamic(url, extractor_type)
            elif use_dynamic:
                # Force dynamic scraping
                return self._scrape_dynamic(url, extractor_type)
            else:
                # Force static scraping
                return self._scrape_static(url, extractor_type)

        except Exception as e:
            # Classify the error
            error_info = classify_and_handle_error(e, {"url": url})
            last_error_info = error_info

            # Log detailed error information
            self.logger.error(
                f"Error scraping {url}: {error_info.category.value} ({error_info.severity.value}) - {error_info.message}"
            )

            # Check if we should retry
            if should_retry_operation(error_info, retry_count, max_retries):
                retry_count += 1
                delay = get_retry_delay_for_error(error_info, retry_count)

                self.logger.info(
                    f"Retrying {url} in {delay}s (attempt {retry_count}/{max_retries})"
                )
                time.sleep(delay)

                # Try fallback strategy
                return self._try_fallback_strategy(
                    url, extractor_type, error_info, use_dynamic
                )
            else:
                # Don't retry, mark as failed
                self.failed_urls.add(url)
                return self._create_error_result(url, error_info)
        finally:
            # Memory optimization after each scrape
            memory_manager.force_garbage_collection()

    def _try_fallback_strategy(
        self, url: str, extractor_type: str, error_info, use_dynamic: bool = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try fallback strategies based on error classification.

        Args:
            url: URL to scrape
            extractor_type: Type of extractor to use
            error_info: Classified error information
            use_dynamic: Original dynamic setting

        Returns:
            Scraped data or error result
        """
        strategy = get_error_recovery_strategy(error_info)

        if not strategy:
            return self._create_error_result(url, error_info)

        self.logger.info(f"Trying fallback strategy: {strategy}")

        try:
            if strategy == "use_static_scraping":
                # Fallback to static scraping
                return self._scrape_static(url, extractor_type)

            elif strategy == "use_dynamic_scraping":
                # Fallback to dynamic scraping
                return self._scrape_dynamic(url, extractor_type)

            elif strategy == "change_user_agent":
                # Try with different user agent
                original_agent = self.user_agent
                self.user_agent = random.choice(self.user_agents)
                self.session.headers.update({"User-Agent": self.user_agent})

                try:
                    result = self._scrape_static(url, extractor_type)
                    if result:
                        return result
                finally:
                    # Restore original user agent
                    self.user_agent = original_agent
                    self.session.headers.update({"User-Agent": self.user_agent})

            elif strategy == "increase_timeout":
                # Try with increased timeout
                original_timeout = self.timeout
                self.timeout = min(
                    original_timeout * 2, 120
                )  # Double timeout, max 120s

                try:
                    result = self._scrape_static(url, extractor_type)
                    if result:
                        return result
                finally:
                    self.timeout = original_timeout

            elif strategy == "add_headers":
                # Try with additional headers
                original_headers = dict(self.session.headers)
                additional_headers = {
                    "Accept-Language": "en-US,en;q=0.9",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache",
                }
                self.session.headers.update(additional_headers)

                try:
                    result = self._scrape_static(url, extractor_type)
                    if result:
                        return result
                finally:
                    self.session.headers.update(original_headers)

            elif strategy == "use_session":
                # Create new session
                self.session = self._create_session()
                return self._scrape_static(url, extractor_type)

            elif strategy == "try_different_parser":
                # Try with different parser
                return self._scrape_with_alternative_parser(url, extractor_type)

            elif strategy == "extract_raw_content":
                # Extract raw content without parsing
                return self._extract_raw_content(url)

            else:
                # Unknown strategy, try static scraping as last resort
                self.logger.warning(
                    f"Unknown fallback strategy: {strategy}, trying static scraping"
                )
                return self._scrape_static(url, extractor_type)

        except Exception as fallback_error:
            # Fallback also failed
            fallback_error_info = classify_and_handle_error(
                fallback_error, {"url": url}
            )
            self.logger.error(
                f"Fallback strategy {strategy} failed: {fallback_error_info.message}"
            )
            return self._create_error_result(url, fallback_error_info)

    def _scrape_dynamic(
        self, url: str, extractor_type: str = "generic"
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape dynamic content using Selenium with comprehensive AJAX waiting,
        infinite scroll handling, and cookie consent management.

        Args:
            url: URL to scrape
            extractor_type: Type of extractor to use

        Returns:
            Dictionary containing scraped data or None if failed
        """
        if not SELENIUM_AVAILABLE:
            self.logger.warning(
                "Selenium not available, falling back to static scraping"
            )
            return self._scrape_static(url, extractor_type)

        driver = None
        try:
            self.logger.info(f"Scraping dynamic content from: {url}")

            # Set up WebDriver
            driver = self.dynamic_handler.setup_driver(headless=True, browser="chrome")

            if not driver:
                self.logger.warning(
                    "Failed to set up WebDriver, falling back to static scraping"
                )
                return self._scrape_static(url, extractor_type)

            # Extract dynamic content
            dynamic_content = self.dynamic_handler.extract_dynamic_content(driver, url)

            if dynamic_content.get("error"):
                self.logger.error(
                    f"Dynamic extraction failed: {dynamic_content['error']}"
                )
                return None

            # Use appropriate extractor for content processing
            extractor = self.extractors.get(extractor_type, self.extractors["generic"])

            # Create BeautifulSoup object for extractor
            soup = BeautifulSoup(dynamic_content.get("content", ""), "html.parser")

            # Create mock response object for extractor compatibility
            class MockResponse:
                def __init__(self, url, text):
                    self.url = url
                    self.text = text
                    self.status_code = 200
                    self.headers = {}
                    self.content = text.encode("utf-8")

            mock_response = MockResponse(url, dynamic_content.get("content", ""))

            # Extract content using the appropriate extractor
            scraped_data = extractor.extract(soup, url, mock_response)

            if scraped_data:
                # Add dynamic content metadata
                scraped_data.update(
                    {
                        "dynamic_content_loaded": True,
                        "scroll_attempts": dynamic_content.get("metadata", {}).get(
                            "scroll_attempts", 0
                        ),
                        "page_height": dynamic_content.get("metadata", {}).get(
                            "page_height", 0
                        ),
                        "viewport_height": dynamic_content.get("metadata", {}).get(
                            "viewport_height", 0
                        ),
                        "extractor_type": extractor_type,
                    }
                )

                # Create metadata
                metadata = self._create_metadata(url, scraped_data, mock_response)
                scraped_data["metadata"] = metadata

                self.scraped_urls.add(url)
                self.logger.info(f"Successfully scraped dynamic content from: {url}")
            else:
                self.failed_urls.add(url)
                self.logger.warning(f"No content extracted from dynamic page: {url}")

            return scraped_data

        except Exception as e:
            self.logger.error(f"Error in dynamic scraping {url}: {e}")
            self.failed_urls.add(url)
            return None
        finally:
            # Clean up WebDriver
            if driver:
                try:
                    driver.quit()
                except Exception as e:
                    self.logger.debug(f"Error closing WebDriver: {e}")

            # Memory optimization
            memory_manager.force_garbage_collection()

    def _scrape_static(
        self, url: str, extractor_type: str = "generic"
    ) -> Optional[Dict[str, Any]]:
        """
        Scrape static content using requests and BeautifulSoup.

        Args:
            url: URL to scrape
            extractor_type: Type of extractor to use

        Returns:
            Dictionary containing scraped data or None if failed
        """
        try:
            self.logger.debug(f"Scraping static content from: {url}")

            # Make request with retry mechanism
            response = self.session.get(
                url, timeout=self.timeout, headers={"User-Agent": self.user_agent}
            )
            response.raise_for_status()

            if not response or response.status_code != 200:
                self.logger.warning(f"Failed to get response from {url}")
                return None

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Use appropriate extractor
            extractor = self.extractors.get(extractor_type, self.extractors["generic"])
            scraped_data = extractor.extract(soup, url, response)

            if scraped_data:
                scraped_data.update(
                    {"dynamic_content_loaded": False, "extractor_type": extractor_type}
                )

                # Create metadata
                metadata = self._create_metadata(url, scraped_data, response)
                scraped_data["metadata"] = metadata

                self.scraped_urls.add(url)
                self.logger.debug(f"Successfully scraped static content from: {url}")
            else:
                self.failed_urls.add(url)
                self.logger.debug(f"No content extracted from static page: {url}")

            return scraped_data

        except Exception as e:
            self.logger.error(f"Error in static scraping {url}: {e}")
            self.failed_urls.add(url)
            return None

    def _create_metadata(
        self, url: str, scraped_data: Dict[str, Any], response: requests.Response
    ) -> Metadata:
        """Create metadata for scraped content"""
        try:
            # Prepare source data for metadata extraction
            source_data = {
                "url": url,
                "content": scraped_data.get("content", ""),
                "html": response.text,
                "response": response,
                "custom_fields": {
                    "extractor_type": scraped_data.get("extractor_type", "generic"),
                    "content_length": len(scraped_data.get("content", "")),
                    "links_count": len(scraped_data.get("links", [])),
                    "images_count": len(scraped_data.get("images", [])),
                },
            }

            # Create metadata using metadata handler
            metadata = self.metadata_handler.create_metadata("web", source_data)

            return metadata

        except Exception as e:
            self.logger.error(f"Error creating metadata: {e}")
            # Return basic metadata on error
            return Metadata(
                source_id=f"web_{hashlib.md5(url.encode()).hexdigest()[:12]}",
                source_type="web",
                url=url,
                processing_status="failed",
                processing_errors=[str(e)],
                scraped_at=datetime.now(timezone.utc),
            )

    @chunked_processing(chunk_size=50)
    def scrape_urls_batch(
        self, urls: List[str], extractor_type: str = "generic"
    ) -> List[Dict[str, Any]]:
        """
        Scrape a batch of URLs with chunked processing and metadata integration

        Args:
            urls: List of URLs to scrape
            extractor_type: Type of extractor to use

        Returns:
            List of scraped data dictionaries with metadata
        """
        results = []

        for url in urls:
            try:
                # Add random delay to be respectful
                time.sleep(random.uniform(*self.delay_range))

                scraped_data = self.scrape(url, extractor_type)
                if scraped_data:
                    results.append(scraped_data)

            except Exception as e:
                self.logger.error(f"Error in batch scraping {url}: {e}")
                continue

        return results

    @memory_safe
    def scrape_site(
        self, start_url: Optional[str] = None, extractor_type: str = "generic"
    ) -> List[Dict[str, Any]]:
        """
        Scrape entire site starting from base URL with metadata integration

        Args:
            start_url: Starting URL (if None, uses base_url)
            extractor_type: Type of extractor to use

        Returns:
            List of scraped data dictionaries with metadata
        """
        start_url = start_url or self.base_url
        if not start_url:
            raise ValueError("No start URL provided")

        self.logger.info(
            f"Starting site scraping from: {start_url} with extractor: {extractor_type}"
        )

        # Check memory before starting
        if memory_manager.is_memory_critical():
            self.logger.warning("Memory critical before site scraping, optimizing...")
            memory_manager.optimize_memory()

        all_scraped_data = []
        urls_to_scrape = [start_url]
        pages_scraped = 0

        while urls_to_scrape and pages_scraped < self.max_pages:
            # Process URLs in chunks
            current_chunk = urls_to_scrape[: self.chunk_size]
            urls_to_scrape = urls_to_scrape[self.chunk_size :]

            self.logger.info(
                f"Scraping chunk {pages_scraped//self.chunk_size + 1} ({len(current_chunk)} URLs)"
            )

            # Scrape current chunk
            chunk_results = self.scrape_urls_batch(current_chunk, extractor_type)
            all_scraped_data.extend(chunk_results)

            # Collect new URLs from scraped pages
            new_urls = []
            for result in chunk_results:
                new_urls.extend(result.get("links", []))

            # Filter and add new URLs
            for url in new_urls:
                if (
                    url not in self.scraped_urls
                    and url not in self.failed_urls
                    and url not in urls_to_scrape
                    and self._is_valid_url(url)
                ):
                    urls_to_scrape.append(url)

            pages_scraped += len(current_chunk)

            # Save intermediate results with metadata
            if self.save_intermediate and chunk_results:
                self._save_intermediate_results(chunk_results, pages_scraped)

            # Memory optimization between chunks
            memory_manager.force_garbage_collection()

            # Check memory after each chunk
            if memory_manager.is_memory_critical():
                self.logger.warning(
                    f"Memory critical after chunk {pages_scraped//self.chunk_size}, optimizing..."
                )
                memory_manager.optimize_memory()

            self.logger.info(
                f"Scraped {pages_scraped} pages, {len(all_scraped_data)} successful, {len(urls_to_scrape)} URLs remaining"
            )

        self.logger.info(
            f"Site scraping completed. Total pages scraped: {pages_scraped}, successful: {len(all_scraped_data)}"
        )

        return all_scraped_data

    def _save_intermediate_results(
        self, results: List[Dict[str, Any]], chunk_number: int
    ) -> None:
        """Save intermediate results to disk with metadata"""
        try:
            filename = f"scraped_chunk_{chunk_number}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.intermediate_dir / filename

            # Convert metadata objects to dictionaries for JSON serialization
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                if "metadata" in serializable_result and hasattr(
                    serializable_result["metadata"], "to_dict"
                ):
                    # Convert Metadata object to dictionary
                    serializable_result["metadata"] = serializable_result[
                        "metadata"
                    ].to_dict()
                serializable_results.append(serializable_result)

            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved intermediate results to {filepath}")

        except Exception as e:
            self.logger.error(f"Error saving intermediate results: {e}")

    def load_intermediate_results(self) -> List[Dict[str, Any]]:
        """Load all intermediate results from disk with metadata"""
        all_results = []

        try:
            for filepath in self.intermediate_dir.glob("scraped_chunk_*.json"):
                with open(filepath, "r", encoding="utf-8") as f:
                    chunk_results = json.load(f)
                    all_results.extend(chunk_results)

            self.logger.info(
                f"Loaded {len(all_results)} results from intermediate files"
            )

        except Exception as e:
            self.logger.error(f"Error loading intermediate results: {e}")

        return all_results

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """Save scraped results to file with metadata"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert results to JSON-serializable format
            serializable_results = []
            for result in results:
                serializable_result = result.copy()
                # Convert Metadata object to dictionary if present
                if "metadata" in serializable_result:
                    metadata = serializable_result["metadata"]
                    if hasattr(metadata, "to_dict"):
                        serializable_result["metadata"] = metadata.to_dict()
                    elif isinstance(metadata, dict):
                        # Already a dictionary, ensure it's JSON serializable
                        serializable_result["metadata"] = metadata
                    else:
                        # Remove non-serializable metadata
                        del serializable_result["metadata"]
                serializable_results.append(serializable_result)

            # Save as JSON
            json_path = output_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2, ensure_ascii=False)

            # Save as CSV (flattened with metadata)
            csv_path = output_path.with_suffix(".csv")
            flattened_results = self._flatten_results_for_csv(results)
            df = pd.DataFrame(flattened_results)
            df.to_csv(csv_path, index=False, encoding="utf-8")

            # Export metadata report
            metadata_list = []
            for result in results:
                if result.get("metadata"):
                    metadata = result["metadata"]
                    # Convert to Metadata object if it's a dictionary
                    if isinstance(metadata, dict):
                        metadata = self.metadata_handler.load_metadata_from_dict(
                            metadata
                        )
                    metadata_list.append(metadata)

            if metadata_list:
                metadata_report_path = output_path.with_suffix(".metadata_report")
                self.metadata_handler.export_metadata_report(
                    metadata_list, str(metadata_report_path)
                )

            self.logger.info(
                f"Results saved to {json_path}, {csv_path}, and metadata report"
            )

        except Exception as e:
            self.logger.error(f"Error saving results: {e}")

    def _flatten_results_for_csv(
        self, results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Flatten results for CSV export including metadata"""
        flattened = []

        for result in results:
            flat_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content_length": len(result.get("content", "")),
                "links_count": len(result.get("links", [])),
                "images_count": len(result.get("images", [])),
                "extractor_type": result.get("extractor_type", "generic"),
            }

            # Add metadata fields if available
            if result.get("metadata"):
                metadata = result["metadata"]
                # Handle both Metadata objects and dictionaries
                if hasattr(metadata, "source_id"):  # Metadata object
                    flat_result.update(
                        {
                            "source_id": metadata.source_id,
                            "quality_score": metadata.quality_score,
                            "completeness_score": metadata.completeness_score,
                            "freshness_score": metadata.freshness_score,
                            "processing_status": metadata.processing_status,
                            "author": metadata.author,
                            "publisher": metadata.publisher,
                            "domain": metadata.domain,
                            "language": metadata.language,
                            "created_at": (
                                metadata.created_at.isoformat()
                                if metadata.created_at
                                else ""
                            ),
                            "modified_at": (
                                metadata.modified_at.isoformat()
                                if metadata.modified_at
                                else ""
                            ),
                            "scraped_at": (
                                metadata.scraped_at.isoformat()
                                if metadata.scraped_at
                                else ""
                            ),
                        }
                    )
                else:  # Dictionary
                    flat_result.update(
                        {
                            "source_id": metadata.get("source_id", ""),
                            "quality_score": metadata.get("quality_score", ""),
                            "completeness_score": metadata.get(
                                "completeness_score", ""
                            ),
                            "freshness_score": metadata.get("freshness_score", ""),
                            "processing_status": metadata.get("processing_status", ""),
                            "author": metadata.get("author", ""),
                            "publisher": metadata.get("publisher", ""),
                            "domain": metadata.get("domain", ""),
                            "language": metadata.get("language", ""),
                            "created_at": metadata.get("created_at", ""),
                            "modified_at": metadata.get("modified_at", ""),
                            "scraped_at": metadata.get("scraped_at", ""),
                        }
                    )

            flattened.append(flat_result)

        return flattened

    def get_scraping_stats(self) -> Dict[str, Any]:
        """Get scraping statistics with metadata information"""
        stats = {
            "total_urls_scraped": len(self.scraped_urls),
            "total_urls_failed": len(self.failed_urls),
            "success_rate": (
                len(self.scraped_urls)
                / (len(self.scraped_urls) + len(self.failed_urls))
                if (len(self.scraped_urls) + len(self.failed_urls)) > 0
                else 0
            ),
            "base_url": self.base_url,
            "max_pages": self.max_pages,
            "user_agent": self.user_agent,
            "config_source": "centralized",
            "memory_usage": memory_manager.get_memory_report(),
        }

        return stats

    def cleanup(self) -> None:
        """Clean up scraper resources"""
        self.session.close()
        self.scraped_urls.clear()
        self.failed_urls.clear()

        # Clean up intermediate files if requested
        if self.save_intermediate and self.intermediate_dir.exists():
            try:
                for filepath in self.intermediate_dir.glob("scraped_chunk_*.json"):
                    filepath.unlink()
                self.logger.info("Cleaned up intermediate files")
            except Exception as e:
                self.logger.error(f"Error cleaning up intermediate files: {e}")

        self.logger.info("Web scraper cleaned up")

    def analyze_scraped_content(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze scraped content for quality and patterns with metadata"""
        if not results:
            return {"error": "No results to analyze"}

        analysis = {
            "total_pages": len(results),
            "content_length_stats": {
                "min": min(len(r.get("content", "")) for r in results),
                "max": max(len(r.get("content", "")) for r in results),
                "avg": sum(len(r.get("content", "")) for r in results) / len(results),
            },
            "domains": {},
            "extractor_types": {},
            "content_quality": {
                "pages_with_title": sum(1 for r in results if r.get("title")),
                "pages_with_links": sum(1 for r in results if r.get("links")),
                "pages_with_images": sum(1 for r in results if r.get("images")),
            },
            "metadata_quality": {
                "pages_with_metadata": sum(1 for r in results if r.get("metadata")),
                "high_quality_pages": sum(
                    1
                    for r in results
                    if r.get("metadata")
                    and r["metadata"].quality_score
                    and r["metadata"].quality_score > 0.8
                ),
                "complete_pages": sum(
                    1
                    for r in results
                    if r.get("metadata")
                    and r["metadata"].completeness_score
                    and r["metadata"].completeness_score > 0.8
                ),
            },
        }

        # Analyze domains and extractor types
        for result in results:
            domain = result.get("domain", "unknown")
            analysis["domains"][domain] = analysis["domains"].get(domain, 0) + 1

            extractor_type = result.get("extractor_type", "generic")
            analysis["extractor_types"][extractor_type] = (
                analysis["extractor_types"].get(extractor_type, 0) + 1
            )

        # Analyze metadata quality if available
        metadata_results = [r for r in results if r.get("metadata")]
        if metadata_results:
            quality_scores = [
                r["metadata"].quality_score
                for r in metadata_results
                if r["metadata"].quality_score
            ]
            completeness_scores = [
                r["metadata"].completeness_score
                for r in metadata_results
                if r["metadata"].completeness_score
            ]
            freshness_scores = [
                r["metadata"].freshness_score
                for r in metadata_results
                if r["metadata"].freshness_score
            ]

            analysis["metadata_statistics"] = {
                "average_quality_score": (
                    sum(quality_scores) / len(quality_scores) if quality_scores else 0
                ),
                "average_completeness_score": (
                    sum(completeness_scores) / len(completeness_scores)
                    if completeness_scores
                    else 0
                ),
                "average_freshness_score": (
                    sum(freshness_scores) / len(freshness_scores)
                    if freshness_scores
                    else 0
                ),
            }

        return analysis

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid based on filtering rules"""
        try:
            parsed = urlparse(url)

            # Check domain restrictions
            if self.allowed_domains and parsed.netloc not in self.allowed_domains:
                return False

            # Check excluded patterns
            for pattern in self.excluded_patterns:
                if re.search(pattern, url):
                    return False

            # Check required patterns
            if self.required_patterns:
                for pattern in self.required_patterns:
                    if not re.search(pattern, url):
                        return False

            return True

        except Exception:
            return False


class GenericWebExtractor:
    """Generic web content extractor"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content using generic approach"""
        try:
            # Extract title
            title = ""
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                h1_tag = soup.find("h1")
                if h1_tag:
                    title = h1_tag.get_text(strip=True)

            # Extract content
            content = self._extract_main_content(soup)

            # Extract links
            links = self._extract_links(soup, url)

            # Extract images
            images = self._extract_images(soup, url)

            return {
                "url": url,
                "title": title,
                "content": content,
                "links": links,
                "images": images,
                "extractor_type": "generic",
                "domain": urlparse(url).netloc,
            }

        except Exception as e:
            self.logger.error(f"Error in generic extraction: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract main content from HTML"""
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()

        # Try to find main content area
        main_content = None

        # Common content selectors
        content_selectors = [
            "main",
            "article",
            ".content",
            ".main-content",
            "#content",
            "#main",
            ".post-content",
            ".entry-content",
        ]

        for selector in content_selectors:
            main_content = soup.select_one(selector)
            if main_content:
                break

        # If no main content found, use body
        if not main_content:
            main_content = soup.find("body")

        if main_content:
            # Get text content
            text = main_content.get_text(separator=" ", strip=True)

            # Clean up whitespace
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return ""

    def _extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract links from page"""
        links = []
        for link in soup.find_all("a", href=True):
            href = link["href"]
            absolute_url = urljoin(base_url, href)
            links.append(absolute_url)

        return list(set(links))  # Remove duplicates

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract image URLs from page"""
        images = []
        for img in soup.find_all("img", src=True):
            src = img["src"]
            absolute_url = urljoin(base_url, src)
            images.append(absolute_url)

        return list(set(images))  # Remove duplicates


class NewsWebExtractor(GenericWebExtractor):
    """News website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from news websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # News-specific enhancements
            result["extractor_type"] = "news"

            # Extract publication date
            date = self._extract_publication_date(soup)
            if date:
                result["publication_date"] = date

            # Extract author
            author = self._extract_author(soup)
            if author:
                result["author"] = author

            # Extract category/tags
            category = self._extract_category(soup)
            if category:
                result["category"] = category

            return result

        except Exception as e:
            self.logger.error(f"Error in news extraction: {e}")
            return None

    def _extract_publication_date(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date from news articles"""
        date_selectors = [
            "time[datetime]",
            ".published-date",
            ".article-date",
            ".post-date",
            '[class*="date"]',
            '[class*="time"]',
        ]

        for selector in date_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get("datetime") or element.get_text(strip=True)

        return None

    def _extract_author(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract author from news articles"""
        author_selectors = [
            ".author",
            ".byline",
            ".article-author",
            '[class*="author"]',
            '[rel="author"]',
        ]

        for selector in author_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None

    def _extract_category(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract category from news articles"""
        category_selectors = [
            ".category",
            ".section",
            ".tag",
            '[class*="category"]',
            '[class*="section"]',
        ]

        for selector in category_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None


class BlogWebExtractor(GenericWebExtractor):
    """Blog website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from blog websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # Blog-specific enhancements
            result["extractor_type"] = "blog"

            # Extract tags
            tags = self._extract_tags(soup)
            if tags:
                result["tags"] = tags

            # Extract comments count
            comments_count = self._extract_comments_count(soup)
            if comments_count:
                result["comments_count"] = comments_count

            return result

        except Exception as e:
            self.logger.error(f"Error in blog extraction: {e}")
            return None

    def _extract_tags(self, soup: BeautifulSoup) -> List[str]:
        """Extract tags from blog posts"""
        tags = []
        tag_selectors = [".tags a", ".tag a", '[class*="tag"] a', ".post-tags a"]

        for selector in tag_selectors:
            elements = soup.select(selector)
            for element in elements:
                tag = element.get_text(strip=True)
                if tag and tag not in tags:
                    tags.append(tag)

        return tags

    def _extract_comments_count(self, soup: BeautifulSoup) -> Optional[int]:
        """Extract comments count from blog posts"""
        comment_selectors = [".comments-count", ".comment-count", '[class*="comment"]']

        for selector in comment_selectors:
            element = soup.select_one(selector)
            if element:
                text = element.get_text(strip=True)
                # Extract number from text
                numbers = re.findall(r"\d+", text)
                if numbers:
                    return int(numbers[0])

        return None


class DocumentationWebExtractor(GenericWebExtractor):
    """Documentation website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from documentation websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # Documentation-specific enhancements
            result["extractor_type"] = "documentation"

            # Extract code blocks
            code_blocks = self._extract_code_blocks(soup)
            if code_blocks:
                result["code_blocks"] = code_blocks

            # Extract table of contents
            toc = self._extract_table_of_contents(soup)
            if toc:
                result["table_of_contents"] = toc

            return result

        except Exception as e:
            self.logger.error(f"Error in documentation extraction: {e}")
            return None

    def _extract_code_blocks(self, soup: BeautifulSoup) -> List[str]:
        """Extract code blocks from documentation"""
        code_blocks = []
        code_selectors = ["pre code", "code", ".highlight", ".code-block"]

        for selector in code_selectors:
            elements = soup.select(selector)
            for element in elements:
                code = element.get_text(strip=True)
                if code and len(code) > 10:
                    code_blocks.append(code)

        return code_blocks

    def _extract_table_of_contents(self, soup: BeautifulSoup) -> List[str]:
        """Extract table of contents from documentation"""
        toc = []
        toc_selectors = [".toc a", ".table-of-contents a", ".nav a", '[class*="toc"] a']

        for selector in toc_selectors:
            elements = soup.select(selector)
            for element in elements:
                item = element.get_text(strip=True)
                if item and item not in toc:
                    toc.append(item)

        return toc


class EcommerceWebExtractor(GenericWebExtractor):
    """E-commerce website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from e-commerce websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # E-commerce-specific enhancements
            result["extractor_type"] = "ecommerce"

            # Extract product information
            product_info = self._extract_product_info(soup)
            if product_info:
                result.update(product_info)

            # Extract price information
            price_info = self._extract_price_info(soup)
            if price_info:
                result.update(price_info)

            return result

        except Exception as e:
            self.logger.error(f"Error in e-commerce extraction: {e}")
            return None

    def _extract_product_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract product information"""
        product_info = {}

        # Product name
        name_selectors = [
            ".product-name",
            ".product-title",
            'h1[class*="product"]',
            '[class*="product-name"]',
        ]

        for selector in name_selectors:
            element = soup.select_one(selector)
            if element:
                product_info["product_name"] = element.get_text(strip=True)
                break

        # Product description
        desc_selectors = [
            ".product-description",
            ".product-summary",
            '[class*="description"]',
        ]

        for selector in desc_selectors:
            element = soup.select_one(selector)
            if element:
                product_info["product_description"] = element.get_text(strip=True)
                break

        return product_info

    def _extract_price_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract price information"""
        price_info = {}

        # Price
        price_selectors = [
            ".price",
            ".product-price",
            '[class*="price"]',
            "[data-price]",
        ]

        for selector in price_selectors:
            element = soup.select_one(selector)
            if element:
                price_text = element.get_text(strip=True)
                # Extract price from text
                price_match = re.search(r"[\$£€]?\d+\.?\d*", price_text)
                if price_match:
                    price_info["price"] = price_match.group()
                break

        return price_info


class GovernmentWebExtractor(GenericWebExtractor):
    """Government website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from government websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # Government-specific enhancements
            result["extractor_type"] = "government"

            # Extract document information
            doc_info = self._extract_document_info(soup)
            if doc_info:
                result.update(doc_info)

            # Extract contact information
            contact_info = self._extract_contact_info(soup)
            if contact_info:
                result["contact_info"] = contact_info

            return result

        except Exception as e:
            self.logger.error(f"Error in government extraction: {e}")
            return None

    def _extract_document_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract document information from government sites"""
        doc_info = {}

        # Document number
        doc_selectors = [
            ".document-number",
            ".reference-number",
            '[class*="document"]',
            '[class*="reference"]',
        ]

        for selector in doc_selectors:
            element = soup.select_one(selector)
            if element:
                doc_info["document_number"] = element.get_text(strip=True)
                break

        return doc_info

    def _extract_contact_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract contact information from government sites"""
        contact_info = {}

        # Phone numbers
        phone_pattern = r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phone_elements = soup.find_all(text=re.compile(phone_pattern))
        if phone_elements:
            contact_info["phone_numbers"] = [elem.strip() for elem in phone_elements]

        # Email addresses
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        email_elements = soup.find_all(text=re.compile(email_pattern))
        if email_elements:
            contact_info["email_addresses"] = [elem.strip() for elem in email_elements]

        return contact_info


class AcademicWebExtractor(GenericWebExtractor):
    """Academic website content extractor"""

    def extract(
        self, soup: BeautifulSoup, url: str, response: requests.Response
    ) -> Dict[str, Any]:
        """Extract content from academic websites"""
        try:
            # Get base extraction
            result = super().extract(soup, url, response)
            if not result:
                return None

            # Academic-specific enhancements
            result["extractor_type"] = "academic"

            # Extract citation information
            citation_info = self._extract_citation_info(soup)
            if citation_info:
                result.update(citation_info)

            # Extract abstract
            abstract = self._extract_abstract(soup)
            if abstract:
                result["abstract"] = abstract

            return result

        except Exception as e:
            self.logger.error(f"Error in academic extraction: {e}")
            return None

    def _extract_citation_info(self, soup: BeautifulSoup) -> Dict[str, Any]:
        """Extract citation information from academic papers"""
        citation_info = {}

        # DOI
        doi_selectors = [".doi", '[class*="doi"]', 'a[href*="doi.org"]']

        for selector in doi_selectors:
            element = soup.select_one(selector)
            if element:
                doi = element.get_text(strip=True) or element.get("href", "")
                if "doi.org" in doi:
                    citation_info["doi"] = doi
                break

        # Journal/Conference
        journal_selectors = [
            ".journal",
            ".conference",
            '[class*="journal"]',
            '[class*="conference"]',
        ]

        for selector in journal_selectors:
            element = soup.select_one(selector)
            if element:
                citation_info["journal"] = element.get_text(strip=True)
                break

        return citation_info

    def _extract_abstract(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract abstract from academic papers"""
        abstract_selectors = [
            ".abstract",
            '[class*="abstract"]',
            ".summary",
            '[class*="summary"]',
        ]

        for selector in abstract_selectors:
            element = soup.select_one(selector)
            if element:
                return element.get_text(strip=True)

        return None
