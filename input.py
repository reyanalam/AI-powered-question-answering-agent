from preprocessing import DocumentationCrawler
import requests

start_url = input("Enter a documentation/help URL to crawl: ").strip()

try:
        head_response = requests.head(start_url, timeout=5)
        head_response.raise_for_status()
        
        crawler = DocumentationCrawler(start_url, max_depth=2)
        content = crawler.crawl()
except requests.exceptions.RequestException as e:
        print(f"[Error] The provided URL is not accessible: {e}")
except Exception as e:
        print(f"[Error] Something went wrong during initialization: {e}")

