import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin, urlparse
import tldextract

class DocumentationCrawler:
    def __init__(self, base_url, max_depth=2):
        self.base_url = base_url
        self.domain = self.get_domain(base_url)
        self.subdomain = self.get_subdomain(base_url)
        self.path_prefix = self.get_path_prefix(base_url)
        self.visited = set()
        self.max_depth = max_depth

    def get_domain(self, url):
        extracted = tldextract.extract(url)
        return f"{extracted.domain}.{extracted.suffix}"

    def get_subdomain(self, url):
        extracted = tldextract.extract(url)
        return extracted.subdomain

    def get_path_prefix(self, url):
        parsed = urlparse(url)
        return parsed.path.split('/')[1] if parsed.path.count('/') >= 1 else ""

    def is_valid_url(self, url):
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ['http', 'https']:
                return False

            extracted = tldextract.extract(url)
            full_domain = f"{extracted.domain}.{extracted.suffix}"

            if not extracted.domain or not extracted.suffix:
                return False

            if full_domain != self.domain:
                return False

            if extracted.subdomain and extracted.subdomain not in ['docs', 'help', self.subdomain]:
                return False

            if self.path_prefix and not parsed.path.startswith(f"/{self.path_prefix}"):
                return False

            return True
        except Exception as e:
            print(f"[URL validation error]: {e}")
            return False

    def fetch_page(self, url):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content_type = response.headers.get('Content-Type', '')
            if 'text/html' not in content_type:
                print(f"[Skipped non-HTML content] {url} -> {content_type}")
                return None
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"[Error fetching {url}]: {e}")
        return None

    def extract_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        for tag in soup(['nav', 'header', 'footer', 'aside', 'script', 'style']):
            tag.decompose()
        main_content = soup.body or soup
        return self.parse_elements(main_content)

    def parse_elements(self, container):
        content = []

        def walk(node, level=0):
            if isinstance(node, NavigableString):
                text = node.strip()
                if text:
                    content.append(("  " * level) + text)
            elif isinstance(node, Tag):
                if node.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    content.append("\n" + ("#" * int(node.name[1])) + " " + node.get_text(strip=True))
                elif node.name == 'ul':
                    for li in node.find_all('li', recursive=False):
                        content.append(("  " * level) + "- " + li.get_text(strip=True))
                elif node.name == 'ol':
                    for i, li in enumerate(node.find_all('li', recursive=False), start=1):
                        content.append(("  " * level) + f"{i}. " + li.get_text(strip=True))
                elif node.name == 'table':
                    rows = []
                    for row in node.find_all('tr'):
                        cols = [col.get_text(strip=True) for col in row.find_all(['th', 'td'])]
                        rows.append(cols)
                    content.append("\n[Table]")
                    for r in rows:
                        content.append(" | ".join(r))
                else:
                    for child in node.children:
                        walk(child, level + 1)

        walk(container)
        return "\n".join(content)

    def crawl(self, url=None, depth=0):
        if depth > self.max_depth:
            return ""

        url = url or self.base_url
        if url in self.visited:
            return ""

        if not self.is_valid_url(url):
            print(f"[Skipped] Invalid or external URL: {url}")
            return ""

        self.visited.add(url)
        html = self.fetch_page(url)
        all_content = ""
        if html:
            print(f"\n Crawled: {url}")
            content = self.extract_content(html)
            all_content += f"\n\n=== {url} ===\n{content}"
            soup = BeautifulSoup(html, 'html.parser')
            for link in soup.find_all('a', href=True):
                next_url = urljoin(url, link['href'])
                if next_url not in self.visited:
                    sub_content = self.crawl(next_url, depth + 1)
                    if sub_content:
                        all_content += f"\n\n{sub_content}"
        
        return all_content
