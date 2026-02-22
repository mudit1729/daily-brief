import logging
import re
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 15
MAX_HTML_BYTES = 512 * 1024  # 512 KB limit to prevent memory issues
USER_AGENT = 'SignalBriefBot/1.0 (+https://github.com/signal-brief-engine)'


def extract(url):
    """
    Fetch full HTML from URL and extract article content.
    Primary: readability-lxml. Fallback: BeautifulSoup heuristic.
    Returns dict with: title, text, og_image_url, author, published_at, raw_html
    """
    try:
        resp = requests.get(
            url,
            timeout=REQUEST_TIMEOUT,
            headers={'User-Agent': USER_AGENT},
            stream=True,
        )
        resp.raise_for_status()

        # Read up to MAX_HTML_BYTES to prevent huge pages from causing OOM/crashes
        chunks = []
        total = 0
        for chunk in resp.iter_content(chunk_size=64 * 1024, decode_unicode=True):
            chunks.append(chunk)
            total += len(chunk)
            if total > MAX_HTML_BYTES:
                logger.debug(f"Truncating HTML at {MAX_HTML_BYTES} bytes for {url}")
                break
        resp.close()
        html = ''.join(chunks)
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    result = _extract_readability(html, url)
    if not result or not result.get('text') or len(result['text']) < 100:
        result = _extract_bs4(html, url)

    if result:
        result['raw_html'] = html
        meta = _extract_meta(html)
        result['og_image_url'] = meta.get('og_image_url')
        if not result.get('author'):
            result['author'] = meta.get('author')

    return result


def _extract_readability(html, url):
    """Extract using readability-lxml."""
    try:
        from readability import Document
        doc = Document(html, url=url)
        title = doc.title()
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, 'lxml')
        text = soup.get_text(separator=' ', strip=True)
        return {'title': title, 'text': text, 'author': None}
    except Exception as e:
        logger.debug(f"Readability failed for {url}: {e}")
        return None


def _extract_bs4(html, url):
    """Fallback extraction using BeautifulSoup."""
    try:
        soup = BeautifulSoup(html, 'lxml')
        title = ''
        title_tag = soup.find('title')
        if title_tag:
            title = title_tag.get_text(strip=True)

        article = soup.find('article') or soup.find('main') or soup.find('div', class_=re.compile(r'content|article|post'))
        if article:
            text = article.get_text(separator=' ', strip=True)
        else:
            body = soup.find('body')
            text = body.get_text(separator=' ', strip=True) if body else ''

        return {'title': title, 'text': text, 'author': None}
    except Exception as e:
        logger.debug(f"BS4 extraction failed for {url}: {e}")
        return None


def _extract_meta(html):
    """Extract OG image and author from meta tags."""
    meta = {}
    try:
        soup = BeautifulSoup(html, 'lxml')
        og_image = soup.find('meta', property='og:image')
        if og_image:
            meta['og_image_url'] = og_image.get('content')

        author_meta = soup.find('meta', attrs={'name': 'author'})
        if author_meta:
            meta['author'] = author_meta.get('content')
    except Exception:
        pass
    return meta
