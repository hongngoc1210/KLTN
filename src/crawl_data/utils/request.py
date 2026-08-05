import requests
import urllib3

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/138.0 Safari/537.36"
    )
}

# Tạo Session dùng chung
session = requests.Session()

retry = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"]
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("http://", adapter)
session.mount("https://", adapter)

session.headers.update(HEADERS)


def get_html(url):
    """Lấy nội dung HTML."""
    response = session.get(
        url,
        verify=False,
        timeout=30
    )
    response.raise_for_status()
    return response.text


def get_response(url):
    """Lấy response (dùng để download PDF)."""
    response = session.get(
        url,
        verify=False,
        timeout=60,
        stream=True
    )
    response.raise_for_status()
    return response