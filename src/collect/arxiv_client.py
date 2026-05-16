import urllib.request
import urllib.error
import xml.etree.ElementTree as ET

ARXIV_API = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
    "arxiv": "http://arxiv.org/schemas/atom",
}


def fetch_category_stats(code: str, max_results: int = 100) -> tuple[int, float]:
    url = f"{ARXIV_API}?search_query=cat:{code}&max_results={max_results}"
    with urllib.request.urlopen(url, timeout=30) as response:
        xml_bytes = response.read()

    root = ET.fromstring(xml_bytes)

    total_el = root.find("opensearch:totalResults", NS)
    total = int(total_el.text) if total_el is not None else 0

    entries = root.findall("atom:entry", NS)
    primary_count = sum(
        1
        for e in entries
        if (pc := e.find("arxiv:primary_category", NS)) is not None
        and pc.get("term") == code
    )
    primary_pct = (primary_count / len(entries) * 100) if entries else 0.0

    return total, primary_pct
