from __future__ import annotations

import io
import json
import random
import tarfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests


UNPAYWALL_EMAIL = "fernver62@gmail.com"


def _clean_doi(doi: str) -> str:
    return (doi or "").replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def _get_pdf_url_unpaywall(doi: str, timeout: int = 15) -> list[str]:
    doi_clean = _clean_doi(doi)
    if not doi_clean:
        return []
    url = f"https://api.unpaywall.org/v2/{doi_clean}?email={UNPAYWALL_EMAIL}"
    try:
        r = requests.get(url, timeout=timeout, headers={"User-Agent": "paperforge-v3/1.0"})
        if r.status_code != 200:
            return []
        data = r.json()
        urls = []
        for loc in data.get("oa_locations", []):
            u = loc.get("url_for_pdf") or loc.get("url", "")
            if u:
                urls.append(u)
        return urls
    except Exception:
        return []


def _get_pdf_urls_crossref(doi: str, timeout: int = 15) -> list[str]:
    doi_clean = _clean_doi(doi)
    if not doi_clean:
        return []
    try:
        r = requests.get(
            f"https://api.crossref.org/works/{doi_clean}",
            timeout=timeout,
            headers={"User-Agent": f"paperforge-v3/1.0 (mailto:{UNPAYWALL_EMAIL})"},
        )
        if r.status_code != 200:
            return []
        links = r.json().get("message", {}).get("link", [])
        return [
            link["URL"]
            for link in links
            if "pdf" in link.get("content-type", "").lower()
            or "pdf" in link.get("URL", "").lower()
        ]
    except Exception:
        return []


def _mdpi_cdn_url(redirect_url: str) -> str | None:
    try:
        filename = redirect_url.rstrip("/").split("/")[-1]
        if not filename.endswith(".pdf"):
            return None
        slug = filename[:-4]
        parts = slug.rsplit("-", 2)
        if len(parts) != 3:
            return None
        journal = parts[0]
        return f"https://res.mdpi.com/{journal}/{slug}/article_deploy/{slug}.pdf"
    except Exception:
        return None


def _fetch_pdf(url: str, timeout: int = 60, referer: str = "", delay: float = 1.5) -> bytes | None:
    time.sleep(delay + random.uniform(0, 0.8))
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }
    if referer:
        headers["Referer"] = referer
    try:
        r = requests.get(url, headers=headers, allow_redirects=True, timeout=timeout)
        if r.status_code == 200 and r.content[:4] == b"%PDF":
            return r.content
        return None
    except Exception:
        return None


def download_pdf(pmcid: str, doi: str, out_path: Path, timeout: int = 60) -> bool:
    """Download one PDF using the current PaperForge fallback chain."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(data: bytes, label: str) -> bool:
        out_path.write_bytes(data)
        print(f"    OK [{label}] {len(data) // 1024} KB")
        return True

    if doi:
        unpaywall_urls = _get_pdf_url_unpaywall(doi, timeout=15)
        mdpi_cdn_candidates = []
        for url in unpaywall_urls:
            if "pmc.ncbi.nlm.nih.gov" in url and url.endswith(".pdf"):
                cdn = _mdpi_cdn_url(url)
                if cdn:
                    mdpi_cdn_candidates.append(cdn)

        for url in unpaywall_urls:
            data = _fetch_pdf(url, timeout=timeout)
            if data:
                return save(data, "unpaywall")
            print(f"    unpaywall failed: {url[:70]}")

        for cdn in mdpi_cdn_candidates:
            data = _fetch_pdf(cdn, timeout=timeout, delay=0.5)
            if data:
                return save(data, f"mdpi-cdn {cdn[:70]}")
            print(f"    mdpi-cdn failed: {cdn[:70]}")

    if pmcid:
        pmc_url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
        try:
            r = requests.get(
                pmc_url,
                headers={"User-Agent": "paperforge-v3/1.0"},
                allow_redirects=True,
                timeout=timeout,
            )
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return save(r.content, "pmc-direct")
            if "pmc.ncbi.nlm.nih.gov" in r.url:
                cdn = _mdpi_cdn_url(r.url)
                if cdn:
                    data = _fetch_pdf(cdn, timeout=timeout, delay=0.5)
                    if data:
                        return save(data, f"mdpi-cdn(pmc-redirect) {cdn[:60]}")
            print(f"    pmc-direct ({r.status_code}) final_url={r.url[:70]}")
        except Exception as exc:
            print(f"    pmc-direct failed: {exc}")

        epmc_url = f"https://europepmc.org/backend/ptpmcrender.fcgi?accid={pmcid}&blobtype=pdf"
        data = _fetch_pdf(epmc_url, timeout=timeout, delay=1.0)
        if data:
            return save(data, "europepmc")
        print("    europepmc failed")

    if doi:
        crossref_urls = _get_pdf_urls_crossref(doi, timeout=15)
        for url in crossref_urls:
            data = _fetch_pdf(url, timeout=timeout)
            if data:
                return save(data, f"crossref {url[:50]}")
        if crossref_urls:
            print("    crossref: links found but all failed")

        doi_clean = _clean_doi(doi)
        try:
            time.sleep(1.5)
            r = requests.get(
                f"https://doi.org/{doi_clean}",
                headers={"User-Agent": "paperforge-v3/1.0", "Accept": "application/pdf"},
                allow_redirects=True,
                timeout=timeout,
            )
            if r.status_code == 200 and r.content[:4] == b"%PDF":
                return save(r.content, "doi-resolver")
        except Exception as exc:
            print(f"    doi-resolver failed: {exc}")

    if pmcid:
        try:
            import ftplib

            r = requests.get(
                f"https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id={pmcid}",
                headers={"User-Agent": "paperforge-v3/1.0"},
                timeout=15,
            )
            if r.status_code == 200:
                root = ET.fromstring(r.text)
                link = root.find(".//link[@format='tgz']")
                if link is not None:
                    ftp_href = link.attrib["href"]
                    ftp_path = ftp_href.replace("ftp://ftp.ncbi.nlm.nih.gov", "")
                    print(f"    PMC OA FTP: {ftp_path}")
                    buf = io.BytesIO()
                    ftp = ftplib.FTP("ftp.ncbi.nlm.nih.gov", timeout=120)
                    ftp.login()
                    ftp.retrbinary(f"RETR {ftp_path}", buf.write)
                    ftp.quit()
                    buf.seek(0)
                    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
                        pdfs = [m for m in tar.getmembers() if m.name.endswith(".pdf")]
                        if pdfs:
                            pdf_bytes = tar.extractfile(pdfs[0]).read()
                            if pdf_bytes[:4] == b"%PDF":
                                return save(pdf_bytes, f"pmc-oa-ftp {pdfs[0].name}")
        except Exception as exc:
            print(f"    pmc-oa-ftp failed: {exc}")

    print(f"    ALL methods failed for {pmcid}")
    return False


def download_pdf_for_paper(
    pmcid: str,
    doi: str,
    paper_dir: str | Path,
    reuse_existing: bool = True,
) -> Path:
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = paper_dir / "paper.pdf"
    if reuse_existing and pdf_path.exists() and pdf_path.stat().st_size > 0:
        return pdf_path
    ok = download_pdf(pmcid, doi, pdf_path)
    if not ok or not pdf_path.exists():
        raise RuntimeError(f"pdf_download_failed:{pmcid}")
    return pdf_path


def write_paper_context(paper_dir: str | Path, paper: dict) -> Path:
    paper_dir = Path(paper_dir)
    paper_dir.mkdir(parents=True, exist_ok=True)
    context_path = paper_dir / "context.json"
    context_path.write_text(json.dumps(paper, indent=2, ensure_ascii=False), encoding="utf-8")
    text_clean = str(paper.get("text_clean", "") or "").strip()
    if text_clean:
        (paper_dir / "paper_context.txt").write_text(text_clean, encoding="utf-8")
    return context_path
