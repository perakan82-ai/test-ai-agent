#!/usr/bin/env python3
"""reach.py - ให้ AI อ่านเว็บ / YouTube / GitHub / RSS
ใช้: python reach.py <web|yt|gh|rss|doctor> <url หรือ owner/repo>
ทำงานได้ด้วย Python ล้วน ๆ ถ้ามี trafilatura / feedparser / yt-dlp จะได้ผลดีขึ้น
"""
import sys, re, json, html, glob, os, shutil, subprocess, tempfile
import urllib.request
import xml.etree.ElementTree as ET

UA = {"User-Agent": "Mozilla/5.0 (reach.py)"}
MAX = 20000  # ตัดข้อความไม่ให้ยาวเกิน


def fetch(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def strip_html(raw):
    raw = re.sub(r"(?is)<(script|style|nav|footer|header|noscript).*?</\1>", " ", raw)
    raw = re.sub(r"(?s)<[^>]+>", " ", raw)
    return re.sub(r"\s+", " ", html.unescape(raw)).strip()


def cmd_web(url):
    try:
        import trafilatura
        text = trafilatura.extract(trafilatura.fetch_url(url) or "")
        if text:
            return text[:MAX]
    except ImportError:
        pass
    return strip_html(fetch(url))[:MAX]


def cmd_yt(url):
    if not shutil.which("yt-dlp"):
        return "ERROR: ยังไม่มี yt-dlp -> pip install yt-dlp"
    d = tempfile.mkdtemp()
    subprocess.run(["yt-dlp", "--skip-download", "--write-subs", "--write-auto-subs",
                    "--sub-langs", "th,en.*,en", "--sub-format", "vtt",
                    "-o", os.path.join(d, "%(title)s.%(ext)s"), url],
                   capture_output=True, text=True)
    files = glob.glob(os.path.join(d, "*.vtt"))
    if not files:
        return "ERROR: คลิปนี้ไม่มีซับ"
    lines, last = [], None
    for ln in open(files[0], encoding="utf-8"):
        ln = re.sub(r"<[^>]+>", "", ln).strip()
        if not ln or "-->" in ln or ln.startswith(("WEBVTT", "Kind:", "Language:")):
            continue
        if ln != last:
            lines.append(ln)
            last = ln
    title = os.path.basename(files[0]).rsplit(".", 2)[0]
    return f"# {title}\n" + " ".join(lines)[:MAX]


def cmd_gh(repo):
    repo = re.sub(r"^https?://github\.com/", "", repo).strip("/")
    owner_repo = "/".join(repo.split("/")[:2])
    info = json.loads(fetch(f"https://api.github.com/repos/{owner_repo}"))
    out = [f"# {info['full_name']}", info.get("description") or "",
           f"stars: {info['stargazers_count']}  language: {info.get('language')}  "
           f"updated: {info['pushed_at']}"]
    try:
        readme = fetch(f"https://raw.githubusercontent.com/{owner_repo}/"
                       f"{info['default_branch']}/README.md")
        out.append("\n## README\n" + readme)
    except Exception:
        out.append("(ไม่พบ README.md)")
    return "\n".join(out)[:MAX]


def cmd_rss(url, n=15):
    try:
        import feedparser
        f = feedparser.parse(url)
        return "\n".join(f"- {e.get('title')} | {e.get('published', '')} | {e.get('link')}"
                         for e in f.entries[:n])
    except ImportError:
        pass
    root = ET.fromstring(fetch(url))
    items = root.findall(".//item") or root.findall(".//{http://www.w3.org/2005/Atom}entry")
    out = []
    for it in items[:n]:
        g = lambda tag: (it.findtext(tag) or it.findtext("{http://www.w3.org/2005/Atom}" + tag) or "")
        link = g("link")
        if not link:
            el = it.find("{http://www.w3.org/2005/Atom}link")
            link = el.get("href") if el is not None else ""
        out.append(f"- {g('title').strip()} | {g('pubDate') or g('updated')} | {link}")
    return "\n".join(out)


def cmd_doctor(_=None):
    rows = []
    for mod in ("trafilatura", "feedparser"):
        try:
            __import__(mod); rows.append(f"OK   {mod}")
        except ImportError:
            rows.append(f"--   {mod} (ไม่บังคับ: pip install {mod})")
    rows.append(("OK   yt-dlp" if shutil.which("yt-dlp") else "--   yt-dlp (ต้องมีถ้าจะอ่าน YouTube)"))
    try:
        fetch("https://api.github.com"); rows.append("OK   อินเทอร์เน็ต")
    except Exception as e:
        rows.append(f"FAIL อินเทอร์เน็ต: {e}")
    return "\n".join(rows)


CMDS = {"web": cmd_web, "yt": cmd_yt, "gh": cmd_gh, "rss": cmd_rss, "doctor": cmd_doctor}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CMDS:
        print(__doc__); sys.exit(1)
    try:
        print(CMDS[sys.argv[1]](*sys.argv[2:3]))
    except Exception as e:
        print(f"ERROR: {e}"); sys.exit(1)
