import aiohttp, asyncio, json, os, time
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from huggingface_hub import HfApi
import urllib.robotparser

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "blazeaura69/Wallpapers"
TARGET_FILE = "wallpapers.jsonl"

MAX_URLS = 1_000_000
MAX_RUNTIME = 4 * 3600  # 4 hours
CONCURRENCY = 200       # tune for 4 vCPU

seen = set()
results = {}
robots_cache = {}

def can_fetch(url, user_agent="*"):
    domain = urlparse(url).netloc
    if domain not in robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"https://{domain}/robots.txt")
        try:
            rp.read()
        except Exception:
            return False  # if robots.txt not accessible, be safe
        robots_cache[domain] = rp
    return robots_cache[domain].can_fetch(user_agent, url)

async def fetch(session, url):
    if not can_fetch(url):
        return
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200 or "text/html" not in resp.headers.get("content-type",""):
                return
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")

            title = soup.title.string.strip() if soup.title else ""
            domain = urlparse(url).netloc
            domain_url = f"https://{domain}"
            favicon = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
            source_name = domain.split(".")[0]

            entry = {
                "url": url,
                "title": title,
                "favicon": favicon,
                "source_name": source_name,
                "domain_url": domain_url
            }

            if url not in seen:
                seen.add(url)
                results[url] = entry

    except Exception:
        return

async def crawl(seed_urls):
    start = time.time()
    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(CONCURRENCY)
        tasks = []

        async def bound_fetch(u):
            async with sem:
                await fetch(session, u)

        for u in seed_urls:
            if len(results) >= MAX_URLS or (time.time() - start) > MAX_RUNTIME:
                break
            tasks.append(bound_fetch(u))

        await asyncio.gather(*tasks)

def write_jsonl_prepend(entries, filename):
    old_lines = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            old_lines = f.readlines()
    with open(filename, "w") as f:
        for e in list(entries.values())[::-1]:  # newest first
            f.write(json.dumps(e) + "\n")
        f.writelines(old_lines)

def upload_to_hf(filename):
    api = HfApi()
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=TARGET_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )

if __name__ == "__main__":
    # Replace with your own discovery logic or seed list
    seeds = [
        "https://wallhaven.cc/waterfall",
        "https://unsplash.com/nature",
    ]
    asyncio.run(crawl(seeds))
    write_jsonl_prepend(results, TARGET_FILE)
    upload_to_hf(TARGET_FILE)
