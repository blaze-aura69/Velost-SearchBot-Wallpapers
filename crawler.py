import aiohttp, asyncio, json, os, time
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
from huggingface_hub import HfApi
from PIL import Image
import requests
from io import BytesIO

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "blaze-aura69/Wallpapers"
TARGET_FILE = "wallpapers.jsonl"

MAX_URLS = 1_000_000
MAX_RUNTIME = 4 * 3600  # 4 hours
CONCURRENCY = 200

seen_urls = set()
results = []

async def fetch(session, url, queue):
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200 or "text/html" not in resp.headers.get("content-type",""):
                return
            html = await resp.text()
            soup = BeautifulSoup(html, "html.parser")

            # Extract images
            for img in soup.find_all("img"):
                src = img.get("src") or img.get("data-src")
                if not src or src in seen_urls:
                    continue

                # Try to check aspect ratio
                try:
                    r = requests.get(src, timeout=10)
                    im = Image.open(BytesIO(r.content))
                    w, h = im.size
                    ratio = w / h
                    if abs(ratio - (9/16)) > 0.05:
                        continue
                except Exception:
                    continue

                domain = urlparse(url).netloc
                domain_url = f"https://{domain}"
                favicon = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
                source_name = domain.split(".")[0]
                title = soup.title.string.strip() if soup.title else ""

                entry = {
                    "url": src,
                    "title": title,
                    "favicon": favicon,
                    "source_name": source_name,
                    "domain_url": domain_url
                }

                seen_urls.add(src)
                results.append(entry)

            # Discover new links
            for a in soup.find_all("a", href=True):
                link = urljoin(url, a["href"])
                if link not in seen_urls and link.startswith("http"):
                    await queue.put(link)

    except Exception:
        return

async def crawl(seed_urls):
    start = time.time()
    queue = asyncio.Queue()
    for u in seed_urls:
        await queue.put(u)

    async with aiohttp.ClientSession() as session:
        sem = asyncio.Semaphore(CONCURRENCY)

        async def worker():
            while True:
                if len(results) >= MAX_URLS or (time.time() - start) > MAX_RUNTIME:
                    return
                try:
                    url = await queue.get()
                except Exception:
                    return
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                async with sem:
                    await fetch(session, url, queue)
                queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
        await queue.join()
        for t in tasks:
            t.cancel()

def write_jsonl_prepend(entries, filename):
    old_lines = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            old_lines = f.readlines()
    with open(filename, "w") as f:
        for e in entries[::-1]:  # newest first
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
    seeds = [
        "https://unsplash.com/wallpapers/nature",
        "https://unsplash.com/wallpapers/animals",
        "https://www.pexels.com/search/nature%20wallpaper/",
        "https://www.pexels.com/search/animal%20wallpaper/",
        "https://pixabay.com/images/search/nature%20wallpaper/",
        "https://pixabay.com/images/search/animal%20wallpaper/"
    ]
    asyncio.run(crawl(seeds))
    write_jsonl_prepend(results, TARGET_FILE)
    upload_to_hf(TARGET_FILE)
