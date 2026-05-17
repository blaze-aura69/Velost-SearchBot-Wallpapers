import asyncio, json, os, time
from urllib.parse import urlparse, urljoin
from huggingface_hub import HfApi
from playwright.async_api import async_playwright

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "blaze-aura69/Wallpapers"
TARGET_FILE = "wallpapers.jsonl"

# Limits
ALLOWED_DOMAINS = ["unsplash.com", "pexels.com", "pixabay.com"]
MAX_URLS = 1_000_000
MAX_RUNTIME = 4 * 3600   # 4 hours
UPLOAD_INTERVAL = 300    # 5 minutes
CONCURRENCY = 3          # keep low for Playwright

seen = set()
results = []
start_time = None

def is_allowed(url):
    domain = urlparse(url).netloc
    return any(domain.endswith(d) for d in ALLOWED_DOMAINS)

def append_jsonl(entry, filename):
    with open(filename, "a") as f:
        f.write(json.dumps(entry) + "\n")
    if len(results) % 100 == 0:
        print(f"[WRITE] Total {len(results)} entries written so far")

def upload_to_hf(filename):
    print(f"[UPLOAD] Uploading {filename} to Hugging Face dataset {DATASET_REPO}")
    api = HfApi()
    api.upload_file(
        path_or_fileobj=filename,
        path_in_repo=TARGET_FILE,
        repo_id=DATASET_REPO,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    print("[UPLOAD] Completed")

async def periodic_uploader():
    while True:
        await asyncio.sleep(UPLOAD_INTERVAL)
        if os.path.exists(TARGET_FILE):
            upload_to_hf(TARGET_FILE)

async def process_page(page, url, queue):
    try:
        print(f"[FETCH] Visiting {url}")
        await page.goto(url, timeout=30000)
        title = await page.title()

        # Extract images
        imgs = await page.query_selector_all("img")
        print(f"[INFO] Found {len(imgs)} <img> tags on {url}")
        for img in imgs:
            src = await img.get_attribute("src")
            if not src or src in seen:
                continue
            domain = urlparse(url).netloc
            entry = {
                "url": src,
                "title": title,
                "favicon": f"https://icons.duckduckgo.com/ip3/{domain}.ico",
                "source_name": domain.split(".")[0],
                "domain_url": f"https://{domain}"
            }
            seen.add(src)
            results.append(entry)
            append_jsonl(entry, TARGET_FILE)

        # Discover links (only allowed domains)
        links = await page.query_selector_all("a")
        print(f"[INFO] Found {len(links)} links on {url}")
        for a in links:
            href = await a.get_attribute("href")
            if href:
                link = urljoin(url, href)
                if link.startswith("http") and is_allowed(link) and link not in seen:
                    await queue.put(link)
                    print(f"[QUEUE] Added {link}")

    except Exception as e:
        print(f"[ERROR] {url}: {e}")

async def crawl(seeds):
    global start_time
    start_time = time.time()
    queue = asyncio.Queue()
    for u in seeds:
        await queue.put(u)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        async def worker():
            while True:
                # Stop if limits reached
                if len(results) >= MAX_URLS:
                    print("[STOP] Max URL limit reached")
                    return
                if (time.time() - start_time) > MAX_RUNTIME:
                    print("[STOP] Max runtime reached (4h)")
                    return
                try:
                    url = await queue.get()
                except Exception:
                    return
                if url in seen:
                    continue
                seen.add(url)
                page = await browser.new_page()
                await process_page(page, url, queue)
                await page.close()
                queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(CONCURRENCY)]
        uploader_task = asyncio.create_task(periodic_uploader())
        await queue.join()
        for t in tasks:
            t.cancel()
        uploader_task.cancel()
        await browser.close()

if __name__ == "__main__":
    if not os.path.exists(TARGET_FILE):
        open(TARGET_FILE, "w").close()
        print(f"[INIT] Created empty {TARGET_FILE}")

    seeds = [
        "https://unsplash.com/wallpapers/nature",
        "https://unsplash.com/wallpapers/animals",
        "https://www.pexels.com/search/nature%20wallpaper/",
        "https://www.pexels.com/search/animal%20wallpaper/",
        "https://pixabay.com/images/search/nature%20wallpaper/",
        "https://pixabay.com/images/search/animal%20wallpaper/"
    ]
    asyncio.run(crawl(seeds))
    upload_to_hf(TARGET_FILE)
    print(f"[DONE] Crawler finished with {len(results)} images collected in {(time.time()-start_time)/60:.1f} minutes")
