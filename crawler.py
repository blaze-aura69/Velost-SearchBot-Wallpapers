import asyncio, json, os, time
from urllib.parse import urlparse, urljoin
from huggingface_hub import HfApi
from playwright.async_api import async_playwright

HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_REPO = "blaze-aura69/Wallpapers"
TARGET_FILE = "wallpapers.jsonl"

MAX_URLS = 1_000_000
MAX_RUNTIME = 4 * 3600  # 4 hours
CONCURRENCY = 10        # browsers are heavier, keep lower concurrency

seen = set()
results = []

async def process_page(page, url, queue):
    try:
        await page.goto(url, timeout=30000)
        html = await page.content()

        # Extract images
        imgs = await page.query_selector_all("img")
        for img in imgs:
            src = await img.get_attribute("src")
            if not src or src in seen:
                continue

            domain = urlparse(url).netloc
            domain_url = f"https://{domain}"
            favicon = f"https://icons.duckduckgo.com/ip3/{domain}.ico"
            source_name = domain.split(".")[0]
            title = await page.title()

            entry = {
                "url": src,
                "title": title,
                "favicon": favicon,
                "source_name": source_name,
                "domain_url": domain_url
            }

            seen.add(src)
            results.append(entry)

        # Discover links
        links = await page.query_selector_all("a")
        for a in links:
            href = await a.get_attribute("href")
            if href:
                link = urljoin(url, href)
                if link.startswith("http") and link not in seen:
                    await queue.put(link)

    except Exception:
        return

async def crawl(seed_urls):
    start = time.time()
    queue = asyncio.Queue()
    for u in seed_urls:
        await queue.put(u)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        async def worker():
            while True:
                if len(results) >= MAX_URLS or (time.time() - start) > MAX_RUNTIME:
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
        await queue.join()
        for t in tasks:
            t.cancel()
        await browser.close()

def write_jsonl_prepend(entries, filename):
    old_lines = []
    if os.path.exists(filename):
        with open(filename, "r") as f:
            old_lines = f.readlines()
    with open(filename, "w") as f:
        for e in entries[::-1]:
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
