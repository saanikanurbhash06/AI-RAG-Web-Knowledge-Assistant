import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ANAKIN_API_KEY")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
}

def scrape_with_anakin(url):

    payload = {
        "url": url,
        "country": "us",
        "useBrowser": False,
        "generateJson": False
    }

    # 1. Submit job
    response = requests.post(
        "https://api.anakin.io/v1/url-scraper",
        headers=HEADERS,
        json=payload
    )

    data = response.json()

    if "jobId" not in data:
        raise Exception(f"API Error: {data}")

    job_id = data["jobId"]

    # 2. Poll result
    while True:
        result = requests.get(
            f"https://api.anakin.io/v1/url-scraper/{job_id}",
            headers=HEADERS
        )

        result_data = result.json()
        status = result_data.get("status")

        if status == "completed":
            break
        elif status == "failed":
            raise Exception(result_data)

        time.sleep(2)

    # 3. EXTRACT ONLY REAL TEXT (IMPORTANT FIX)
    content = (
        result_data.get("markdown")
        or result_data.get("data", {}).get("markdown")
        or result_data.get("result", {}).get("markdown")
        or ""
    )

    return content