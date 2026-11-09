"""One polite, retrying HTTP helper shared by every API client."""

import time

import requests

USER_AGENT = "gb-battery-trader (portfolio project)"


def get_json(url: str, params: dict | None = None, retries: int = 5, timeout: int = 60):
    """GET a URL and return the parsed JSON, retrying on rate limits and outages.

    Waits 1, 2, 4, 8... seconds between attempts (exponential back-off).
    A 400-type error means our request is wrong, so it fails straight away.
    """
    for attempt in range(retries):
        try:
            r = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}
            )
        except (requests.ConnectionError, requests.Timeout):
            time.sleep(2**attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2**attempt)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"Request failed after {retries} attempts: {url} {params}")
