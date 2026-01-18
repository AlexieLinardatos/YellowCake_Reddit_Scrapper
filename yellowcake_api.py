import argparse
import json
import os
import re
import time
from urllib.parse import urlparse
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests


DEFAULT_ENDPOINT = "https://api.yellowcake.dev/v1/extract-stream"


def _parse_sse_lines(lines: Iterable[bytes]) -> Iterable[Tuple[str, str]]:
    event = ""
    data_lines: List[str] = []
    for raw in lines:
        if raw is None:
            continue
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            if data_lines:
                yield event or "message", "\n".join(data_lines)
                event = ""
                data_lines = []
            continue
        if line.startswith("event:"):
            event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].strip())
    if data_lines:
        yield event or "message", "\n".join(data_lines)


def yellowcake_extract(
    url: str,
    prompt: str,
    api_key: str,
    *,
    throttle: Optional[bool] = None,
    login_url: Optional[str] = None,
    authorized_urls: Optional[List[str]] = None,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout_sec: int = 1800,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"url": url, "prompt": prompt}
    if throttle is not None:
        payload["throttle"] = bool(throttle)
    if login_url:
        payload["loginURL"] = login_url
    if authorized_urls:
        payload["authorizedURLs"] = authorized_urls

    headers = {
        "Content-Type": "application/json",
        "X-API-Key": api_key,
    }

    resp = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        stream=True,
        timeout=timeout_sec,
    )
    resp.raise_for_status()

    last_json: Optional[Dict[str, Any]] = None
    for event, data in _parse_sse_lines(resp.iter_lines()):
        if not data:
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            parsed = None
        if parsed is not None:
            last_json = parsed
        if event == "complete" and parsed is not None:
            return parsed

    if last_json is None:
        raise RuntimeError("Yellowcake stream ended without a JSON payload.")
    return last_json


POSITIVE_PATTERNS = [
    (r"\b(i own|i own the|i've owned|i have owned)\b", 0.45),
    (r"\b(i bought|i purchased|i ordered|i picked up)\b", 0.35),
    (r"\b(i use|i've used|i have used|i'm using|i am using)\b", 0.25),
    (r"\b(i've had|i have had|i had)\b", 0.20),
    (r"\b(my|mine)\b", 0.10),
    (r"\bfor my (kid|kids|son|daughter|wife|husband|partner)\b", 0.20),
    (r"\bwe (own|use|bought|purchased)\b", 0.20),
]

NEGATIVE_PATTERNS = [
    (r"\b(i don't own|i do not own|never owned|haven't owned)\b", -0.60),
    (r"\b(i haven't bought|i have not bought|haven't purchased)\b", -0.45),
    (r"\b(i'm thinking about buying|considering buying|plan to buy|might buy)\b", -0.25),
    (r"\b(i wish i had|if i had)\b", -0.20),
]


def ownership_confidence(comment: str) -> float:
    text = (comment or "").lower()
    if not text:
        return 0.0

    score = 0.10
    for pat, weight in POSITIVE_PATTERNS:
        if re.search(pat, text):
            score += weight
    for pat, weight in NEGATIVE_PATTERNS:
        if re.search(pat, text):
            score += weight

    if re.search(r"\b(serial|model|version|sku|order)\b", text):
        score += 0.10
    if re.search(r"\b(mine broke|my broke|my unit|my device)\b", text):
        score += 0.10

    score = max(0.0, min(1.0, score))
    return round(score, 3)


NAME_KEYS = ["name", "full_name", "person", "author", "username", "user"]
COMMENT_KEYS = ["comment", "text", "review", "content", "body"]
POST_TITLE_KEYS = ["title", "post_title", "headline", "name"]
POST_AUTHOR_KEYS = ["username", "author", "user", "poster", "posted_by"]
POST_COMMENTS_KEYS = ["comments", "replies", "responses"]


def simple_sentiment(comment: str) -> float:
    text = (comment or "").lower()
    if not text:
        return 0.0
    pos_words = {
        "love", "great", "awesome", "amazing", "helpful", "easy", "perfect", "like",
        "favorite", "best", "nice", "solid", "happy", "satisfied", "recommend",
    }
    neg_words = {
        "hate", "bad", "broken", "frustrating", "annoying", "disappointing", "terrible",
        "awful", "poor", "worse", "worst", "return", "refund", "defect", "issue",
    }
    tokens = re.findall(r"[a-z']+", text)
    if not tokens:
        return 0.0
    pos = sum(t in pos_words for t in tokens)
    neg = sum(t in neg_words for t in tokens)
    raw = (pos - neg) / max(1, (pos + neg))
    return round(max(-1.0, min(1.0, raw)), 3)


def extract_name_comment(item: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    name = None
    for k in NAME_KEYS:
        if k in item and str(item[k]).strip():
            name = str(item[k]).strip()
            break
    comment = None
    for k in COMMENT_KEYS:
        if k in item and str(item[k]).strip():
            comment = str(item[k]).strip()
            break
    return name, comment


def _first_list_value(data: Any) -> Optional[List[Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "items", "results", "posts", "rows"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        for value in data.values():
            if isinstance(value, list):
                return value
    return None


def _extract_post_rows(payload: Any) -> List[Dict[str, Any]]:
    rows = _first_list_value(payload)
    if isinstance(rows, list):
        return [r for r in rows if isinstance(r, dict)]
    return []


def ownership_scores_from_yellowcake(data: Any, *, include_sentiment: bool = False) -> List[Dict[str, Any]]:
    if isinstance(data, dict) and "data" in data:
        rows = data.get("data", [])
    else:
        rows = data

    if not isinstance(rows, list):
        raise ValueError("Yellowcake payload is not a list of records.")

    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        name, comment = extract_name_comment(item)
        if not name or not comment:
            continue
        row = {"name": name, "confidence": ownership_confidence(comment)}
        if include_sentiment:
            row["sentiment"] = simple_sentiment(comment)
        out.append(row)
    return out


def _normalize_subreddit_url(subreddit_url: str) -> str:
    url = subreddit_url.strip()
    if not url.startswith("http"):
        url = "https://www.reddit.com/r/" + url.strip("/").replace("r/", "")
    return url


def _listing_sort(sort: Optional[str]) -> str:
    sort_value = (sort or "hot").strip().lower()
    if sort_value not in {"top", "new", "hot"}:
        return "hot"
    return sort_value


def _listing_url(base: str, sort: Optional[str]) -> str:
    sort_value = _listing_sort(sort)
    return f"{base}/{sort_value}.json"


def _listing_params(sort: Optional[str], limit: int, after: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"limit": limit}
    if after:
        params["after"] = after
    if _listing_sort(sort) == "top":
        params["t"] = "all"
    return params


def fetch_post_comments(
    base_url: str,
    permalink: str,
    *,
    max_comments: int = 50,
    sort: str = "top",
    sleep_ms: int = 300,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    comments: List[Dict[str, Any]] = []
    headers = {"User-Agent": "yellowcake-cli/1.0 (contact: local)"}
    params = {"limit": max_comments, "sort": _listing_sort(sort)}

    url = f"{base_url}{permalink}.json"
    for attempt in range(max_retries + 1):
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        if resp.status_code == 429 and attempt < max_retries:
            time.sleep(max(0.0, sleep_ms / 1000.0) * (attempt + 1))
            continue
        resp.raise_for_status()
        break
    payload = resp.json()
    if not isinstance(payload, list) or len(payload) < 2:
        return comments

    children = payload[1].get("data", {}).get("children", [])
    for child in children:
        if child.get("kind") != "t1":
            continue
        data = child.get("data", {})
        author = (data.get("author") or "").strip()
        body = (data.get("body") or "").strip()
        if author and body:
            comments.append({"commenter_username": author, "comment": body})
        if len(comments) >= max_comments:
            break

    return comments


def fetch_subreddit_posts(
    subreddit_url: str,
    *,
    max_posts: int = 100,
    comments_per_post: int = 50,
    sort: str = "top",
    sleep_ms: int = 300,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    url = _normalize_subreddit_url(subreddit_url)
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    out: List[Dict[str, Any]] = []
    after = None
    headers = {"User-Agent": "yellowcake-cli/1.0 (contact: local)"}

    while len(out) < max_posts:
        limit = min(100, max_posts - len(out))
        params = _listing_params(sort, limit, after)

        listing_url = _listing_url(base, sort)
        for attempt in range(max_retries + 1):
            resp = requests.get(listing_url, headers=headers, params=params, timeout=20)
            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(max(0.0, sleep_ms / 1000.0) * (attempt + 1))
                continue
            resp.raise_for_status()
            break
        payload = resp.json()

        children = payload.get("data", {}).get("children", [])
        if not children and listing_url != f"{base}.json":
            fallback_url = f"{base}.json"
            resp = requests.get(fallback_url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            children = payload.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            data = child.get("data", {})
            title = (data.get("title") or "").strip()
            author = (data.get("author") or "").strip()
            permalink = (data.get("permalink") or "").strip()
            if title and author:
                post = {"title": title, "username": author}
                if permalink:
                    try:
                        post["comments"] = fetch_post_comments(
                            f"{parsed.scheme}://{parsed.netloc}",
                            permalink,
                            max_comments=comments_per_post,
                            sort=sort,
                            sleep_ms=sleep_ms,
                            max_retries=max_retries,
                        )
                    except requests.RequestException:
                        post["comments"] = []
                out.append(post)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
            if len(out) >= max_posts:
                break

        after = payload.get("data", {}).get("after")
        if not after:
            break

    return out


def fetch_subreddit_posts_yellowcake(
    subreddit_url: str,
    *,
    api_key: str,
    max_posts: int = 100,
    comments_per_post: int = 50,
    sort: str = "top",
    throttle: Optional[bool] = None,
) -> List[Dict[str, Any]]:
    prompt = (
        "Extract posts from this subreddit page. "
        f"Return up to {max_posts} posts. "
        "Each post should have: title, username, comments. "
        f"Comments should include up to {comments_per_post} items, each with commenter_username and comment. "
        "Return a JSON array only."
    )
    payload = yellowcake_extract(
        subreddit_url,
        prompt,
        api_key,
        throttle=throttle,
    )
    rows = _extract_post_rows(payload)
    if not rows:
        raise ValueError("Yellowcake payload did not contain a list of posts.")
    out = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        post_obj = item
        if not any(k in item for k in POST_TITLE_KEYS + POST_AUTHOR_KEYS) and isinstance(item.get("post"), dict):
            post_obj = item["post"]

        title = ""
        for key in POST_TITLE_KEYS:
            if post_obj.get(key):
                title = str(post_obj.get(key)).strip()
                break
        username = ""
        for key in POST_AUTHOR_KEYS:
            if post_obj.get(key):
                username = str(post_obj.get(key)).strip()
                break
        comments = None
        for key in POST_COMMENTS_KEYS:
            if key in item:
                comments = item.get(key)
                break
        if comments is None:
            comments = item.get("comment") or item.get("responses") or post_obj.get("comments") or []
        if not title or not username:
            continue
        clean_comments = []
        if isinstance(comments, list):
            for c in comments[:comments_per_post]:
                if not isinstance(c, dict):
                    continue
                commenter = (c.get("commenter_username") or c.get("username") or c.get("author") or "").strip()
                comment = (c.get("comment") or c.get("text") or c.get("body") or "").strip()
                if commenter and comment:
                    clean_comments.append({"commenter_username": commenter, "comment": comment})
        out.append({"title": title, "username": username, "comments": clean_comments})
        if len(out) >= max_posts:
            break
    return out


def fetch_subreddit_post_permalinks(
    subreddit_url: str,
    *,
    max_posts: int = 100,
    sort: str = "top",
    sleep_ms: int = 300,
    max_retries: int = 3,
) -> List[str]:
    url = _normalize_subreddit_url(subreddit_url)
    parsed = urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}{parsed.path}".rstrip("/")
    out: List[str] = []
    after = None
    headers = {"User-Agent": "yellowcake-cli/1.0 (contact: local)"}

    while len(out) < max_posts:
        limit = min(100, max_posts - len(out))
        params = _listing_params(sort, limit, after)

        listing_url = _listing_url(base, sort)
        for attempt in range(max_retries + 1):
            resp = requests.get(listing_url, headers=headers, params=params, timeout=20)
            if resp.status_code == 429 and attempt < max_retries:
                time.sleep(max(0.0, sleep_ms / 1000.0) * (attempt + 1))
                continue
            resp.raise_for_status()
            break

        payload = resp.json()
        children = payload.get("data", {}).get("children", [])
        if not children and listing_url != f"{base}.json":
            fallback_url = f"{base}.json"
            resp = requests.get(fallback_url, headers=headers, params=params, timeout=20)
            resp.raise_for_status()
            payload = resp.json()
            children = payload.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            data = child.get("data", {})
            permalink = (data.get("permalink") or "").strip()
            if permalink:
                out.append(f"{parsed.scheme}://{parsed.netloc}{permalink}")
            if len(out) >= max_posts:
                break

        after = payload.get("data", {}).get("after")
        if not after:
            break

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    return out


def fetch_subreddit_posts_via_yellowcake_per_post(
    subreddit_url: str,
    *,
    api_key: str,
    max_posts: int = 100,
    comments_per_post: int = 50,
    sort: str = "top",
    throttle: Optional[bool] = None,
    sleep_ms: int = 300,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    permalinks = fetch_subreddit_post_permalinks(
        subreddit_url,
        max_posts=max_posts,
        sort=sort,
        sleep_ms=sleep_ms,
        max_retries=max_retries,
    )
    out: List[Dict[str, Any]] = []
    last_error: Optional[Exception] = None
    for link in permalinks:
        prompt = (
            "Extract the post title, the author's username, and comments from this Reddit thread. "
            f"Return up to {comments_per_post} comments. "
            "Return a JSON object with keys: title, username, comments. "
            "Each comment has: commenter_username, comment."
        )
        try:
            payload = yellowcake_extract(
                link,
                prompt,
                api_key,
                throttle=throttle,
            )
        except requests.RequestException as exc:
            last_error = exc
            continue

        item = payload.get("data", payload)
        if isinstance(item, list) and item:
            item = item[0]
        if not isinstance(item, dict):
            continue

        title = (item.get("title") or "").strip()
        username = (item.get("username") or item.get("author") or "").strip()
        comments = item.get("comments") or []
        if not title or not username:
            continue

        clean_comments = []
        if isinstance(comments, list):
            for c in comments[:comments_per_post]:
                if not isinstance(c, dict):
                    continue
                commenter = (c.get("commenter_username") or c.get("username") or c.get("author") or "").strip()
                comment = (c.get("comment") or c.get("text") or c.get("body") or "").strip()
                if commenter and comment:
                    clean_comments.append({"commenter_username": commenter, "comment": comment})

        out.append({"title": title, "username": username, "comments": clean_comments})
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)

    if not out and last_error is not None:
        raise last_error

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Yellowcake extract + ownership confidence scoring.")
    parser.add_argument("--url", required=False, help="Target URL to extract from.")
    parser.add_argument("--prompt", required=False, help="Extraction prompt for Yellowcake.")
    parser.add_argument("--api-key", default=os.getenv("YELLOWCAKE_API_KEY"), help="Yellowcake API key.")
    parser.add_argument("--login-url", default=None, help="Optional login URL for auth-walled sites.")
    parser.add_argument("--authorized-urls", default=None, help="Comma-separated list of authorized URLs.")
    parser.add_argument("--throttle", action="store_true", help="Enable throttle mode.")
    parser.add_argument("--print-reddit-prompt", action="store_true", help="Print a suggested Reddit extraction prompt.")
    parser.add_argument("--dump-raw", default=None, help="Optional path to write raw Yellowcake JSON.")
    parser.add_argument("--include-sentiment", action="store_true", help="Include a sentiment score per comment.")
    parser.add_argument("--subreddit-url", default=None, help="Fetch subreddit posts (title + author) without Yellowcake.")
    parser.add_argument("--max-posts", type=int, default=100, help="Max subreddit posts to fetch.")
    parser.add_argument("--comments-per-post", type=int, default=50, help="Max comments per post to fetch.")
    parser.add_argument("--sort", default="top", choices=["top", "new", "hot"], help="Sort order for subreddit posts.")
    parser.add_argument("--output-json", default=None, help="Write output JSON to a file.")
    parser.add_argument("--sleep-ms", type=int, default=300, help="Delay between requests to avoid rate limits.")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries when rate-limited (HTTP 429).")
    parser.add_argument("--use-yellowcake", action="store_true", help="Use Yellowcake for subreddit extraction.")
    args = parser.parse_args()

    if args.print_reddit_prompt:
        print(
            "Extract each comment's username (author) and comment text from this Reddit thread. "
            "Return a JSON array with keys: username, comment. Only include actual comment content."
        )
        return

    if args.subreddit_url:
        if args.use_yellowcake:
            if not args.api_key:
                raise SystemExit("Missing API key. Pass --api-key or set YELLOWCAKE_API_KEY.")
            posts = fetch_subreddit_posts_via_yellowcake_per_post(
                args.subreddit_url,
                api_key=args.api_key,
                max_posts=args.max_posts,
                comments_per_post=args.comments_per_post,
                sort=args.sort,
                throttle=args.throttle or None,
                sleep_ms=args.sleep_ms,
                max_retries=args.max_retries,
            )
        else:
            posts = fetch_subreddit_posts(
                args.subreddit_url,
                max_posts=args.max_posts,
                comments_per_post=args.comments_per_post,
                sort=args.sort,
                sleep_ms=args.sleep_ms,
                max_retries=args.max_retries,
            )
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(posts, f, ensure_ascii=True, indent=2)
            print(f"Wrote {len(posts)} posts to {args.output_json}")
        else:
            print(json.dumps(posts, indent=2))
        return

    if not args.url or not args.prompt:
        raise SystemExit("Missing --url/--prompt. Use --subreddit-url for subreddit posts.")

    if not args.api_key:
        raise SystemExit("Missing API key. Pass --api-key or set YELLOWCAKE_API_KEY.")

    authorized = None
    if args.authorized_urls:
        authorized = [u.strip() for u in args.authorized_urls.split(",") if u.strip()]

    payload = yellowcake_extract(
        args.url,
        args.prompt,
        args.api_key,
        throttle=args.throttle or None,
        login_url=args.login_url,
        authorized_urls=authorized,
    )

    if args.dump_raw:
        with open(args.dump_raw, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=True, indent=2)

    people = ownership_scores_from_yellowcake(payload, include_sentiment=args.include_sentiment)
    print(json.dumps(people, indent=2))


if __name__ == "__main__":
    main()
