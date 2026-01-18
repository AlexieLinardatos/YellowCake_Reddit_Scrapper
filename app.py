from __future__ import annotations

import json
import os
from typing import Any, Dict

from flask import Flask, jsonify, request, send_from_directory
import requests

import yellowcake_api


app = Flask(__name__, static_folder=".", static_url_path="")


@app.get("/")
def index() -> Any:
    return send_from_directory(".", "index.html")


@app.post("/run")
def run_extract() -> Any:
    payload: Dict[str, Any] = request.get_json(silent=True) or {}
    subreddit_url = str(payload.get("url", "")).strip()
    api_key = str(payload.get("api_key", "")).strip() or os.getenv("YELLOWCAKE_API_KEY", "")
    save_path = str(payload.get("save_path", "")).strip()

    if not subreddit_url:
        return jsonify({"error": "Missing subreddit URL."}), 400
    if not api_key:
        return jsonify({"error": "Missing API key. Provide one or set YELLOWCAKE_API_KEY on the server."}), 400

    max_posts = int(payload.get("posts", 50))
    comments_per_post = int(payload.get("comments", 25))
    sort = str(payload.get("sort", "top")).strip().lower()
    use_yellowcake = bool(payload.get("use_yellowcake", True))

    try:
        if use_yellowcake:
            posts = yellowcake_api.fetch_subreddit_posts_yellowcake(
                subreddit_url,
                api_key=api_key,
                max_posts=max_posts,
                comments_per_post=comments_per_post,
                sort=sort,
            )
        else:
            posts = yellowcake_api.fetch_subreddit_posts(
                subreddit_url,
                max_posts=max_posts,
                comments_per_post=comments_per_post,
                sort=sort,
            )
    except requests.RequestException as exc:
        return jsonify(
            {
                "error": "Yellowcake request failed.",
                "detail": str(exc),
                "used_yellowcake": use_yellowcake,
                "api_key_present": bool(api_key),
            }
        ), 502
    except Exception as exc:
        return jsonify(
            {
                "error": "Extraction failed.",
                "detail": str(exc),
                "used_yellowcake": use_yellowcake,
                "api_key_present": bool(api_key),
            }
        ), 500

    if save_path:
        save_dir = os.path.dirname(save_path)
        if save_dir and not os.path.isdir(save_dir):
            return jsonify({"error": "Save path directory does not exist."}), 400
        with open(save_path, "w", encoding="utf-8") as handle:
            json.dump(posts, handle, ensure_ascii=True, indent=2)

    return jsonify(
        {
            "data": posts,
            "count": len(posts),
            "used_yellowcake": use_yellowcake,
            "api_key_present": bool(api_key),
            "saved_path": save_path or None,
        }
    )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="127.0.0.1", port=port, debug=True)
