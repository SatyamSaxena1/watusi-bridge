import json
import sys
import requests


def main(url: str = "http://localhost:8000/auto-reply"):
    payload = {
        "date": "2025-09-08 10:00:00",
        "jid": "12345@s.whatsapp.net",
        "name": "Alice",
        "text": "Hey! Are you free for lunch today?",
    }
    r = requests.post(url, json=payload, timeout=30)
    print(f"Status: {r.status_code}")
    try:
        print(json.dumps(r.json(), indent=2))
    except Exception:
        print(r.text)


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000/auto-reply"
    main(url)
