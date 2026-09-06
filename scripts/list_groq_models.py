"""Quick script to list available Groq models."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backend.config as cfg
import urllib.request, json

key = cfg.GROQ_API_KEY
print(f"Key available: {bool(key)}, starts with: {key[:8] if key else 'N/A'}")

req = urllib.request.Request(
    "https://api.groq.com/openai/v1/models",
    headers={"Authorization": f"Bearer {key}"}
)
with urllib.request.urlopen(req) as resp:
    data = json.loads(resp.read())
    print(f"Total models: {len(data['data'])}")
    print("\nAll model IDs:")
    for m in sorted(data["data"], key=lambda x: x["id"]):
        print(f"  {m['id']}")
