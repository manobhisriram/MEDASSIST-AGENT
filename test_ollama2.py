import requests
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "medassist", "prompt": "What is fever?", "stream": False},
    timeout=300
)
print(response.status_code)
print(response.json().get("response", "")[:200])