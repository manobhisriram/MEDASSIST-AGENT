import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "medassist",
        "prompt": "A patient has fever for 3 days, joint pain, and red spots on skin. What could this be? Answer in 2 sentences.",
        "stream": False
    }
    response = requests.post(url, json=payload)
    result = response.json()
    print("MedAssist response:")
    print(result["response"])
    print("\nOllama medassist model working correctly.")

if __name__ == "__main__":
    test_ollama()