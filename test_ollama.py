import requests
import json

def test_ollama():
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3",
        "prompt": "A patient has fever for 3 days, joint pain, and red spots on skin. What could this be? Answer in 2 sentences.",
        "stream": False
    }
    response = requests.post(url, json=payload)
    result = response.json()
    print("Ollama response:")
    print(result["response"])
    print("\nOllama is working correctly.")

if __name__ == "__main__":
    test_ollama()