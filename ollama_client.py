import requests
from typing import Dict, Any


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
    
    def list_models(self) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.api_url}/tags")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to list models: {str(e)}"}
    
    def generate_response(self, model: str, prompt: str) -> Dict[str, Any]:
        payload = {"model": model, "prompt": prompt, "stream": False}
        try:
            response = requests.post(f"{self.api_url}/generate", json=payload, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {"error": f"Failed to generate response: {str(e)}"}
    
    def check_model_exists(self, model: str) -> bool:
        models = self.list_models()
        if "error" in models:
            return False
        model_names = [m["name"] for m in models.get("models", [])]
        return any(model in name for name in model_names)