"""
core/openrouter.py — OpenRouter API client
"""
import base64
import time
import json
import re
import httpx
import numpy as np
import cv2
from typing import Optional
from config import config


def _extract_json_safe(text: str) -> dict:
    """
    Robustly extract JSON from LLM response.
    Handles: markdown fences, extra text before/after, truncated responses.
    """
    if not text or not text.strip():
        raise ValueError("Empty response from LLM")

    # Remove markdown fences
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object via brace matching
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON object found in response: {text[:200]}")

    depth = 0
    end = -1
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # Try to fix truncated JSON by closing open braces
        open_count = text.count("{") - text.count("}")
        fixed = text[start:] + "}" * open_count
        try:
            return json.loads(fixed)
        except Exception:
            raise ValueError(f"Malformed JSON in response: {text[start:start+300]}")

    return json.loads(text[start:end])


class OpenRouterClient:
    def __init__(self):
        self.api_key = config.openrouter.api_key
        self.base_url = config.openrouter.base_url
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://humanoid-agent.research",
            "X-Title": "Humanoid Robot Learning Agent",
        }

    def _encode_frame(self, frame: np.ndarray) -> str:
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buf).decode("utf-8")

    def call_vision(
        self,
        prompt: str,
        frame: np.ndarray,
        model: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> dict:
        model = model or config.openrouter.vision_model
        image_b64 = self._encode_frame(frame)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
        }]
        return self._call(model, messages, max_tokens, temperature)

    def call_text(
        self,
        system: str,
        user: str,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> dict:
        model = model or config.openrouter.reasoning_model
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self._call(model, messages, max_tokens, temperature)

    def _call(
        self,
        model: str,
        messages: list,
        max_tokens: int,
        temperature: float,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        for attempt in range(3):
            try:
                with httpx.Client(timeout=90.0) as client:
                    r = client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload,
                    )
                    r.raise_for_status()
                    return r.json()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    wait = 2 ** attempt
                    print(f"  Rate limit — retrying in {wait}s...")
                    time.sleep(wait)
                elif e.response.status_code == 401:
                    raise RuntimeError(
                        "401 Unauthorized — check OPENROUTER_API_KEY in .env"
                    ) from e
                else:
                    raise
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(1)
        raise RuntimeError("API call failed after 3 retries")

    def extract_text(self, response: dict) -> str:
        try:
            return response["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as e:
            raise ValueError(f"Cannot parse response: {e}\n{response}") from e

    def extract_json(self, response: dict) -> dict:
        return _extract_json_safe(self.extract_text(response))
