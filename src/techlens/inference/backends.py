# -*- coding: utf-8 -*-
"""推理后端抽象。mock: gold/noisy模式测评估链路；llama_server: OpenAI兼容接口。"""
import copy
import json
import random
import time
import urllib.request


class InferenceBackend:
    name = "base"

    def generate(self, system: str, user: str, meta: dict | None = None) -> dict:
        raise NotImplementedError


class MockBackend(InferenceBackend):
    def __init__(self, mode="gold", noise_rate=0.3, seed=0):
        self.name = f"mock-{mode}"
        self.mode = mode
        self.noise_rate = noise_rate
        self.rng = random.Random(seed)

    def generate(self, system, user, meta=None):
        expected = (meta or {}).get("expected", {"status": "ABORT", "reason": "mock"})
        out = copy.deepcopy(expected)
        if self.mode == "noisy" and self.rng.random() < self.noise_rate:
            kind = self.rng.choice(["invent_level", "kdj_drift", "analyze_error", "markdown", "broken"])
            if kind == "invent_level" and out.get("status") == "OK":
                out["support"] = 12.34  # 编造价位
            elif kind == "kdj_drift" and out.get("status") == "OK":
                out["kdj"]["K"] = round(out["kdj"]["K"] + 1.5, 2)  # KDJ值漂移
            elif kind == "analyze_error" and out.get("status") == "ABORT":
                out = {"status": "OK", "stock_code": "600000", "trend": "neutral",
                       "volume_price": "量价平稳", "support": "暂不设定", "resistance": "暂不设定",
                       "kdj": {"K": 50.0, "D": 50.0, "J": 50.0, "signal": "waiting"},
                       "confidence": "low", "summary": "硬着头皮分析"}
            elif kind == "markdown":
                return {"text": "```json\n" + json.dumps(out, ensure_ascii=False) + "\n```",
                        "latency_s": 0.01, "first_token_s": 0.005}
            else:
                return {"text": '{"status":"OK","trend":', "latency_s": 0.01, "first_token_s": 0.005}
        return {"text": json.dumps(out, ensure_ascii=False), "latency_s": 0.01, "first_token_s": 0.005}


class LlamaServerBackend(InferenceBackend):
    def __init__(self, base_url="http://127.0.0.1:8080", model="techlens",
                 temperature=0.0, max_tokens=384, timeout=120):
        self.name = f"llama-server@{base_url}"
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout

    def generate(self, system, user, meta=None):
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": self.temperature, "max_tokens": self.max_tokens,
        }).encode("utf-8")
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"}, method="POST")
        t0 = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return {"text": data["choices"][0]["message"]["content"],
                "latency_s": time.time() - t0, "first_token_s": None}


def create_backend(config: dict) -> InferenceBackend:
    kind = config.get("kind", "mock")
    if kind == "mock":
        return MockBackend(mode=config.get("mode", "gold"),
                           noise_rate=config.get("noise_rate", 0.3), seed=config.get("seed", 0))
    if kind == "llama_server":
        return LlamaServerBackend(base_url=config.get("base_url", "http://127.0.0.1:8080"))
    raise ValueError(f"unknown backend kind: {kind}")
