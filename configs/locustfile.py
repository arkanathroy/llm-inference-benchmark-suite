"""
locustfile.py
=============
Locust load test targeting a running vLLM OpenAI-compatible server
(launched via `vllm serve` or the notebook's Phase 6).

Run standalone (outside the notebook) with:
    locust -f configs/locustfile.py --host http://localhost:8000 \
           --users 32 --spawn-rate 4 --run-time 3m --headless \
           --csv=benchmarks/locust_results

WHY Locust over a hand-rolled asyncio loop for the FINAL reported
numbers (the notebook's internal benchmark_runner.py is used for the
fine-grained per-request TTFT/TPOT breakdown during development):
Locust's distributed, battle-tested load generator produces industry-
standard percentile/RPS statistics and a live web UI -- since the JD
explicitly names Locust as a required tool, the deliverable benchmark
report is generated from real Locust output, not just an equivalent
custom script, to directly demonstrate hands-on tool proficiency.
"""

import json
import random
import time

from locust import HttpUser, task, between, events

PROMPT_MIX = {
    "short": (32, 128, 0.4),
    "medium": (128, 512, 0.4),
    "long": (512, 1024, 0.2),
}

_TOPIC_BANK = [
    "billing dispute on account", "order tracking status update",
    "password reset assistance request", "product return policy question",
    "subscription cancellation process", "technical support for device",
]


def _sample_prompt() -> str:
    categories = list(PROMPT_MIX.keys())
    weights = [PROMPT_MIX[c][2] for c in categories]
    category = random.choices(categories, weights=weights, k=1)[0]
    min_tok, max_tok, _ = PROMPT_MIX[category]
    n_words = int(random.randint(min_tok, max_tok) / 1.3)
    words = [random.choice(_TOPIC_BANK) for _ in range(max(1, n_words // 4))]
    return " ".join(words) + ". Please provide a detailed response."


class TTFTTracker:
    """
    Custom timer capturing Time To First Token separately from total
    request time, using vLLM's OpenAI-compatible streaming endpoint.
    Locust's built-in response_time only captures total request
    duration -- TTFT requires reading the FIRST chunk of a streamed
    response, which needs this custom instrumentation.
    """
    @staticmethod
    def measure_stream(response) -> tuple:
        t_start = time.time()
        ttft = None
        n_tokens = 0
        for line in response.iter_lines():
            if not line:
                continue
            if ttft is None:
                ttft = time.time() - t_start
            if line.startswith(b"data: ") and line != b"data: [DONE]":
                n_tokens += 1
        e2e = time.time() - t_start
        return ttft or 0.0, e2e, n_tokens


class VLLMUser(HttpUser):
    # WHAT: wait_time between consecutive tasks for the SAME simulated
    #       user. WHY between(0.5, 2.0): approximates human think-time
    #       between conversational turns in a voice/chat agent context
    #       rather than a zero-wait hammer test, which would measure raw
    #       server saturation but not realistic sustained load.
    wait_time = between(0.5, 2.0)

    @task
    def chat_completion(self):
        prompt = _sample_prompt()
        payload = {
            "model": "meta-llama/Llama-3.2-3B-Instruct",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "temperature": 0.7,
            "stream": True,
        }

        t0 = time.time()
        with self.client.post(
            "/v1/chat/completions",
            json=payload,
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Got status code {response.status_code}")
                return

            ttft, e2e, n_tokens = TTFTTracker.measure_stream(response)

            events.request.fire(
                request_type="TTFT",
                name="chat_completion_ttft",
                response_time=ttft * 1000,
                response_length=0,
                exception=None,
                context={},
            )
            response.success()


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("Load test starting -- target:", environment.host)
    print("Prompt mix:", json.dumps(PROMPT_MIX, indent=2))
