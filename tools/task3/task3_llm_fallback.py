#!/usr/bin/env python3

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

from openai import OpenAI


FALLBACK_BASE_URL_ENV = "TASK3_FALLBACK_BASE_URL"
FALLBACK_API_KEY_ENV = "TASK3_FALLBACK_API_KEY"
FALLBACK_MODEL_ENV = "TASK3_FALLBACK_MODEL"


@dataclass(frozen=True)
class LLMRoute:
    name: str
    base_url: str
    api_key: str
    model: str


def build_routes(*, base_url: str, api_key: str, model: str) -> List[LLMRoute]:
    routes = [LLMRoute(name="primary", base_url=base_url, api_key=api_key, model=model)]
    fallback_base_url = os.environ.get(FALLBACK_BASE_URL_ENV, "").strip()
    fallback_api_key = os.environ.get(FALLBACK_API_KEY_ENV, "").strip() or api_key
    fallback_model = os.environ.get(FALLBACK_MODEL_ENV, "").strip() or model
    if fallback_base_url:
        fallback_route = LLMRoute(
            name="fallback",
            base_url=fallback_base_url,
            api_key=fallback_api_key,
            model=fallback_model,
        )
        if fallback_route != routes[0]:
            routes.append(fallback_route)
    return routes


def build_clients(routes: List[LLMRoute], *, timeout: int = 180) -> List[tuple[LLMRoute, OpenAI]]:
    return [
        (route, OpenAI(base_url=route.base_url, api_key=route.api_key, timeout=timeout, max_retries=0))
        for route in routes
    ]
