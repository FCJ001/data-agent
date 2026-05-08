from typing import Optional

import httpx

from app.conf.app_config import EmbeddingConfig, app_config


class EmbeddingClientManager:
    def __init__(self, config: EmbeddingConfig):
        self.client: Optional[httpx.AsyncClient] = None
        self.config = config

    def _get_url(self):
        return f"http://{self.config.host}:{self.config.port}"

    def init(self):
        self.client = httpx.AsyncClient(base_url=self._get_url(), timeout=30.0)

    async def close(self):
        if self.client:
            await self.client.aclose()

    async def embed(self, text: str) -> list[float]:
        resp = await self.client.post("/embed", json={"inputs": text})
        resp.raise_for_status()
        return resp.json()

    async def embed_many(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.post("/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()


embedding_client_manager = EmbeddingClientManager(app_config.embedding)
