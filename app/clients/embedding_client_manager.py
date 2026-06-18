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

    async def embed_query(self, text: str) -> list[float]:
        resp = await self.client.post("/embed", json={"inputs": text})
        resp.raise_for_status()
        return resp.json()

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        resp = await self.client.post("/embed", json={"inputs": texts})
        resp.raise_for_status()
        return resp.json()


embedding_client_manager = EmbeddingClientManager(app_config.embedding)

if __name__ == '__main__':
    import asyncio

    embedding_client_manager.init()

    async def test():
        # 单条文本向量化
        single = await embedding_client_manager.embed_query("华南地区的销售额是多少")
        print(f"单条向量维度: {len(single)}")
        print(f"前5个值: {single[:5]}")

        # 批量文本向量化
        texts = ["广东省", "华东地区", "销售额", "订单数量"]
        batch = await embedding_client_manager.embed_documents(texts)
        print(f"\n批量向量: {len(batch)} 条, 每条维度: {len(batch[0])}")

    asyncio.run(test())
