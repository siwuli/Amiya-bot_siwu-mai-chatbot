# -*- coding: utf-8 -*-
"""OpenAI 兼容聊天客户端（默认 DeepSeek）"""

import asyncio
from typing import List, Dict, Optional, Tuple


class DeepSeekClient:
    """异步聊天客户端，兼容 DeepSeek / OpenAI 接口"""

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.deepseek.com',
        default_model: str = 'deepseek-chat',
        max_concurrent: int = 3,
    ):
        if not api_key:
            raise ValueError('API Key 不能为空')
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.default_model = default_model
        self._session = None
        # 限制并发，避免生成与画像/风格学习同时打爆接口导致空 content
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    def _get_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    @staticmethod
    def _extract_content(message: Dict) -> str:
        """只取最终回复 content；绝不把 reasoning_content 当可见回复。"""
        if not isinstance(message, dict):
            return ''
        content = message.get('content')
        if isinstance(content, list):
            parts = []
            for p in content:
                if isinstance(p, str):
                    parts.append(p)
                elif isinstance(p, dict):
                    # 跳过 reasoning / thought 类型分片
                    ptype = str(p.get('type') or '').lower()
                    if ptype in ('reasoning', 'thinking', 'thought', 'reason'):
                        continue
                    parts.append(str(p.get('text') or p.get('content') or ''))
            content = ''.join(parts)
        if content is None:
            return ''
        return str(content).strip()

    async def chat_raw(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.85,
        max_tokens: int = 512,
        **kwargs,
    ) -> Tuple[str, str, Dict]:
        """返回 (content, finish_reason, 原始 choice 摘要)。"""
        session = self._get_session()
        payload = {
            'model': model or self.default_model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        payload.update(kwargs)

        async with self._sem:
            async with session.post(f'{self.base_url}/chat/completions', json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f'LLM API error {resp.status}: {text[:300]}')
                import json
                data = json.loads(text)
                choices = data.get('choices') or []
                if not choices:
                    return '', 'no_choices', {'raw_keys': list(data.keys())}
                choice = choices[0] or {}
                message = choice.get('message') or {}
                content = self._extract_content(message)
                finish = str(choice.get('finish_reason') or '')
                meta = {
                    'finish_reason': finish,
                    'model': data.get('model'),
                    'content_len': len(content),
                    'has_reasoning': bool(message.get('reasoning_content') or message.get('reasoning')),
                }
                return content, finish, meta

    async def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.85,
        max_tokens: int = 512,
        **kwargs,
    ) -> str:
        content, _, _ = await self.chat_raw(
            messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        return content

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None


class EmbeddingClient:
    """OpenAI 兼容 embedding 客户端（POST /embeddings）。

    注意：DeepSeek 官方接口没有 embedding；请配置支持 /embeddings 的服务，例如：
      硅基流动  base_url=https://api.siliconflow.cn/v1  model=BAAI/bge-m3
      OpenAI    base_url=https://api.openai.com/v1       model=text-embedding-3-small
    换模型后维度变化不影响检索（余弦相似度按行比较，维度不一致自动跳过）。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = 'https://api.siliconflow.cn/v1',
        model: str = 'text-embedding-3-small',
        max_concurrent: int = 2,
    ):
        if not api_key:
            raise ValueError('Embedding API Key 不能为空')
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.model = model
        self._session = None
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))

    def _get_session(self):
        if self._session is None:
            import aiohttp
            self._session = aiohttp.ClientSession(
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json',
                },
                timeout=aiohttp.ClientTimeout(total=60),
            )
        return self._session

    async def embed(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，返回与 texts 等长的向量列表；失败抛异常由调用方兜底。"""
        texts = [str(t).strip() for t in texts if str(t).strip()]
        if not texts:
            return []
        session = self._get_session()
        payload = {'model': self.model, 'input': texts}
        async with self._sem:
            async with session.post(f'{self.base_url}/embeddings', json=payload) as resp:
                text = await resp.text()
                if resp.status != 200:
                    raise RuntimeError(f'Embedding API error {resp.status}: {text[:300]}')
                import json
                data = json.loads(text)
                rows = sorted(
                    (data.get('data') or []),
                    key=lambda d: int(d.get('index', 0)),
                )
                return [list(d['embedding']) for d in rows if isinstance(d, dict) and d.get('embedding')]

    async def embed_one(self, text: str) -> List[float]:
        vecs = await self.embed([text])
        return vecs[0] if vecs else []

    async def close(self):
        if self._session:
            await self._session.close()
            self._session = None
