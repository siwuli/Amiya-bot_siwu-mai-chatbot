# -*- coding: utf-8 -*-
"""持续风格学习：提示词可由配置注入"""

import json
import time
from typing import List, Dict, Optional

from .llm import DeepSeekClient
from .memory import A_Memorix
from .utils import extract_json

DEFAULT_EXPRESSION_PROMPT = """从最近群聊里提取有"群友特征"的说话习惯（口头禅、常用句式、整体语气）。

消息：
{messages}

输出 JSON（每条 1~4 个，不要堆）：
{{
  "habits": ["口头禅", "语气词"],
  "sentence_patterns": ["句式特征"],
  "tone": "整体语气简短描述"
}}

只输出 JSON。"""

DEFAULT_JARGON_PROMPT = """从最近群聊里提取可能是群友专用的新词、俚语或黑话。

消息：
{messages}

输出 JSON：
{{
  "jargons": [{{"term": "词", "meaning": "意思"}}]
}}

没发现就输出 {{"jargons": []}}。只输出 JSON。"""


class StyleLearner:
    def __init__(
        self,
        llm: DeepSeekClient,
        memory: A_Memorix,
        expression_cooldown: int = 10800,
        jargon_cooldown: int = 21600,
        min_samples: int = 15,
    ):
        self.llm = llm
        self.memory = memory
        self._expression_cooldown = expression_cooldown
        self._jargon_cooldown = jargon_cooldown
        self._min_samples = min_samples
        self._last_expression = 0.0
        self._last_jargon = 0.0
        self.expression_prompt = DEFAULT_EXPRESSION_PROMPT
        self.jargon_prompt = DEFAULT_JARGON_PROMPT

    def update_config(
        self,
        expression_cooldown: int,
        jargon_cooldown: int,
        min_samples: int,
        expression_prompt: Optional[str] = None,
        jargon_prompt: Optional[str] = None,
    ):
        self._expression_cooldown = expression_cooldown
        self._jargon_cooldown = jargon_cooldown
        self._min_samples = min_samples
        if expression_prompt and str(expression_prompt).strip():
            self.expression_prompt = str(expression_prompt).strip()
        if jargon_prompt and str(jargon_prompt).strip():
            self.jargon_prompt = str(jargon_prompt).strip()

    async def learn_expressions(self, group_id: str, messages: List[Dict]):
        if time.time() - self._last_expression < self._expression_cooldown:
            return
        msgs = [m for m in messages if m.get('message')]
        if len(msgs) < self._min_samples:
            return
        self._last_expression = time.time()

        msg_lines = [f"{m.get('nickname', '?')}: {m['message']}" for m in msgs[-30:]]
        joined = '\n'.join(msg_lines)
        try:
            try:
                prompt = self.expression_prompt.format(messages=joined)
            except Exception:
                prompt = DEFAULT_EXPRESSION_PROMPT.format(messages=joined)
            response = await self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            data = json.loads(extract_json(response) or '{"habits":[],"sentence_patterns":[],"tone":""}')
            for p in data.get('habits', []) + data.get('sentence_patterns', []):
                if isinstance(p, str) and p.strip():
                    self.memory.learn_pattern(group_id, 'group', p.strip())
        except Exception:
            pass

    async def mine_jargons(self, group_id: str, messages: List[Dict]):
        if time.time() - self._last_jargon < self._jargon_cooldown:
            return
        msgs = [m for m in messages if m.get('message')]
        if len(msgs) < self._min_samples:
            return
        self._last_jargon = time.time()

        joined = '\n'.join(m['message'] for m in msgs[-30:])
        try:
            try:
                prompt = self.jargon_prompt.format(messages=joined)
            except Exception:
                prompt = DEFAULT_JARGON_PROMPT.format(messages=joined)
            response = await self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=200,
            )
            data = json.loads(extract_json(response) or '{"jargons":[]}')
            for item in data.get('jargons', []):
                term = item.get('term', '')
                if isinstance(term, str) and len(term) >= 2:
                    self.memory.learn_jargon(group_id, term.strip(), meaning=item.get('meaning'), level=1)
        except Exception:
            pass
