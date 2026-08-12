# -*- coding: utf-8 -*-
"""持续风格学习：表达习惯 + 黑话一次调用合并提取，提示词可由配置注入"""

import json
import time
from typing import List, Dict, Optional

from .llm import DeepSeekClient
from .memory import A_Memorix
from .utils import extract_json

DEFAULT_GROUP_PROMPT = """从最近群聊里提取有"群友特征"的说话习惯，以及群内专用黑话。

消息：
{messages}

输出 JSON：
{{
  "habits": ["口头禅", "语气词"],
  "sentence_patterns": ["句式特征"],
  "tone": "整体语气简短描述",
  "jargons": [{{"term": "词", "meaning": "意思"}}]
}}

规则：
- 习惯与句式 1~4 个，不要堆。
- 黑话只记录新人不了解、老群友才懂的词；普通网络词（哈哈/确实/笑死）与一次性玩笑不要记录。
- 没发现就输出空数组。
只输出 JSON。"""


class StyleLearner:
    """表达风格与黑话合并在一次 LLM 调用里提取，省一次调用，回复路径不受影响。"""

    def __init__(
        self,
        llm: DeepSeekClient,
        memory: A_Memorix,
        group_cooldown: int = 10800,
        min_samples: int = 15,
    ):
        self.llm = llm
        self.memory = memory
        self._group_cooldown = group_cooldown
        self._min_samples = min_samples
        self._last_group = 0.0
        self.group_prompt = DEFAULT_GROUP_PROMPT

    def update_config(
        self,
        group_cooldown: int,
        min_samples: int,
        group_prompt: Optional[str] = None,
    ):
        self._group_cooldown = group_cooldown
        self._min_samples = min_samples
        if group_prompt and str(group_prompt).strip():
            self.group_prompt = str(group_prompt).strip()

    async def learn_group(self, group_id: str, messages: List[Dict]):
        """一次调用提取表达习惯 + 黑话；样本数量与消息全文保留，不影响学习质量。"""
        if time.time() - self._last_group < self._group_cooldown:
            return
        msgs = [m for m in messages if m.get('message')]
        if len(msgs) < self._min_samples:
            return
        self._last_group = time.time()

        msg_lines = [f"{m.get('nickname', '?')}: {m['message']}" for m in msgs[-30:]]
        joined = '\n'.join(msg_lines)
        try:
            try:
                prompt = self.group_prompt.format(messages=joined)
            except Exception:
                prompt = DEFAULT_GROUP_PROMPT.format(messages=joined)
            response = await self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.3,
                max_tokens=300,
            )
            data = json.loads(
                extract_json(response) or '{"habits":[],"sentence_patterns":[],"tone":"","jargons":[]}'
            )
            for p in data.get('habits', []) + data.get('sentence_patterns', []):
                if isinstance(p, str) and p.strip():
                    self.memory.learn_pattern(group_id, 'group', p.strip())
            for item in data.get('jargons', []):
                term = item.get('term', '')
                if isinstance(term, str) and len(term) >= 2:
                    self.memory.learn_jargon(group_id, term.strip(), meaning=item.get('meaning'), level=1)
        except Exception:
            pass
