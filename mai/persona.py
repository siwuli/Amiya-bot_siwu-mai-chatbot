# -*- coding: utf-8 -*-
"""人格画像：提示词可由配置注入"""

import json
import re
import time
from threading import Lock
from typing import List, Dict, Optional

from .llm import DeepSeekClient
from .memory import A_Memorix
from .utils import extract_json

DEFAULT_PERSONA_PROMPT = """从用户最近的聊天记录里提取稳定的画像信息。输出严格 JSON：

{
  "traits": {"特质名": "简短依据"},
  "preferences": {"偏好名": "简短依据"},
  "speaking_style": {"风格名": "简短依据"},
  "stable_facts": {"事实名": "内容"}
}

规则：
- 只保留较稳定、可复用的信息；一时兴起的玩笑不要写。
- 没有就用空对象 {}。
- 只输出 JSON。"""

DEFAULT_PERSONA_USER_TEMPLATE = '用户最近的聊天记录：\n{conversation}'

_TS_PREFIX = re.compile(r'^\[\d{2}-\d{2}\s+\d{2}:\d{2}\]\s*')


class PersonaExtractor:
    def __init__(self, llm: DeepSeekClient, cooldown_seconds: int = 21600, min_samples: int = 12):
        self.llm = llm
        self._pending: Dict[str, List[str]] = {}
        self._lock = Lock()
        self._last_extract: Dict[str, float] = {}
        self._cooldown = cooldown_seconds
        self._min_samples = min_samples
        self.system_prompt = DEFAULT_PERSONA_PROMPT
        self.user_template = DEFAULT_PERSONA_USER_TEMPLATE

    def update_config(
        self,
        cooldown_seconds: int,
        min_samples: int,
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
    ):
        self._cooldown = cooldown_seconds
        self._min_samples = min_samples
        if system_prompt and str(system_prompt).strip():
            self.system_prompt = str(system_prompt).strip()
        if user_template and str(user_template).strip():
            self.user_template = str(user_template).strip()

    def add_messages(self, user_id: str, messages: List[str]):
        if not user_id or not messages:
            return
        cleaned = [m.strip() for m in messages if m and str(m).strip()]
        if not cleaned:
            return
        with self._lock:
            if user_id not in self._pending:
                self._pending[user_id] = []
            self._pending[user_id].extend(cleaned)
            if len(self._pending[user_id]) > 80:
                self._pending[user_id] = self._pending[user_id][-80:]

    def _gather_messages(self, user_id: str, memory: A_Memorix) -> List[str]:
        """合并内存样本与库里近期 chat，避免重启/重建核心后永远攒不够。"""
        with self._lock:
            msgs = list(self._pending.get(user_id, []))

        seen = set(msgs)
        try:
            db_msgs = memory.get_recent_chunks(
                user_id, hours=336, limit=40, chunk_types=('chat',)
            )
        except Exception:
            db_msgs = []

        for raw in reversed(db_msgs):
            text = _TS_PREFIX.sub('', (raw or '').strip()).strip()
            if text and text not in seen:
                seen.add(text)
                msgs.append(text)
        return msgs

    async def extract_if_needed(self, user_id: str, memory: A_Memorix) -> Optional[Dict]:
        now = time.time()
        if now - self._last_extract.get(user_id, 0) < self._cooldown:
            return None

        msgs = self._gather_messages(user_id, memory)
        if len(msgs) < self._min_samples:
            return None

        profile = await self._do_extract(user_id, msgs, memory)
        if profile is None:
            return None

        # 仅成功后进入冷却，并保留尾部样本；失败不扣样本、不加锁
        with self._lock:
            self._pending[user_id] = msgs[-12:]
            self._last_extract[user_id] = time.time()
        return profile

    async def _do_extract(self, user_id: str, messages: List[str], memory: A_Memorix) -> Optional[Dict]:
        conversation = '\n'.join(f'- {m}' for m in messages[-25:])
        try:
            try:
                user_content = self.user_template.format(conversation=conversation)
            except Exception:
                user_content = DEFAULT_PERSONA_USER_TEMPLATE.format(conversation=conversation)

            response = await self.llm.chat(
                messages=[
                    {'role': 'system', 'content': self.system_prompt},
                    {'role': 'user', 'content': user_content},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            json_str = extract_json(response)
            if not json_str:
                return None
            profile = json.loads(json_str)
            if not isinstance(profile, dict):
                return None

            traits = profile.get('traits') if isinstance(profile.get('traits'), dict) else {}
            preferences = profile.get('preferences') if isinstance(profile.get('preferences'), dict) else {}
            speaking_style = (
                profile.get('speaking_style')
                if isinstance(profile.get('speaking_style'), dict)
                else {}
            )
            stable_facts = (
                profile.get('stable_facts')
                if isinstance(profile.get('stable_facts'), dict)
                else {}
            )

            # 全空也算一次成功，避免同一批样本反复烧 LLM；写库留下可观察痕迹
            memory.write_profile(
                user_id,
                traits=traits or {},
                preferences=preferences or {},
                speaking_style=speaking_style or {},
                stable_facts=stable_facts or {},
            )
            for entity, val in stable_facts.items():
                memory.write_graph(
                    user_id,
                    str(entity),
                    'is',
                    str(val),
                    weight=0.9,
                    evidence=conversation[:200],
                )
                memory.write_chunk(
                    user_id=user_id,
                    content=f'{entity}: {val}',
                    chunk_type='fact',
                    importance=0.85,
                )
            return {
                'traits': traits,
                'preferences': preferences,
                'speaking_style': speaking_style,
                'stable_facts': stable_facts,
            }
        except Exception:
            return None
