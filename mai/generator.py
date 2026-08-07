# -*- coding: utf-8 -*-
"""对话生成：人设 / 模板均可由配置注入"""

import random
import re
from typing import List, Dict, Optional, Callable, Awaitable

from .llm import DeepSeekClient
from .memory import A_Memorix
from .utils import strip_thinking

DEFAULT_SYSTEM_PROMPT = """你是阿米娅，群里大家都叫你「兔兔」。你在一个 QQ/群聊里和大家闲聊。

说话规则（必须遵守）：
1. 像普通女孩子日常聊天：口语、自然、有一点情绪，不要公文腔。
2. 绝大多数回复 1~2 句，最多 3 句；别写长段、别列点、别用 markdown。
3. 直接回应当前这句话和最近几句上下文，不要复述人设，不要自我介绍。
4. 称呼对方可用「博士」或自然地用「你」，看气氛；不要每句都叫博士。
5. 可以吐槽、接梗、反问；不懂就坦诚说不熟。
6. 禁止：我是AI / 作为人工智能 / 很抱歉无法 / 有什么可以帮你 / 请问还有别的问题。
7. 不要输出「兔兔:」「阿米娅:」这类前缀，直接说内容。
8. 可用语气词（啊呢嘛吧呀……），可用省略号；不需要正式句号收尾。
9. 已被选中回复时必须输出可见中文，禁止空内容/沉默/不回复。"""

DEFAULT_USER_PROMPT_TEMPLATE = (
    '最近群聊：\n{history}\n\n'
    '现在轮到你接话。{speaker_name}刚刚说：{focus_text}\n'
    '请用{bot_name}的口吻直接回复一句或两句可见中文，不要空着。'
)

DEFAULT_BANNED_PHRASES = [
    '作为人工智能',
    '作为一个AI',
    '我是AI',
    '我是一个语言模型',
    '很抱歉，我无法',
    '有什么可以帮',
    '请问还有什么',
]

# 模型返回空内容时的保底短句（已决定要回时绝不哑火）
FALLBACK_REPLIES = [
    '啊？',
    '嗯？',
    '哈哈',
    '真的假的',
    '有点东西',
    '咋了',
    '懂了',
    '行吧',
]


class ChatGenerator:
    def __init__(
        self,
        llm: DeepSeekClient,
        memory: A_Memorix,
        model: str = 'deepseek-chat',
        embedder: Optional[Callable[[str], Awaitable[Optional[List[float]]]]] = None,
    ):
        self.llm = llm
        self.memory = memory
        self.model = model
        # 兼容两种用法：直接传 EmbeddingClient，或传 MaiCore._safe_embed_one（带熔断）
        self.embedder = embedder
        self.bot_name = '兔兔'
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        self.user_prompt_template = DEFAULT_USER_PROMPT_TEMPLATE
        self.banned_phrases = list(DEFAULT_BANNED_PHRASES)
        self._temperature: float = 0.95
        self._max_tokens: int = 180
        self._max_length: int = 100
        self._memory_hours: int = 168
        self._max_context: int = 16
        self.last_empty_meta: Dict = {}

    def update_config(
        self,
        temperature: float,
        max_tokens: int,
        max_length: int,
        memory_hours: int,
        max_context: int,
        model: Optional[str] = None,
        bot_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
        user_prompt_template: Optional[str] = None,
        banned_phrases: Optional[List[str]] = None,
    ):
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_length = max_length
        self._memory_hours = memory_hours
        self._max_context = max_context
        if model:
            self.model = model
        if bot_name:
            self.bot_name = bot_name
        if system_prompt and str(system_prompt).strip():
            self.system_prompt = str(system_prompt).strip()
        if user_prompt_template and str(user_prompt_template).strip():
            self.user_prompt_template = str(user_prompt_template).strip()
        if banned_phrases is not None:
            self.banned_phrases = [str(x).strip() for x in banned_phrases if str(x).strip()]

    def _build_system(
        self,
        user_id: str,
        group_id: str,
        query: str,
        query_vec: Optional[List[float]] = None,
    ) -> str:
        parts = [self.system_prompt]

        profile = self.memory.get_profile(user_id)
        profile_bits = []
        if profile.get('traits'):
            profile_bits.append('性格印象：' + '、'.join(f'{k}' for k in list(profile['traits'].keys())[:5]))
        if profile.get('preferences'):
            profile_bits.append('偏好：' + '、'.join(f'{k}' for k in list(profile['preferences'].keys())[:5]))
        if profile.get('speaking_style'):
            profile_bits.append(
                '说话习惯：' + '、'.join(f'{k}' for k in list(profile['speaking_style'].keys())[:4])
            )
        if profile.get('recent_info'):
            info = profile['recent_info']
            if isinstance(info, dict) and info:
                profile_bits.append(
                    '已知事实：' + '；'.join(f'{k}={v}' for k, v in list(info.items())[:6])
                )
        if profile_bits:
            parts.append('关于当前说话的人：\n- ' + '\n- '.join(profile_bits))

        patterns = self.memory.get_top_patterns(group_id, limit=4)
        if patterns:
            parts.append('这群常有的说法：' + '、'.join(patterns))

        # 黑话按当前消息检索，不灌 TopN
        jargons = self.memory.search_jargons(group_id, query, limit=5)
        if jargons:
            bits = []
            for j in jargons:
                term = j.get('term') or ''
                meaning = (j.get('meaning') or '').strip()
                if not term:
                    continue
                bits.append(f'{term}({meaning})' if meaning else term)
            if bits:
                parts.append('与当前话题相关的群黑话：' + '、'.join(bits))

        memories = self.memory.search_relevant(group_id, query, limit=3, query_vec=query_vec)
        user_mem = self.memory.search_relevant(user_id, query, limit=2, query_vec=query_vec)
        merged = []
        for m in memories + user_mem:
            if m not in merged:
                merged.append(m)
        if merged:
            parts.append('你隐约记得：\n' + '\n'.join(f'· {m}' for m in merged[:4]))

        return '\n\n'.join(parts)

    def _format_history(self, recent_messages: List[Dict], bot_user_id: str) -> str:
        lines = []
        bot_ids = {str(bot_user_id or ''), 'bot'}
        bot_names = {str(self.bot_name or ''), '兔兔', '阿米娅'}
        for m in recent_messages[-self._max_context:]:
            nick = m.get('nickname') or '某人'
            msg = (m.get('message') or '').strip()
            if not msg:
                continue
            uid = str(m.get('user_id') or '')
            if m.get('is_bot') or uid in bot_ids or nick in bot_names:
                lines.append(f'{self.bot_name}: {msg}')
            else:
                lines.append(f'{nick}: {msg}')
        return '\n'.join(lines)

    async def generate(
        self,
        user_id: str,
        group_id: str,
        recent_messages: List[Dict],
        bot_user_id: str = 'bot',
        focus_text: str = '',
        speaker_name: str = '群友',
    ) -> str:
        if not focus_text and recent_messages:
            focus_text = recent_messages[-1].get('message', '')

        # 语义检索用当前消息的向量；embedding 失败时自动退回纯 FTS
        query_vec = None
        if focus_text and getattr(self, 'embedder', None) is not None:
            try:
                embed_fn = getattr(self.embedder, 'embed_one', None)
                if embed_fn is not None:
                    query_vec = await embed_fn(focus_text)
                else:
                    query_vec = await self.embedder(focus_text)
            except Exception:
                query_vec = None

        system = self._build_system(user_id, group_id, focus_text or '', query_vec=query_vec)
        history = self._format_history(recent_messages, bot_user_id)

        try:
            user_prompt = self.user_prompt_template.format(
                history=history or '（暂无）',
                speaker_name=speaker_name or '群友',
                focus_text=focus_text or '',
                bot_name=self.bot_name,
            )
        except Exception:
            user_prompt = DEFAULT_USER_PROMPT_TEMPLATE.format(
                history=history or '（暂无）',
                speaker_name=speaker_name or '群友',
                focus_text=focus_text or '',
                bot_name=self.bot_name,
            )

        messages = [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user_prompt},
        ]

        reply = await self._call_once(messages, self._temperature)
        if not reply:
            # 空内容常见于：内容过滤 / 模型“选择沉默” / 接口偶发空 content
            reply = await self._call_once(
                messages,
                max(0.3, self._temperature - 0.25),
                force_line=True,
            )

        if not reply:
            reply = random.choice(FALLBACK_REPLIES)
            self.last_empty_meta = getattr(self, 'last_empty_meta', {}) or {}
            self.last_empty_meta['used_fallback'] = True
        return reply

    async def _call_once(
        self,
        messages: List[Dict],
        temperature: float,
        force_line: bool = False,
    ) -> str:
        msgs = messages
        if force_line:
            msgs = list(messages) + [
                {
                    'role': 'user',
                    'content': '上一次你没有输出任何文字。请现在直接回复一句可见的中文短句，禁止空着。',
                }
            ]
        try:
            chat_raw = getattr(self.llm, 'chat_raw', None)
            if callable(chat_raw):
                raw, finish, meta = await chat_raw(
                    messages=msgs,
                    model=self.model,
                    temperature=min(1.4, temperature + random.random() * 0.1),
                    max_tokens=self._max_tokens,
                )
                self.last_empty_meta = {
                    'finish_reason': finish,
                    'meta': meta,
                    'raw_len': len(raw or ''),
                    'force_line': force_line,
                }
            else:
                raw = await self.llm.chat(
                    messages=msgs,
                    model=self.model,
                    temperature=min(1.4, temperature + random.random() * 0.1),
                    max_tokens=self._max_tokens,
                )
                self.last_empty_meta = {'raw_len': len(raw or ''), 'force_line': force_line}
            return self._post_process(raw)
        except Exception as e:
            self.last_empty_meta = {'error': str(e), 'force_line': force_line}
            raise

    def _post_process(self, text: str) -> str:
        text = strip_thinking(text or '')
        text = text.strip()
        if not text:
            return ''

        # 模型有时用「（沉默）」「不回复」表示闭嘴——视为空，走重试/兜底
        silence = {
            '（沉默）', '(沉默)', '沉默', '不回复', '不说话', '……', '...', '…',
            '（不回复）', '(不回复)', '无', '无。', '/', '-',
        }
        if text in silence:
            return ''

        prefix_pat = rf'^(assistant|{re.escape(self.bot_name)}|阿米娅|Amiya|AI)[:：]\s*'
        text = re.sub(prefix_pat, '', text, flags=re.I)
        text = text.strip().strip('"\'「」『』')
        text = re.sub(r'^[-*•]\s+', '', text, flags=re.M)
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'`([^`]*)`', r'\1', text)

        for b in self.banned_phrases:
            if b and b in text:
                text = text.replace(b, '')

        text = re.sub(r'\n{2,}', '\n', text).strip()
        if text in silence:
            return ''

        if len(text) > self._max_length:
            cut = text[: self._max_length]
            for sep in ('。', '！', '？', '…', '~', '～', '!', '?', '，', ','):
                idx = cut.rfind(sep)
                if idx >= 12:
                    text = cut[: idx + len(sep)]
                    break
            else:
                text = cut.rstrip('，,、') + '…'

        return text.strip()
