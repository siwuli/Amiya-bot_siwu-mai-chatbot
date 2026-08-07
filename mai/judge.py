# -*- coding: utf-8 -*-
"""轻量接话评分（Reply Judge）

小模型只输出布尔标签 JSON，程序按规则算分，再叠加状态惩罚后概率抽样。
"""

import json
import random
import time
from typing import List, Dict, Optional, Any, Tuple

from .llm import DeepSeekClient
from .utils import extract_json

DEFAULT_JUDGE_PROMPT = (
    '你是群聊回复分析器。\n'
    '根据聊天上下文，判断当前消息具有哪些特征。\n'
    '不要猜测机器人应该说什么，只分析聊天状态。\n\n'
    '机器人：{bot_name}\n\n'
    '最近聊天：\n'
    '{history}\n\n'
    '当前发言人：{speaker_name}\n'
    '当前消息：{focus_text}\n\n'
    '字段说明：\n'
    'mentioned: 是否直接提到机器人（@、名字、昵称等）\n'
    'reply_to_bot: 是否是在回应机器人上一句话\n'
    'question: 是否向群里提出问题\n'
    'open_topic: 是否留下明显接话空间\n'
    'group_chat: 是否属于多人公共聊天，而不是两个人私聊式对话\n'
    'already_answered: 当前消息是否已经被别人自然接住\n'
    'interrupt_risk: 机器人插话是否容易打断别人\n'
    'command: 是否属于机器人无需参与的指令/通知/刷屏\n'
    'requires_bot: 是否明显希望机器人参与\n\n'
    '只输出 JSON：\n'
    '{{"mentioned":false,'
    '"reply_to_bot":false,'
    '"question":false,'
    '"open_topic":false,'
    '"group_chat":true,'
    '"already_answered":false,'
    '"interrupt_risk":false,'
    '"command":false,'
    '"requires_bot":false}}'
)

TAG_KEYS = (
    'mentioned',
    'reply_to_bot',
    'question',
    'open_topic',
    'group_chat',
    'already_answered',
    'interrupt_risk',
    'command',
    'requires_bot',
)


class ReplyJudge:
    """轻量模型接话评分器：标签 → 程序算分 → 状态惩罚 → 概率抽样"""

    def __init__(self, llm: Optional[DeepSeekClient] = None, model: Optional[str] = None):
        self.llm = llm
        self.model = model
        self.enabled = False
        self.prompt = DEFAULT_JUDGE_PROMPT
        self.bot_name = '兔兔'
        self.max_history = 8
        self.temperature = 0.1
        self.max_tokens = 200
        # 状态惩罚参数（秒）；与 TimingGate.min_interval 对齐时可外部覆盖
        self.cooldown_seconds = 60

    def update_config(
        self,
        enabled: bool = False,
        model: Optional[str] = None,
        prompt: Optional[str] = None,
        bot_name: Optional[str] = None,
        max_history: Optional[int] = None,
        llm: Optional[DeepSeekClient] = None,
        cooldown_seconds: Optional[int] = None,
    ):
        self.enabled = bool(enabled)
        if model:
            self.model = model
        if prompt and str(prompt).strip():
            self.prompt = str(prompt).strip()
        if bot_name:
            self.bot_name = bot_name
        if max_history is not None:
            self.max_history = max(2, int(max_history))
        if llm is not None:
            self.llm = llm
        if cooldown_seconds is not None:
            self.cooldown_seconds = max(0, int(cooldown_seconds))

    def _format_history(self, recent_messages: List[Dict], bot_user_id: str) -> str:
        lines = []
        bot_ids = {str(bot_user_id or ''), 'bot'}
        bot_names = {str(self.bot_name or ''), '兔兔', '阿米娅'}
        for m in recent_messages[-self.max_history:]:
            nick = str(m.get('nickname') or '')
            uid = str(m.get('user_id') or '')
            if m.get('is_bot') or uid in bot_ids or nick in bot_names:
                role = self.bot_name
            else:
                role = nick or '某人'
            text = (m.get('message') or '').strip()
            if text:
                lines.append(f'{role}: {text}')
        return '\n'.join(lines) if lines else '（暂无）'

    @staticmethod
    def _as_bool(val: Any) -> bool:
        if isinstance(val, bool):
            return val
        if isinstance(val, (int, float)):
            return bool(val)
        if isinstance(val, str):
            return val.strip().lower() in ('1', 'true', 'yes', 'y', '是')
        return False

    @classmethod
    def parse_tags(cls, raw: str) -> Dict[str, bool]:
        """解析模型 JSON → 布尔标签；解析失败则全 false"""
        defaults = {k: False for k in TAG_KEYS}
        if not raw:
            return defaults
        blob = extract_json(raw) or raw.strip()
        try:
            data = json.loads(blob)
        except Exception:
            return defaults
        if not isinstance(data, dict):
            return defaults
        return {k: cls._as_bool(data.get(k, False)) for k in TAG_KEYS}

    @staticmethod
    def compute_tag_score(tags: Dict[str, bool]) -> int:
        """仅按布尔标签计分（未夹紧、未扣状态惩罚）。command → 0。"""
        if tags.get('command'):
            return 0
        score = 0
        if tags.get('mentioned'):
            score += 70
        if tags.get('reply_to_bot'):
            score += 50
        if tags.get('requires_bot'):
            score += 40
        if tags.get('question'):
            score += 20
        if tags.get('open_topic'):
            score += 15
        if tags.get('group_chat'):
            score += 10
        if tags.get('already_answered'):
            score -= 25
        if tags.get('interrupt_risk'):
            score -= 35
        return score

    @staticmethod
    def calc_recent_reply_penalty(
        recent_messages: List[Dict],
        bot_user_id: str,
    ) -> int:
        """距机器人上一次发言越近，惩罚越高。"""
        # 从尾部往前数（跳过当前这条用户消息）
        msgs = list(recent_messages or [])
        if len(msgs) >= 1:
            msgs = msgs[:-1]
        since = 0
        found = False
        bot_ids = {str(bot_user_id or ''), 'bot'}
        for m in reversed(msgs):
            if m.get('is_bot') or str(m.get('user_id', '')) in bot_ids:
                found = True
                break
            since += 1
        if not found:
            return 0
        if since <= 0:
            return 45
        if since == 1:
            return 30
        if since <= 3:
            return 15
        if since <= 5:
            return 8
        return 0

    def calc_cooldown_penalty(self, last_reply_ts: float) -> int:
        """距上次机器人回复越近，惩罚越高；超过 cooldown_seconds 为 0。"""
        if not last_reply_ts or self.cooldown_seconds <= 0:
            return 0
        elapsed = time.time() - float(last_reply_ts)
        if elapsed >= self.cooldown_seconds:
            return 0
        if elapsed < 0:
            elapsed = 0
        ratio = 1.0 - (elapsed / float(self.cooldown_seconds))
        return int(round(40 * ratio))

    @classmethod
    def compute_score(
        cls,
        tags: Dict[str, bool],
        recent_reply_penalty: int = 0,
        cooldown_penalty: int = 0,
    ) -> int:
        if tags.get('command'):
            return 0
        score = cls.compute_tag_score(tags)
        score -= int(recent_reply_penalty or 0)
        score -= int(cooldown_penalty or 0)
        return max(0, min(100, score))

    @staticmethod
    def format_tags(tags: Dict[str, bool]) -> str:
        return ','.join(f'{k}={"1" if tags.get(k) else "0"}' for k in TAG_KEYS)

    async def score_message(
        self,
        recent_messages: List[Dict],
        focus_text: str,
        speaker_name: str = '群友',
        bot_user_id: str = 'bot',
        last_reply_ts: float = 0,
    ) -> Tuple[int, Dict[str, bool], str, int, int]:
        """返回 (分数, 标签, 模型原文, recent_penalty, cooldown_penalty)。"""
        empty_tags = {k: False for k in TAG_KEYS}
        if not self.enabled or not self.llm:
            return 0, empty_tags, '', 0, 0
        focus = (focus_text or '').strip()
        if not focus:
            return 0, empty_tags, '', 0, 0

        history = self._format_history(recent_messages, bot_user_id)
        try:
            prompt = self.prompt.format(
                bot_name=self.bot_name,
                history=history,
                speaker_name=speaker_name or '群友',
                focus_text=focus,
            )
        except Exception:
            prompt = DEFAULT_JUDGE_PROMPT.format(
                bot_name=self.bot_name,
                history=history,
                speaker_name=speaker_name or '群友',
                focus_text=focus,
            )
        result = await self.llm.chat(
            messages=[{'role': 'user', 'content': prompt}],
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        raw = (result or '').strip()
        tags = self.parse_tags(raw)
        recent_pen = self.calc_recent_reply_penalty(recent_messages, bot_user_id)
        cool_pen = self.calc_cooldown_penalty(last_reply_ts)
        score = self.compute_score(tags, recent_pen, cool_pen)
        return score, tags, raw, recent_pen, cool_pen

    async def should_reply(
        self,
        recent_messages: List[Dict],
        focus_text: str,
        speaker_name: str = '群友',
        bot_user_id: str = 'bot',
        last_reply_ts: float = 0,
    ) -> tuple:
        """评分 + 概率抽样。返回 (是否接话, 分数, 原因)。"""
        focus = (focus_text or '').strip()
        if not focus:
            return False, 0, 'empty'
        if not self.enabled or not self.llm:
            return False, 0, 'disabled'

        score, tags, raw, recent_pen, cool_pen = await self.score_message(
            recent_messages=recent_messages,
            focus_text=focus,
            speaker_name=speaker_name,
            bot_user_id=bot_user_id,
            last_reply_ts=last_reply_ts,
        )
        tag_hint = self.format_tags(tags)
        pen = f'recent_pen={recent_pen} cool_pen={cool_pen}'
        if tags.get('command'):
            return False, 0, f'command tags={{{tag_hint}}} {pen}'
        if score <= 0:
            return False, score, f'score0 tags={{{tag_hint}}} {pen}'

        roll = random.random() * 100
        hit = roll < float(score)
        base = f'tags={{{tag_hint}}} score={score} {pen}'
        if hit:
            return True, score, f'hit roll={roll:.1f}<{score} {base}'
        return False, score, f'miss roll={roll:.1f}>={score} {base}'
