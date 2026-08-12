# -*- coding: utf-8 -*-
"""智能发言时机（Timing Gate）

Token 优化策略：
1. 规则判断优先（消息间隔、是否被 @、是否包含触发词、消息密度等）
2. 只在规则不确定时（介于"可以发言"和"不该发言"之间）才调用 LLM
3. 距离上次发言越近，越保守；时间越久，越倾向主动
"""

import time
from typing import List, Dict, Optional

from .llm import DeepSeekClient

DEFAULT_TIMING_PROMPT = """你是群聊节奏控制员。判断现在机器人是否应该主动说一句。

最近群聊尾巴：
{context}

判定依据：
- 群活跃度（最近是否有连续对话）
- 是否有人在引导话题或求救
- 距离上次机器人发言过去了多久（越久越可以接一句）
- 机器人最近是否被冷落（如果很久没说话，可以接一句）

只输出一个词：
- "reply"  = 可以说一句
- "wait"   = 再等等

只输出一个词。"""


class TimingGate:
    """判断该不该开口的子系统，默认纯规则，LLM 仅作 fallback"""

    def __init__(self, llm: Optional[DeepSeekClient] = None,
                 min_interval: int = 300,
                 max_interval: int = 1800,
                 llm_fallback: bool = True):
        self.llm = llm
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.llm_fallback = llm_fallback
        self.timing_prompt = DEFAULT_TIMING_PROMPT
        self.bot_nickname = '兔兔'
        self._last_reply: Dict[str, float] = {}
        self._group_msg_times: Dict[str, List[float]] = {}
        self._max_track = 60

    def update_config(
        self,
        min_interval: int,
        max_interval: int,
        llm_fallback: bool,
        timing_prompt: Optional[str] = None,
        bot_nickname: Optional[str] = None,
    ):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self.llm_fallback = llm_fallback
        if timing_prompt and str(timing_prompt).strip():
            self.timing_prompt = str(timing_prompt).strip()
        if bot_nickname:
            self.bot_nickname = bot_nickname

    def _track_message(self, group_id: str):
        """记录消息时间用于活跃度估算"""
        now = time.time()
        if group_id not in self._group_msg_times:
            self._group_msg_times[group_id] = []
        times = self._group_msg_times[group_id]
        times.append(now)
        # 清理 10 分钟以前的
        cutoff = now - 600
        self._group_msg_times[group_id] = [t for t in times if t > cutoff][-self._max_track:]

    def _msg_density_per_min(self, group_id: str) -> float:
        """估算最近 5 分钟的消息密度（条/分钟）"""
        times = self._group_msg_times.get(group_id, [])
        if not times:
            return 0.0
        cutoff = time.time() - 300
        recent = sum(1 for t in times if t > cutoff)
        return recent / 5.0

    async def should_proactive_reply(
        self,
        group_id: str,
        recent_messages: List[Dict],
        bot_user_id: str,
    ) -> bool:
        """主动发言判断：纯规则优先，LLM 兜底。返回 True 表示说一句。

        注意：这里只做判断，不写 last_reply；真正发言成功后由
        MaiCore.mark_replied 记录，避免“判定命中但被其他插件抢走/发送失败”
        却仍然进入冷却。
        """
        self._track_message(group_id)
        now = time.time()
        last = self._last_reply.get(group_id, 0)
        elapsed = now - last

        # 规则 1：刚发过言 → 静默
        if elapsed < self.min_interval:
            return False

        # 规则 2：很久没发言 → 主动发言
        if elapsed > self.max_interval:
            return True

        # 规则 3：群里非常冷（密度小于 0.2 条/分钟）→ 静默
        density = self._msg_density_per_min(group_id)
        if density < 0.2:
            return False

        # 规则 4：群里非常活跃（密度大于 5 条/分钟）→ 静默
        if density > 5:
            return False

        # 规则 5：最近几条提到机器人名字 → 回复
        bot_nickname = self.bot_nickname or '兔兔'
        recent_text = ' '.join(m.get('message', '') for m in recent_messages[-3:])
        if bot_nickname in recent_text:
            return True

        # 规则 6：最近 1 分钟内有 2 条以上消息且间隔适中 → 让 LLM 决策
        if density > 0.5 and self.llm and self.llm_fallback:
            times = self._group_msg_times.get(group_id, [])
            recent_minute = [t for t in times if t > now - 60]
            if len(recent_minute) >= 2:
                return await self._llm_decide(recent_messages, bot_user_id)

        # 默认：随机小概率主动（被动模式下不主动）
        return False

    async def _llm_decide(self, recent_messages: List[Dict], bot_user_id: str) -> bool:
        """LLM 兜底决策"""
        context_lines = []
        for m in recent_messages[-8:]:
            role = '我' if m.get('user_id') == bot_user_id else (m.get('nickname', '某人'))
            context_lines.append(f'{role}: {m.get("message", "")}')
        context = '\n'.join(context_lines)
        try:
            try:
                prompt = self.timing_prompt.format(context=context)
            except Exception:
                prompt = DEFAULT_TIMING_PROMPT.format(context=context)
            result = await self.llm.chat(
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0.1,
                max_tokens=8,
            )
            decision = result.strip().lower()
            return 'reply' in decision
        except Exception:
            return False