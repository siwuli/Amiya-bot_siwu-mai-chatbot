# -*- coding: utf-8 -*-
"""通用工具函数"""

import re
from typing import Optional


def extract_json(text: str) -> Optional[str]:
    """从 LLM 输出中提取 JSON 字符串"""
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        return m.group(1).strip()
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        return m.group(0)
    return None


def strip_thinking(text: str) -> str:
    """去掉模型思考过程，只保留最终可见回复。"""
    if not text:
        return ''
    s = str(text)

    # DeepSeek / 常见推理标签（含未闭合）
    patterns = [
        r'<think\b[^>]*>[\s\S]*?</think\s*>',
        r'<thinking\b[^>]*>[\s\S]*?</thinking\s*>',
        r'<reason(?:ing)?\b[^>]*>[\s\S]*?</reason(?:ing)?\s*>',
        r'<reflection\b[^>]*>[\s\S]*?</reflection\s*>',
        r'【思考】[\s\S]*?【(?:回答|回复|答案)】',
        r'(?:^|\n)\s*(?:思考过程|推理过程|分析)\s*[:：][\s\S]*?(?=\n\s*(?:回复|回答|最终)[:：]|$)',
    ]
    for pat in patterns:
        s = re.sub(pat, '', s, flags=re.I)

    # 残留开闭标签 / 单独出现的 </think>
    s = re.sub(r'</?(?:think|thinking|reason(?:ing)?|reflection)\b[^>]*>', '', s, flags=re.I)

    # 「回复：」「最终回答：」之后才是正文
    m = re.search(r'(?:^|\n)\s*(?:最终)?(?:回复|回答)\s*[:：]\s*', s)
    if m:
        after = s[m.end():].strip()
        if after:
            s = after

    return s.strip()