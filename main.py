# -*- coding: utf-8 -*-
"""
siwu-mai-chatbot 入口（兔兔 / 阿米娅）

关键修复：
1. 阿米娅全局前缀会剥掉「兔兔」→ 必须用 text_original / text_prefix 判断召唤
2. choice_handlers 很快就写好 data.verify → 兜底延迟改为约 1 秒
3. 配置变更自动重建核心；生成失败打日志
4. 人格/风格/黑话在观察后学习；黑话生成时关键词检索注入
"""

import os
import random
import asyncio
import hashlib

from core import AmiyaBotPluginInstance, log
from core import Message
from amiyabot.builtin.messageChain import Chain

from .mai import MaiCore

curr_dir = os.path.dirname(os.path.abspath(__file__))

bot = AmiyaBotPluginInstance(
    name='兔兔 - 智能聊天',
    version='1.12.0',
    plugin_id='siwu-mai-chatbot',
    plugin_type='functional',
    description='明日方舟阿米娅人设的群聊智能体。前缀/@召唤，评分接话，画像/黑话学习与检索，不抢其他插件。',
    document=f'{curr_dir}/README.md',
    global_config_default=f'{curr_dir}/config_default.yaml',
    global_config_schema=f'{curr_dir}/jsonSchema.json',
)

_mai_core: MaiCore = None
_mai_core_sig: str = ''
_mai_fallback_done: set = set()
_mai_pending_tasks: dict = {}


def _cfg(key: str, default=None, old_key: str = None):
    val = bot.get_config(key, channel_id=None)
    if val is not None:
        return val
    if old_key:
        val = bot.get_config(old_key, channel_id=None)
        if val is not None:
            return val
    return default


def _parse_list(raw, default=None):
    if default is None:
        default = []
    if raw is None:
        return list(default)
    if isinstance(raw, str):
        return [w.strip() for w in raw.replace('，', ',').split(',') if w.strip()]
    if isinstance(raw, list):
        return [str(w).strip() for w in raw if str(w).strip()]
    return list(default)


def _parse_trigger_words():
    return _parse_list(_cfg('mai_trigger_words', ['兔兔', '阿米娅']), ['兔兔', '阿米娅'])


def _mai_log(msg: str, force: bool = False):
    """调试日志：mai_debug_log=true 时输出；force 时始终输出"""
    if force or bool(_cfg('mai_debug_log', True)):
        log.info(f'[Mai] {msg}')


def _short(text: str, n: int = 40) -> str:
    text = (text or '').replace('\n', ' ').strip()
    if len(text) <= n:
        return text
    return text[:n] + '…'


PROMPT_KEYS = [
    'mai_bot_name',
    'mai_system_prompt',
    'mai_user_prompt_template',
    'mai_banned_phrases',
    'mai_persona_prompt',
    'mai_persona_user_template',
    'mai_style_expression_prompt',
    'mai_style_learn_prompt',
    'mai_timing_prompt',
    'mai_judge_prompt',
]


def _config_signature() -> str:
    keys = [
        'mai_api_key', 'mai_model', 'mai_base_url', 'mai_trigger_words',
        'mai_min_reactive_interval', 'mai_max_reactive_interval', 'mai_llm_fallback',
        'mai_persona_cooldown_hours', 'mai_persona_min_samples',
        'mai_expression_cooldown_hours', 'mai_jargon_cooldown_hours', 'mai_style_min_samples',
        'mai_temperature', 'mai_max_tokens', 'mai_max_reply_length',
        'mai_memory_window_hours', 'mai_max_context', 'mai_fallback_delay',
        'mai_followup_window_seconds', 'mai_followup_max_turns',
        'mai_judge_enabled', 'mai_judge_api_key', 'mai_judge_model', 'mai_judge_base_url',
        'mai_embedding_enabled', 'mai_embedding_api_key', 'mai_embedding_base_url', 'mai_embedding_model',
        'mai_debug_log', 'mai_proactive', 'mai_proactive_rate',
        *PROMPT_KEYS,
    ]
    blob = '|'.join(f'{k}={_cfg(k, "")}' for k in keys)
    return hashlib.md5(blob.encode('utf-8', errors='ignore')).hexdigest()


def _build_core() -> MaiCore:
    api_key = (_cfg('mai_api_key', '') or '').strip()
    if not api_key:
        raise ValueError('未配置 DeepSeek API Key（mai_api_key）')

    model = _cfg('mai_model', 'deepseek-chat') or 'deepseek-chat'
    base_url = (_cfg('mai_base_url', 'https://api.deepseek.com') or 'https://api.deepseek.com').rstrip('/')
    trigger_words = _parse_trigger_words()
    bot_name = (_cfg('mai_bot_name', '兔兔') or '兔兔').strip()

    judge_api_key = (_cfg('mai_judge_api_key', '') or '').strip() or None
    judge_model = (_cfg('mai_judge_model', '') or '').strip() or None
    judge_base_url = (_cfg('mai_judge_base_url', '') or '').strip() or None

    embedding_enabled = bool(_cfg('mai_embedding_enabled', False))
    embedding_api_key = (_cfg('mai_embedding_api_key', '') or '').strip() or None
    embedding_base_url = (_cfg('mai_embedding_base_url', '') or '').strip() or None
    embedding_model = (_cfg('mai_embedding_model', '') or '').strip() or None

    core = MaiCore(
        api_key=api_key,
        model=model,
        base_url=base_url,
        trigger_words=trigger_words,
        bot_name=bot_name,
        judge_api_key=judge_api_key,
        judge_model=judge_model,
        judge_base_url=judge_base_url,
        embedding_enabled=embedding_enabled,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url,
        embedding_model=embedding_model,
    )

    core.timing.update_config(
        min_interval=int(_cfg('mai_min_reactive_interval', 300, old_key='mai_min_reply_interval')),
        max_interval=int(_cfg('mai_max_reactive_interval', 1800, old_key='mai_max_reply_interval')),
        llm_fallback=bool(_cfg('mai_llm_fallback', True)),
        timing_prompt=_cfg('mai_timing_prompt', ''),
        bot_nickname=bot_name,
    )
    core.persona.update_config(
        cooldown_seconds=int(_cfg('mai_persona_cooldown_hours', 6)) * 3600,
        min_samples=int(_cfg('mai_persona_min_samples', 12)),
        system_prompt=_cfg('mai_persona_prompt', ''),
        user_template=_cfg('mai_persona_user_template', ''),
    )
    core.style_learner.update_config(
        # 表达 + 黑话合并为一次调用，冷却取两个配置中较小值（表达仍 3h，黑话顺带更勤）
        group_cooldown=min(
            int(_cfg('mai_expression_cooldown_hours', 3)) * 3600,
            int(_cfg('mai_jargon_cooldown_hours', 6)) * 3600,
        ),
        min_samples=int(_cfg('mai_style_min_samples', 15)),
        # 优先新合并提示词；未配则回退旧「表达」提示词（可能无 jargons 段），最后用内置默认
        group_prompt=_cfg('mai_style_learn_prompt', '') or _cfg('mai_style_expression_prompt', ''),
    )
    core.generator.update_config(
        temperature=float(_cfg('mai_temperature', 0.95)),
        max_tokens=int(_cfg('mai_max_tokens', 180)),
        max_length=int(_cfg('mai_max_reply_length', 100)),
        memory_hours=int(_cfg('mai_memory_window_hours', 168)),
        max_context=int(_cfg('mai_max_context', 16)),
        model=model,
        bot_name=bot_name,
        system_prompt=_cfg('mai_system_prompt', ''),
        user_prompt_template=_cfg('mai_user_prompt_template', ''),
        banned_phrases=_parse_list(_cfg('mai_banned_phrases', None), []) or None,
    )
    core.judge.update_config(
        enabled=bool(_cfg('mai_judge_enabled', True)),
        model=judge_model or model,
        prompt=_cfg('mai_judge_prompt', ''),
        bot_name=bot_name,
        max_history=int(_cfg('mai_max_context', 16)),
    )
    # 对话续聊窗口：被召唤回复后，窗口内别人继续说话会直接续聊
    core.FOLLOWUP_WINDOW_SECONDS = max(0, int(_cfg('mai_followup_window_seconds', 120)))
    core.FOLLOWUP_MAX_TURNS = max(1, int(_cfg('mai_followup_max_turns', 4)))
    j_model = judge_model or model
    j_url = judge_base_url or base_url
    _mai_log(
        f'核心刷新 model={model} judge={j_model}@{j_url} '
        f'proactive={bool(_cfg("mai_proactive", False))} '
        f'judge_enabled={bool(_cfg("mai_judge_enabled", True))} '
        f'embedding={"on" if embedding_enabled else "off"}'
        f'{"/默认模型" if embedding_enabled and not embedding_model else ""} '
        f'debug={bool(_cfg("mai_debug_log", True))}',
        force=True,
    )
    return core


def _get_core() -> MaiCore:
    global _mai_core, _mai_core_sig
    sig = _config_signature()
    if _mai_core is None or sig != _mai_core_sig:
        old_history = getattr(_mai_core, '_history', None) if _mai_core else None
        old_bot_uid = getattr(_mai_core, 'bot_user_id', None) if _mai_core else None
        old_last_reply = None
        if _mai_core and getattr(_mai_core, 'timing', None):
            old_last_reply = dict(getattr(_mai_core.timing, '_last_reply', {}) or {})
        _mai_core = _build_core()
        _mai_core_sig = sig
        # 改配置重建核心时保留短期历史（含兔兔自己的发言），避免评分丢上下文
        if isinstance(old_history, dict) and old_history:
            _mai_core._history = old_history
        if old_bot_uid:
            _mai_core.set_bot_user_id(str(old_bot_uid))
        if old_last_reply and getattr(_mai_core, 'timing', None):
            _mai_core.timing._last_reply.update(old_last_reply)
    return _mai_core

def install():
    global _mai_core, _mai_core_sig
    _mai_core = None
    _mai_core_sig = ''


def _is_mai_enabled() -> bool:
    return bool(_cfg('mai_enabled', True))


def _message_texts(data: Message):
    """返回 (展示/记忆用原文, 前缀剥掉后的正文)"""
    original = (getattr(data, 'text_original', None) or data.text or '').strip()
    body = (data.text or '').strip()
    prefix = (getattr(data, 'text_prefix', None) or '').strip()
    return original, body, prefix


def _address_reason(data: Message) -> str:
    """返回召唤原因；未召唤则为空串"""
    if bool(getattr(data, 'is_at', False)):
        return 'is_at'
    at_target = getattr(data, 'at_target', None) or []
    try:
        bot_id = str(getattr(getattr(data, 'instance', None), 'appid', '') or '')
        if bot_id and str(bot_id) in [str(x) for x in at_target]:
            return 'at_target'
    except Exception:
        pass
    original, body, prefix = _message_texts(data)
    if prefix:
        return f'prefix:{prefix}'
    triggers = _parse_trigger_words()
    haystack = original or body
    for w in triggers:
        if w and w in haystack:
            return f'trigger:{w}'
    return ''


def _is_addressed_to_bot(data: Message) -> bool:
    """用户是否在召唤机器人"""
    return bool(_address_reason(data))


def _other_plugin_claimed(data: Message) -> bool:
    """choice_handlers 选中其他功能后会写入 data.verify"""
    verify = getattr(data, 'verify', None)
    if verify is None:
        return False
    try:
        if hasattr(verify, 'result'):
            return bool(verify.result)
        return bool(verify)
    except Exception:
        return False


async def _send_mai_reply(data: Message, source: str = ''):
    original, body, prefix = _message_texts(data)
    group_id = str(data.channel_id) if data.channel_id else 'dm'
    user_id = str(data.user_id) if data.user_id else 'unknown'
    nickname = getattr(data, 'nickname', None) or '群友'
    focus = body or original

    try:
        core = _get_core()
    except ValueError as e:
        log.warning(f'[Mai] {e}')
        return

    try:
        appid = str(getattr(data.instance, 'appid', '') or '')
        if appid:
            core.set_bot_user_id(appid)
    except Exception:
        pass

    _mai_log(f'开始生成回复 group={group_id} user={nickname}({user_id}) focus={_short(focus)}')
    try:
        replies = await core.generate_reply(
            user_id=user_id,
            group_id=group_id,
            focus_text=focus,
            speaker_name=nickname,
        )
    except Exception as e:
        log.warning(f'[Mai] 生成失败: {e}')
        return

    meta = getattr(getattr(core, 'generator', None), 'last_empty_meta', None) or {}
    if meta.get('used_fallback'):
        _mai_log(f'生成空结果已兜底 meta={meta}', force=True)
    elif not replies:
        log.warning(f'[Mai] 生成结果为空 meta={meta}')
        return

    # 连发多条：第一条立即发送，后续每条间隔 1~1.5s，模拟真人分段说话
    is_followup = str(source).startswith('followup')
    for i, text in enumerate(replies):
        if i > 0:
            await asyncio.sleep(1.0 + random.random() * 0.5)
        _mai_log(f'发送回复 group={group_id} [{i + 1}/{len(replies)}] len={len(text)} text={_short(text, 60)}')
        await data.send(Chain(data, at=False).text(text))

    # 被召唤 / 主动发言触发的回复 → 开启对话窗口（窗口内别人继续说话会直接续聊）；续聊 → 计数
    try:
        if is_followup:
            core.followup_replied(group_id)
        elif str(source).startswith('addressed') or str(source) in ('judge', 'legacy'):
            await core.enter_followup(group_id)
    except Exception as e:
        log.warning(f'[Mai] 对话窗口更新失败: {e}')


def _spawn_bg(coro):
    """创建后台任务并吞掉未捕获异常，避免 "Task exception was never retrieved" 刷屏。"""
    task = asyncio.create_task(coro)

    def _on_done(t):
        if t.cancelled():
            return
        exc = t.exception()
        if exc:
            log.warning(f'[Mai] 后台任务异常: {exc}')

    task.add_done_callback(_on_done)
    return task


def _schedule_fallback(data: Message, source: str):
    """排队兜底回复；限制 pending 数量，避免消息洪峰下无限增长。"""
    msg_key = getattr(data, 'message_id', None) or id(data)
    if msg_key in _mai_pending_tasks:
        return
    # 清掉已完成的任务，防止 pending 无限增长
    if len(_mai_pending_tasks) > 100:
        for k, t in list(_mai_pending_tasks.items()):
            if t.done():
                _mai_pending_tasks.pop(k, None)
    _mai_log(f'排队兜底回复 source={source} msg={msg_key}')
    _mai_pending_tasks[msg_key] = asyncio.create_task(
        _fallback_check_and_reply(data, source=source)
    )


async def _fallback_check_and_reply(data: Message, source: str = 'unknown'):
    msg_key = getattr(data, 'message_id', None) or id(data)
    if msg_key in _mai_fallback_done:
        _mai_log(f'兜底跳过(已处理) source={source} msg={msg_key}')
        return
    _mai_fallback_done.add(msg_key)
    if len(_mai_fallback_done) > 2000:
        for i in list(_mai_fallback_done)[:1000]:
            _mai_fallback_done.discard(i)

    delay = float(_cfg('mai_fallback_delay', 1.0))
    delay = max(0.3, min(delay, 5.0))
    _mai_log(f'兜底等待 {delay:.1f}s source={source} msg={msg_key}')
    await asyncio.sleep(delay)

    if _other_plugin_claimed(data):
        verify = getattr(data, 'verify', None)
        _mai_log(f'让出：其他插件已认领 source={source} verify={verify!r}')
        _mai_pending_tasks.pop(msg_key, None)
        return

    _mai_log(f'兜底无人认领，进入生成 source={source}')
    try:
        await _send_mai_reply(data, source=source)
    except Exception as e:
        log.warning(f'[Mai] 兜底回复失败: {e}')
    finally:
        _mai_pending_tasks.pop(msg_key, None)


@bot.message_created
async def _mai_observe(data: Message, _):
    if not _is_mai_enabled():
        return

    original, body, prefix = _message_texts(data)
    if not original and not body:
        return

    user_id = str(data.user_id) if data.user_id else 'unknown'

    try:
        core = _get_core()
    except ValueError as e:
        _mai_log(f'跳过：核心未就绪 ({e})', force=True)
        return

    bot_uid = str(getattr(getattr(data, 'instance', None), 'appid', '') or core.bot_user_id)
    if bot_uid:
        core.set_bot_user_id(bot_uid)

    group_id = str(data.channel_id) if data.channel_id else 'dm'
    nickname = getattr(data, 'nickname', None) or '群友'
    observe_text = original or body

    # 自己的消息也要进历史（评分靠它判断 reply_to_bot）；不做主动接话
    if core.is_bot_user(user_id) or user_id == 'bot':
        core.observe_message(group_id, user_id, core.BOT_NAME, observe_text)
        _mai_log(f'记录自身消息 group={group_id} text={_short(observe_text)}')
        return

    core.observe_message(group_id, user_id, nickname, observe_text)
    # 统一记录消息时间：主动发言的活跃度估算依赖它（judge / legacy 共用一份）
    try:
        core.track_message(group_id)
    except Exception:
        pass
    # 画像/风格/黑话学习挂在观察后：此前只在成功回复后触发，未召唤时永远不写画像
    _spawn_bg(core.on_message_post(group_id, user_id, nickname, observe_text))

    reason = _address_reason(data)
    addressed = bool(reason)
    proactive = bool(_cfg('mai_proactive', False))
    judge_on = bool(_cfg('mai_judge_enabled', True))
    _mai_log(
        f'观察 group={group_id} user={nickname}({user_id}) '
        f'addressed={addressed}{f"({reason})" if reason else ""} '
        f'proactive={proactive} judge={judge_on} '
        f'text={_short(observe_text)}'
    )

    # 主动发言 / 轻量评分接话：被召唤以外的消息
    # 对话续聊窗口优先：刚被召唤回复过，窗口内别人继续说话 → 直接续聊，不评分
    if not addressed:
        try:
            in_window = await core.in_followup_window(group_id, bot_uid)
        except Exception as e:
            in_window = False
            log.warning(f'[Mai] 对话窗口判断失败: {e}')
        if in_window:
            _mai_log(f'对话窗口内续聊 group={group_id} user={nickname}({user_id}) text={_short(observe_text)}')
            _schedule_fallback(data, source='followup')
            return

    if not addressed and proactive:
        observe_for_reply = original or body
        should_try = False
        path = 'none'

        if judge_on:
            try:
                hit, score, detail = await core.maybe_judge_reply(
                    group_id=group_id,
                    focus_text=observe_for_reply,
                    speaker_name=nickname,
                    bot_user_id=bot_uid,
                )
                _mai_log(f'接话评分 score={score} hit={hit} detail={detail}')
                if hit:
                    should_try = True
                    path = 'judge'
            except Exception as e:
                log.warning(f'[Mai] 接话评分失败: {e}')
        else:
            rate = float(_cfg('mai_proactive_rate', 0.08))
            roll = random.random()
            if roll <= rate:
                gate = await core.maybe_proactive_reply(group_id, bot_uid)
                _mai_log(f'旧主动逻辑 rate={rate} roll={roll:.3f} timing_gate={gate}')
                if gate:
                    should_try = True
                    path = 'legacy'
            else:
                _mai_log(f'旧主动逻辑未过概率 rate={rate} roll={roll:.3f}')

        if should_try:
            _schedule_fallback(data, source=path)
            return
        return

    if not addressed and not proactive:
        _mai_log('未召唤且未开主动发言 → 仅观察')
        return

    # 被召唤：等其他插件；没人要再回
    if addressed:
        _schedule_fallback(data, source=f'addressed:{reason}')
