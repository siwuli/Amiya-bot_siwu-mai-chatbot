# -*- coding: utf-8 -*-
"""Mai 核心调度：观察、记忆、生成、学习"""

import asyncio
import os
import time
from threading import Lock
from typing import Optional, Dict, List

from .llm import DeepSeekClient, EmbeddingClient
from .memory import A_Memorix
from .persona import PersonaExtractor
from .timing import TimingGate
from .style import StyleLearner
from .generator import ChatGenerator
from .judge import ReplyJudge


def _mai_log(level: str, msg: str):
    """延迟导入框架日志；非 Amiya 环境（如单测）下静默。"""
    try:
        from core import log as _log
        getattr(_log, level, _log.info)(f'[Mai] {msg}')
    except Exception:
        pass


class MaiCore:
    BOT_NAME = '兔兔'
    MAX_MSG_RECORD_LEN = 240
    # 只有看起来「值得记住」的消息才落库，避免记忆=聊天流水账
    MEMORABLE_MIN_LEN = 8

    def __init__(
        self,
        api_key: str,
        model: str = 'deepseek-chat',
        base_url: str = 'https://api.deepseek.com',
        data_dir: Optional[str] = None,
        trigger_words: Optional[List[str]] = None,
        bot_name: str = '兔兔',
        judge_api_key: Optional[str] = None,
        judge_model: Optional[str] = None,
        judge_base_url: Optional[str] = None,
        embedding_enabled: bool = False,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.model = model
        self.BOT_NAME = bot_name or '兔兔'
        self.llm = DeepSeekClient(api_key, base_url=base_url, default_model=model)

        # 接话评分可用独立轻量模型（key/url 缺省时复用主模型）
        j_key = (judge_api_key or api_key or '').strip()
        j_url = (judge_base_url or base_url or '').rstrip('/') or base_url
        j_model = (judge_model or model or 'deepseek-chat').strip()
        if j_key == api_key and j_url.rstrip('/') == base_url.rstrip('/'):
            self.judge_llm = self.llm
        else:
            self.judge_llm = DeepSeekClient(j_key, base_url=j_url, default_model=j_model)

        if data_dir is None:
            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                '..',
                '..',
                'data',
                'mai_memory.db',
            )
        data_dir = os.path.abspath(data_dir)
        os.makedirs(os.path.dirname(data_dir), exist_ok=True)
        self.memory = A_Memorix(data_dir)

        # 向量记忆：OpenAI 兼容 /embeddings。DeepSeek 无 embedding 接口，需另配服务
        # 未启用 / 未配全 / 服务不可用 → 自动停用并提示，绝不影响聊天主流程
        self.embedder = None
        self._embed_fail_streak = 0
        if embedding_enabled:
            emb_key = (embedding_api_key or api_key or '').strip()
            emb_url = (embedding_base_url or '').strip() or base_url
            emb_model = (embedding_model or '').strip() or 'text-embedding-3-small'
            if emb_key and emb_url:
                try:
                    self.embedder = EmbeddingClient(emb_key, base_url=emb_url, model=emb_model)
                except Exception:
                    self.embedder = None
            if self.embedder is not None:
                if 'deepseek.com' in emb_url:
                    _mai_log(
                        'warning',
                        f'向量记忆已启用但 base_url 指向 DeepSeek（{emb_url}），其接口没有 /embeddings，'
                        f'语义检索将无法工作。请在控制台把 mai_embedding_base_url 换成支持 embedding 的服务'
                        f'（如 https://api.siliconflow.cn/v1 + BAAI/bge-m3），或关闭 mai_embedding_enabled。',
                    )
                if not (embedding_model or '').strip():
                    _mai_log('info', f'向量记忆已启用，未填写 embedding 模型，默认使用 {emb_model}')

        self.persona = PersonaExtractor(self.llm)
        self.timing = TimingGate(llm=self.llm)
        self.style_learner = StyleLearner(self.llm, self.memory)
        # embedder 传熔断保护的封装函数，保证读取路径也跟随自动停用
        self.generator = ChatGenerator(
            self.llm,
            self.memory,
            model=model,
            embedder=self._safe_embed_one if self.embedder is not None else None,
        )
        self.generator.bot_name = self.BOT_NAME
        self.judge = ReplyJudge(llm=self.judge_llm, model=j_model)
        self.judge.bot_name = self.BOT_NAME

        self.trigger_words = trigger_words or ['兔兔']
        self._history: Dict[str, List[Dict]] = {}
        self._lock = Lock()
        self.bot_user_id = 'bot'

        # 启动后台回填：给历史记忆（含 persona 写的 fact）补向量
        if self.embedder is not None:
            self._spawn(self._backfill_embeddings())

    def _spawn(self, coro):
        """在运行中的事件循环里派发后台协程；无事件循环（如单元测试）时静默跳过。"""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            pass

    def _disable_embedder(self, reason: str):
        """连续失败熔断：停用向量检索，避免对错误配置反复发请求。"""
        if self.embedder is None:
            return
        self.embedder = None
        _mai_log(
            'warning',
            f'向量记忆已自动停用（{reason}）。聊天不受影响，仅失去语义检索；'
            f'请检查 mai_embedding_api_key / base_url / model 配置。',
        )

    async def _safe_embed_one(self, text: str) -> Optional[List[float]]:
        """单条向量化 + 熔断；任何失败返回 None，不向上抛。"""
        embedder = self.embedder
        if embedder is None:
            return None
        try:
            vec = await embedder.embed_one(text)
            self._embed_fail_streak = 0
            return vec
        except Exception as e:
            self._embed_fail_streak += 1
            if self._embed_fail_streak >= 3:
                self._disable_embedder(f'连续 {self._embed_fail_streak} 次调用失败: {e}')
            return None

    async def _safe_embed_many(self, texts: List[str]) -> List[List[float]]:
        """批量向量化 + 熔断；失败返回空列表。"""
        if not texts:
            return []
        embedder = self.embedder
        if embedder is None:
            return []
        try:
            vecs = await embedder.embed(texts)
            self._embed_fail_streak = 0
            return vecs
        except Exception as e:
            self._embed_fail_streak += 1
            if self._embed_fail_streak >= 3:
                self._disable_embedder(f'连续 {self._embed_fail_streak} 次调用失败: {e}')
            return []

    async def _embed_chunk(self, scope_id: str, content: str):
        """给单条记忆生成并补写向量；任何失败都不影响主流程。"""
        try:
            vec = await self._safe_embed_one(content)
            if vec:
                self.memory.update_chunk_embedding(scope_id, content, vec)
        except Exception:
            pass

    async def _backfill_embeddings(self):
        """分批把库内没有向量的记忆全部补上，避免旧数据永远搜不到。"""
        try:
            while True:
                pending = self.memory.pending_embeddings(limit=20)
                if not pending:
                    break
                vecs = await self._safe_embed_many([c for _, c in pending])
                if not vecs:
                    break
                if len(vecs) != len(pending):
                    break
                for (uid, content), vec in zip(pending, vecs):
                    if vec:
                        self.memory.update_chunk_embedding(uid, content, vec)
                await asyncio.sleep(0.5)
        except Exception:
            pass

    def set_bot_user_id(self, bot_user_id: str):
        if bot_user_id:
            self.bot_user_id = str(bot_user_id)

    def _truncate(self, s: str) -> str:
        if len(s) > self.MAX_MSG_RECORD_LEN:
            return s[: self.MAX_MSG_RECORD_LEN] + '…'
        return s

    def _ensure_group(self, group_id: str):
        if group_id not in self._history:
            # 冷启动 / 重建核心后：从库恢复含机器人发言的短期历史
            loaded = []
            try:
                loaded = self.memory.load_chat_tail(group_id, limit=80)
            except Exception:
                loaded = []
            self._history[group_id] = loaded

    def record_message(
        self,
        group_id: str,
        user_id: str,
        nickname: str,
        message: str,
        is_bot: bool = False,
    ):
        self._ensure_group(group_id)
        text = self._truncate(message)
        if not text:
            return
        now = time.time()
        msg = {
            'user_id': str(user_id or ''),
            'nickname': nickname or ('兔兔' if is_bot else '群友'),
            'message': text,
            'time': now,
            'is_bot': bool(is_bot),
        }
        with self._lock:
            hist = self._history[group_id]
            # 去重：发送后平台回传自己的消息时避免双记
            if hist:
                last = hist[-1]
                if (
                    last.get('is_bot') == msg['is_bot']
                    and str(last.get('user_id')) == msg['user_id']
                    and (last.get('message') or '') == text
                    and now - float(last.get('time') or 0) < 8
                ):
                    return
            hist.append(msg)
            if len(hist) > 80:
                self._history[group_id] = hist[-80:]

        try:
            self.memory.append_chat_tail(
                group_id=group_id,
                user_id=msg['user_id'],
                nickname=msg['nickname'],
                message=text,
                is_bot=bool(is_bot),
                created_at=now,
                keep=80,
            )
        except Exception:
            pass

    def is_bot_user(self, user_id: str) -> bool:
        uid = str(user_id or '')
        if not uid or uid == 'unknown':
            return False
        if uid in ('bot', str(self.bot_user_id)):
            return True
        return False

    def _looks_memorable(self, message: str) -> bool:
        text = (message or '').strip()
        if len(text) < self.MEMORABLE_MIN_LEN:
            return False
        # 纯表情/刷屏/短应答不入库
        noise = {'好', '哈哈', '哈哈哈', '嗯', '哦', '啊', '？', '?', '草', '卧槽', '6', '66', '666'}
        if text in noise:
            return False
        if len(set(text)) <= 2 and len(text) <= 6:
            return False
        return True

    def observe_message(self, group_id: str, user_id: str, nickname: str, message: str):
        """观察：记短期上下文 + 喂人格样本；重要内容再写长期记忆。

        若是机器人自己的消息：只记历史（供评分识别 reply_to_bot），不做画像/学习。
        """
        if self.is_bot_user(user_id):
            self.record_message(
                group_id,
                self.bot_user_id or user_id,
                self.BOT_NAME,
                message,
                is_bot=True,
            )
            return

        self.record_message(group_id, user_id, nickname, message, is_bot=False)

        # 人格管道：必须喂消息，否则永远不会画像
        self.persona.add_messages(user_id, [message])
        if nickname:
            try:
                self.memory.write_profile(user_id, nickname=nickname)
            except Exception:
                pass

        if self._looks_memorable(message):
            try:
                # 群维度 + 用户维度各留一份，便于检索
                stamp = time.strftime('%m-%d %H:%M')
                group_content = f'[{stamp}] {nickname}: {self._truncate(message)}'
                user_content = f'[{stamp}] {self._truncate(message)}'
                self.memory.write_chunk(
                    user_id=group_id,
                    content=group_content,
                    chunk_type='chat',
                    importance=0.4,
                )
                self.memory.write_chunk(
                    user_id=user_id,
                    content=user_content,
                    chunk_type='chat',
                    importance=0.45,
                )
                self._spawn(self._embed_chunk(group_id, group_content))
                self._spawn(self._embed_chunk(user_id, user_content))
            except Exception:
                pass

    def get_recent_messages(self, group_id: str) -> List[Dict]:
        self._ensure_group(group_id)
        return list(self._history.get(group_id, []))

    def should_reply_passive(self, group_id: str, text: str, is_at: bool) -> bool:
        matches = TimingGate.match_trigger_word(text, self.trigger_words)
        return is_at or matches

    async def maybe_proactive_reply(self, group_id: str, bot_user_id: str) -> bool:
        if bot_user_id:
            self.set_bot_user_id(bot_user_id)
        recent = self.get_recent_messages(group_id)
        return await self.timing.should_proactive_reply(group_id, recent, bot_user_id or self.bot_user_id)

    async def maybe_judge_reply(
        self,
        group_id: str,
        focus_text: str,
        speaker_name: str = '群友',
        bot_user_id: Optional[str] = None,
    ) -> tuple:
        """轻量模型标签评分 + 状态惩罚 + 概率抽样。

        返回 (是否接话, 分数, 原因)。
        """
        if not self.judge.enabled:
            return False, 0, 'disabled'

        if bot_user_id:
            self.set_bot_user_id(bot_user_id)
        uid = bot_user_id or self.bot_user_id
        recent = self.get_recent_messages(group_id)
        last_ts = float(self.timing._last_reply.get(group_id, 0) or 0)
        # 与 TimingGate 间隔对齐，用作 cooldown_penalty 衰减窗口（软惩罚，不硬拦截）
        self.judge.cooldown_seconds = int(getattr(self.timing, 'min_interval', 60) or 60)
        try:
            hit, score, reason = await self.judge.should_reply(
                recent_messages=recent,
                focus_text=focus_text,
                speaker_name=speaker_name,
                bot_user_id=uid,
                last_reply_ts=last_ts,
            )
        except Exception as e:
            return False, 0, f'error:{e}'
        # 注意：真正发言成功后再写 last_reply，避免评分命中但最终没发出去仍冷却
        return hit, score, reason

    def mark_replied(self, group_id: str):
        self.timing._last_reply[group_id] = time.time()

    async def generate_reply(
        self,
        user_id: str,
        group_id: str,
        focus_text: str = '',
        speaker_name: str = '群友',
    ) -> str:
        recent = self.get_recent_messages(group_id)
        reply = await self.generator.generate(
            user_id,
            group_id,
            recent,
            bot_user_id=self.bot_user_id,
            focus_text=focus_text,
            speaker_name=speaker_name,
        )
        if reply:
            # 必须写入历史：评分靠 is_bot / reply_to_bot 识别自己说过什么
            self.record_message(group_id, self.bot_user_id, self.BOT_NAME, reply, is_bot=True)
            self.mark_replied(group_id)
            try:
                stamp = time.strftime('%m-%d %H:%M')
                content = f'[{stamp}] {self.BOT_NAME}: {self._truncate(reply)}'
                self.memory.write_chunk(
                    user_id=group_id,
                    content=content,
                    chunk_type='chat',
                    importance=0.55,
                )
                self._spawn(self._embed_chunk(group_id, content))
            except Exception:
                pass
        return reply

    async def on_message_post(self, group_id: str, user_id: str, nickname: str, message: str):
        """每条群友消息后异步学习（画像 / 群风格 / 黑话），不依赖是否回复。"""
        if self.is_bot_user(user_id):
            return
        recent = self.get_recent_messages(group_id)
        asyncio.create_task(self.style_learner.learn_expressions(group_id, recent))
        asyncio.create_task(self.style_learner.mine_jargons(group_id, recent))
        asyncio.create_task(self._extract_persona_safe(user_id))

    async def _extract_persona_safe(self, user_id: str):
        try:
            profile = await self.persona.extract_if_needed(user_id, self.memory)
            if profile is not None:
                n_traits = len(profile.get('traits') or {})
                n_prefs = len(profile.get('preferences') or {})
                n_style = len(profile.get('speaking_style') or {})
                n_facts = len(profile.get('stable_facts') or {})
                # 延迟导入，避免非 Amiya 环境下硬依赖
                try:
                    from core import log as _log
                    _log.info(
                        f'[Mai] 画像已更新 user={user_id} '
                        f'traits={n_traits} prefs={n_prefs} style={n_style} facts={n_facts}'
                    )
                except Exception:
                    pass
        except Exception as e:
            try:
                from core import log as _log
                _log.warning(f'[Mai] 画像提取失败 user={user_id}: {e}')
            except Exception:
                pass

    async def close(self):
        await self.llm.close()
        if self.judge_llm is not self.llm:
            await self.judge_llm.close()
        if self.embedder is not None:
            try:
                await self.embedder.close()
            except Exception:
                pass
