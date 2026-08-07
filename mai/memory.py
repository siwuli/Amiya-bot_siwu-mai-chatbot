# -*- coding: utf-8 -*-
"""轻量记忆引擎：FTS 检索 + 用户画像 + 群风格"""

import json
import re
import time
import sqlite3
from threading import Lock
from typing import Optional, List, Dict, Any


class A_Memorix:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._lock = Lock()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        return conn

    def _init_db(self):
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id     TEXT NOT NULL,
                chunk_type  TEXT NOT NULL DEFAULT 'chat',
                content     TEXT NOT NULL,
                embedding   BLOB,
                importance  REAL DEFAULT 0.5,
                created_at  REAL DEFAULT (unixepoch()),
                source_msg  TEXT,
                UNIQUE(user_id, content)
            )
        ''')

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                content,
                content='memory_chunks',
                content_rowid='id'
            )
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS memory_ai_insert
            AFTER INSERT ON memory_chunks BEGIN
                INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS memory_ai_delete
            AFTER DELETE ON memory_chunks BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content)
                    VALUES('delete', old.id, old.content);
            END
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS memory_graph (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                entity     TEXT NOT NULL,
                relation   TEXT NOT NULL,
                target     TEXT,
                weight     REAL DEFAULT 1.0,
                evidence   TEXT,
                created_at REAL DEFAULT (unixepoch()),
                UNIQUE(user_id, entity, relation, target)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id        TEXT PRIMARY KEY,
                nickname       TEXT,
                traits         TEXT,
                preferences    TEXT,
                speaking_style TEXT,
                recent_info    TEXT,
                updated_at     REAL DEFAULT (unixepoch())
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS group_style (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                pattern    TEXT NOT NULL,
                count      INTEGER DEFAULT 1,
                updated_at REAL DEFAULT (unixepoch()),
                UNIQUE(group_id, pattern)
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS jargon_table (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id   TEXT NOT NULL,
                term       TEXT NOT NULL,
                meaning    TEXT,
                level      INTEGER DEFAULT 0,
                count      INTEGER DEFAULT 1,
                updated_at REAL DEFAULT (unixepoch()),
                UNIQUE(group_id, term)
            )
        ''')

        # 短期群聊尾巴（含机器人自己的发言），供接话评分 / 生成上下文，重启不丢
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_tail (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                group_id   TEXT NOT NULL,
                user_id    TEXT NOT NULL,
                nickname   TEXT,
                message    TEXT NOT NULL,
                is_bot     INTEGER DEFAULT 0,
                created_at REAL DEFAULT (unixepoch())
            )
        ''')
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_chat_tail_group_time
            ON chat_tail(group_id, created_at DESC)
        ''')

        conn.commit()
        conn.close()

    def append_chat_tail(
        self,
        group_id: str,
        user_id: str,
        nickname: str,
        message: str,
        is_bot: bool = False,
        created_at: Optional[float] = None,
        keep: int = 80,
    ):
        if not group_id or not message or not str(message).strip():
            return
        ts = float(created_at or time.time())
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute(
                    '''
                    INSERT INTO chat_tail (group_id, user_id, nickname, message, is_bot, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ''',
                    (
                        str(group_id),
                        str(user_id or ''),
                        nickname or '',
                        str(message).strip(),
                        1 if is_bot else 0,
                        ts,
                    ),
                )
                # 只保留最近 keep 条
                cursor.execute(
                    '''
                    DELETE FROM chat_tail
                    WHERE group_id = ? AND id NOT IN (
                        SELECT id FROM chat_tail
                        WHERE group_id = ?
                        ORDER BY created_at DESC, id DESC
                        LIMIT ?
                    )
                    ''',
                    (str(group_id), str(group_id), max(10, int(keep))),
                )
            except Exception:
                pass
            conn.commit()
            conn.close()

    def load_chat_tail(self, group_id: str, limit: int = 80) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(
                '''
                SELECT user_id, nickname, message, is_bot, created_at
                FROM chat_tail
                WHERE group_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                ''',
                (str(group_id), max(1, int(limit))),
            )
            rows = cursor.fetchall()
        except Exception:
            rows = []
        conn.close()
        # 库里是新→旧，内存历史要旧→新
        out = []
        for r in reversed(rows):
            out.append(
                {
                    'user_id': str(r[0] or ''),
                    'nickname': r[1] or '',
                    'message': r[2] or '',
                    'is_bot': bool(r[3]),
                    'time': float(r[4] or 0),
                }
            )
        return out

    def write_chunk(
        self,
        user_id: str,
        content: str,
        chunk_type: str = 'chat',
        embedding: Optional[List[float]] = None,
        importance: float = 0.5,
        source_msg: Optional[str] = None,
    ):
        if not content or not content.strip():
            return
        content = content.strip()
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            emb_bytes = json.dumps(embedding).encode() if embedding else None
            try:
                cursor.execute('''
                    INSERT INTO memory_chunks
                        (user_id, chunk_type, content, embedding, importance, source_msg)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, content) DO UPDATE SET
                        importance=MAX(importance, excluded.importance),
                        created_at=unixepoch()
                ''', (user_id, chunk_type, content, emb_bytes, importance, source_msg))
            except Exception:
                pass
            conn.commit()
            conn.close()

    def update_chunk_embedding(self, user_id: str, content: str, embedding: List[float]):
        """给已写入的记忆片段补向量（写入时可能还没向量，后台异步补齐）。"""
        if not content or not embedding:
            return
        with self._lock:
            conn = self._get_conn()
            try:
                conn.execute(
                    'UPDATE memory_chunks SET embedding=? WHERE user_id=? AND content=?',
                    (json.dumps(embedding).encode(), str(user_id), str(content).strip()),
                )
            except Exception:
                pass
            conn.commit()
            conn.close()

    def pending_embeddings(self, limit: int = 50) -> List[tuple]:
        """返回还没有向量的记忆片段 (user_id, content)，供后台回填。"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                'SELECT user_id, content FROM memory_chunks '
                'WHERE embedding IS NULL ORDER BY id LIMIT ?',
                (max(1, int(limit)),),
            ).fetchall()
        except Exception:
            rows = []
        conn.close()
        return [(str(r[0] or ''), str(r[1] or '')) for r in rows if r[1]]

    def write_graph(
        self,
        user_id: str,
        entity: str,
        relation: str,
        target: Optional[str] = None,
        weight: float = 1.0,
        evidence: Optional[str] = None,
    ):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO memory_graph (user_id, entity, relation, target, weight, evidence)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(user_id, entity, relation, target)
                        DO UPDATE SET weight=MAX(weight, excluded.weight),
                                      evidence=COALESCE(excluded.evidence, evidence)
                ''', (user_id, entity, relation, target, weight, evidence))
            except Exception:
                pass
            conn.commit()
            conn.close()

    def write_profile(self, user_id: str, **fields):
        """更新用户画像。允许 traits/preferences/speaking_style/recent_info/nickname。
        stable_facts 会合并进 recent_info。
        """
        allowed = {'traits', 'preferences', 'speaking_style', 'recent_info', 'nickname'}
        if 'stable_facts' in fields and fields['stable_facts']:
            existing = self.get_profile(user_id).get('recent_info') or {}
            if isinstance(existing, dict) and isinstance(fields['stable_facts'], dict):
                merged = {**existing, **fields['stable_facts']}
            else:
                merged = fields['stable_facts']
            fields['recent_info'] = merged

        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            for key, val in fields.items():
                if key not in allowed:
                    continue
                if key == 'nickname':
                    payload = str(val) if val is not None else ''
                else:
                    # 合并字典字段，避免覆盖旧画像
                    if isinstance(val, dict):
                        old = {}
                        cursor.execute(f'SELECT {key} FROM user_profile WHERE user_id=?', (user_id,))
                        row = cursor.fetchone()
                        if row and row[0]:
                            try:
                                old = json.loads(row[0]) or {}
                            except Exception:
                                old = {}
                        if isinstance(old, dict):
                            val = {**old, **val}
                    payload = json.dumps(val, ensure_ascii=False)
                cursor.execute(f'''
                    INSERT INTO user_profile (user_id, {key}, updated_at)
                    VALUES (?, ?, unixepoch())
                    ON CONFLICT(user_id) DO UPDATE SET {key}=?, updated_at=unixepoch()
                ''', (user_id, payload, payload))
            conn.commit()
            conn.close()

    @staticmethod
    def _fts_query(query: str) -> str:
        """把自然语言收成 FTS5 可接受的词串"""
        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}', query or '')
        # 去重保序
        seen = set()
        cleaned = []
        for t in tokens:
            if t not in seen:
                seen.add(t)
                cleaned.append(t)
        if not cleaned:
            return ''
        return ' OR '.join(cleaned[:8])

    def search_fts(self, user_id: str, query: str, limit: int = 5) -> List[Dict]:
        q = self._fts_query(query)
        if not q:
            return []
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT m.id, m.content, m.chunk_type, m.importance, m.created_at
                FROM memory_chunks m
                JOIN memory_fts f ON m.id = f.rowid
                WHERE m.user_id = ? AND memory_fts MATCH ?
                ORDER BY bm25(memory_fts) LIMIT ?
            ''', (user_id, q, limit))
            rows = cursor.fetchall()
        except Exception:
            rows = []
        conn.close()
        return [
            {'id': r[0], 'content': r[1], 'type': r[2], 'importance': r[3], 'created_at': r[4]}
            for r in rows
        ]

    @staticmethod
    def _cosine(a, b) -> float:
        """余弦相似度；维度不一致或零向量返回 0。"""
        if not a or not b or len(a) != len(b):
            return 0.0
        dot = na = nb = 0.0
        for x, y in zip(a, b):
            dot += x * y
            na += x * x
            nb += y * y
        if na == 0.0 or nb == 0.0:
            return 0.0
        return dot / ((na * nb) ** 0.5)

    def search_vector(self, scope_id: str, query_vec: List[float], limit: int = 3) -> List[str]:
        """语义检索：对库内该 scope 全部已嵌入记忆做余弦相似度排序。

        数据量（几千条内）直接全量算，无需额外索引依赖。
        """
        if not query_vec:
            return []
        conn = self._get_conn()
        try:
            rows = conn.execute(
                'SELECT content, embedding FROM memory_chunks '
                'WHERE user_id = ? AND embedding IS NOT NULL',
                (str(scope_id),),
            ).fetchall()
        except Exception:
            rows = []
        conn.close()
        scored = []
        for content, emb_bytes in rows:
            if not emb_bytes:
                continue
            try:
                emb = json.loads(emb_bytes)
            except Exception:
                continue
            sim = self._cosine(emb, query_vec)
            if sim > 0:
                scored.append((sim, content))
        scored.sort(key=lambda x: -x[0])
        return [c for _, c in scored[: max(1, int(limit))]]

    def search_relevant(
        self,
        scope_id: str,
        query: str,
        limit: int = 5,
        query_vec: Optional[List[float]] = None,
    ) -> List[str]:
        """相关记忆：向量语义 + FTS 关键词 混合，不足用近 72h summary/fact 补。

        传 query_vec 时优先向量召回（语义相近但字面不同也能命中）；
        不传则保持原来的纯 FTS 行为。
        """
        contents: List[str] = []
        if query_vec:
            contents.extend(self.search_vector(scope_id, query_vec, limit=limit))
        fts = self.search_fts(scope_id, query, limit=limit)
        for item in fts:
            if item not in contents:
                contents.append(item)
        if len(contents) >= limit:
            return contents[:limit]
        recent = self.get_recent_chunks(scope_id, hours=72, limit=limit, chunk_types=('summary', 'fact'))
        for item in recent:
            if item not in contents:
                contents.append(item)
            if len(contents) >= limit:
                break
        return contents[:limit]

    def search_graph(self, user_id: str, entity: str, limit: int = 10) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT entity, relation, target, weight, evidence
            FROM memory_graph
            WHERE user_id = ? AND (entity LIKE ? OR target LIKE ?)
            ORDER BY weight DESC LIMIT ?
        ''', (user_id, f'%{entity}%', f'%{entity}%', limit))
        rows = cursor.fetchall()
        conn.close()
        return [
            {'entity': r[0], 'relation': r[1], 'target': r[2], 'weight': r[3], 'evidence': r[4]}
            for r in rows
        ]

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            'SELECT traits, preferences, speaking_style, recent_info, nickname '
            'FROM user_profile WHERE user_id = ?',
            (user_id,),
        )
        row = cursor.fetchone()
        conn.close()
        if not row:
            return {}

        def _loads(v):
            if not v:
                return {}
            try:
                return json.loads(v)
            except Exception:
                return {}

        return {
            'traits': _loads(row[0]),
            'preferences': _loads(row[1]),
            'speaking_style': _loads(row[2]),
            'recent_info': _loads(row[3]),
            'nickname': row[4],
        }

    def get_recent_chunks(
        self,
        user_id: str,
        hours: int = 24,
        limit: int = 20,
        chunk_types: tuple = ('chat', 'summary', 'fact'),
    ) -> List[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cutoff = time.time() - hours * 3600
        placeholders = ','.join('?' * len(chunk_types))
        cursor.execute(f'''
            SELECT content FROM memory_chunks
            WHERE user_id = ? AND created_at > ? AND chunk_type IN ({placeholders})
            ORDER BY importance DESC, created_at DESC LIMIT ?
        ''', (user_id, cutoff, *chunk_types, limit))
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rows

    def learn_pattern(self, group_id: str, user_id: str, pattern: str):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO group_style (group_id, user_id, pattern)
                    VALUES (?, ?, ?)
                    ON CONFLICT(group_id, pattern) DO UPDATE SET
                        count=count+1, user_id=COALESCE(excluded.user_id, user_id),
                        updated_at=unixepoch()
                ''', (group_id, user_id, pattern))
            except Exception:
                pass
            conn.commit()
            conn.close()

    def get_top_patterns(self, group_id: str, limit: int = 10) -> List[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT pattern FROM group_style
            WHERE group_id = ? ORDER BY count DESC LIMIT ?
        ''', (group_id, limit))
        rows = [r[0] for r in cursor.fetchall()]
        conn.close()
        return rows

    def learn_jargon(self, group_id: str, term: str, meaning: Optional[str] = None, level: int = 0):
        with self._lock:
            conn = self._get_conn()
            cursor = conn.cursor()
            try:
                cursor.execute('''
                    INSERT INTO jargon_table (group_id, term, meaning, level)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(group_id, term) DO UPDATE SET
                        count=count+1,
                        meaning=COALESCE(excluded.meaning, meaning),
                        level=MAX(level, excluded.level),
                        updated_at=unixepoch()
                ''', (group_id, term, meaning, level))
            except Exception:
                pass
            conn.commit()
            conn.close()

    def get_jargons(self, group_id: str, limit: int = 20) -> List[Dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT term, meaning, level, count FROM jargon_table
            WHERE group_id = ? AND level >= 1 ORDER BY count DESC LIMIT ?
        ''', (group_id, max(1, int(limit))))
        rows = [
            {'term': r[0], 'meaning': r[1], 'level': r[2], 'count': r[3]}
            for r in cursor.fetchall()
        ]
        conn.close()
        return rows

    def search_jargons(self, group_id: str, query: str, limit: int = 5) -> List[Dict]:
        """按当前消息检索相关黑话：词出现在消息里 / 消息词命中释义。

        只返回相关命中，不回退到「热度 TopN」，避免无关黑话污染提示词。
        """
        all_rows = self.get_jargons(group_id, limit=200)
        if not all_rows:
            return []

        q = (query or '').strip()
        if not q:
            return []

        tokens = re.findall(r'[\u4e00-\u9fff]{2,}|[A-Za-z0-9_]{2,}', q)
        scored: List[tuple] = []
        for item in all_rows:
            term = (item.get('term') or '').strip()
            meaning = (item.get('meaning') or '').strip()
            if not term:
                continue
            score = 0.0
            if term in q:
                # 整词命中权重最高；更长短语优先
                score += 20.0 + min(len(term), 12)
            for tok in tokens:
                if tok == term:
                    score += 12.0
                elif tok in term or term in tok:
                    score += 6.0
                elif meaning and tok in meaning:
                    score += 2.5
            if score <= 0:
                continue
            score += min(float(item.get('count') or 1), 8) * 0.15
            scored.append((score, item))

        scored.sort(key=lambda x: (-x[0], -(x[1].get('count') or 0)))
        return [it for _, it in scored[: max(1, int(limit))]]
