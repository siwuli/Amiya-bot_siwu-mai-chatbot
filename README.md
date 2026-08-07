# 兔兔 - AmiyaBot 群聊智能体插件

基于明日方舟 **阿米娅** 人设的 **AmiyaBot** 群聊智能体插件。被前缀 / `@` / 触发词召唤后像真人一样短聊接话，并且**不抢**其他指令插件；可选开启主动模式，在合适的时机像老群友一样自己接话。

> 插件 id：`siwu-mai-chatbot`　当前版本：`1.9.0`

## 特性

- **召唤聊天**：`兔兔…` / `@兔兔…` / 自定义触发词 → 由 LLM 生成符合人设的短回复
- **评分接话（可选）**：开启 `mai_proactive` 后，未召唤消息先由轻量模型打标签、程序计分，按概率决定是否接话
- **持续学习**：观察每条群友消息后异步学习用户画像、群风格、群黑话（不依赖机器人是否回复）
- **语义记忆（v1.9，可选）**：接入 OpenAI 兼容 embedding 服务，记忆检索升级为「向量 + 关键词」混合召回
- **不抢插件**：其他插件认领的消息自动让出

## 安装

1. 将打包好的 `siwu-mai-chatbot-1.0.zip` 放入 AmiyaBot 根目录的 `plugins` 目录；或把源码解压到 `pluginsServer/siwu-mai-chatbot-1_0`
2. 在控制台（或 `config_default.yaml`）填写 `mai_api_key`（DeepSeek 等 OpenAI 兼容服务的 Key）
3. 重启 AmiyaBot，或在插件管理里重载插件

源码方式重新打包（在插件目录内执行）：

```bash
python build.py
```

产物为 AmiyaBot 根目录下 `plugins/siwu-mai-chatbot-1.0.zip`。

## 接话时机（v1.7）

开启 `mai_proactive` 后，未召唤的消息会先走**轻量标签模型**：

1. 模型只输出布尔标签：`mentioned` `reply_to_bot` `question` `open_topic` `group_chat` `already_answered` `interrupt_risk` `command` `requires_bot`
2. 程序计分：`+70/+50/+40/+20/+15/+10/-25/-35`；`command=true` → 0；再扣 `recent_reply_penalty` / `cooldown_penalty`
3. 夹到 0~100 后按分数概率抽样，命中再走现有生成逻辑

`cooldown_penalty` 的衰减窗口对齐 `mai_min_reactive_interval`（软扣分，不硬拦截）。

## 学习与检索（v1.8）

每条群友消息在**观察后**就会异步学习（不依赖机器人是否回复）：

| 能力 | 触发 | 说明 |
|---|---|---|
| 用户画像 | 样本够且过冷却 | 合并内存样本 + 库内近期 chat；成功才进入冷却 |
| 群风格 | 群消息够且过冷却 | 写入 `group_style` |
| 黑话挖掘 | 群消息够且过冷却 | 写入 `jargon_table` |

生成回复时，黑话**按当前消息关键词检索**注入（最多 5 条），不再固定灌 TopN：

- 黑话原词出现在当前消息 → 高分
- 消息切词命中原词/释义 → 加分
- 得分 ≤ 0 不注入

成功更新画像时日志会出现：`[Mai] 画像已更新 user=…`

## 语义记忆（v1.9，可选）

默认关闭（`mai_embedding_enabled: false`），不配置不影响任何使用。

开启后：记忆写入时同步生成向量，检索时按当前消息做「向量相似度 + FTS 关键词」**混合召回**——语义相近但字面不同的记忆也能命中（例如聊「抽卡」能想起之前聊过的「原神」）。

> 注意：DeepSeek 官方接口**没有** `/embeddings`，需要另配一个 OpenAI 兼容的 embedding 服务：

| 配置项 | 示例 |
|---|---|
| `mai_embedding_enabled` | `true` |
| `mai_embedding_base_url` | `https://api.siliconflow.cn/v1`（或 OpenAI `/v1`） |
| `mai_embedding_api_key` | 对应服务 Key（留空复用 `mai_api_key`） |
| `mai_embedding_model` | `BAAI/bge-m3`（留空默认 `text-embedding-3-small`） |

开启后会自动**后台回填**历史记忆的向量。未配全 / 服务不可用 / 配置错误时会**自动停用并提示**，聊天不受影响，只退回关键词检索。

## 生成时注入的上下文

| 数据 | 上限 | 取法 |
|---|---|---|
| 人设 system | 整段 | `mai_system_prompt` |
| 用户画像 | traits≤5 / prefs≤5 / style≤4 / 事实≤6 | 当前发言人 `user_profile` |
| 群说法 | 4 | `group_style` 按 count |
| 相关黑话 | 0～5 | 相对当前消息关键词检索 |
| 相关记忆 | ≤4 | 向量语义（群≤3 + 用户≤2）+ FTS 关键词混合，不足用近 72h summary/fact 补 |
| 群聊历史 | ≤`mai_max_context`（默认 16） | 短期历史（内存 + `chat_tail` 持久化） |

## 提示词可配置

以下项都在控制台 / `config_default.yaml`，改完会热刷新：

| 配置项 | 作用 | 占位符 |
|---|---|---|
| `mai_bot_name` | 自称 | - |
| `mai_system_prompt` | 人设 system（最影响说话风格） | - |
| `mai_user_prompt_template` | 生成时的 user 模板 | `{history}` `{speaker_name}` `{focus_text}` `{bot_name}` |
| `mai_banned_phrases` | 回复后处理删除的套话 | - |
| `mai_persona_prompt` | 画像提取 system | - |
| `mai_persona_user_template` | 画像提取 user | `{conversation}` |
| `mai_style_expression_prompt` | 群风格学习 | `{messages}`（JSON 花括号写成 `{{` `}}`） |
| `mai_style_jargon_prompt` | 黑话挖掘 | `{messages}` |
| `mai_timing_prompt` | 主动发言时机（评分关闭时） | `{context}` |
| `mai_judge_prompt` | 接话特征标签 JSON | `{bot_name}` `{history}` `{speaker_name}` `{focus_text}` |

## 记忆库表说明

数据库默认路径：`data/mai_memory.db`

| 表 | 存什么 |
|---|---|
| `memory_chunks` | 长期记忆片段（群聊摘要、用户事实等），按 `user_id`（群号或用户号）隔离；`embedding` 列存向量（可选） |
| `memory_fts` | `memory_chunks` 的 FTS5 全文索引（虚拟表，勿手改） |
| `memory_graph` | 实体关系三元组（如「职业 is 学生」），来自画像抽取 |
| `user_profile` | 用户画像：性格/偏好/说话风格/已知事实/昵称 |
| `group_style` | 群常用说法、口头禅（风格学习结果） |
| `jargon_table` | 群黑话/新词及释义 |
| `chat_tail` | 短期群聊尾巴（含机器人自己的发言），供评分/生成，重启可恢复 |

另有触发器 `memory_ai_insert` / `memory_ai_delete`：同步维护 FTS 索引。

## 怎么触发

| 用户怎么说 | 其他插件 | 兔兔 |
|---|---|---|
| `兔兔你好` / `阿米娅在吗` | 没人回 | ✅ |
| `兔兔签到` | 签到插件回 | ❌ 让出 |
| `@兔兔 …` | 没人回 | ✅ |
| 无前缀闲聊（`mai_proactive`+评分命中） | 没人回 | ✅ |
| 无前缀闲聊（评分 0 / 未命中） | - | ❌ |

## 调试日志（v1.4）

控制台打开 `mai_debug_log`（默认开）后，每条相关消息会打印类似：

- `观察 … addressed=… proactive=…` — 是否召唤、是否走主动
- `接话评分 score=… hit=… detail=…` — 标签、程序算分、概率命中/未命中
- `召唤排队兜底` / `排队兜底回复` — 进入等待其他插件
- `让出：其他插件已认领` — 不抢指令
- `开始生成回复` / `发送回复` — 真正调用主模型并发送
- `画像已更新` — 人格字段写入成功
- `向量记忆已自动停用` — embedding 配置异常被熔断（聊天不受影响）

## 版本记录

每次发版在表格最上方追加一行。

| 版本 | 更新内容 |
|---|---|
| `1.9.0` | 记忆库新增可选语义检索：OpenAI 兼容 `/embeddings` 生成向量，向量 + FTS 混合召回；自动后台回填历史记忆；未配全/服务异常自动熔断停用，不影响聊天 |
| `1.8.3` | 自身发言写入 `chat_tail` 并持久化；评分/生成前恢复历史；识别机器人 QQ 回传；改配置重建核心时保留历史与冷却；发送成功后再更新 `last_reply`，`reply_to_bot` 判断更准 |
| `1.8.2` | 剥离思考过程（`<think>` / `【思考】` 等）；不再把 `reasoning_content` 当可见回复 |
| `1.8.1` | 空结果自动重试并强制输出；仍空则短句兜底；提示词禁止空回复；LLM 并发限制与空结果 `meta` 日志 |
| `1.8` | 观察后异步学习画像/风格/黑话；画像样本合并库内 chat；黑话改为关键词检索注入；同步配置与文档 |
| `1.7` | 轻量标签接话评分 + 概率抽样；独立评分模型 `mai_judge_*`；`mai_debug_log` |
| `1.4` | 调试日志：观察 / 召唤 / 评分 / 兜底 / 生成 |
| 更早 | 前缀/@/触发词召唤；兜底不抢插件；DeepSeek 对话；画像/风格/黑话与记忆库 |

## 致谢

本插件在设计与实现上参考了 [MaiBot（麦麦）](https://github.com/Mai-with-u/MaiBot)——一个基于大语言模型、专注于群组聊天的可交互智能体。人设化对话、群聊风格与黑话学习、用户画像与长期记忆等设计思路均受其启发。感谢 MaiBot 项目及其社区的开源贡献（MaiBot 基于 GPL-3.0 许可证开源）。

## 项目地址

<https://github.com/siwuli/Amiya-bot_siwu-mai-chatbot>
