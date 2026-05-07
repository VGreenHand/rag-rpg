# RAG-RPG 记忆引擎

基于 RAG（检索增强生成）的交互式叙事记忆引擎，为 SillyTavern 角色扮演平台提供持久化对话记忆与剧情自动约束能力。

## 项目概述

RAG-RPG 是一个**对话记忆与剧情约束系统**，专门为 SillyTavern 平台的文字角色扮演游戏设计。核心思路是：**让大模型在生成回复时，能够"记住"历史对话中的关键信息，并受到游戏世界观设定和技能体系的自然约束，从而产生一致、连贯、沉浸的叙事体验。**

---

## 核心特性

### 1. 对话记忆自动管道

| 阶段 | 功能 | 说明 |
|------|------|------|
| 捕获 | MESSAGE_RECEIVED 事件监听 | SillyTavern JS 扩展自动捕获每轮对话 |
| 清洗 | HTML/标记语言移除 | 去除 `<i>`、`**` 等格式标记，保留纯文本 |
| 提取 | 关键术语自动识别 | 从对话中识别技能名称、机制关键词等 |
| 持久化 | TXT 文件 + ChromaDB | 每日对话写入 TXT，同时向量化存入向量库 |
| 批量 | `[TYPE]` 标记文件导入 | 支持 ingest_new.py 格式的批量导入 |

### 2. 上下文感知语义查询

- **多策略查询生成**：从最近 N 轮对话中提取用户消息、AI 回复、摘要、动作描述等多个查询变体
- **跨集合融合检索**：同时在角色技能库、对话记忆库、剧情库中检索，去重后按相关度排序
- **降级保护**：当某个集合查询超时或失败时，自动跳过该集合，不影响其他结果
- **格式化输出**：将检索结果整理为结构化的文本，直接可供大模型理解

### 3. 剧情约束规则引擎

- **自然语言约束**：将检索到的记忆转化为自然流畅的剧情引导指令，而非生硬的规则列表
- **类型化模板**：支持 技能约束 / 机制约束 / 世界观约束 / 剧情线索 / 记忆回调 五类模板
- **冷却控制**：同一条目在短时间内不会重复出现，避免 AI 反复提及相同内容
- **自适应权重**：根据约束类型被 AI 实际采用的频率动态调整优先级

### 4. 断点续执行与容错

- **超时保护**：所有 ChromaDB 操作、embedding 编码、文件 I/O 均有独立超时（5-15s）
- **进度持久化**：批量导入时每处理一批数据就记录 checkpoint
- **心跳监控**：后台线程每 5 秒写入心跳文件，可检测是否存活
- **安全恢复**：进程崩溃重启后，可从上一个 checkpoint 继续批量导入，无需从头开始

### 5. 灵活的服务端 API

| 端点 | 功能 |
|------|------|
| `GET /api/health` | 全面的健康检查（无需认证） |
| `GET /api/status` | 引擎状态与向量库统计 |
| `POST /api/dialogue/ingest` | 接收单轮对话，全流程处理 |
| `POST /api/dialogue/query` | 上下文搜索 + 约束生成 |
| `POST /api/batch/ingest` | 批量导入 TXT 标记文件 |
| `GET/POST /api/checkpoint/*` | 断点状态查询、续点执行、清除 |
| `POST /api/skill/update` | 更新技能条目 |
| `POST /api/feedback` | 约束采用反馈，调整权重 |

---

## 系统架构

```
SillyTavern (浏览器/客户端)
  │
  ├── MESSAGE_RECEIVED ──→ POST /api/dialogue/ingest
  ├── GENERATION_BEFORE  ──→ POST /api/dialogue/query
  │                            ↓
  │                      setExtensionPrompt() ← 注入约束/记忆
  │
  ▼
┌──────────────────────────────────────────────────────────────┐
│                  Python FastAPI 服务端                        │
│                                                              │
│  pipeline.py        清洗 → 提取 → TXT写入 → 向量入库        │
│  query_engine.py    上下文分析 → 多查询生成 → 融合检索      │
│  constraint_engine.py  检索结果 → 类型化模板 → 冷却控制     │
│  checkpoint_manager.py 超时保护 → 进度持久化 → 续点执行     │
│  server.py          统一 API 入口 + 健康检查                 │
│  config.py          全局配置参数                              │
└──────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────────────────────────────────────────────────────┐
│                      ChromaDB 向量库                          │
│  character_skills    角色技能/机制条目（10-50 条）            │
│  my_rag_memory       自定义记忆/剧情条目                      │
│  dialogue_memory     实时对话历史记录（持续增长）             │
└──────────────────────────────────────────────────────────────┘
```

---

## 环境要求

| 组件 | 最低要求 | 推荐 |
|------|----------|------|
| Python | 3.10+ | 3.12+ |
| 操作系统 | Windows 10+ / macOS / Linux | Windows 11 |
| 内存 | 4 GB | 8 GB |
| 磁盘空间 | 1 GB | 2 GB |
| 网络 | embedding 模型首次下载需联网 | 宽带连接 |
| SillyTavern | 无版本依赖 | 最新稳定版 |

---

## 安装步骤

### 第一步：克隆项目

```powershell
git clone https://github.com/VGreenHand/rag-rpg.git
cd rag-rpg
```

### 第二步：创建虚拟环境

```powershell
python -m venv venv
.\venv\Scripts\activate     # Windows
# source venv/bin/activate  # Linux/macOS
```

### 第三步：安装依赖

```powershell
pip install -r requirements.txt
```

依赖清单：
- `chromadb>=0.4.0` — 向量数据库
- `sentence-transformers>=2.2.0` — 文本向量化模型
- `fastapi>=0.104.0` — Web 服务框架
- `uvicorn[standard]>=0.24.0` — ASGI 服务器
- `pydantic>=2.0.0` — 数据验证

### 第四步：初始化向量库

```powershell
python scripts/ingest_initial.py
```

这将从 `data/CharacterInfo/Characterdesign.json` 中读取角色技能和世界观条目，生成向量并存入 ChromaDB。

首次运行会自动下载 embedding 模型 `shibing624/text2vec-base-chinese`（约 400 MB）。

### 第五步：启动服务端

```powershell
python server.py
```

服务将在 `http://127.0.0.1:8765` 启动。

### 第六步：安装 SillyTavern 扩展

将 `st_extension/` 文件夹复制到 SillyTavern 的扩展目录：

| 安装位置 | 路径 |
|----------|------|
| 用户扩展 | `SillyTavern/data/default-user/extensions/RAG-RPG/` |
| 第三方扩展 | `SillyTavern/public/scripts/extensions/third-party/RAG-RPG/` |

重启 SillyTavern，在扩展面板中启用 **RAG-RPG 记忆引擎**。

### 第七步：验证安装

```powershell
python -c "import sys; sys.path.insert(0,'.'); from pipeline import get_pipeline; p=get_pipeline(); print('Pipeline:', p.get_stats())"
```

预期输出类似：
```
Pipeline: {'dialogue_memory': 0, 'character_skills': 10, 'my_rag_memory': 0, 'dialogue_txt_files': 0}
```

---

## 基本操作流程

### 日常使用流程

```
1. 启动 Python 后端
   └─ python server.py

2. 启动 SillyTavern，加载角色卡（如"天际省叙事者"）

3. 开始角色扮演对话
   └─ 扩展自动捕获每轮对话并存入记忆库
   └─ AI 生成回复前自动查询相关记忆并注入约束

4. AI 回复受世界观和技能体系的自然引导
```

### 完整数据流

```
用户输入 "我想使用光剑精通攻击敌人"
  │
  ▼
SillyTavern 扩展捕获 → POST /api/dialogue/ingest
  ├── 清洗: 移除格式标记
  ├── 提取: [光剑精通]
  ├── TXT写入: dialogues/dialogue_2026-05-06.txt
  ├── 向量入库: dialogue_memory 集合
  └── batch标记: [DIALOGUE] 对话提及: skill_lightsaber_mastery | ...

AI 即将生成回复
  │
  ▼
SillyTavern 扩展触发 → POST /api/dialogue/query
  ├── 分析上下文: "光剑精通" + "攻击"
  ├── 检索 character_skills → 光剑精通熟练度12/100
  ├── 检索 dialogue_memory → 历史战斗记录
  ├── 约束引擎 → "角色拥有技能'光剑精通'，熟练度12/100，请在回复中展现..."
  └── setExtensionPrompt → AI 回复时自然融入技能描述
```

---

## 常用功能模块

### 管理工具脚本

所有脚本位于 `scripts/` 目录下：

| 脚本 | 用途 | 用法示例 |
|------|------|----------|
| `check_env.py` | 检查 Python 环境与依赖 | `python scripts/check_env.py` |
| `ingest_initial.py` | 从 JSON 初始化向量库 | `python scripts/ingest_initial.py` |
| `ingest_new.py` | 从 `[TYPE]` 标记 TXT 追加导入 | 编辑 `new_batch.txt` 后执行 |
| `quick_search.py` | 交互式语义检索 | `python scripts/quick_search.py` |
| `update_skill.py` | 更新指定技能条目 | 修改 entry_key 和内容后执行 |
| `check_keys.py` | 验证 JSON 与向量库一致性 | `python scripts/check_keys.py` |
| `check_metadata.py` | 查看所有条目的 entry_key | `python scripts/check_metadata.py` |

### 批量导入格式

创建 `new_batch.txt`，每行使用 `[TYPE]` 标记：

```text
[SKILL] 技能：冰霜吐息。龙裔释放冰霜龙吼，冻结前方敌人。
[PLOT] 剧情：玩家在寒落神殿发现了龙石碎片，触发主线任务。
[SETTING] 设定：冬堡学院以幻术系魔法闻名，院长是萨沃斯·阿冉。
```

然后执行：

```powershell
python scripts/ingest_new.py
```

### 断点续执行

当批量导入因故中断后（如进程崩溃），恢复执行：

```powershell
curl -X POST http://127.0.0.1:8765/api/checkpoint/resume -H "X-API-Key: rag-rpg-local"
```

查询断点状态：

```powershell
curl http://127.0.0.1:8765/api/checkpoint/status -H "X-API-Key: rag-rpg-local"
```

### 技能更新

在 SillyTavern 中完成战斗后，更新技能熟练度：

```powershell
curl -X POST http://127.0.0.1:8765/api/skill/update ^
  -H "X-API-Key: rag-rpg-local" ^
  -H "Content-Type: application/json" ^
  -d "{\"entry_key\": \"skill_lightsaber_mastery\", \"new_content\": \"技能：光剑精通。当前熟练度 25/100。[type:skill]\"}"
```

### 扩展面板设置

在 SillyTavern 扩展面板中可配置：

| 设置项 | 默认值 | 说明 |
|--------|--------|------|
| API URL | `http://127.0.0.1:8765` | Python 后端地址 |
| API Key | `rag-rpg-local` | 认证密钥 |
| 自动捕获 | 开启 | 自动捕获每轮对话 |
| 自动查询 | 开启 | AI 生成前自动检索记忆 |
| 注入约束 | 开启 | 将检索结果作为剧情约束注入 |
| 查询轮次 | 6 | 检索时考虑的最近对话轮数 |
| 调试模式 | 关闭 | 输出详细日志到浏览器控制台 |

---

## 数据文件结构

```
rag-rpg/
├── data/                              ← 游戏数据文件
│   ├── CharacterInfo/
│   │   ├── Characterdesign.json       ← 角色技能/机制（SillyTavern 格式）
│   │   └── 角色设定.json              ← 角色设定（冗余备份）
│   ├── WorldInfo/
│   │   └── main_Skyrim Lorebook...json ← 世界观设定
│   └── 分支记录.txt                   ← 游戏剧情分支记录
│
├── scripts/                           ← 管理工具脚本
│   ├── ingest_initial.py             ← 从 JSON 初始化向量库
│   ├── ingest_new.py                 ← 从 TXT 追加导入
│   ├── quick_search.py               ← 交互式语义检索
│   ├── update_skill.py               ← 更新技能条目
│   ├── check_env.py                  ← 环境检查
│   ├── check_keys.py                 ← 一致性验证
│   └── check_metadata.py             ← 元数据查看
│
├── tests/                             ← 测试套件
│   ├── test_suite.py                 ← 148 条全方位测试
│   └── test_checkpoint_resume.py     ← 75 条断点续执行测试
│
├── st_extension/                      ← SillyTavern 扩展
│   ├── manifest.json                 ← 扩展清单
│   └── index.js                      ← 扩展核心逻辑
│
├── config.py                          ← 全局配置（端口/路径/参数）
├── pipeline.py                        ← 对话处理管道
├── query_engine.py                    ← 语义查询引擎
├── constraint_engine.py               ← 剧情约束引擎
├── checkpoint_manager.py              ← 断点续执行管理器
├── server.py                          ← FastAPI 服务端
├── requirements.txt                   ← Python 依赖
├── .gitignore                         ← Git 忽略规则
└── README.md                          ← 本文档
```

---

## 配置文件说明

`config.py` 是项目的唯一配置入口，可修改以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | `127.0.0.1` | 服务监听地址 |
| `API_PORT` | `8765` | 服务端口 |
| `API_KEY` | `rag-rpg-local` | API 认证密钥（可通过环境变量 `RAG_RPG_API_KEY` 覆盖） |
| `MODEL_NAME` | `shibing624/text2vec-base-chinese` | embedding 模型名称 |
| `MAX_CONTEXT_TURNS` | `5` | 查询时考虑的最近对话轮数 |
| `TOP_K_RESULTS` | `3` | 每次检索返回的最大结果数 |
| `MIN_RELEVANCE` | `0.25` | 最低相关度阈值 |
| `MAX_CONSTRAINT_CHARS` | `800` | 约束文本最大长度 |

---

## 运行测试

项目包含两套独立的测试套件：

### 全面功能测试

```powershell
python tests/test_suite.py
```

148 条测试，覆盖：环境检查、配置验证、管道清洗/入库/提取、查询引擎多策略检索、约束引擎模板生成/冷却/反馈权重、FastAPI 全端点测试、兼容性测试（英文/中英混合/超长文本/JSON序列化）、性能基准测试。

### 断点续执行专项测试

```powershell
python tests/test_checkpoint_resume.py
```

75 条测试，覆盖：checkpoint 持久化/加载、续点执行全流程、超时保护、心跳监控、SafeTimer 安全执行、管道安全操作、查询引擎降级模式、断点 API 端点测试、并发执行压力测试。

---

## 错误排查

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|----------|----------|
| 服务启动报错 `chromadb` 未安装 | 依赖未安装 | `pip install -r requirements.txt` |
| SillyTavern 扩展不生效 | 扩展目录错误 | 检查扩展路径是否正确 |
| `ingest_initial.py` 找不到 JSON | 路径变化 | 确认目标路径: `data/CharacterInfo/Characterdesign.json` |
| 查询返回空结果 | 向量库未初始化 | 先执行 `python scripts/ingest_initial.py` |
| 出现 `invalid api key` | API Key 不匹配 | 确认 SillyTavern 扩展设置中的 API Key 与 config.py 一致 |
| embedding 模型下载失败 | 网络问题 | 确保网络畅通，可尝试设置 `HF_TOKEN` |
| 测试卡死在 `init_task` | 旧版 checkpoint 残留 | `python -c "import shutil; shutil.rmtree('.checkpoints', ignore_errors=True)"` |
| `process_batch_txt` 部分失败 | ChromaDB 写入超时 | 检查磁盘负载，可减小 `batch_size` |

### 日志查看

服务端日志输出到控制台，格式如下：

```
2026-05-06 14:30:22,456 [INFO] 已处理 Turn#12 | user | 关键术语: ['光剑精通', '里·鬼剑术']
2026-05-06 14:30:23,012 [WARNING] 集合查询超时: my_rag_memory，降级跳过
```

SillyTavern 扩展日志在浏览器控制台中查看（需开启调试模式）。

---

## 预期用途

### 核心场景

- **文字角色扮演游戏**：为 SillyTavern 中的 RPG 角色扮演提供长程记忆和剧情一致性
- **互动小说创作**：管理复杂分支剧情，确保 AI 始终遵循已建立的世界观
- **AI 叙事助手**：帮助 AI 在长篇叙事中保持角色性格、技能体系和剧情连贯性

### 具体应用示例

| 场景 | 效果 |
|------|------|
| 玩家学会新技能 | 系统将技能纳入向量库，此后 AI 会在适当时机引用该技能 |
| 触发关键剧情 | AI 自动检索相关线索，在后续对话中自然推进 |
| 战斗升级 | 技能熟练度动态更新，AI 对战斗效果的描述随之变化 |
| 跨会话记忆 | 关闭游戏后重新开启，AI 仍记得之前的冒险经历 |
| 多角色交互 | 不同角色的设定和关系在向量库中独立管理 |

---

## 适用人群

| 人群 | 价值点 |
|------|--------|
| **SillyTavern 玩家** | 获得持久记忆、一致剧情、沉浸式 RPG 体验 |
| **AI 角色扮演爱好者** | 告别 AI "失忆" 问题，享受连续发展的故事线 |
| **互动小说作者** | 利用剧情约束机制自动管理复杂分支叙事 |
| **游戏模组开发者** | 可定制角色技能体系、世界观条目和剧情事件 |
| **NLP/LLM 爱好者** | 研究 RAG 在交互式叙事中的应用实践 |
| **Python/FastAPI 开发者** | 参考本项目的架构设计和模块划分 |

---

## 扩展方向

### 短期（可行）

- **多角色卡支持**：自动识别当前角色卡，加载对应的技能和世界观数据
- **对话摘要生成**：对长对话进行自动摘要，节省 token 并保留核心信息
- **可视化管理界面**：基于 FastAPI 提供 Web 管理面板，可视化查看/编辑记忆库
- **多 embedding 模型切换**：支持 OpenAI/Cohere 等商业 embedding API

### 中期（需一定开发）

- **剧情分支图**：追踪关键决策点，可视化展示剧情分支结构
- **自动技能熟练度**：AI 根据对话中技能使用频率自动更新熟练度数值
- **多语言支持**：扩展英文及日文 embedding 模型，服务非中文用户
- **导出/导入格式**：支持 JSON/YAML 格式的剧情包导入导出

### 长期（愿景）

- **独立叙事引擎**：脱离 SillyTavern，作为独立的 AI 叙事平台运行
- **多模态记忆**：支持图片、音频等非文本信息的记忆和检索
- **群体记忆系统**：多用户共享世界状态，实现 MMORPG 风格的持续世界
- **自适应剧情生成**：根据玩家的历史行为动态调整主线/支线叙事权重

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12+ | 后端核心语言 |
| FastAPI + Uvicorn | REST API 服务 |
| ChromaDB | 向量数据库 |
| sentence-transformers | 文本向量化 |
| shibing624/text2vec-base-chinese | 中文 embedding 模型 |
| JavaScript (SillyTavern 扩展) | 前端事件监听与交互 |

---

## 许可证

本项目为个人学习和实践项目，代码可用于学习和参考。
