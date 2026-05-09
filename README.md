# RAG-RPG 记忆引擎

基于 RAG（检索增强生成）的交互式叙事记忆引擎，为 SillyTavern 角色扮演平台提供持久化对话记忆与剧情自动约束能力。

给你的 AI 角色装一个不会失忆的外接大脑——自动记住每轮对话，在恰当的时机注入相关的剧情约束，让长篇角色扮演保持连贯、沉浸、不跳戏。预设数据完全可选，没有角色设定文件也能正常游玩。

---

## 快速开始

```cmd
:: 1. 克隆项目
git clone https://github.com/VGreenHand/rag-rpg.git
cd rag-rpg

:: 2. 创建环境
conda create -n rag-rpg python=3.12 -y
conda activate rag-rpg

:: 3. 安装依赖
pip install -r requirements.txt

:: 4. （可选）如果有自己的角色 JSON，放入 data/CharacterInfo/ 后初始化
python scripts\ingest_initial.py

:: 5. 启动服务
python server.py
```

然后将 `st_extension/` 复制到 SillyTavern 的扩展目录，在扩展面板中启用即可。

---

## 工作原理

### 数据流

```
用户输入 → SillyTavern 扩展捕获
             ↓
          POST /api/dialogue/ingest
             ├── 清洗格式标记（HTML、Markdown）
             ├── 提取关键术语（技能名、机制词）
             ├── 写入对话日志 TXT
             └── 向量化存入 ChromaDB

AI 即将回复 → SillyTavern 扩展触发
             ↓
          POST /api/dialogue/query
             ├── 分析最近 N 轮对话，生成多策略查询
             ├── 跨集合检索（技能库、记忆库、对话历史）
             ├── 约束引擎生成自然语言引导
             └── setExtensionPrompt → AI 回复时自然融入
```

### 核心能力

| 能力 | 说明 |
|------|------|
| 对话记忆 | 自动捕获每轮对话，清洗、提取、持久化到向量库 |
| 语义检索 | 分析对话上下文，多策略生成查询，跨集合融合检索 |
| 剧情约束 | 将检索到的记忆转化为自然语言引导，非生硬规则列表 |
| 断点续执行 | 批量导入带超时保护、进度持久化、崩溃恢复 |
| 零数据启动 | 对话记忆功能开箱即用，无需任何预设文件 |

---

## 数据文件说明

`data/` 目录存放的是**不同游戏的静态预设数据**（角色技能、世界观信息），由用户自行管理。

- 没有这些文件：对话记忆功能正常工作，纯记忆模式即可游玩
- 有这些文件：角色技能和世界观会被 AI 检索引用，体验更沉浸
- 多设备：把自己的 JSON 文件复制到新设备，`ingest_initial.py` 重建即可

```
data/
├── CharacterInfo/          ← 角色技能/机制 JSON（SillyTavern World Info 格式）
│   ├── Characterdesign.json
│   └── 角色设定.json
├── WorldInfo/              ← 世界观设定 JSON
│   └── 游戏名 Lorebook.json
└── 分支记录.txt            ← 剧情分支记录（手动维护）
```

---

## 安装

### 环境要求

| 组件 | 要求 |
|------|------|
| Python | 3.10+（推荐 3.12） |
| 操作系统 | Windows / macOS / Linux |
| 内存 | 4 GB（推荐 8 GB） |
| 磁盘 | 1 GB（不含 embedding 模型，模型约 400 MB） |
| 网络 | 首次运行需下载 embedding 模型 |

### 详细步骤

**第一步：克隆项目**

```cmd
git clone https://github.com/VGreenHand/rag-rpg.git
cd rag-rpg
```

**第二步：创建 Python 环境**

```cmd
conda create -n rag-rpg python=3.12 -y
conda activate rag-rpg
```

**第三步：安装依赖**

```cmd
pip install -r requirements.txt
```

依赖清单：`chromadb>=0.4.0`、`sentence-transformers>=2.2.0`、`fastapi>=0.104.0`、`uvicorn[standard]>=0.24.0`、`pydantic>=2.0.0`。

**第四步（可选）：初始化角色技能库**

把自己的角色设定 JSON 放入 `data/CharacterInfo/` 后执行：

```cmd
python scripts\ingest_initial.py
```

首次运行会自动下载 embedding 模型 `shibing624/text2vec-base-chinese`（约 400 MB）。

**第五步：启动服务端**

```cmd
python server.py
```

服务就绪后会显示 `http://127.0.0.1:8765`。

**第六步：安装 SillyTavern 扩展**

将 `st_extension/` 文件夹复制到：

| 安装位置 | 路径 |
|----------|------|
| 用户扩展 | `SillyTavern/data/default-user/extensions/RAG-RPG/` |
| 第三方扩展 | `SillyTavern/public/scripts/extensions/third-party/RAG-RPG/` |

重启 SillyTavern，在扩展面板中启用。

**第七步：验证**

```cmd
python -c "from pipeline import get_pipeline; p=get_pipeline(); print(p.get_stats())"
```

---

## 管理工具

所有脚本位于 `scripts/` 目录下。

### 日常工具

| 脚本 | 用途 | 用法 |
|------|------|------|
| `ingest_initial.py` | 从 JSON 初始化向量库 | `python scripts\ingest_initial.py` |
| `ingest_new.py` | 从标记 TXT 追加导入 | 编辑 `new_batch.txt` 后执行 |
| `quick_search.py` | 交互式语义检索 | `python scripts\quick_search.py` |
| `update_skill.py` | 更新技能条目 | 修改 entry_key 和内容后执行 |
| `check_env.py` | 检查 Python 环境和依赖 | `python scripts\check_env.py` |

### 环境与校验工具

| 脚本 | 用途 | 用法 |
|------|------|------|
| `setup_test_env.py` | 一键环境检测 + 重建 + 测试 | `python scripts\setup_test_env.py` |
| `check_keys.py` | ChromaDB ↔ JSON 条目一致性 | `python scripts\check_keys.py --full` |
| `check_metadata.py` | 查看所有条目的 entry_key | `python scripts\check_metadata.py` |
| `data_manifest.py` | 数据文件传输完整性校验 | `python scripts\data_manifest.py generate` |
| `validate_character_consistency.py` | JSON 文件结构有效性校验 | `python scripts\validate_character_consistency.py` |

### 批量导入格式

创建 `new_batch.txt`，每行使用 `[TYPE]` 标记：

```text
[SKILL] 技能：冰霜吐息。龙裔释放冰霜龙吼，冻结前方敌人。
[PLOT] 剧情：玩家在寒落神殿发现了龙石碎片，触发主线任务。
[SETTING] 设定：冬堡学院以幻术系魔法闻名。
```

---

## 跨设备开发

### 同一用户在不同设备上的操作流程

```
设备 A（已有数据）:
  把自己的 JSON 放入 data/CharacterInfo/
  python scripts/ingest_initial.py      ← 构建 ChromaDB
  (开始使用)

设备 B（新机器）:
  git clone / git pull
  pip install -r requirements.txt
  从设备 A 复制 data/ 下的 JSON 到同样位置
  python scripts\setup_test_env.py      ← 一键检测 + 重建 + 测试
  (开始使用)
```

### Git Hooks 保护

项目提供了 Git hooks 用于日常保护：

- **pre-commit**：提交前检查暂存区 JSON 文件格式（跳过 `data/`，不影响用户数据）
- **post-merge**：拉取后检测脚本/测试更新，提醒用户

启用一次即可：

```cmd
git config core.hooksPath .git/hooks
```

---

## API 端点

| 端点 | 功能 |
|------|------|
| `GET /api/health` | 健康检查（无需认证） |
| `GET /api/status` | 引擎状态与向量库统计 |
| `POST /api/dialogue/ingest` | 接收单轮对话，全流程处理 |
| `POST /api/dialogue/query` | 上下文搜索 + 约束生成 |
| `POST /api/batch/ingest` | 批量导入 |
| `GET/POST /api/checkpoint/*` | 断点状态查询、续点、清除 |
| `POST /api/skill/update` | 更新技能条目 |
| `POST /api/feedback` | 约束采用反馈，调整权重 |
| `GET /api/constraints/current` | 获取当前生效的约束 |

---

## 测试

```cmd
:: 环境初始化 + 重建 + 测试（推荐新设备使用）
python scripts\setup_test_env.py --rebuild --test

:: 仅运行测试套件
python tests\test_suite.py              165+ 条
python tests\test_checkpoint_resume.py  75 条
```

测试覆盖：功能测试、兼容性测试、性能基准测试、断点续执行、数据结构校验。

---

## 配置

`config.py` 是唯一配置入口：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | `127.0.0.1` | 监听地址 |
| `API_PORT` | `8765` | 监听端口 |
| `API_KEY` | `rag-rpg-local` | 认证密钥（可环境变量 `RAG_RPG_API_KEY` 覆盖） |
| `MODEL_NAME` | `shibing624/text2vec-base-chinese` | embedding 模型 |
| `MAX_CONTEXT_TURNS` | `5` | 查询考虑的最近对话轮数 |
| `TOP_K_RESULTS` | `3` | 每次检索最大结果数 |
| `MIN_RELEVANCE` | `0.25` | 最低相关度阈值 |
| `MAX_CONSTRAINT_CHARS` | `800` | 约束文本最大长度 |

---

## 项目结构

```
rag-rpg/
├── data/                           ← （可选）用户预设数据
├── scripts/                        ← 管理工具脚本
├── tests/                          ← 测试套件
├── st_extension/                   ← SillyTavern 扩展
│
├── server.py                       ← FastAPI 服务端
├── pipeline.py                     ← 对话处理管道
├── query_engine.py                 ← 语义查询引擎
├── constraint_engine.py            ← 剧情约束引擎
├── checkpoint_manager.py           ← 断点续执行管理器
├── config.py                       ← 全局配置
├── requirements.txt                ← Python 依赖
└── README.md                       ← 本文档
```

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 服务启动报错 `chromadb` 未安装 | 依赖未装 | `pip install -r requirements.txt` |
| 扩展不生效 | 目录错误 | 检查扩展复制路径 |
| JSON 找不到 | data/ 下无预设文件 | 正常，可跳过。如需角色技能，放入 JSON 后重试 |
| 查询空结果 | 首次使用 | 对话记忆会自动积累。角色技能需先 `ingest_initial.py` |
| `invalid api key` | 密钥不匹配 | 确认 SillyTavern 设置中的 API Key 与 config.py 一致 |
| embedding 模型下载失败 | 网络问题 | 确保网络畅通 |
| 测试卡死/checkpoint 残留 | 旧状态未清理 | `python -c "import shutil; shutil.rmtree('.checkpoints', ignore_errors=True)"` |

---

## 技术栈

| 技术 | 用途 |
|------|------|
| Python 3.12+ | 后端 |
| FastAPI + Uvicorn | REST API |
| ChromaDB | 向量数据库 |
| sentence-transformers | 文本向量化 |
| shibing624/text2vec-base-chinese | 中文 embedding 模型 |
| JavaScript (SillyTavern 扩展) | 前端事件监听 |
