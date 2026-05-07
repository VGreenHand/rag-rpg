"""
RAG-RPG 全面测试套件
覆盖：功能测试 / 兼容性测试 / 性能测试
运行方式：D:\Study\Project\rag-rpg\test_env\Scripts\python.exe test_suite.py
"""
import sys
import os
import json
import time
import shutil
import tempfile
import uuid
import re
import io
from pathlib import Path
from datetime import datetime

os.environ["RAG_RPG_API_KEY"] = "rag-rpg-local"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOTAL_TESTS = 0
PASSED = 0
FAILED = 0
ERRORS_LIST = []
PERF_METRICS = {}
RESULTS_BY_CATEGORY = {}


def check(condition, test_name, detail=""):
    global TOTAL_TESTS, PASSED, FAILED
    TOTAL_TESTS += 1
    if condition:
        PASSED += 1
        return True
    else:
        FAILED += 1
        ERRORS_LIST.append(f"  [FAIL] {test_name}: {detail}")
        return False


def run_section(name, func):
    global RESULTS_BY_CATEGORY
    print(f"\n{'=' * 65}")
    print(f"  {name}")
    print(f"{'=' * 65}")
    before = TOTAL_TESTS
    func()
    section_count = TOTAL_TESTS - before
    RESULTS_BY_CATEGORY[name] = section_count
    print(f"  ── 本节测试: {section_count} 条 ──")


# ═══════════════════════════════════════════════════════════
#  SECTION 1: ENVIRONMENT & IMPORT
# ═══════════════════════════════════════════════════════════

def test_environment():
    import platform
    ver = sys.version_info
    check(ver.major == 3 and ver.minor >= 10, "Python版本≥3.10",
          f"实际: {ver.major}.{ver.minor}.{ver.micro}")
    check(platform.system() == "Windows", "操作系统Windows")

    pkgs = {
        "chromadb": ("chromadb", "__version__"),
        "sentence_transformers": ("sentence_transformers", "__version__"),
        "fastapi": ("fastapi", "__version__"),
        "uvicorn": ("uvicorn", "__version__"),
        "pydantic": ("pydantic", "__version__"),
    }
    for name, (mod_name, attr) in pkgs.items():
        try:
            mod = __import__(mod_name)
            ver_str = getattr(mod, attr, "unknown")
            check(True, f"导入 {name}", f"v{ver_str}")
        except ImportError as e:
            check(False, f"导入 {name}", str(e))

    check("test_env" in sys.executable.lower(), "运行在test_env而非base",
          sys.executable)


# ═══════════════════════════════════════════════════════════
#  SECTION 2: CONFIG MODULE
# ═══════════════════════════════════════════════════════════

def test_config():
    import config
    check(isinstance(config.BASE_DIR, Path), "BASE_DIR是Path对象")
    check(config.API_HOST == "127.0.0.1", "API_HOST=127.0.0.1")
    check(config.API_PORT == 8765, "API_PORT=8765")
    check(config.API_KEY == "rag-rpg-local", "API_KEY正确")
    check(config.MODEL_NAME == "shibing624/text2vec-base-chinese", "MODEL_NAME正确")
    check(config.TOP_K_RESULTS == 3, "TOP_K_RESULTS=3")
    check(config.MAX_CONTEXT_TURNS == 5, "MAX_CONTEXT_TURNS=5")
    check(config.MIN_RELEVANCE == 0.25, "MIN_RELEVANCE=0.25")
    check(config.MAX_CONSTRAINT_CHARS == 800, "MAX_CONSTRAINT_CHARS=800")
    check(config.CONSTRAINT_COOLDOWN_TURNS == 3, "CONSTRAINT_COOLDOWN_TURNS=3")
    check(config.MAX_ACTIVE_CONSTRAINTS == 5, "MAX_ACTIVE_CONSTRAINTS=5")
    check(config.COLLECTION_SKILLS == "character_skills", "COLLECTION_SKILLS名称")
    check(config.COLLECTION_MEMORY == "my_rag_memory", "COLLECTION_MEMORY名称")
    check(config.COLLECTION_DIALOGUE == "dialogue_memory", "COLLECTION_DIALOGUE名称")
    check(config.COLLECTION_PLOT_STATE == "plot_state", "COLLECTION_PLOT_STATE名称")
    check(isinstance(config.DIALOGUE_DIR, Path), "DIALOGUE_DIR是Path对象")


# ═══════════════════════════════════════════════════════════
#  SECTION 3: PIPELINE MODULE
# ═══════════════════════════════════════════════════════════

def test_pipeline():
    from pipeline import get_pipeline
    pipeline = get_pipeline()
    check(pipeline is not None, "Pipeline实例化")
    check(pipeline.model is not None, "Pipeline embedding模型加载")
    check(pipeline.client is not None, "Pipeline ChromaDB客户端连接")

    # 3a. 文本清洗
    raw1 = "**拔出霜噬之刃**，<i>向前猛地一挥</i>\n\n一道冰霜剑气划破空气！"
    cleaned = pipeline._clean_text(raw1)
    check("霜噬之刃" in cleaned, "清洗-保留中文内容")
    check("**" not in cleaned, "清洗-移除Markdown粗体")
    check("<i>" not in cleaned, "清洗-移除HTML标签")
    check(len(cleaned) < len(raw1), f"清洗-文本缩短 raw={len(raw1)} clean={len(cleaned)}")
    check("\n\n\n" not in cleaned, "清洗-合并连续3+换行")

    # 空输入 边界
    check(pipeline._clean_text("") == "", "清洗-空字符串返回空")
    check(pipeline._clean_text("   ") == "", "清洗-纯空白返回空")

    # 特殊字符
    special = "～★◆《》「」【】〖〗"
    cleaned_sp = pipeline._clean_text(special)
    check(cleaned_sp == special, "清洗-保留CJK特殊字符")

    # 3b. 术语提取
    terms_empty = pipeline._extract_key_terms("")
    check(terms_empty == [], "术语提取-空文本返回空列表")

    # 无匹配
    terms_unm = pipeline._extract_key_terms("今天天气不错啊")
    check(isinstance(terms_unm, list), "术语提取-无匹配返回列表")

    # 3c. TXT写入
    today = datetime.now().strftime("%Y-%m-%d")
    txt_path = pipeline._write_txt("user", "测试角色", "测试对话内容", 1, ["skill_lightsaber_mastery"])
    check(os.path.exists(txt_path), f"TXT写入-文件创建 ({txt_path})")
    check(today in txt_path, "TXT写入-文件名含日期")

    with open(txt_path, "r", encoding="utf-8") as f:
        content = f.read()
    check("Turn #1" in content, "TXT写入-包含轮次")
    check("测试角色" in content, "TXT写入-包含角色名")
    check("测试对话内容" in content, "TXT写入-包含对话内容")
    check("skill_lightsaber_mastery" in content, "TXT写入-包含关键术语")

    # 3d. 向量入库
    doc_id = pipeline._embed_and_store(
        "测试内容：光剑精通是剑魂的核心能力",
        "user", "测试者", 1, ["光剑精通"]
    )
    check(doc_id is not None, "向量入库-返回doc_id")
    check(len(doc_id) > 10, "向量入库-doc_id格式有效")

    # 重复入库
    doc_id2 = pipeline._embed_and_store(
        "另一条测试内容", "ai", "AI助手", 2, []
    )
    check(doc_id2 != doc_id, "向量入库-不同内容产生不同ID")

    # 3e. 完整管道 process_turn
    result = pipeline.process_turn(
        speaker="user",
        name="无名者",
        content="**拔出霜噬之刃**，施展里·鬼剑术向前攻击！",
        turn=99
    )
    check(result["status"] == "ok", f"process_turn-状态OK")
    check(result["turn"] == 99, "process_turn-轮次正确")
    check(result["raw_length"] >= result["cleaned_length"], "process_turn-清洗后不增长")
    check(isinstance(result["key_terms_found"], list), "process_turn-术语返回为列表")
    check(os.path.exists(result["txt_path"]), "process_turn-TXT文件存在")

    # 3f. 统计
    stats = pipeline.get_stats()
    check("dialogue_memory" in stats, "get_stats-含dialogue_memory")
    check("character_skills" in stats, "get_stats-含character_skills")
    check(isinstance(stats["dialogue_memory"], int), "get_stats-统计值是整数")

    # 3g. 批处理
    result_batch = pipeline.process_batch_txt()
    check(result_batch["status"] in ("ok", "empty"), "process_batch_txt-状态OK或empty")


# ═══════════════════════════════════════════════════════════
#  SECTION 4: QUERY ENGINE
# ═══════════════════════════════════════════════════════════

def test_query_engine():
    from query_engine import get_query_engine
    qe = get_query_engine()
    check(qe is not None, "QueryEngine实例化")
    check(qe.model is not None, "QueryEngine embedding模型加载")

    # 4a. build_queries
    ctx_empty = []
    qs_empty = qe.build_queries(ctx_empty)
    check(isinstance(qs_empty, list), "build_queries-空上下文返回列表")

    ctx_basic = [
        {"speaker": "user", "content": "我想使用里·鬼剑术攻击敌人"},
        {"speaker": "ai", "content": "你拔出霜噬之刃，摆出里·鬼剑术的起手式"},
        {"speaker": "user", "content": "好的，向前发动连续斩击"},
    ]
    qs = qe.build_queries(ctx_basic)
    check(len(qs) > 0, f"build_queries-生成{len(qs)}条查询")
    check(any("里·鬼剑术" in q for q in qs), "build_queries-查询含关键内容")

    # 只有AI消息
    ctx_ai = [{"speaker": "ai", "content": "前方出现一名持剑的敌人"}]
    qs_ai = qe.build_queries(ctx_ai)
    check(len(qs_ai) > 0, "build_queries-纯AI上下文生成查询")

    # 长上下文截断
    ctx_long = [
        {"speaker": "user", "content": "x" * 500},
        {"speaker": "ai", "content": "y" * 500},
    ]
    qs_long = qe.build_queries(ctx_long)
    check(len(qs_long) > 0, "build_queries-长文本正常处理")

    # 4b. search 单查询
    results = qe.search("光剑精通")
    check(isinstance(results, list), "search-返回列表")
    check(len(results) > 0, f"search-检索到{len(results)}条结果")
    if results:
        r = results[0]
        check("collection" in r, "search-结果含collection字段")
        check("document" in r, "search-结果含document字段")
        check("score" in r, "search-结果含score字段")
        check(0 <= r["score"] <= 1, f"search-score在[0,1]范围: {r['score']}")

    # 指定集合
    results_skills = qe.search("技能", collections=["character_skills"])
    check(isinstance(results_skills, list), "search-指定集合返回列表")

    # 查询不存在的集合
    results_bad = qe.search("test", collections=["non_existent_collection"])
    check(isinstance(results_bad, list), "search-不存在的集合不报错")

    # 4c. multi_search
    ms = qe.multi_search(ctx_basic)
    check("results" in ms, "multi_search-含results键")
    check("total_hits" in ms, "multi_search-含total_hits键")
    check("queries_used" in ms, "multi_search-含queries_used键")
    check(ms["queries_used"] > 0, f"multi_search-使用了{ms['queries_used']}条查询")
    check(isinstance(ms["total_hits"], int), "multi_search-total_hits是整数")

    # 去重
    ms2 = qe.multi_search(ctx_basic, k=1)
    check(ms2["total_hits"] <= ms["total_hits"] + 3, "multi_search-k参数影响结果数量")

    # 空上下文
    ms_empty = qe.multi_search([])
    check(ms_empty["total_hits"] == 0, "multi_search-空上下文返回0结果")

    # 4d. format_for_llm
    formatted = qe.format_for_llm(ms)
    check(isinstance(formatted, str), "format_for_llm-返回字符串")
    if ms["total_hits"] > 0:
        check("[RAG-RPG" in formatted, "format_for_llm-含标签头")
        check("相关度:" in formatted, "format_for_llm-含相关度")

    fmt_empty = qe.format_for_llm({"results": [], "total_hits": 0})
    check(fmt_empty == "", "format_for_llm-空结果返回空字符串")


# ═══════════════════════════════════════════════════════════
#  SECTION 5: CONSTRAINT ENGINE
# ═══════════════════════════════════════════════════════════

def test_constraint_engine():
    from constraint_engine import get_constraint_engine
    ce = get_constraint_engine()
    check(ce is not None, "ConstraintEngine实例化")

    # 5a. generate_constraints - 空结果
    empty_sr = {"results": [], "total_hits": 0}
    constraint = ce.generate_constraints(empty_sr, [])
    check(constraint == "", "generate_constraints-空结果返回空字符串")

    # 构造模拟检索结果
    mock_results = {
        "results": [
            {
                "collection": "character_skills",
                "id": "skill_lightsaber_mastery",
                "document": "技能：光剑精通。剑魂对光剑的专属掌控能力，提升攻击速度、命中率与暴击概率。当前熟练度 12/100。",
                "metadata": {"type": "skill", "entry_key": "skill_lightsaber_mastery"},
                "score": 0.85,
            },
            {
                "collection": "character_skills",
                "id": "mechanic_lightsaber",
                "document": "机制：光剑特性。光剑是以能量为刃的轻型武器，极高攻速与低硬直，但基础攻击力较低。",
                "metadata": {"type": "mechanic", "entry_key": "光剑特性"},
                "score": 0.72,
            },
            {
                "collection": "dialogue_memory",
                "id": str(uuid.uuid4()),
                "document": "对话：你拔出霜噬之刃，准备迎战前方的敌人",
                "metadata": {"type": "dialogue"},
                "score": 0.55,
            },
        ],
        "total_hits": 3,
    }

    ctx = [
        {"speaker": "user", "content": "我拔出光剑准备战斗"},
        {"speaker": "ai", "content": "霜噬之刃发出幽幽蓝光"},
    ]
    c = ce.generate_constraints(mock_results, ctx)
    check(len(c) > 0, f"generate_constraints-生成约束文本 len={len(c)}")
    check("光剑精通" in c or "技能约束" in c, "generate_constraints-含技能约束")
    check("[RAG-RPG 剧情约束" in c or "[RAG-RPG" in c, "generate_constraints-含标签头")
    check(len(c) <= 800 + 200, f"generate_constraints-长度可控 (len={len(c)})")

    # 5b. 冷却机制
    c2 = ce.generate_constraints(mock_results, ctx)
    check(isinstance(c2, str), "冷却-二次约束生成不报错")

    # 重置冷却并测试
    ce._cooldowns.clear()
    c3 = ce.generate_constraints(mock_results, ctx)
    check(len(c3) > 0, "冷却-重置后可重新生成")

    # 5c. _build_constraint 各类型
    skill_r = {"document": "光剑精通内容", "metadata": {"type": "skill"}, "score": 0.9, "collection": "character_skills"}
    skill_c = ce._build_constraint(skill_r, [])
    check("技能约束" in skill_c, "build_constraint-skill类型包含标签")

    mech_r = {"document": "机制内容", "metadata": {"type": "mechanic"}, "score": 0.8, "collection": "character_skills"}
    mech_c = ce._build_constraint(mech_r, [])
    check("机制约束" in mech_c, "build_constraint-mechanic类型包含标签")

    setting_r = {"document": "世界观设定", "metadata": {"type": "setting"}, "score": 0.7, "collection": "character_skills"}
    setting_c = ce._build_constraint(setting_r, [])
    check("世界观约束" in setting_c, "build_constraint-setting类型包含标签")

    # 5d. feedback 权重
    initial_w = ce.get_weight("skill")
    ce.update_feedback("skill", True)
    w_up = ce.get_weight("skill")
    check(w_up > initial_w, f"feedback-正向反馈提升权重 ({initial_w:.2f}→{w_up:.2f})")

    ce.update_feedback("skill", False)
    w_down = ce.get_weight("skill")
    check(w_down < w_up, f"feedback-负向反馈降低权重 ({w_up:.2f}→{w_down:.2f})")

    check(ce.get_weight("unknown_type") == 1.0, "get_weight-未知类型返回默认1.0")

    # 权重边界
    for _ in range(50):
        ce.update_feedback("skill", False)
    check(ce.get_weight("skill") >= 0.3, f"feedback-权重不低于0.3: {ce.get_weight('skill'):.4f}")

    for _ in range(50):
        ce.update_feedback("skill", True)
    check(ce.get_weight("skill") <= 3.0, f"feedback-权重不超过3.0: {ce.get_weight('skill'):.4f}")


# ═══════════════════════════════════════════════════════════
#  SECTION 6: FASTAPI SERVER (via TestClient)
# ═══════════════════════════════════════════════════════════

def test_server():
    print("  (启动 TestClient...)")
    from fastapi.testclient import TestClient
    from server import app
    client = TestClient(app)

    headers = {"X-API-Key": "rag-rpg-local"}

    # 6a. status
    r = client.get("/api/status", headers=headers)
    check(r.status_code == 200, f"GET /api/status → {r.status_code}")
    data = r.json()
    check(data["status"] == "running", "/api/status-状态running")
    check("vector_db" in data, "/api/status-含vector_db统计")

    # 6b. status - 无认证
    r_noauth = client.get("/api/status")
    check(r_noauth.status_code == 403, f"GET /api/status 无认证 → 403 (实际:{r_noauth.status_code})")

    # 6c. status - 错误Key
    r_badkey = client.get("/api/status", headers={"X-API-Key": "wrong-key"})
    check(r_badkey.status_code == 403, f"GET /api/status 错误Key → 403 (实际:{r_badkey.status_code})")

    # 6d. ingest
    ingest_payload = {
        "speaker": "user",
        "name": "测试用户",
        "content": "使用光剑精通技能斩杀敌人",
        "turn": 1,
    }
    r_ingest = client.post("/api/dialogue/ingest", json=ingest_payload, headers=headers)
    check(r_ingest.status_code == 200, f"POST /api/dialogue/ingest → {r_ingest.status_code}")
    ingest_data = r_ingest.json()
    check(ingest_data["status"] == "ok", "ingest-状态OK")
    check(ingest_data["turn"] == 1, "ingest-轮次一致")
    check("doc_id" in ingest_data, "ingest-含doc_id")

    # 6e. ingest - 空内容
    r_empty = client.post("/api/dialogue/ingest",
                          json={"speaker": "user", "name": "", "content": "", "turn": 0},
                          headers=headers)
    check(r_empty.status_code == 200, f"ingest-空内容不崩溃 → {r_empty.status_code}")

    # 6f. query
    query_payload = {
        "context": [
            {"speaker": "user", "content": "使用光剑攻击"},
            {"speaker": "ai", "content": "你拔出武器"},
        ],
        "max_results": 2,
        "generate_constraint": True,
    }
    r_query = client.post("/api/dialogue/query", json=query_payload, headers=headers)
    check(r_query.status_code == 200, f"POST /api/dialogue/query → {r_query.status_code}")
    qdata = r_query.json()
    check("raw_results" in qdata, "query-含raw_results")
    check("formatted" in qdata, "query-含formatted")
    check("constraint_text" in qdata, "query-含constraint_text")

    # 查询不生成约束
    q_no_const = {
        "context": [{"speaker": "user", "content": "测试"}],
        "max_results": 1,
        "generate_constraint": False,
    }
    r_nc = client.post("/api/dialogue/query", json=q_no_const, headers=headers)
    check(r_nc.status_code == 200, f"query-无约束模式 → {r_nc.status_code}")

    # 6g. skill update
    upd_payload = {
        "entry_key": "skill_lightsaber_mastery",
        "new_content": "技能：光剑精通。测试更新内容。熟练度 50/100。[type:skill]",
    }
    r_upd = client.post("/api/skill/update", json=upd_payload, headers=headers)
    check(r_upd.status_code in (200, 404), f"POST /api/skill/update → {r_upd.status_code}")

    # 更新不存在的技能
    upd_bad = {"entry_key": "non_existent_skill_xyz", "new_content": "test"}
    r_bad_upd = client.post("/api/skill/update", json=upd_bad, headers=headers)
    check(r_bad_upd.status_code == 404, f"POST /api/skill/update 不存在 → 404 (实际:{r_bad_upd.status_code})")

    # 6h. feedback
    fb_payload = {"entry_type": "skill", "was_used": True}
    r_fb = client.post("/api/feedback", json=fb_payload, headers=headers)
    check(r_fb.status_code == 200, f"POST /api/feedback → {r_fb.status_code}")
    fb_data = r_fb.json()
    check("new_weight" in fb_data, "feedback-含new_weight")
    check(fb_data["status"] == "ok", "feedback-状态OK")

    # 6i. batch ingest
    r_batch = client.post("/api/batch/ingest", headers=headers)
    check(r_batch.status_code == 200, f"POST /api/batch/ingest → {r_batch.status_code}")


# ═══════════════════════════════════════════════════════════
#  SECTION 7: COMPATIBILITY & EDGE CASES
# ═══════════════════════════════════════════════════════════

def test_compatibility():
    from pipeline import get_pipeline
    from query_engine import get_query_engine
    from constraint_engine import get_constraint_engine

    pipeline = get_pipeline()
    qe = get_query_engine()
    ce = get_constraint_engine()

    # 7a. 纯英文
    eng = pipeline._clean_text("The hero draws his **light saber** and charges forward!")
    check("light saber" in eng, "兼容-英文文本保留内容")
    check("**" not in eng, "兼容-英文文本清洗加粗")

    # 7b. 混合中英文
    mixed = pipeline._clean_text("使用Light Saber**攻击**敌人 with great force!")
    check("Light Saber" in mixed, "兼容-中英混合保留英文")
    check("攻击" in mixed, "兼容-中英混合保留中文")
    check("**" not in mixed, "兼容-中英混合清洗标记")

    # 7c. 超长文本
    long_text = "探索" * 300
    cleaned_long = pipeline._clean_text(long_text)
    check(len(cleaned_long) <= len(long_text), f"兼容-超长文本清洗完成 len={len(cleaned_long)}")

    # 7d. 全标点
    punct = "！？。，、；：""''（）【】《》…—"
    cleaned_punct = pipeline._clean_text(punct)
    check(len(cleaned_punct) >= 0, "兼容-全标点文本不崩溃")

    # 7e. 换行符处理
    multiline = "第一行\n第二行\n\n第三行\r\n第四行"
    cl = pipeline._clean_text(multiline)
    check("第二行" in cl, "兼容-换行符保留内容")

    # 7f. 嵌入向量维度一致性
    test_texts = [
        "技能测试一：光剑精通",
        "一个完全不同的主题内容天气很好今天",
    ]
    emb1 = pipeline.model.encode(test_texts[0]).tolist()
    emb2 = pipeline.model.encode(test_texts[1]).tolist()
    check(len(emb1) == len(emb2), f"兼容-向量维度一致 ({len(emb1)})")
    check(len(emb1) > 0, "兼容-向量非空")
    check(all(isinstance(x, float) for x in emb1[:5]), "兼容-向量元素为浮点数")

    # 7g. 元数据字段
    from config import COLLECTION_DIALOGUE
    col = pipeline.client.get_collection(name=COLLECTION_DIALOGUE)
    results = col.get()
    if results["ids"]:
        meta = results["metadatas"][0]
        for key in ["speaker", "turn", "timestamp"]:
            check(key in meta, f"兼容-元数据含{key}字段")

    # 7h. ChromaDB查询的entry_key过滤
    from config import COLLECTION_SKILLS
    skill_col = pipeline.client.get_collection(name=COLLECTION_SKILLS)
    skill_results = skill_col.get()
    if skill_results["ids"]:
        check(len(skill_results["metadatas"]) > 0, "兼容-skills集合有元数据")
        check("entry_key" in skill_results["metadatas"][0], "兼容-skills元数据含entry_key")

    # 7i. JSON序列化兼容性
    test_dict = {
        "key": "value",
        "chinese": "中文字符串",
        "number": 123.456,
        "list": [1, 2, 3],
    }
    json_str = json.dumps(test_dict, ensure_ascii=False)
    parsed = json.loads(json_str)
    check(parsed == test_dict, "兼容-中英文JSON序列化往返一致")

    # 7j. 模块单例一致性
    from pipeline import get_pipeline as gp1
    from pipeline import get_pipeline as gp2
    check(gp1() is gp2(), "兼容-Pipeline单例一致")

    from query_engine import get_query_engine as gq1
    from query_engine import get_query_engine as gq2
    check(gq1() is gq2(), "兼容-QueryEngine单例一致")

    from constraint_engine import get_constraint_engine as gc1
    from constraint_engine import get_constraint_engine as gc2
    check(gc1() is gc2(), "兼容-ConstraintEngine单例一致")


# ═══════════════════════════════════════════════════════════
#  SECTION 8: PERFORMANCE TESTS
# ═══════════════════════════════════════════════════════════

def test_performance():
    from pipeline import get_pipeline
    from query_engine import get_query_engine
    from constraint_engine import get_constraint_engine

    pipeline = get_pipeline()
    qe = get_query_engine()
    ce = get_constraint_engine()

    # 8a. 文本清洗吞吐量
    sample_texts = [
        "**拔出霜噬之刃**，施展*里·鬼剑术*向前攻击！",
        "你感受到剑身传来一阵冰冷的气息，仿佛有什么在呼唤你",
        "前方出现三名手持武器的敌人，<i>他们的眼中闪烁着红光</i>",
    ] * 10

    t0 = time.perf_counter()
    for text in sample_texts:
        pipeline._clean_text(text)
    t_clean = time.perf_counter() - t0
    rate = len(sample_texts) / t_clean if t_clean > 0 else 0
    PERF_METRICS["文本清洗"] = {"total": len(sample_texts), "time_s": round(t_clean, 4), "rate_per_s": round(rate, 1)}
    check(t_clean < 2.0, f"性能-清洗30条文本<2s: {t_clean:.4f}s", f"{t_clean:.4f}s")
    check(rate > 100, f"性能-清洗速率>100/s: {rate:.0f}/s", f"{rate:.0f}/s")

    # 8b. 向量编码延迟
    test_emb_texts = ["测试文本" + str(i) for i in range(5)]
    t0 = time.perf_counter()
    _ = pipeline.model.encode(test_emb_texts[:1]).tolist()
    t_single = time.perf_counter() - t0
    PERF_METRICS["单条向量编码"] = {"time_s": round(t_single, 4)}
    check(t_single < 5.0, f"性能-单条embedding<5s: {t_single:.4f}s", f"{t_single:.4f}s")

    t0 = time.perf_counter()
    _ = pipeline.model.encode(test_emb_texts).tolist()
    t_batch = time.perf_counter() - t0
    PERF_METRICS["批量向量编码(5条)"] = {"time_s": round(t_batch, 4), "avg_per_item_s": round(t_batch / 5, 4)}
    check(t_batch < 10.0, f"性能-批量5条embedding<10s: {t_batch:.4f}s", f"{t_batch:.4f}s")

    # 8c. ChromaDB 写入延迟
    t0 = time.perf_counter()
    doc_id = pipeline._embed_and_store(
        "性能测试写入内容", "user", "perf_tester", 999, ["test"]
    )
    t_write = time.perf_counter() - t0
    PERF_METRICS["ChromaDB写入"] = {"time_s": round(t_write, 4)}
    check(t_write < 5.0, f"性能-向量写入<5s: {t_write:.4f}s", f"{t_write:.4f}s")

    # 8d. 查询延迟
    t0 = time.perf_counter()
    _ = qe.search("光剑")
    t_search = time.perf_counter() - t0
    PERF_METRICS["单查询检索"] = {"time_s": round(t_search, 4)}
    check(t_search < 5.0, f"性能-单查询检索<5s: {t_search:.4f}s", f"{t_search:.4f}s")

    # 8e. 多查询检索延迟
    ctx = [
        {"speaker": "user", "content": "使用光剑攻击"},
        {"speaker": "ai", "content": "你拔出了武器"},
    ] * 3
    t0 = time.perf_counter()
    ms = qe.multi_search(ctx)
    t_multi = time.perf_counter() - t0
    PERF_METRICS["多查询检索"] = {"time_s": round(t_multi, 4), "hits": ms["total_hits"]}
    check(t_multi < 20.0, f"性能-多查询检索<20s: {t_multi:.4f}s", f"{t_multi:.4f}s")

    # 8f. 约束生成延迟
    mock = {
        "results": [
            {
                "id": "test", "document": "测试" * 20,
                "metadata": {"type": "skill"}, "score": 0.9,
                "collection": "character_skills",
            }
        ],
        "total_hits": 1,
    }
    ce._cooldowns.clear()
    t0 = time.perf_counter()
    _ = ce.generate_constraints(mock, ctx)
    t_constraint = time.perf_counter() - t0
    PERF_METRICS["约束生成"] = {"time_s": round(t_constraint, 4)}
    check(t_constraint < 0.1, f"性能-约束生成<0.1s: {t_constraint:.4f}s", f"{t_constraint:.4f}s")

    # 8g. process_turn 端到端延迟
    t0 = time.perf_counter()
    _ = pipeline.process_turn("user", "test", "测试端到端管道性能", 1000)
    t_e2e = time.perf_counter() - t0
    PERF_METRICS["端到端process_turn"] = {"time_s": round(t_e2e, 4)}
    check(t_e2e < 10.0, f"性能-端到端延迟<10s: {t_e2e:.4f}s", f"{t_e2e:.4f}s")

    # 8h. ChromaDB 集合大小统计
    stats = pipeline.get_stats()
    total_docs = sum(v for k, v in stats.items() if k != "dialogue_txt_files")
    PERF_METRICS["向量库大小"] = stats
    check(total_docs >= 3, f"性能-向量库至少3条测试数据: {total_docs}", f"实际: {total_docs}")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    global PERF_METRICS
    start_time = time.perf_counter()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       RAG-RPG 全面测试套件 v1.0                              ║")
    print("║       Python:", f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}".ljust(40), "║")
    print(f"║       时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".ljust(55), "║")
    print("╚══════════════════════════════════════════════════════════════╝")

    run_section("[1] 环境与依赖导入", test_environment)
    run_section("[2] 配置模块", test_config)
    run_section("[3] 对话处理管道 (pipeline.py)", test_pipeline)
    run_section("[4] 查询引擎 (query_engine.py)", test_query_engine)
    run_section("[5] 剧情约束引擎 (constraint_engine.py)", test_constraint_engine)
    run_section("[6] FastAPI服务端 (server.py)", test_server)
    run_section("[7] 兼容性与边界条件", test_compatibility)
    run_section("[8] 性能测试", test_performance)

    elapsed = time.perf_counter() - start_time

    # ── 测试报告 ──
    print(f"\n{'═' * 65}")
    print(f"  测试报告")
    print(f"{'═' * 65}")
    print(f"  总测试数 : {TOTAL_TESTS}")
    print(f"  通过     : {PASSED} [OK]")
    print(f"  失败     : {FAILED} [FAIL]")
    print(f"  通过率   : {PASSED / TOTAL_TESTS * 100:.1f}%" if TOTAL_TESTS > 0 else "  N/A")
    print(f"  总耗时   : {elapsed:.2f}s")
    print()

    if ERRORS_LIST:
        print(f"  {'─' * 55}")
        print(f"  失败详情:")
        for err in ERRORS_LIST:
            print(err)
        print()

    print(f"  {'─' * 55}")
    print(f"  分类统计:")
    for cat, count in RESULTS_BY_CATEGORY.items():
        print(f"    {cat}: {count} 条测试")

    print(f"\n  {'─' * 55}")
    print(f"  性能指标:")
    for name, data in PERF_METRICS.items():
        if isinstance(data, dict):
            items = ", ".join(f"{k}={v}" for k, v in data.items())
            print(f"    {name}: {items}")
        else:
            print(f"    {name}: {data}")

    # 环境信息
    print(f"\n  {'─' * 55}")
    print(f"  环境信息:")
    print(f"    Python: {sys.version}")
    print(f"    Executable: {sys.executable}")
    try:
        import chromadb
        print(f"    chromadb: {chromadb.__version__}")
    except Exception:
        pass
    try:
        import sentence_transformers
        print(f"    sentence-transformers: {sentence_transformers.__version__}")
    except Exception:
        pass
    try:
        import torch
        print(f"    torch: {torch.__version__}  (CUDA: {torch.cuda.is_available()})")
    except Exception:
        pass
    try:
        import fastapi
        print(f"    fastapi: {fastapi.__version__}")
    except Exception:
        pass

    print(f"\n{'═' * 65}")
    if FAILED == 0:
        print("    [PASS] 全部测试通过！")
    else:
        print(f"    [WARN] 存在 {FAILED} 条失败测试，请检查上述详情")
    print(f"{'═' * 65}\n")

    return FAILED == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
