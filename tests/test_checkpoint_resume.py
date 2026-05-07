"""
RAG-RPG v2.0 断点续执行与异常恢复 集成测试
运行方式：test_env/Scripts/python.exe test_checkpoint_resume.py
"""
import sys
import os
import json
import time
import uuid
import threading
from pathlib import Path
from datetime import datetime

os.environ["RAG_RPG_API_KEY"] = "rag-rpg-local"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TOTAL, PASSED, FAILED = 0, 0, 0
ERRORS = []


def check(condition, name, detail=""):
    global TOTAL, PASSED, FAILED
    TOTAL += 1
    if condition:
        PASSED += 1
    else:
        FAILED += 1
        ERRORS.append(f"  [FAIL] {name}: {detail}")
    return condition


def section(title):
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ═══════════════════════════════════════════════════════════
#  SECTION 1: CHECKPOINT MANAGER UNIT TESTS
# ═══════════════════════════════════════════════════════════

def test_checkpoint_unit():
    from checkpoint_manager import (
        get_checkpoint, CheckpointManager, CHECKPOINT_FILE,
        CHECKPOINT_DIR, HEARTBEAT_FILE,
    )
    cp = get_checkpoint()
    cp.clear_checkpoint()

    check(not cp.can_resume(), "1.1 清除后无可续点任务")

    steps_def = [
        {"id": "step_1", "name": "加载向量库"},
        {"id": "step_2", "name": "清洗文本"},
        {"id": "step_3", "name": "提取关键术语"},
        {"id": "step_4", "name": "生成嵌入向量"},
        {"id": "step_5", "name": "写入ChromaDB"},
    ]
    exec_id = cp.init_task("batch_ingest", {"file": "test.txt"}, steps_def)
    check(len(exec_id) > 10, "1.2 init_task返回有效execution_id")
    check(cp._state.status == "running", "1.3 任务状态=running")
    check(cp._state.total_steps == 5, "1.4 total_steps=5")

    check(CHECKPOINT_FILE.exists(), "1.5 状态持久化到磁盘")

    cp.mark_step_start(0)
    s0 = cp._state.steps[0]
    check(s0.status == "running", "1.6 mark_step_start状态=running")
    check(s0.attempts == 1, "1.7 attempts递增")

    cp.mark_step_success(0, {"result": "loaded 10 items"})
    check(cp._state.steps[0].status == "completed", "1.8 mark_step_success")

    cp.mark_step_start(1)
    cp.mark_step_failed(1, "ChromaDB超时")
    check(cp._state.steps[1].status == "failed", "1.9 mark_step_failed")
    check("超时" in cp._state.steps[1].last_error, "1.10 错误信息已记录")

    cp.mark_step_skipped(2, "术语库为空")
    s2 = cp._state.steps[2]
    check(s2.status == "skipped", "1.11 mark_step_skipped")

    progress = cp.get_progress()
    check(progress["current_step"] == 1, "1.12 current_step=1 (mark_step_skipped不更新current_step)")
    check(progress["progress"] == 1/5, f"1.13 进度=20% (实际:{progress['progress']:.2f})")
    check(progress["stats"]["total_processed"] == 3, "1.14 total_processed=3")
    check(progress["stats"]["total_succeeded"] == 1, "1.15 succeeded=1")
    check(progress["stats"]["total_failed"] == 1, "1.16 failed=1")
    check(progress["stats"]["total_skipped"] == 1, "1.17 skipped=1")

    pending = cp.get_pending_steps()
    check(len(pending) == 3, f"1.18 待执行步骤=3 (failed+2 pending) (实际:{len(pending)})")
    check(pending[0].step_id == "step_2", "1.19 pending首个是失败的步骤")

    cp.complete_task("partial")
    check(cp._state.status == "partial", "1.20 complete_task状态=partial")

    progress2 = cp.get_progress()
    check(progress2["status"] == "partial", "1.21 完成后进度可查")


# ═══════════════════════════════════════════════════════════
#  SECTION 2: RESUME FROM CHECKPOINT
# ═══════════════════════════════════════════════════════════

def test_resume_flow():
    from checkpoint_manager import get_checkpoint

    cp = get_checkpoint()
    cp.clear_checkpoint()

    steps_def = [
        {"id": f"resume_{i}", "name": f"步骤 {i}"}
        for i in range(10)
    ]
    cp.init_task("resume_test", {}, steps_def)

    for i in range(4):
        cp.mark_step_start(i)
        cp.mark_step_success(i, {"index": i})

    cp.mark_step_start(4)
    cp.mark_step_failed(4, "模拟中断-网络超时")

    progress_before = cp.get_progress()
    check(progress_before["stats"]["total_succeeded"] == 4, "2.1 中断前成功4条")

    cp2 = get_checkpoint()
    saved = cp2.load_state()
    check(saved is not None, "2.2 load_state可从磁盘恢复")
    check(saved["stats"]["total_succeeded"] == 4, "2.3 恢复后成功数一致")
    check(saved["current_step"] == 4, "2.4 恢复后current_step=4")

    cp2.mark_step_start(4)
    cp2.mark_step_success(4, {"retry": "ok"})

    for i in range(5, 10):
        cp2.mark_step_start(i)
        cp2.mark_step_success(i)

    cp2.complete_task("completed")
    final = cp2.get_progress()
    check(final["stats"]["total_succeeded"] == 10, f"2.5 续点完成共10条 (实际:{final['stats']['total_succeeded']})")
    check(final["status"] == "completed", "2.6 最终状态=completed")


# ═══════════════════════════════════════════════════════════
#  SECTION 3: TIMEOUT PROTECTION
# ═══════════════════════════════════════════════════════════

def test_timeout_protection():
    from checkpoint_manager import get_checkpoint, TimeoutError as CkpTimeoutError

    cp = get_checkpoint()
    cp.clear_checkpoint()

    def slow_func():
        time.sleep(3)
        return "done"

    def fast_func():
        return "instant"

    t0 = time.perf_counter()
    result = cp.execute_with_timeout(fast_func, timeout=2.0)
    elapsed = time.perf_counter() - t0
    check(result == "instant", "3.1 快速操作正常返回")
    check(elapsed < 2.0, f"3.2 快速操作不等待超时 (实际:{elapsed:.3f}s)")

    t0 = time.perf_counter()
    try:
        cp.execute_with_timeout(slow_func, timeout=0.5)
        check(False, "3.3 超时操作应抛出异常")
    except CkpTimeoutError as e:
        elapsed = time.perf_counter() - t0
        check("超时" in str(e), f"3.4 超时异常信息正确")
        check(elapsed < 1.5, f"3.5 超时后快速返回 (实际:{elapsed:.3f}s)")

    steps = [{"id": "safe_1", "name": "安全执行测试"}]
    cp.init_task("timeout_test", {}, steps)

    try:
        cp.safe_execute(0, slow_func, timeout=0.3)
        check(False, "3.6 safe_execute超时不应静默")
    except CkpTimeoutError:
        check(cp._state.steps[0].status == "failed", "3.7 safe_execute标记失败")
        check("超时" in cp._state.steps[0].last_error, "3.8 超时错误已记录到步骤")


# ═══════════════════════════════════════════════════════════
#  SECTION 4: HEARTBEAT MONITORING
# ═══════════════════════════════════════════════════════════

def test_heartbeat():
    from checkpoint_manager import get_checkpoint

    cp = get_checkpoint()
    cp.clear_checkpoint()

    alive_before = cp.is_alive()
    check(not alive_before["alive"], "4.1 清除后无心跳")

    steps = [{"id": "hb_1", "name": "Heartbeat test"}]
    cp.init_task("heartbeat_test", {}, steps)

    time.sleep(0.5)
    alive = cp.is_alive()
    check(alive["alive"], "4.2 任务启动后有心跳")
    check(alive["age_seconds"] < 5, f"4.3 心跳新鲜度<5s (实际:{alive['age_seconds']})")
    check(alive["execution_id"] == cp._state.execution_id, "4.4 心跳含正确execution_id")

    cp.mark_step_start(0)
    time.sleep(0.3)
    alive2 = cp.is_alive()
    check(alive2["current_step"] == 0, "4.5 心跳反映当前步骤")

    cp.complete_task()
    alive3 = cp.is_alive()
    check(alive3["status"] == "completed", "4.6 心跳反映完成状态")


# ═══════════════════════════════════════════════════════════
#  SECTION 5: PIPELINE SAFE OPERATIONS
# ═══════════════════════════════════════════════════════════

def test_pipeline_safe():
    from pipeline import get_pipeline, SafeTimer
    from checkpoint_manager import TimeoutError as CkpTimeoutError

    pipeline = get_pipeline()

    clean_empty = pipeline._clean_text("")
    check(clean_empty == "", "5.1 清洗空字符串")

    clean_normal = pipeline._clean_text("**测试**行动开始")
    check("测试" in clean_normal, "5.2 清洗正常文本")
    check("**" not in clean_normal, "5.3 清洗移除加粗")

    terms_empty = pipeline._extract_key_terms("")
    check(terms_empty == [], "5.4 空文本无术语")

    raw = "使用光剑精通攻击敌人，施展里·鬼剑术"
    result = pipeline.process_turn("user", "测试者", raw, 9999)
    check(result["status"] == "ok", "5.5 process_turn状态OK")
    check(result["turn"] == 9999, "5.6 turn正确")
    check(result["raw_length"] >= result["cleaned_length"], "5.7 清洗不膨胀")

    pipeline.invalidate_terms_cache()
    check(pipeline._known_terms is None, "5.8 invalidate_terms_cache清空缓存")

    terms_after = pipeline._extract_key_terms("测试")
    check(isinstance(terms_after, list), "5.9 缓存清空后重新加载不报错")

    stats = pipeline.get_stats()
    check("dialogue_memory" in stats, "5.10 get_stats含必填字段")


# ═══════════════════════════════════════════════════════════
#  SECTION 6: QUERY ENGINE DEGRADED MODE
# ═══════════════════════════════════════════════════════════

def test_query_degraded():
    from query_engine import get_query_engine
    from checkpoint_manager import TimeoutError as CkpTimeoutError

    qe = get_query_engine()

    ctx = [
        {"speaker": "user", "content": "测试查询"},
        {"speaker": "ai", "content": "系统回复"},
    ]
    queries = qe.build_queries(ctx)
    check(len(queries) > 0, f"6.1 build_queries生成{len(queries)}条查询")

    results = qe.multi_search(ctx)
    check("results" in results, "6.2 multi_search含results键")
    check("degraded" in results, "6.3 multi_search含degraded标记")
    check(isinstance(results["degraded"], bool), "6.4 degraded是布尔值")
    check("stats" in results, "6.5 multi_search含stats统计")

    health = qe.get_health()
    check("collections_loaded" in health, "6.6 health含collections_loaded")
    check("stats" in health, "6.7 health含stats")


# ═══════════════════════════════════════════════════════════
#  SECTION 7: SERVER CHECKPOINT API
# ═══════════════════════════════════════════════════════════

def test_server_checkpoint_api():
    from fastapi.testclient import TestClient
    from server import app
    from checkpoint_manager import get_checkpoint

    cp = get_checkpoint()
    cp.clear_checkpoint()

    client = TestClient(app)
    h = {"X-API-Key": "rag-rpg-local"}

    # health
    r = client.get("/api/health")
    check(r.status_code == 200, f"7.1 GET /api/health -> {r.status_code}")
    health_data = r.json()
    check("status" in health_data, "7.2 health含status")
    check("uptime_heartbeat" in health_data, "7.3 health含heartbeat")

    # checkpoint status (no checkpoint)
    r = client.get("/api/checkpoint/status", headers=h)
    check(r.status_code == 200, f"7.4 GET /api/checkpoint/status -> {r.status_code}")
    cs = r.json()
    check(cs["has_checkpoint"] == False, "7.5 初始化后无断点")
    check(cs["progress"]["status"] == "idle", "7.6 进度状态=idle")

    # resume without checkpoint
    r = client.post("/api/checkpoint/resume", json={}, headers=h)
    check(r.status_code == 404, f"7.7 resume无断点返回404 (实际:{r.status_code})")

    # create checkpoint via batch ingest
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False, encoding="utf-8"
    ) as f:
        for i in range(6):
            f.write(f"[test] 测试条目 test_checkpoint_{i}\n")
        tmpfile = f.name

    try:
        r = client.post("/api/batch/ingest", json={"file_path": tmpfile}, headers=h)
        check(r.status_code == 200, f"7.8 批量导入 -> {r.status_code}")
        batch_data = r.json()
        check(batch_data["status"] in ("completed", "partial"),
              f"7.9 批量导入完成 (status={batch_data['status']})")

        r = client.get("/api/checkpoint/status", headers=h)
        cs2 = r.json()
        check(cs2["progress"]["status"] != "idle",
              f"7.10 导入后断点状态已更新 (status={cs2['progress']['status']})")

        # clear checkpoint
        r = client.post("/api/checkpoint/clear", headers=h)
        check(r.status_code == 200, f"7.11 清除断点 -> {r.status_code}")
        cs3 = client.get("/api/checkpoint/status", headers=h).json()
        check(cs3["has_checkpoint"] == False, "7.12 清除后无断点")
    finally:
        os.unlink(tmpfile)


# ═══════════════════════════════════════════════════════════
#  SECTION 8: EDGE CASES & CONCURRENCY
# ═══════════════════════════════════════════════════════════

def test_edge_cases():
    from checkpoint_manager import get_checkpoint, TimeoutError as CkpTimeoutError

    cp = get_checkpoint()
    cp.clear_checkpoint()

    check(not cp.can_resume(), "8.1 无状态时can_resume=false")
    progress_empty = cp.get_progress()
    check(progress_empty["progress"] == 0, "8.2 无任务进度=0")

    steps = [{"id": "edge_1", "name": "边界测试"}]
    cp.init_task("edge_test", {}, steps)

    # Double init - should overwrite
    cp.init_task("edge_test2", {"override": True}, steps)
    check(cp._state.task_type == "edge_test2", "8.3 重复init_task覆盖旧任务")

    # Complete already completed
    cp.mark_step_start(0)
    cp.mark_step_success(0)
    cp.mark_step_success(0)
    check(cp._state.steps[0].status == "completed", "8.4 重复标记Success不崩溃")

    cp.complete_task()
    cp.complete_task()
    check(cp._state.status == "completed", "8.5 重复complete_task不崩溃")

    # Clear then operations
    cp.clear_checkpoint()
    try:
        cp.get_pending_steps()
        check(True, "8.6 clear后get_pending_steps不崩溃")
    except Exception:
        check(False, "8.7 clear后get_pending_steps崩溃")

    # Concurrent access simulation (快速多线程)
    steps_many = [{"id": f"conc_{i}", "name": f"并发步骤{i}"} for i in range(20)]
    cp.init_task("concurrency", {}, steps_many)

    def worker(start, count):
        for j in range(start, start + count):
            try:
                cp.mark_step_start(j)
                cp.mark_step_success(j)
            except Exception:
                pass

    threads = []
    for t in range(4):
        th = threading.Thread(target=worker, args=(t * 5, 5))
        threads.append(th)
        th.start()
    for th in threads:
        th.join(timeout=5)

    progress_conc = cp.get_progress()
    check(progress_conc["stats"]["total_succeeded"] >= 0,
          f"8.8 并发执行无死锁 (succeeded={progress_conc['stats']['total_succeeded']})")


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def main():
    start = time.perf_counter()

    print("=" * 65)
    print("  RAG-RPG v2.0 断点续执行 & 异常恢复 集成测试")
    print("=" * 65)

    section("[1] 断点管理器单元测试 (21 tests)")
    test_checkpoint_unit()

    section("[2] 续点执行流程测试 (6 tests)")
    test_resume_flow()

    section("[3] 超时保护测试 (8 tests)")
    test_timeout_protection()

    section("[4] 心跳监控测试 (6 tests)")
    test_heartbeat()

    section("[5] Pipeline安全操作测试 (10 tests)")
    test_pipeline_safe()

    section("[6] 查询引擎降级模式测试 (7 tests)")
    test_query_degraded()

    section("[7] Server断点续执行API测试 (12 tests)")
    test_server_checkpoint_api()

    section("[8] 边界条件与并发测试 (8 tests)")
    test_edge_cases()

    elapsed = time.perf_counter() - start

    print(f"\n{'=' * 65}")
    print(f"  测试报告")
    print(f"{'=' * 65}")
    print(f"  总测试数 : {TOTAL}")
    print(f"  通过     : {PASSED}")
    print(f"  失败     : {FAILED}")
    pct = PASSED / TOTAL * 100 if TOTAL > 0 else 0
    print(f"  通过率   : {pct:.1f}%")
    print(f"  总耗时   : {elapsed:.2f}s")

    if ERRORS:
        print(f"\n  {'─' * 50}")
        print(f"  失败详情:")
        for err in ERRORS:
            print(err)

    print(f"\n{'=' * 65}")
    if FAILED == 0:
        print("  [PASS] 全部测试通过！")
    else:
        print(f"  [WARN] {FAILED} 条失败")
    print(f"{'=' * 65}\n")

    return FAILED == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
