"""
RAG-RPG 记忆引擎服务端：为 SillyTavern 提供对话记忆和剧情约束 API
v2.0: 断点续执行 + 健康检查 + 超时保护

启动: python server.py  或  uvicorn server:app --host 127.0.0.1 --port 8765
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT, API_KEY
from pipeline import get_pipeline
from query_engine import get_query_engine
from constraint_engine import get_constraint_engine
from checkpoint_manager import get_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag-rpg")


# ─── 请求/响应模型 ──────────────────────────────────────────

class DialogueTurn(BaseModel):
    speaker: str = Field(..., description="user 或 ai")
    name: str = Field(default="", description="说话者名称")
    content: str = Field(..., description="对话内容")
    turn: int = Field(default=0, description="对话轮次序号")
    timestamp: Optional[str] = Field(default=None)


class QueryRequest(BaseModel):
    context: list[dict] = Field(..., description="最近N轮对话上下文")
    max_results: int = Field(default=3)
    collections: Optional[list[str]] = Field(default=None)
    generate_constraint: bool = Field(default=True)


class BatchIngestRequest(BaseModel):
    file_path: Optional[str] = Field(default=None)
    resume: bool = Field(default=False)


class SkillUpdateRequest(BaseModel):
    entry_key: str = Field(...)
    new_content: str = Field(...)


class FeedbackRequest(BaseModel):
    entry_type: str = Field(...)
    was_used: bool = Field(default=True)


class IngestParams(BaseModel):
    file_path: Optional[str] = Field(default=None)
    resume: bool = Field(default=False)


# ─── 认证 ───────────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="无效的 API Key")
    return x_api_key


# ─── 生命周期 ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化 RAG-RPG 记忆引擎 v2.0 ...")
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    logger.info(f"向量库状态: {stats}")

    cp = get_checkpoint()
    if cp.can_resume():
        progress = cp.get_progress()
        logger.warning(
            f"检测到未完成的断点任务: {progress['execution_id']} "
            f"(进度: {progress['current_step']}/{progress['total_steps']})"
        )

    cp._start_heartbeat()
    logger.info(f"服务已就绪 → http://{API_HOST}:{API_PORT}")
    yield
    cp._stop_heartbeat.set()
    logger.info("RAG-RPG 服务关闭")


app = FastAPI(
    title="RAG-RPG Memory Engine",
    description="SillyTavern 对话记忆与剧情约束引擎 v2.0",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── 异常处理 ───────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"未捕获异常: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "error_type": type(exc).__name__,
            "detail": str(exc),
        },
    )


# ═══════════════════════════════════════════════════════════
#  健康检查与监控
# ═══════════════════════════════════════════════════════════

@app.get("/api/health")
async def health_check():
    """全面的系统健康检查"""
    pipeline = get_pipeline()
    qe = get_query_engine()
    cp = get_checkpoint()

    health = {
        "status": "healthy",
        "version": "2.0.0",
        "uptime_heartbeat": cp.is_alive(),
    }

    try:
        health["vector_db"] = pipeline.get_stats()
        health["query_engine"] = qe.get_health()
    except Exception as e:
        health["status"] = "degraded"
        health["error"] = str(e)

    if cp.can_resume():
        health["checkpoint"] = cp.get_progress()
        health["status"] = "needs_attention"

    return health


@app.get("/api/status")
async def get_status(_: str = Depends(verify_api_key)):
    pipeline = get_pipeline()
    cp = get_checkpoint()
    progress = cp.get_progress() if cp.can_resume() else None
    return {
        "status": "running",
        "version": "2.0.0",
        "vector_db": pipeline.get_stats(),
        "checkpoint": progress,
    }


# ═══════════════════════════════════════════════════════════
#  对话处理
# ═══════════════════════════════════════════════════════════

@app.post("/api/dialogue/ingest")
async def ingest_dialogue(
    turn: DialogueTurn,
    _: str = Depends(verify_api_key),
):
    pipeline = get_pipeline()
    result = pipeline.process_turn(
        speaker=turn.speaker,
        name=turn.name or ("用户" if turn.speaker == "user" else "AI"),
        content=turn.content,
        turn=turn.turn,
    )
    logger.info(
        f"已处理 Turn#{turn.turn} | {turn.speaker} | "
        f"关键术语: {result['key_terms_found'][:5]}"
    )
    return result


@app.post("/api/dialogue/query")
async def query_dialogue(
    req: QueryRequest,
    _: str = Depends(verify_api_key),
):
    engine = get_query_engine()
    search_results = engine.multi_search(
        dialogue_context=req.context,
        collections=req.collections,
        k=req.max_results,
    )

    response = {
        "raw_results": search_results["results"],
        "formatted": engine.format_for_llm(search_results),
        "constraint_text": "",
        "total_hits": search_results["total_hits"],
        "degraded": search_results.get("degraded", False),
    }

    if req.generate_constraint and search_results["results"]:
        ce = get_constraint_engine()
        constraint = ce.generate_constraints(
            search_results=search_results,
            dialogue_context=req.context,
        )
        response["constraint_text"] = constraint

    return response


# ═══════════════════════════════════════════════════════════
#  断点续执行 API
# ═══════════════════════════════════════════════════════════

@app.post("/api/batch/ingest")
async def batch_ingest(
    req: BatchIngestRequest = None,
    _: str = Depends(verify_api_key),
):
    file_path = req.file_path if req else None
    resume = req.resume if req else False

    pipeline = get_pipeline()
    result = pipeline.process_batch_txt(file_path=file_path, resume=resume)
    return result


@app.get("/api/checkpoint/status")
async def checkpoint_status(_: str = Depends(verify_api_key)):
    """查询当前断点/续点执行状态"""
    cp = get_checkpoint()
    return {
        "has_checkpoint": cp.can_resume(),
        "progress": cp.get_progress(),
        "heartbeat": cp.is_alive(),
    }


@app.post("/api/checkpoint/resume")
async def checkpoint_resume(
    req: IngestParams = None,
    _: str = Depends(verify_api_key),
):
    """从断点恢复批量导入"""
    cp = get_checkpoint()
    if not cp.can_resume():
        raise HTTPException(
            status_code=404,
            detail="没有可恢复的断点任务。请先执行批量导入。",
        )

    progress = cp.get_progress()
    logger.info(f"从断点恢复: {progress['execution_id']}")

    pipeline = get_pipeline()
    file_path = req.file_path if req else None
    result = pipeline.process_batch_txt(file_path=file_path, resume=True)
    return {
        "resumed_from": progress["execution_id"],
        "result": result,
    }


@app.post("/api/checkpoint/clear")
async def checkpoint_clear(_: str = Depends(verify_api_key)):
    """清除所有断点数据（放弃未完成任务）"""
    cp = get_checkpoint()
    progress = cp.get_progress()
    cp.clear_checkpoint()
    logger.info(f"已清除断点: {progress.get('execution_id', 'N/A')}")
    return {"status": "ok", "cleared_execution_id": progress.get("execution_id")}


# ═══════════════════════════════════════════════════════════
#  技能更新 & 反馈
# ═══════════════════════════════════════════════════════════

@app.post("/api/skill/update")
async def update_skill(
    req: SkillUpdateRequest,
    _: str = Depends(verify_api_key),
):
    import chromadb
    from sentence_transformers import SentenceTransformer
    from config import CHROMA_PATH, COLLECTION_SKILLS, MODEL_NAME

    model = SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_SKILLS)

    results = collection.get(where={"entry_key": req.entry_key})
    if not results["ids"]:
        raise HTTPException(
            status_code=404,
            detail=f"未找到 entry_key='{req.entry_key}' 的技能条目"
        )

    old_id = results["ids"][0]
    new_emb = model.encode(req.new_content).tolist()
    collection.update(
        ids=[old_id],
        documents=[req.new_content],
        embeddings=[new_emb],
    )

    pipeline = get_pipeline()
    pipeline.invalidate_terms_cache()

    return {
        "status": "ok",
        "entry_key": req.entry_key,
        "message": f"条目 '{req.entry_key}' 已更新",
    }


@app.post("/api/feedback")
async def submit_feedback(
    req: FeedbackRequest,
    _: str = Depends(verify_api_key),
):
    ce = get_constraint_engine()
    ce.update_feedback(req.entry_type, req.was_used)
    return {
        "status": "ok",
        "entry_type": req.entry_type,
        "new_weight": ce.get_weight(req.entry_type),
    }


# ─── 启动入口 ───────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动 RAG-RPG v2.0 服务 → http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
