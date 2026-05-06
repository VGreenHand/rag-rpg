"""
RAG-RPG 记忆引擎服务端：为 SillyTavern 提供对话记忆和剧情约束 API
启动: python server.py  或  uvicorn server:app --host 127.0.0.1 --port 8765
"""
import logging
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import API_HOST, API_PORT, API_KEY
from pipeline import get_pipeline
from query_engine import get_query_engine
from constraint_engine import get_constraint_engine

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


class SkillUpdateRequest(BaseModel):
    entry_key: str = Field(...)
    new_content: str = Field(...)


class FeedbackRequest(BaseModel):
    entry_type: str = Field(...)
    was_used: bool = Field(default=True)


# ─── 认证依赖 ───────────────────────────────────────────────

def verify_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="无效的 API Key")
    return x_api_key


# ─── 应用生命周期 ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("正在初始化 RAG-RPG 记忆引擎...")
    pipeline = get_pipeline()
    stats = pipeline.get_stats()
    logger.info(f"向量库状态: {stats}")
    logger.info(f"服务已就绪 → http://{API_HOST}:{API_PORT}")
    yield
    logger.info("RAG-RPG 服务关闭")


app = FastAPI(
    title="RAG-RPG Memory Engine",
    description="SillyTavern 对话记忆与剧情约束引擎",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── API 端点 ───────────────────────────────────────────────

@app.get("/api/status")
async def get_status(_: str = Depends(verify_api_key)):
    """获取引擎运行状态"""
    pipeline = get_pipeline()
    return {
        "status": "running",
        "version": "1.0.0",
        "vector_db": pipeline.get_stats(),
    }


@app.post("/api/dialogue/ingest")
async def ingest_dialogue(
    turn: DialogueTurn,
    _: str = Depends(verify_api_key),
):
    """
    接收单轮对话并全量处理：
    1. 文本清洗与关键信息提取
    2. 写入按日期命名的TXT文件
    3. 向量化存入 ChromaDB
    4. 生成 batch 格式条目
    """
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
    """
    上下文感知查询：
    1. 分析对话上下文生成多角度查询
    2. 在技能库/记忆库/对话库中检索
    3. 返回格式化约束文本
    """
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
    }

    if req.generate_constraint and search_results["results"]:
        ce = get_constraint_engine()
        constraint = ce.generate_constraints(
            search_results=search_results,
            dialogue_context=req.context,
        )
        response["constraint_text"] = constraint

    return response


@app.post("/api/batch/ingest")
async def batch_ingest(
    req: BatchIngestRequest = None,
    _: str = Depends(verify_api_key),
):
    """批量导入标记TXT文件到向量库"""
    pipeline = get_pipeline()
    result = pipeline.process_batch_txt(
        file_path=req.file_path if req else None
    )
    return result


@app.post("/api/skill/update")
async def update_skill(
    req: SkillUpdateRequest,
    _: str = Depends(verify_api_key),
):
    """更新技能条目内容并重新向量化"""
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
    """反馈某类约束是否被AI采用，用于调整查询权重"""
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
    logger.info(f"启动 RAG-RPG 服务 → http://{API_HOST}:{API_PORT}")
    uvicorn.run(
        "server:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
