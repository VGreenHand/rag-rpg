"""
独立模型嵌入服务：将 SentenceTransformer 加载到独立进程，通过 HTTP 提供嵌入接口。
主服务重启时无需重新加载模型，实现秒级重启。

启动: python embedding_service.py
端口: 8766（可在 config.py 中修改）
"""
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from config import MODEL_NAME, EMBEDDING_SERVICE_HOST, EMBEDDING_SERVICE_PORT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("rag-rpg.embedding-service")

model: SentenceTransformer = None


class EmbedRequest(BaseModel):
    texts: list[str] = Field(..., description="待编码文本列表")


class EmbedResponse(BaseModel):
    embeddings: list[list[float]] = Field(..., description="嵌入向量列表")
    model: str = Field(...)
    dimensions: int = Field(...)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info(f"正在加载嵌入模型: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    logger.info(f"模型加载完成，向量维度: {model.get_sentence_embedding_dimension()}")
    yield
    logger.info("嵌入服务关闭")


app = FastAPI(
    title="RAG-RPG Embedding Service",
    description="为 RAG-RPG 主服务提供文本嵌入的独立模型服务",
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


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest):
    embeddings = model.encode(req.texts).tolist()
    return EmbedResponse(
        embeddings=embeddings,
        model=MODEL_NAME,
        dimensions=len(embeddings[0]) if embeddings else 0,
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "dimensions": model.get_sentence_embedding_dimension() if model else 0,
    }


if __name__ == "__main__":
    logger.info(
        f"启动嵌入服务 → http://{EMBEDDING_SERVICE_HOST}:{EMBEDDING_SERVICE_PORT}"
    )
    uvicorn.run(
        "embedding_service:app",
        host=EMBEDDING_SERVICE_HOST,
        port=EMBEDDING_SERVICE_PORT,
        reload=False,
        log_level="info",
    )
