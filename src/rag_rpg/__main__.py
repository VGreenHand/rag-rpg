"""rag-rpg 包入口：python -m rag_rpg 启动主服务"""
import sys
from pathlib import Path

_root = Path(__file__).resolve().parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import uvicorn
from config import API_HOST, API_PORT


def main():
    """启动 RAG-RPG 服务"""
    from server import app
    uvicorn.run("server:app", host=API_HOST, port=API_PORT, reload=False, log_level="info")


if __name__ == "__main__":
    sys.exit(main())
