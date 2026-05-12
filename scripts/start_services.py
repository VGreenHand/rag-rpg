"""
一键启动 RAG-RPG 全部服务
  1. 启动嵌入服务 embedding_service（后台，永不重启）
  2. 等待服务就绪（健康检查）
  3. 启动主服务 server
  4. 运行时按 r + Enter 热重启主服务（改完代码立刻测试）
  5. Ctrl+C 或输入 q + Enter 停止所有服务

用法:
  conda activate rag-rpg
  python scripts/start_services.py
"""
import sys
import os
import time
import signal
import subprocess
import urllib.request
import urllib.error
import json
import logging
import threading
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("start-services")

BASE_DIR = Path(__file__).parent.parent
EMBEDDING_SERVICE_URL = "http://127.0.0.1:8766"
EMBEDDING_HEALTH_URL = f"{EMBEDDING_SERVICE_URL}/health"
MAIN_SERVICE_URL = "http://127.0.0.1:8765"
MAIN_HEALTH_URL = f"{MAIN_SERVICE_URL}/api/health"
HEALTH_CHECK_INTERVAL = 3
EMBEDDING_TIMEOUT = 120
SERVER_TIMEOUT = 30

# 数据统计相关配置
CHROMA_DB_PATH = BASE_DIR / "chroma_db"
DIALOGUE_DIR = BASE_DIR / "dialogues"
CHARACTER_DIR = BASE_DIR / "data" / "CharacterInfo"
WORLDINFO_DIR = BASE_DIR / "data" / "WorldInfo"
COLLECTION_NAMES = {
    "dialogue_memory": "对话记忆",
    "character_skills": "角色技能",
    "my_rag_memory": "长期记忆",
    "plot_state": "剧情状态",
}

python = sys.executable
server_proc: subprocess.Popen = None
emb_proc: subprocess.Popen = None
keep_running = True


def collect_data_stats() -> dict:
    """收集数据记录统计信息"""
    stats = {}

    # 扫描所有 profile 目录
    profile_dirs = [CHROMA_DB_PATH]
    if CHROMA_DB_PATH.exists():
        for item in CHROMA_DB_PATH.iterdir():
            if item.is_dir() and (item / "chroma.sqlite3").exists():
                profile_dirs.append(item)

    # ChromaDB 集合统计
    try:
        import chromadb
        for profile_path in profile_dirs:
            profile_name = profile_path.name if profile_path != CHROMA_DB_PATH else "default"
            try:
                client = chromadb.PersistentClient(path=str(profile_path))
                for col_key, col_label in COLLECTION_NAMES.items():
                    try:
                        col = client.get_collection(name=col_key)
                        count = col.count()
                        label = f"{col_label}({profile_name})"
                        stats[label] = stats.get(label, 0) + count
                    except Exception:
                        label = f"{col_label}({profile_name})"
                        stats.setdefault(label, 0)
            except Exception:
                pass
    except Exception:
        for col_key, col_label in COLLECTION_NAMES.items():
            stats[col_label] = -1

    # 对话 TXT 文件统计
    txt_count = 0
    if DIALOGUE_DIR.exists():
        txt_count = len(list(DIALOGUE_DIR.rglob("*.txt")))
    stats["对话文件"] = txt_count

    # 角色信息文件统计
    char_count = 0
    if CHARACTER_DIR.exists():
        char_count = len(list(CHARACTER_DIR.glob("*.json")))
    stats["角色档案"] = char_count

    # 世界观信息文件统计
    world_count = 0
    if WORLDINFO_DIR.exists():
        world_count = len(list(WORLDINFO_DIR.glob("*.json")))
    stats["世界观"] = world_count

    return stats


def print_data_stats(stats: dict):
    """打印数据统计表格"""
    print("+--------------------------------------------------+")
    print("|  数据记录统计                                      |")
    print("+--------------------------------------------------+")
    for key, value in stats.items():
        if value == -1:
            val_str = "N/A"
        else:
            val_str = str(value)
        label = f"  {key}"
        print(f"| {label:<22} {val_str:>20} |")
    print("+--------------------------------------------------+")


def cleanup(signum=None, frame=None):
    global keep_running
    keep_running = False
    logger.info("正在停止所有服务...")
    for proc in [server_proc, emb_proc]:
        if proc and proc.poll() is None:
            proc.terminate()
    for proc in [server_proc, emb_proc]:
        if proc and proc.poll() is None:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    logger.info("所有服务已停止")


def wait_for_service(url: str, name: str, timeout: int = 120) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            resp = urllib.request.urlopen(f"{url}", timeout=3)
            if resp.status < 300:
                elapsed = time.time() - start
                logger.info(f"{name} 就绪（耗时 {elapsed:.0f}秒）")
                return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionResetError):
            pass
        time.sleep(HEALTH_CHECK_INTERVAL)
    logger.error(f"{name} 启动超时（{timeout}秒）")
    return False


def read_output(proc: subprocess.Popen, prefix: str = ""):
    try:
        for line in iter(proc.stdout.readline, ""):
            if line:
                print(f"{prefix}{line}", end="", flush=True)
    except ValueError:
        pass


def start_server() -> subprocess.Popen:
    global server_proc
    logger.info("正在启动主服务...")
    p = subprocess.Popen(
        [python, "server.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True,
    )
    reader = threading.Thread(target=read_output, args=(p,), daemon=True)
    reader.start()

    if not wait_for_service(MAIN_HEALTH_URL, "主服务", SERVER_TIMEOUT):
        logger.error("主服务启动失败，退出")
        p.terminate()
        return None
    server_proc = p
    return p


def stop_server():
    global server_proc
    if server_proc and server_proc.poll() is None:
        logger.info("正在停止主服务...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_proc.wait(timeout=3)
        logger.info("主服务已停止")
        server_proc = None


def restart_server():
    stop_server()
    print()
    logger.info("重新加载 server.py ...")
    return start_server()


def listen_input():
    global keep_running
    while keep_running:
        try:
            cmd = input().strip().lower()
            if cmd == "r":
                print()
                logger.info(">>> 热重启主服务...")
                if restart_server() is None:
                    logger.error("主服务重启失败")
                else:
                    logger.info(">>> 主服务重启完成")
                print()
                data_stats = collect_data_stats()
                print_data_stats(data_stats)
                print()
                print("+--------------------------------------------------+")
                print("|  按 r + Enter 重启主服务                          |")
                print("|  按 q + Enter 关闭所有服务                       |")
                print("+--------------------------------------------------+")
            elif cmd == "q":
                logger.info("用户请求退出")
                cleanup()
                sys.exit(0)
        except (EOFError, ValueError):
            break


def main():
    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    global emb_proc, keep_running
    os.chdir(BASE_DIR)

    print()
    print("+--------------------------------------------------+")
    print("|        RAG-RPG 服务一键启动                        |")
    print("+--------------------------------------------------+")
    print("|  嵌入服务 -> http://127.0.0.1:8766                 |")
    print("|  主服务   -> http://127.0.0.1:8765                 |")
    print("+--------------------------------------------------+")
    print()

    logger.info("正在启动嵌入服务...")
    logger.info(f"首次启动可能需下载模型，请耐心等待（最长 {EMBEDDING_TIMEOUT}秒）")
    emb_proc = subprocess.Popen(
        [python, "embedding_service.py"],
        stdout=None,
        stderr=None,
    )

    if not wait_for_service(EMBEDDING_HEALTH_URL, "嵌入服务", EMBEDDING_TIMEOUT):
        logger.error("嵌入服务启动失败，退出")
        cleanup()
        sys.exit(1)

    if start_server() is None:
        cleanup()
        sys.exit(1)

    print()
    print("+--------------------------------------------------+")
    print("|  所有服务已就绪 (OK)                               |")
    print("|                                                   |")
    print("|  按 r + Enter  只重启主服务（改代码后测试用）      |")
    print("|  按 q + Enter  关闭所有服务                       |")
    print("|  也可以直接 Ctrl+C                                 |")
    print("+--------------------------------------------------+")
    print()

    data_stats = collect_data_stats()
    print_data_stats(data_stats)
    print()

    input_thread = threading.Thread(target=listen_input, daemon=True)
    input_thread.start()

    try:
        while keep_running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        cleanup()


if __name__ == "__main__":
    main()
