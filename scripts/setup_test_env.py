"""
RAG-RPG 测试环境初始化与一键重建工具

在新设备上克隆项目后，一键完成：
  1. 环境检测（Python版本 / 依赖包）
  2. 数据文件结构检查（不校验内容）
  3. ChromaDB 重建（从用户自己的 JSON 初始化）
  4. 运行测试套件
  5. 生成环境快照报告

用法：
  python scripts/setup_test_env.py                  # 全流程（检查→重建→测试）
  python scripts/setup_test_env.py --check          # 仅检查环境，不重建
  python scripts/setup_test_env.py --rebuild        # 重建 ChromaDB
  python scripts/setup_test_env.py --test           # 仅运行测试
  python scripts/setup_test_env.py --report         # 仅生成环境报告
"""
import sys
import os
import json
import subprocess
import platform
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
REQUIREMENTS_FILE = BASE_DIR / "requirements.txt"
REQUIRED_PYTHON = (3, 10)
RECOMMENDED_PACKAGES = {
    "chromadb": "chromadb",
    "sentence_transformers": "sentence_transformers",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}

CHECKS = []


def record(name: str, status: str, detail: str = ""):
    CHECKS.append({"name": name, "status": status, "detail": detail})
    icon = {"pass": "✅", "fail": "❌", "warn": "⚠️", "info": "ℹ️"}.get(status, "•")
    print(f"  {icon} [{name}] {detail}")


def check_python():
    ver = sys.version_info
    ok = ver.major == REQUIRED_PYTHON[0] and ver.minor >= REQUIRED_PYTHON[1]
    label = f"{ver.major}.{ver.minor}.{ver.micro}"
    record("Python版本", "pass" if ok else "fail",
           f"{label} (要求 ≥{REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})")


def check_packages():
    for pkg_name, mod_name in RECOMMENDED_PACKAGES.items():
        try:
            mod = __import__(mod_name)
            ver = getattr(mod, "__version__", "unknown")
            record("依赖包", "pass", f"{pkg_name} v{ver}")
        except ImportError:
            record("依赖包", "fail", f"{pkg_name} 未安装 (pip install {pkg_name})")


def check_data_files():
    data_dir = BASE_DIR / "data"
    if not data_dir.exists():
        record("数据目录", "warn", "data/ 目录不存在")
        return

    json_files = list(data_dir.rglob("*.json"))
    if not json_files:
        record("数据文件", "warn", "data/ 下没有 JSON 文件")
        return

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                json.load(f)
            record("数据文件", "pass", str(json_file.relative_to(BASE_DIR)))
        except json.JSONDecodeError as e:
            record("数据文件", "fail",
                   f"{json_file.relative_to(BASE_DIR)}: {e}")


def check_chromadb():
    try:
        import chromadb
        from config import CHROMA_PATH
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        from config import COLLECTION_SKILLS
        try:
            col = client.get_collection(name=COLLECTION_SKILLS)
            count = col.count()
            record("向量数据库", "pass" if count > 0 else "warn",
                   f"skills 集合: {count} 条")
        except Exception:
            record("向量数据库", "warn",
                   f"skills 集合未初始化（需运行 ingest_initial.py）")
    except ImportError:
        record("向量数据库", "fail", "chromadb 未安装")


def rebuild_chromadb() -> bool:
    ingest_script = BASE_DIR / "scripts" / "ingest_initial.py"
    if not ingest_script.exists():
        record("重建向量库", "fail", "ingest_initial.py 不存在")
        return False

    print("  ⏳ 正在从 JSON 重建 ChromaDB 向量库...")
    result = subprocess.run(
        [sys.executable, str(ingest_script)],
        capture_output=True, text=True, cwd=str(BASE_DIR)
    )
    if result.returncode == 0:
        last_line = [l for l in result.stdout.strip().split("\n") if l][-1]
        record("重建向量库", "pass", last_line)
        return True
    else:
        record("重建向量库", "fail",
               result.stderr.strip()[:200])
        return False


def run_tests() -> bool:
    test_files = [
        BASE_DIR / "tests" / "test_suite.py",
        BASE_DIR / "tests" / "test_checkpoint_resume.py",
    ]
    all_pass = True
    for test_file in test_files:
        if not test_file.exists():
            record("运行测试", "warn", f"{test_file.name} 不存在，跳过")
            continue
        print(f"\n  ⏳ 运行 {test_file.name}...")
        result = subprocess.run(
            [sys.executable, str(test_file)],
            capture_output=True, text=True, cwd=str(BASE_DIR)
        )
        output = result.stdout.strip()
        # 提取通过率信息
        for line in output.split("\n"):
            if "通过率" in line or "全部测试通过" in line:
                record("运行测试", "pass" if result.returncode == 0 else "fail",
                       f"{test_file.name}: {line.strip()}")
                break
        else:
            record("运行测试", "pass" if result.returncode == 0 else "fail",
                   f"{test_file.name} (exit={result.returncode})")
        if result.returncode != 0:
            all_pass = False
    return all_pass


def generate_report():
    from config import (
        API_HOST, API_PORT, MODEL_NAME, COLLECTION_SKILLS,
        COLLECTION_MEMORY, COLLECTION_DIALOGUE,
    )

    report = {
        "generated_at": datetime.now().isoformat(),
        "platform": platform.system(),
        "python_version": sys.version,
        "python_executable": sys.executable,
        "project_root": str(BASE_DIR),
        "config_snapshot": {
            "API_HOST": API_HOST,
            "API_PORT": API_PORT,
            "MODEL_NAME": MODEL_NAME,
            "COLLECTION_SKILLS": COLLECTION_SKILLS,
            "COLLECTION_MEMORY": COLLECTION_MEMORY,
            "COLLECTION_DIALOGUE": COLLECTION_DIALOGUE,
        },
        "checks": CHECKS,
    }

    report_file = BASE_DIR / "environment_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    record("环境报告", "info", f"已保存至 {report_file}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG-RPG 环境初始化工具")
    parser.add_argument("--check", action="store_true", help="仅检查环境")
    parser.add_argument("--rebuild", action="store_true", help="仅重建 ChromaDB")
    parser.add_argument("--test", action="store_true", help="仅运行测试")
    parser.add_argument("--report", action="store_true", help="仅生成环境报告")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════╗")
    print("║  RAG-RPG 环境初始化工具 v2.0                       ║")
    print(f"║  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}                          ║")
    print("║  你的数据，你的角色，你的世界——直接初始化即可         ║")
    print("╚════════════════════════════════════════════════════╝")

    # ── 模式选择 ──
    mode_single = args.check or args.rebuild or args.test or args.report
    do_check = args.check or (not mode_single)
    do_rebuild = args.rebuild or (not mode_single)
    do_test = args.test or (not mode_single)
    do_report = args.report or (not mode_single)

    # ── 1. 环境检测 ──
    if do_check:
        print(f"\n{'─' * 50}")
        print("  [1] 环境检测")
        print(f"{'─' * 50}")
        check_python()
        check_packages()
        check_data_files()
        check_chromadb()

    # ── 2. 重建向量库 ──
    if do_rebuild:
        print(f"\n{'─' * 50}")
        print("  [2] ChromaDB 重建")
        print(f"{'─' * 50}")
        rebuild_chromadb()

    # ── 3. 运行测试 ──
    if do_test:
        print(f"\n{'─' * 50}")
        print("  [3] 运行测试套件")
        print(f"{'─' * 50}")
        run_tests()

    # ── 4. 环境报告 ──
    if do_report:
        print(f"\n{'─' * 50}")
        print("  [4] 环境报告")
        print(f"{'─' * 50}")
        generate_report()

    # ── 汇总 ──
    passed = sum(1 for c in CHECKS if c["status"] == "pass")
    failed = sum(1 for c in CHECKS if c["status"] == "fail")
    warned = sum(1 for c in CHECKS if c["status"] == "warn")

    print(f"\n{'=' * 50}")
    print(f"  汇总: ✅ {passed}  ⚠️ {warned}  ❌ {failed}")
    print(f"{'=' * 50}")

    if failed:
        print("\n📌 有检查项未通过，请根据上方提示修复。")
        print("   常见修复: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("\n✅ 环境就绪！现在可以正常使用 RAG-RPG。")


if __name__ == "__main__":
    main()
