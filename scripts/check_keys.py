"""
在你运行 ingest_initial.py 构建自己的向量库后，
用这个工具确认 ChromaDB 中的数据是否与你的 JSON 源文件一致。

场景：同一用户的跨设备开发
  ├─ 设备 A: 放入自己的 JSON → ingest_initial.py → ChromaDB 就绪
  ├─ 设备 B: git pull → 放入同一份 JSON → ingest_initial.py → check_keys.py
  └─ 验证: ChromaDB 中的条目与 JSON 中的 entries 完全匹配

用法：
  python scripts/check_keys.py                    # 标准校验
  python scripts/check_keys.py --verbose          # 详细输出每条记录
  python scripts/check_keys.py --full             # 全量检查（含所有集合）
  python scripts/check_keys.py --json             # JSON 格式输出报告
"""
import sys
import os
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

COLLECTION_NAME = "character_skills"
JSON_FILE = "data/CharacterInfo/Characterdesign.json"
ALL_COLLECTIONS = ["character_skills", "dialogue_memory", "my_rag_memory", "plot_state"]

FAILURES = []
WARNINGS = []


def fail(msg: str):
    FAILURES.append(msg)


def warn(msg: str):
    WARNINGS.append(msg)


def check_json_consistency(verbose: bool) -> dict:
    """校验 Characterdesign.json 的完整性"""
    json_path = Path(__file__).parent.parent / JSON_FILE
    if not json_path.exists():
        warn(f"JSON 文件不存在: {json_path}（无预设数据，跳过）")
        return {}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    entries = data.get("entries", {})
    json_keys = set(entries.keys())

    if verbose:
        print(f"\n📄 Characterdesign.json 结构:")
        print(f"   顶级键: {list(data.keys())}")
        print(f"   条目数: {len(entries)}")
        print(f"   条目 keys: {sorted(json_keys, key=int)}")

    return data


def check_chromadb_consistency(verbose: bool) -> dict:
    """连接 ChromaDB 获取 character_skills 集合数据"""
    try:
        import chromadb
    except ImportError:
        fail("chromadb 未安装，无法进行数据库校验")
        return {}

    from config import CHROMA_PATH

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    try:
        collection = client.get_collection(name=COLLECTION_NAME)
    except Exception:
        fail(f"ChromaDB 中不存在集合 '{COLLECTION_NAME}'，请先运行 ingest_initial.py")
        return {}

    chroma_count = collection.count()

    if verbose:
        print(f"\n🗄️  ChromaDB 集合 '{COLLECTION_NAME}':")
        print(f"   条目数: {chroma_count}")

    if chroma_count == 0:
        fail(f"集合 '{COLLECTION_NAME}' 为空")

    all_data = collection.get()
    return {
        "client": client,
        "collection": collection,
        "ids": set(all_data["ids"]),
        "all_data": all_data,
        "count": chroma_count,
    }


def check_id_key_mismatch(chroma_data: dict, verbose: bool):
    """检查 ChromaDB 中 id 与 entry_key 的匹配关系"""
    all_data = chroma_data["all_data"]
    mismatch_count = 0

    if verbose:
        print(f"\n🔍 ChromaDB id ↔ entry_key 匹配检查:")

    for i, (cid, meta) in enumerate(zip(all_data["ids"], all_data["metadatas"])):
        entry_key = meta.get("entry_key", "N/A")
        uid = meta.get("uid", "N/A")
        content_preview = all_data["documents"][i][:80].replace("\n", " ")
        status = "✅" if cid == entry_key else "❌"
        if cid != entry_key:
            mismatch_count += 1
            fail(f"id/entry_key 不匹配: id={cid} | entry_key={entry_key} | uid={uid}")
        if verbose or cid != entry_key:
            print(f"   {status} id={cid:>4} | entry_key={entry_key:>4} | "
                  f"uid={uid:>4} | {content_preview}...")

    if mismatch_count == 0 and verbose:
        print("   所有条目的 id 与 entry_key 均一致 ✅")


def check_json_vs_chromadb(json_data: dict, chroma_data: dict):
    """校验 JSON 与 ChromaDB 之间的 keys 一致性"""
    entries = json_data.get("entries", {})
    json_keys = set(entries.keys())
    chroma_ids = chroma_data["ids"]
    json_keys_by_entry_key = set()

    for key, entry in entries.items():
        key_list = entry.get("key", [])
        if key_list and isinstance(key_list, list):
            json_keys_by_entry_key.add(key_list[0])

    all_json_identifiers = json_keys | json_keys_by_entry_key

    only_in_json = all_json_identifiers - chroma_ids
    only_in_chroma = chroma_ids - all_json_identifiers

    if only_in_json:
        fail(f"仅在 JSON 中但不在 ChromaDB: {sorted(only_in_json)}")
    if only_in_chroma:
        fail(f"仅在 ChromaDB 中但不在 JSON: {sorted(only_in_chroma)}")

    if not only_in_json and not only_in_chroma:
        print("   ✅ JSON ↔ ChromaDB: 条目完全一致")
    else:
        print(f"   ⚠️  JSON ({len(all_json_identifiers)} 标识) ↔ "
              f"ChromaDB ({len(chroma_ids)} 条)")


def check_all_collections(verbose: bool):
    """检查所有 ChromaDB 集合的状态"""
    try:
        import chromadb
    except ImportError:
        return

    from config import CHROMA_PATH
    client = chromadb.PersistentClient(path=CHROMA_PATH)

    if verbose:
        print(f"\n📊 所有集合状态:")

    for col_name in ALL_COLLECTIONS:
        try:
            col = client.get_collection(name=col_name)
            count = col.count()
            status = "✅" if count > 0 else "⚠️ 空"
            if verbose:
                print(f"   {status} {col_name}: {count} 条")
        except Exception:
            if verbose:
                print(f"   ❌ {col_name}: 无法访问")


def main():
    parser = argparse.ArgumentParser(description="ChromaDB ↔ JSON 一致性校验")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--full", action="store_true", help="全量校验（含所有集合）")
    parser.add_argument("--json", action="store_true", help="JSON 格式输出报告")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════╗")
    print("║  RAG-RPG ChromaDB ↔ JSON 一致性校验                ║")
    print("╚════════════════════════════════════════════════════╝")

    json_data = check_json_consistency(args.verbose)
    chroma_data = check_chromadb_consistency(args.verbose)

    if json_data and chroma_data:
        print(f"\n{'─' * 55}")
        print("  [校验 1/3] JSON ↔ ChromaDB 条目一致性")
        print(f"{'─' * 55}")
        check_json_vs_chromadb(json_data, chroma_data)

        print(f"\n{'─' * 55}")
        print("  [校验 2/3] id ↔ entry_key 匹配检查")
        print(f"{'─' * 55}")
        check_id_key_mismatch(chroma_data, args.verbose)

    if args.full:
        print(f"\n{'─' * 55}")
        print("  [校验 3/3] 全量集合检查")
        print(f"{'─' * 55}")
        check_all_collections(args.verbose)

    if args.json:
        report = {
            "summary": {
                "failures": len(FAILURES),
                "warnings": len(WARNINGS),
            },
            "failures": FAILURES,
            "warnings": WARNINGS,
            "suggestion": (
                "运行以下命令重建向量库以修复一致性问题:\n"
                "  python scripts/ingest_initial.py\n"
                "然后重新运行本校验:\n"
                "  python scripts/check_keys.py"
            ),
        }
        print(f"\n{json.dumps(report, ensure_ascii=False, indent=2)}")

    print(f"\n{'=' * 55}")
    print(f"  结果: 失败={len(FAILURES)}  警告={len(WARNINGS)}")
    print(f"{'=' * 55}")
    for item in FAILURES:
        print(f"  ❌ {item}")
    for item in WARNINGS:
        print(f"  ⚠️  {item}")

    if not FAILURES:
        print("✅ 所有校验通过！数据一致。")
    else:
        print(f"\n📌 建议修复: 运行 'python scripts/ingest_initial.py' 重建向量库")
        sys.exit(1)


if __name__ == "__main__":
    main()
