"""
角色设定文件结构有效性校验工具

功能：检查用户提供的角色设定 JSON 文件的结构是否有效，确保其能被
  ingest_initial.py 正常处理。不校验具体数据内容——不同用户有各自
  的角色设定和世界观，只要结构合法即可。

校验项：
  1. JSON 文件可解析
  2. 顶级键完整性（至少包含 "entries"）
  3. 每个条目必需字段检查（uid, key, content）
  4. key 列表去重检查
  5. uid 唯一性检查
  6. [type:xxx] 标签格式检查

用法：
  python scripts/validate_character_consistency.py                              # 默认检查 data/CharacterInfo/ 下所有 JSON
  python scripts/validate_character_consistency.py --file data/你的角色文件.json   # 指定文件
  python scripts/validate_character_consistency.py --json                       # JSON 格式报告
"""
import sys
import os
import json
import re
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
TYPE_PATTERN = re.compile(r'\[type:(\w+)\]')
REQUIRED_ENTRY_FIELDS = ["uid", "key", "content"]

FAILURES = []
WARNINGS = []


def fail(check_name: str, detail: str):
    FAILURES.append(f"  ❌ [{check_name}] {detail}")


def warn(check_name: str, detail: str):
    WARNINGS.append(f"  ⚠️  [{check_name}] {detail}")


def validate_json_structure(data: dict, file_label: str):
    """校验 JSON 文件的结构完整性"""
    if not isinstance(data, dict):
        fail("顶级结构", f"{file_label}: 根节点必须是 JSON 对象 (dict)")
        return

    if "entries" not in data:
        fail("顶级结构", f"{file_label}: 缺少顶级键 'entries'")
        return

    entries = data["entries"]
    if not isinstance(entries, dict):
        fail("顶级结构", f"{file_label}: 'entries' 必须是字典类型 (dict)")
        return

    if len(entries) == 0:
        fail("顶级结构", f"{file_label}: 'entries' 为空")
        return

    print(f"\n📄 {file_label}:")
    print(f"   顶级键: {list(data.keys())}")
    print(f"   条目数: {len(entries)}")

    uid_set = set()
    all_keys = []
    all_type_tags = []

    for entry_key, entry in entries.items():
        # 必需字段检查
        missing = [f for f in REQUIRED_ENTRY_FIELDS if f not in entry]
        if missing:
            fail("必需字段", f"{file_label} 条目 {entry_key}: 缺少 {missing}")

        # uid 唯一性
        uid = entry.get("uid")
        if uid is not None:
            if uid in uid_set:
                fail("UID唯一性", f"{file_label} 条目 {entry_key}: uid={uid} 重复")
            uid_set.add(uid)

        # key 列表检查
        key_list = entry.get("key", [])
        if isinstance(key_list, list):
            all_keys.extend(key_list)
        else:
            fail("Key格式", f"{file_label} 条目 {entry_key}: key 不是列表类型")

        # content 中的 [type:xxx] 标签
        content = entry.get("content", "")
        match = TYPE_PATTERN.search(content)
        if match:
            tag = match.group(1).strip().lower()
            all_type_tags.append(tag)
        else:
            warn("Type标签",
                 f"{file_label} 条目 {entry_key}: content 中缺少 [type:xxx] 标签")

    # key 去重检查
    key_duplicates = [k for k in set(all_keys) if all_keys.count(k) > 1]
    if key_duplicates:
        warn("Key重复",
             f"{file_label}: 以下关键词在多个条目中重复: {set(key_duplicates)}")

    # type 标签统计
    if all_type_tags:
        tag_counts = {}
        for t in all_type_tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1
        print(f"   [type:xxx] 标签分布: {tag_counts}")

    print(f"   结构检查完成 ({len(entries)} 个条目)")


def main():
    parser = argparse.ArgumentParser(description="角色设定文件结构有效性校验")
    parser.add_argument("--file", "-f", type=str, default=None,
                        help="指定 JSON 文件路径（默认扫描 data/CharacterInfo/）")
    parser.add_argument("--json", action="store_true",
                        help="以 JSON 格式输出报告")
    args = parser.parse_args()

    print("╔════════════════════════════════════════════════════╗")
    print("║  RAG-RPG 角色设定文件结构校验工具 v2.0             ║")
    print("║  仅校验文件结构，不校验具体数据内容                  ║")
    print("╚════════════════════════════════════════════════════╝")

    if args.file:
        target_files = [Path(args.file)]
    else:
        char_dir = BASE_DIR / "data" / "CharacterInfo"
        if char_dir.exists():
            target_files = sorted(char_dir.glob("*.json"))
        else:
            target_files = []

    if not target_files:
        print("\nℹ️  data/CharacterInfo/ 下没有 JSON 文件。")
        print("   角色设定数据是可选的，没有预设文件也能正常游玩（纯对话记忆模式）。")
        print("   如果只是想检查某个 JSON 文件的结构，请指定文件路径:")
        print("     python scripts/validate_character_consistency.py --file <路径>")
        sys.exit(0)

    for json_path in target_files:
        if not json_path.exists():
            fail("文件访问", f"文件不存在: {json_path}")
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            check_result = "✅ 可解析"
        except json.JSONDecodeError as e:
            fail("JSON解析", f"{json_path.name}: {e}")
            continue

        label = f"{json_path.name} ({check_result})"
        validate_json_structure(data, label)

    print(f"\n{'=' * 50}")
    print(f"  结果: 失败={len(FAILURES)}  警告={len(WARNINGS)}")
    print(f"{'=' * 50}")

    for item in FAILURES:
        print(item)
    for item in WARNINGS:
        print(item)

    if args.json:
        report = {
            "summary": {"failures": len(FAILURES), "warnings": len(WARNINGS)},
            "failures": FAILURES,
            "warnings": WARNINGS,
        }
        print(f"\n{json.dumps(report, ensure_ascii=False, indent=2)}")

    if FAILURES:
        print("\n❌ 存在结构问题，请修复后重新运行。")
        sys.exit(1)
    else:
        print("\n✅ 结构校验通过！文件可正常用于 ingest_initial.py。")


if __name__ == "__main__":
    main()
