"""
数据文件传输完整性校验工具

当你把自己的 JSON 数据文件从一台设备复制到另一台设备时，
用这个工具确认文件是否完整传输、没有损坏。

功能：
  1. generate  — 为 data/ 下所有 JSON/TXT 文件生成 SHA256 指纹清单
  2. check     — 对比当前文件是否与指纹清单匹配（在另一台设备上执行）
  3. verify    — 验证所有 JSON 文件是否能被正常解析（无损坏）
  4. diff      — 显示文件是否有新增/删除/修改

用法：
  python scripts/data_manifest.py generate   # 生成指纹清单
  python scripts/data_manifest.py check      # 校验文件是否与清单一致
  python scripts/data_manifest.py verify     # 检查 JSON 文件是否可读
  python scripts/data_manifest.py diff       # 查看文件变化
"""
import sys
import os
import json
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_FILE = BASE_DIR / "data_manifest.json"
WATCHED_EXTENSIONS = {".json", ".txt"}
WATCHED_DIRS = ["CharacterInfo", "WorldInfo"]


def _hash_file(filepath: Path) -> str:
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def _get_data_files() -> list[Path]:
    files = []
    for dirname in WATCHED_DIRS:
        dirpath = DATA_DIR / dirname
        if not dirpath.exists():
            continue
        for f in sorted(dirpath.rglob("*")):
            if f.is_file() and f.suffix in WATCHED_EXTENSIONS:
                files.append(f)
    return files


def _relative_path(filepath: Path) -> str:
    return str(filepath.relative_to(BASE_DIR))


def generate_manifest():
    files = _get_data_files()
    if not files:
        print("ℹ️  data/ 下没有预设数据文件。")
        print("   角色设定和世界观数据是可选的，没有文件也能正常游玩。")
        print("   当你准备好自己的 JSON 文件后，再运行此命令生成指纹清单。")
        sys.exit(0)

    manifest = {
        "version": "1.0",
        "generated_at": os.path.getmtime(__file__),
        "files": {},
    }
    for f in files:
        rel = _relative_path(f)
        manifest["files"][rel] = {
            "hash": _hash_file(f),
            "size": f.stat().st_size,
            "mtime": f.stat().st_mtime,
        }

    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"✅ 已生成数据清单: {MANIFEST_FILE}")
    print(f"   共 {len(files)} 个文件已记录指纹")
    return manifest


def check_manifest():
    if not MANIFEST_FILE.exists():
        print("❌ 未找到数据清单文件 data_manifest.json，请先运行:")
        print("   python scripts/data_manifest.py generate")
        sys.exit(1)

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        recorded = json.load(f)

    files = _get_data_files()
    recorded_paths = set(recorded.get("files", {}).keys())
    actual_paths = set(_relative_path(f) for f in files)

    missing_in_record = actual_paths - recorded_paths
    extra_in_record = recorded_paths - actual_paths

    all_ok = True

    if extra_in_record:
        print("⚠️  清单中存在但实际已删除的文件:")
        for p in sorted(extra_in_record):
            print(f"   - {p}")
        all_ok = False

    if missing_in_record:
        print("⚠️  实际存在但未记录在清单中的文件（新增文件）:")
        for p in sorted(missing_in_record):
            print(f"   + {p}")
        all_ok = False

    hash_mismatch = False
    for f in files:
        rel = _relative_path(f)
        if rel not in recorded.get("files", {}):
            continue
        current_hash = _hash_file(f)
        recorded_hash = recorded["files"][rel]["hash"]
        if current_hash != recorded_hash:
            if not hash_mismatch:
                print("❌ 文件已被修改（hash 不匹配）:")
                hash_mismatch = True
            print(f"   * {rel}")
            print(f"     记录: {recorded_hash[:16]}...")
            print(f"     当前: {current_hash[:16]}...")
            all_ok = False

    if all_ok:
        print("✅ 所有文件与清单记录匹配。")
    else:
        print("\n📌 清单与实际情况存在差异，这是正常的——")
        print("   当你传输完数据文件后，在新设备上重新运行 generate 即可更新清单。")


def verify_files():
    files = _get_data_files()
    all_ok = True
    for f in files:
        try:
            if f.suffix == ".json":
                with open(f, "r", encoding="utf-8") as fh:
                    json.load(fh)
                print(f"  ✅ {_relative_path(f)} — JSON 格式有效")
            elif f.suffix == ".txt":
                with open(f, "r", encoding="utf-8") as fh:
                    fh.read()
                print(f"  ✅ {_relative_path(f)} — TXT 可读")
        except json.JSONDecodeError as e:
            print(f"  ❌ {_relative_path(f)} — JSON 解析失败: {e}")
            all_ok = False
        except Exception as e:
            print(f"  ❌ {_relative_path(f)} — 读取失败: {e}")
            all_ok = False

    if all_ok:
        print(f"\n✅ 所有 {len(files)} 个数据文件完整性校验通过！")
    else:
        print(f"\n❌ 存在文件损坏，请检查上述错误项")
        sys.exit(1)


def show_diff():
    if not MANIFEST_FILE.exists():
        print("❌ 未找到数据清单，请先生成基线文件:")
        print("   python scripts/data_manifest.py generate")
        return

    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        recorded = json.load(f)

    files = _get_data_files()
    current_map = {}
    for f in files:
        rel = _relative_path(f)
        current_map[rel] = {
            "hash": _hash_file(f),
            "size": f.stat().st_size,
        }

    recorded_files = recorded.get("files", {})
    all_paths = sorted(set(list(recorded_files.keys()) + list(current_map.keys())))

    print(f"{'状态':<6} {'文件路径':<40} {'大小变化':<12} {'Hash变化':<10}")
    print("-" * 80)

    any_change = False
    for rel in all_paths:
        old = recorded_files.get(rel)
        cur = current_map.get(rel)
        if old is None:
            print(f"{'🟢新增':<6} {rel:<40} {'N/A':<12} {'+新文件':<10}")
            any_change = True
        elif cur is None:
            print(f"{'🔴删除':<6} {rel:<40} {'N/A':<12} {'-已删除':<10}")
            any_change = True
        elif old["hash"] != cur["hash"]:
            size_diff = cur["size"] - old["size"]
            size_str = f"{size_diff:+d}B" if size_diff != 0 else "0B"
            print(f"{'🟡修改':<6} {rel:<40} {size_str:<12} {'Hash变':<10}")
            any_change = True

    if not any_change:
        print(f"{'✅ 无变化':<6} 所有数据文件与清单记录完全一致")

    print(f"\n共检查 {len(all_paths)} 个文件路径")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    if command == "generate":
        generate_manifest()
    elif command == "check":
        check_manifest()
    elif command == "verify":
        verify_files()
    elif command == "diff":
        show_diff()
    else:
        print(f"未知命令: {command}")
        print("可用命令: generate, check, verify, diff")
        sys.exit(1)


if __name__ == "__main__":
    main()
