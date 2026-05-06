"""
检查 Python 环境、chromadb、sentence-transformers 是否正常。
首次运行会自动下载 embedding 模型（约几百 MB，需网络通畅）。
"""
import sys

print("Python 版本:", sys.version)

# 检查 chromadb
try:
    import chromadb
    print(f"chromadb 已安装，版本: {chromadb.__version__}")
except ImportError:
    print("❌ chromadb 未安装，请在当前环境运行: pip install chromadb")
    sys.exit(1)

# 检查 sentence-transformers 并加载模型
try:
    from sentence_transformers import SentenceTransformer
    print("sentence-transformers 已安装，正在加载 embedding 模型...")
    model = SentenceTransformer('shibing624/text2vec-base-chinese')
    print("✅ embedding 模型加载成功！环境一切就绪。")
except ImportError:
    print("❌ sentence-transformers 未安装，请运行: pip install sentence-transformers")
    sys.exit(1)
except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    sys.exit(1)