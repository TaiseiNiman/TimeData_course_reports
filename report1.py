import sys
print("使用中のPython実行ファイル:", sys.executable)

try:
    import scipy
    print("NumPy version:", scipy.__version__)
except ImportError:
    print("NumPyはインストールされていません")