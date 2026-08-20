# 1. 自動作成された Python 3.14 の .venv を削除
rm -rf /home/hiroyoshi/workspace/sotuken_learn/.venv
rm -rf .venv

# 2. script ディレクトリ内に Python 3.12 の環境を明示的に作成
uv venv --python 3.12 --no-project .venv

# 3. 2.で作った 3.12 環境の中に PyTorch (CUDA 12.4版) と matplotlib を導入
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 --python .venv/bin/python --no-cache
uv pip install matplotlib --python .venv/bin/python

# 実行
uv run --no-project --python .venv/bin/python bainarize_connect.py





# 1. Python 3.12 でプロジェクトを作成
uv init my_project --python 3.12
cd my_project

# 2. CUDA 12.4 版 PyTorch と torchvision を追加
uv add torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. その他必要なライブラリ（matplotlib 等）を追加
uv add matplotlib