# 初始化子模块（只需一次）
git submodule update --init --recursive

# 创建 Python 3.8 环境（只需一次）
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# 运行评估
python examples/libero/main.py --args.task-suite-name libero_spatial
