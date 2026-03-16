#!/bin/bash
cd ~/dpno
# 初始化Git仓库
git init

# 创建.gitignore文件（非常重要，特别是对于深度学习项目）
cat > .gitignore << EOF
# 数据文件
*.data
*.pickle
*.h5
*.hdf5
*.npy
*.npz
*.png
*.ipynb

# 模型文件
*.pt
*.pth
*.ckpt
*.weights
*.bin

# 日志和输出
experiments/
logs/
runs/
results/
outputs/
checkpoints/

# 虚拟环境
venv/
env/
envs/
.venv
.env

# IDE配置
.vscode/
.idea/
*.swp
*.swo

# 数据集
data/
datasets/
__pycache__/
*.pyc

# 大文件
*.zip
*.tar
*.gz
*.rar
EOF

# 添加所有文件
git add .

# 首次提交
git commit -m "Initial commit: Deep learning project"