# AI Infra

物理机写代码，容器中运行测试。

## 环境搭建

### 1. 安装 nvidia-container-toolkit

```bash
# 添加源
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.sources <<'EOF'
Enabled: yes
Types: deb
URIs: https://nvidia.github.io/libnvidia-container/stable/deb/amd64
Suites: /
Components:
Signed-By: /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
EOF

# 安装
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 验证
docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu24.04 nvidia-smi
```

### 2. 拉取镜像

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: <你的 NGC API Key>
# 获取 key: https://ngc.nvidia.com/setup/api-key

docker pull nvcr.io/nvidia/pytorch:26.04-py3
```

### 3. 创建并启动容器

```bash
docker run -d \
  --name aiinfra \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v $(pwd):/workspace \
  -w /workspace \
  -p 8888:8888 \
  nvcr.io/nvidia/pytorch:26.04-py3 \
  tail -f /dev/null
```

### 4. 进入容器

```bash
docker exec -it aiinfra bash
```

## 日常使用

```bash
# 启动已有容器（重启后）
docker start aiinfra

# 进入容器
docker exec -it aiinfra bash

# 在容器中运行测试
docker exec -it aiinfra pytest tests/ -v

# Jupyter Lab (容器内)
docker exec -it aiinfra jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
# 然后访问 http://localhost:8888

# 停止容器
docker stop aiinfra
```

## 工作流

```
物理机 (host)  ←→  容器 (/workspace)
  写代码              跑测试
  git操作             python训练/推理
```

项目目录通过 `-v $(pwd):/workspace` 挂载进容器，物理机上改代码，容器内立刻生效。
