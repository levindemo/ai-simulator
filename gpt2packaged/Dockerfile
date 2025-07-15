# 使用官方Python镜像作为基础
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV MODEL_NAME=gpt2 \
    PORT=80 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY main.py .

# 预先下载GPT-2模型到镜像中
RUN python -c "from transformers import GPT2LMHeadModel, GPT2Tokenizer; \
    tokenizer = GPT2Tokenizer.from_pretrained('$MODEL_NAME'); \
    model = GPT2LMHeadModel.from_pretrained('$MODEL_NAME'); \
    tokenizer.save_pretrained('/app/model'); \
    model.save_pretrained('/app/model')"

# 暴露端口
EXPOSE $PORT

# 启动命令
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT"]
