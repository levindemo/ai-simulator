# 本地 GPT-4o 模拟服务

这是一个使用 FastAPI 和 GPT-2 模型构建的本地服务，模拟了 OpenAI GPT-4o 的 API 接口格式，可以用于开发和测试。

## 功能特点

- 模拟 OpenAI GPT-4o 的`/v1/chat/completions`接口
- 支持 Docker 容器化部署
- 默认绑定 80 端口，方便直接使用
- 自动生成 API 文档
- 支持通过环境变量配置模型和端口

## 快速开始

### 使用 Docker 构建和运行

1. 构建 Docker 镜像：docker build -t local-gpt4o .
2. 运行容器：docker run -d -p 80:80 --name gpt4o-simulator local-gpt4o
3. 测试接口：curl -X POST http://localhost/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
   "model": "gpt-4o",
   "messages": [
   {"role": "user", "content": "什么是人工智能？"}
   ],
   "max_tokens": 150
   }'

### 自定义配置

- 使用更大的模型（需要更多内存）：
  docker run -d -p 80:80 -e MODEL_NAME=gpt2-large --name gpt4o-simulator local-gpt4o
- 更改端口：
  docker run -d -p 8080:8080 -e PORT=8080 --name gpt4o-simulator local-gpt4o

## API 文档

服务启动后，可以通过以下地址访问自动生成的 API 文档：

- Swagger UI: http://localhost/docs
- ReDoc: http://localhost/redoc

## 健康检查

可以通过访问 http://localhost/health 检查服务状态
