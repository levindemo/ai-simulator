from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import uuid
import os

# 从环境变量获取配置
MODEL_NAME = os.getenv("MODEL_NAME", "gpt2")  # 默认使用gpt2模型
PORT = int(os.getenv("PORT", 80))  # 默认端口80

# 初始化FastAPI应用
app = FastAPI(
    title="本地GPT-4o模拟服务",
    description="模拟OpenAI GPT-4o API接口的本地服务",
    version="1.0.0"
)

# 加载GPT模型和分词器
print(f"Loading {MODEL_NAME} model and tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)

# 将模型移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# 设置pad_token为eos_token，因为GPT-2默认没有pad_token
tokenizer.pad_token = tokenizer.eos_token

# 定义请求和响应的数据模型


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"  # 默认模拟gpt-4o
    messages: List[Message]
    max_tokens: Optional[int] = Field(100, ge=1, le=1000)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=1.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: Usage


def generate_response(prompt: str, max_tokens: int = 100, temperature: float = 0.7, top_p: float = 1.0) -> str:
    """使用GPT模型生成响应"""
    # 对输入进行编码
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    # 生成文本
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # 解码生成的文本
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 只返回新生成的部分（去除提示词）
    prompt_length = len(tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=True))
    return response[prompt_length:].strip()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """模拟OpenAI的聊天完成接口，兼容GPT-4o格式"""
    # 构建提示词
    prompt = "\n".join(
        [f"{msg.role}: {msg.content}" for msg in request.messages]) + "\nassistant:"

    # 生成响应
    response_text = generate_response(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    # 计算token使用量
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(response_text))
    total_tokens = prompt_tokens + completion_tokens

    # 构建响应
    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "created": int(time.time()),
        "model": request.model,  # 返回请求中指定的模型名称
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                }
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model": MODEL_NAME,
        "device": device,
        "simulated_model": "gpt-4o"
    }

if __name__ == '__main__':
    import uvicorn
    print(f"Starting GPT-4o simulation API server with {MODEL_NAME}...")
    print(f"Server will run on port {PORT}")
    print("Available endpoints:")
    print("POST /v1/chat/completions - 模拟OpenAI GPT-4o聊天接口")
    print("GET /health - 服务健康检查")
    print(f"访问 http://localhost:{PORT}/docs 查看API文档")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
