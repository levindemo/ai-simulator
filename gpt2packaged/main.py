from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import torch
import time
import uuid
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 初始化FastAPI应用
app = FastAPI(
    title="本地GPT-4o模拟服务",
    description="包含预打包GPT-2模型的OpenAI兼容接口",
    version="1.0.0"
)

# 从本地加载预下载的模型和分词器
print("Loading GPT-2 model from local files...")
model_path = "/app/model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# 设置设备（GPU如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# 设置pad_token
tokenizer.pad_token = tokenizer.eos_token

# 数据模型定义


class Message(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
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
    """生成响应文本"""
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prompt_length = len(tokenizer.decode(
        inputs["input_ids"][0], skip_special_tokens=True))
    return response[prompt_length:].strip()


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    """模拟OpenAI聊天接口"""
    prompt = "\n".join(
        [f"{msg.role}: {msg.content}" for msg in request.messages]) + "\nassistant:"

    response_text = generate_response(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p
    )

    # 计算token使用量
    prompt_tokens = len(tokenizer.encode(prompt))
    completion_tokens = len(tokenizer.encode(response_text))

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "created": int(time.time()),
        "model": request.model,
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
            "total_tokens": prompt_tokens + completion_tokens
        }
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model": "gpt2",
        "device": device
    }
