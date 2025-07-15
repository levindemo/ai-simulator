from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import time
import uuid

app = Flask(__name__)

# 加载GPT-2模型和分词器
print("Loading GPT-2 model and tokenizer...")
model_name = "gpt2"  # 可以替换为"gpt2-medium", "gpt2-large", "gpt2-xl"等更大的模型
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# 将模型移动到GPU（如果可用）
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded on {device}")

# 设置pad_token为eos_token，因为GPT-2默认没有pad_token
tokenizer.pad_token = tokenizer.eos_token


def generate_response(prompt, max_tokens=100, temperature=0.7, top_p=1.0):
    """使用GPT-2生成响应"""
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


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """模拟OpenAI的聊天完成接口"""
    data = request.json

    # 解析请求参数
    messages = data.get('messages', [])
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 1.0)

    # 构建提示词
    prompt = "\n".join(
        [f"{msg['role']}: {msg['content']}" for msg in messages]) + "\nassistant:"

    # 生成响应
    start_time = time.time()
    response_text = generate_response(prompt, max_tokens, temperature, top_p)
    end_time = time.time()

    # 构建符合OpenAI格式的响应
    response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(response_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
        }
    }

    return jsonify(response)


@app.route('/v1/completions', methods=['POST'])
def completions():
    """模拟OpenAI的完成接口"""
    data = request.json

    # 解析请求参数
    prompt = data.get('prompt', '')
    max_tokens = data.get('max_tokens', 100)
    temperature = data.get('temperature', 0.7)
    top_p = data.get('top_p', 1.0)

    # 生成响应
    start_time = time.time()
    response_text = generate_response(prompt, max_tokens, temperature, top_p)
    end_time = time.time()

    # 构建符合OpenAI格式的响应
    response = {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(tokenizer.encode(prompt)),
            "completion_tokens": len(tokenizer.encode(response_text)),
            "total_tokens": len(tokenizer.encode(prompt)) + len(tokenizer.encode(response_text))
        }
    }

    return jsonify(response)


if __name__ == '__main__':
    print("Starting GPT-2 API server...")
    print("Available endpoints:")
    print("POST /v1/chat/completions - Simulates OpenAI chat completions")
    print("POST /v1/completions - Simulates OpenAI text completions")
    app.run(host='0.0.0.0', port=5000, debug=True)
