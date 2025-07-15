from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# 生成文本
inputs = tokenizer.encode("Hello, my dog is very", return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# 生成文本
inputs = tokenizer.encode("what is AI?", return_tensors="pt")
outputs = model.generate(inputs, max_length=50, num_return_sequences=1)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
