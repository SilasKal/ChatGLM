from transformers import AutoTokenizer, AutoModel
# import sentencepiece as spm
# import torch
# import cuda
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
model = model.eval()
response, history = model.chat(tokenizer, "Name 10 words that are as different from each other as possible.", history=[])
print(response)
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)
