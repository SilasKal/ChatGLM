from transformers import AutoTokenizer, AutoModel
# import sentencepiece as spm
# import torch
# import cuda
# import openai
from flask import Flask
app = Flask(__name__)

@app.route('/chatglm/<user_input>')
def chatglm(user_input):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, user_input, history=[])
    print(response)
    return response

app.run()
# response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history)
# print(response)

