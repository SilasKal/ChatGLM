from transformers import AutoTokenizer, AutoModel
import pandas as pd
# import sentencepiece as spm
# import torch
# import cuda
# import openai
# from flask import Flask
# app = Flask(__name__)
#
# @app.route('/chatglm/<user_input>')
disassociationprompt = "Please name 10 words that are as different from each other as possible, " \
                       "in all meanings and uses of " \
                       "the words. Follow the following rules: " \
                       "Only single words in English, only nouns (e.g., things, objects, concepts), " \
                       "no proper nouns (e.g., no specific people or places) and no specialised vocabulary " \
                       "(e.g., no technical terms). "

disassociationprompt2 = "Please name 10 words that are as different from each other as possible, " \
                       "in all meanings and uses of " \
                       "the words. Maximize the unrelatedness of the words! Follow the following rules: " \
                       "Only single words in English, only nouns (e.g., things, objects, concepts), " \
                       "no proper nouns (e.g., no specific people or places) and no specialised vocabulary " \
                       "(e.g., no technical terms). "


def chatglm(user_input):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, user_input, history=[])
    print(user_input, response)
    return response


def dat_txt_tsv(filename, filename2):
    columns = ['id']
    for i in range(1, 11):
        columns.append('word' + str(i))
    df = pd.DataFrame(columns = columns)
    print(columns)
    with open(filename, 'r', encoding='utf8') as f:
        for counter, line in enumerate(f):
            print([counter] + line.strip().split(',')[1:11])
            line = line.strip()
            df.loc[len(df)] = [counter] + line.split(',')[1:11].encode('utf8', 'replace')
    print(df)
    df.to_csv(filename2, sep='\t')

def dat(prompt, filename_raw, filename2, num_responses):
    # df = pd.DataFrame(columns = ['word ' + str(i) for i in range(1, 11)])
    for i in range(num_responses):
        response = chatglm(prompt)
        response = ''.join(filter(lambda x: (not x.isdigit() and not x in ['.', ' ']), response))
        response = response.split('\n')
        with open(filename_raw, 'a+', encoding='utf8') as f:
            print(i)
            string_to_write = ",".join(response)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)
    dat_txt_tsv(filename_raw, filename2)


# dat(disassociationprompt, 'response_100_prompt1.txt', 'response_prompt1_chatglm.tsv', 100)
dat_txt_tsv('response_100_prompt1.txt', 'response_prompt1_chatglm.tsv')
dat(disassociationprompt2, 'response_100_prompt2.txt', 'response_prompt2_chatglm.tsv', 100)
