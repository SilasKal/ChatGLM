import re

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


def chatglm(user_input, message_history = []):
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True)
    model = AutoModel.from_pretrained("THUDM/chatglm-6b-int8", trust_remote_code=True).half().cuda()
    model = model.eval()
    response, history = model.chat(tokenizer, user_input, history=message_history)
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
            df.loc[len(df)] = [counter] + line.split(',')[1:11]
    print(df)
    df.to_csv(filename2, sep='\t', encoding='utf8')

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

def filter_responses(filename, filename_new):
    # Open the file in read mode
    lines_lst = []
    with open(filename, 'r', encoding='utf8') as file:
        for line in file:
            new_list = []
            curr_list = line.strip().split(',')
            # print(curr_list)
            for i in curr_list:
                i = re.sub('[\u4e00-\u9fff]+', '', i)
                i= re.sub('-[^ ]*', '', i)
                # i = i.replace('(', '')
                # i = i.replace(')', '')
                i= re.sub('[^a-zA-Z]+', '', i)
                # print(filtered_string)
                if not (i.startswith('These') or i.startswith('Note') or i.endswith(':') or i.endswith('!') or i.startswith('here') or i in ['', ',', 'Okay'] or i.startswith('suchas')):
                    new_list.append(i)
                    # print(i)
                else:
                    pass
                    # print(i)
            # print(new_list)
            lines_lst.append(new_list)
    file.close()
    # Open the file in write mode and write the modified contents back to the file
    with open(filename_new, 'w', encoding='utf8') as f:
        for i, response in enumerate(lines_lst):
            print(response)
            # Convert the list of strings to a single string separated by commas
            string_to_write = ",".join(response)
            # Write the string to the file
            if i != 0:
                f.write('\n' + string_to_write)
            else:
                f.write(string_to_write)
    f.close()
# dat(disassociationprompt, 'response_100_prompt1.txt', 'response_prompt1_chatglm.tsv', 100)
# dat_txt_tsv('response_100_prompt1.txt', 'response_prompt1_chatglm.tsv')
# dat(disassociationprompt2, 'response_100_prompt2.txt', 'response_prompt2_chatglm.tsv', 100)
# filter_responses('response_100_prompt1.txt', 'response_100_prompt1_filtered.txt')
# dat_txt_tsv('response_100_prompt1_filtered.txt', 'response_prompt1_chatglm.tsv')


chatglm(input('User input >'))
