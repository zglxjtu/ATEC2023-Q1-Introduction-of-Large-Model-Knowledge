import json
import numpy as np
import os

# load train_corpus
idx_list = []
q_list = []
a_list = []
data_path = './data/train_corpus.jsonl'
idx = 0
with open(data_path, 'r') as f:
    for line in f:
        text = json.loads(line)
        q_temp = text['content'].split('\n内容：')[0]
        if '标题：\n' in q_temp:
            q = q_temp.split('标题：\n')[1]
        else:
            q = q_temp.split('标题：')[1]
        a = text['content'].split('\n内容：')[1]
        q_list.append(q)
        a_list.append(a)
        idx_list.append(idx)
        idx += 1

output_prediction_file = os.path.join('./data/', "train_ambrose.jsonl")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    for id, q, a in zip(idx_list, q_list, a_list):
        res = json.dumps({"id": id, "question": q, "answer": a}, ensure_ascii=False)
        writer.write(f"{res}\n")
