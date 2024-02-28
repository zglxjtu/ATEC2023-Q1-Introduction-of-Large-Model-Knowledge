import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
from text2vec import cos_sim, semantic_search
import os
from tqdm import tqdm
import random

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain.retrievers import BM25Retriever
from langchain.schema import Document
import jieba
import re

Place = '北京市 上海市 天津市 黑龙江省 吉林省 哈尔滨市 长春市 辽宁省 沈阳市 内蒙古自治区 呼和浩特市 河北省 石家庄市 新疆维吾尔自治区 乌鲁木齐市 甘肃省 兰州市 青海省 西宁市 陕西省 西安市 宁夏回族自治区 银川市 河南省 郑州市 山东省 济南市 山西省 太原市 安徽省 合肥市 湖北省 武汉市 湖南省 长沙市 江苏省 南京市 四川省 成都市 贵州省 贵阳市 云南省 昆明市 西藏自治区 拉萨市 浙江省 杭州市 江西省 南昌市 福建省 福州市 广东省 广州市 台湾省 台北市 海南省 海口市 重庆市 香港特别行政区 澳门特别行政区'
StopWords = ['什么', '时候', '需要', '满足', '怎么', '多少', '关于', '怎样', '可以', '不可以', '如何', '哪里', '才能', '到位', '相关', '提交', '哪类', '开始', '为什么', '几点', '分别', '受到', '经过', '几种', '情况', '哪几种', '分为', '看到', '何时', '情形', '总共', '哪些', '多久', '哪查', '哪一年', '包含', '内容', '多长时间', '下会']

class BM25_Retriever(object):
    def __init__(self, documents):
        doc = []
        all_docs = []
        for idx, line in enumerate(documents):
            if(len(line)<2):
                continue
            tokens = " ".join(jieba.cut_for_search(line))
            doc.append(Document(page_content=tokens, metadata={"id": idx}))
            all_docs.append(Document(page_content=line, metadata={"id": idx}))
        self.documents = doc
        self.full_documents = all_docs
        self.retriever = self._init_bm25()

    def _init_bm25(self):
        return BM25Retriever.from_documents(self.documents)

    def GetBM25TopK(self, query, topk):
        self.retriever.k = topk
        query = " ".join(jieba.cut_for_search(query))
        ans_docs = self.retriever.get_relevant_documents(query)
        ans = []
        for line in ans_docs:
            ans.append(self.full_documents[line.metadata["id"]])
        return ans

def sliding_window(strs, max_len=512, overlap_len=128):
    cleaned_chunks = []
    i = 0
    strs = strs.replace('\n\t', ' ')
    strs = strs.replace('\u3000\u30001', '')
    strs = strs.replace('\u3000\u3000', '')
    while i < len(strs):
        cur_s = strs[i:i + max_len]
        if len(cur_s) > 10:
            cleaned_chunks.append(cur_s)
        i += (max_len - overlap_len)

    return cleaned_chunks

def seed_everything(seed: int): # fix the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def text_embedding(question):
    question_id = tokenizer.encode(question, add_special_tokens=False)
    question_id = np.array(question_id).astype(np.int64)
    question_id = torch.from_numpy(question_id).to('cuda')

    with torch.no_grad():
        question_embeds = model.transformer.embedding(question_id)
    question_embeds = torch.mean(question_embeds, dim=-1)
    return question_embeds

seed_everything(42)

model_path = '../chatglm2-6b'
# template = """请使用以下检索到的上下文来回答问题。问题是：{question},上下文: {summaries},答案是:"""
template = '''根据以下参考文档和问题，给出对应的答案。如果参考文档明确指出问题答案，请给出答案并保证答案的正确性、连贯性和完整性。如果根据参考文档无法得出明确的答案，请回答“对不起，根据参考资料无法回答。”。答案字数限定在512以内，仅回答与问题相关的答案即可。
问题: {question}
参考文档: {summaries}
答案:
'''

# load train_corpus
idx_list = []
train_corpus_q = []
train_corpus_q2 = []
train_corpus_q_key = []
train_corpus_q_key2 = []
train_corpus_a = []
train_corpus = []
train_corpus_dict = {}
data_path = '../data/train_ambrose.jsonl'
idx = 0
with open(data_path, 'r') as f:
    for line in f:
        text = json.loads(line)
        keywords = [item for item in jieba.lcut(text['question']) if len(item) != 1]
        keywords = [item for item in keywords if item not in StopWords]
        train_corpus_q_key.append(''.join(keywords))
        train_corpus_q_key2.append(keywords)

        train_corpus_q2.append(text['question'])
        train_corpus_q.append(text['question'].replace('?', '？').split('？')[0])
        train_corpus_a.append(text['answer'])
        if text['question'] in text['answer']:
            train_corpus.append(text['answer'])
            train_corpus_dict[text['question']] = text['answer']
        else:
            train_corpus.append(text['question'] + '\n' + text['answer'])
            train_corpus_dict[text['question']] = text['question'] + '\n' + text['answer']
        idx_list.append(idx)
        idx += 1

# embedding
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
pre_seq_len = 128
checkpoint_path = "../output/checkpoint-1600"
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, pre_seq_len=pre_seq_len)
model = AutoModel.from_pretrained(model_path, config=config, device='cuda', trust_remote_code=True)
prefix_state_dict = torch.load(os.path.join(checkpoint_path, "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
model = model.to('cuda')
model.eval()

# tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
# model = AutoModel.from_pretrained(model_path, device="cuda", trust_remote_code=True)
# model.eval()

corpus_embeddings = []
for idx, text in enumerate(train_corpus_q):
    text_embeds = text_embedding(text)
    corpus_embeddings.append(text_embeds)
corpus_embeddings = torch.stack(corpus_embeddings, dim=0)

corpus_embeddings_key = []
for idx, text in enumerate(train_corpus_q_key):
    text_embeds = text_embedding(text)
    corpus_embeddings_key.append(text_embeds)
corpus_embeddings_key = torch.stack(corpus_embeddings_key, dim=0)

content_chunks = []
for content in train_corpus:
    content_chunks+=sliding_window(content, max_len=512, overlap_len=128)

bm25_title = BM25_Retriever(train_corpus_q2)
bm25_content = BM25_Retriever(content_chunks)

test = []
with open('../data/test.txt', 'r') as file:
    for line in file:
        test.append(line.split('\n')[0])


ids = []
queries = []
answers = []
id = 0
for t in tqdm(test):
    query = t
    query_ori = query
    query = Special_qa.get(query, query)
    query = query.replace('?', '？').split('？')[0]

    query_embedding = text_embedding(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
    hits = hits[0]  # Get the hits for the first query
    if ' ' in query:
        flag = 1
    else:
        flag = 0
    if hits[0]['score'] > 0.85:
        topk_corpus = []
        for hit in hits:
            topk_corpus.append(train_corpus[hit['corpus_id']])

        topk_corpus_text = ''
        for text in topk_corpus:
            topk_corpus_text += text
    else:
        keywords = [item for item in jieba.lcut(query) if len(item) != 1]
        keywords = [item for item in keywords if item not in StopWords]
        if len(keywords) >= 5:
            query_key_omit1 = ''.join(keywords[:-1])
            query_key_omit2 = ''.join(keywords[1:])
            query_key_omit3 = ''.join(keywords[:-2])
        query_key = ''.join(keywords)
        query_key_embedding = text_embedding(query_key)
        hits_key = semantic_search(query_key_embedding, corpus_embeddings_key, top_k=1)
        hits_key = hits_key[0]  # Get the hits for the first query
        if hits_key[0]['score'] > 0.85:
            topk_corpus_key = []
            for hit_key in hits_key:
                topk_corpus_key.append(train_corpus[hit_key['corpus_id']])

            topk_corpus_text = ''
            for text in topk_corpus_key:
                topk_corpus_text += text
        else:
            keywords = [item for item in jieba.lcut(query) if len(item) != 1]
            keywords = [item for item in keywords if item not in StopWords]
            topk_corpus_text = ''
            count_list = []

            for item_q_key2, item_q_key, item_q, item_a in zip(train_corpus_q_key2, train_corpus_q_key, train_corpus_q,
                                                               train_corpus_a):
                item = item_q + item_a

                if len(keywords) >= 5:
                    if query in item or query_key_omit1 in item or query_key_omit2 in item or query_key_omit3 in item:
                        if item_q in item_a:
                            topk_corpus_text = item_a
                        else:
                            topk_corpus_text = item_q + '\n' + item_a
                        break
                else:
                    if query in item:
                        if item_q in item_a:
                            topk_corpus_text = item_a
                        else:
                            topk_corpus_text = item_q + '\n' + item_a
                        break

                if len(item_q_key2) > 2 and len(keywords) > 2:
                    if set(item_q_key2).issubset(set(keywords)) or set(keywords).issubset(set(item_q_key2)):
                        if item_q in item_a:
                            topk_corpus_text = item_a
                        else:
                            topk_corpus_text = item_q + '\n' + item_a
                        break

                count = 0
                place = []
                for key in keywords:
                    if '省' in key or '市' in key or key in Place:
                        place.append(key)

                    if key in item:
                        count += 1
                if len(place) != 0:
                    p_count = 0
                    for p in place:
                        p = p.replace('省', '').replace('市', '')
                        if p in item:
                            p_count += 1
                    if p_count != len(place):
                        count = 0

                if '《' in query:
                    law = query.split('《')[-1].split('》')[0]
                    if law not in item:
                        count = 0

                count_list.append(count)

            if topk_corpus_text == '':
                max_value = max(count_list)
                if count_list.count(max_value) == 1 and flag == 0:
                    max_index = count_list.index(max_value)
                    topk_corpus_text = train_corpus[max_index]
                else:
                    match_title = bm25_title.GetBM25TopK(query_ori, 2)
                    match_title = [train_corpus_dict[p.page_content] for p in match_title]
                    match_title_chunks = []
                    for content in match_title:
                        match_title_chunks += sliding_window(content, max_len=512, overlap_len=128)
                    #
                    bm25_title2 = BM25_Retriever(match_title_chunks)
                    match_title = bm25_title2.GetBM25TopK(query_ori, 2)
                    match_content = bm25_content.GetBM25TopK(query_ori, 2)
                    match_res = [p.page_content for p in match_title]
                    for res in match_content:
                        if res.page_content not in match_res:
                            match_res.append(res.page_content)
                    cnt = 1
                    topk_corpus_text = ''
                    for p in match_res:
                        topk_corpus_text += f'文档{cnt}：{p}\n\n'
                        cnt += 1

    # if len(topk_corpus_text) > 8000:
    #     topk_corpus_text = topk_corpus_text[:4096]
    question_input = template.replace('{question}', query_ori).replace('{summaries}', topk_corpus_text)

    # response, history = model.chat(tokenizer,
    #                                question_input,
    #                                history=[],
    #                                max_length=1024,
    #                                top_p=0.7,
    #                                temperature=0.95)

    response, history = model.chat(tokenizer,
                                   question_input,
                                   history=[],
                                   top_p=0.8,
                                   temperature=0.6
                                   )
    ids.append(id)
    queries.append(query_ori)
    answers.append(response)
    id += 1

output_prediction_file = os.path.join('./', "predictions.jsonl")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    for id, q, a in zip(ids, queries, answers):
        res = json.dumps({"id": id, "question": q, "answer": a}, ensure_ascii=False)
        writer.write(f"{res}\n")
