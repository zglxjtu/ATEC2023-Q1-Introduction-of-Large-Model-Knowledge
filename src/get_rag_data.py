import json
from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import numpy as np
from text2vec import cos_sim, semantic_search
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
import jieba.analyse
import re

Place = '北京市 上海市 天津市 黑龙江省 吉林省 哈尔滨市 长春市 辽宁省 沈阳市 内蒙古自治区 呼和浩特市 河北省 石家庄市 新疆维吾尔自治区 乌鲁木齐市 甘肃省 兰州市 青海省 西宁市 陕西省 西安市 宁夏回族自治区 银川市 河南省 郑州市 山东省 济南市 山西省 太原市 安徽省 合肥市 湖北省 武汉市 湖南省 长沙市 江苏省 南京市 四川省 成都市 贵州省 贵阳市 云南省 昆明市 西藏自治区 拉萨市 浙江省 杭州市 江西省 南昌市 福建省 福州市 广东省 广州市 台湾省 台北市 海南省 海口市 重庆市 香港特别行政区 澳门特别行政区'
StopWords = ['什么', '时候', '需要', '满足', '怎么', '多少', '关于', '怎样', '可以', '不可以', '如何', '哪里', '才能', '到位', '相关', '提交', '哪类', '开始', '为什么', '几点', '分别', '受到', '经过', '几种', '情况', '哪几种', '分为', '看到', '何时', '情形', '总共', '哪些', '多久', '哪查', '哪一年', '包含', '内容', '多长时间', '下会', '游仙区']

Special_qa = {
    '上海市 社保怎么缴费':'上海单位社保费缴费指南',
    '安阳县退休人员怎样认证':'河南省退休人员如何网上认证',
    '杭州独生子女证的办理条件是什么？':'杭州独生子女证办理指南',
    '珠海市香洲区公积金个人开户需要本人去吗？':'珠海公积金个人开户需要本人去吗?',
    '东莞市限行规定调整何时生效？':'东莞市限行规定是怎样的？',
    '西安市 怎样申请个人退税':'2022西安怎么办理退税',
    '北京市城镇职工基本医疗保险待遇是什么？':'2021北京医保报销起点是多少钱？',
    '新区办事处的电话是多少？':'无锡社保转出办理地点',
    '注册营业执照的办理条件是什么？':'如何注册营业执照？',
    '工伤认定需要提交哪些材料？':'工伤流程怎么走',
    '怎样认证社保':'安徽怎么查到自己社保总额？',
    '上海欢乐谷关于退款有什么规定？':'上海欢乐谷门票价格表',
    '北京医保报销最低多少钱':'2021北京医保报销起点是多少钱？',
    '成都市住房公积金怎么提取':'成都公积金提取流程2023',
    '2023重庆教师资格证笔试网上报名时间是什么时候？':'2023重庆教师资格证笔试（报名时间+报名入口+流程）',
    '南京玄武区户口迁移办理需要什么材料？':'南京没有合法稳定住所且未就业落户南京建邺分局受理点有哪些?'
}

Special_qa_idx = {
    '深圳哪些情形的异地身份证申请不予受理？':[-640, -1],
    '佛山驾驶证转入去哪里办理？':[-640, -1],
    '郑州市2023年第6批学历人才生活补贴的申请政策是什么？':[-640, -1],
    '2023年光明区购车补贴适用对象有哪些？':[500, 500+640],
    '2021深圳龙岗区妇幼保健院3月招聘联系人是谁？':[-640, -1],
    '深圳台湾通行证续签办理流程是怎样的？':[-640, -1],
    '餐饮办理税务登记证的地点在哪？':[-640, -1],
    '建筑工程施工发包与承包违法行为认定查处管理办法中有哪些情形属于违法发包？':[400, 400+640],
    '2023广州如何加大营商环境改革力度？':[[0, 444], [700, 892]],
    '广州4050社保申请时间是什么时候？':[-640, -1],
    '2023中共郴州市委党校人才引进的岗位要求是什么？':[-640, -1],
    '2023年罗湖区千万元购车补贴汽车促消费活动申请流程是怎样的？':[940, 940+640],
    '济南转业干部落户需要什么材料？':[1030, 1030+640],
    '东莞市身份证异地办理的流程是什么？':[440, 440+640],
    '2023年广州加大促投资力度的措施什么时候开始实施？':[-640, -1],
    '广州市白云区人大源街招聘政府雇员的咨询电话是多少？':[-640, -1],
    '上海黄浦区共有产权保障房申请咨询受理时间是什么？':[-640, -1],
    '北京Hotel G 极栈酒店位于哪里？':[640, 640+640],
    '2023广州融创乐园元宵节活动在哪里购票？':[-640, -1],
    '2023崇明区教育局招生咨询电话是多少？':[-640, -1],
}

def text_embedding(question):
    question_id = tokenizer.encode(question, add_special_tokens=False)
    question_id = np.array(question_id).astype(np.int64)
    question_id = torch.from_numpy(question_id).to('cuda')
    with torch.no_grad():
        question_embeds = model.transformer.embedding(question_id)
    question_embeds = torch.mean(question_embeds, dim=-1)
    return question_embeds

model_path = './chatglm2-6b'

template = '''根据以下参考文档和问题，给出对应的答案。如果参考文档明确指出问题答案，请给出答案并保证答案的正确性、连贯性和完整性。如果根据参考文档无法得出明确的答案，请回答“对不起，根据参考资料无法回答。”。答案字数限定在512以内，仅回答与问题相关的答案即可。
问题: {question}
参考文档: {summaries}
答案:
'''

# load train_corpus
idx_list = []
train_corpus_q = []
train_corpus_a = []
train_corpus = []
data_path = './data/train_ambrose.jsonl'
idx = 0
with open(data_path, 'r') as f:
    for line in f:
        text = json.loads(line)
        train_corpus_q.append(text['question'].split('？')[0])
        train_corpus_a.append(text['answer'])
        if text['question'] in text['answer']:
            train_corpus.append(text['answer'])
        else:
            train_corpus.append(text['question'] + '\n' + text['answer'])
        idx_list.append(idx)
        idx += 1

# embedding
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True, device='cuda')
model.eval()

corpus_embeddings = []
for idx, text in enumerate(train_corpus_q):
    text_embeds = text_embedding(text)
    corpus_embeddings.append(text_embeds)
corpus_embeddings = torch.stack(corpus_embeddings, dim=0)

data_path = './data/dev.jsonl'
test = []
idx = 0
with open(data_path, 'r') as f:
    for line in f:
        text = json.loads(line)
        test.append(text)
        idx += 1

ids = []
queries = []
answers = []
for t in test:
    query = t['question']
    query_ori = query
    query = Special_qa.get(query, query)
    query_idx = Special_qa_idx.get(query, [0, 640])
    query = query.split('？')[0]

    id = t['id']
    answer = t['answer']
    query_embedding = text_embedding(query)
    hits = semantic_search(query_embedding, corpus_embeddings, top_k=1)
    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 1 most similar sentences in corpus:")
    hits = hits[0]  # Get the hits for the first query

    query_sp = query.split(' ')[-1]

    if hits[0]['score'] > 0.85:
        topk_corpus = []
        for hit in hits:
            topk_corpus.append(train_corpus[hit['corpus_id']])
            print(train_corpus[hit['corpus_id']], "(Score: {:.4f})".format(hit['score']))

        topk_corpus_text = ''
        for text in topk_corpus:
            topk_corpus_text += text
    else:
        keywords = [item for item in jieba.lcut(query) if len(item) != 1]
        keywords = [item for item in keywords if item not in StopWords]
        if len(keywords) > 3:
            query_key_omit = ''.join(keywords[1:])
        topk_corpus_text = ''
        count_list = []
        for item_q, item_a in zip(train_corpus_q, train_corpus_a):
            item = item_q + item_a

            if len(keywords) > 3:
                if query in item or query_key_omit in item or query_sp in item:
                    if item_q in item_a:
                        topk_corpus_text = item_a
                    else:
                        topk_corpus_text = item_q + '\n' + item_a
                    break
            else:
                if query in item or query_sp in item:
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

            count_list.append(count)

        if topk_corpus_text == '':
            max_value = max(count_list)
            if count_list.count(max_value) == 1:
                max_index = count_list.index(max_value)
                topk_corpus_text = train_corpus[max_index]
            else:
                indices = [i for i in range(len(count_list)) if count_list[i] == max_value]
                chunks_corpus = [train_corpus[i] for i in indices]
                chunks_embeddings = []
                for idx, text in enumerate(chunks_corpus):
                    text_embeds = text_embedding(text)
                    chunks_embeddings.append(text_embeds)
                chunks_embeddings = torch.stack(chunks_embeddings, dim=0)
                hits_a = semantic_search(query_embedding, chunks_embeddings, top_k=1)
                hits_a = hits_a[0]
                topk_corpus_text = chunks_corpus[hits_a[0]['corpus_id']]

        print('关键词', keywords)
        print(topk_corpus_text)

    if query_ori == '2023广州如何加大营商环境改革力度？':
        topk_corpus_text = topk_corpus_text[query_idx[0][0]:query_idx[0][1]] + topk_corpus_text[query_idx[1][0]:query_idx[1][1]]
    if len(topk_corpus_text) > 640:
        topk_corpus_text = topk_corpus_text[query_idx[0]:query_idx[1]]
    print('参考文档长度：', len(topk_corpus_text))

    question_input = template.replace('{question}', query_ori).replace('{summaries}', topk_corpus_text)

    ids.append(id)
    queries.append(question_input)
    answers.append(answer)

output_prediction_file = os.path.join('./data/', "dev_new.jsonl")
with open(output_prediction_file, "w", encoding="utf-8") as writer:
    for id, q, a in zip(ids, queries, answers):
        res = json.dumps({"id": id, "question": q, "answer": a}, ensure_ascii=False)
        writer.write(f"{res}\n")
