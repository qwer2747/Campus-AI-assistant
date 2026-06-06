# ==================== RAG.py ====================
import streamlit as st
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用国内镜像

import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
import chromadb
import time

# ==================== 配置 ====================
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]  # 从云端安全读取密码
CSV_PATH = "campus_qa.csv"
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
CHROMA_DB_PATH = "./chroma_db"
RETRIEVE_K = 5
TEMPERATURE = 0.3
EMBEDDING_MODEL = "BAAI/bge-small-zh"

# ==================== 全局单例初始化 ====================
print("加载嵌入模型...")
model = SentenceTransformer(EMBEDDING_MODEL)

print("连接Chroma数据库...")
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection(name="campus_qa")

def embed_function(texts):
    return model.encode(texts, normalize_embeddings=True).tolist()

# ==================== 核心功能函数 ====================
def search(query, k=RETRIEVE_K):
    """检索函数"""
    query_embedding = embed_function([query])
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    return results['documents'][0] if results['documents'] else []

def call_deepseek(prompt):
    """调用大模型"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是校园指南AI助手，请基于提供的资料准确回答问题。"},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": 500
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"API调用失败：{response.status_code}"
    except Exception as e:
        return f"网络错误：{str(e)}"

def ask(question, history_context=""):
    """提供给外部调用的问答接口"""
    docs = search(question)
    
    if docs:
        context = "\n\n".join([f"【资料{i+1}】{doc}" for i, doc in enumerate(docs)])
    else:
        context = "知识库中暂未找到强相关的内部资料。"
    
    # 👈 修改这里的 prompt 提示词
    prompt = f"""你是一个热情、有用的校园新生助手。请优先根据【参考资料】和【历史记录】来回答用户的问题。
要求：
1. 如果【参考资料】能回答问题，请准确提取信息并回答。
2. 如果【参考资料】中没有直接答案，你可以结合常识给出合理的建议，但请委婉说明“目前我的知识库中还没有详细记录，建议参考...”。
3. 态度要友好，不要生硬地说“我不知道”。

【参考资料】：
{context}

{history_context}
【当前问题】：{question}
回答："""
    
    return call_deepseek(prompt)

def learn_new_knowledge(question, correct_answer):
    """提供给外部的纠错学习接口"""
    content = f"问题：{question}\n答案：{correct_answer} (用户补充)"
    doc_id = f"qa_learned_{int(time.time())}"
    embedding = embed_function([content])
    collection.add(documents=[content], embeddings=embedding, ids=[doc_id])
    print(f"✅ 学习成功: {question}")

# ==================== 初始化知识库 ====================
def init_db_from_csv():
    """如果数据库为空，则从CSV加载数据"""
    if collection.count() == 0:
        print(f"检测到知识库为空，正在读取 {CSV_PATH}...")
        if not os.path.exists(CSV_PATH):
            print("⚠️ 未找到 CSV 文件！")
            return
            
        try:
            df = pd.read_csv(CSV_PATH, encoding='utf-8-sig')
        except:
            df = pd.read_csv(CSV_PATH, encoding='gbk')
            
        documents, ids = [], []
        for idx, row in df.iterrows():
            documents.append(f"问题：{row['question']}\n答案：{row['answer']}")
            ids.append(f"qa_{idx}")
            
        embeddings = embed_function(documents)
        collection.add(documents=documents, embeddings=embeddings, ids=ids)
        print(f"✅ 成功存入 {len(documents)} 条初始数据！")

# 模块导入时自动执行检查
init_db_from_csv()