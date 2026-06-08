# ==================== web.py ====================
import streamlit as st
import sqlite3
from datetime import datetime
from RAG import ask, learn_new_knowledge  # 直接调用 RAG.py 里的接口

DB_PATH = "user_memory.db"

# ==================== 页面配置 ====================
st.set_page_config(page_title="校园新生指南",
                   page_icon="💬", 
                   layout="wide", 
                   initial_sidebar_state="expanded")


# ==================== SQLite 数据库初始化 (线程安全版) ====================
def init_sqlite_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS chat_memory 
                     (user_id TEXT, role TEXT, content TEXT, timestamp DATETIME)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS feedback_log 
                     (user_id TEXT, question TEXT, ai_answer TEXT, correct_answer TEXT)''')

init_sqlite_db()

def save_message(user_id, role, content):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("INSERT INTO chat_memory VALUES (?, ?, ?, ?)", 
                     (user_id, role, content, datetime.now()))

def load_user_memory(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT role, content FROM chat_memory WHERE user_id=? ORDER BY timestamp ASC", (user_id,))
        return [{"role": row[0], "content": row[1]} for row in cursor.fetchall()]

def clear_user_memory(user_id):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM chat_memory WHERE user_id=?", (user_id,))

# ==================== 样式 ====================
st.markdown("""
<style>
    .stApp { background-color: #f4f7f6; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    /* 聊天气泡样式 */
    [data-testid="stChatMessage"] { border-radius: 15px; padding: 15px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
    [data-testid="stChatMessage"][data-baseweb="block"]:has([alt="assistant avatar"]) { background-color: #ffffff; border-left: 5px solid #8A2BE2; }
    [data-testid="stChatMessage"][data-baseweb="block"]:has([alt="user avatar"]) { background-color: #f3e5f5; border-right: 5px solid #9C27B0; }
</style>
""", unsafe_allow_html=True)

# ==================== 主界面标题 ====================
st.markdown("<h1 style='text-align: center; color: #333;'>💭 校园新生指南</h1>", unsafe_allow_html=True)


# ==================== 侧边栏 ====================
with st.sidebar:
    st.markdown("### 👤 身份认证")
    current_user = st.text_input("请输入学号/昵称开启记忆：", value="Guest")
    
    if "messages" not in st.session_state or current_user != st.session_state.get("last_user", ""):
        st.session_state.messages = load_user_memory(current_user)
        if not st.session_state.messages:
            welcome_msg = f"你好，{current_user}！我是你的校园助手！"
            st.session_state.messages.append({"role": "assistant", "content": welcome_msg})
            save_message(current_user, "assistant", welcome_msg)
        st.session_state.last_user = current_user
        st.session_state.quick = None

    st.markdown("---")
    st.markdown("### 📌 快捷提问")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📚 图书馆", use_container_width=True): st.session_state.quick = "图书馆几点开门？"
        if st.button("🍔 食堂", use_container_width=True): st.session_state.quick = "学校有几个食堂？"
    with c2:
        if st.button("💳 校园卡", use_container_width=True): st.session_state.quick = "学生卡丢了怎么办？"
        if st.button("📖 选课", use_container_width=True): st.session_state.quick = "选课系统怎么登录？"

    st.markdown("---")
    if st.button("🗑️ 清空当前记忆", use_container_width=True):
        clear_user_memory(current_user)
        st.session_state.messages = []
        st.rerun()

# ==================== 主界面对话 ====================
for i, msg in enumerate(st.session_state.messages):
    avatar = "🧑‍🎓" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        
        # 反馈纠错按钮 (仅在最后一条AI回复下显示)
        if msg["role"] == "assistant" and i == len(st.session_state.messages) - 1 and i > 0:
            last_user_msg = st.session_state.messages[i-1]["content"]
            with st.expander("💡 答案有误？帮我纠错进化"):
                correct_ans = st.text_area("正确答案应该是：", key=f"correct_{i}")
                if st.button("提交并让我学习", key=f"btn_{i}"):
                    if correct_ans:
                        learn_new_knowledge(last_user_msg, correct_ans) # 调用 RAG.py 写入向量库
                        with sqlite3.connect(DB_PATH) as conn:
                            conn.execute("INSERT INTO feedback_log VALUES (?, ?, ?, ?)", 
                                         (current_user, last_user_msg, msg["content"], correct_ans))
                        st.success("🎉 太棒了！我已经将这个知识点记入大脑。")

# ==================== 处理输入 ====================
prompt = st.session_state.quick if st.session_state.get("quick") else st.chat_input("💬 输入你的问题...")
if st.session_state.get("quick"): st.session_state.quick = None 

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    save_message(current_user, "user", prompt)
    with st.chat_message("user", avatar="🧑‍🎓"):
        st.markdown(prompt)
    
    # 提取最近上下文（最多4条）
    context_str = ""
    if len(st.session_state.messages) > 1:
        recent = st.session_state.messages[-5:-1]
        context_str = "【历史记录】\n" + "\n".join([f"{m['role']}: {m['content']}" for m in recent]) + "\n\n"
    
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 检索知识库中..."):
            answer = ask(question=prompt, history_context=context_str) 
        st.markdown(answer)
    
    st.session_state.messages.append({"role": "assistant", "content": answer})
    save_message(current_user, "assistant", answer)
    st.rerun()
