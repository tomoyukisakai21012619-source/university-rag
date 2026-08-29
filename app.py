import os
import json
import streamlit as st
from query import query_rag

st.set_page_config(page_title="大学認証評価 Q&Aシステム", page_icon="🎓", layout="wide")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
VOYAGE_API_KEY    = os.environ.get("VOYAGE_API_KEY", "")
PINECONE_API_KEY  = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX    = os.environ.get("PINECONE_INDEX", "university-rag")

# ユーザー情報: 環境変数 AUTH_USERS に JSON 形式で格納
# 例: {"sakai": "password123", "yamada": "pass456"}
AUTH_USERS = json.loads(os.environ.get("AUTH_USERS", "{}"))


def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        st.title("🎓 大学認証評価 Q&Aシステム")
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.subheader("ログイン")
            username = st.text_input("ユーザー名")
            password = st.text_input("パスワード", type="password")
            if st.button("ログイン", use_container_width=True):
                if username in AUTH_USERS and AUTH_USERS[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.rerun()
                else:
                    st.error("ユーザー名またはパスワードが違います")
        return False
    return True


def main():
    if not check_login():
        return

    st.title("🎓 大学認証評価 Q&Aシステム")

    with st.sidebar:
        st.markdown(f"👤 {st.session_state.get('username', '')}")
        if st.button("ログアウト"):
            st.session_state.logged_in = False
            st.rerun()
        st.markdown("---")
        st.markdown("### 使い方")
        st.markdown("""
- 大学名・地域・規模など自由に質問できます
- 例：「関西の私立大学で地域連携が評価された事例は？」
- 例：「偏差値50以上の大学の改善指摘事項を教えて」
- 例：「○○大学の認証評価結果は？」
        """)
        st.markdown("---")
        top_k = st.slider("参照チャンク数", min_value=5, max_value=30, value=20)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("質問を入力してください...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("検索・回答生成中..."):
                try:
                    response_text, matches, filters = query_rag(
                        question=question,
                        anthropic_api_key=ANTHROPIC_API_KEY,
                        voyage_api_key=VOYAGE_API_KEY,
                        pinecone_api_key=PINECONE_API_KEY,
                        pinecone_index_name=PINECONE_INDEX,
                        top_k=top_k,
                        history=st.session_state.messages,
                    )

                    st.markdown(response_text)

                    if filters:
                        active = {k: v for k, v in filters.items() if v and k != "地域名"}
                        if active:
                            with st.expander("🔍 適用された絞り込み条件"):
                                st.json(active)

                    with st.expander(f"📄 参照した資料（{len(matches)}件）"):
                        for i, match in enumerate(matches[:10], 1):
                            meta = match["metadata"]
                            st.markdown(
                                f"**{i}.** {meta.get('大学名','不明')} — {meta.get('報告書種別','不明')} p.{meta.get('page','?')} "
                                f"（スコア: {match['score']:.3f}）"
                            )

                    st.session_state.messages.append({"role": "assistant", "content": response_text})

                except Exception as e:
                    st.error(f"エラーが発生しました: {e}")


if __name__ == "__main__":
    main()
