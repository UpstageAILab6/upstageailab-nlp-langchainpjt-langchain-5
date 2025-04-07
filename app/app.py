# app/app.py
import os
import streamlit as st
from model.model import load_chain
from langchain.callbacks import tracing_v2_enabled
from langsmith import Client

def run_app():
    st.title("스포츠 룰 질의응답 시스템")
    # 예시: data/pdf 폴더 내 PDF 파일들을 사용
    pdf_source = "data/pdf/"
    chain = load_chain(pdf_source)

    # 세션 상태에 대화 내역 초기화 (없을 경우)  
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    question = st.text_input("궁금한 내용을 입력하세요:")
    if st.button("질문 전송") and question:
        with st.spinner("답변 생성 중..."):
            # answer = chain.invoke(question)
            history_str = "\n".join(st.session_state.conversation_history[-10:]) if st.session_state.conversation_history else ""
            with tracing_v2_enabled():
                answer = chain.invoke({"question": question, "history": history_str})

        st.subheader("답변")
        # st.write(answer)
        st.markdown(answer)
        # 대화 내역에 현재 대화 추가
        st.session_state.conversation_history.append("User: " + question)
        st.session_state.conversation_history.append("Assistant: " + answer)
