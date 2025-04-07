# main.py
import os

from model.model import load_chain
from langsmith import Client
from langchain.callbacks import tracing_v2_enabled
from app.app import run_app
from streamlit.runtime.scriptrunner import script_run_context as stctx

def run_cli():
    """
    CLI 모드: 터미널에서 실행 시 질문을 입력받아 답변을 출력합니다.
    """
    import os
    pdf_path = "data/pdf/"
    if not os.path.exists(pdf_path):
        print("지정한 PDF 파일이 존재하지 않습니다.")
        return

    chain = load_chain(pdf_path)
    conversation_history = []  # 대화 내역 리스트 생성
    MAX_HISTORY = 10           # 최근 10개 챗 히스토리 기억 설정

    print("알고 싶은 경기 규칙을 입력하세요...")
    while True:
        question = input("질문 (종료하려면 '종료' 입력): ").strip()
        if question.lower() in ['종료', 'exit', 'quit']:
            print("프로그램을 종료합니다.")
            break 
        
        # 최근 대화 내역을 문자열로 생성 (대화 히스토리가 있을 경우)
        history_str = "\n".join(conversation_history[-MAX_HISTORY:]) if conversation_history else ""
        
        # tracing 기능을 활성화한 상태에서 질문과 히스토리를 체인에 전달합니다.
        with tracing_v2_enabled():
            answer = chain.invoke({"question": question, "history": history_str})

        print("답변:")
        print(answer)
        
        # 대화 내역에 현재 대화 추가
        conversation_history.append("User: " + question)
        conversation_history.append("Assistant: " + answer)



def run_streamlit():
    """
    Streamlit 모드: app 모듈의 run_app() 함수를 호출하여 웹 인터페이스를 실행합니다.
    """
    run_app()

def main():
    """
    실행 환경에 따라 CLI 모드 또는 Streamlit 모드를 선택.
    streamlit.runtime.scriptrunner.script_run_context.get_script_run_ctx()를 이용해
    Streamlit 런타임 환경을 감지합니다.
    """
    
    try:
        if stctx.get_script_run_ctx() is not None:
            run_streamlit()
        else:
            run_cli()
    except Exception:
        run_cli()


    

if __name__ == "__main__":
    main()


# 모듈 테스트 예시
# if __name__ == "__main__":
#     sample_question = "최근 AAPL 주식 가격은 어떻게 되나요?"
#     try:
#         answer = MCPChain.invoke(sample_question)
#         print("답변:", answer)
#     except Exception as e:
#         print("오류 발생:", e)

