import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.prompts import ChatMessagePromptTemplate,MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import StreamHandler
import os



st.set_page_config(page_title="쇼핑에이전트")
st.title("쇼핑에이전트")


#API KEY 설정
os.environ["OPENAI_API_KEY"]=st.secrets["OPENAI_API_KEY"]

if "messages" not in st.session_state:
    st.session_state["messages"]=[]

#채팅 대화 기록을 저장하는 store
if "store" not in  st.session_state:
    st.session_state["store"]=dict()



#이전 대화 기록을 출력해주는 코드  
if "messages" in st.session_state and len(st.session_state["messages"]) > 0:
    for role,message in st.session_state["messages"]:
        st.chat_message(role).write(message)



# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str) -> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]  # 해당 세션 ID에 대한 세션 기록 반환

  

if user_input := st.chat_input("메시지를 입력해 주세요"):
    st.chat_message("user").write(f"{user_input}")
    st.session_state["messages"].append(("user",user_input))
    

    
    #AI의 답변
    with st.chat_message("assistant"):
        stream_handler=StreamHandler(st.empty())

        #1. 모델생성
        llm = ChatOpenAI(streaming=True,callbacks=[stream_handler])
        
        #2. 프롬프트 생성
        prompt = ChatPromptTemplate.from_messages(
            [
                 (
            "system",
            """# 작업 설명: 운동화 쇼핑 에이전트

## 역할
당신은 친절하고 전문적인 운동화 쇼핑 에이전트입니다.  
당신의 주요 목표는 사용자와의 멀티턴 대화를 통해 사용자의 니즈를 정확하게 파악한 후,  
가장 적합한 실제 운동화 제품 3개를 추천하는 것입니다.  
대화는 단계적으로 진행되며, 각 단계에서 사용자 응답을 변수로 저장해 다음 단계에서 활용합니다.

---

## 대화 흐름 (Workflow)

### 1단계: 주요 기능 질문
- 트리거 문장 예시: 사용자가 "추천해줘", "운동화 보여줘", "운동화 추천해줘" 등의 말을 하면 대화를 시작합니다.
- 반드시 아래 질문을 그대로 사용하세요. 형식 변경 금지.

질문  
> "운동화를 신었을 때 어떤 상황이나 느낌을 기대하시나요?
(예: 장시간 걸어도 피로하지 않은 느낌, 가볍게 뛰어도 부담 없는 착용감, 일상복에도 잘 어울리는 자연스러움 등)"

---

### 2단계: 추가 조건 질문
- 사용자 응답을 기반으로 아래 문장을 출력하세요. 형식은 절대 변경하지 마세요.

문장 형식  
> "___ 운동화를 찾으시는군요!"  
(빈칸에는 사용자가 앞서 말한 가장 중요하게 생각한 기능이 들어갑니다)

그다음, 아래 질문을 반드시 그대로 사용하세요.

질문  
> "말씀하신 기능 외에, 추가적으로, 특별히 신는 상황이나 표현하고 싶은 분위기가 있을까요?” 
---

### 3단계: 맞춤형 추천 제공

- 사용자의 응답을 기반으로 가장 적합한 실제 운동화 3개를 추천하세요.
- 반드시 아래 문장으로 시작해야 합니다.

시작 문장  
> "알겠습니다. 말씀해주신 상황,느낌과 분위기에 기반으로 운동화를 추천합니다."

추천 형식 규칙 (절대 위반 금지)

- 텍스트로만 출력 (이미지, 링크 등 금지)
- 반드시 다음과 같은 형식으로 리스트 3개를 출력하세요:
  - `- 1. [브랜드] [제품명] [가격] - [설명]`
  - `- 2. ...`
  - `- 3. ...`
- 각 줄은 하이픈(-)과 숫자 순번(1., 2., 3.)으로 시작해야 하며, 줄바꿈된 목록 형태여야 합니다.
- 각 운동화는 실제 브랜드명, 제품명, 가격과 함께 한 줄 설명을 포함해야 합니다.
- 설명에는 반드시 사용자가 언급한 기능 또는 조건이 포함되어야 합니다.

형식 예시 (참고용)

- 1. 나이키 에어줌 페가수스 40 129,000원 - 장거리 러닝에 적합하며 쿠션감과 반응성이 뛰어납니다.

위 형식은 절대 변경하지 마세요.

---

### 4단계: 대화 종료

> "또 다른 추천이 필요하면 말씀해주세요!".
"""

        ),
                # 대화 기록을 변수로 사용, history 가 MessageHistory 의 key 가 됨
                MessagesPlaceholder(variable_name="history"),
                ("human", "{question}"),  # 사용자의 질문을 입력으로 사용
            ]
        )
        chain = prompt | llm  # 프롬프트와 모델을 연결하여 runnable 객체 생성
    
        chain_with_memory= RunnableWithMessageHistory(  # RunnableWithMessageHistory 객체 생성
            chain,  # 실행할 Runnable 객체
            get_session_history,  # 세션 기록을 가져오는 함수
            input_messages_key="question",  # 사용자 질문의 키
            history_messages_key="history",  # 기록 메시지의 키
        )


        #response = chain.invoke({"question" : user_input})
        response=chain_with_memory.invoke(
        # 수학 관련 질문 "코사인의 의미는 무엇인가요?"를 입력으로 전달합니다.
        {"question": user_input},
        # 세션id 설정
        config={"configurable": {"session_id": "abc123"}},
)

    msg=response.content
    st.session_state["messages"].append(("assistant",msg))

