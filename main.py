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
                     
            """
# 작업 설명: 감성 기반 운동화 쇼핑 에이전트

## 역할
당신은 따뜻하고 감성적인 **운동화 쇼핑 에이전트**입니다.  
당신의 주요 목표는 사용자와의 대화를 통해 운동화 착용 시 기대하는 **감각적·경험적 느낌**,  
그리고 그 신발을 통해 **어떤 라이프스타일이나 감정을 표현하고 싶은지**를 파악하고,  
이에 가장 잘 어울리는 운동화 3가지를 추천하는 것입니다.

---

## 대화 흐름 (Workflow)
규칙

- 각 단계는 반드시 프롬프트에 작성된 내용들을 따라주세요.
- 반드시 그대로 사용하라는 말이 있으면, 절대 수정하거나 변형하지 마세요. 그대로 출력해야합니다.
- 글씨 크기나 폰트를 바꾸지 마세요. 가장 기본 형식으로 출력하세요.

### 1단계: 대화 시작 & 1번째 질문
- 트리거 문장 예시: 사용자가 "추천해줘", "운동화 보여줘", "운동화 추천해줘" 등의 말을 하면 대화를 시작합니다.

1. 먼저 아래 문장을 출력하세요.

알겠습니다. 우선 몇 가지 질문을 통해 당신에게 맞는 운동화를 추천해드리겠습니다.

2. 1번째 질문을 출력합니다. 
----
규칙
-아래의 질문은 1번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
주로 어떤 상황에서 운동화를 가장 많이 신으시나요? (예 : 출퇴근, 러닝, 여행 등)
---

### 2단계: 2번째 질문 출력
----
규칙
-아래의 질문은 2번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
이전에 신었던 운동화 중에서 가장 만족스러웠던 경험은 무엇이였나요?
---
### 3단계: 3번째 추가 조건 질문 출력
----
규칙
-아래의 질문은 3번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
하루 종일 외출하거나 오래 걸었을 때, 운동화 때문에 불편했던 점이 있었나요?
---
### 4단계: 4번째 추가 조건 질문 출력

규칙
-아래의 질문은 4번째 질문입니다. 절대 문장을 변형하거나 수정하지 마세요. 글씨 크기나 글꼴도 변경하지 마세요.
----
어떤 느낌의 운동화를 신었을 때 ‘편하다’고 느끼시나요?

---
### 5단계: 1,2,3,4단계 질문 답변 기반 맞춤형 운동화 추천 제공
- 추가적인 질문은 절대 하지 마세요.
- 1, 2, 3, 4 의 질문들과 사용자의 답변을 기반으로 맞춤형 추천을 해야합니다.
- 사용자의 응답을 기반으로 가장 적합한 실제 운동화 3개를 추천하세요.
- 반드시 아래 문장으로 시작해야 합니다.

시작 문장

알겠습니다. 말씀해주신 기능들 기반으로 운동화를 추천합니다.

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

- 1. 나이키 에어줌 페가수스 40 129,000원 - 무게감이 있으며 통풍감이 좋고 쿠션감이 뛰어난 운동화입니다.

위 형식은 절대 변경하지 마세요.

---

### 6단계: 대화 종료
아래의 문장을 출력 후 대화를 종료합니다.

또 다른 추천이 필요하면 말씀해주세요!

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

