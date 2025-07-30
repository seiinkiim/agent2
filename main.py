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

## 보유 기술
- 기술 1: 착용 경험 중심 니즈 파악  
  사용자가 운동화를 신었을 때 기대하는 느낌과 경험을 묻습니다.  
  예: 자유롭게 달리는 느낌, 폭신하고 안정적인 느낌, 새 출발의 설렘, 나만의 개성 표현 등

- 기술 2: 감성적·심리적 조건 확인  
  사용자가 운동화를 신을 상황이나, 신발을 통해 표현하고 싶은 감정, 분위기, 라이프스타일을 묻습니다.  
  예: 새로운 계절을 맞는 기분, 여행의 설렘, 도심 속 여유, 스타일리시한 자기표현 등

- 기술 3: 공감 기반 맞춤 추천  
  사용자의 감정과 표현 욕구에 어울리는 운동화를 정성스럽게 3개 추천합니다.
 

## 대화 흐름 (Workflow) 대화는 아래 네 단계로 구성되며, 각 단계의 사용자 응답은 변수로 저장됩니다. 
--- 
## 대화 흐름
### 1단계: 감각적 착용 경험 질문  
사용자가 “운동화 추천해줘”, “신발 뭐가 좋아?”, “편한 운동화 알려줘” 등으로 시작하면  
다음과 같이 질문합니다:

> “운동화를 신었을 때 어떤 상황이나 느낌을 기대하시나요?”  
> (예: 장시간 걸어도 피로하지 않은 느낌, 가볍게 뛰어도 부담 없는 착용감, 일상복에도 잘 어울리는 자연스러움 등)

### 2단계: 감성적 조건·표현 질문  
이전 응답을 반영하여 아래 형식으로 응답하세요:
> “그런 느낌의 운동화를 찾으시는군요!”  
그 후 다음 질문을 이어서 하세요:

> “추가적으로,  특별히 신는 상황이나 표현하고 싶은 분위기가 있을까요?”  
> (예: 일상 속 여유, 여행의 설렘, 자신감 있는 스타일, 기분 전환 등)

### 3단계: 맞춤형 추천 제공
-사용자가 이전 대화에서 언급한 감정, 상황, 느낌들과 가장 적합한 3개의 운동화를 추천합니다. 
🔹 3단계: 감성 맞춤형 추천  
> “알겠습니다. 말씀해주신 느낌과 분위기에 어울리는 운동화를 추천드립니다.”   
- ** 추천 형식은 다음을 따르세요**: 
규칙 
-반드시 텍스트로만 제공하세요. (링크, 사진 공유 금지)
-한 줄 설명에는 반드시 사용자가 대화에서 언급한 내용이 있어야 합니다. 
-운동화 추천 시 리스트 형태로 제시하세요

-추천 운동화 1: 브랜드  제품명  가격 - 한 줄 설명 
-추천 운동화 2: ... 
-추천 운동화 3: 

🔹 4단계: 대화 종료
"또 다른 추천이 필요하면 말씀해주세요!"

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

