import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Union, List
from urllib.parse import quote
import math
import random
import gspread
from google.oauth2 import service_account
import gspread
from datetime import datetime
import numpy as np
import pandas as pd
import redis
import requests
import uvicorn
from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
    RedisChatMessageHistory,
)
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, OpenAI
from pydantic import BaseModel
from oauth2client.service_account import ServiceAccountCredentials
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema,
    DataType, Collection
)
from langdetect import detect
from openai import OpenAI as OpenAIClient

# 트리거 키워드 정의 (한국어, 영어만)
GREETING_KEYWORDS = [
    # 한국어 인사말
    "안녕", "안녕하세요", "안녕하십니까", "반갑습니다", "안뇽", "하이", "헬로", "헬로우",
    "시작", "출발", "시작하기",
    
    # 영어 인사말
    "hi", "hello", "hey", "welcome", "greetings", "start"
]

AI_SEARCH_KEYWORDS = [
    # 한국어
    "ai 검색", "ai검색", "ai 상품 검색", "ai상품검색", "ai 추천", "ai추천",
    "상품 찾기", "상품찾기", "상품 검색", "상품검색", "제품 찾기", "제품찾기",
    "ai 픽", "ai픽", "상품 추천", "상품추천",
    
    # 영어
    "ai search", "ai product search", "ai recommendation", "ai picks",
    "product search", "find product", "search product", "recommend product",
    "ai shopping", "smart search", "intelligent search"
]

executor = ThreadPoolExecutor()

# ✅ 환경 변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
REDIS_URL = "redis://localhost:6379/0"
VERIFY_TOKEN = os.getenv('VERIFY_TOKEN')
PAGE_ACCESS_TOKEN = os.getenv('PAGE_ACCESS_TOKEN')

LLM_MODEL = "gpt-4.1-mini-2025-04-14"
EMB_MODEL = "text-embedding-3-small"

CSV_PATH = "카테고리목록.csv"
# "카테고리" 목록 로드 (엑셀/CSV)
df_categories = pd.read_csv(CSV_PATH, encoding="utf-8-sig")
categories = df_categories['카테고리목록'].dropna().unique().tolist()

# 클라이언트 및 래퍼
client = OpenAIClient(api_key=API_KEY)
llm = OpenAI(api_key=API_KEY, model=LLM_MODEL, temperature=0)
embedder = OpenAIEmbeddings(api_key=API_KEY, model=EMB_MODEL)

# API_URL 설정 (URL로 변경)
API_URL = "https://port-0-chatmall2-mddsxz1wc930914e.sel5.cloudtype.app"
# print(f"🔍 로드된 VERIFY_TOKEN: {VERIFY_TOKEN}")
# print(f"🔍 로드된 PAGE_ACCESS_TOKEN: {PAGE_ACCESS_TOKEN}")
# print(f"🔍 로드된 API_KEY: {API_KEY}")
# print(f"🔍 로드된 API_URL: {API_URL}")

# ─── Milvus import & 연결 ───────────────────────────────────────────────
connections.connect(
    alias="default",
    host="114.110.135.96",
    port="19530"
)
print("✅ Milvus에 연결되었습니다.")

# 컬렉션 이름
collection_name = "ownerclan_weekly_0428"
collection = Collection(name=collection_name)

# OpenAI Embedding 모델 (쿼리용)
emb_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    openai_api_key=os.getenv("OPENAI_API_KEY")
)
print(f"\n저장된 엔트리 수: {collection.num_entities}")
print("사용자 정보는 JSON 파일에 저장됩니다.")

# JSON 파일 기반 사용자 데이터 저장소
USER_DATA_FILE = "user_data.json"
ORDER_DATA_FILE = "order_data.json"
CONVERSATION_DATA_FILE = "facebook_conversations.json"

# 상품 캐시 (전역 선언)
PRODUCT_CACHE = {}
PROCESSED_MESSAGES = set()
BOT_MESSAGES = set()
PROCESSING_USERS = set()  # 처리 중인 사용자 추적

def load_json_data(file_path: str) -> dict:
    """JSON 파일에서 데이터 로드"""
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    except Exception as e:
        print(f"JSON 로드 오류 ({file_path}): {e}")
        return {}

def save_json_data(file_path: str, data: dict) -> bool:
    """JSON 파일에 데이터 저장"""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"JSON 저장 오류 ({file_path}): {e}")
        return False

def convert_to_serializable(obj):
    """JSON 직렬화를 위한 int 변환 함수"""
    if isinstance(obj, (np.int64, np.int32, np.float32, np.float64)):
        return obj.item()
    return obj

class UserDataManager:
    """사용자별 데이터 관리 클래스 - 주문 진행용 임시 저장소"""
    
    @staticmethod
    def get_user_data(sender_id: str):
        """사용자 데이터 조회 (없으면 기본값 생성)"""
        all_data = load_json_data(USER_DATA_FILE)
        if sender_id not in all_data:
            all_data[sender_id] = {
                "product_code": None,
                "product_name": None,
                "selected_option": None,
                "extra_price": 0,
                "quantity": 0,
                "unit_price": 0,
                "shipping_fee": 0,
                "total_price": 0,
                "bundle_size": 0,  # 묶음당 수량 추가
                "bundles_needed": 1,  # 필요한 묶음 수 추가
                "product_info": {},
                "order_status": "none",
                "last_updated": time.time()
            }
            save_json_data(USER_DATA_FILE, all_data)
        return all_data[sender_id]
    
    @staticmethod
    def update_user_data(sender_id: str, **kwargs):
        """사용자 데이터 업데이트 (주문 진행 중에만 사용)"""
        all_data = load_json_data(USER_DATA_FILE)
        if sender_id not in all_data:
            all_data[sender_id] = {
                "product_code": None,
                "product_name": None,
                "selected_option": None,
                "extra_price": 0,
                "quantity": 0,
                "unit_price": 0,
                "shipping_fee": 0,
                "total_price": 0,
                "max_quantity": 0,
                "product_info": {},
                "order_status": "none",
                "last_updated": time.time()
            }
        
        all_data[sender_id].update(kwargs)
        all_data[sender_id]["last_updated"] = time.time()
        
        if save_json_data(USER_DATA_FILE, all_data):
            print(f"[TEMP_STORAGE] 사용자 {sender_id} 임시 데이터 저장: {kwargs}")
        else:
            print(f"[TEMP_STORAGE] 사용자 {sender_id} 데이터 저장 실패")

    @staticmethod
    def clear_user_data(sender_id: str):
        """사용자 JSON 데이터만 삭제"""
        try:
            all_data = load_json_data(USER_DATA_FILE)
            if sender_id in all_data:
                del all_data[sender_id]
                save_json_data(USER_DATA_FILE, all_data)
                print(f"[JSON_CLEAR] 사용자 {sender_id} JSON 데이터 삭제 완료")
                return True
            return True
        except Exception as e:
            print(f"JSON 데이터 삭제 오류: {e}")
            return False
        
class OrderDataManager:
    """주문 정보 관리 클래스"""
    
    @staticmethod
    def get_order_data(sender_id: str):
        """주문 데이터 조회 (없으면 기본값 생성)"""
        all_data = load_json_data(ORDER_DATA_FILE)
        if sender_id not in all_data:
            all_data[sender_id] = {
                "receiver_name": None,
                "address": None,
                "phone_number": None,
                "email": None,
                "product_name": None,
                "selected_option": None,
                "quantity": 0,
                "total_price": 0,
                "facebook_name": None,
                "order_status": "none",
                "last_updated": time.time()
            }
            save_json_data(ORDER_DATA_FILE, all_data)
        return all_data[sender_id]
    
    @staticmethod
    def update_order_data(sender_id: str, **kwargs):
        """주문 데이터 업데이트"""
        all_data = load_json_data(ORDER_DATA_FILE)
        if sender_id not in all_data:
            all_data[sender_id] = {
                "receiver_name": None,
                "address": None,
                "phone_number": None,
                "email": None,
                "product_name": None,
                "selected_option": None,
                "quantity": 0,
                "total_price": 0,
                "facebook_name": None,
                "order_status": "none",
                "last_updated": time.time()
            }
        
        all_data[sender_id].update(kwargs)
        all_data[sender_id]["last_updated"] = time.time()
        
        if save_json_data(ORDER_DATA_FILE, all_data):
            print(f"[ORDER_DATA] 주문 정보 저장: {kwargs}")
        else:
            print(f"[ORDER_DATA] 주문 정보 저장 실패")
    
    @staticmethod
    def clear_order_data(sender_id: str):
        """주문 데이터 완전 삭제"""
        try:
            all_data = load_json_data(ORDER_DATA_FILE)
            if sender_id in all_data:
                del all_data[sender_id]
                save_json_data(ORDER_DATA_FILE, all_data)
                print(f"[ORDER_CLEAR] 주문 데이터 완전 삭제: {sender_id}")
                return True
            return True
        except Exception as e:
            print(f"주문 데이터 삭제 오류: {e}")
            return False

class ConversationLogger:
    """Facebook 대화내용 사용자별 JSON 저장 클래스"""
    
    @staticmethod
    def load_conversations() -> dict:
        """저장된 대화 기록 로드"""
        try:
            if os.path.exists(CONVERSATION_DATA_FILE):
                with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"대화 기록 로드 오류: {e}")
            return {}

    @staticmethod
    def save_conversations(data: dict) -> bool:
        """대화 기록 저장"""
        try:
            with open(CONVERSATION_DATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"대화 기록 저장 오류: {e}")
            return False
    
    @staticmethod
    def log_message(sender_id: str, message_type: str, content: str) -> bool:
        """
        개별 메시지 로그 저장
        
        Args:
            sender_id: Facebook 사용자 ID
            message_type: 'user' 또는 'bot'
            content: 메시지 내용
        """
        try:
            conversations = ConversationLogger.load_conversations()
            
            # 사용자별 대화 기록 초기화
            if sender_id not in conversations:
                conversations[sender_id] = []
            
            # 메시지 데이터 구성
            message_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "type": message_type,  # 'user' 또는 'bot'
                "message": content
            }
            
            # 대화 기록에 추가
            conversations[sender_id].append(message_data)
            
            # JSON 파일에 저장
            success = ConversationLogger.save_conversations(conversations)
            
            if success:
                print(f"[CONVERSATION] 메시지 저장: {sender_id} - {message_type}")
            
            return success
            
        except Exception as e:
            print(f"[CONVERSATION] 메시지 로그 오류: {e}")
            return False
    
    @staticmethod
    def log_user_message(sender_id: str, user_message: str) -> bool:
        """사용자 메시지 로그"""
        return ConversationLogger.log_message(sender_id, "user", user_message)
    
    @staticmethod
    def log_bot_message(sender_id: str, bot_message: str) -> bool:
        """봇 메시지 로그"""
        return ConversationLogger.log_message(sender_id, "bot", bot_message)

def init_google_sheets():
    """Google Sheets 연결 초기화 (환경변수 + Fallback)"""
    try:
        scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        
        # 환경변수에서 JSON 내용 가져오기
        credentials_json = os.getenv('GOOGLE_CREDENTIALS_JSON')
        
        if credentials_json:
            print("[SHEETS] 환경변수에서 인증 정보 로드 중...")
            try:
                # JSON 문자열을 딕셔너리로 파싱
                credentials_dict = json.loads(credentials_json)
                credential = ServiceAccountCredentials.from_json_keyfile_dict(
                    credentials_dict, scope
                )
                print("✅ 환경변수 인증 성공")
                
            except json.JSONDecodeError as e:
                print(f"❌ 환경변수 JSON 파싱 실패: {e}")
                print("환경변수가 올바른 JSON 형식인지 확인하세요")
                return None
                
        else:
            print("[SHEETS] 환경변수 없음 - 로컬 파일 사용")
            # Fallback: 로컬 JSON 파일 사용
            json_key_path = "facebook-chatbot-2025-f79c8cbf74cf.json"
            
            if not os.path.exists(json_key_path):
                print(f"❌ JSON 파일을 찾을 수 없습니다: {json_key_path}")
                return None
                
            credential = ServiceAccountCredentials.from_json_keyfile_name(
                json_key_path, scope
            )
            print("✅ 로컬 파일 인증 성공")
        
        # gspread 클라이언트 생성
        gc = gspread.authorize(credential)
        
        # 스프레드시트 연결
        spreadsheet_url = "https://docs.google.com/spreadsheets/d/1N-aD64bTw1tKHRKyDksbG2xvvVNsOZPFMcQIxgTUm_4/edit?pli=1&gid=0#gid=0"
        doc = gc.open_by_url(spreadsheet_url)
        
        # 워크시트 선택
        sheet = doc.worksheet("Orders")
        
        print("✅ Google Sheets 연결 성공")
        return sheet
        
    except Exception as e:
        print(f"❌ Google Sheets 연결 실패: {e}")
        return None

def send_order_to_sheets(sender_id: str) -> bool:
    """주문 정보를 Google Sheets에 전송"""
    try:
        print(f"[SHEETS] Google Sheets로 주문 정보 전송 시작 - sender_id: {sender_id}")
        
        # Google Sheets 연결
        sheet = init_google_sheets()
        if not sheet:
            print("[SHEETS] Google Sheets 연결 실패")
            # 실패 시 안내 메시지와 홈 버튼 제공
            send_facebook_message(sender_id, 
                "There was a temporary issue with our order processing system.\n")
            import time
            time.sleep(1)
            send_go_home_card(sender_id)
            return False
        
        # 주문 데이터 및 사용자 데이터 가져오기
        order_data = OrderDataManager.get_order_data(sender_id)
        user_data = UserDataManager.get_user_data(sender_id)
        
        # 현재 시간
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 헤더 가져오기
        headers = sheet.row_values(1)  # 첫 번째 행(헤더)만 가져오기
        print(f"[SHEETS] 시트 헤더: {headers}")
        
        # 드롭다운 컬럼 목록 (건드리지 않을 컬럼들)
        dropdown_columns = [
            "Deposit Confirmed?",
            "Order Placed on Korean Shopping Mall?",
            "Order Received by Customer?"
        ]
        
        # 전송할 데이터 준비
        data_mapping = {
            "Order Date": current_time,
            "Who ordered?": order_data.get('facebook_name', ''),
            "Receiver's Name": order_data.get('receiver_name', ''),
            "What did they order?": order_data.get('product_name', ''),
            "Cart Total": f"{order_data.get('total_price', 0):,}원",
            "Grand Total": f"{order_data.get('total_price', 0):,}원",
            "Delivery Address": order_data.get('address', ''),
            "Email": order_data.get('email', ''),
            "phone_number": order_data.get('phone_number', ''),
            "option": order_data.get('selected_option', ''),
            "quantity": order_data.get('quantity', 0),
            "product_code": user_data.get('product_code', ''),
        }
        
        # 드롭다운 컬럼은 건너뛰고 데이터 배열 생성
        row_data = []
        for header in headers:
            if header in dropdown_columns:
                # 드롭다운 컬럼 건너뛰기
                print(f"[SHEETS] 드롭다운 컬럼 건너뛰기: {header}")
                continue
            else:
                # 일반 컬럼은 데이터 추가
                value = data_mapping.get(header, "")
                row_data.append(str(value))
                print(f"[SHEETS] 데이터 추가: {header} = {value}")
        
        print(f"[SHEETS] 전송할 데이터 행: {row_data}")
        
        # 개별 셀 업데이트로 드롭다운 보존
        # 먼저 새 행 번호 찾기
        all_values = sheet.get_all_values()
        next_row = len(all_values) + 1  # 다음 빈 행
        
        print(f"[SHEETS] 새 행 번호: {next_row}")
        
        # 드롭다운이 아닌 컬럼들만 개별 업데이트
        for col_index, header in enumerate(headers, start=1):
            if header not in dropdown_columns:
                value = data_mapping.get(header, "")
                if value:  # 빈 값이 아닌 경우만 업데이트
                    sheet.update_cell(next_row, col_index, str(value))
                    print(f"[SHEETS] 셀 업데이트: 행{next_row}, 열{col_index} ({header}) = {value}")
            else:
                print(f"[SHEETS] 드롭다운 셀 보존: 행{next_row}, 열{col_index} ({header})")
        
        print(f"[SHEETS] 주문 정보 전송 완료!")
        return True
            
    except Exception as e:
        print(f"[SHEETS] 주문 정보 전송 오류: {e}")
        import traceback
        print(f"[SHEETS] 상세 오류:\n{traceback.format_exc()}")
        return False

def clear_user_data(sender_id: str, clear_type: str = "all"):
    """
    사용자 관련 모든 데이터 초기화
    """
    try:
        print(f"[CLEAR_ALL] 사용자 데이터 초기화 시작 - sender_id: {sender_id}, type: {clear_type}")
        
        # 1. Redis 대화 기록 초기화
        try:
            clear_message_history(sender_id)
            print(f"Redis 대화 기록 초기화 완료")
        except Exception as e:
            print(f"Redis 초기화 오류: {e}")
        
        # 2. JSON 임시 데이터 초기화
        try:
            UserDataManager.clear_user_data(sender_id)
            print(f"JSON 임시 데이터 초기화 완료")
        except Exception as e:
            print(f"JSON 데이터 초기화 오류: {e}")
        
        # 3. 주문 데이터 초기화
        if clear_type in ["all", "go_home", "reset", "order_complete", "order_cancel", "ai_search", "greeting"]:
            try:
                OrderDataManager.clear_order_data(sender_id)
                print(f"주문 데이터 초기화 완료")
            except Exception as e:
                print(f"주문 데이터 초기화 오류: {e}")
        
        print(f"사용자 데이터 초기화 완료 - type: {clear_type}")
        return True
        
    except Exception as e:
        print(f"사용자 데이터 초기화 전체 오류: {e}")
        return False

def get_redis():
    return redis.Redis.from_url(REDIS_URL)

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisChatMessageHistory(session_id=session_id, url=REDIS_URL)

def clear_message_history(sender_id: str):
    """
    Redis에 저장된 특정 세션의 대화 기록을 초기화합니다.
    """
    try:
        history = RedisChatMessageHistory(session_id=sender_id, url=REDIS_URL)
        history.clear()  # Redis에서 해당 세션의 모든 메시지 삭제
        print(f"[REDIS_CLEAR] 세션 {sender_id}의 Redis 대화 기록이 초기화되었습니다.")
        return True
    except Exception as e:
        print(f"[REDIS_ERROR] Redis 초기화 오류: {e}")
        return False

# 트리거 함수들
def is_greeting_message(message: str) -> bool:
    """메시지가 인사말인지 확인"""
    message_lower = message.lower().strip()
    return any(keyword in message_lower for keyword in GREETING_KEYWORDS)

def is_ai_search_trigger(message: str) -> bool:
    """메시지가 AI 검색 트리거인지 확인"""
    message_lower = message.lower().strip()
    return any(keyword in message_lower for keyword in AI_SEARCH_KEYWORDS)

def check_and_handle_previous_data(sender_id: str) -> bool:
    """이전 데이터 확인 함수 - 항상 False 반환 (이전 데이터 사용 안함)"""
    # 이전 데이터를 사용하지 않으므로 항상 False 반환
    return False

def clean_html_content(html_raw: str) -> str:
    try:
        html_cleaned = html_raw.replace('\n', '').replace('\r', '')
        html_cleaned = html_cleaned.replace(""", "\"").replace(""", "\"").replace("'", "'").replace("'", "'")
        if html_cleaned.count("<center>") > html_cleaned.count("</center>"):
            html_cleaned += "</center>"
        if html_cleaned.count("<p") > html_cleaned.count("</p>"):
            html_cleaned += "</p>"
        return html_cleaned
    except Exception as e:
        print(f"HTML 정제 오류: {e}")
        return html_raw

def safe_int(val):
    try:
        return int(float(str(val).replace(",", "").replace("원", "").strip()))
    except:
        return 0
    
# FastAPI 인스턴스 생성
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        API_URL,  # 새로운 CloudType URL
        "http://localhost:5050",  # 로컬 개발용
        "https://polecat-precious-termite.ngrok-free.app",  # 기존 ngrok (백업용)
        "https://port-0-chatmall2-mddsxz1wc930914e.sel5.cloudtype.app"  # 명시적 추가
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("response_time_logger")

# 응답 속도 측정을 위한 미들웨어 추가
@app.middleware("http")
async def measure_response_time(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    response.headers["ngrok-skip-browser-warning"] = "1"
    response.headers["X-Frame-Options"] = "ALLOWALL"
    response.headers["Content-Security-Policy"] = "frame-ancestors *"

    if request.url.path == "/webhook":
        print(f"[TEST] Endpoint: {request.url.path}, 처리 시간: {process_time:.4f} 초")
        logger.info(f"[Endpoint: {request.url.path}] 처리 시간: {process_time:.4f} 초")
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="templates")

# 요청 모델
class QueryRequest(BaseModel):
    query: str

class Product_Selections(BaseModel):
    sender_id: str
    product_code: str

class QuantityInput(BaseModel):
    sender_id: str
    product_quantity: int
    product_code: str

# 중복 처리 방지를 위한 전역 변수
PROCESSED_MESSAGES = set()
BOT_MESSAGES = set()  # 봇이 보낸 메시지 추적

def send_facebook_message(sender_id: str, text: str):
    """Facebook Messenger API로 텍스트 메시지 전송 (개선된 버전)"""
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        'recipient': {'id': sender_id},
        'message': {'text': text},
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    try:
        # 타임아웃 설정으로 응답 속도 개선
        response = requests.post(url, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            result = response.json()
            message_id = result.get("message_id")
            print(f"메시지 전송 성공: {text[:50]}... (ID: {message_id})")

            ConversationLogger.log_bot_message(sender_id, text)
            # 봇이 보낸 메시지 ID 기록
            if message_id:
                BOT_MESSAGES.add(message_id)
                
            # 메시지 캐시 정리
            cleanup_message_cache()
        else:
            print(f"메시지 전송 실패: {response.status_code} - {response.text}")
    except requests.exceptions.Timeout:
        print(f"메시지 전송 타임아웃 (10초)")
    except Exception as e:
        print(f"메시지 전송 오류: {e}")

def send_facebook_carousel(sender_id: str, products: list):
    """Facebook Messenger API로 카루셀 메시지 전송 (중복 방지)"""
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    elements = []
    for product in products[:10]:  # 최대 10개
        try:
            price = int(float(product.get("가격", 0)))
            shipping = int(float(product.get("배송비", 0)))
        except:
            price = 0
            shipping = 0
        
        product_code = product.get("상품코드", "")
        
        element = {
            "title": product.get("제목", "상품")[:80],  # 80자 제한
            "subtitle": f"가격: {price:,}원\n배송비: {shipping:,}원\n원산지: {product.get('원산지', '')}",
            "image_url": product.get("이미지", ""),
            "buttons": [
                {
                    "type": "web_url",
                    "url": product.get("상품링크", "#"),
                    "title": "View Product"
                },
                {
                    "type": "postback",
                    "title": "Buy Now",
                    "payload": f"BUY_{product_code}"
                }
            ]
        }
        elements.append(element)
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': elements
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'  # 응답 타입 명시
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            result = response.json()
            message_id = result.get("message_id")
            print(f"카루셀 메시지 전송 성공 (ID: {message_id})")
            
            # Facebook에 전송된 카루셀 내용 그대로 로깅
            carousel_message = "[카루셀 메시지]\n"
            for i, element in enumerate(elements, 1):
                carousel_message += f"카드 {i}:\n"
                carousel_message += f"제목: {element['title']}\n"
                carousel_message += f"{element['subtitle']}\n"
                carousel_message += f"버튼: View Product, Buy Now\n\n"
            
            ConversationLogger.log_bot_message(sender_id, carousel_message.strip())
            
            # ✅ 봇이 보낸 메시지 ID 기록
            if message_id:
                BOT_MESSAGES.add(message_id)
        else:
            print(f"카루셀 메시지 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"카루셀 메시지 전송 오류: {e}")

def get_user_name(sender_id: str) -> str:
    """
    Facebook에서 사용자의 full_name만 가져오기
    
    Args:
        sender_id: Facebook 사용자 ID (예: "8127128490722875")
    
    Returns:
        str: 사용자 이름 (예: "홍길동") 또는 빈 문자열
    """
    try:
        # Graph API URL 구성
        url = f"https://graph.facebook.com/v18.0/{sender_id}"
        
        # name(full_name)만 가져오기
        params = {
            'fields': 'name',
            'access_token': PAGE_ACCESS_TOKEN
        }
        
        print(f"[GET_NAME] 사용자 이름 요청: {sender_id}")
        
        # API 호출
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            user_info = response.json()
            user_name = user_info.get('name', '')
            
            print(f"[GET_NAME] 이름 가져오기 성공: {user_name}")
            return user_name
        else:
            print(f"[GET_NAME] API 호출 실패: {response.status_code}")
            return ""
            
    except Exception as e:
        print(f"[GET_NAME] 사용자 이름 가져오기 오류: {e}")
        return ""

def send_welcome_message(sender_id: str):
    """환영 메시지와 버튼 메뉴 전송 (Facebook 기준)"""
    import time as time_module
    
    # 사용자 이름 가져오기
    user_name = get_user_name(sender_id)

    # 환영 텍스트 메시지
    if user_name:
        welcome_text = (
            f"Welcome! {user_name}\n"
            f"Thank you for contacting ChatMall.\n"
            f"Our Chatbot helps foreigners in Korea shop easily.\n"
            f"Looking for something? Just type it in!"
        )
    else:
        welcome_text = (
            f"Welcome! {user_name}\n"
            f"Thank you for contacting ChatMall.\n"
            f"Our Chatbot helps foreigners in Korea shop easily.\n"
            f"Looking for something? Just type it in!"
        )
    
    send_facebook_message(sender_id, welcome_text)
    time_module.sleep(1)
    
    # ✅ 3단계: 버튼 카드 전송 (title/subtitle 비워둠)
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': '',  
                            'subtitle': '',  
                            'image_url': 'https://drive.google.com/uc?export=view&id=156l_KbzB2bcyyuOXvYiyFrA_bAe1PHk_',
                            'buttons': [
                                {
                                    'type': 'web_url',
                                    'url': 'https://www.chatmall.kr/',
                                    'title': '🌐 Let’s Go ChatMall'
                                },
                                {
                                    'type': 'postback',
                                    'title': '👤 Sign Up Now',
                                    'payload': 'REGISTER'
                                },
                                {
                                    'type': 'postback',
                                    'title': '📦 Track Order',
                                    'payload': 'TRACK_ORDER'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"환영 버튼 카드 전송 성공")
            # AI 검색 버튼을 별도 메시지로 전송
            time_module.sleep(1)
            send_ai_search_button(sender_id)
        else:
            print(f"환영 버튼 카드 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"환영 버튼 카드 전송 오류: {e}")

def send_ai_search_button(sender_id: str):
    """AI 검색 버튼을 Quick Reply로 전송"""
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': '',
                            'subtitle': '',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '🤖 Start My AI Picks',
                                    'payload': 'AI_SEARCH'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"AI 검색 Quick Reply 버튼 전송 성공")
        else:
            print(f"AI 검색 Quick Reply 버튼 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"AI 검색 Quick Reply 버튼 전송 오류: {e}")

def send_ai_search_prompt(sender_id: str):
    """AI 검색 안내 메시지 전송"""
    import time as time_module
    print(f"[DEBUG] send_ai_search_prompt 호출됨! - sender_id: {sender_id}")
    
    # 첫 번째 메시지: AI 검색 안내
    ai_search_text = (
        "🤖 AI Product Search is ON! ✨\n\n"
        "🔎 What are you shopping for today?\n\n"
        "✨ AI picks, just for you! Enter what you're looking for.\n\n"
    )
    
    print(f"[DEBUG] 첫 번째 메시지 전송 중...")
    send_facebook_message(sender_id, ai_search_text)
    print(f"[DEBUG] 첫 번째 메시지 전송 완료!")
    
    # 짧은 딜레이 후 두 번째 메시지
    # print(f"[DEBUG] 0.5초 대기 중...")
    # time_module.sleep(0.5)
    
    # print(f"[DEBUG] 네비게이션 버튼 메시지 전송 중...")
    # send_navigation_buttons(sender_id)
    # print(f"[DEBUG] send_ai_search_prompt 완료!")

def send_navigation_buttons(sender_id: str):
    """네비게이션 버튼 메시지 전송"""
    import time as time_module
    
    print(f"[DEBUG] send_navigation_buttons 시작 - sender_id: {sender_id}")
    
    # 1. 먼저 텍스트 메시지 전송
    navigation_text = (
        "🧹 Click \"Reset\" below to reset the conversation history 🕓\n\n"
        "🏠 Click \"Go Home\" to return to main menu 🏠"
    )
    
    send_facebook_message(sender_id, navigation_text)
    
    # 2. 짧은 딜레이 후 카드 버튼 전송
    time_module.sleep(0.5)
    
    # 3. Generic Template 카드 전송
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Choose your next action from the options below',
                            'subtitle': '',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '♻️ Reset Conversation',
                                    'payload': 'RESET_CONVERSATION'
                                },
                                {
                                    'type': 'postback',
                                    'title': '🏠 Go Home',
                                    'payload': 'GO_HOME'
                                },
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"네비게이션 카드 메시지 전송 성공")
        else:
            print(f"네비게이션 카드 메시지 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"네비게이션 카드 메시지 전송 오류: {e}")
    
    print(f"[DEBUG] send_navigation_buttons 완료")

def send_go_home_card(sender_id: str):
    """Go Home 카드 버튼 전송"""
    import time as time_module
    
    print(f"[DEBUG] send_go_home_card 시작 - sender_id: {sender_id}")
    
    # 짧은 딜레이
    time_module.sleep(0.5)
    
    url = f"https://graph.facebook.com/v18.0/me/messages"

    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Navigation',
                            'subtitle': 'Return to main menu or continue shopping:',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '🏠 Go Home',
                                    'payload': 'GO_HOME'
                                },
                                {
                                    'type': 'postback',
                                    'title': '🤖 AI Search',
                                    'payload': 'AI_SEARCH'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"Go Home 카드 메시지 전송 성공")
        else:
            print(f"Go Home 카드 메시지 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"Go Home 카드 메시지 전송 오류: {e}")
    
    print(f"[DEBUG] send_go_home_card 완료")


def send_quantity_selection(sender_id: str, product_code: str):
    """수량 직접 입력 요청 메시지 전송"""
    import time as time_module
    
    print(f"[QUANTITY_INPUT] 수량 직접 입력 요청 - sender_id: {sender_id}, product_code: {product_code}")
    
    # 상품 정보 확인
    product = PRODUCT_CACHE.get(product_code)
    if not product:
        print(f"[QUANTITY] 상품 캐시에서 찾을 수 없음: {product_code}")
        
        # 사용자 데이터에서 상품 정보 복구 시도
        user_data = UserDataManager.get_user_data(sender_id)
        stored_product = user_data.get("product_info", {})
        
        if stored_product and stored_product.get("상품코드") == product_code:
            print(f"🔄 [QUANTITY_RESTORE] 사용자 데이터에서 상품 정보 복구")
            PRODUCT_CACHE[product_code] = stored_product
            product = stored_product
        else:
            print(f"[QUANTITY] 사용자 데이터에서도 상품 정보 없음")
            send_facebook_message(sender_id, "❌ 상품 정보를 찾을 수 없습니다. 다시 검색해주세요.")
            return
    
    # 묶음배송 수량 확인
    bundle_size_raw = product.get('최대구매수량', 0)
    try:
        bundle_size = int(float(bundle_size_raw)) if bundle_size_raw else 0
    except (ValueError, TypeError):
        bundle_size = 0
    
    print(f"📦 [BUNDLE] 묶음배송 정보:")
    print(f"   - 원본 데이터: {bundle_size_raw}")
    print(f"   - 묶음당 수량: {bundle_size}개")
    
    # 사용자 데이터에 상태 저장
    UserDataManager.update_user_data(
        sender_id,
        product_code=product_code,
        order_status="waiting_quantity",
        bundle_size=bundle_size,
        product_info=product  # 상품 정보 다시 저장
    )
    
    # 묶음배송 안내 메시지 생성
    if bundle_size > 0:
        quantity_message = (
            f"🧮 How many do you want? 🔢\n\n"
            f"📦 Product: {product.get('제목', '상품')}\n"
            f"📊 Combined Shipping: packaged in sets of {bundle_size}\n"
            f"💡 ex: Our bundled shipping rate applies to every 3 items. If you order 6 items, they will be sent in two separate packages (3 items each), and the shipping fee will be applied twice.\n\n"
            f"💬 Please enter the quantity.\n"
            f"(예: 1, 25, 50, 100, 150)"
        )
    else:
        quantity_message = (
            f"🧮 How many do you want? 🔢\n\n"
            f"📦 Product: {product.get('제목', '상품')}\n"
            f"📊 Single Shipment (No Quantity Limit)\n\n"
            f"💬 Please enter the quantity.\n"
            f"(예: 1, 10, 50, 100)"
        )
    
    send_facebook_message(sender_id, quantity_message)
    print(f"✅ [QUANTITY] 수량 입력 안내 메시지 전송 완료")

def handle_quantity_input(sender_id: str, user_message: str) -> bool:
    """수량 입력 처리 함수"""
    try:
        print(f"[QUANTITY_INPUT] 수량 입력 처리 시작: '{user_message}'")
        
        # 사용자 데이터 확인
        user_data = UserDataManager.get_user_data(sender_id)
        
        # 수량 입력 대기 상태인지 확인
        order_status = user_data.get("order_status")
        if order_status != "waiting_quantity":
            return False
        
        product_code = user_data.get("product_code")
        if not product_code:
            send_facebook_message(sender_id, "❌ Product information not found.\n Please try again.")
            return True
        
        # 입력값에서 숫자 추출
        import re
        numbers = re.findall(r'\d+', user_message.strip())
        
        if not numbers:
            send_facebook_message(sender_id, 
                "❌ Please enter a valid number.\n"
                "예: 1, 25, 50, 100")
            return True
        
        try:
            quantity = int(numbers[0])
        except ValueError:
            send_facebook_message(sender_id, 
                "❌ Please enter a valid number.\n"
                "예: 1, 25, 50, 100")
            return True
        
        # 수량 유효성 검사
        if quantity <= 0:
            send_facebook_message(sender_id, 
                "❌ Please enter a quantity of at least 1.")
            return True
        
        # 묶음배송 정보로 배송비 미리 계산해서 안내
        bundle_size = user_data.get("bundle_size", 0)
        product = PRODUCT_CACHE.get(product_code)
        
        if product and bundle_size > 0:
            shipping_fee = int(float(product.get("배송비", 0)))
            bundles_needed = math.ceil(quantity / bundle_size)
            total_shipping = shipping_fee * bundles_needed
            
            # 묶음배송 안내 메시지
            bundle_info = (
                f"📦 Bundled Shipping Details:\n"
                f"   Quantity: {quantity} items\n"
                f"   Bundles: {bundle_size} items/bundle × {bundles_needed} bundles\n"
                f"   Shipping Fee: KRW {shipping_fee:,} × {bundles_needed} = KRW {total_shipping:,}"
            )
            send_facebook_message(sender_id, bundle_info)
            
            import time
            time.sleep(1)
        
        # 수량이 유효하면 주문 확인으로 진행
        print(f"[QUANTITY_VALID] 유효한 수량 입력: {quantity}개")
        send_order_confirmation(sender_id, product_code, quantity)
        return True
        
    except Exception as e:
        print(f"❌ [QUANTITY_ERROR] 수량 처리 오류: {e}")
        send_facebook_message(sender_id, 
            "❌ An error occurred while processing the quantity.\n Please try again.")
        return True

def send_order_confirmation(sender_id: str, product_code: str, quantity: int):
    """주문 확인 카드 버튼 전송"""
    import time as time_module
    
    print(f"🛒 [ORDER_CONFIRM] 주문 확인 카드 전송")
    print(f"   - 상품코드: {product_code}")
    print(f"   - 주문수량: {quantity}개")
    
    # 상품 정보 확인
    product = PRODUCT_CACHE.get(product_code)
    if not product:
        send_facebook_message(sender_id, "❌ Product information not found.")
        return
    
    # 사용자 데이터에서 추가 정보 가져오기
    user_data = UserDataManager.get_user_data(sender_id)
    extra_price = user_data.get("extra_price", 0)
    
    # 가격 및 배송비 계산
    try:
        unit_price = int(float(product.get("가격", 0)))
        shipping_fee_per_bundle = int(float(product.get("배송비", 0)))
        
        # 묶음배송 수량 (max_quantity = 묶음당 수량)
        bundle_size_raw = product.get("최대구매수량", 0)
        try:
            bundle_size = int(float(bundle_size_raw)) if bundle_size_raw else 0
        except (ValueError, TypeError):
            bundle_size = 0
        
        # 옵션 추가 금액 포함한 단가
        item_price = unit_price + extra_price
        total_item_cost = item_price * quantity
        
        # ✅ 정확한 묶음배송 계산
        if bundle_size > 0:
            # 묶음배송: 수량을 묶음 크기로 나누어 필요한 묶음 수 계산
            bundles_needed = math.ceil(quantity / bundle_size)
            total_shipping_cost = shipping_fee_per_bundle * bundles_needed
            
            print(f"📦 [BUNDLE_CALC] 묶음배송 계산 결과:")
            print(f"   - 주문 수량: {quantity}개")
            print(f"   - 묶음 크기: {bundle_size}개")
            print(f"   - 필요 묶음 수: {bundles_needed}묶음")
            print(f"   - 묶음당 배송비: {shipping_fee_per_bundle:,}원")
            print(f"   - 총 배송비: {total_shipping_cost:,}원")
            
            shipping_detail = f"{shipping_fee_per_bundle:,}원 × {bundles_needed}묶음"
        else:
            # 단일 배송: 수량에 관계없이 배송비 1회
            total_shipping_cost = shipping_fee_per_bundle
            bundles_needed = 1
            
            print(f"📦 [SINGLE_CALC] 단일 배송:")
            print(f"   - 주문 수량: {quantity}개")
            print(f"   - 배송비: {total_shipping_cost:,}원")
            
            shipping_detail = f"{total_shipping_cost:,}원"
            
        total_price = total_item_cost + total_shipping_cost
        
    except Exception as e:
        print(f"[PRICE_CALC] 가격 계산 오류: {e}")
        unit_price = 0
        total_shipping_cost = 0
        total_price = 0
        item_price = 0
        bundles_needed = 1
        shipping_detail = "계산 오류"
    
    # 사용자 데이터 업데이트
    UserDataManager.update_user_data(
        sender_id,
        quantity=quantity,
        total_price=total_price,
        shipping_fee=total_shipping_cost,
        bundles_needed=bundles_needed,
        order_status="confirmed"
    )
    
    # 상세한 주문 확인 메시지
    confirmation_text = (
        f"🛒 Would you like to continue with your order?\n\n"
        f"📦 Product: {product.get('제목', '상품')}\n"
        f"🔢 Quantity: {quantity} items\n"
        f"💰 Unit Price: KRW {unit_price:,}"
    )
    
    if extra_price > 0:
        confirmation_text += f"➕ Add-on: KRW {extra_price:,}\n\n"
        
    # 묶음배송 정보 상세 표시
    if bundle_size > 0 and bundles_needed > 1:
        confirmation_text += (
        f"📦 Bundled Shipping: {bundle_size} items/bundle × {bundles_needed} bundles\n"
        f"🚚 Shipping Fee: {shipping_detail} = KRW {total_shipping_cost:,}\n"
        )
    else:
        confirmation_text += f"🚚 Shipping Fee: KRW {total_shipping_cost:,}\n"
        
    confirmation_text += f"💳 Total: KRW {total_price:,}"
    
    send_facebook_message(sender_id, confirmation_text)
    
    time_module.sleep(0.5)
    
    # 확인/취소 버튼 카드 전송
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Your Order',
                            'subtitle': 'Please review your order and use the buttons below to confirm or cancel.',
                            'image_url': product.get('이미지', ''),
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '✅ Correct',
                                    'payload': f'CONFIRM_{product_code}_{quantity}'
                                },
                                {
                                    'type': 'postback',
                                    'title': '✖️ Incorrect',
                                    'payload': f'CANCEL_{product_code}'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"주문 확인 카드 전송 성공")
        else:
            print(f"주문 확인 카드 전송 실패: {response.status_code} - {response.text}")
    except Exception as e:
        print(f"주문 확인 카드 전송 오류: {e}")

def send_final_order_complete(sender_id: str, product_code: str, quantity: int):
    """최종 주문 완료 처리"""
    print(f"[FINAL_ORDER] 주문 정보 수집 시작 - sender_id: {sender_id}")
    
    # 상품 정보 확인
    product = PRODUCT_CACHE.get(product_code)
    if not product:
        send_facebook_message(sender_id, "❌ Product information not found.")
        return
    
    # 주문 정보 수집 시작
    start_order_info_collection(sender_id, product_code, quantity)

def send_option_selection_buttons(sender_id: str, product_code: str):
    """모든 옵션을 Button Template 방식으로 처리"""
    import time as time_module
    
    print(f"🔧 [OPTION] 옵션 선택 시작 - sender_id: {sender_id}, product_code: {product_code}")
    
    product = PRODUCT_CACHE.get(product_code)
    if not product:
        print(f"[OPTION] 상품 정보 없음")
        send_facebook_message(sender_id, "❌ Product information not found.")
        return
    
    options_raw = product.get("조합형옵션", "")
    print(f"🔍 [OPTION] 원본 옵션: '{options_raw}'")
    
    # 옵션이 없는 경우
    if not options_raw or str(options_raw).lower() in ["nan", "", "none", "null"]:
        print(f"[OPTION] 옵션 없음 - 수량 입력으로 이동")
        send_facebook_message(sender_id, "🧾 This item has a single option — please enter the quantity.")
        time_module.sleep(1)
        send_quantity_selection(sender_id, product_code)
        return

    # 옵션 파싱
    try:
        # 줄바꿈이 없는 경우 대비한 파싱
        options_str = str(options_raw).strip()
        
        # 먼저 줄바꿈으로 분리 시도
        if '\n' in options_str:
            options = options_str.split("\n")
        else:
            # 줄바꿈이 없으면 패턴으로 분리 (옵션명,가격,재고 형태)
            import re
            # "이름,숫자,숫자" 패턴을 찾아서 분리
            pattern = r'([^,]+,\d+,\d+)'
            matches = re.findall(pattern, options_str)
            options = matches if matches else [options_str]
        
        # 빈 옵션 제거
        options = [opt.strip() for opt in options if opt.strip()]
        option_count = len(options)
        
        print(f"[OPTION] 총 옵션 개수: {option_count}개")
        print(f"[OPTION] 옵션 리스트: {options}")
        
        if option_count == 0:
            print(f"⚠️ [OPTION] 파싱 후 옵션 없음")
            send_quantity_selection(sender_id, product_code)
            return
        
        # ===== 모든 옵션을 Button Template 방식으로 처리 =====
        print(f"[OPTION] Button Template 방식 사용 ({option_count}개)")
        
        # 안내 메시지 먼저 전송
        send_facebook_message(sender_id, "⚙️ Please select an option:")
        time_module.sleep(1.5)
        
        # 총 메시지 수 계산
        total_messages = math.ceil(len(options) / 3)
        successful_messages = 0
        
        print(f"[OPTION] 전송할 총 메시지 수: {total_messages}")
        
        # 3개씩 그룹으로 나누어 각각 별도 메시지로 전송
        for i in range(0, len(options), 3):
            message_count = (i // 3) + 1
            option_group = options[i:i+3]
            
            print(f"[OPTION] ===== 메시지 {message_count}/{total_messages} 시작 =====")
            print(f"[OPTION] 이번 그룹 옵션: {option_group}")
            
            buttons = []
            
            for j, opt in enumerate(option_group):
                try:
                    print(f"[OPTION] 옵션 {j+1} 처리 중: '{opt}'")
                    parts = opt.split(",")
                    if len(parts) >= 2:
                        name = parts[0].strip()
                        extra_price = float(parts[1].strip()) if parts[1].strip() else 0
                        
                        caption = f"{name}"
                        if extra_price > 0:
                            caption += f" (+{int(extra_price):,}원)"
                        
                        # Facebook 버튼 제목 길이 제한 (20자)
                        if len(caption) > 20:
                            caption = caption[:17] + "..."
                        
                        payload = f'OPTION_{product_code}_{name}_{int(extra_price)}'
                        
                        buttons.append({
                            'type': 'postback',
                            'title': caption,
                            'payload': payload
                        })
                        
                        print(f"[OPTION] 버튼 {j+1} 생성 완료: {caption}")
                    else:
                        print(f"[OPTION] 옵션 형식 오류: {opt} (parts: {parts})")
                        
                except Exception as e:
                    print(f"[OPTION] 개별 옵션 파싱 실패: {opt} → {e}")
                    continue
            
            print(f"[OPTION] 메시지 {message_count} 생성된 버튼 수: {len(buttons)}")
            
            # 버튼이 있을 때만 메시지 전송
            if buttons:
                print(f"[OPTION] 메시지 {message_count} 전송 시작")
                
                url = f"https://graph.facebook.com/v18.0/me/messages"
                
                # Facebook Button Template 사용
                data = {
                    'recipient': {'id': sender_id},
                    'message': {
                        'attachment': {
                            'type': 'template',
                            'payload': {
                                'template_type': 'button',
                                'text': f"📌 Pick your preferred option ({message_count}/{total_messages}):",
                                'buttons': buttons[:3]  # Facebook 제한: 최대 3개 버튼
                            }
                        }
                    },
                    'access_token': PAGE_ACCESS_TOKEN,
                    'messaging_type': 'RESPONSE'
                }
                
                headers = {'Content-Type': 'application/json'}
                
                try:
                    print(f"[OPTION] HTTP 요청 전송 중... (메시지 {message_count})")
                    response = requests.post(url, headers=headers, json=data, timeout=25)
                    
                    print(f"[OPTION] 메시지 {message_count} 응답 상태: {response.status_code}")
                    
                    if response.status_code == 200:
                        result = response.json()
                        message_id = result.get("message_id")
                        print(f"[OPTION] 메시지 {message_count} 전송 성공! (ID: {message_id})")
                        successful_messages += 1
                        
                        # 봇이 보낸 메시지 ID 기록
                        if message_id:
                            BOT_MESSAGES.add(message_id)
                    else:
                        print(f"[OPTION] 메시지 {message_count} 전송 실패: {response.status_code}")
                        print(f"[OPTION] 오류 응답: {response.text}")
                        
                except requests.exceptions.Timeout:
                    print(f"[OPTION] 메시지 {message_count} 요청 타임아웃 (25초)")
                except Exception as e:
                    print(f"[OPTION] 메시지 {message_count} 전송 오류: {e}")
                
                # 메시지 간 딜레이 - 항상 적용 (마지막 메시지 제외)
                if i + 3 < len(options):  # 마지막 메시지가 아니면
                    print(f"[OPTION] 메시지 {message_count} 후 딜레이 시작 (3초)...")
                    time_module.sleep(3.0)
                    print(f"[OPTION] 메시지 {message_count} 딜레이 완료")
            else:
                print(f"[OPTION] 메시지 {message_count} 건너뜀 - 버튼 없음")
            
            print(f"[OPTION] ===== 메시지 {message_count} 완료 =====\n")
        
        print(f"[OPTION] Button Template 전송 완료 - 성공: {successful_messages}/{total_messages}개 메시지")
        
        if successful_messages == 0:
            print(f"[OPTION] 모든 메시지 전송 실패 - 수량 선택으로 이동")
            send_facebook_message(sender_id, "⚠️ 옵션 처리 중 오류가 발생했습니다. 수량을 입력해주세요.")
            time_module.sleep(1)
            send_quantity_selection(sender_id, product_code)
        elif successful_messages < total_messages:
            print(f"[OPTION] 일부 메시지만 전송됨 ({successful_messages}/{total_messages})")
            # 그래도 일부는 전송되었으므로 계속 진행
        else:
            print(f"[OPTION] 모든 옵션 메시지 전송 완료!")
            
    except Exception as e:
        print(f"[OPTION] 옵션 파싱 오류: {e}")
        send_quantity_selection(sender_id, product_code)
        return

def handle_option_selection_from_payload(sender_id: str, payload: str):
    """옵션 선택 처리 (Postback과 Quick Reply 공통)"""
    try:
        print(f"[OPTION_PARSE] payload 파싱 시작: {payload}")
        
        # OPTION_ 제거 후 분리
        parts = payload.replace("OPTION_", "").split("_")
        print(f"[OPTION_PARSE] 분리된 parts: {parts}")
        
        if len(parts) >= 3:
            product_code = parts[0]
            # 마지막 부분이 숫자(가격)이므로, 나머지를 옵션명으로 합침
            try:
                extra_price = int(parts[-1]) if parts[-1].isdigit() or (parts[-1].startswith('-') and parts[-1][1:].isdigit()) else 0
                option_name = "_".join(parts[1:-1])  # 중간 부분들을 다시 합침
            except (ValueError, IndexError):
                extra_price = 0
                option_name = "_".join(parts[1:])  # 전체를 옵션명으로 사용
            
            print(f"⚙️ [OPTION_SELECT] 옵션 선택됨")
            print(f"   상품코드: {product_code}")
            print(f"   옵션명: {option_name}")
            print(f"   추가금액: {extra_price}원")
            
            # 상품 정보 확인 및 캐시 보존
            product = PRODUCT_CACHE.get(product_code)
            if not product:
                print(f"[OPTION_SELECT] 상품 캐시에서 찾을 수 없음: {product_code}")
                print(f"[CACHE_STATUS] 현재 캐시 상품 수: {len(PRODUCT_CACHE)}")
                print(f"[CACHE_KEYS] 캐시 키 목록: {list(PRODUCT_CACHE.keys())}")
                
                # 사용자 데이터에서 상품 정보 복구 시도
                user_data = UserDataManager.get_user_data(sender_id)
                stored_product = user_data.get("product_info", {})
                
                if stored_product and stored_product.get("상품코드") == product_code:
                    print(f"🔄 [CACHE_RESTORE] 사용자 데이터에서 상품 정보 복구")
                    PRODUCT_CACHE[product_code] = stored_product
                    product = stored_product
                else:
                    print(f"[CACHE_RESTORE] 사용자 데이터에서도 상품 정보 없음")
                    send_facebook_message(sender_id, "❌ Product information not found.\n Please try again.")
                    return
            
            # 사용자 데이터에 옵션 정보 저장 (상품 정보도 함께 보존)
            UserDataManager.update_user_data(
                sender_id,
                product_code=product_code,
                product_name=product.get('제목', '상품'),
                selected_option=f"{option_name} (+{extra_price:,}원)" if extra_price > 0 else option_name,
                extra_price=extra_price,
                unit_price=int(float(product.get("가격", 0) or 0)),
                shipping_fee=int(float(product.get("배송비", 0) or 0)),
                max_quantity=int(float(product.get("최대구매수량", 0) or 0)),
                product_info=product,  # 상품 정보 보존
                order_status="option_selected"
            )
            
            # 옵션 선택 완료 메시지
            option_display = f"{option_name}"
            if extra_price > 0:
                option_display += f" (+{extra_price:,}원)"
            
            send_facebook_message(sender_id, f"Selected Options: {option_display}")
            
            # 다음 단계: 수량 선택으로 이동
            import time
            time.sleep(1.5)
            print(f"[NEXT_STEP] 수량 선택 단계로 이동")
            send_quantity_selection(sender_id, product_code)
            
        else:
            print(f"[OPTION_SELECT] 잘못된 payload 형식: {payload}")
            print(f"[OPTION_SELECT] parts 길이: {len(parts)}, parts: {parts}")
            send_facebook_message(sender_id, "❌ An error occurred while selecting your options.\n Please try again.")
            
    except Exception as e:
        print(f"[OPTION_SELECT] 옵션 선택 처리 오류: {e}")
        import traceback
        print(f"[OPTION_SELECT] 상세 오류: {traceback.format_exc()}")
        send_facebook_message(sender_id, "❌ An error occurred while selecting your options.\n Please try again.")

def handle_postback(sender_id: str, payload: str):
    """Postback 처리 함수"""
    import time as time_module
    
    print(f"🔘 [POSTBACK] 처리 시작: {payload}")
    
    try:
        # ===== WELCOME_MESSAGE 처리 (GET_STARTED 대신) =====
        if payload == 'WELCOME_MESSAGE':
            print(f"🎯 [WELCOME_MESSAGE] 시작하기 버튼 클릭 - 웰컴 메시지 전송")
            clear_user_data(sender_id, "get_started")
            send_welcome_message(sender_id)
            return True
        
        # ===== 기본 메뉴 처리 =====
        if payload == 'REGISTER':
            clear_user_data(sender_id, "register")
            send_facebook_message(sender_id,
                "👤 Register as a member to enjoy:\n"
                "• Exclusive discounts 💰\n"
                "• Fast checkout 🚀\n"
                "• Order tracking 📦\n"
                "• Special offers 🎁\n\n"
                "Visit: https://www.chatmall.kr/bbs/register.php")
            time_module.sleep(1)
            send_go_home_card(sender_id)
            return True
        
        elif payload == 'TRACK_ORDER':
            clear_user_data(sender_id, "track_order")
            send_facebook_message(sender_id,
                "📦 Track your order:\n"
                "Please provide your order number or visit:\n"
                "https://www.chatmall.kr/shop/mypage.php\n\n"
                "Need help? Just ask! 😊")
            time_module.sleep(1)
            send_go_home_card(sender_id)
            return True
        
        elif payload == 'AI_SEARCH':
            print(f"🤖 [AI_SEARCH] AI 검색 시작 - 데이터 초기화")
            clear_user_data(sender_id, "ai_search")
            send_ai_search_prompt(sender_id)
            return True
        
        elif payload == 'RESET_CONVERSATION':
            print(f"🔄 [RESET] 명시적 초기화 요청")
            clear_user_data(sender_id, "reset")
            
            search_prompt_text = (
                "🔄 Chat history cleared! ✨\n\n"
                "💬 Now enter what you're looking for:\n\n"
                "For example: portable fan, striped tee, women's light shoes, 100 paper cups\n\n"
                "What are you shopping for today? 😊"
            )
            send_facebook_message(sender_id, search_prompt_text)
            time_module.sleep(1)
            send_navigation_buttons(sender_id)
            return True
            
        elif payload == 'GO_HOME':
            print(f"🏠 [GO_HOME] 홈으로 이동 - 데이터 초기화")
            clear_user_data(sender_id, "go_home")
            send_welcome_message(sender_id)
            return True
        
        # ===== 구매하기 버튼 처리 =====
        elif payload.startswith("BUY_"):
            product_code = payload.replace("BUY_", "")
            print(f"🛒 [BUY] 새로운 주문 시작 - product_code: {product_code}")
            
            # 기존 데이터 초기화
            clear_user_data(sender_id, "new_order")
            
            product = PRODUCT_CACHE.get(product_code)
            if product:
                # 상품 정보를 사용자 데이터에 저장 (캐시 보존)
                UserDataManager.update_user_data(
                    sender_id,
                    product_code=product_code,
                    product_name=product.get('제목', '상품'),
                    unit_price=int(float(product.get("가격", 0) or 0)),
                    shipping_fee=int(float(product.get("배송비", 0) or 0)),
                    max_quantity=int(float(product.get("최대구매수량", 0) or 0)),
                    product_info=product,  # 상품 정보 보존
                    order_status="selecting"
                )
                
                print(f"[BUY] 상품 정보 저장 완료: {product_code}")
                print(f"[CACHE_STATUS] 상품 캐시 저장 확인 - 키: {product_code}")
                
                # 구매 확인 메시지 전송
                purchase_message = (
                    f"🛒 You selected:\n\n"
                    f"📦 {product.get('제목', '상품')}\n"
                    f"💰 Price: {int(float(product.get('가격', 0) or 0)):,}원\n"
                    f"🚚 Shipping: {int(float(product.get('배송비', 0) or 0)):,}원\n\n"
                    f"Let's proceed with your order! 😊"
                )
                send_facebook_message(sender_id, purchase_message)
                time_module.sleep(1)
                
                # 옵션 선택으로 이동
                send_option_selection_buttons(sender_id, product_code)
            else:
                print(f"[BUY] 상품 정보 없음: {product_code}")
                print(f"[CACHE_STATUS] 현재 캐시 상품 수: {len(PRODUCT_CACHE)}")
                send_facebook_message(sender_id, "❌ Product information not found.\n Please search again.")
            return True
        
        # ===== 옵션 선택 버튼 처리 =====
        elif payload.startswith("OPTION_"):
            print(f"[OPTION] 옵션 선택 처리 시작: {payload}")
            handle_option_selection_from_payload(sender_id, payload)
            return True
        
        # ===== 주문 확인 버튼 처리 =====
        elif payload.startswith("CONFIRM_"):
            parts = payload.replace("CONFIRM_", "").split("_")
            if len(parts) >= 2:
                product_code = parts[0]
                try:
                    quantity = int(parts[1])
                    print(f"[CONFIRM] 주문 확인 - product_code: {product_code}, quantity: {quantity}")
                    
                    # 주문 확인 메시지
                    send_facebook_message(sender_id, "Order confirmed! Let's collect your delivery information.")
                    time_module.sleep(1)
                    
                    # 주문 정보 수집 시작
                    send_final_order_complete(sender_id, product_code, quantity)
                except ValueError:
                    print(f"[CONFIRM] 잘못된 수량 값: {parts[1]}")
                    send_facebook_message(sender_id, "❌ An error occurred while processing your order.")
            else:
                print(f"[CONFIRM] 잘못된 payload 형식: {payload}")
                send_facebook_message(sender_id, "❌ An error occurred while processing your order.")
            return True
        
        # ===== 주문 취소 =====
        elif payload.startswith("CANCEL_"):
            product_code = payload.replace("CANCEL_", "")
            print(f"[CANCEL] 주문 취소 - product_code: {product_code}")
            
            # 데이터 즉시 삭제
            clear_user_data(sender_id, "order_cancel")
            
            send_facebook_message(sender_id, 
                "❌ Order cancelled successfully!\n"
                "🔄 Feel free to browse other products or try again. 😊")
            
            time_module.sleep(1)
            send_go_home_card(sender_id)
            return True
        
        # ===== 주문 정보 확인 처리 =====
        elif payload == 'ORDER_CORRECT':
            print(f"[ORDER] 주문 정보 확인 완료")
            send_facebook_message(sender_id, "✅ Perfect! Your order information is confirmed.")
            time_module.sleep(1)
            send_payment_instructions(sender_id)
            return True
        
        elif payload == 'ORDER_INCORRECT':
            print(f"[ORDER] 주문 정보 수정 요청")
            send_facebook_message(sender_id, "No problem! Let's fix that information.")
            time_module.sleep(1)
            send_correction_options(sender_id)
            return True
        
        # ===== 주문 정보 수정 처리 =====
        elif payload == 'CORRECT_NAME':
            print(f"[CORRECT] 이름 수정 요청")
            OrderDataManager.update_order_data(sender_id, order_status="correcting_name")
            send_facebook_message(sender_id, "Please enter the correct name:")
            return True
        
        elif payload == 'CORRECT_ADDRESS':
            print(f"[CORRECT] 주소 수정 요청")
            OrderDataManager.update_order_data(sender_id, order_status="correcting_address")
            send_facebook_message(sender_id, "Please enter the correct address:")
            return True
        
        elif payload == 'CORRECT_PHONE':
            print(f"[CORRECT] 전화번호 수정 요청")
            OrderDataManager.update_order_data(sender_id, order_status="correcting_phone")
            send_facebook_message(sender_id, "Please enter the correct phone number:")
            return True
        
        elif payload == 'CORRECT_EMAIL':
            print(f"[CORRECT] 이메일 수정 요청")
            OrderDataManager.update_order_data(sender_id, order_status="correcting_email")
            send_facebook_message(sender_id, "Please enter the correct email address:")
            return True
        
        elif payload == 'CORRECT_ALL':
            print(f"[CORRECT] 전체 정보 다시 입력")
            OrderDataManager.clear_order_data(sender_id)
            clear_user_data(sender_id, "restart_order")
            send_facebook_message(sender_id, "Let's start over with the order information.")
            time_module.sleep(1)
            ask_receiver_name(sender_id)
            return True
        
        # ===== 결제 처리 =====
        elif payload == 'PAYMENT_SENT':
            print(f"[PAYMENT] 결제 완료 확인")
            send_facebook_message(sender_id, "🎉 Payment confirmation received!")
            time_module.sleep(1)
            send_payment_confirmation(sender_id)
            return True
        
        # ===== 기타 알 수 없는 payload 처리 =====
        else:
            print(f"[POSTBACK] 알 수 없는 payload: {payload}")
            
            # 일반적인 응답 처리
            if payload.upper() in ['HELP', 'SUPPORT']:
                send_facebook_message(sender_id, 
                    "🤝 Need help? Here's what I can do:\n"
                    "• Search for products 🔍\n"
                    "• Help with orders 📦\n"
                    "• Track deliveries 🚚\n"
                    "• Answer questions 💬\n\n"
                    "Just ask me anything! 😊")
                time_module.sleep(1)
                send_go_home_card(sender_id)
            else:
                send_facebook_message(sender_id, 
                    "🤔 I'm not sure about that request.\n"
                    "Let me help you with something else! 😊")
                time_module.sleep(1)
                send_go_home_card(sender_id)
            return True
    
    except Exception as e:
        print(f"[POSTBACK] 처리 오류: {e}")
        import traceback
        print(f"[POSTBACK] 상세 오류:\n{traceback.format_exc()}")
        
        # 오류 발생 시 안전한 응답
        try:
            send_facebook_message(sender_id, 
                "😅 Something went wrong, but I'm here to help!\n"
                "Let's try again. 🔄")
            time_module.sleep(1)
            send_go_home_card(sender_id)
        except Exception as fallback_error:
            print(f"[POSTBACK] 폴백 메시지 전송 실패: {fallback_error}")
        
        return True
    
    finally:
        print(f"🔄 [POSTBACK] 처리 완료: {payload}")
    
    return True

def minimal_clean_with_llm(latest_input: str, previous_inputs: List[str]) -> str:
    """
    최신 입력과 Redis에서 가져온 과거 입력을 함께 LLM에게 전달하여,
    최소한의 정제 + 충돌 문맥 제거를 수행한 한 문장 반환
    """
    try:
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        API_KEY = os.environ["OPENAI_API_KEY"]

        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", openai_api_key=API_KEY)

        context_message = "\n".join(previous_inputs)

        system_prompt = f"""
            당신은 사용자의 과거 대화 기록과 최신 입력을 분석하여 문장에 맞게 의미 있는 검색 쿼리 문장을 재구성하는 전문가입니다.\n

            System:
        당신은 (1) 검색 엔진의 전처리를 담당하는 AI이자, (2) 쇼핑몰 검색 및 분류 전문가입니다.
        어떤 언어로 입력이 되든 반드시 한국어로 문장 의미에 맞게 번역 먼저 합니다.
        아래는 엑셀에서 로드된 **가능한 카테고리 목록**입니다.  
        모든 예측은 이 목록 안에서만 이루어져야 합니다:
        
        {categories}
        
        다음 순서대로 응답하세요:
        
        1) **전처리 단계**  
           - 사용자 원문(query)에서 오타를 바로잡고, 중복 표현을 제거한 뒤  
           - 핵심 키워드와 의미만 남긴 깔끔한 검색 쿼리로 바꿔주세요.  
           - 문장의 의미가 맞다면 문장 통으로 입력되어도 괜찮습니다.  
        
        2) **카테고리 예측 단계**  
           - 전처리된 쿼리를 바탕으로 직관적으로 최상위 카테고리 하나를 예측하세요.
        
        3) **검색 결과 재정렬 단계**  
           - 이미 Milvus 벡터 검색을 통해 얻은 TOP N 결과 리스트(search_results)를 입력받아  
           - 각 결과의 메타데이터(id, 상품명, 카테고리, 가격, URL 등)를 활용해  
           - 2번에서 예측한 카테고리와 매칭되거나 인접한 결과를 우선 정렬하세요.
        
        4) **출력 형식**은 반드시 아래와 같습니다:
        
        Raw Query: "<query>"  
        Preprocessed Query: "<전처리된_쿼리>"  
        Predicted Category: "<예측된_최상위_카테고리>" 

        
            다음 기준을 철저히 따르세요:\n
            1. 이전 입력 중 **최신 입력과 의미가 충돌하는 문장**은 완전히 제거합니다.\n
            2. **충돌이 없는 이전 입력은 유지**하며, **최신 입력을 반영**해 전체 흐름을 자연스럽게 이어가세요.\n
            3. 문장의 단어 순서나 표현은 원문을 최대한 유지합니다.\n
            4. 오타, 띄어쓰기, 맞춤법만 교정하세요.\n
            5. 절대로 결과에 설명을 추가하지 마세요. **한 문장만 출력**합니다.\n
            \n
            ---\n
            \n
            # 예시 1:\n
            이전 입력:\n
            - 강아지 옷 찿아줘\n
            - 밝은색 으로다시찾아\n
            - 겨울 용이면 더조아\n
            \n
            최신 입력:\n
            - 여름용으로 바꿔줘\n
            \n
            → 결과: "강아지 옷 여름용 밝은 색"\n
            \n
            ---\n
            \n
            # 예시 2:\n
            이전 입력:\n
            - 아이폰\n
            - 프로 모델 이면 좋겠 어\n
            - 실버 색상으로\n
            \n
            최신 입력:\n
            - 갤럭시로 \n
            \n
            → 결과: "갤럭시 실버 색상"\n
            \n
            ---\n
            \n
            # 예시 3:\n
            이전 입력:\n
            - 운동화250mm사이즈찿아줘\n
            - 흰 색 계열이 좋아\n
            - 쿠션감있는거 위주로\n
            \n
            최신 입력:\n
            - 260mm로 바꿔줘\n
            \n
            → 결과: "운동화 260mm 흰색 쿠션감 있는 걸로 찾아줘"\n
            """
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"이전 대화: {context_message}\n최신 입력: {latest_input}")
        ])

        if not hasattr(response, "content") or not isinstance(response.content, str):
            raise ValueError("LLM 응답이 유효하지 않습니다.")

        return response.content.strip()

    except Exception as e:
        print(f"[ERROR] minimal_clean_with_llm 실패: {e}")
        return latest_input

def start_order_info_collection(sender_id: str, product_code: str, quantity: int):
    """주문 정보 수집 시작"""
    import time as time_module
    
    print(f"[ORDER_INFO] 주문 정보 수집 시작")
    
    # 상품 정보 확인
    product = PRODUCT_CACHE.get(product_code)
    if not product:
        send_facebook_message(sender_id, "Product information not found.")
        return
    
    # 사용자 데이터에서 주문 정보 가져오기
    user_data = UserDataManager.get_user_data(sender_id)
    total_price = user_data.get("total_price", 0)
    selected_option = user_data.get("selected_option", "기본옵션")
    
    # Facebook 사용자 이름 가져오기
    facebook_name = get_user_name(sender_id)
    
    # 주문 데이터 초기화 및 설정
    OrderDataManager.update_order_data(
        sender_id,
        product_name=product.get('제목', '상품'),
        selected_option=selected_option,
        quantity=quantity,
        total_price=total_price,
        facebook_name=facebook_name,
        order_status="collecting_info"
    )
    
    # 배송 정보 수집 시작 메시지
    info_message = "To deliver your items safely and quickly, please provide the required information."
    send_facebook_message(sender_id, info_message)
    
    time_module.sleep(1)
    
    # 첫 번째 질문: 수령인 이름
    ask_receiver_name(sender_id)

def ask_receiver_name(sender_id: str):
    """수령인 이름 질문"""
    OrderDataManager.update_order_data(sender_id, order_status="waiting_name")
    send_facebook_message(sender_id, "What is your name or the recipient's full name?")

def ask_address(sender_id: str):
    """주소 질문"""
    OrderDataManager.update_order_data(sender_id, order_status="waiting_address")
    send_facebook_message(sender_id, "Thank you! What is the shipping address?")

def ask_phone_number(sender_id: str):
    """전화번호 질문"""
    OrderDataManager.update_order_data(sender_id, order_status="waiting_phone")
    send_facebook_message(sender_id, "Almost done! May I have your phone number?")

def ask_email(sender_id: str):
    """이메일 질문"""
    OrderDataManager.update_order_data(sender_id, order_status="waiting_email")
    send_facebook_message(sender_id, "Last step! What is your email address?")

def send_order_confirmation_review(sender_id: str):
    """주문 정보 확인 카드 전송"""
    import time as time_module
    
    order_data = OrderDataManager.get_order_data(sender_id)
    
    confirmation_text = f"""Kindly review and confirm the info below is correct:

🙋‍♂️Name: {order_data.get('receiver_name', '')}
🏠Address: {order_data.get('address', '')}
📞Contact #: {order_data.get('phone_number', '')}
📧Email: {order_data.get('email', '')}

📦Product: {order_data.get('product_name', '')}
☑️Option: {order_data.get('selected_option', '')}

🔢quantity: {order_data.get('quantity', '')}
💰Total_money: {order_data.get('total_price', 0):,}원"""
    
    send_facebook_message(sender_id, confirmation_text)
    
    time_module.sleep(1)
    
    # 확인/수정 버튼 카드 전송
    url = f"https://graph.facebook.com/v18.0/me/messages"

    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Order Information Review',
                            'subtitle': 'Please confirm if all information is correct',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '✅ Correct',
                                    'payload': 'ORDER_CORRECT'
                                },
                                {
                                    'type': 'postback',
                                    'title': '❌ Incorrect',
                                    'payload': 'ORDER_INCORRECT'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ 주문 확인 카드 전송 성공")
            OrderDataManager.update_order_data(sender_id, order_status="review_confirmation")
        else:
            print(f"❌ 주문 확인 카드 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 주문 확인 카드 전송 오류: {e}")

    send_go_home_card(sender_id)

def send_correction_options(sender_id: str):
    """수정 옵션 버튼 전송"""
    import time as time_module
    
    send_facebook_message(sender_id, "Which is incorrect?")
    
    time_module.sleep(1)
    
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Select Information to Correct',
                            'subtitle': 'Choose what you want to update',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': 'Name',
                                    'payload': 'CORRECT_NAME'
                                },
                                {
                                    'type': 'postback',
                                    'title': 'Address',
                                    'payload': 'CORRECT_ADDRESS'
                                },
                                {
                                    'type': 'postback',
                                    'title': 'Contact Number',
                                    'payload': 'CORRECT_PHONE'
                                }
                            ]
                        },
                        {
                            'title': 'More Options',
                            'subtitle': 'Additional correction options',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': 'Email',
                                    'payload': 'CORRECT_EMAIL'
                                },
                                {
                                    'type': 'postback',
                                    'title': 'ALL',
                                    'payload': 'CORRECT_ALL'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ 수정 옵션 카드 전송 성공")
        else:
            print(f"❌ 수정 옵션 카드 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 수정 옵션 카드 전송 오류: {e}")

def send_payment_instructions(sender_id: str):
    """결제 안내 메시지 전송 - 분리 방식"""
    import time as time_module
    
    payment_message = """For secure payment processing, please make a deposit to the account below.

Bank Name: 하나은행 / Hana Bank
Account Number: 841-910015-85404
Account Name: (주)나로수

After sending the payment, please click the "PAYMENT SENT" button so we can process your order faster!"""
    
    # ✅ 1단계: 결제 안내 메시지
    send_facebook_message(sender_id, payment_message)
    time_module.sleep(1)
    
    # ✅ 3단계: 버튼만 있는 깔끔한 카드
    url = f"https://graph.facebook.com/v18.0/me/messages"
    
    data = {
        'recipient': {'id': sender_id},
        'message': {
            'attachment': {
                'type': 'template',
                'payload': {
                    'template_type': 'generic',
                    'elements': [
                        {
                            'title': 'Payment Status',
                            'subtitle': 'Click after completing your payment',
                            'image_url': '',
                            'buttons': [
                                {
                                    'type': 'postback',
                                    'title': '✅ PAYMENT SENT',
                                    'payload': 'PAYMENT_SENT'
                                }
                            ]
                        }
                    ]
                }
            }
        },
        'access_token': PAGE_ACCESS_TOKEN,
        'messaging_type': 'RESPONSE'
    }
    
    headers = {'Content-Type': 'application/json'}
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            print(f"✅ 결제 확인 버튼 전송 성공")
            OrderDataManager.update_order_data(sender_id, order_status="waiting_payment")
        else:
            print(f"❌ 결제 확인 버튼 전송 실패: {response.status_code}")
    except Exception as e:
        print(f"❌ 결제 확인 버튼 전송 오류: {e}")

def handle_quick_reply(sender_id: str, payload: str):
    """Quick Reply 처리 함수 - 통합된 버전"""
    print(f"🔘 [QUICK_REPLY] Quick Reply 처리: {payload}")
    
    if payload.startswith('OPTION_'):
        print(f"⚙️ [QUICK_REPLY] 옵션 선택 처리 시작: {payload}")
        handle_option_selection_from_payload(sender_id, payload)
    elif payload == 'AI_SEARCH':
        print(f"🤖 [QUICK_REPLY] AI 검색 시작")
        clear_user_data(sender_id, "ai_search")
        send_ai_search_prompt(sender_id)
    else:
        print(f"⚠️ [QUICK_REPLY] 알 수 없는 payload: {payload}")
        handle_postback(sender_id, payload)

def send_payment_confirmation(sender_id: str):
    """결제 확인 메시지 전송"""
    import time as time_module
    from datetime import datetime

    send_facebook_message(sender_id, "💳 Payment Confirmation")
    time_module.sleep(1)
    
    # 첫 번째 확인 메시지
    confirmation_message = """Once we confirm your payment, we'll process your order right away! 🚚💨

Please give us a moment while our ChatMall team confirms your payment. ⏳💳"""
    
    send_facebook_message(sender_id, confirmation_message)
    time_module.sleep(2)
    
    # ✅ Google Sheets에 주문 정보 전송
    print(f"📊 [ORDER_COMPLETE] Google Sheets 전송 시작")
    sheets_success = send_order_to_sheets(sender_id)
    
    if sheets_success:
        print(f"✅ [ORDER_COMPLETE] Google Sheets 전송 성공 ✨")
    else:
        print(f"❌ [ORDER_COMPLETE] Google Sheets 전송 실패")
    
    # 주문 상세 정보 생성
    order_data = OrderDataManager.get_order_data(sender_id)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    order_summary = f"""✅ Order Completed Successfully! 🎉

📅 Order Time: {current_time}
👤 Customer: {order_data.get('facebook_name', '')}
📋 Receiver: {order_data.get('receiver_name', '')}
🏠 Address: {order_data.get('address', '')}
📞 Contact: {order_data.get('phone_number', '')}
📧 Email: {order_data.get('email', '')}

🛍️ Order Details:
📦 Product: {order_data.get('product_name', '')}
⚙️ Option: {order_data.get('selected_option', '')}
🔢 Quantity: {order_data.get('quantity', 0)}
💰 Total: {order_data.get('total_price', 0):,}원

🚚 We'll start processing your order right away!
Thank you for shopping with ChatMall! 😊"""
    
    send_facebook_message(sender_id, order_summary)
    
    # 주문 완료 후 데이터 정리
    time_module.sleep(1)
    print(f"🧹 [ORDER_COMPLETE] 주문 완료 후 데이터 정리 시작")
    
    # 주문 데이터는 보존하고 임시 데이터만 정리
    UserDataManager.clear_user_data(sender_id)  # 임시 데이터만 삭제
    # OrderDataManager.clear_order_data(sender_id)  # 주문 데이터는 보존
    
    print(f"✅ [ORDER_COMPLETE] 주문 완료 처리 끝")
    
    # 홈으로 돌아가기 버튼 제공
    time_module.sleep(2)
    send_go_home_card(sender_id)

def handle_order_info_input(sender_id: str, user_message: str) -> bool:
    """주문 정보 입력 처리"""
    try:
        order_data = OrderDataManager.get_order_data(sender_id)
        order_status = order_data.get("order_status")
        
        if order_status == "waiting_name":
            OrderDataManager.update_order_data(sender_id, receiver_name=user_message.strip())
            ask_address(sender_id)
            return True
        
        elif order_status == "waiting_address":
            OrderDataManager.update_order_data(sender_id, address=user_message.strip())
            ask_phone_number(sender_id)
            return True
        
        elif order_status == "waiting_phone":
            OrderDataManager.update_order_data(sender_id, phone_number=user_message.strip())
            ask_email(sender_id)
            return True
        
        elif order_status == "waiting_email":
            OrderDataManager.update_order_data(sender_id, email=user_message.strip())
            send_order_confirmation_review(sender_id)
            return True
        
        # 수정 모드 처리
        elif order_status == "correcting_name":
            OrderDataManager.update_order_data(sender_id, receiver_name=user_message.strip())
            send_order_confirmation_review(sender_id)
            return True
        
        elif order_status == "correcting_address":
            OrderDataManager.update_order_data(sender_id, address=user_message.strip())
            send_order_confirmation_review(sender_id)
            return True
        
        elif order_status == "correcting_phone":
            OrderDataManager.update_order_data(sender_id, phone_number=user_message.strip())
            send_order_confirmation_review(sender_id)
            return True
        
        elif order_status == "correcting_email":
            OrderDataManager.update_order_data(sender_id, email=user_message.strip())
            send_order_confirmation_review(sender_id)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ [ORDER_INFO] 주문 정보 입력 처리 오류: {e}")
        return False

async def process_ai_response(sender_id: str, user_message: str, processing_key: str = None):
    """AI 응답 처리 (타임아웃 추가)"""
    try:
        print(f"🕒 [AI 처리 시작] 유저 ID: {sender_id}, 메시지: {user_message}")

        # 타임아웃 설정 (120초)
        try:
            loop = asyncio.get_running_loop()
            bot_response = await asyncio.wait_for(
                loop.run_in_executor(executor, external_search_and_generate_response, user_message, sender_id),
                timeout=120.0
            )
        except asyncio.TimeoutError:
            print(f"⏱️ [AI_TIMEOUT] AI 처리 타임아웃 (120초)")
            send_facebook_message(sender_id, "The request timed out. Please try again.")
            return

        if isinstance(bot_response, dict):
            combined_message_text = bot_response.get("combined_message_text", "")
            results = bot_response.get("results", [])

            # 상품 캐시에 저장
            for product in results:
                product_code = product.get("상품코드")
                if product_code:
                    PRODUCT_CACHE[product_code] = product

            # 응답이 없으면 기본 메시지
            if not combined_message_text and not results:
                send_facebook_message(sender_id, "Sorry, we couldn't find any results.\n Please try again with different keywords.")
                return

            # AI 응답 메시지 먼저 전송
            if combined_message_text:
                send_facebook_message(sender_id, combined_message_text)
                await asyncio.sleep(1)

            # 상품 카루셀 전송
            if results:
                send_facebook_carousel(sender_id, results)

            print(f"✅ [메시지 전송 완료] - 키: {processing_key}")
            
            # 네비게이션 버튼 전송
            try:
                await asyncio.sleep(1)
                send_navigation_buttons(sender_id)
                print(f"✅ [NAVIGATION_BUTTONS] 네비게이션 버튼 전송 완료")
            except Exception as nav_error:
                print(f"❌ [NAVIGATION_BUTTONS] 네비게이션 버튼 전송 실패: {nav_error}")
        else:
            print(f"❌ AI 응답 오류 발생")
            send_facebook_message(sender_id, "Sorry, an error occurred while processing.")

    except Exception as e:
        print(f"❌ AI 응답 처리 오류: {e}")
        send_facebook_message(sender_id, "Sorry, an error occurred while processing.")

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Facebook 웹훅 인증"""
    try:
        mode = request.query_params.get("hub.mode")
        token = request.query_params.get("hub.verify_token")
        challenge = request.query_params.get("hub.challenge")
        
        print(f"🔍 받은 Verify Token: {token}")
        print(f"🔍 서버 Verify Token: {VERIFY_TOKEN}")
        
        if mode == "subscribe" and token == VERIFY_TOKEN:
            print("✅ 웹훅 인증 성공")
            return int(challenge)
        else:
            print("❌ 웹훅 인증 실패")
            return {"status": "error", "message": "Invalid token"}
    except Exception as e:
        print(f"❌ 인증 처리 오류: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/webhook")
async def handle_webhook(request: Request, background_tasks: BackgroundTasks):
    """Facebook 웹훅 메시지 처리"""
    try:
        data = await request.json()
        
        # 불필요한 이벤트 필터링
        should_log = False
        for entry in data.get("entry", []):
            for messaging in entry.get("messaging", []):
                # 실제 메시지나 postback만 출력
                if "message" in messaging and messaging["message"].get("text"):
                    should_log = True
                elif "postback" in messaging:
                    should_log = True
        
        # 의미있는 이벤트만 출력
        if should_log:
            print(f"📥 받은 메시지: {json.dumps(data, indent=2, ensure_ascii=False)}")
        
        # 즉시 성공 응답 반환 (5초 제한 준수)
        if data.get("object") == "page":
            # 백그라운드에서 메시지 처리
            background_tasks.add_task(process_webhook_data, data)
            return {"status": "success"}  # 즉시 응답
        
        return {"status": "success"}

    except Exception as e:
        print(f"❌ 웹훅 처리 오류: {e}")
        return {"status": "success"}  # 오류 발생해도 성공 응답 (재시도 방지)

async def process_webhook_data(data: dict):
    """웹훅 데이터 백그라운드 처리"""
    try:
        for entry in data.get("entry", []):
            for messaging in entry.get("messaging", []):
                sender_id = messaging.get("sender", {}).get("id")
                
                # 메시지 중복 처리 강화
                message_key = generate_message_key(messaging)
                if message_key in PROCESSED_MESSAGES:
                    print(f"🔄 [DUPLICATE] 중복 메시지 스킵: {message_key}")
                    continue
                
                PROCESSED_MESSAGES.add(message_key)
                
                # 처리 중인 사용자 체크 (동시 처리 방지)
                if sender_id in PROCESSING_USERS:
                    print(f"⏳ [PROCESSING] 사용자 {sender_id} 이미 처리 중")
                    continue
                
                PROCESSING_USERS.add(sender_id)
                
                try:
                    # Postback 처리
                    if "postback" in messaging:
                        postback = messaging["postback"]
                        payload = postback.get("payload", "")
                        print(f"🔘 [POSTBACK] 버튼 클릭: {payload}")
                        handle_postback(sender_id, payload)
                    
                    # 메시지 처리
                    elif "message" in messaging:
                        message = messaging["message"]
                        user_message = message.get("text", "").strip()
                        message_id = message.get("mid")
                        
                        # Echo 및 봇 메시지 필터링 강화
                        if (message.get("is_echo") or 
                            message_id in BOT_MESSAGES or 
                            not user_message):
                            continue
                        ConversationLogger.log_user_message(sender_id, user_message)
                        
                        # 퀵 리플라이 처리
                        quick_reply = message.get("quick_reply")
                        if quick_reply:
                            payload = quick_reply.get("payload")
                            print(f"🔘 [QUICK_REPLY] 버튼 클릭: {payload}")
                            handle_quick_reply(sender_id, payload)
                            continue
                        
                        # 일반 메시지 처리
                        if user_message:
                            await handle_user_message(sender_id, user_message)
                
                finally:
                    # 처리 완료 후 사용자 제거
                    PROCESSING_USERS.discard(sender_id)
                    
    except Exception as e:
        print(f"❌ 백그라운드 처리 오류: {e}")

async def handle_user_message(sender_id: str, user_message: str):
    """사용자 메시지 처리"""
    try:
        print(f"💬 [USER_MESSAGE] 처리 시작: {user_message}")
        
        # 인사말 처리
        if is_greeting_message(user_message):
            clear_user_data(sender_id, "greeting")
            send_welcome_message(sender_id)
            return
        
        # AI 검색 트리거 처리
        if is_ai_search_trigger(user_message):
            clear_user_data(sender_id, "ai_search")
            send_ai_search_prompt(sender_id)
            return
        
        # 수량 입력 처리
        if handle_quantity_input(sender_id, user_message):
            return
        
        # 주문 정보 입력 처리
        if handle_order_info_input(sender_id, user_message):
            return
        
        # AI 응답 처리
        await process_ai_response(sender_id, user_message)
        
    except Exception as e:
        print(f"❌ [USER_MESSAGE] 처리 오류: {e}")
        send_facebook_message(sender_id, "Sorry, an error occurred while processing.")

def cleanup_message_cache():
    """메시지 캐시 정리 (메모리 누수 방지)"""
    global PROCESSED_MESSAGES, BOT_MESSAGES
    
    # 5분 이상 된 메시지 캐시 정리
    current_time = time.time()
    if not hasattr(cleanup_message_cache, "last_cleanup"):
        cleanup_message_cache.last_cleanup = current_time
    
    if current_time - cleanup_message_cache.last_cleanup > 300:  # 5분
        PROCESSED_MESSAGES.clear()
        BOT_MESSAGES.clear()
        cleanup_message_cache.last_cleanup = current_time
        print("🧹 메시지 캐시 정리 완료")

def generate_message_key(messaging: dict) -> str:
    """메시지 고유 키 생성"""
    if "message" in messaging:
        message_id = messaging["message"].get("mid")
        sender_id = messaging.get("sender", {}).get("id")
        timestamp = messaging.get("timestamp", 0)
        return f"{sender_id}_{message_id}_{timestamp}"
    elif "postback" in messaging:
        sender_id = messaging.get("sender", {}).get("id")
        payload = messaging["postback"].get("payload", "")
        timestamp = messaging.get("timestamp", 0)
        return f"{sender_id}_{payload}_{timestamp}"
    else:
        return f"unknown_{time.time()}"

def external_search_and_generate_response(request: Union[QueryRequest, str], session_id: str = None) -> dict:
    try:
        # 입력 쿼리 추출 및 타입 확인
        query = request if isinstance(request, str) else request.query
        print(f"🔍 사용자 검색어: {query}")
        
        if not isinstance(query, str):
            raise TypeError(f"❌ [ERROR] 잘못된 query 타입: {type(query)}")
    
        # 세션 초기화 명령 처리
        if query.lower() == "reset":
            if session_id:
                clear_message_history(session_id)
            return {"message": f"세션 {session_id}의 대화 기록이 초기화되었습니다."}
    
        # Redis 세션 기록 불러오기 및 최신 입력 저장
        session_history = get_session_history(session_id)
        session_history.add_user_message(query)
    
        previous_queries = [msg.content for msg in session_history.messages if isinstance(msg, HumanMessage)]
        if query in previous_queries:
            previous_queries.remove(query)
        
        # 전체 중복 제거
        previous_queries = list(dict.fromkeys(previous_queries))

        # LLM으로 정제된 쿼리 생성
        UserMessage = minimal_clean_with_llm(query, previous_queries)
        print("\n🧾 [최종 정제된 문장] →", UserMessage)
        print("📚 [원본 전체 문맥] →", " | ".join(previous_queries + [query]))
        
        raw = detect(query)
        lang_code = raw.lower().split("-")[0]

        # 언어 코드 매핑
        lang_map = {
            "ko": "한국어",
            "en": "English",
            "zh": "中文",
            "ja": "日本語",
            "vi": "Tiếng Việt",
            "th": "ไทย",
        }
        
        target_lang = lang_map.get(lang_code, "English")
        print("[Debug] Detected language →", target_lang)
        
        llm_response = UserMessage
        print("[Debug] LLM full response:\n", llm_response)
        
        # LLM 응답 파싱
        lines = [l.strip() for l in llm_response.splitlines() if l.strip()]
        try:
            preprocessed_query = next(
                l.split(":",1)[1].strip().strip('"')
                for l in lines if l.lower().startswith("preprocessed query")
            )
            predicted_category = next(
                l.split(":",1)[1].strip().strip('"')
                for l in lines if l.lower().startswith("predicted category")
            )
        except:
            preprocessed_query = UserMessage
            predicted_category = "일반상품"
        
        top_category = predicted_category.split(">")[0]
        
        print("[Debug] Preprocessed Query →", preprocessed_query)
        print("[Debug] top_category →", top_category)
        
        # 쿼리 임베딩 생성
        q_vec = embedder.embed_query(preprocessed_query)
        print(f"[Debug] q_vec length: {len(q_vec)}, sample: {q_vec[:5]}")
        
        # Stage1: 직접 문자열 검색
        print("[Stage1] Direct name search 시작")
        
        tokens = [t for t in re.sub(r"[용\s]+", " ", preprocessed_query).split() if t]
        query_expr = " && ".join(f'market_product_name like "%{tok}%"' for tok in tokens)
        
        print("[Debug] Stage1 expr:", query_expr)
        direct_hits = collection.query(
            expr=query_expr,
            output_fields=[
                "product_code", "category_code", "category_name", "market_product_name",
                "market_price", "shipping_fee", "shipping_type", "max_quantity",
                "composite_options", "image_url", "manufacturer", "model_name",
                "origin", "keywords", "description", "return_shipping_fee",
            ]
        )
        print("[Stage1] Direct hits count:", len(direct_hits))

        # 50개로 제한
        n = 50
        if len(direct_hits) > n:
            direct_hits = random.sample(direct_hits, n)

        raw_candidates = []
        for row in direct_hits:
            try:
                html_raw = row.get("description", "") or ""
                html_cleaned = clean_html_content(html_raw)
                if isinstance(html_raw, bytes):
                    html_raw = html_raw.decode("cp949")
                encoded_html = base64.b64encode(
                    html_cleaned.encode("utf-8", errors="ignore")
                ).decode("utf-8")
                safe_html = urllib.parse.quote_plus(encoded_html)
                preview_url = f"{API_URL}/preview?html={safe_html}"
            except Exception as err:
                print(f"⚠️ 본문 처리 오류: {err}")
                preview_url = "https://naver.com"
    
            product_link = row.get("product_link", "")
            if not product_link or product_link in ["링크 없음", "#", None]:
                product_link = preview_url
    
            option_raw = str(row.get("composite_options", "")).strip()
            option_display = "없음"
            if option_raw.lower() not in ["", "nan"]:
                parsed = []
                for line in option_raw.splitlines():
                    try:
                        name, extra, _ = line.split(",")
                        extra = int(float(extra))
                        parsed.append(
                            f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                        )
                    except Exception:
                        parsed.append(line.strip())
                option_display = "\n".join(parsed)
    
            result_info = {
                "상품코드": str(row.get("product_code", "없음")),
                "제목": row.get("market_product_name", "제목 없음"),
                "가격": convert_to_serializable(row.get("market_price", 0)),
                "배송비": convert_to_serializable(row.get("shipping_fee", 0)),
                "이미지": row.get("image_url", "이미지 없음"),
                "원산지": row.get("origin", "정보 없음"),
                "상품링크": product_link,
                "옵션": option_display,
                "조합형옵션": option_raw,
                "최대구매수량": convert_to_serializable(row.get("max_quantity", 0)),
            }
            result_info_cleaned = {}
            for k, v in result_info.items():
                if isinstance(v, str):
                    v = v.replace("\n", "").replace("\r", "").replace("\t", "")
                result_info_cleaned[k] = v
            raw_candidates.append(result_info_cleaned)

        # Stage2: 벡터 유사도 검색
        milvus_results = collection.search(
            data=[q_vec],
            anns_field="emb",
            param={"metric_type": "L2", "params": {"nprobe": 128}},
            limit=50,
            output_fields=[
                "product_code", "category_code", "category_name", "market_product_name",
                "market_price", "shipping_fee", "shipping_type", "max_quantity",
                "composite_options", "image_url", "manufacturer", "model_name",
                "origin", "keywords", "description", "return_shipping_fee",
            ]
        )
        print(f"[Stage2] Vector hits count: {len(milvus_results[0])}")

        # 벡터 검색 결과 추가
        for hits in milvus_results:
            for hit in hits:
                e = hit.entity
                try:
                    html_raw = e.get("description", "") or ""
                    html_cleaned = clean_html_content(html_raw)
                    if isinstance(html_raw, bytes):
                        html_raw = html_raw.decode("cp949")
                    encoded_html = base64.b64encode(
                        html_cleaned.encode("utf-8", errors="ignore")
                    ).decode("utf-8")
                    safe_html = urllib.parse.quote_plus(encoded_html)
                    preview_url = f"{API_URL}/preview?html={safe_html}"
                except Exception as err:
                    print(f"⚠️ 본문 처리 오류: {err}")
                    preview_url = "https://naver.com"
        
                product_link = e.get("product_link", "")
                if not product_link or product_link in ["링크 없음", "#", None]:
                    product_link = preview_url
        
                option_raw = str(e.get("composite_options", "")).strip()
                option_display = "없음"
                if option_raw.lower() not in ["", "nan"]:
                    parsed = []
                    for line in option_raw.splitlines():
                        try:
                            name, extra, _ = line.split(",")
                            extra = int(float(extra))
                            parsed.append(
                                f"{name.strip()}{f' (＋{extra:,}원)' if extra>0 else ''}"
                            )
                        except Exception:
                            parsed.append(line.strip())
                    option_display = "\n".join(parsed)
        
                result_info = {
                    "상품코드": str(e.get("product_code", "없음")),
                    "제목": e.get("market_product_name", "제목 없음"),
                    "가격": convert_to_serializable(e.get("market_price", 0)),
                    "배송비": convert_to_serializable(e.get("shipping_fee", 0)),
                    "이미지": e.get("image_url", "이미지 없음"),
                    "원산지": e.get("origin", "정보 없음"),
                    "상품링크": product_link,
                    "옵션": option_display,
                    "조합형옵션": option_raw,
                    "최대구매수량": convert_to_serializable(e.get("max_quantity", 0)),
                }
                result_info_cleaned = {}
                for k, v in result_info.items():
                    if isinstance(v, str):
                        v = v.replace("\n", "").replace("\r", "").replace("\t", "")
                    result_info_cleaned[k] = v
                raw_candidates.append(result_info_cleaned)
        
        message_history = [
            {"type": type(msg).__name__, "content": msg.content if hasattr(msg, "content") else str(msg)}
            for msg in session_history.messages
        ]

        # Stage4: LLM으로 최종 5개 선택
        print("[Stage4] LLM 최종 후보 선정 시작")
        candidate_list = "\n".join(
            f"{i+1}. {info['제목']} [{info.get('카테고리', predicted_category)}]"
            for i, info in enumerate(raw_candidates)
        )

        raw_results_json = json.dumps(candidate_list[:5], ensure_ascii=False)
        raw_history_json = json.dumps(message_history, ensure_ascii=False)
        escaped_results = raw_results_json.replace("{", "{{").replace("}", "}}")
        escaped_history = raw_history_json.replace("{", "{{").replace("}", "}}")

        print("[Stage4] LLM에 넘길 후보 리스트:\n", candidate_list[:500], "...")
        print(f"target_lang: {target_lang}")

        # LangChain 기반 프롬프트 및 LLM 실행 설정
        llm = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", openai_api_key=API_KEY)
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""
            **⚠️ 답변은 반드시 "{target_lang}" 언어로 답변 해주세요.**
            System: 당신은 쇼핑몰에 대해서 전문지식을 갖춘 직원 입니다. 최대한 친근하고 정중한 말투로 상품을 물음표로 권유합니다.
            User Query: "{query}"
            예측된 카테고리: "{predicted_category}"
            아래 후보들은 모두 이 카테고리에 속합니다. 
            후보리스트 : {candidate_list}.
            반드시 후보리스트만 보고 사용자에게 후보리스트 내용정보 안에서만 상품을 추천하는 질문을 만들어서 물음표로 권유합니다.
            입력된 모든 상품을 가지고 카테고리, 제목등 찾은 결과의 내용들을 종합해서 넣어서 원하는 상품을 없다는면 원하는 상품을 좁혀나가는 질문을 반드시 400자로 생성 합니다.
            그리고 나서 이 중 사용자 의도에 가장 적합한 5개 항목의 번호만 JSON 배열 형태로 반환하세요:

            candidate_list 상품 번호는 너만 보기만 하고 LLM답변으로 출력은 절대 하지마.
         """),
            MessagesPlaceholder(variable_name="message_history"),
            ("system", f"[검색 결과 - 내부 참고용 JSON]\n{escaped_results}"),
            ("system", f"[이전 대화 내용]\n{escaped_history}"),
            ("human", query)
        ]) 
    
        runnable = prompt | llm
        with_message_history = RunnableWithMessageHistory(
            runnable,
            get_session_history,
            input_messages_key="input",
            history_messages_key="message_history",
        )
        
        # 응답 생성 및 시간 측정
        start_response = time.time()
        print("▶️ [LLM 호출 시작] with_message_history.invoke() 직전")

        resp2 = with_message_history.invoke(
            {
              "input": query,
              "query": query,
              "predicted_category": predicted_category,
              "target_lang": target_lang
            },
            config={"configurable": {"session_id": session_id}}
        )
        
        print(f"📊 [LLM 응답 시간] {time.time() - start_response:.2f}초")
        print("🤖 응답 결과:", resp2.content)

        selection = resp2.content.strip()
        
        # JSON 마크다운 제거
        clean = re.sub(r'```.*?\n', '', selection).replace('```','').strip()
        
        match = re.search(r'\[(?:\s*\d+\s*,?)+\s*\]', clean)
        if match:
            arr_text = match.group(0)
            try:
                chosen_idxs = json.loads(arr_text)
            except json.JSONDecodeError:
                chosen_idxs = []
        else:
            chosen_idxs = []
        
        max_n = len(raw_candidates)
        valid_idxs = [i for i in chosen_idxs if 1 <= i <= max_n]
        if len(valid_idxs) < len(chosen_idxs):
            print(f"⚠️ 잘못된 인덱스 제거됨: {set(chosen_idxs) - set(valid_idxs)}")
        if not valid_idxs:
            print("⚠️ 유효 인덱스 없음, 상위 5개로 Fallback")
            valid_idxs = list(range(1, min(6, max_n+1)))
        chosen_idxs = valid_idxs
        print("[Stage4] Final chosen indices:", chosen_idxs)
        
        # 최종 결과 매핑
        final_results = [raw_candidates[i-1] for i in chosen_idxs]
        print("\n✅ 최종 추천 5개 상품:")
        
        # 10개 이상이면 앞 10개만 사용
        if len(final_results) > 10:
            final_results = final_results[:10]
        
        for idx, info in enumerate(final_results, start=1):
            # ✅ 상품 캐시에 저장 (키 확인)
            product_code = info.get("상품코드")
            if product_code:
                PRODUCT_CACHE[product_code] = info
                print(f"💾 [CACHE_SAVE] 상품 캐시 저장: {product_code} -> {info.get('제목', '제목없음')}")
            else:
                print(f"⚠️ [CACHE_SAVE] 상품코드 없음: {info}")
            
            print(f"\n[{idx}] {info['제목']}")
            print(f"   상품코드   : {info['상품코드']}")
            print(f"   가격       : {info['가격']}원")
            print(f"   배송비     : {info['배송비']}원")
            print(f"   이미지     : {info['이미지']}")
            print(f"   원산지     : {info['원산지']}")
            print(f"   상품링크   : {info['상품링크']}")
        
        # 최종 결과 반환
        result_payload = {
            "query": query,
            "UserMessage": UserMessage,
            "RawContext": previous_queries + [query],
            "results": final_results,
            "combined_message_text": resp2.content,
            "message_history": message_history
        }
        return result_payload
    
    except Exception as e:
        print(f"❌ external_search_and_generate_response 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
# ✨ 새로운 API 엔드포인트: 주문 내역 조회
@app.get("/orders/{sender_id}")
async def get_user_orders(sender_id: str):
    """사용자의 주문 내역 조회"""
    user_data = UserDataManager.get_user_data(sender_id)
    order_history = user_data.get("order_history", [])
    
    return {"orders": order_history}

# ✨ 새로운 API 엔드포인트: 상품 캐시 조회
@app.get("/products")
async def get_cached_products():
    """캐시된 상품 목록 조회"""
    return {"products": list(PRODUCT_CACHE.values())}

# ✨ 새로운 API 엔드포인트: 사용자 데이터 조회
@app.get("/user/{sender_id}")
async def get_user_data_api(sender_id: str):
    """사용자 데이터 상태 조회"""
    user_data = UserDataManager.get_user_data(sender_id)
    return {"user_data": user_data}

# ✨ 새로운 API 엔드포인트: 사용자 데이터 초기화
@app.post("/user/{sender_id}/clear")
async def clear_user_data_api(sender_id: str, clear_type: str = "all"):
    """API를 통한 사용자 데이터 초기화"""
    try:
        success = UserDataManager.clear_user_data(sender_id, clear_type)
        return {
            "success": success,
            "message": f"사용자 {sender_id}의 {clear_type} 데이터가 초기화되었습니다."
        }
    except Exception as e:
        return {"success": False, "error": str(e)}

# ✨ 새로운 API 엔드포인트: JSON 데이터 백업
@app.get("/backup")
async def backup_data():
    """JSON 데이터 백업"""
    try:
        user_data = load_json_data(USER_DATA_FILE)
        
        backup_data = {
            "users": user_data,
            "backup_time": time.time()
        }
        
        return backup_data
    except Exception as e:
        return {"error": str(e)}

# ✅ 루트 경로 - HTML 페이지 렌더링
@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/preview", response_class=HTMLResponse)
async def product_preview(html: str):
    try:
        decoded_html = base64.b64decode(html).decode("utf-8")
        return f"""
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <title>상품 상세 페이지</title>
            <style>
                body {{
                    font-family: '맑은 고딕', sans-serif;
                    padding: 20px;
                    max-width: 800px;
                    margin: auto;
                    line-height: 1.5;
                }}
                img {{
                    max-width: 100%;
                    height: auto;
                    display: block;
                    margin: 20px auto;
                }}
            </style>
        </head>
        <body>
            {decoded_html}
        </body>
        </html>
        """
    except Exception as e:
        return HTMLResponse(content=f"<h1>오류 발생</h1><p>{e}</p>", status_code=400)






# ✅ 디버깅용 요청 모델 추가
class ExtendedChatmallRequest(BaseModel):
    # 기존 검색용
    query: Optional[str] = None
    session_id: Optional[str] = None
    
    # 주문 과정용 필드들 추가
    action: Optional[str] = "search"  # "search", "select_product", "select_option", "set_quantity", "submit_info", "complete", "go_home", "reset"
    
    # 상품 선택용
    product_code: Optional[str] = None
    
    # 옵션 선택용
    option_name: Optional[str] = None
    extra_price: Optional[int] = 0
    
    # 수량 설정용
    quantity: Optional[int] = 1
    
    # 주문 정보용
    receiver_name: Optional[str] = None
    address: Optional[str] = None
    phone_number: Optional[str] = None
    email: Optional[str] = None



# 웹 주문 관리 클래스 추가
class WebOrderManager:
    """웹 주문 세션 관리 클래스"""
    
    @staticmethod
    def get_session_data(session_id: str):
        """웹 세션 데이터 조회"""
        all_data = load_json_data("web_orders.json")
        if session_id not in all_data:
            all_data[session_id] = {
                "product_code": None,
                "product_name": None,
                "selected_option": None,
                "extra_price": 0,
                "quantity": 0,
                "unit_price": 0,
                "shipping_fee": 0,
                "total_price": 0,
                "bundle_size": 0,
                "bundles_needed": 1,
                "product_info": {},
                "receiver_name": None,
                "address": None,
                "phone_number": None,
                "email": None,
                "step": "none",
                "last_updated": time.time()
            }
            save_json_data("web_orders.json", all_data)
        return all_data[session_id]
    
    @staticmethod
    def update_session_data(session_id: str, **kwargs):
        """웹 세션 데이터 업데이트"""
        all_data = load_json_data("web_orders.json")
        if session_id not in all_data:
            all_data[session_id] = {
                "product_code": None,
                "product_name": None,
                "selected_option": None,
                "extra_price": 0,
                "quantity": 0,
                "unit_price": 0,
                "shipping_fee": 0,
                "total_price": 0,
                "bundle_size": 0,
                "bundles_needed": 1,
                "product_info": {},
                "receiver_name": None,
                "address": None,
                "phone_number": None,
                "email": None,
                "step": "none",
                "last_updated": time.time()
            }
        
        all_data[session_id].update(kwargs)
        all_data[session_id]["last_updated"] = time.time()
        
        if save_json_data("web_orders.json", all_data):
            print(f"[WEB_SESSION] 세션 {session_id} 데이터 저장: {kwargs}")
        else:
            print(f"[WEB_SESSION] 세션 {session_id} 데이터 저장 실패")

    @staticmethod
    def clear_session_data(session_id: str):
        """웹 세션 데이터 삭제"""
        try:
            all_data = load_json_data("web_orders.json")
            if session_id in all_data:
                del all_data[session_id]
                save_json_data("web_orders.json", all_data)
                print(f"[WEB_SESSION] 세션 {session_id} 데이터 삭제 완료")
                return True
            return True
        except Exception as e:
            print(f"웹 세션 데이터 삭제 오류: {e}")
            return False

async def send_order_to_sheets_unified(session_id: str, session_data: dict) -> bool:
    """통합 구글 시트 전송 함수"""
    try:
        print(f"[SHEETS_UNIFIED] Google Sheets로 주문 정보 전송 시작 - session_id: {session_id}")
        
        # Google Sheets 연결
        sheet = init_google_sheets()
        if not sheet:
            print("[SHEETS_UNIFIED] Google Sheets 연결 실패")
            return False
        
        # 현재 시간
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 헤더 가져오기
        headers = sheet.row_values(1)
        print(f"[SHEETS_UNIFIED] 시트 헤더: {headers}")
        
        # 드롭다운 컬럼 목록
        dropdown_columns = [
            "Deposit Confirmed?",
            "Order Placed on Korean Shopping Mall?",
            "Order Received by Customer?"
        ]
        
        # 전송할 데이터 준비
        data_mapping = {
            "Order Date": current_time,
            "Who ordered?": session_data.get('receiver_name', ''),
            "Receiver's Name": session_data.get('receiver_name', ''),
            "What did they order?": session_data.get('product_name', ''),
            "Cart Total": f"{session_data.get('total_price', 0):,}원",
            "Grand Total": f"{session_data.get('total_price', 0):,}원",
            "Delivery Address": session_data.get('address', ''),
            "Email": session_data.get('email', ''),
            "phone_number": session_data.get('phone_number', ''),
            "option": session_data.get('selected_option', ''),
            "quantity": session_data.get('quantity', 0),
            "product_code": session_data.get('product_code', ''),
        }
        
        # 새 행 번호 찾기
        all_values = sheet.get_all_values()
        next_row = len(all_values) + 1
        
        print(f"[SHEETS_UNIFIED] 새 행 번호: {next_row}")
        
        # 드롭다운이 아닌 컬럼들만 개별 업데이트
        for col_index, header in enumerate(headers, start=1):
            if header not in dropdown_columns:
                value = data_mapping.get(header, "")
                if value:
                    sheet.update_cell(next_row, col_index, str(value))
                    print(f"[SHEETS_UNIFIED] 셀 업데이트: 행{next_row}, 열{col_index} ({header}) = {value}")
        
        print(f"[SHEETS_UNIFIED] 주문 정보 전송 완료!")
        return True
            
    except Exception as e:
        print(f"[SHEETS_UNIFIED] 주문 정보 전송 오류: {e}")
        import traceback
        print(f"[SHEETS_UNIFIED] 상세 오류:\n{traceback.format_exc()}")
        return False

@app.post("/chatmall")
async def extended_chatmall_endpoint(data: ExtendedChatmallRequest):
    """
    Facebook 챗봇과 동일한 트리거 메시지를 포함한 챗몰 통합 엔드포인트
    """
    try:
        print(f"🔍 [EXTENDED_CHATMALL] 요청: action={data.action}")
        
        # 세션 ID 생성/확인
        session_id = data.session_id or f"chatmall_{int(time.time())}_{random.randint(1000, 9999)}"
        # session_id = data.session_id
        # 액션별 처리
        if data.action == "search":
            return await handle_chatmall_search_with_triggers(data, session_id)
        
        elif data.action == "select_product":
            return await handle_chatmall_select_product_with_triggers(data, session_id)
        
        elif data.action == "select_option":
            return await handle_chatmall_select_option_with_triggers(data, session_id)
        
        elif data.action == "set_quantity":
            return await handle_chatmall_set_quantity_with_triggers(data, session_id)
        
        elif data.action == "submit_info":
            return await handle_chatmall_submit_info_with_triggers(data, session_id)
        
        elif data.action == "complete":
            return await handle_chatmall_complete_with_triggers(data, session_id)
        
        elif data.action == "reset":
            return await handle_chatmall_reset_with_triggers(data, session_id)
        
        # 기본값: 검색 (기존 호환성 유지)
        else:
            if data.query:
                return await handle_chatmall_search_with_triggers(data, session_id)
            else:
                return JSONResponse(
                    status_code=400,
                    content={
                        "status": "error",
                        "error": "query 또는 유효한 action이 필요합니다",
                        "valid_actions": ["search", "select_product", "select_option", "set_quantity", "submit_info", "complete", "go_home", "reset"]
                    }
                )
        
    except Exception as e:
        print(f"❌ [EXTENDED_CHATMALL] 오류 발생: {e}")
        import traceback
        error_detail = traceback.format_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "error": str(e),
                "error_detail": error_detail,
                "action": data.action
            }
        )

# ============================================================================
# 각 단계별 처리 함수들 (트리거 메시지 포함)
# ============================================================================

async def handle_chatmall_search_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """1단계: 검색 처리 (트리거 메시지 포함)"""
    try:
        if not data.query:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "검색어가 필요합니다"}
            )
        
        print(f"🔍 [CHATMALL_SEARCH] 검색 요청: {data.query}")
        print(f" 유저 ID: {session_id}")
        # AI 검색 실행
        result = external_search_and_generate_response(data.query, session_id)
        
        # 상품 캐시에 저장
        products = result.get("results", [])
        for product in products:
            product_code = product.get("상품코드")
            if product_code:
                PRODUCT_CACHE[product_code] = product
        
        # Facebook 챗봇 스타일 트리거 메시지
        trigger_message = (
            f"AI Product Search is ON!\n\n"
            f"What are you shopping for today?\n\n"
            f"AI picks, just for you! Enter what you're looking for."
        )
        
        # 응답 구조
        response = {
            "status": "success",
            "action": "search",
            "session_id": session_id,
            "trigger_message": trigger_message,
            "ai_message": result.get("combined_message_text", ""),
            "query": data.query,
            "total_results": len(products),
            "products": products,
            "next_action": "select_product",
            "navigation": {
                "can_reset": True
            }
        }
        
        return JSONResponse(content=response)
        
    except Exception as e:
        print(f"❌ [CHATMALL_SEARCH] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "search"}
        )

async def handle_chatmall_select_product_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """2단계: 상품 선택 처리 (트리거 메시지 포함)"""
    try:
        if not data.product_code:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "product_code가 필요합니다"}
            )
        
        product = PRODUCT_CACHE.get(data.product_code)
        if not product:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "유효하지 않은 상품입니다. 먼저 검색을 수행해주세요."}
            )
        
        print(f"📦 [CHATMALL_SELECT] 상품 선택: {data.product_code}")
        
        # 상품 정보 추출
        product_name = product.get('제목', '상품')
        unit_price = int(float(product.get("가격", 0) or 0))
        shipping_fee = int(float(product.get("배송비", 0) or 0))
        bundle_size = int(float(product.get("최대구매수량", 0) or 0))
        
        # 세션에 상품 정보 저장
        WebOrderManager.update_session_data(
            session_id,
            product_code=data.product_code,
            product_name=product_name,
            unit_price=unit_price,
            shipping_fee=shipping_fee,
            bundle_size=bundle_size,
            product_info=product,
            step="product_selected"
        )
        
        # Facebook 챗봇 스타일 트리거 메시지
        trigger_message = (
            f"You selected:\n\n"
            f"Product: {product_name}\n"
            f"Price: {unit_price:,}원\n"
            f"Shipping: {shipping_fee:,}원\n\n"
            f"Let's proceed with your order!"
        )
        
        # 옵션 파싱
        options = []
        options_raw = product.get("조합형옵션", "")
        
        if options_raw and str(options_raw).lower() not in ["nan", "", "none", "null"]:
            option_lines = str(options_raw).split("\n") if '\n' in str(options_raw) else [str(options_raw)]
            
            for line in option_lines:
                line = line.strip()
                if line:
                    try:
                        parts = line.split(",")
                        if len(parts) >= 2:
                            name = parts[0].strip()
                            extra_price = int(float(parts[1].strip())) if parts[1].strip() else 0
                            options.append({
                                "name": name,
                                "extra_price": extra_price,
                                "display": f"{name}" + (f" (+{extra_price:,}원)" if extra_price > 0 else "")
                            })
                    except:
                        continue
        
        # 옵션 선택 안내 메시지 (항상 카루셀 카드 방식)
        if options:
            guidance_message = "Please select an option:"
        else:
            guidance_message = "This item has a single option — please enter the quantity."
        
        return JSONResponse(content={
            "status": "success",
            "action": "select_product",
            "session_id": session_id,
            "trigger_message": trigger_message,
            "guidance_message": guidance_message,
            "selected_product": {
                "code": data.product_code,
                "name": product_name,
                "price": unit_price,
                "shipping": shipping_fee,
                "image": product.get('이미지', ''),
                "bundle_size": bundle_size
            },
            "options": options,
            "has_options": len(options) > 0,
            "option_count": len(options),
            "option_display_method": "carousel_cards",  # 항상 카루셀 카드 방식
            "options_per_card": 3,  # 카드당 최대 3개 옵션
            "total_cards": math.ceil(len(options) / 3) if options else 0,
            "next_action": "select_option" if options else "set_quantity",
            "navigation": {
                "can_reset": True,
                "can_go_back": True,
                "back_action": "search"
            }
        })
        
    except Exception as e:
        print(f"❌ [CHATMALL_SELECT] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "select_product"}
        )

async def handle_chatmall_select_option_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """3단계: 옵션 선택 처리 (트리거 메시지 포함)"""
    try:
        option_name = data.option_name or "기본옵션"
        extra_price = data.extra_price or 0
        
        print(f"⚙️ [CHATMALL_OPTION] 옵션 선택: {option_name}, 추가금액: {extra_price}")
        
        # 세션에 옵션 정보 저장
        selected_option_display = f"{option_name}" + (f" (+{extra_price:,}원)" if extra_price > 0 else "")
        
        WebOrderManager.update_session_data(
            session_id,
            selected_option=selected_option_display,
            extra_price=extra_price,
            step="option_selected"
        )
        
        # Facebook 챗봇 스타일 트리거 메시지
        trigger_message = f"Selected Options: {selected_option_display}"
        
        # 수량 입력 안내 메시지 생성
        session_data = WebOrderManager.get_session_data(session_id)
        product_name = session_data.get("product_name", "상품")
        bundle_size = session_data.get("bundle_size", 0)
        
        if bundle_size > 0:
            guidance_message = (
                f"How many do you want?\n\n"
                f"Product: {product_name}\n"
                f"Combined Shipping: packaged in sets of {bundle_size}\n"
                f"ex: Our bundled shipping rate applies to every {bundle_size} items. "
                f"If you order {bundle_size * 2} items, they will be sent in 2 separate packages "
                f"({bundle_size} items each), and the shipping fee will be applied twice.\n\n"
                f"Please enter the quantity.\n"
                f"(예: 1, 25, 50, 100, 150)"
            )
        else:
            guidance_message = (
                f"How many do you want?\n\n"
                f"Product: {product_name}\n"
                f"Single Shipment (No Quantity Limit)\n\n"
                f"Please enter the quantity.\n"
                f"(예: 1, 10, 50, 100)"
            )
        
        return JSONResponse(content={
            "status": "success",
            "action": "select_option",
            "session_id": session_id,
            "trigger_message": trigger_message,
            "guidance_message": guidance_message,
            "selected_option": {
                "name": option_name,
                "extra_price": extra_price,
                "display": selected_option_display
            },
            "next_action": "set_quantity",
            "navigation": {
                "can_reset": True,
                "can_go_back": True,
                "back_action": "select_product"
            }
        })
        
    except Exception as e:
        print(f"❌ [CHATMALL_OPTION] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "select_option"}
        )

async def handle_chatmall_set_quantity_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """4단계: 수량 설정 처리 (트리거 메시지 포함)"""
    try:
        quantity = data.quantity or 1
        if quantity <= 0:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "수량은 1개 이상이어야 합니다"}
            )
        
        print(f"🔢 [CHATMALL_QUANTITY] 수량 설정: {quantity}")
        
        # 가격 계산
        session_data = WebOrderManager.get_session_data(session_id)
        unit_price = session_data.get("unit_price", 0)
        extra_price = session_data.get("extra_price", 0)
        shipping_fee = session_data.get("shipping_fee", 0)
        bundle_size = session_data.get("bundle_size", 0)
        product_name = session_data.get("product_name", "상품")
        selected_option = session_data.get("selected_option", "기본옵션")
        
        item_price = unit_price + extra_price
        item_total = item_price * quantity
        
        # 묶음배송 계산
        if bundle_size > 0:
            bundles_needed = math.ceil(quantity / bundle_size)
            total_shipping = shipping_fee * bundles_needed
        else:
            bundles_needed = 1
            total_shipping = shipping_fee
        
        total_price = item_total + total_shipping
        
        # 세션에 수량 및 가격 정보 저장
        WebOrderManager.update_session_data(
            session_id,
            quantity=quantity,
            total_price=total_price,
            calculated_shipping=total_shipping,
            bundles_needed=bundles_needed,
            step="quantity_set"
        )
        
        # 묶음배송 안내 메시지 (Facebook 챗봇 스타일)
        bundle_message = None
        if bundle_size > 0 and bundles_needed > 1:
            bundle_message = (
                f"Bundled Shipping Details:\n"
                f"   Quantity: {quantity} items\n"
                f"   Bundles: {bundle_size} items/bundle × {bundles_needed} bundles\n"
                f"   Shipping Fee: KRW {shipping_fee:,} × {bundles_needed} = KRW {total_shipping:,}"
            )
        
        # 주문 확인 메시지 (Facebook 챗봇 스타일)
        trigger_message = (
            f"Would you like to continue with your order?\n\n"
            f"Product: {product_name}\n"
            f"Quantity: {quantity} items\n"
            f"Unit Price: KRW {unit_price:,}"
        )
        
        if extra_price > 0:
            trigger_message += f"\nAdd-on: KRW {extra_price:,}"
        
        if bundle_size > 0 and bundles_needed > 1:
            trigger_message += f"\n\nBundled Shipping: {bundle_size} items/bundle × {bundles_needed} bundles"
        
        trigger_message += f"\nShipping Fee: KRW {total_shipping:,}"
        trigger_message += f"\nTotal: KRW {total_price:,}"
        
        # 배송 정보 수집 안내
        guidance_message = (
            "Order confirmed! Let's collect your delivery information.\n\n"
            "To deliver your items safely and quickly, please provide the required information."
        )
        
        return JSONResponse(content={
            "status": "success",
            "action": "set_quantity",
            "session_id": session_id,
            "trigger_message": trigger_message,
            "bundle_message": bundle_message,
            "guidance_message": guidance_message,
            "quantity": quantity,
            "price_summary": {
                "unit_price": unit_price,
                "extra_price": extra_price,
                "item_total": item_total,
                "shipping_fee": total_shipping,
                "total_price": total_price,
                "bundle_info": {
                    "bundle_size": bundle_size,
                    "bundles_needed": bundles_needed
                } if bundle_size > 0 else None
            },
            "next_action": "submit_info",
            "input_steps": [
                {
                    "field": "receiver_name",
                    "question": "What is your name or the recipient's full name?",
                    "type": "text"
                },
                {
                    "field": "address", 
                    "question": "Thank you! What is the shipping address?",
                    "type": "textarea"
                },
                {
                    "field": "phone_number",
                    "question": "Almost done! May I have your phone number?", 
                    "type": "tel"
                },
                {
                    "field": "email",
                    "question": "Last step! What is your email address?",
                    "type": "email"
                }
            ],
            "navigation": {
                "can_reset": True,
                "can_go_back": True,
                "back_action": "select_option"
            }
        })
        
    except Exception as e:
        print(f"❌ [CHATMALL_QUANTITY] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "set_quantity"}
        )

async def handle_chatmall_submit_info_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """5단계: 주문자 정보 입력 처리 (트리거 메시지 포함)"""
    try:
        # 필수 필드 검증
        required_fields = {
            "receiver_name": data.receiver_name,
            "address": data.address,
            "phone_number": data.phone_number,
            "email": data.email
        }
        
        for field_name, field_value in required_fields.items():
            if not field_value:
                return JSONResponse(
                    status_code=400,
                    content={"status": "error", "error": f"{field_name}는 필수 입력 항목입니다"}
                )
        
        print(f"📝 [CHATMALL_INFO] 주문 정보 입력: {data.receiver_name}")
        
        # 세션에 주문자 정보 저장
        WebOrderManager.update_session_data(
            session_id,
            receiver_name=data.receiver_name,
            address=data.address,
            phone_number=data.phone_number,
            email=data.email,
            step="info_submitted"
        )
        
        # 주문 요약 정보 생성
        session_data = WebOrderManager.get_session_data(session_id)
        
        # Facebook 챗봇 스타일 주문 확인 메시지
        trigger_message = (
            f"Kindly review and confirm the info below is correct:\n\n"
            f"Name: {data.receiver_name}\n"
            f"Address: {data.address}\n"
            f"Contact #: {data.phone_number}\n"
            f"Email: {data.email}\n\n"
            f"Product: {session_data.get('product_name', '')}\n"
            f"Option: {session_data.get('selected_option', '')}\n"
            f"Quantity: {session_data.get('quantity', '')}\n"
            f"Total_money: {session_data.get('total_price', 0):,}원"
        )
        
        # 결제 안내 메시지
        payment_guidance = (
            "For secure payment processing, please make a deposit to the account below.\n\n"
            "Bank Name: 하나은행 / Hana Bank\n"
            "Account Number: 841-910015-85404\n"
            "Account Name: (주)나로수\n\n"
            "After sending the payment, please click the \"COMPLETE ORDER\" button so we can process your order faster!"
        )
        
        return JSONResponse(content={
            "status": "success",
            "action": "submit_info",
            "session_id": session_id,
            "trigger_message": trigger_message,
            "payment_guidance": payment_guidance,
            "order_summary": {
                "customer_info": {
                    "receiver_name": data.receiver_name,
                    "address": data.address,
                    "phone_number": data.phone_number,
                    "email": data.email
                },
                "product_info": {
                    "product_name": session_data.get("product_name"),
                    "selected_option": session_data.get("selected_option"),
                    "quantity": session_data.get("quantity"),
                    "total_price": session_data.get("total_price")
                }
            },
            "next_action": "complete",
            "navigation": {
                "can_reset": True,
                "can_go_back": True,
                "back_action": "set_quantity"
            }
        })
        
    except Exception as e:
        print(f"❌ [CHATMALL_INFO] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "submit_info"}
        )

async def handle_chatmall_complete_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """6단계: 주문 완료 처리 (트리거 메시지 포함)"""
    try:
        print(f"🎉 [CHATMALL_COMPLETE] 주문 완료 처리")
        
        # 세션 데이터 확인
        session_data = WebOrderManager.get_session_data(session_id)
        
        if not session_data.get("receiver_name") or not session_data.get("product_name"):
            return JSONResponse(
                status_code=400,
                content={"status": "error", "error": "주문 정보가 완전하지 않습니다"}
            )
        
        # 결제 확인 메시지 (Facebook 챗봇 스타일)
        payment_confirmation_message = (
            "Payment Confirmation\n\n"
            "Once we confirm your payment, we'll process your order right away!\n\n"
            "Please give us a moment while our ChatMall team confirms your payment."
        )
        
        # 구글 시트 전송
        try:
            sheet_success = await send_order_to_sheets_unified(session_id, session_data)
            
            if sheet_success:
                WebOrderManager.update_session_data(session_id, step="completed")
                order_number = f"CHATMALL{int(time.time())}"
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Facebook 챗봇 스타일 주문 완료 메시지
                completion_message = (
                    f"Order Completed Successfully!\n\n"
                    f"Order Time: {timestamp}\n"
                    f"Customer: {session_data.get('receiver_name', '')}\n"
                    f"Receiver: {session_data.get('receiver_name', '')}\n"
                    f"Address: {session_data.get('address', '')}\n"
                    f"Contact: {session_data.get('phone_number', '')}\n"
                    f"Email: {session_data.get('email', '')}\n\n"
                    f"Order Details:\n"
                    f"Product: {session_data.get('product_name', '')}\n"
                    f"Option: {session_data.get('selected_option', '')}\n"
                    f"Quantity: {session_data.get('quantity', 0)}\n"
                    f"Total: {session_data.get('total_price', 0):,}원\n\n"
                    f"We'll start processing your order right away!\n"
                    f"Thank you for shopping with ChatMall!"
                )
                
                return JSONResponse(content={
                    "status": "success",
                    "action": "complete",
                    "session_id": session_id,
                    "trigger_message": payment_confirmation_message,
                    "completion_message": completion_message,
                    "order_number": order_number,
                    "message": "주문이 성공적으로 완료되었습니다!",
                    "order_details": {
                        "receiver_name": session_data.get("receiver_name"),
                        "product_name": session_data.get("product_name"),
                        "quantity": session_data.get("quantity"),
                        "total_price": session_data.get("total_price"),
                        "timestamp": timestamp
                    },
                    "navigation": {
                        "can_start_new_order": True
                    }
                })
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "status": "error", 
                        "error": "주문 처리 중 오류가 발생했습니다",
                        "trigger_message": "There was a temporary issue with our order processing system."
                    }
                )
                
        except Exception as e:
            print(f"❌ [CHATMALL_COMPLETE] 구글 시트 오류: {e}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "error", 
                    "error": "구글 시트 전송에 실패했습니다",
                    "trigger_message": "There was a temporary issue with our order processing system."
                }
            )
        
    except Exception as e:
        print(f"❌ [CHATMALL_COMPLETE] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "complete"}
        )

async def handle_chatmall_reset_with_triggers(data: ExtendedChatmallRequest, session_id: str):
    """대화 초기화 처리 (트리거 메시지 포함)"""
    try:
        print(f"♻️ [CHATMALL_RESET] 대화 초기화: {session_id}")
        
        # 세션 데이터 완전 초기화
        try:
            WebOrderManager.clear_session_data(session_id)
            print(f"♻️ [CHATMALL_RESET] 세션 데이터 완전 삭제: {session_id}")
        except Exception as e:
            print(f"♻️ [CHATMALL_RESET] 세션 삭제 오류: {e}")
        
        # Redis 대화 기록 초기화
        try:
            if isinstance(session_id, str):
                clear_message_history(session_id)
                print(f"♻️ [CHATMALL_RESET] Redis 대화 기록 초기화 완료")
        except Exception as e:
            print(f"♻️ [CHATMALL_RESET] Redis 초기화 오류: {e}")
        
        # Facebook 챗봇 스타일 리셋 메시지
        reset_message = (
            "Chat history cleared!\n\n"
            "Now enter what you're looking for:\n\n"
            "For example: portable fan, striped tee, women's light shoes, 100 paper cups\n\n"
            "What are you shopping for today?"
        )
        
        navigation_guidance = (
            "Click \"Reset\" below to reset the conversation history\n\n"
        )
        
        return JSONResponse(content={
            "status": "success",
            "action": "reset",
            "session_id": session_id,
            "trigger_message": reset_message,
            "navigation_guidance": navigation_guidance,
            "message": "대화 기록이 초기화되었습니다. 새로운 대화를 시작하세요!",
            "reset_completed": True,
            "conversation_cleared": True,
            "next_action": "search",
            "navigation": {
                "can_search": True,
                "show_navigation_buttons": True
            },
            "navigation_buttons": [
                {
                    "title": "Reset Conversation",
                    "action": "reset",
                    "type": "postback"
                }
            ]
        })
        
    except Exception as e:
        print(f"❌ [CHATMALL_RESET] 오류: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e), "action": "reset"}
        )

# 테스트
@app.get("/view-conversations", response_class=HTMLResponse)
async def view_conversations_web():
    """웹 브라우저에서 대화 기록을 예쁘게 볼 수 있는 페이지"""
    try:
        # JSON 파일 읽기
        if os.path.exists(CONVERSATION_DATA_FILE):
            with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        else:
            conversations = {}
        
        # HTML 생성
        html_content = """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Facebook 대화 기록</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .header {
                    background: white;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 20px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .user-card {
                    background: white;
                    margin-bottom: 20px;
                    border-radius: 8px;
                    overflow: hidden;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .user-header {
                    background: #1877f2;
                    color: white;
                    padding: 15px 20px;
                    font-weight: bold;
                    cursor: pointer;
                }
                .user-header:hover {
                    background: #166fe5;
                }
                .messages {
                    max-height: 400px;
                    overflow-y: auto;
                    padding: 0;
                    display: none;
                }
                .message {
                    padding: 10px 20px;
                    border-bottom: 1px solid #eee;
                    display: flex;
                    align-items: flex-start;
                    gap: 10px;
                }
                .message:last-child {
                    border-bottom: none;
                }
                .message.user {
                    background: #f0f2f5;
                }
                .message.bot {
                    background: #e3f2fd;
                }
                .message-type {
                    font-weight: bold;
                    min-width: 40px;
                    font-size: 12px;
                }
                .user-type { color: #1877f2; }
                .bot-type { color: #4caf50; }
                .message-content {
                    flex: 1;
                    word-break: break-word;
                }
                .timestamp {
                    font-size: 11px;
                    color: #666;
                    margin-top: 5px;
                }
                .stats {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }
                .stat-card {
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .stat-number {
                    font-size: 24px;
                    font-weight: bold;
                    color: #1877f2;
                }
                .stat-label {
                    font-size: 14px;
                    color: #666;
                    margin-top: 5px;
                }
                .refresh-btn {
                    background: #1877f2;
                    color: white;
                    border: none;
                    padding: 10px 20px;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 14px;
                }
                .refresh-btn:hover {
                    background: #166fe5;
                }
                .search-box {
                    width: 100%;
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-bottom: 20px;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📱 Facebook 대화 기록</h1>
                    <button class="refresh-btn" onclick="location.reload()">🔄 새로고침</button>
                    <br><br>
                    <input type="text" class="search-box" id="searchBox" placeholder="사용자 ID 또는 메시지 내용으로 검색..." onkeyup="searchMessages()">
                </div>
                
                <div class="stats">
        """
        
        # 통계 계산
        total_users = len(conversations)
        total_messages = sum(len(msgs) for msgs in conversations.values())
        
        html_content += f"""
                    <div class="stat-card">
                        <div class="stat-number">{total_users}</div>
                        <div class="stat-label">총 사용자</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{total_messages}</div>
                        <div class="stat-label">총 메시지</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-number">{datetime.now().strftime('%H:%M')}</div>
                        <div class="stat-label">마지막 업데이트</div>
                    </div>
                </div>
        """
        
        # 사용자별 대화 기록
        for sender_id, messages in conversations.items():
            message_count = len(messages)
            last_message = messages[-1]['timestamp'] if messages else "없음"
            
            html_content += f"""
                <div class="user-card">
                    <div class="user-header" onclick="toggleMessages('{sender_id}')">
                        👤 사용자: {sender_id} | 메시지: {message_count}개 | 최근: {last_message}
                    </div>
                    <div class="messages" id="messages-{sender_id}">
            """
            
            # 메시지들
            for msg in messages:
                msg_type = msg['type']
                content = msg['message']
                timestamp = msg['timestamp']
                
                type_class = 'user' if msg_type == 'user' else 'bot'
                type_label = '👤' if msg_type == 'user' else '🤖'
                type_color = 'user-type' if msg_type == 'user' else 'bot-type'
                
                html_content += f"""
                        <div class="message {type_class}">
                            <div class="message-type {type_color}">{type_label}</div>
                            <div class="message-content">
                                {content}
                                <div class="timestamp">{timestamp}</div>
                            </div>
                        </div>
                """
            
            html_content += """
                    </div>
                </div>
            """
        
        html_content += """
            </div>
            
            <script>
                function toggleMessages(userId) {
                    const messages = document.getElementById('messages-' + userId);
                    if (messages.style.display === 'none' || messages.style.display === '') {
                        messages.style.display = 'block';
                    } else {
                        messages.style.display = 'none';
                    }
                }
                
                function searchMessages() {
                    const searchTerm = document.getElementById('searchBox').value.toLowerCase();
                    const userCards = document.querySelectorAll('.user-card');
                    
                    userCards.forEach(card => {
                        const header = card.querySelector('.user-header').textContent.toLowerCase();
                        const messages = card.querySelectorAll('.message-content');
                        let hasMatch = header.includes(searchTerm);
                        
                        if (!hasMatch) {
                            messages.forEach(msg => {
                                if (msg.textContent.toLowerCase().includes(searchTerm)) {
                                    hasMatch = true;
                                }
                            });
                        }
                        
                        card.style.display = hasMatch ? 'block' : 'none';
                    });
                }
                
                // 5분마다 자동 새로고침
                setInterval(() => {
                    location.reload();
                }, 300000);
            </script>
        </body>
        </html>
        """
        
        return html_content
        
    except Exception as e:
        return f"<h1>오류 발생</h1><p>{str(e)}</p>"

@app.get("/conversations-json")
async def get_conversations_json():
    """JSON 형태로 대화 기록 반환"""
    try:
        if os.path.exists(CONVERSATION_DATA_FILE):
            with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        else:
            conversations = {}
        
        return JSONResponse(content={
            "status": "success",
            "data": conversations,
            "total_users": len(conversations),
            "total_messages": sum(len(msgs) for msgs in conversations.values()),
            "last_updated": datetime.now().isoformat()
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/download-conversations")
async def download_conversations():
    """JSON 파일 다운로드"""
    try:
        if os.path.exists(CONVERSATION_DATA_FILE):
            with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                content = f.read()
            
            from fastapi.responses import Response
            
            return Response(
                content=content,
                media_type="application/json",
                headers={
                    "Content-Disposition": f"attachment; filename=facebook_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                }
            )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "대화 기록 파일이 없습니다"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/conversation/{sender_id}")
async def get_single_conversation(sender_id: str):
    """특정 사용자의 대화 기록만 조회"""
    try:
        if os.path.exists(CONVERSATION_DATA_FILE):
            with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
        else:
            conversations = {}
        
        if sender_id in conversations:
            return JSONResponse(content={
                "status": "success",
                "sender_id": sender_id,
                "messages": conversations[sender_id],
                "message_count": len(conversations[sender_id])
            })
        else:
            return JSONResponse(
                status_code=404,
                content={"error": f"사용자 {sender_id}의 대화 기록이 없습니다"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/conversations")
async def clear_all_conversations():
    """모든 대화 기록 삭제 (주의!)"""
    try:
        if os.path.exists(CONVERSATION_DATA_FILE):
            os.remove(CONVERSATION_DATA_FILE)
            return JSONResponse(content={
                "status": "success",
                "message": "모든 대화 기록이 삭제되었습니다"
            })
        else:
            return JSONResponse(content={
                "status": "success", 
                "message": "삭제할 파일이 없습니다"
            })
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.delete("/conversation/{sender_id}")
async def clear_user_conversation(sender_id: str):
    """특정 사용자의 대화 기록만 삭제"""
    try:
        if os.path.exists(CONVERSATION_DATA_FILE):
            with open(CONVERSATION_DATA_FILE, 'r', encoding='utf-8') as f:
                conversations = json.load(f)
            
            if sender_id in conversations:
                del conversations[sender_id]
                
                with open(CONVERSATION_DATA_FILE, 'w', encoding='utf-8') as f:
                    json.dump(conversations, f, ensure_ascii=False, indent=2)
                
                return JSONResponse(content={
                    "status": "success",
                    "message": f"사용자 {sender_id}의 대화 기록이 삭제되었습니다"
                })
            else:
                return JSONResponse(
                    status_code=404,
                    content={"error": f"사용자 {sender_id}의 대화 기록이 없습니다"}
                )
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "대화 기록 파일이 없습니다"}
            )
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# 파일 시스템 정보 확인용
@app.get("/file-info")
async def get_file_info():
    """JSON 파일 존재 여부 및 정보 확인"""
    try:
        file_info = {
            "file_path": CONVERSATION_DATA_FILE,
            "exists": os.path.exists(CONVERSATION_DATA_FILE),
            "current_directory": os.getcwd(),
            "files_in_directory": os.listdir(".")
        }
        
        if file_info["exists"]:
            stat = os.stat(CONVERSATION_DATA_FILE)
            file_info.update({
                "file_size": stat.st_size,
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "creation_time": datetime.fromtimestamp(stat.st_ctime).isoformat()
            })
        
        return JSONResponse(content=file_info)
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


# FastAPI 서버 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5051))
    uvicorn.run(app, host="0.0.0.0", port=port)
