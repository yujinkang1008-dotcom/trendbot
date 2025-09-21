"""
환경 설정 및 API 키 관리
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("trendbot.env")

class Config:
    """설정 클래스"""
    
    # API 키
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # 타임존
    TIMEZONE = os.getenv("TZ", "Asia/Seoul")
    
    # 데이터베이스 설정
    DB_PATH = "data/trendbot.db"
    CHROMA_PATH = "data/chroma_db"
    
    # 수집 설정
    DEFAULT_PERIOD_DAYS = 365  # 기본 1년
    MAX_KEYWORDS = 5  # 최대 키워드 수
    
    # 분석 설정
    TOP_KEYWORDS_COUNT = 10
    TOP_TOPICS_COUNT = 10
    
    @classmethod
    def validate_config(cls):
        """설정 검증"""
        required_keys = [
            ("GEMINI_API_KEY", cls.GEMINI_API_KEY),
            ("NAVER_CLIENT_ID", cls.NAVER_CLIENT_ID),
            ("NAVER_CLIENT_SECRET", cls.NAVER_CLIENT_SECRET)
        ]
        
        missing_keys = []
        for key_name, key_value in required_keys:
            if not key_value:
                missing_keys.append(key_name)
        
        if missing_keys:
            raise ValueError(f"다음 환경 변수가 설정되지 않았습니다: {', '.join(missing_keys)}")
        
        return True
