"""
공통 설정 모듈
"""
import os
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("trendbot.env")

class Config:
    """설정 클래스"""
    
    # 디버그 모드
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    STRICT_FETCH = os.getenv("STRICT_FETCH", "True").lower() == "true"
    
    # API 키
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    NAVER_CLIENT_ID = os.getenv("NAVER_CLIENT_ID")
    NAVER_CLIENT_SECRET = os.getenv("NAVER_CLIENT_SECRET")
    HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
    
    # 타임존
    TIMEZONE = os.getenv("TZ", "Asia/Seoul")
    
    # 데이터베이스 설정
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trendbot.db")
    
    # 디버그 폴더
    DEBUG_FOLDER = "data/debug"
    
    @classmethod
    def ensure_debug_folder(cls):
        """디버그 폴더 생성"""
        if cls.DEBUG:
            import os
            os.makedirs(cls.DEBUG_FOLDER, exist_ok=True)
            print(f"✅ 디버그 폴더 생성: {cls.DEBUG_FOLDER}")
    
    @classmethod
    def validate_config(cls):
        """설정 검증"""
        missing_keys = []
        
        if not cls.GEMINI_API_KEY:
            missing_keys.append("GEMINI_API_KEY")
        if not cls.NAVER_CLIENT_ID:
            missing_keys.append("NAVER_CLIENT_ID")
        if not cls.NAVER_CLIENT_SECRET:
            missing_keys.append("NAVER_CLIENT_SECRET")
        if not cls.HUGGINGFACE_API_KEY:
            missing_keys.append("HUGGINGFACE_API_KEY")
        
        if missing_keys:
            print(f"⚠️ 누락된 환경변수: {', '.join(missing_keys)}")
            print("trendbot.env 파일을 확인해주세요.")
        else:
            print("✅ 모든 환경변수 설정 완료")
        
        # 디버그 폴더 생성
        cls.ensure_debug_folder()
    
    @classmethod
    def get_masked_api_keys(cls):
        """마스킹된 API 키 반환"""
        def mask_key(key):
            if not key:
                return "None"
            return key[:4] + "*" * (len(key) - 4) if len(key) > 4 else "*" * len(key)
        
        return {
            "GEMINI_API_KEY": mask_key(cls.GEMINI_API_KEY),
            "NAVER_CLIENT_ID": mask_key(cls.NAVER_CLIENT_ID),
            "NAVER_CLIENT_SECRET": mask_key(cls.NAVER_CLIENT_SECRET),
            "HUGGINGFACE_API_KEY": mask_key(cls.HUGGINGFACE_API_KEY)
        }
