"""
디버그 추적 모듈
"""
import os
import pandas as pd
from src.common.config import Config

def snapshot_df(df, name):
    """
    DEBUG 모드일 때만 DataFrame을 CSV로 저장
    
    Args:
        df: 저장할 DataFrame
        name: 파일명 (확장자 제외)
    """
    if not Config.DEBUG:
        return
    
    if df is None or df.empty:
        print(f"⚠️ {name}: 빈 DataFrame, 저장 건너뛰기")
        return
    
    try:
        # 디버그 폴더 확인
        Config.ensure_debug_folder()
        
        # 파일 경로
        file_path = os.path.join(Config.DEBUG_FOLDER, f"{name}.csv")
        
        # CSV 저장
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        print(f"📁 {name} 저장: {file_path} ({df.shape})")
        
    except Exception as e:
        print(f"❌ {name} 저장 실패: {e}")

def log_shape(df, name):
    """
    DataFrame의 shape와 columns를 표준 출력
    
    Args:
        df: 로그할 DataFrame
        name: 식별자
    """
    if df is None:
        print(f"📊 {name}: None")
        return
    
    if df.empty:
        print(f"📊 {name}: 빈 DataFrame")
        return
    
    print(f"📊 {name}: {df.shape} - 컬럼: {list(df.columns)}")
    
    # DEBUG 모드일 때 샘플 데이터도 출력
    if Config.DEBUG and len(df) > 0:
        print(f"   샘플 데이터 (상위 3행):")
        print(df.head(3).to_string())
        print()
