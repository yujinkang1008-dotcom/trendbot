"""
트렌드 데이터 병합 모듈
"""
import pandas as pd
from src.common.trace import snapshot_df, log_shape

def merge_trends(google_df, naver_df, how="outer"):
    """
    Google Trends와 Naver DataLab 데이터를 병합
    
    Args:
        google_df: Google Trends DataFrame (period + 키워드 컬럼들)
        naver_df: Naver DataLab DataFrame (period + 키워드 컬럼들)
        how: 병합 방식 ("outer", "inner", "left", "right")
        
    Returns:
        pd.DataFrame: 병합된 트렌드 데이터
        
    Raises:
        ValueError: 병합 실패 또는 빈 결과
    """
    try:
        print("🔄 트렌드 데이터 병합 시작")
        
        # 입력 데이터 검증
        if google_df is None or google_df.empty:
            raise ValueError("Google Trends 데이터가 없습니다")
        
        if naver_df is None or naver_df.empty:
            raise ValueError("Naver DataLab 데이터가 없습니다")
        
        # period 컬럼 확인
        if 'period' not in google_df.columns:
            raise ValueError("Google Trends에 period 컬럼이 없습니다")
        
        if 'period' not in naver_df.columns:
            raise ValueError("Naver DataLab에 period 컬럼이 없습니다")
        
        log_shape(google_df, "google_trends")
        log_shape(naver_df, "naver_datalab")
        
        # period 기준으로 병합
        merged_df = google_df.merge(naver_df, on='period', how=how)
        
        if merged_df.empty:
            raise ValueError("병합된 데이터가 비어있습니다")
        
        # period 기준으로 정렬
        merged_df = merged_df.sort_values('period').reset_index(drop=True)
        
        log_shape(merged_df, "trends_merged")
        snapshot_df(merged_df, "trends_merged")
        
        print(f"✅ 트렌드 데이터 병합 완료: {merged_df.shape}")
        return merged_df
        
    except Exception as e:
        print(f"❌ 트렌드 데이터 병합 실패: {e}")
        raise

def prepare_for_visualization(df):
    """
    시각화를 위한 데이터 전처리 (결측값 처리)
    
    Args:
        df: 병합된 트렌드 DataFrame
        
    Returns:
        pd.DataFrame: 시각화용 데이터 (결측값 처리됨)
    """
    try:
        print("🎨 시각화용 데이터 전처리")
        
        if df is None or df.empty:
            raise ValueError("전처리할 데이터가 없습니다")
        
        # period 컬럼 제외하고 숫자 컬럼만 처리
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # forward fill 후 0으로 채우기
        df_viz = df.copy()
        df_viz[numeric_cols] = df_viz[numeric_cols].fillna(method='ffill').fillna(0)
        
        log_shape(df_viz, "trends_final")
        snapshot_df(df_viz, "trends_final")
        
        print(f"✅ 시각화용 데이터 준비 완료: {df_viz.shape}")
        return df_viz
        
    except Exception as e:
        print(f"❌ 시각화용 데이터 전처리 실패: {e}")
        raise
