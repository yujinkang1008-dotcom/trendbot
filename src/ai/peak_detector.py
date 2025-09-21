"""
피크 탐지 모듈
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from typing import List, Dict, Any
from datetime import datetime, timedelta
import plotly.graph_objects as go

class PeakDetector:
    """피크 탐지 클래스"""
    
    def __init__(self):
        pass
    
    def detect_peaks(self, data: pd.Series, prominence: float = 0.1, distance: int = 7) -> List[Dict]:
        """
        시계열 데이터에서 피크 탐지
        
        Args:
            data: 시계열 데이터 (pandas Series)
            prominence: 피크의 두드러짐 임계값
            distance: 피크 간 최소 거리
            
        Returns:
            List[Dict]: 피크 정보 리스트
        """
        if data.empty or len(data) < 3:
            return []
        
        try:
            # 데이터 정규화
            normalized_data = (data - data.min()) / (data.max() - data.min())
            
            # 피크 탐지
            peaks, properties = find_peaks(
                normalized_data,
                prominence=prominence,
                distance=distance
            )
            
            # 피크 정보 추출
            peak_info = []
            for i, peak_idx in enumerate(peaks):
                peak_date = data.index[peak_idx]
                peak_value = data.iloc[peak_idx]
                
                # 피크 강도 계산
                prominence_value = properties['prominences'][i] if 'prominences' in properties else 0
                
                peak_info.append({
                    'date': peak_date,
                    'value': peak_value,
                    'index': peak_idx,
                    'prominence': prominence_value,
                    'strength': prominence_value * peak_value
                })
            
            # 강도순 정렬
            peak_info.sort(key=lambda x: x['strength'], reverse=True)
            
            return peak_info
            
        except Exception as e:
            print(f"피크 탐지 오류: {e}")
            return []
    
    def detect_news_peaks(self, news_data: Dict[str, List[Dict]], keywords: List[str]) -> Dict[str, List[Dict]]:
        """
        뉴스 데이터에서 피크 탐지
        
        Args:
            news_data: 뉴스 데이터 (소스별)
            keywords: 검색 키워드
            
        Returns:
            Dict: 키워드별 피크 정보
        """
        peak_results = {}
        
        for keyword in keywords:
            keyword_peaks = []
            
            # 각 소스별로 피크 탐지
            for source, articles in news_data.items():
                if not articles:
                    continue
                
                # 날짜별 기사 수 계산
                df = pd.DataFrame(articles)
                if 'pub_date' in df.columns:
                    df['date'] = pd.to_datetime(df['pub_date'], errors='coerce')
                    daily_counts = df.groupby(df['date'].dt.date).size()
                    
                    # 피크 탐지
                    peaks = self.detect_peaks(daily_counts)
                    
                    for peak in peaks:
                        peak_date = peak['date']
                        peak_value = peak['value']
                        
                        # 해당 날짜의 기사들 추출
                        peak_articles = df[df['date'].dt.date == peak_date].to_dict('records')
                        
                        keyword_peaks.append({
                            'source': source,
                            'date': peak_date,
                            'value': peak_value,
                            'articles': peak_articles,
                            'strength': peak['strength']
                        })
            
            # 강도순 정렬
            keyword_peaks.sort(key=lambda x: x['strength'], reverse=True)
            peak_results[keyword] = keyword_peaks
        
        return peak_results
    
    def create_peak_visualization(self, data: pd.Series, peaks: List[Dict]) -> go.Figure:
        """
        피크 시각화 생성
        
        Args:
            data: 시계열 데이터
            peaks: 피크 정보
            
        Returns:
            go.Figure: 피크 시각화 차트
        """
        fig = go.Figure()
        
        # 기본 데이터 플롯
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data.values,
            mode='lines',
            name='데이터',
            line=dict(color='blue', width=2)
        ))
        
        # 피크 표시
        if peaks:
            peak_dates = [peak['date'] for peak in peaks]
            peak_values = [peak['value'] for peak in peaks]
            
            fig.add_trace(go.Scatter(
                x=peak_dates,
                y=peak_values,
                mode='markers',
                name='피크',
                marker=dict(
                    size=12,
                    color='red',
                    symbol='diamond'
                ),
                hovertemplate='<b>피크</b><br>' +
                             '날짜: %{x}<br>' +
                             '값: %{y}<br>' +
                             '<extra></extra>'
            ))
        
        fig.update_layout(
            title='피크 탐지 결과',
            xaxis_title='날짜',
            yaxis_title='값',
            template='plotly_white',
            hovermode='x unified'
        )
        
        return fig
    
    def analyze_peak_causes(self, peak_date: str, articles: List[Dict], gemini_analyzer) -> str:
        """
        피크 원인 분석
        
        Args:
            peak_date: 피크 날짜
            articles: 해당 날짜의 기사들
            gemini_analyzer: Gemini 분석기
            
        Returns:
            str: 피크 원인 분석 결과
        """
        if not articles:
            return "해당 날짜의 기사가 없습니다."
        
        # 기사 텍스트 결합
        article_texts = []
        for article in articles[:10]:  # 최대 10개 기사만
            title = article.get('title', '')
            description = article.get('description', '')
            if title or description:
                article_texts.append(f"제목: {title}\n내용: {description}")
        
        if not article_texts:
            return "분석할 기사 내용이 없습니다."
        
        combined_text = "\n\n".join(article_texts)
        
        prompt = f"""
        {peak_date} 날짜에 검색량이 급증한 원인을 다음 기사들을 바탕으로 분석해주세요:
        
        {combined_text}
        
        다음 사항을 포함해서 분석해주세요:
        1. 주요 사건이나 이슈
        2. 급증 원인
        3. 사회적 영향
        4. 관련 키워드
        
        한국어로 작성해주세요.
        """
        
        try:
            response = gemini_analyzer.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"피크 원인 분석 오류: {e}")
            return "피크 원인 분석 중 오류가 발생했습니다."
    
    def get_peak_summary(self, peak_results: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """
        피크 요약 정보 생성
        
        Args:
            peak_results: 피크 탐지 결과
            
        Returns:
            Dict: 피크 요약 정보
        """
        summary = {
            'total_peaks': 0,
            'top_peaks': [],
            'keyword_stats': {}
        }
        
        all_peaks = []
        
        for keyword, peaks in peak_results.items():
            summary['keyword_stats'][keyword] = {
                'peak_count': len(peaks),
                'max_strength': max([p['strength'] for p in peaks]) if peaks else 0
            }
            
            all_peaks.extend([(keyword, peak) for peak in peaks])
            summary['total_peaks'] += len(peaks)
        
        # 전체 피크를 강도순으로 정렬
        all_peaks.sort(key=lambda x: x[1]['strength'], reverse=True)
        
        # 상위 5개 피크 선택
        summary['top_peaks'] = all_peaks[:5]
        
        return summary
