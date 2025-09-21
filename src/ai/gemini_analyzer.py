"""
Gemini AI 분석 모듈
"""
import google.generativeai as genai
from typing import List, Dict, Any
import json

class GeminiAnalyzer:
    """Gemini AI 분석 클래스"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def summarize_articles(self, articles: List[Dict]) -> str:
        """
        기사 요약
        
        Args:
            articles: 기사 데이터 리스트
            
        Returns:
            str: 요약 텍스트
        """
        if not articles:
            return "요약할 기사가 없습니다."
        
        # 기사 텍스트 결합
        article_texts = []
        for article in articles[:10]:  # 최대 10개 기사만
            title = article.get('title', '')
            description = article.get('description', '')
            article_texts.append(f"제목: {title}\n내용: {description}")
        
        combined_text = "\n\n".join(article_texts)
        
        prompt = f"""
        다음 뉴스 기사들을 분석하여 핵심 내용을 요약해주세요:
        
        {combined_text}
        
        요약 시 다음 사항을 포함해주세요:
        1. 주요 이슈 3가지
        2. 핵심 내용 요약
        3. 시사점
        
        한국어로 작성해주세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"기사 요약 오류: {e}")
            return "요약 생성 중 오류가 발생했습니다."
    
    def analyze_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        감성 분석
        
        Args:
            texts: 분석할 텍스트 리스트
            
        Returns:
            Dict: 감성 분석 결과 {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
        """
        if not texts:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 텍스트 결합
        combined_text = " ".join(texts[:20])  # 최대 20개 텍스트만
        
        prompt = f"""
        다음 텍스트들의 감성을 분석해주세요:
        
        {combined_text}
        
        긍정, 중립, 부정의 비율을 JSON 형태로 반환해주세요.
        예시: {{"positive": 0.3, "neutral": 0.5, "negative": 0.2}}
        
        숫자는 0과 1 사이의 값이며, 합계는 1.0이어야 합니다.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # JSON 파싱 시도
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()
            
            result = json.loads(result_text)
            
            # 값 검증
            total = sum(result.values())
            if total > 0:
                for key in result:
                    result[key] = result[key] / total
            
            return result
            
        except Exception as e:
            print(f"감성 분석 오류: {e}")
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
    
    def extract_topics(self, texts: List[str], max_topics: int = 10) -> List[Dict]:
        """
        토픽 추출
        
        Args:
            texts: 분석할 텍스트 리스트
            max_topics: 최대 토픽 수
            
        Returns:
            List[Dict]: 토픽 리스트 [{'topic': 'AI', 'count': 50}, ...]
        """
        if not texts:
            return []
        
        # 텍스트 결합
        combined_text = " ".join(texts[:30])  # 최대 30개 텍스트만
        
        prompt = f"""
        다음 텍스트들에서 주요 토픽을 추출해주세요:
        
        {combined_text}
        
        상위 {max_topics}개 토픽을 빈도와 함께 JSON 형태로 반환해주세요.
        예시: [{{"topic": "인공지능", "count": 50}}, {{"topic": "머신러닝", "count": 30}}]
        
        토픽은 한국어로, 명사 형태로 추출해주세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result_text = response.text.strip()
            
            # JSON 파싱 시도
            if result_text.startswith('```json'):
                result_text = result_text.replace('```json', '').replace('```', '').strip()
            elif result_text.startswith('```'):
                result_text = result_text.replace('```', '').strip()
            
            result = json.loads(result_text)
            
            # 결과 검증 및 정렬
            if isinstance(result, list):
                result = sorted(result, key=lambda x: x.get('count', 0), reverse=True)
                return result[:max_topics]
            
            return []
            
        except Exception as e:
            print(f"토픽 추출 오류: {e}")
            return []
    
    def generate_report(self, trend_data: Dict[str, Any]) -> str:
        """
        자동 리포트 생성
        
        Args:
            trend_data: 트렌드 분석 데이터
            
        Returns:
            str: 자동 생성된 리포트
        """
        # 데이터 요약
        data_summary = f"""
        분석 데이터:
        - 검색 키워드: {', '.join(trend_data.get('keywords', []))}
        - 뉴스 기사 수: {len(trend_data.get('news_data', {}).get('naver_news', []))}
        - 논문 수: {len(trend_data.get('papers_data', []))}
        - 분석 기간: {trend_data.get('period', 'N/A')}
        """
        
        prompt = f"""
        다음 트렌드 분석 데이터를 바탕으로 종합 리포트를 작성해주세요:
        
        {data_summary}
        
        리포트에 다음 내용을 포함해주세요:
        1. 핵심 이슈 Top 3
        2. 급상승 키워드 및 원인 분석
        3. 뉴스 담론 vs 학술 연구 비교
        4. 향후 전망 및 시사점
        
        전문적이면서도 이해하기 쉽게 한국어로 작성해주세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"리포트 생성 오류: {e}")
            return "리포트 생성 중 오류가 발생했습니다."
    
    def analyze_peak_cause(self, peak_date: str, articles: List[Dict]) -> str:
        """
        피크 원인 분석
        
        Args:
            peak_date: 피크 날짜
            articles: 해당 날짜의 기사들
            
        Returns:
            str: 피크 원인 분석 결과
        """
        if not articles:
            return "해당 날짜의 기사가 없습니다."
        
        # 기사 텍스트 결합
        article_texts = []
        for article in articles[:5]:  # 최대 5개 기사만
            title = article.get('title', '')
            description = article.get('description', '')
            article_texts.append(f"제목: {title}\n내용: {description}")
        
        combined_text = "\n\n".join(article_texts)
        
        prompt = f"""
        {peak_date} 날짜에 검색량이 급증한 원인을 다음 기사들을 바탕으로 분석해주세요:
        
        {combined_text}
        
        다음 사항을 포함해서 분석해주세요:
        1. 주요 사건이나 이슈
        2. 급증 원인
        3. 사회적 영향
        
        한국어로 작성해주세요.
        """
        
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            print(f"피크 원인 분석 오류: {e}")
            return "피크 원인 분석 중 오류가 발생했습니다."
