"""
arXiv 학술 논문 데이터 수집기
"""
import requests
import feedparser
from typing import List, Dict, Any
from datetime import datetime, timedelta
import time
import pandas as pd
from src.nlp.clean import normalize_for_topics

class ArxivCollector:
    """arXiv 학술 논문 데이터 수집 클래스"""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        
    def collect_papers(self, keywords: List[str], max_results: int = 500, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        arXiv에서 논문 검색 및 텍스트 정제 + 디버깅
        
        Args:
            keywords: 검색 키워드 리스트
            max_results: 최대 검색 결과 수
            start_date: 시작 날짜 (YYYY-MM-DD, 선택사항)
            end_date: 종료 날짜 (YYYY-MM-DD, 선택사항)
            
        Returns:
            pd.DataFrame: 논문 데이터 (title, url, published, summary, text_clean)
            
        Raises:
            ValueError: 데이터 수집 실패 또는 빈 결과
        """
        print(f"🔍 arXiv 논문 수집 시작: {keywords}, 최대 {max_results}개")
        
        papers_data = []
        
        for keyword in keywords:
            try:
                print(f"📚 키워드 '{keyword}' 검색 중...")
                
                # arXiv API 파라미터 (한국어 키워드 처리)
                # 한국어 키워드인 경우 영어 번역 시도
                search_query = keyword
                if any('\uac00' <= char <= '\ud7a3' for char in keyword):  # 한글 포함 여부
                    # 한국어 키워드를 영어로 매핑
                    korean_to_english = {
                        '인공지능': 'artificial intelligence',
                        '머신러닝': 'machine learning',
                        '딥러닝': 'deep learning',
                        '자연어처리': 'natural language processing',
                        '컴퓨터비전': 'computer vision',
                        '데이터사이언스': 'data science',
                        '빅데이터': 'big data',
                        '블록체인': 'blockchain',
                        '사이버보안': 'cybersecurity',
                        '클라우드': 'cloud computing'
                    }
                    search_query = korean_to_english.get(keyword, 'artificial intelligence')
                    print(f"  🔄 한국어 키워드 '{keyword}' -> 영어 '{search_query}'로 변환")
                
                params = {
                    'search_query': f'all:{search_query}',
                    'start': 0,
                    'max_results': max_results,
                    'sortBy': 'submittedDate',
                    'sortOrder': 'descending'
                }
                
                print(f"📡 API 파라미터: {params}")
                
                # API 호출
                response = requests.get(self.base_url, params=params)
                response.raise_for_status()
                
                # XML 파싱
                feed = feedparser.parse(response.content)
                
                print(f"📊 파싱된 엔트리 수: {len(feed.entries)}")
                
                if not feed.entries:
                    print(f"⚠️ '{keyword}'에 대한 결과 없음")
                    continue
                
                keyword_papers = 0
                for i, entry in enumerate(feed.entries):
                    title = entry.get('title', '').strip()
                    summary = entry.get('summary', '').strip()
                    published = entry.get('published', '')
                    url = entry.get('link', '')
                    
                    # 기간 필터링 (published 기준)
                    if start_date or end_date:
                        try:
                            pub_date = datetime.strptime(published[:10], '%Y-%m-%d')
                            if start_date and pub_date < datetime.strptime(start_date, '%Y-%m-%d'):
                                continue
                            if end_date and pub_date > datetime.strptime(end_date, '%Y-%m-%d'):
                                continue
                        except:
                            pass  # 날짜 파싱 실패 시 건너뛰기
                    
                    # title + summary 결합
                    text_raw = f"{title} {summary}".strip()
                    
                    if not text_raw:
                        continue
                    
                    # 텍스트 정제
                    text_clean = normalize_for_topics(text_raw)
                    
                    # 디버깅: 처음 3개 논문만 출력
                    if keyword_papers < 3:
                        print(f"  📄 논문 {keyword_papers+1}:")
                        print(f"    제목: {title[:50]}...")
                        print(f"    원본 텍스트 길이: {len(text_raw)}")
                        print(f"    정제된 텍스트 길이: {len(text_clean)}")
                        print(f"    정제된 텍스트 샘플: {text_clean[:100]}...")
                    
                    papers_data.append({
                        'title': title,
                        'url': url,
                        'published': published,
                        'summary': summary,
                        'text_clean': text_clean,
                        'keyword': keyword
                    })
                    
                    keyword_papers += 1
                
                print(f"✅ '{keyword}': {keyword_papers}개 논문 수집")
                
                # API 호출 제한 고려
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ arXiv 검색 오류 ({keyword}): {e}")
                continue
        
        if not papers_data:
            print("❌ arXiv: 수집된 논문이 없습니다")
            raise ValueError("arXiv: 수집된 논문이 없습니다")
        
        # DataFrame 생성
        df = pd.DataFrame(papers_data)
        
        print(f"📊 DataFrame 생성: {df.shape}")
        
        # 필수 컬럼 확인
        required_cols = ['title', 'url', 'published', 'summary', 'text_clean']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"❌ 필수 컬럼 누락: {missing_cols}")
            raise ValueError(f"arXiv: 필수 컬럼 누락 {missing_cols}")
        
        # text_clean 품질 검증
        empty_clean = df['text_clean'].isna() | (df['text_clean'] == '')
        clean_ratio = (len(df) - empty_clean.sum()) / len(df) if len(df) > 0 else 0
        
        print(f"📈 text_clean 품질: {clean_ratio:.2%} ({len(df) - empty_clean.sum()}/{len(df)})")
        
        if clean_ratio < 0.5:
            print("⚠️ text_clean 품질이 낮습니다. 텍스트 정제 로직을 확인하세요.")
        
        print(f"✅ arXiv 수집 완료: {len(df)}개 논문")
        return df

    # 기존 호환성을 위한 메서드
    def search_papers(self, keywords: List[str], max_results: int = 100) -> List[Dict]:
        """기존 호환성용 메서드"""
        try:
            df = self.collect_papers(keywords, max_results)
            return df.to_dict('records')
        except Exception as e:
            print(f"arXiv 검색 오류: {e}")
            return []