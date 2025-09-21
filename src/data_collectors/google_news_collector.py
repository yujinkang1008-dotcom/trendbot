"""
Google News RSS 수집기 - 실제 본문 추출
"""
import feedparser
import requests
import trafilatura
import pandas as pd
import urllib.parse as up
from typing import List
from src.nlp.clean import normalize_for_topics

def collect_google_news(keywords: List[str], max_articles: int = 5) -> pd.DataFrame:
    """
    Google News RSS에서 뉴스 수집 및 본문 추출
    
    Args:
        keywords: 검색 키워드 리스트
        max_articles: 키워드당 최대 수집 기사 수
        
    Returns:
        pd.DataFrame: 뉴스 데이터 (title, url, published, text_raw, text_clean)
        
    Raises:
        ValueError: 데이터 수집 실패 또는 빈 결과
    """
    all_articles = []
    
    for keyword in keywords:
        try:
            # URL 인코딩
            encoded_keyword = up.quote(keyword)
            rss_url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
            
            # RSS 피드 파싱
            feed = feedparser.parse(rss_url)
            
            if not feed.entries:
                continue
                
            # 각 기사 처리
            for entry in feed.entries[:max_articles]:
                title = entry.get('title', '')
                url = entry.get('link', '')
                published = entry.get('published', '')
                summary = entry.get('summary', '')
                
                # 본문 추출 시도
                text_raw = ""
                try:
                    response = requests.get(url, timeout=10, headers={
                        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                    })
                    if response.status_code == 200:
                        extracted = trafilatura.extract(response.text)
                        if extracted:
                            text_raw = extracted
                except Exception as e:
                    print(f"본문 추출 실패 ({url}): {e}")
                
                # 본문 추출 실패 시 제목+요약 사용
                if not text_raw:
                    text_raw = f"{title} {summary}".strip()
                
                # 빈 텍스트면 건너뛰기
                if not text_raw:
                    continue
                
                # 텍스트 정제
                text_clean = normalize_for_topics(text_raw)
                
                all_articles.append({
                    'title': title,
                    'url': url,
                    'published': published,
                    'text_raw': text_raw,
                    'text_clean': text_clean,
                    'keyword': keyword
                })
                
        except Exception as e:
            print(f"Google News 수집 오류 ({keyword}): {e}")
            continue
    
    if not all_articles:
        raise ValueError("Google News: 수집된 기사가 없습니다")
    
    # DataFrame 생성
    df = pd.DataFrame(all_articles)
    
    # 필수 컬럼 확인
    required_cols = ['title', 'url', 'published', 'text_raw', 'text_clean']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Google News: 필수 컬럼 누락 {missing_cols}")
    
    print(f"✅ Google News 수집 완료: {len(df)}개 기사")
    return df
