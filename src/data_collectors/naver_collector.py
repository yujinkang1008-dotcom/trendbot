"""
Naver 데이터 수집기 (뉴스, 블로그, DataLab)
"""
import os
import urllib.parse as up
import pandas as pd
import datetime as dt
import requests
from typing import List, Dict, Any

class NaverCollector:
    """Naver 데이터 수집 클래스"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
    
    def _headers(self, json=False):
        """API 호출용 헤더 생성"""
        h = {
            "X-Naver-Client-Id": self.client_id,
            "X-Naver-Client-Secret": self.client_secret
        }
        if json:
            h["Content-Type"] = "application/json"
        return h
    
    def search_news(self, query: str, display: int = 20) -> pd.DataFrame:
        """네이버 뉴스 검색 (개선된 버전)"""
        print(f"🔍 NaverCollector.search_news 호출됨")
        print(f"📝 쿼리: {query}")
        print(f"📝 쿼리 타입: {type(query)}")
        print(f"📝 display: {display}")
        print(f"📝 API 키 확인: {self.client_id[:4] if self.client_id else 'None'}...")
        print(f"📝 API 시크릿 확인: {self.client_secret[:4] if self.client_secret else 'None'}...")
        
        if not self.client_id or not self.client_secret:
            print("❌ API 키가 설정되지 않았습니다!")
            return pd.DataFrame()
        
        try:
            # 키워드 전처리 및 확장
            processed_queries = self._process_query(query)
            print(f"📝 처리된 쿼리들: {processed_queries}")
            
            all_results = []
            
            # 각 쿼리로 검색 수행
            for i, processed_query in enumerate(processed_queries):
                try:
                    print(f"📡 검색 쿼리 {i+1}/{len(processed_queries)}: {processed_query}")
                    
                    # 키워드를 더 정확하게 인코딩
                    q = up.quote(processed_query, safe='')
                    url = f"https://openapi.naver.com/v1/search/news.json?query={q}&display={min(display, 100)}&start=1&sort=date"
                    
                    print(f"📝 인코딩된 쿼리: {q}")
                    print(f"📝 전체 URL: {url}")
                    
                    headers = {
                        "X-Naver-Client-Id": self.client_id,
                        "X-Naver-Client-Secret": self.client_secret
                    }
                    
                    response = requests.get(url, headers=headers)
                    print(f"📡 응답 상태: {response.status_code}")
                    
                    if response.status_code != 200:
                        print(f"❌ API 호출 실패: {response.text}")
                        continue
                    
                    js = response.json()
                    items = js.get("items", [])
                    print(f"📊 쿼리 '{processed_query}' 응답 아이템 수: {len(items)}")
                    
                    if items:
                        # 결과에 쿼리 정보 추가
                        for item in items:
                            item['search_query'] = processed_query
                            item['relevance_score'] = self._calculate_relevance(item, query)
                        
                        all_results.extend(items)
                        print(f"✅ 쿼리 '{processed_query}': {len(items)}개 결과 추가")
                    
                except Exception as e:
                    print(f"❌ 쿼리 '{processed_query}' 검색 실패: {e}")
                    continue
            
            if not all_results:
                print("⚠️ 모든 쿼리에서 검색 결과가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame 생성
            df = pd.DataFrame([{
                "title": it.get("title", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&"),
                "url": it.get("link", ""),
                "published": it.get("pubDate", ""),
                "desc": it.get("description", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&"),
                "search_query": it.get("search_query", query),
                "relevance_score": it.get("relevance_score", 0)
            } for it in all_results])
            
            # 중복 제거 (URL 기준)
            df = df.drop_duplicates(subset=['url'], keep='first')
            
            # 관련성 점수 순으로 정렬
            df = df.sort_values('relevance_score', ascending=False)
            
            # 결과 개수 제한
            df = df.head(display)
            
            print(f"📊 최종 DataFrame 생성 완료: {df.shape}")
            print(f"📊 관련성 점수 분포: {df['relevance_score'].describe()}")
            
            return df
            
        except Exception as e:
            print(f"❌ search_news 에러: {e}")
            import traceback
            print(f"❌ 상세 에러: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _process_query(self, query: str) -> List[str]:
        """쿼리 전처리 및 확장"""
        queries = [query]  # 기본 쿼리
        
        # 키워드별 특별 처리
        if "ai" in query.lower() or "인공지능" in query:
            # AI 관련 쿼리 확장
            if "생성형" in query or "generative" in query.lower():
                queries.extend([
                    f'"{query}"',  # 정확한 구문 검색
                    f"{query} 기술",
                    f"{query} 도구",
                    f"{query} 서비스"
                ])
            else:
                queries.extend([
                    f'"{query}"',
                    f"{query} 기술",
                    f"{query} 발전",
                    f"{query} 동향"
                ])
        else:
            # 일반 키워드 처리
            queries.extend([
                f'"{query}"',  # 정확한 구문 검색
                f"{query} 관련",
                f"{query} 동향"
            ])
        
        # 중복 제거 및 순서 유지
        seen = set()
        unique_queries = []
        for q in queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_queries.append(q)
        
        return unique_queries[:5]  # 최대 5개 쿼리로 제한
    
    def _calculate_relevance(self, item: Dict, original_query: str) -> int:
        """관련성 점수 계산"""
        try:
            title = str(item.get("title", "")).lower()
            desc = str(item.get("description", "")).lower()
            query = original_query.lower()
            
            score = 0
            
            # 정확한 매칭 (가장 높은 점수)
            if query in title:
                score += 20
            if query in desc:
                score += 10
            
            # 구문 매칭 (따옴표로 감싸진 경우)
            if f'"{query}"' in title or f'"{query}"' in desc:
                score += 15
            
            # 단어별 매칭
            query_words = query.split()
            for word in query_words:
                if len(word) > 2:  # 2글자 이상인 단어만
                    if word in title:
                        score += 5
                    if word in desc:
                        score += 2
            
            # 키워드별 추가 점수
            if "ai" in query.lower() or "인공지능" in query:
                ai_terms = ["ai", "인공지능", "머신러닝", "딥러닝", "chatgpt", "gpt"]
                for term in ai_terms:
                    if term in title:
                        score += 3
                    if term in desc:
                        score += 1
            
            return score
            
        except Exception as e:
            print(f"❌ 관련성 점수 계산 실패: {e}")
            return 0
    
    def search_blog(self, query: str, display: int = 20) -> pd.DataFrame:
        """네이버 블로그 검색 (개선된 버전)"""
        print(f"🔍 NaverCollector.search_blog 호출됨")
        print(f"📝 쿼리: {query}")
        print(f"📝 쿼리 타입: {type(query)}")
        print(f"📝 display: {display}")
        print(f"📝 API 키 확인: {self.client_id[:4] if self.client_id else 'None'}...")
        print(f"📝 API 시크릿 확인: {self.client_secret[:4] if self.client_secret else 'None'}...")
        
        if not self.client_id or not self.client_secret:
            print("❌ API 키가 설정되지 않았습니다!")
            return pd.DataFrame()
        
        try:
            # 키워드 전처리 및 확장
            processed_queries = self._process_query(query)
            print(f"📝 처리된 쿼리들: {processed_queries}")
            
            all_results = []
            
            # 각 쿼리로 검색 수행
            for i, processed_query in enumerate(processed_queries):
                try:
                    print(f"📡 검색 쿼리 {i+1}/{len(processed_queries)}: {processed_query}")
                    
                    # 키워드를 더 정확하게 인코딩
                    q = up.quote(processed_query, safe='')
                    url = f"https://openapi.naver.com/v1/search/blog.json?query={q}&display={min(display, 100)}&start=1&sort=date"
                    
                    print(f"📝 인코딩된 쿼리: {q}")
                    print(f"📝 전체 URL: {url}")
                    
                    headers = {
                        "X-Naver-Client-Id": self.client_id,
                        "X-Naver-Client-Secret": self.client_secret
                    }
                    
                    response = requests.get(url, headers=headers)
                    print(f"📡 응답 상태: {response.status_code}")
                    
                    if response.status_code != 200:
                        print(f"❌ API 호출 실패: {response.text}")
                        continue
                    
                    js = response.json()
                    items = js.get("items", [])
                    print(f"📊 쿼리 '{processed_query}' 응답 아이템 수: {len(items)}")
                    
                    if items:
                        # 결과에 쿼리 정보 추가
                        for item in items:
                            item['search_query'] = processed_query
                            item['relevance_score'] = self._calculate_relevance(item, query)
                        
                        all_results.extend(items)
                        print(f"✅ 쿼리 '{processed_query}': {len(items)}개 결과 추가")
                    
                except Exception as e:
                    print(f"❌ 쿼리 '{processed_query}' 검색 실패: {e}")
                    continue
            
            if not all_results:
                print("⚠️ 모든 쿼리에서 검색 결과가 없습니다.")
                return pd.DataFrame()
            
            # DataFrame 생성
            df = pd.DataFrame([{
                "title": it.get("title", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&"),
                "url": it.get("link", ""),
                "published": it.get("postdate", ""),
                "desc": it.get("description", "").replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&"),
                "bloggername": it.get("bloggername", ""),
                "search_query": it.get("search_query", query),
                "relevance_score": it.get("relevance_score", 0)
            } for it in all_results])
            
            # 중복 제거 (URL 기준)
            df = df.drop_duplicates(subset=['url'], keep='first')
            
            # 관련성 점수 순으로 정렬
            df = df.sort_values('relevance_score', ascending=False)
            
            # 결과 개수 제한
            df = df.head(display)
            
            print(f"📊 최종 DataFrame 생성 완료: {df.shape}")
            print(f"📊 관련성 점수 분포: {df['relevance_score'].describe()}")
            
            return df
            
        except Exception as e:
            print(f"❌ search_blog 에러: {e}")
            import traceback
            print(f"❌ 상세 에러: {traceback.format_exc()}")
            return pd.DataFrame()

# 환경 변수에서 API 키 로드
NAVER_ID = os.getenv("NAVER_CLIENT_ID")
NAVER_SECRET = os.getenv("NAVER_CLIENT_SECRET")

# STRICT_FETCH 설정 (환경 변수에서 로드, 기본값 False)
STRICT_FETCH = os.getenv("STRICT_FETCH", "False").lower() == "true"

def _headers(json=False):
    """API 호출용 헤더 생성"""
    h = {
        "X-Naver-Client-Id": NAVER_ID,
        "X-Naver-Client-Secret": NAVER_SECRET
    }
    if json:
        h["Content-Type"] = "application/json"
    return h

def _assert(df, cols, name="dataset", min_rows=3):
    """데이터프레임 검증 유틸리티"""
    if df is None or len(df) < min_rows:
        raise ValueError(f"{name}: too few rows (expected >= {min_rows}, got {len(df) if df is not None else 0})")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")

def search_news(query: str, display: int = 20) -> pd.DataFrame:
    """
    Naver 뉴스 검색 (GET + 헤더 2개) + 디버깅
    
    Args:
        query: 검색 키워드
        display: 검색 결과 수
        
    Returns:
        pd.DataFrame: 뉴스 데이터
    """
    print(f"🔍 Naver 뉴스 검색 시작: '{query}', {display}개 결과")
    print(f"📝 입력 파라미터 - query: {query}, display: {display}")
    print(f"📝 query 타입: {type(query)}")
    print(f"📝 API 키 확인: {NAVER_ID[:4] if NAVER_ID else 'None'}...")
    
    try:
        q = up.quote(query)
        url = f"https://openapi.naver.com/v1/search/news.json?query={q}&display={display}&start=1&sort=date"
        
        print(f"📡 API URL: {url}")
        print(f"📡 헤더: {_headers()}")
        
        response = requests.get(url, headers=_headers())
        print(f"📡 응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API 호출 실패: {response.status_code}")
            print(f"❌ 응답 내용: {response.text}")
            response.raise_for_status()
        
        js = response.json()
        print(f"📊 JSON 응답: {js}")
        
        items = js.get("items", [])
        print(f"📊 API 응답: {len(items)}개 아이템")
        
        if not items and STRICT_FETCH:
            raise ValueError(f"Naver 뉴스 검색 실패: {query}")
        
        # 데이터 품질 검증
        valid_items = []
        for i, it in enumerate(items):
            title = it.get("title", "")
            desc = it.get("description", "")
            
            print(f"📝 아이템 {i}: title='{title}', desc='{desc[:50]}...'")
            
            # HTML 태그 제거 확인
            clean_title = title.replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
            clean_desc = desc.replace("<b>", "").replace("</b>", "").replace("&quot;", '"').replace("&amp;", "&")
            
            if i < 3:  # 처음 3개 아이템만 디버깅 출력
                print(f"  📰 아이템 {i+1}:")
                print(f"    제목: {clean_title[:50]}...")
                print(f"    설명: {clean_desc[:50]}...")
            
            valid_items.append({
                "title": clean_title,
                "url": it.get("link", ""),
                "published": it.get("pubDate", ""),
                "desc": clean_desc
            })
        
        df = pd.DataFrame(valid_items)
        
        print(f"✅ Naver 뉴스 데이터 생성: {df.shape}")
        _assert(df, ["title", "url", "published"], "naver_news")
        return df
        
    except Exception as e:
        if STRICT_FETCH:
            raise
        print(f"❌ Naver 뉴스 검색 오류 ({query}): {e}")
        return pd.DataFrame(columns=["title", "url", "published", "desc"])

def search_blog(query: str, display: int = 20) -> pd.DataFrame:
    """
    Naver 블로그 검색 (GET + 헤더 2개)
    
    Args:
        query: 검색 키워드
        display: 검색 결과 수
        
    Returns:
        pd.DataFrame: 블로그 데이터
    """
    print(f"🔍 Naver 블로그 검색 시작: '{query}', {display}개 결과")
    print(f"📝 입력 파라미터 - query: {query}, display: {display}")
    print(f"📝 query 타입: {type(query)}")
    print(f"📝 API 키 확인: {NAVER_ID[:4] if NAVER_ID else 'None'}...")
    
    try:
        q = up.quote(query)
        url = f"https://openapi.naver.com/v1/search/blog.json?query={q}&display={display}&start=1&sort=date"
        
        print(f"📡 API URL: {url}")
        print(f"📡 헤더: {_headers()}")
        
        response = requests.get(url, headers=_headers())
        print(f"📡 응답 상태 코드: {response.status_code}")
        
        if response.status_code != 200:
            print(f"❌ API 호출 실패: {response.status_code}")
            print(f"❌ 응답 내용: {response.text}")
            response.raise_for_status()
        
        js = response.json()
        print(f"📊 JSON 응답: {js}")
        
        items = js.get("items", [])
        print(f"📊 API 응답: {len(items)}개 아이템")
        
        if not items and STRICT_FETCH:
            raise ValueError(f"Naver 블로그 검색 실패: {query}")
        
        print(f"📊 DataFrame 생성 시작...")
        df = pd.DataFrame([{
            "title": it.get("title", ""),
            "url": it.get("link", ""),
            "published": it.get("postdate", ""),
            "desc": it.get("description", ""),
            "bloggername": it.get("bloggername", "")
        } for it in items])
        
        print(f"📊 DataFrame 생성 완료: {df.shape}")
        print(f"📊 DataFrame 컬럼: {df.columns.tolist()}")
        print(f"📊 DataFrame 샘플: {df.head()}")
        
        _assert(df, ["title", "url", "published"], "naver_blog")
        print(f"✅ 검증 통과")
        return df
        
    except Exception as e:
        print(f"❌ search_blog 에러: {e}")
        print(f"❌ 에러 타입: {type(e)}")
        import traceback
        print(f"❌ 상세 에러: {traceback.format_exc()}")
        
        if STRICT_FETCH:
            raise
        print(f"Naver 블로그 검색 오류 ({query}): {e}")
        return pd.DataFrame(columns=["title", "url", "published", "desc", "bloggername"])

def datalab_timeseries_multi(keywords: list, start: str, end: str, timeUnit: str = "date") -> pd.DataFrame:
    """
    Naver DataLab 다중 키워드 시계열 데이터 수집 (POST + Content-Type: application/json + JSON 바디)
    
    Args:
        keywords: 검색 키워드 리스트
        start: 시작 날짜 (YYYY-MM-DD)
        end: 종료 날짜 (YYYY-MM-DD)
        timeUnit: 시간 단위 (date, week, month)
        
    Returns:
        pd.DataFrame: 시계열 데이터 (period + 각 키워드 컬럼)
        
    Raises:
        ValueError: 데이터 수집 실패 또는 빈 결과
    """
    from src.common.trace import snapshot_df, log_shape
    
    try:
        print(f"Naver DataLab 다중 수집 시작: {keywords}, 기간: {start} ~ {end}")
        
        # 키워드 그룹 생성
        keyword_groups = []
        for keyword in keywords:
            keyword_groups.append({
                "groupName": keyword,
                "keywords": [keyword]
            })
        
        payload = {
            "startDate": start,
            "endDate": end,
            "timeUnit": timeUnit,
            "keywordGroups": keyword_groups,
            "device": "",
            "ages": [],
            "gender": ""
        }
        
        response = requests.post(
            "https://openapi.naver.com/v1/datalab/search",
            headers=_headers(json=True),
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results:
            raise ValueError(f"Naver DataLab: 빈 결과 {keywords}")
        
        # 결과를 wide 형태로 머지
        merged_df = None
        for result in results:
            keyword_name = result["title"]
            keyword_data = pd.DataFrame(result["data"])
            
            if keyword_data.empty:
                print(f"⚠️ {keyword_name}: 빈 데이터")
                continue
                
            keyword_data = keyword_data.rename(columns={"ratio": keyword_name})
            keyword_data = keyword_data.set_index("period")
            
            if merged_df is None:
                merged_df = keyword_data
            else:
                merged_df = merged_df.join(keyword_data, how='outer')
        
        if merged_df is None or merged_df.empty:
            raise ValueError(f"Naver DataLab: 최종 빈 결과 {keywords}")
        
        # 인덱스를 다시 컬럼으로
        merged_df = merged_df.reset_index()
        
        # 키워드 컬럼 확인
        missing_keywords = [kw for kw in keywords if kw not in merged_df.columns]
        if missing_keywords:
            raise ValueError(f"Naver DataLab: 누락된 키워드 {missing_keywords}")
        
        print(f"✅ Naver DataLab 다중 수집 완료: {merged_df.shape}")
        snapshot_df(merged_df, "trends_naver")
        return merged_df
        
    except Exception as e:
        print(f"❌ Naver DataLab 다중 수집 실패: {e}")
        if STRICT_FETCH:
            raise
        return pd.DataFrame()

def datalab_timeseries(keyword: str, start: str, end: str, timeUnit: str = "date") -> pd.DataFrame:
    """
    Naver DataLab 시계열 데이터 수집 (POST + Content-Type: application/json + JSON 바디)
    
    Args:
        keyword: 검색 키워드
        start: 시작 날짜 (YYYY-MM-DD)
        end: 종료 날짜 (YYYY-MM-DD)
        timeUnit: 시간 단위 (date, week, month)
        
    Returns:
        pd.DataFrame: 시계열 데이터
    """
    try:
        payload = {
            "startDate": start,
            "endDate": end,
            "timeUnit": timeUnit,
            "keywordGroups": [{"groupName": keyword, "keywords": [keyword]}],
            "device": "",
            "ages": [],
            "gender": ""
        }
        
        response = requests.post(
            "https://openapi.naver.com/v1/datalab/search",
            headers=_headers(json=True),
            json=payload
        )
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        if not results and STRICT_FETCH:
            raise ValueError(f"Naver DataLab 검색 실패: {keyword}")
        
        if results:
            df = pd.DataFrame(results[0]["data"])  # period, ratio
            _assert(df, ["period", "ratio"], "naver_datalab")
            return df
        else:
            return pd.DataFrame(columns=["period", "ratio"])
            
    except Exception as e:
        if STRICT_FETCH:
            raise
        print(f"Naver DataLab 검색 오류 ({keyword}): {e}")
        return pd.DataFrame(columns=["period", "ratio"])

class NaverCollector:
    """Naver 데이터 수집 클래스 (기존 호환성 유지)"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
    
    def search_news(self, keywords: List[str], display: int = 100) -> List[Dict]:
        """뉴스 검색 (기존 호환성)"""
        news_data = []
        
        for keyword in keywords:
            try:
                df = search_news(keyword, display)
                
                for _, row in df.iterrows():
                    news_data.append({
                        'keyword': keyword,
                        'title': row.get('title', ''),
                        'description': row.get('desc', ''),
                        'link': row.get('url', ''),
                        'pub_date': row.get('published', ''),
                        'source': 'naver_news'
                    })
                    
            except Exception as e:
                print(f"Naver 뉴스 검색 오류 ({keyword}): {e}")
                continue
        
        return news_data
    
    def search_blog(self, keywords: List[str], display: int = 100) -> List[Dict]:
        """블로그 검색 (기존 호환성)"""
        blog_data = []
        
        for keyword in keywords:
            try:
                df = search_blog(keyword, display)
                
                for _, row in df.iterrows():
                    blog_data.append({
                        'keyword': keyword,
                        'title': row.get('title', ''),
                        'description': row.get('desc', ''),
                        'link': row.get('url', ''),
                        'bloggername': row.get('bloggername', ''),
                        'postdate': row.get('published', ''),
                        'source': 'naver_blog'
                    })
                    
            except Exception as e:
                print(f"Naver 블로그 검색 오류 ({keyword}): {e}")
                continue
        
        return blog_data
    
    def get_google_news_rss(self, keywords: List[str]) -> List[Dict]:
        """Google News RSS (기존 호환성)"""
        # 기존 구현 유지
        import feedparser
        
        news_data = []
        
        for keyword in keywords:
            try:
                rss_url = f"https://news.google.com/rss/search?q={keyword}&hl=ko&gl=KR&ceid=KR:ko"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries:
                    news_data.append({
                        'keyword': keyword,
                        'title': entry.get('title', ''),
                        'description': entry.get('summary', ''),
                        'link': entry.get('link', ''),
                        'pub_date': entry.get('published', ''),
                        'source': 'google_news_rss'
                    })
                    
            except Exception as e:
                print(f"Google News RSS 수집 오류 ({keyword}): {e}")
                continue
        
        return news_data
    
    def collect_all_news(self, keywords: List[str]) -> Dict[str, List[Dict]]:
        """모든 뉴스 소스에서 데이터 수집 (기존 호환성)"""
        result = {
            'naver_news': self.search_news(keywords),
            'naver_blog': self.search_blog(keywords),
            'google_news': self.get_google_news_rss(keywords)
        }
        
        return result