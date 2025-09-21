"""
Google Trends 데이터 수집기
"""
import pandas as pd
from pytrends.request import TrendReq
from typing import List, Dict, Any
import time
from datetime import datetime, timedelta

class GoogleTrendsCollector:
    """Google Trends 데이터 수집 클래스"""
    
    def __init__(self):
        # Google Trends API 설정 (간단하게)
        import urllib3
        
        # SSL 경고 비활성화
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        # 기본 설정으로 초기화
        self.pytrends = TrendReq(hl='ko', tz=360)
        
    def google_trends_multi(self, keywords: List[str], timeframe: str = "today 12-m") -> pd.DataFrame:
        """
        여러 키워드 동시 수집 (wide 형태)
        실패 시 키워드 단위로 순차 재시도
        
        Args:
            keywords: 키워드 리스트
            timeframe: 기간 (예: "today 12-m", "today 3-m")
            
        Returns:
            pd.DataFrame: wide 형태 데이터 (period + 각 키워드 컬럼)
            
        Raises:
            ValueError: 데이터 수집 실패 또는 빈 결과
        """
        from src.common.trace import snapshot_df, log_shape
        
        try:
            print(f"Google Trends 다중 수집 시작: {keywords}, 기간: {timeframe}")
            
            # 1차 시도: 모든 키워드 동시 수집
            try:
                self.pytrends.build_payload(kw_list=keywords, timeframe=timeframe, geo="")
                time.sleep(1)  # API 제한 고려
                
                interest_df = self.pytrends.interest_over_time()
                
                if not interest_df.empty:
                    # isPartial 컬럼 제거
                    if 'isPartial' in interest_df.columns:
                        interest_df = interest_df.drop(columns=['isPartial'])
                    
                    # 인덱스(date)를 period 컬럼으로 변환
                    interest_df = interest_df.reset_index()
                    interest_df = interest_df.rename(columns={'date': 'period'})
                    
                    # 키워드 컬럼 확인
                    missing_keywords = [kw for kw in keywords if kw not in interest_df.columns]
                    if not missing_keywords:
                        print(f"✅ Google Trends 다중 수집 완료: {interest_df.shape}")
                        snapshot_df(interest_df, "trends_google")
                        return interest_df
                    
                    print(f"⚠️ 일부 키워드 누락: {missing_keywords}, 순차 재시도")
                    
            except Exception as e:
                print(f"⚠️ 동시 수집 실패: {e}, 순차 재시도")
            
            # 2차 시도: 키워드 단위로 순차 수집
            print("🔄 키워드 단위 순차 수집 시작")
            all_dfs = []
            
            for keyword in keywords:
                try:
                    print(f"  - {keyword} 수집 중...")
                    self.pytrends.build_payload(kw_list=[keyword], timeframe=timeframe, geo="")
                    time.sleep(2)  # API 제한 고려
                    
                    keyword_df = self.pytrends.interest_over_time()
                    
                    if not keyword_df.empty:
                        # isPartial 컬럼 제거
                        if 'isPartial' in keyword_df.columns:
                            keyword_df = keyword_df.drop(columns=['isPartial'])
                        
                        # 인덱스(date)를 period 컬럼으로 변환
                        keyword_df = keyword_df.reset_index()
                        keyword_df = keyword_df.rename(columns={'date': 'period'})
                        
                        # 키워드 컬럼명 정리
                        if keyword in keyword_df.columns:
                            all_dfs.append(keyword_df[['period', keyword]])
                            print(f"    ✅ {keyword} 수집 성공")
                        else:
                            print(f"    ❌ {keyword} 컬럼 누락")
                    else:
                        print(f"    ❌ {keyword} 빈 데이터")
                        
                except Exception as e:
                    print(f"    ❌ {keyword} 수집 실패: {e}")
                    continue
            
            if not all_dfs:
                raise ValueError(f"Google Trends: 모든 키워드 수집 실패 {keywords}")
            
            # 모든 DataFrame을 period 기준으로 outer join
            merged_df = all_dfs[0]
            for df in all_dfs[1:]:
                merged_df = merged_df.merge(df, on='period', how='outer')
            
            # 최종 검증
            missing_keywords = [kw for kw in keywords if kw not in merged_df.columns]
            if missing_keywords:
                raise ValueError(f"Google Trends: 최종 누락 키워드 {missing_keywords}")
            
            if merged_df.empty:
                raise ValueError(f"Google Trends: 최종 빈 결과 {keywords}")
            
            print(f"✅ Google Trends 순차 수집 완료: {merged_df.shape}")
            snapshot_df(merged_df, "trends_google")
            return merged_df
            
        except Exception as e:
            print(f"❌ Google Trends 다중 수집 실패: {e}")
            raise

    def get_trends_data(self, keywords: List[str], period_days: int = 365) -> Dict[str, Any]:
        """
        Google Trends 데이터 수집
        
        Args:
            keywords: 검색 키워드 리스트
            period_days: 수집 기간 (일)
            
        Returns:
            Dict: 검색량 데이터, 연관 검색어, 인기 검색어 등
        """
        try:
            # 기간 설정
            if period_days <= 30:
                timeframe = 'today 1-m'
            elif period_days <= 90:
                timeframe = 'today 3-m'
            elif period_days <= 180:
                timeframe = 'today 6-m'
            else:
                timeframe = 'today 12-m'
            
            print(f"Google Trends 데이터 수집 시작: {keywords}, 기간: {timeframe}")
            
            # 검색량 데이터 수집
            self.pytrends.build_payload(
                keywords, 
                cat=0, 
                timeframe=timeframe,
                geo='KR',  # 한국
                gprop=''
            )
            
            # 잠시 대기 (API 호출 제한 고려)
            import time
            time.sleep(1)
            
            # 검색량 추이 데이터
            try:
                interest_over_time = self.pytrends.interest_over_time()
                print(f"✅ 검색량 데이터 수집 성공: {interest_over_time.shape}")
            except Exception as e:
                print(f"❌ 검색량 데이터 수집 실패: {e}")
                interest_over_time = pd.DataFrame()
            
            # 연관 검색어 수집
            related_queries = {}
            for keyword in keywords:
                try:
                    # 각 키워드별로 개별적으로 연관 검색어 수집
                    print(f"연관 검색어 수집 중: {keyword}")
                    self.pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='KR', gprop='')
                    time.sleep(1)  # API 호출 제한 고려
                    keyword_related = self.pytrends.related_queries()
                    if keyword in keyword_related and keyword_related[keyword]['top'] is not None:
                        # 입력 키워드 제외
                        related_df = keyword_related[keyword]['top']
                        filtered_df = related_df[~related_df['query'].str.contains(keyword, case=False, na=False)]
                        related_queries[keyword] = {'top': filtered_df, 'rising': keyword_related[keyword]['rising']}
                        print(f"✅ {keyword} 연관 검색어 수집 성공 ({len(filtered_df)}개)")
                    else:
                        related_queries[keyword] = {'top': None, 'rising': None}
                        print(f"⚠️ {keyword} 연관 검색어 없음")
                except Exception as e:
                    print(f"❌ 키워드 {keyword} 연관 검색어 수집 실패: {e}")
                    related_queries[keyword] = {'top': None, 'rising': None}
            
            # 인기 검색어 (지역별)
            try:
                trending_searches = self.pytrends.trending_searches(pn='south_korea')
            except:
                trending_searches = pd.DataFrame()
            
            # 인구통계학적 데이터
            try:
                interest_by_region = self.pytrends.interest_by_region()
            except:
                interest_by_region = pd.DataFrame()
            
            result = {
                'search_volume': interest_over_time,
                'related_queries': related_queries,
                'trending_searches': trending_searches,
                'interest_by_region': interest_by_region,
                'keywords': keywords,
                'period': {
                    'start': '12개월 전',
                    'end': '현재',
                    'timeframe': timeframe
                }
            }
            
            return result
            
        except Exception as e:
            print(f"Google Trends 데이터 수집 오류: {e}")
            # 빈 데이터 반환
            return {
                'search_volume': pd.DataFrame(),
                'related_queries': {},
                'trending_searches': pd.DataFrame(),
                'interest_by_region': pd.DataFrame(),
                'keywords': keywords,
                'error': str(e)
            }
    
    def get_related_keywords(self, keywords: List[str], limit: int = 10) -> List[Dict]:
        """
        연관 검색어 추출
        
        Args:
            keywords: 검색 키워드 리스트
            limit: 반환할 연관 검색어 수
            
        Returns:
            List[Dict]: 연관 검색어 리스트
        """
        try:
            self.pytrends.build_payload(keywords, cat=0, timeframe='today 12-m', geo='KR')
            related_queries = self.pytrends.related_queries()
            
            related_keywords = []
            for keyword in keywords:
                if keyword in related_queries and related_queries[keyword]['top'] is not None:
                    top_related = related_queries[keyword]['top'].head(limit)
                    for _, row in top_related.iterrows():
                        related_keywords.append({
                            'keyword': keyword,
                            'related': row['query'],
                            'value': row['value']
                        })
            
            return related_keywords
            
        except Exception as e:
            print(f"연관 검색어 수집 오류: {e}")
            return []
    
