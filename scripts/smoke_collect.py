"""
스모크 테스트 스크립트 - 데이터 수집기 검증
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("trendbot.env")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_naver_search():
    """Naver 검색 테스트"""
    print("=== Naver 검색 테스트 ===")
    try:
        from src.data_collectors.naver_collector import search_news
        
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            print("❌ Naver API 키 누락")
            return False
        
        # 뉴스 검색 테스트
        df = search_news("ai", display=3)
        print(f"✅ Naver 뉴스: {df.shape} - 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 확인
        required_cols = ['title', 'url', 'published']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ 누락된 컬럼: {missing}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Naver 검색 오류: {e}")
        return False

def test_naver_datalab_multi():
    """Naver DataLab 다중 키워드 테스트"""
    print("=== Naver DataLab 다중 키워드 테스트 ===")
    try:
        from src.data_collectors.naver_collector import datalab_timeseries_multi
        
        # 날짜 설정 (최근 30일)
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
        
        keywords = ["ai", "영어공부", "에듀테크"]
        df = datalab_timeseries_multi(keywords, start_date, end_date)
        
        print(f"✅ Naver DataLab: {df.shape} - 컬럼: {list(df.columns)}")
        
        # 키워드 컬럼 확인
        missing_keywords = [kw for kw in keywords if kw not in df.columns]
        if missing_keywords:
            print(f"❌ 누락된 키워드: {missing_keywords}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Naver DataLab 오류: {e}")
        return False

def test_google_news():
    """Google News RSS 테스트"""
    print("=== Google News RSS 테스트 ===")
    try:
        from src.data_collectors.google_news_collector import collect_google_news
        
        keywords = ["ai", "영어공부", "에듀테크"]
        df = collect_google_news(keywords, max_articles=5)
        
        print(f"✅ Google News: {df.shape} - 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 확인
        required_cols = ['title', 'url', 'published', 'text_raw', 'text_clean']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ 누락된 컬럼: {missing}")
            return False
        
        # text_clean 비율 확인 (≥ 0.8)
        total_records = len(df)
        non_empty_clean = df['text_clean'].dropna()
        clean_ratio = len(non_empty_clean) / total_records if total_records > 0 else 0
        
        if clean_ratio < 0.8:
            print(f"❌ text_clean 비율 부족: {clean_ratio:.2f} < 0.8")
            return False
        
        print(f"✅ text_clean 비율: {clean_ratio:.2f} (≥ 0.8)")
        return True
        
    except Exception as e:
        print(f"❌ Google News 오류: {e}")
        return False

def test_google_trends_multi():
    """Google Trends 다중 키워드 테스트"""
    print("=== Google Trends 다중 키워드 테스트 ===")
    try:
        from src.data_collectors.google_trends_collector import GoogleTrendsCollector
        
        collector = GoogleTrendsCollector()
        keywords = ["ai", "영어공부", "에듀테크"]
        
        df = collector.google_trends_multi(keywords, "today 12-m")
        
        print(f"✅ Google Trends: {df.shape} - 컬럼: {list(df.columns)}")
        
        # 키워드 컬럼 확인
        missing_keywords = [kw for kw in keywords if kw not in df.columns]
        if missing_keywords:
            print(f"❌ 누락된 키워드: {missing_keywords}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Google Trends 오류: {e}")
        return False

def test_arxiv():
    """arXiv 테스트"""
    print("=== arXiv 테스트 ===")
    try:
        from src.data_collectors.arxiv_collector import ArxivCollector
        
        collector = ArxivCollector()
        keywords = ["generative AI"]
        
        df = collector.collect_papers(keywords, max_results=5)
        
        print(f"✅ arXiv: {df.shape} - 컬럼: {list(df.columns)}")
        
        # 필수 컬럼 확인
        required_cols = ['title', 'url', 'published', 'summary', 'text_clean']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            print(f"❌ 누락된 컬럼: {missing}")
            return False
        
        # text_clean 비율 확인 (≥ 0.8)
        total_records = len(df)
        non_empty_clean = df['text_clean'].dropna()
        clean_ratio = len(non_empty_clean) / total_records if total_records > 0 else 0
        
        if clean_ratio < 0.8:
            print(f"❌ text_clean 비율 부족: {clean_ratio:.2f} < 0.8")
            return False
        
        print(f"✅ text_clean 비율: {clean_ratio:.2f} (≥ 0.8)")
        return True
        
    except Exception as e:
        print(f"❌ arXiv 오류: {e}")
        return False

def test_topic_quality():
    """토픽 품질 테스트"""
    print("=== 토픽 품질 테스트 ===")
    try:
        from src.ai.topic_extractor import TopicExtractor
        
        # 테스트 코퍼스 생성
        test_corpus = [
            "인공지능 기술이 발전하고 있습니다",
            "머신러닝 알고리즘을 연구합니다",
            "딥러닝 모델을 개발합니다",
            "자연어 처리 기술을 적용합니다",
            "컴퓨터 비전 시스템을 구축합니다"
        ]
        
        extractor = TopicExtractor()
        topics = extractor.top_topics(test_corpus, k=5)
        
        print(f"✅ 토픽 추출: {topics}")
        
        # 금지 토큰 검증
        forbidden_tokens = ['nbsp', 'font', 'href', 'https', 'com', 'google', 'news']
        for topic in topics:
            if any(forbidden in topic.lower() for forbidden in forbidden_tokens):
                print(f"❌ 금지 토큰 포함: '{topic}'")
                return False
        
        print("✅ 토픽 품질 검증 통과")
        return True
        
    except Exception as e:
        print(f"❌ 토픽 품질 테스트 오류: {e}")
        return False

def main():
    """메인 테스트 실행"""
    print("🚀 데이터 수집기 스모크 테스트 시작")
    print("=" * 50)
    
    tests = [
        ("Naver 검색", test_naver_search),
        ("Naver DataLab 다중", test_naver_datalab_multi),
        ("Google News RSS", test_google_news),
        ("Google Trends 다중", test_google_trends_multi),
        ("arXiv", test_arxiv),
        ("토픽 품질", test_topic_quality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name} 테스트 중...")
        try:
            success = test_func()
            results.append((test_name, success))
            if success:
                print(f"✅ {test_name} 테스트 통과")
            else:
                print(f"❌ {test_name} 테스트 실패")
        except Exception as e:
            print(f"❌ {test_name} 테스트 오류: {e}")
            results.append((test_name, False))
    
    print("\n" + "=" * 50)
    print("📊 테스트 결과 요약")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 통과" if success else "❌ 실패"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n총 {len(results)}개 테스트 중 {passed}개 통과")
    
    if passed == len(results):
        print("🎉 모든 테스트 통과!")
        return True
    else:
        print("⚠️ 일부 테스트 실패")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
