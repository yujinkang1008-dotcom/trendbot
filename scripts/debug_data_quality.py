"""
데이터 품질 디버깅 스크립트
각 데이터 소스별로 품질을 검증하고 문제점을 찾습니다.
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("trendbot.env")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def debug_naver_news_quality():
    """Naver 뉴스 데이터 품질 디버깅"""
    print("=" * 60)
    print("🔍 NAVER 뉴스 데이터 품질 디버깅")
    print("=" * 60)
    
    try:
        from src.data_collectors.naver_collector import search_news
        from src.nlp.clean import normalize_for_topics
        
        # 테스트 키워드
        test_keywords = ["인공지능", "AI", "머신러닝"]
        
        for keyword in test_keywords:
            print(f"\n📰 키워드: '{keyword}'")
            
            # 뉴스 검색
            df = search_news(keyword, display=5)
            
            if df.empty:
                print(f"❌ '{keyword}'에 대한 뉴스 없음")
                continue
            
            print(f"✅ {len(df)}개 뉴스 수집")
            
            # 각 뉴스의 품질 검증
            for i, (_, row) in enumerate(df.iterrows()):
                title = row.get('title', '')
                desc = row.get('desc', '')
                
                print(f"\n  📄 뉴스 {i+1}:")
                print(f"    원본 제목: {title}")
                print(f"    원본 설명: {desc[:100]}...")
                
                # 텍스트 정제 테스트
                combined_text = f"{title} {desc}"
                clean_text = normalize_for_topics(combined_text)
                
                print(f"    정제된 텍스트: {clean_text}")
                print(f"    정제 비율: {len(clean_text)/len(combined_text):.2%}")
                
                # 문제 토큰 검사
                problematic_tokens = ['nbsp', 'font', 'href', 'https', 'ai', 'rss', 'xml']
                found_problems = [token for token in problematic_tokens if token in clean_text.lower()]
                
                if found_problems:
                    print(f"    ⚠️ 문제 토큰 발견: {found_problems}")
                else:
                    print(f"    ✅ 문제 토큰 없음")
        
    except Exception as e:
        print(f"❌ Naver 뉴스 디버깅 오류: {e}")

def debug_arxiv_quality():
    """arXiv 논문 데이터 품질 디버깅"""
    print("=" * 60)
    print("🔍 ARXIV 논문 데이터 품질 디버깅")
    print("=" * 60)
    
    try:
        from src.data_collectors.arxiv_collector import ArxivCollector
        from src.nlp.clean import normalize_for_topics
        
        collector = ArxivCollector()
        test_keywords = ["machine learning", "artificial intelligence"]
        
        for keyword in test_keywords:
            print(f"\n📚 키워드: '{keyword}'")
            
            # 논문 검색
            df = collector.collect_papers([keyword], max_results=3)
            
            if df.empty:
                print(f"❌ '{keyword}'에 대한 논문 없음")
                continue
            
            print(f"✅ {len(df)}개 논문 수집")
            
            # 각 논문의 품질 검증
            for i, (_, row) in enumerate(df.iterrows()):
                title = row.get('title', '')
                summary = row.get('summary', '')
                text_clean = row.get('text_clean', '')
                
                print(f"\n  📄 논문 {i+1}:")
                print(f"    제목: {title}")
                print(f"    요약: {summary[:100]}...")
                print(f"    정제된 텍스트: {text_clean[:100]}...")
                print(f"    정제 비율: {len(text_clean)/len(title + ' ' + summary):.2%}")
                
                # 문제 토큰 검사
                problematic_tokens = ['nbsp', 'font', 'href', 'https', 'ai', 'rss', 'xml']
                found_problems = [token for token in problematic_tokens if token in text_clean.lower()]
                
                if found_problems:
                    print(f"    ⚠️ 문제 토큰 발견: {found_problems}")
                else:
                    print(f"    ✅ 문제 토큰 없음")
        
    except Exception as e:
        print(f"❌ arXiv 디버깅 오류: {e}")

def debug_topic_extraction():
    """토픽 추출 과정 디버깅"""
    print("=" * 60)
    print("🔍 토픽 추출 과정 디버깅")
    print("=" * 60)
    
    try:
        from src.ai.topic_extractor import TopicExtractor
        from src.nlp.clean import normalize_for_topics
        
        # 테스트 코퍼스 생성
        test_corpus = [
            "인공지능 기술이 발전하고 있습니다",
            "머신러닝 알고리즘을 연구합니다",
            "딥러닝 모델을 개발합니다",
            "자연어 처리 기술을 적용합니다",
            "컴퓨터 비전 시스템을 구축합니다",
            "AI 기술의 발전이 가속화되고 있습니다",
            "nbsp font href https 문제가 있습니다",
            "RSS XML JSON API 관련 내용입니다"
        ]
        
        print("📊 테스트 코퍼스:")
        for i, text in enumerate(test_corpus):
            print(f"  {i+1}: {text}")
        
        # 텍스트 정제 테스트
        print("\n🧹 텍스트 정제 결과:")
        cleaned_corpus = []
        for i, text in enumerate(test_corpus):
            clean_text = normalize_for_topics(text)
            cleaned_corpus.append(clean_text)
            print(f"  {i+1}: {text[:30]}... -> {clean_text}")
        
        # 토픽 추출 테스트
        print("\n🎯 토픽 추출 테스트:")
        extractor = TopicExtractor()
        
        # 원본 코퍼스로 테스트
        print("\n📋 원본 코퍼스 토픽 추출:")
        topics_original = extractor.top_topics(test_corpus, k=5)
        print(f"결과: {topics_original}")
        
        # 정제된 코퍼스로 테스트
        print("\n📋 정제된 코퍼스 토픽 추출:")
        topics_cleaned = extractor.top_topics(cleaned_corpus, k=5)
        print(f"결과: {topics_cleaned}")
        
        # 문제 토큰 확인
        print("\n🚫 문제 토큰 검사:")
        problematic_tokens = ['nbsp', 'font', 'href', 'https', 'ai', 'rss', 'xml', 'json', 'api']
        
        for topic_list, name in [(topics_original, "원본"), (topics_cleaned, "정제")]:
            found_problems = []
            for topic in topic_list:
                for problem_token in problematic_tokens:
                    if problem_token in topic.lower():
                        found_problems.append(f"{topic}({problem_token})")
            
            if found_problems:
                print(f"  {name}: ⚠️ 문제 토큰 발견 - {found_problems}")
            else:
                print(f"  {name}: ✅ 문제 토큰 없음")
        
    except Exception as e:
        print(f"❌ 토픽 추출 디버깅 오류: {e}")

def debug_data_collection_pipeline():
    """전체 데이터 수집 파이프라인 디버깅"""
    print("=" * 60)
    print("🔍 전체 데이터 수집 파이프라인 디버깅")
    print("=" * 60)
    
    try:
        from src.data_collectors.naver_collector import NaverCollector
        from src.data_collectors.arxiv_collector import ArxivCollector
        from src.ai.topic_extractor import TopicExtractor
        
        # 설정 검증
        client_id = os.getenv("NAVER_CLIENT_ID")
        client_secret = os.getenv("NAVER_CLIENT_SECRET")
        
        if not client_id or not client_secret:
            print("❌ Naver API 키 누락")
            return
        
        print("✅ 환경 변수 설정 확인")
        
        # 데이터 수집기 초기화
        naver_collector = NaverCollector(client_id, client_secret)
        arxiv_collector = ArxivCollector()
        
        test_keywords = ["인공지능"]
        
        # 뉴스 데이터 수집
        print(f"\n📰 뉴스 데이터 수집: {test_keywords}")
        news_data = naver_collector.collect_all_news(test_keywords)
        
        total_news = sum(len(articles) for articles in news_data.values())
        print(f"✅ 총 {total_news}개 뉴스 수집")
        
        # 논문 데이터 수집
        print(f"\n📚 논문 데이터 수집: {test_keywords}")
        papers_data = arxiv_collector.search_papers(test_keywords, max_results=5)
        print(f"✅ {len(papers_data)}개 논문 수집")
        
        # 텍스트 추출 및 품질 검증
        print(f"\n🧹 텍스트 품질 검증:")
        
        all_texts = []
        
        # 뉴스 텍스트
        for source, articles in news_data.items():
            for article in articles:
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    all_texts.append(text)
        
        # 논문 텍스트
        for paper in papers_data:
            text = f"{paper.get('title', '')} {paper.get('summary', '')}"
            if text.strip():
                all_texts.append(text)
        
        print(f"📊 총 {len(all_texts)}개 텍스트 수집")
        
        # 토픽 추출
        if all_texts:
            print(f"\n🎯 토픽 추출:")
            extractor = TopicExtractor()
            topics = extractor.top_topics(all_texts, k=10)
            print(f"✅ {len(topics)}개 토픽 추출")
            print(f"🎯 추출된 토픽: {topics}")
            
            # 문제 토큰 검사
            problematic_tokens = ['nbsp', 'font', 'href', 'https', 'ai', 'rss', 'xml', 'json', 'api']
            found_problems = []
            for topic in topics:
                for problem_token in problematic_tokens:
                    if problem_token in topic.lower():
                        found_problems.append(f"{topic}({problem_token})")
            
            if found_problems:
                print(f"⚠️ 문제 토큰 발견: {found_problems}")
            else:
                print(f"✅ 문제 토큰 없음")
        
    except Exception as e:
        print(f"❌ 파이프라인 디버깅 오류: {e}")

def main():
    """메인 디버깅 함수"""
    print("🚀 데이터 품질 디버깅 시작")
    print("=" * 60)
    
    # 각 단계별 디버깅
    debug_naver_news_quality()
    debug_arxiv_quality()
    debug_topic_extraction()
    debug_data_collection_pipeline()
    
    print("\n" + "=" * 60)
    print("✅ 데이터 품질 디버깅 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
