"""
의미 없는 단어들이 어디서 오는지 분석하는 스크립트
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv("trendbot.env")

# 프로젝트 루트를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def analyze_raw_data_sources():
    """원본 데이터 소스에서 의미 없는 단어 분석"""
    print("=" * 60)
    print("🔍 원본 데이터 소스 분석")
    print("=" * 60)
    
    try:
        from src.data_collectors.naver_collector import search_news
        from src.data_collectors.arxiv_collector import ArxivCollector
        
        # Naver 뉴스 원본 데이터 분석
        print("\n📰 Naver 뉴스 원본 데이터 분석:")
        df = search_news("인공지능", display=3)
        
        for i, (_, row) in enumerate(df.iterrows()):
            title = row.get('title', '')
            desc = row.get('desc', '')
            
            print(f"\n  📄 뉴스 {i+1}:")
            print(f"    제목: {title}")
            print(f"    설명: {desc}")
            
            # HTML 태그나 특수 문자 분석
            if '<' in title or '<' in desc:
                print(f"    ⚠️ HTML 태그 발견")
            if '&' in title or '&' in desc:
                print(f"    ⚠️ HTML 엔티티 발견")
            if 'http' in title or 'http' in desc:
                print(f"    ⚠️ URL 발견")
        
        # arXiv 논문 원본 데이터 분석
        print("\n📚 arXiv 논문 원본 데이터 분석:")
        collector = ArxivCollector()
        df_papers = collector.collect_papers(["artificial intelligence"], max_results=2)
        
        for i, (_, row) in enumerate(df_papers.iterrows()):
            title = row.get('title', '')
            summary = row.get('summary', '')
            
            print(f"\n  📄 논문 {i+1}:")
            print(f"    제목: {title}")
            print(f"    요약: {summary[:200]}...")
            
            # 특수 문자 분석
            if '<' in title or '<' in summary:
                print(f"    ⚠️ HTML 태그 발견")
            if '&' in title or '&' in summary:
                print(f"    ⚠️ HTML 엔티티 발견")
            if 'http' in title or 'http' in summary:
                print(f"    ⚠️ URL 발견")
        
    except Exception as e:
        print(f"❌ 원본 데이터 분석 오류: {e}")

def analyze_text_cleaning_process():
    """텍스트 정제 과정 분석"""
    print("=" * 60)
    print("🔍 텍스트 정제 과정 분석")
    print("=" * 60)
    
    try:
        from src.nlp.clean import normalize_for_topics
        import re
        import html
        
        # 테스트 텍스트들
        test_texts = [
            "인공지능(AI) 기술이 발전하고 있습니다",
            "&nbsp;font href=https://example.com AI 기술",
            "RSS XML JSON API 관련 내용입니다",
            "<b>인공지능</b> &amp; 머신러닝 기술",
            "AI 기술의 발전이 가속화되고 있습니다"
        ]
        
        for i, text in enumerate(test_texts):
            print(f"\n📝 테스트 텍스트 {i+1}:")
            print(f"  원본: {text}")
            
            # 단계별 정제 과정 추적
            step1 = html.unescape(text)
            print(f"  HTML 디코드: {step1}")
            
            step2 = re.sub(r'<[^>]+>', '', step1)
            print(f"  HTML 태그 제거: {step2}")
            
            step3 = re.sub(r'&[a-zA-Z0-9#]+;', '', step2)
            print(f"  HTML 엔티티 제거: {step3}")
            
            step4 = re.sub(r'https?://[^\s]+', '', step3)
            step4 = re.sub(r'www\.[^\s]+', '', step4)
            print(f"  URL 제거: {step4}")
            
            step5 = re.sub(r'[^\w\s가-힣]', ' ', step4)
            print(f"  특수문자 제거: {step5}")
            
            final = normalize_for_topics(text)
            print(f"  최종 결과: {final}")
            
            # 의미 없는 단어 체크
            meaningless_words = ['rss', 'xml', 'json', 'api', 'http', 'www', 'com']
            found = [word for word in meaningless_words if word in final.lower()]
            if found:
                print(f"  ⚠️ 의미 없는 단어 발견: {found}")
            else:
                print(f"  ✅ 의미 없는 단어 없음")
        
    except Exception as e:
        print(f"❌ 텍스트 정제 분석 오류: {e}")

def analyze_topic_extraction_process():
    """토픽 추출 과정 분석"""
    print("=" * 60)
    print("🔍 토픽 추출 과정 분석")
    print("=" * 60)
    
    try:
        from src.ai.topic_extractor import TopicExtractor
        from src.data_collectors.naver_collector import NaverCollector
        from src.data_collectors.arxiv_collector import ArxivCollector
        
        # 실제 데이터로 테스트
        naver_collector = NaverCollector(os.getenv("NAVER_CLIENT_ID"), os.getenv("NAVER_CLIENT_SECRET"))
        arxiv_collector = ArxivCollector()
        
        # 뉴스 데이터 수집
        news_data = naver_collector.collect_all_news(["인공지능"])
        all_texts = []
        
        for source, articles in news_data.items():
            for article in articles[:3]:  # 처음 3개만
                text = f"{article.get('title', '')} {article.get('description', '')}"
                if text.strip():
                    all_texts.append(text)
        
        print(f"📊 수집된 텍스트: {len(all_texts)}개")
        
        # 각 텍스트의 원본 내용 확인
        print("\n📋 원본 텍스트 샘플:")
        for i, text in enumerate(all_texts[:3]):
            print(f"  {i+1}: {text[:100]}...")
            
            # 문제가 될 수 있는 패턴 체크
            if 'AI' in text:
                print(f"    ⚠️ 'AI' 발견")
            if '<' in text:
                print(f"    ⚠️ HTML 태그 발견")
            if '&' in text:
                print(f"    ⚠️ HTML 엔티티 발견")
            if 'http' in text:
                print(f"    ⚠️ URL 발견")
        
        # 토픽 추출
        print("\n🎯 토픽 추출 결과:")
        extractor = TopicExtractor()
        topics = extractor.top_topics(all_texts, k=10)
        
        print(f"추출된 토픽: {topics}")
        
        # 의미 없는 단어 체크
        meaningless_words = ['ai', 'rss', 'xml', 'json', 'api', 'http', 'www', 'com', 'target', 'oc']
        found = []
        for topic in topics:
            for word in meaningless_words:
                if word in topic.lower():
                    found.append(f"{topic}({word})")
        
        if found:
            print(f"⚠️ 의미 없는 단어 발견: {found}")
        else:
            print(f"✅ 의미 없는 단어 없음")
        
    except Exception as e:
        print(f"❌ 토픽 추출 분석 오류: {e}")

def analyze_tfidf_features():
    """TF-IDF 특성 분석"""
    print("=" * 60)
    print("🔍 TF-IDF 특성 분석")
    print("=" * 60)
    
    try:
        from src.ai.topic_extractor import TopicExtractor
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # 테스트 코퍼스
        test_corpus = [
            "인공지능 기술이 발전하고 있습니다",
            "머신러닝 알고리즘을 연구합니다",
            "딥러닝 모델을 개발합니다",
            "AI 기술의 발전이 가속화되고 있습니다",  # 문제가 될 수 있는 텍스트
            "RSS XML JSON API 관련 내용입니다"  # 문제가 될 수 있는 텍스트
        ]
        
        print("📊 테스트 코퍼스:")
        for i, text in enumerate(test_corpus):
            print(f"  {i+1}: {text}")
        
        # TF-IDF 벡터라이저 설정 (토픽 추출기와 동일한 설정)
        from src.nlp.clean import STOPWORDS, GARBAGE_TOKENS
        combined_stopwords = STOPWORDS | GARBAGE_TOKENS
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8,
            stop_words=list(combined_stopwords),
            token_pattern=r'(?u)\b[가-힣a-zA-Z]{2,}\b'
        )
        
        # 텍스트 정제
        extractor = TopicExtractor()
        cleaned_corpus = []
        for text in test_corpus:
            clean_text = extractor.preprocess_text(text)
            cleaned_corpus.append(clean_text)
        
        print("\n🧹 정제된 코퍼스:")
        for i, text in enumerate(cleaned_corpus):
            print(f"  {i+1}: {text}")
        
        # TF-IDF 행렬 생성
        tfidf_matrix = vectorizer.fit_transform(cleaned_corpus)
        feature_names = vectorizer.get_feature_names_out()
        
        print(f"\n📈 TF-IDF 특성:")
        print(f"  특성 수: {len(feature_names)}")
        print(f"  특성 샘플: {feature_names[:20]}")
        
        # 평균 TF-IDF 점수 계산
        mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # 상위 특성들
        top_indices = np.argsort(mean_scores)[::-1][:10]
        top_features = [(feature_names[i], mean_scores[i]) for i in top_indices]
        
        print(f"\n🎯 상위 특성들:")
        for feature, score in top_features:
            print(f"  {feature}: {score:.4f}")
            
            # 의미 없는 단어 체크
            meaningless_words = ['ai', 'rss', 'xml', 'json', 'api', 'http', 'www', 'com']
            if any(word in feature.lower() for word in meaningless_words):
                print(f"    ⚠️ 의미 없는 단어 포함!")
        
    except Exception as e:
        print(f"❌ TF-IDF 분석 오류: {e}")

def main():
    """메인 분석 함수"""
    print("🚀 의미 없는 단어 분석 시작")
    print("=" * 60)
    
    analyze_raw_data_sources()
    analyze_text_cleaning_process()
    analyze_topic_extraction_process()
    analyze_tfidf_features()
    
    print("\n" + "=" * 60)
    print("✅ 의미 없는 단어 분석 완료")
    print("=" * 60)

if __name__ == "__main__":
    main()
