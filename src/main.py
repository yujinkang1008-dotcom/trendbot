"""
트렌드 분석 서비스 메인 애플리케이션 - 한국어 분석 전용
"""
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# 프로젝트 루트 디렉터리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.common.config import Config
from src.common.trace import snapshot_df, log_shape
from data_collectors import NaverCollector
from ai import SentimentAnalyzer, TopicExtractor
from ai.clustering_analyzer import ClusteringAnalyzer
from visualization import ChartGenerator, WordCloudGenerator

class TrendAnalyzer:
    """트렌드 분석 메인 클래스 - 한국어 전용"""
    
    def __init__(self):
        # 설정 검증
        Config.validate_config()
        
        # 데이터 수집기 초기화 (한국어 전용)
        self.naver = NaverCollector(Config.NAVER_CLIENT_ID, Config.NAVER_CLIENT_SECRET)
        
        # AI 분석기 초기화
        self.sentiment_analyzer = SentimentAnalyzer(Config.HUGGINGFACE_API_KEY)
        self.topic_extractor = TopicExtractor()
        self.clustering_analyzer = ClusteringAnalyzer()
        
        # 시각화 도구 초기화
        self.chart_generator = ChartGenerator()
        self.wordcloud_generator = WordCloudGenerator()
    
    def collect_korean_data(self, keywords: list, period_days: int, use_naver_news: bool = True, 
                           use_naver_blog: bool = True, use_naver_datalab: bool = False) -> dict:
        """
        한국어 데이터 수집
        
        Args:
            keywords: 검색 키워드 리스트
            period_days: 수집 기간 (일)
            use_naver_news: 네이버 뉴스 사용 여부
            use_naver_blog: 네이버 블로그 사용 여부
            use_naver_datalab: 네이버 데이터랩 사용 여부
            
        Returns:
            dict: 수집된 데이터
        """
        data = {
            'news_data': {},
            'analysis_timestamp': datetime.now().isoformat(),
            'period_days': period_days,
            'keywords': keywords
        }
        
        # 네이버 뉴스 수집
        if use_naver_news:
            try:
                # 키워드를 문자열로 변환
                query = ' '.join(keywords) if isinstance(keywords, list) else str(keywords)
                
                with st.spinner("🌐 네이버 뉴스 수집 중..."):
                    news_data = self.naver.search_news(query, display=100)
                
                if isinstance(news_data, pd.DataFrame) and not news_data.empty:
                    data['news_data']['naver_news'] = news_data.to_dict('records')
                    st.success(f"✅ 네이버 뉴스 수집 완료: {len(news_data)}개")
                elif isinstance(news_data, pd.DataFrame) and news_data.empty:
                    st.warning("⚠️ 네이버 뉴스 데이터가 없습니다.")
                else:
                    st.error(f"❌ 네이버 뉴스 수집 실패: 잘못된 데이터 타입")
                    
            except Exception as e:
                st.error(f"❌ 네이버 뉴스 수집 실패: {str(e)}")
        
        # 네이버 블로그 수집
        if use_naver_blog:
            try:
                # 키워드를 문자열로 변환
                query = ' '.join(keywords) if isinstance(keywords, list) else str(keywords)
                
                with st.spinner("🌐 네이버 블로그 수집 중..."):
                    blog_data = self.naver.search_blog(query, display=100)
                
                if isinstance(blog_data, pd.DataFrame) and not blog_data.empty:
                    data['news_data']['naver_blog'] = blog_data.to_dict('records')
                    st.success(f"✅ 네이버 블로그 수집 완료: {len(blog_data)}개")
                elif isinstance(blog_data, pd.DataFrame) and blog_data.empty:
                    st.warning("⚠️ 네이버 블로그 데이터가 없습니다.")
                else:
                    st.error(f"❌ 네이버 블로그 수집 실패: 잘못된 데이터 타입")
                    
            except Exception as e:
                st.error(f"❌ 네이버 블로그 수집 실패: {str(e)}")
        
        return data
    
    def analyze_korean_data(self, data: dict, use_morphology: bool = True) -> dict:
        """
        한국어 데이터 분석
        
        Args:
            data: 수집된 데이터
            use_morphology: 형태소 분석 사용 여부
            
        Returns:
            dict: 분석 결과
        """
        results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'use_morphology': use_morphology
        }
        
        # 토픽 분석
        try:
            st.info("🔍 토픽 분석 중...")
            if 'news_data' in data and data['news_data']:
                news_data = data['news_data']
                
                # 뉴스 토픽 분석
                if 'naver_news' in news_data and news_data['naver_news']:
                    news_topics = self.topic_extractor.extract_topics_simple(news_data['naver_news'])
                    results['news_topics'] = news_topics
                    st.success(f"✅ 뉴스 토픽 분석 완료: {len(news_topics)}개")
                
                # 블로그 토픽 분석
                if 'naver_blog' in news_data and news_data['naver_blog']:
                    blog_topics = self.topic_extractor.extract_topics_simple(news_data['naver_blog'])
                    results['blog_topics'] = blog_topics
                    st.success(f"✅ 블로그 토픽 분석 완료: {len(blog_topics)}개")
        except Exception as e:
            st.error(f"❌ 토픽 분석 실패: {str(e)}")
        
        # 감성 분석
        try:
            st.info("😊 감성 분석 중...")
            if 'news_data' in data and data['news_data']:
                all_texts = []
                news_data = data['news_data']
                
                if 'naver_news' in news_data and news_data['naver_news']:
                    for news in news_data['naver_news']:
                        if 'text_clean' in news:
                            all_texts.append(news['text_clean'])
                        elif 'description' in news:
                            all_texts.append(news['description'])
                
                if 'naver_blog' in news_data and news_data['naver_blog']:
                    for blog in news_data['naver_blog']:
                        if 'text_clean' in blog:
                            all_texts.append(blog['text_clean'])
                        elif 'description' in blog:
                            all_texts.append(blog['description'])
                
                if all_texts:
                    sentiment_results = self.sentiment_analyzer.analyze_batch_sentiment(all_texts)
                    results['sentiment_results'] = sentiment_results
                    st.success("✅ 감성 분석 완료")
        except Exception as e:
            st.error(f"❌ 감성 분석 실패: {str(e)}")
        
        # 클러스터링 분석
        try:
            st.info("🔗 클러스터링 분석 중...")
            if 'news_data' in data and data['news_data']:
                all_texts = []
                news_data = data['news_data']
                
                if 'naver_news' in news_data and news_data['naver_news']:
                    for news in news_data['naver_news']:
                        if 'text_clean' in news:
                            all_texts.append(news['text_clean'])
                        elif 'description' in news:
                            all_texts.append(news['description'])
                
                if 'naver_blog' in news_data and news_data['naver_blog']:
                    for blog in news_data['naver_blog']:
                        if 'text_clean' in blog:
                            all_texts.append(blog['text_clean'])
                        elif 'description' in blog:
                            all_texts.append(blog['description'])
                
                if all_texts:
                    clustering_results = self.clustering_analyzer.cluster_documents(all_texts)
                    results['clustering_results'] = clustering_results
                    st.success("✅ 클러스터링 분석 완료")
        except Exception as e:
            st.error(f"❌ 클러스터링 분석 실패: {str(e)}")
        
        # 시각화 생성
        try:
            st.info("📊 시각화 생성 중...")
            visualizations = self._create_visualizations(data, results)
            results['visualizations'] = visualizations
            st.success("✅ 시각화 생성 완료")
        except Exception as e:
            st.error(f"❌ 시각화 생성 실패: {str(e)}")
        
        return results
    
    def _create_visualizations(self, data: dict, results: dict) -> dict:
        """시각화 생성"""
        visualizations = {}
        
        try:
            # 뉴스 주제 차트
            if 'news_topics' in results and results['news_topics']:
                visualizations['news_topics_chart'] = self.chart_generator.create_topic_frequency_chart(results['news_topics'])
            
            # 블로그 주제 차트
            if 'blog_topics' in results and results['blog_topics']:
                visualizations['blog_topics_chart'] = self.chart_generator.create_topic_frequency_chart(results['blog_topics'])
            
            # 뉴스 건수 추이
            if 'news_data' in data and data['news_data'] and 'naver_news' in data['news_data']:
                news_df = pd.DataFrame(data['news_data']['naver_news'])
                if not news_df.empty and 'pub_date' in news_df.columns:
                    news_df['date'] = pd.to_datetime(news_df['pub_date'], errors='coerce')
                    news_df = news_df.dropna(subset=['date'])
                    if not news_df.empty:
                        visualizations['news_count'] = self.chart_generator.create_news_count_chart({
                            'naver_news': news_df
                        })
            
            # 블로그 건수 추이
            if 'news_data' in data and data['news_data'] and 'naver_blog' in data['news_data']:
                blog_df = pd.DataFrame(data['news_data']['naver_blog'])
                if not blog_df.empty and 'pub_date' in blog_df.columns:
                    blog_df['date'] = pd.to_datetime(blog_df['pub_date'], errors='coerce')
                    blog_df = blog_df.dropna(subset=['date'])
                    if not blog_df.empty:
                        visualizations['blog_count'] = self.chart_generator.create_news_count_chart({
                            'naver_blog': blog_df
                        })
            
            # 워드클라우드
            if 'news_data' in data and data['news_data']:
                all_texts = []
                news_data = data['news_data']
                
                if 'naver_news' in news_data and news_data['naver_news']:
                    for news in news_data['naver_news']:
                        if 'text_clean' in news:
                            all_texts.append(news['text_clean'])
                        elif 'description' in news:
                            all_texts.append(news['description'])
                
                if 'naver_blog' in news_data and news_data['naver_blog']:
                    for blog in news_data['naver_blog']:
                        if 'text_clean' in blog:
                            all_texts.append(blog['text_clean'])
                        elif 'description' in blog:
                            all_texts.append(blog['description'])
                
                if all_texts:
                    topic_results = self.topic_extractor.extract_topics_simple(all_texts)
                    if topic_results:
                        word_freq = {t['word']: t['count'] for t in topic_results}
                        visualizations['wordcloud'] = self.wordcloud_generator.generate_from_frequency(word_freq)
            
            # 감성 분석 차트
            if 'sentiment_results' in results and results['sentiment_results']:
                visualizations['sentiment_chart'] = self.chart_generator.create_sentiment_chart(results['sentiment_results'])
        
        except Exception as e:
            st.warning(f"⚠️ 시각화 생성 중 일부 오류: {str(e)}")
        
        return visualizations

def main():
    """메인 애플리케이션 - 한국어 분석 전용"""
    st.set_page_config(
        page_title="Trendbot - 트렌드 분석",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Trendbot - 트렌드 분석")
    st.markdown("**한국어 키워드**로 뉴스, 블로그 데이터를 수집하고 **KOMORAN 형태소 분석**으로 정확한 분석을 제공합니다.")
    
    # 사이드바에 검색창과 설정
    with st.sidebar:
        st.header("🔍 분석 설정")
        
        # 한국어 키워드 입력
        st.subheader("🔤 한국어 키워드")
        korean_keywords = st.text_input(
            "🇰🇷 분석할 키워드를 입력하세요",
            placeholder="예: 생성형 ai, 인공지능, 머신러닝 (쉼표로 구분)",
            key="korean_keywords"
        )
        
        if not korean_keywords:
            st.warning("⚠️ 한국어 키워드를 입력해주세요.")
            analyze_button = False
        else:
            analyze_button = True
        
        # 수집 기간 설정
        period_days = st.selectbox(
            "분석 기간",
            options=[30, 90, 180, 365, 730, 1095, 1460, 1825],
            index=1,  # 기본값: 90일
            format_func=lambda x: f"{x}일 ({x//365}년 {x%365//30}개월)" if x >= 365 else f"{x}일",
            key="main_period_days"
        )
        
        # 데이터 소스 선택
        st.subheader("📊 한국어 데이터 소스")
        use_naver_news = st.checkbox("네이버 뉴스", value=True, key="sidebar_naver_news")
        use_naver_blog = st.checkbox("네이버 블로그", value=True, key="sidebar_naver_blog")
        use_naver_datalab = st.checkbox("네이버 데이터랩", value=False, key="sidebar_naver_datalab")
        
        # 형태소 분석 설정
        st.subheader("🔤 KOMORAN 분석")
        use_morphology = st.checkbox("KOMORAN 형태소 분석 사용", value=True, help="정확한 한국어 형태소 분석", key="sidebar_morphology")
        if use_morphology:
            st.info("✅ 정확한 한국어 형태소 분석을 제공합니다.")
        
        # 분석 시작 버튼
        analyze_button = st.button("🚀 트렌드 분석 시작", type="primary", key="main_analyze", use_container_width=True)
        
        if analyze_button and korean_keywords:
            # 키워드 분리 및 처리
            korean_keyword_list = [kw.strip() for kw in korean_keywords.split(',') if kw.strip()]
            
            # 전역 상태에 키워드와 설정 저장
            st.session_state['korean_keyword_list'] = korean_keyword_list
            st.session_state['period_days'] = period_days
            st.session_state['use_naver_news'] = use_naver_news
            st.session_state['use_naver_blog'] = use_naver_blog
            st.session_state['use_naver_datalab'] = use_naver_datalab
            st.session_state['use_morphology'] = use_morphology
            st.session_state['analysis_completed'] = True
            
            st.success("✅ 트렌드 분석 설정이 저장되었습니다!")
    
    # 키워드 표시 (메인 영역)
    if st.session_state.get('analysis_completed', False):
        st.header("📝 분석 키워드")
        
        korean_keywords = st.session_state.get('korean_keyword_list', [])
        if korean_keywords:
            st.write("**🇰🇷 한국어 키워드:**")
            for keyword in korean_keywords:
                st.write(f"- {keyword}")
        
        # 한국어 분석 실행
        korean_analysis_interface()
    else:
        # 초기 화면
        st.markdown("""
        ## 🎯 트렌드 분석 사용법
        
        1. **왼쪽 사이드바**에서 분석할 **한국어 키워드**를 입력하세요
        2. **분석 기간**을 선택하세요 (최대 5년)
        3. **데이터 소스**를 선택하세요 (Naver 뉴스/블로그)
        4. **트렌드 분석 시작** 버튼을 클릭하세요
        
        ### 📊 제공되는 분석 기능
        
        - **뉴스 분석**: Naver 뉴스 데이터 수집 및 분석
        - **블로그 분석**: Naver 블로그 데이터 수집 및 분석
        - **KOMORAN 형태소 분석**: 정확한 한국어 텍스트 분석
        - **감성 분석**: 한국어 텍스트의 긍정/부정/중립 분석
        - **토픽 분석**: 주요 키워드와 주제 추출
        - **클러스터링**: 유사한 내용 그룹화
        - **시각화**: 차트와 워드클라우드
        
        ### 🔧 KOMORAN 형태소 분석의 장점
        
        - **정확한 품사 분석**: 명사, 동사, 형용사 등 정확한 분류
        - **의미 있는 키워드 추출**: 불용어 제거로 핵심 키워드만 추출
        - **한국어 특화**: 한국어 문법에 최적화된 분석
        """)

def korean_analysis_interface():
    """한국어 데이터 분석 인터페이스"""
    st.header("🇰🇷 한국어 데이터 분석")
    st.markdown("네이버 뉴스, 블로그 등 한국어 데이터를 KOMORAN 형태소 분석기로 분석합니다.")
    
    # 전역 상태에서 키워드 확인
    if st.session_state.get('analysis_completed', False):
        korean_keywords = st.session_state.get('korean_keyword_list', [])
        period_days = st.session_state.get('period_days', 90)
        use_naver_news = st.session_state.get('use_naver_news', True)
        use_naver_blog = st.session_state.get('use_naver_blog', True)
        use_naver_datalab = st.session_state.get('use_naver_datalab', True)
        use_morphology = st.session_state.get('use_morphology', True)
        
        # 한국어 키워드가 있는지 확인
        if not korean_keywords:
            st.warning("⚠️ 한국어 키워드가 입력되지 않았습니다. 한국어권 분석을 위해서는 한국어 키워드를 입력해주세요.")
            return
        
        # 설정 정보 표시
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("한국어 키워드", ', '.join(korean_keywords))
        with col2:
            st.metric("분석 기간", f"{period_days}일")
        with col3:
            sources = []
            if use_naver_news:
                sources.append("네이버 뉴스")
            if use_naver_blog:
                sources.append("네이버 블로그")
            if use_naver_datalab:
                sources.append("네이버 데이터랩")
            st.metric("데이터 소스", ', '.join(sources) if sources else "없음")
        
        # 형태소 분석 설정 표시
        if use_morphology:
            st.success("✅ KOMORAN 형태소 분석 활성화")
        else:
            st.info("ℹ️ 기본 토큰화 사용")
        
        # 자동 분석 시작
        with st.spinner("한국어 데이터를 수집하고 분석 중입니다..."):
            try:
                # 트렌드 분석기 초기화
                analyzer = TrendAnalyzer()
                
                # 한국어 데이터 수집 (개선된 수집량)
                st.info("📊 한국어 데이터 수집 중...")
                data = analyzer.collect_korean_data(korean_keywords, period_days, use_naver_news, use_naver_blog, use_naver_datalab)
                
                # 수집된 데이터 품질 확인
                if data:
                    news_count = len(data.get('news_data', {}).get('naver_news', []))
                    blog_count = len(data.get('news_data', {}).get('naver_blog', []))
                    st.success(f"✅ 데이터 수집 완료: 뉴스 {news_count}개, 블로그 {blog_count}개")
                else:
                    st.warning("⚠️ 수집된 데이터가 없습니다.")
                
                # 분석 수행 (형태소 분석 포함)
                results = analyzer.analyze_korean_data(data, use_morphology)
                
                # 결과 표시
                display_korean_results(results, korean_keywords)
                
            except Exception as e:
                st.error(f"❌ 한국어 분석 중 오류가 발생했습니다: {str(e)}")
                if Config.DEBUG:
                    st.exception(e)

def display_korean_results(results: dict, keywords: list):
    """한국어 분석 결과 표시"""
    st.header("📊 한국어 분석 결과")
    
    # 데이터 요약
    st.subheader("📈 데이터 요약")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("분석 키워드", ', '.join(keywords))
    with col2:
        st.metric("분석 시점", results.get('analysis_timestamp', 'N/A')[:10])
    with col3:
        # 클러스터링 결과 요약
        clustering_results = results.get('clustering_results', {})
        if isinstance(clustering_results, dict) and 'clusters' in clustering_results:
            cluster_count = len(clustering_results['clusters'])
            st.metric("클러스터 수", cluster_count)
        else:
            st.metric("클러스터 수", "N/A")
    
    # 시각화 표시 (한국어 분석 최적화)
    if 'visualizations' in results:
        visualizations = results['visualizations']
        
        if visualizations:
            st.markdown("---")
            st.subheader("📊 한국어 분석 시각화")
            
            # 3열 레이아웃으로 시각화 표시 (한국어 분석에 최적화)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # 뉴스 주제 차트
                if 'news_topics_chart' in visualizations:
                    st.subheader("📰 뉴스 주제 Top 10")
                    st.plotly_chart(visualizations['news_topics_chart'], use_container_width=True)
                
                # 뉴스 건수 추이
                if 'news_count' in visualizations:
                    st.subheader("📈 뉴스 건수 추이")
                    st.plotly_chart(visualizations['news_count'], use_container_width=True)
            
            with col2:
                # 블로그 주제 차트
                if 'blog_topics_chart' in visualizations:
                    st.subheader("📝 블로그 주제 Top 10")
                    st.plotly_chart(visualizations['blog_topics_chart'], use_container_width=True)
                
                # 블로그 건수 추이
                if 'blog_count' in visualizations:
                    st.subheader("📈 블로그 건수 추이")
                    st.plotly_chart(visualizations['blog_count'], use_container_width=True)
            
            with col3:
                # 워드클라우드
                if 'wordcloud' in visualizations:
                    st.subheader("☁️ 키워드 워드클라우드")
                    st.image(visualizations['wordcloud'], caption="KOMORAN 분석 기반 주요 키워드")
                
                # 감성 분석 차트
                if 'sentiment_chart' in visualizations:
                    st.subheader("😊 감성 분석")
                    st.plotly_chart(visualizations['sentiment_chart'], use_container_width=True)
        else:
            st.info("시각화 데이터가 없습니다.")
    
    # Raw 데이터 접기/펼치기
    st.markdown("---")
    with st.expander("🔍 Raw 데이터 보기 (클릭하여 펼치기)", expanded=False):
        st.subheader("📊 수집된 데이터")
        
        # 뉴스 데이터
        news_data = results.get('news_data', {}).get('naver_news', [])
        if news_data:
            st.subheader("📰 네이버 뉴스 데이터")
            news_df = pd.DataFrame(news_data)
            st.dataframe(news_df, use_container_width=True)
        
        # 블로그 데이터
        blog_data = results.get('news_data', {}).get('naver_blog', [])
        if blog_data:
            st.subheader("📝 네이버 블로그 데이터")
            blog_df = pd.DataFrame(blog_data)
            st.dataframe(blog_df, use_container_width=True)
        
        # 분석 결과
        if 'topic_results' in results:
            st.subheader("🔤 토픽 분석 결과")
            topic_results = results['topic_results']
            if isinstance(topic_results, dict):
                for key, value in topic_results.items():
                    st.write(f"**{key}:** {value}")
        
        if 'sentiment_results' in results:
            st.subheader("😊 감성 분석 결과")
            sentiment_results = results['sentiment_results']
            if isinstance(sentiment_results, dict):
                for key, value in sentiment_results.items():
                    st.write(f"**{key}:** {value}")
        
        if 'clustering_results' in results:
            st.subheader("🔗 클러스터링 결과")
            clustering_results = results['clustering_results']
            if isinstance(clustering_results, dict):
                st.json(clustering_results)

if __name__ == "__main__":
    main()