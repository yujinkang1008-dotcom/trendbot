"""
토픽 추출 모듈 (형태소 분석기 통합)
"""
from typing import List, Dict, Any
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from src.nlp.clean import STOPWORDS, GARBAGE_TOKENS
from src.nlp.morphological_analyzer import MorphologicalAnalyzer

class TopicExtractor:
    """토픽 추출 클래스"""
    
    def __init__(self, use_morphology: bool = True):
        """
        토픽 추출기 초기화
        
        Args:
            use_morphology: 형태소 분석기 사용 여부
        """
        self.use_morphology = use_morphology
        
        # 형태소 분석기 초기화 (KOMORAN 우선 사용)
        if self.use_morphology:
            try:
                # KOMORAN을 우선 시도
                self.morph_analyzer = MorphologicalAnalyzer("komoran")
                print("✅ KOMORAN 형태소 분석기 초기화 완료")
            except Exception as e:
                print(f"⚠️ KOMORAN 형태소 분석기 초기화 실패: {e}")
                try:
                    # Okt로 대체 시도
                    self.morph_analyzer = MorphologicalAnalyzer("okt")
                    print("✅ Okt 형태소 분석기 초기화 완료")
                except Exception as e2:
                    print(f"⚠️ Okt 형태소 분석기도 실패: {e2}")
                    print("🔄 기본 토큰화 방식으로 전환")
                    self.use_morphology = False
                    self.morph_analyzer = None
        else:
            self.morph_analyzer = None
        
        # 불용어 리스트 (한국어) - 기본 불용어
        self.stop_words = {
            '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '도', '는', '은',
            '하다', '있다', '없다', '되다', '이다', '아니다', '그', '이', '저', '그것', '이것', '저것',
            '그런', '이런', '저런', '그렇게', '이렇게', '저렇게', '그때', '이때', '저때',
            '그리고', '하지만', '그러나', '또한', '또', '그래서', '따라서', '그러므로',
            '위해', '위한', '대해', '대한', '관해', '관한', '통해', '통한',
            '때문', '때문에', '덕분', '덕분에', '비해', '비해', '대비', '대비해',
            '관련', '관련된', '관련해', '관련해서', '관련되다', '관련된다',
            '경우', '경우에', '경우에는', '경우', '경우에', '경우에는',
            '수', '수도', '수는', '수만', '수만큼', '수만큼이나',
            '것', '것이', '것을', '것에', '것으로', '것으로서', '것으로써',
            '등', '등의', '등이', '등을', '등에', '등으로', '등으로서', '등으로써',
            '및', '그리고', '또한', '또', '그래서', '따라서', '그러므로',
            '하지만', '그러나', '그런데', '그런', '이런', '저런',
            '그렇게', '이렇게', '저렇게', '그때', '이때', '저때',
            '그리고', '하지만', '그러나', '또한', '또', '그래서', '따라서', '그러므로'
        }
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text: 전처리할 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        if not text:
            return ""
        
        # 특수문자 제거
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # 숫자 제거
        text = re.sub(r'\d+', '', text)
        
        # 공백 정리
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_keywords(self, texts: List[str], max_keywords: int = 50) -> List[Dict]:
        """
        키워드 추출 (형태소 분석 기반)
        
        Args:
            texts: 분석할 텍스트 리스트
            max_keywords: 최대 키워드 수
            
        Returns:
            List[Dict]: 키워드 리스트 [{'keyword': 'AI', 'count': 50, 'pos': 'Noun'}, ...]
        """
        if not texts:
            return []
        
        # 형태소 분석기 사용
        if self.use_morphology and self.morph_analyzer:
            return self._extract_keywords_with_morphology(texts, max_keywords)
        else:
            return self._extract_keywords_basic(texts, max_keywords)
    
    def _extract_keywords_with_morphology(self, texts: List[str], max_keywords: int) -> List[Dict]:
        """
        형태소 분석을 이용한 키워드 추출
        
        Args:
            texts: 분석할 텍스트 리스트
            max_keywords: 최대 키워드 수
            
        Returns:
            List[Dict]: 키워드 리스트
        """
        print(f"🔍 형태소 분석 기반 키워드 추출: {len(texts)}개 텍스트")
        
        all_keywords = []
        for i, text in enumerate(texts):
            if text:
                print(f"  📝 텍스트 {i+1}: {text[:50]}...")
                keywords = self.morph_analyzer.extract_keywords(text, max_keywords=100)
                print(f"  🔑 추출된 키워드: {len(keywords)}개")
                if keywords:
                    print(f"  📋 키워드 샘플: {[kw['keyword'] for kw in keywords[:5]]}")
                all_keywords.extend(keywords)
        
        # 빈도 계산
        keyword_counts = Counter([kw['keyword'] for kw in all_keywords])
        
        # 상위 키워드 선택
        top_keywords = keyword_counts.most_common(max_keywords)
        
        # 결과 정리 (HTML/웹 관련 불용어 추가 필터링 - 강화)
        html_stopwords = {
            'nbsp', 'font', 'href', 'blank', 'target', 'src', 'img', 'div', 'span', 'class', 'id',
            'script', 'css', 'js', 'jquery', 'ajax', 'json', 'xml', 'html', 'htm', 'php', 'asp',
            'http', 'https', 'www', 'com', 'net', 'org', 'co', 'kr', 'link', 'url',
            'rss', 'feed', 'atom', 'syndication', 'channel', 'item', 'description', 'pubdate',
            'guid', 'category', 'enclosure', 'articles', 'target', 'oc', 'ios', 'android',
            'windows', 'mac', 'linux', 'google', 'news', 'naver', 'daum', 'yahoo', 'bing',
            'search', 'youtube', 'facebook', 'twitter', 'instagram', 'linkedin', 'github',
            # HTML 속성값들
            'color', 'red', 'blue', 'green', 'yellow', 'black', 'white', 'gray', 'grey',
            'size', 'width', 'height', 'border', 'margin', 'padding', 'background',
            'display', 'position', 'float', 'clear', 'overflow', 'visible', 'hidden',
            'block', 'inline', 'table', 'cell', 'row', 'column', 'header', 'footer',
            'left', 'right', 'center', 'top', 'bottom', 'middle', 'start', 'end',
            'bold', 'italic', 'underline', 'strike', 'normal', 'small', 'large', 'big',
            'tiny', 'huge', 'massive', 'mini', 'micro', 'macro', 'full', 'empty',
            'half', 'quarter', 'double', 'triple', 'single', 'multiple', 'first', 'last',
            'next', 'prev', 'previous', 'back', 'forward', 'up', 'down', 'over', 'under',
            'above', 'below', 'before', 'after', 'during', 'while', 'since', 'until',
            'within', 'without', 'inside', 'outside', 'beside', 'near', 'far', 'close',
            'distant', 'local', 'global', 'national', 'international', 'regional',
            'urban', 'rural', 'domestic', 'foreign', 'internal', 'external', 'public',
            'private', 'personal', 'individual', 'collective', 'group', 'team',
            'organization', 'company', 'business', 'industry', 'sector', 'field',
            'area', 'region', 'country', 'nation', 'state', 'city', 'town', 'village',
            'community', 'society', 'culture', 'tradition', 'custom', 'habit', 'practice',
            'method', 'way', 'approach', 'technique', 'strategy', 'tactic', 'plan',
            'scheme', 'program', 'project', 'initiative', 'campaign', 'movement',
            'trend', 'pattern', 'model', 'framework', 'structure', 'system', 'process',
            'procedure', 'operation', 'activity', 'action', 'behavior', 'conduct',
            'performance', 'function', 'role', 'purpose', 'goal', 'objective', 'target',
            'aim', 'intention', 'policy', 'rule', 'regulation', 'law', 'legislation',
            'act', 'bill', 'proposal', 'suggestion', 'recommendation', 'advice',
            'guidance', 'direction', 'instruction', 'order', 'command', 'request',
            'demand', 'requirement', 'condition', 'criteria', 'standard', 'level',
            'degree', 'extent', 'scope', 'range', 'limit', 'boundary', 'border',
            'edge', 'margin', 'space', 'room', 'place', 'location', 'position',
            'point', 'spot', 'site', 'venue', 'setting', 'environment', 'context',
            'situation', 'circumstance', 'condition', 'state', 'status', 'case',
            'instance', 'example', 'sample', 'specimen', 'template', 'format',
            'style', 'type', 'kind', 'sort', 'category', 'class', 'set', 'collection',
            'series', 'sequence', 'chain', 'link', 'connection', 'relation',
            'relationship', 'association', 'bond', 'tie', 'bridge', 'gap', 'distance',
            'difference', 'similarity', 'comparison', 'contrast', 'distinction',
            'separation', 'division', 'split', 'break', 'cut', 'slice', 'piece',
            'part', 'section', 'segment', 'portion', 'fraction', 'percentage',
            'ratio', 'proportion', 'rate', 'speed', 'velocity', 'acceleration',
            'momentum', 'force', 'power', 'energy', 'strength', 'weakness',
            'advantage', 'disadvantage', 'benefit', 'cost', 'price', 'value',
            'worth', 'importance', 'significance', 'meaning', 'reason', 'cause',
            'effect', 'result', 'outcome', 'consequence', 'impact', 'influence',
            'change', 'transformation', 'development', 'growth', 'progress',
            'advancement', 'improvement', 'enhancement', 'upgrade', 'update',
            'revision', 'modification', 'adjustment', 'adaptation', 'accommodation',
            'integration', 'coordination', 'cooperation', 'collaboration',
            'partnership', 'alliance', 'union', 'merger', 'acquisition',
            'combination', 'mixture', 'blend', 'fusion', 'synthesis', 'creation',
            'production', 'manufacture', 'construction', 'building', 'establishment',
            'foundation', 'base', 'basis', 'ground', 'support', 'backing',
            'assistance', 'help', 'aid', 'service', 'facility', 'resource',
            'asset', 'property', 'possession', 'ownership', 'control', 'management',
            'administration', 'governance', 'leadership', 'supervision', 'oversight',
            'monitoring', 'tracking', 'following', 'pursuing', 'chasing', 'hunting',
            'seeking', 'searching', 'looking', 'finding', 'discovering', 'detecting',
            'identifying', 'recognizing', 'understanding', 'comprehending', 'learning',
            'studying', 'researching', 'investigating', 'exploring', 'examining',
            'analyzing', 'evaluating', 'assessing', 'judging', 'deciding', 'choosing',
            'selecting', 'picking', 'opting', 'preferring', 'favoring', 'liking',
            'loving', 'enjoying', 'appreciating', 'valuing', 'treasuring', 'cherishing',
            'respecting', 'admiring', 'praising', 'commending', 'recommending',
            'suggesting', 'proposing', 'offering', 'providing', 'supplying',
            'delivering', 'giving', 'presenting', 'showing', 'displaying', 'exhibiting',
            'demonstrating', 'proving', 'confirming', 'verifying', 'validating',
            'authenticating', 'certifying', 'guaranteeing', 'ensuring', 'securing',
            'protecting', 'defending', 'guarding', 'watching', 'observing',
            'noticing', 'seeing', 'viewing', 'listening', 'hearing', 'feeling',
            'touching', 'tasting', 'smelling', 'sensing', 'perceiving', 'experiencing',
            'undergoing', 'suffering', 'enduring', 'bearing', 'tolerating', 'accepting',
            'receiving', 'getting', 'obtaining', 'acquiring', 'gaining', 'earning',
            'winning', 'achieving', 'accomplishing', 'completing', 'finishing',
            'ending', 'concluding', 'terminating', 'stopping', 'ceasing', 'halting',
            'pausing', 'waiting', 'staying', 'remaining', 'continuing', 'persisting',
            'lasting', 'enduring', 'surviving', 'living', 'existing', 'being',
            'becoming', 'growing', 'developing', 'changing', 'transforming', 'evolving',
            'progressing', 'advancing', 'moving', 'going', 'coming', 'arriving',
            'reaching', 'getting', 'obtaining', 'acquiring', 'gaining', 'earning',
            'winning', 'achieving', 'accomplishing', 'completing', 'finishing',
            'ending', 'concluding', 'terminating', 'stopping', 'ceasing', 'halting',
            'pausing', 'waiting', 'staying', 'remaining', 'continuing', 'persisting',
            'lasting', 'enduring', 'surviving', 'living', 'existing', 'being',
            'becoming', 'growing', 'developing', 'changing', 'transforming', 'evolving',
            'progressing', 'advancing', 'moving', 'going', 'coming', 'arriving', 'reaching'
        }
        
        results = []
        for keyword, count in top_keywords:
            # HTML/웹 관련 불용어 필터링
            if keyword.lower() in html_stopwords:
                continue
                
            # 해당 키워드의 품사 정보 찾기
            pos_info = next((kw for kw in all_keywords if kw['keyword'] == keyword), None)
            pos = pos_info['pos'] if pos_info else 'Unknown'
            
            results.append({
                'keyword': keyword,
                'count': count,
                'pos': pos,
                'length': len(keyword)
            })
        
        print(f"✅ 형태소 분석 기반 키워드 추출 완료: {len(results)}개")
        return results
    
    def _extract_keywords_basic(self, texts: List[str], max_keywords: int) -> List[Dict]:
        """
        기본 키워드 추출 (형태소 분석 없이)
        
        Args:
            texts: 분석할 텍스트 리스트
            max_keywords: 최대 키워드 수
            
        Returns:
            List[Dict]: 키워드 리스트
        """
        print(f"🔍 기본 키워드 추출: {len(texts)}개 텍스트")
        
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]
        
        if not processed_texts:
            return []
        
        # 단어 추출 및 빈도 계산
        all_words = []
        for text in processed_texts:
            words = text.split()
            # 불용어 제거 및 길이 필터링
            filtered_words = [
                word for word in words 
                if word not in self.stop_words and len(word) > 1
            ]
            all_words.extend(filtered_words)
        
        # 빈도 계산
        word_freq = Counter(all_words)
        
        # 상위 키워드 선택
        top_keywords = word_freq.most_common(max_keywords)
        
        return [
            {'keyword': word, 'count': count, 'pos': 'Unknown'} 
            for word, count in top_keywords
        ]
    
    def extract_topics_tfidf(self, texts: List[str], n_topics: int = 10) -> List[Dict]:
        """
        TF-IDF를 이용한 토픽 추출
        
        Args:
            texts: 분석할 텍스트 리스트
            n_topics: 추출할 토픽 수
            
        Returns:
            List[Dict]: 토픽 리스트
        """
        if not texts or len(texts) < 2:
            return []
        
        # 텍스트 전처리
        processed_texts = [self.preprocess_text(text) for text in texts]
        processed_texts = [text for text in processed_texts if text]
        
        if len(processed_texts) < 2:
            return []
        
        try:
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words=None,  # 한국어 불용어는 별도 처리
                ngram_range=(1, 2)  # 1-gram과 2-gram 사용
            )
            
            tfidf_matrix = vectorizer.fit_transform(processed_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=min(n_topics, len(processed_texts)), random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 클러스터별 상위 키워드 추출
            topics = []
            for i in range(min(n_topics, len(processed_texts))):
                cluster_indices = np.where(cluster_labels == i)[0]
                if len(cluster_indices) > 0:
                    # 클러스터 내 문서들의 TF-IDF 점수 평균
                    cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                    
                    # 상위 키워드 선택
                    top_indices = cluster_tfidf.argsort()[-10:][::-1]
                    top_keywords = [feature_names[idx] for idx in top_indices if cluster_tfidf[idx] > 0]
                    
                    if top_keywords:
                        topics.append({
                            'topic_id': i,
                            'keywords': top_keywords,
                            'document_count': len(cluster_indices)
                        })
            
            return topics
            
        except Exception as e:
            print(f"TF-IDF 토픽 추출 오류: {e}")
            return []
    
    def extract_topics_simple(self, texts, n_topics: int = 10) -> List[Dict]:
        """
        간단한 토픽 추출 (빈도 기반)
        
        Args:
            texts: 분석할 텍스트 리스트 또는 딕셔너리 리스트
            n_topics: 추출할 토픽 수
            
        Returns:
            List[Dict]: 토픽 리스트
        """
        if not texts:
            return []
        
        # 텍스트 추출 (딕셔너리인 경우 'text' 키에서 추출)
        text_list = []
        for item in texts:
            if isinstance(item, dict):
                # 딕셔너리인 경우 'text' 또는 'text_clean' 키에서 텍스트 추출
                text = item.get('text_clean', item.get('text', ''))
                if text:
                    text_list.append(text)
            elif isinstance(item, str):
                text_list.append(item)
        
        if not text_list:
            return []
        
        # 키워드 추출
        keywords = self.extract_keywords(text_list, max_keywords=100)
        
        # 상위 키워드를 토픽으로 사용
        topics = []
        for i, kw in enumerate(keywords[:n_topics]):
            topics.append({
                'topic_id': i,
                'topic': kw['keyword'],
                'count': kw['count'],
                'keywords': [kw['keyword']]
            })
        
        return topics
    
    def cluster_keywords(self, keywords: List[Dict], n_clusters: int = 5) -> List[Dict]:
        """
        키워드 클러스터링
        
        Args:
            keywords: 키워드 리스트
            n_clusters: 클러스터 수
            
        Returns:
            List[Dict]: 클러스터 리스트
        """
        if not keywords or len(keywords) < n_clusters:
            return []
        
        try:
            # 키워드를 텍스트로 변환
            keyword_texts = [kw['keyword'] for kw in keywords]
            
            # TF-IDF 벡터화
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words=None,
                ngram_range=(1, 1)
            )
            
            tfidf_matrix = vectorizer.fit_transform(keyword_texts)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=min(n_clusters, len(keywords)), random_state=42)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # 클러스터별 키워드 그룹화
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(keywords[i])
            
            # 클러스터 결과 정리
            cluster_results = []
            for cluster_id, cluster_keywords in clusters.items():
                cluster_results.append({
                    'cluster_id': cluster_id,
                    'keywords': cluster_keywords,
                    'size': len(cluster_keywords)
                })
            
            return cluster_results
            
        except Exception as e:
            print(f"키워드 클러스터링 오류: {e}")
            return []
    
    def compare_topics(self, topics1: List[str], topics2: List[str]) -> Dict[str, Any]:
        """
        두 토픽 집합 비교
        
        Args:
            topics1: 첫 번째 토픽 리스트
            topics2: 두 번째 토픽 리스트
            
        Returns:
            Dict: 비교 결과
        """
        set1 = set(topics1)
        set2 = set(topics2)
        
        common = set1 & set2
        only1 = set1 - set2
        only2 = set2 - set1
        
        return {
            'common_topics': list(common),
            'only_in_first': list(only1),
            'only_in_second': list(only2),
            'common_count': len(common),
            'first_only_count': len(only1),
            'second_only_count': len(only2),
            'similarity': len(common) / len(set1 | set2) if set1 | set2 else 0
        }
    
    def top_topics(self, corpus: List[str], k: int = 10) -> List[str]:
        """
        TF-IDF 기반 상위 토픽 추출 (쓰레기 단어 제거 + 품질 가드 + 디버깅)
        
        Args:
            corpus: 텍스트 코퍼스 (text_clean 리스트)
            k: 상위 k개 토픽
            
        Returns:
            List[str]: 상위 토픽 리스트
            
        Raises:
            ValueError: 품질 검증 실패
        """
        from src.common.trace import snapshot_df, log_shape
        
        if not corpus:
            print("❌ 토픽 추출: 빈 코퍼스")
            return []
        
        print(f"🔍 토픽 추출 시작: {len(corpus)}개 문서, {k}개 토픽 추출")
        
        # 원본 데이터 디버깅
        print("📊 원본 데이터 샘플:")
        for i, text in enumerate(corpus[:3]):
            print(f"  {i+1}: {text[:100]}...")
        
        # 통합된 불용어 세트
        combined_stopwords = STOPWORDS | GARBAGE_TOKENS
        print(f"🚫 불용어 세트 크기: {len(combined_stopwords)}")
        
        # TF-IDF 벡터라이저 설정 (동적 조정)
        corpus_size = len(corpus)
        min_df = min(3, max(1, corpus_size // 10))  # 동적 min_df
        max_df = 0.8  # 완화된 max_df
        
        print(f"⚙️ 벡터라이저 설정: min_df={min_df}, max_df={max_df}")
        
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  # 1-gram, 2-gram
            min_df=min_df,       # 동적 min_df
            max_df=max_df,       # 완화된 max_df
            stop_words=list(combined_stopwords),
            token_pattern=r'(?u)\b[가-힣a-zA-Z]{2,}\b'  # 숫자/한글자 제외
        )
        
        try:
            # TF-IDF 행렬 생성
            tfidf_matrix = vectorizer.fit_transform(corpus)
            feature_names = vectorizer.get_feature_names_out()
            
            print(f"📈 TF-IDF 행렬 생성: {tfidf_matrix.shape}")
            print(f"🔤 추출된 특성 수: {len(feature_names)}")
            
            if len(feature_names) == 0:
                print("❌ 토픽 추출: 유효한 특성이 없습니다")
                # 원본 텍스트에서 직접 키워드 추출 시도
                return self._fallback_keyword_extraction(corpus, k)
            
            # 평균 TF-IDF 점수 계산
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # 상위 k개 토픽 선택 (더 많은 후보 확보)
            top_indices = np.argsort(mean_scores)[::-1][:k*3]  # 여유분 확보
            top_topics_list = [feature_names[i] for i in top_indices if mean_scores[i] > 0]
            
            print(f"🎯 상위 후보 토픽: {len(top_topics_list)}개")
            print(f"📋 후보 토픽 샘플: {top_topics_list[:10]}")
            
            # 쓰레기 토큰 필터링 (품질 가드) - 더 엄격하게
            filtered_topics = []
            rejected_topics = []
            
            for topic in top_topics_list:
                # 쓰레기 토큰이 포함되지 않은 경우만 추가
                if not any(garbage in topic.lower() for garbage in GARBAGE_TOKENS):
                    # 추가 품질 검증
                    if (len(topic) > 1 and 
                        not topic.isdigit() and 
                        not topic.lower() in ['ai', 'rss', 'xml', 'json'] and
                        not any(char.isdigit() for char in topic)):
                        filtered_topics.append(topic)
                    else:
                        rejected_topics.append(topic)
                else:
                    rejected_topics.append(topic)
            
            print(f"✅ 필터링된 토픽: {len(filtered_topics)}개")
            print(f"❌ 거부된 토픽: {len(rejected_topics)}개")
            if rejected_topics:
                print(f"🚫 거부된 토픽 샘플: {rejected_topics[:10]}")
            
            # 최종 품질 검증
            final_topics = filtered_topics[:k]
            
            # 금지 토큰 재검증 (더 강화된 버전)
            forbidden_tokens = [
                'nbsp', 'font', 'href', 'https', 'com', 'google', 'news', 'rss', 'xml', 'json', 'api',
                'articles', 'target', 'oc', 'feed', 'atom', 'syndication', 'channel', 'item',
                'link', 'description', 'pubdate', 'guid', 'category', 'enclosure', 'www', 'http'
            ]
            final_filtered_topics = []
            
            for topic in final_topics:
                # 토큰별로 분리해서 검사
                topic_tokens = topic.lower().split()
                has_forbidden = False
                
                for token in topic_tokens:
                    if token in forbidden_tokens:
                        print(f"⚠️ 최종 검증에서 거부: '{topic}' (금지 토큰: '{token}')")
                        has_forbidden = True
                        break
                
                if not has_forbidden:
                    final_filtered_topics.append(topic)
            
            # 결과가 부족하면 fallback 사용
            if len(final_filtered_topics) < k // 2:
                print("⚠️ 품질 높은 토픽이 부족하여 fallback 방식 사용")
                return self._fallback_keyword_extraction(corpus, k)
            
            # 결과 저장
            topics_df = pd.DataFrame({"topic": final_filtered_topics})
            snapshot_df(topics_df, "topics_top10")
            
            print(f"✅ 토픽 추출 완료: {len(final_filtered_topics)}개")
            print(f"🎯 최종 토픽: {final_filtered_topics}")
            return final_filtered_topics
            
        except Exception as e:
            print(f"❌ 토픽 추출 실패: {e}")
            print("🔄 Fallback 방식으로 재시도")
            return self._fallback_keyword_extraction(corpus, k)
    
    def _fallback_keyword_extraction(self, corpus: List[str], k: int) -> List[str]:
        """
        Fallback 키워드 추출 방식 (TF-IDF 실패 시)
        
        Args:
            corpus: 텍스트 코퍼스
            k: 추출할 키워드 수
            
        Returns:
            List[str]: 키워드 리스트
        """
        print("🔄 Fallback 키워드 추출 시작")
        
        try:
            # 간단한 빈도 기반 키워드 추출
            all_words = []
            for text in corpus:
                if text:
                    # 토큰화 및 필터링
                    tokens = text.lower().split()
                    filtered_tokens = [
                        token for token in tokens 
                        if (len(token) > 1 and 
                            token not in STOPWORDS and 
                            token not in GARBAGE_TOKENS and
                            not token.isdigit() and
                            not any(char.isdigit() for char in token))
                    ]
                    all_words.extend(filtered_tokens)
            
            if not all_words:
                print("❌ Fallback: 유효한 단어가 없습니다")
                return []
            
            # 빈도 계산
            from collections import Counter
            word_freq = Counter(all_words)
            
            # 상위 키워드 선택
            top_keywords = [word for word, count in word_freq.most_common(k*2)]
            
            # 최종 필터링
            final_keywords = []
            for keyword in top_keywords:
                if (len(keyword) > 1 and 
                    keyword not in ['ai', 'rss', 'xml', 'json', 'api', 'http', 'www'] and
                    not any(char.isdigit() for char in keyword)):
                    final_keywords.append(keyword)
            
            result = final_keywords[:k]
            print(f"✅ Fallback 완료: {len(result)}개 키워드")
            print(f"🎯 Fallback 결과: {result}")
            return result
            
        except Exception as e:
            print(f"❌ Fallback 실패: {e}")
            return []
    
    def doc_topics(self, text: str, n: int = 5) -> List[str]:
        """
        단일 문서의 주요 토픽 추출
        
        Args:
            text: 분석할 텍스트
            n: 상위 n개 토픽
            
        Returns:
            List[str]: 주요 토픽 리스트
        """
        if not text:
            return []
        
        # 간단한 토큰화 및 빈도 분석
        tokens = text.lower().split()
        
        # 불용어 및 쓰레기 토큰 제거
        combined_stopwords = STOPWORDS | GARBAGE_TOKENS
        filtered_tokens = [
            token for token in tokens 
            if len(token) > 1 and token not in combined_stopwords
        ]
        
        if not filtered_tokens:
            return []
        
        # 빈도 계산
        from collections import Counter
        token_counts = Counter(filtered_tokens)
        
        # 상위 n개 반환
        return [token for token, count in token_counts.most_common(n)]
