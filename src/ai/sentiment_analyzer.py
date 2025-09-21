"""
감성 분석 모듈
"""
from typing import List, Dict
import re
from collections import Counter
import requests
import json

# 영어 자연어처리 라이브러리
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

class SentimentAnalyzer:
    """감성 분석 클래스"""
    
    def __init__(self, huggingface_api_key: str = None):
        self.huggingface_api_key = huggingface_api_key
        
        # 영어 자연어처리 초기화
        self._initialize_english_nlp()
        
        # 긍정/부정 단어 사전 (확장)
        self.positive_words = {
            '좋다', '훌륭하다', '훌륭한', '좋은', '좋아', '좋아하는', '좋아요',
            '성공', '성공적', '성공하다', '성공한', '성공적인',
            '개선', '개선되다', '개선된', '개선하다',
            '향상', '향상되다', '향상된', '향상하다',
            '증가', '증가하다', '증가한', '증가하다',
            '상승', '상승하다', '상승한', '상승하다',
            '긍정적', '긍정', '긍정적이다',
            '유리한', '유리하다', '유리',
            '장점', '장점이', '장점이다',
            '혜택', '혜택이', '혜택이다',
            '기대', '기대하다', '기대되는', '기대된다',
            '희망', '희망적', '희망이다',
            '낙관', '낙관적', '낙관적이다',
            '효과적', '효과', '효과적이다',
            '유용', '유용하다', '유용한',
            '편리', '편리하다', '편리한',
            '쉽다', '쉬운', '쉽게',
            '빠르다', '빠른', '빠르게',
            '정확', '정확하다', '정확한',
            '안전', '안전하다', '안전한',
            '신뢰', '신뢰하다', '신뢰할',
            '만족', '만족하다', '만족스러운',
            '높다', '높은', '높게',
            '크다', '큰', '크게',
            '많다', '많은', '많이',
            '강하다', '강한', '강하게',
            '뛰어나다', '뛰어난', '뛰어나게',
            '우수', '우수하다', '우수한',
            '최고', '최고다', '최고의',
            '최적', '최적화', '최적화하다',
            '혁신', '혁신적', '혁신하다',
            '발전', '발전하다', '발전된',
            '성장', '성장하다', '성장한',
            '진보', '진보하다', '진보한',
            '개발', '개발하다', '개발된',
            '도움', '도움이', '도움이 된다',
            '효과', '효과가', '효과가 있다',
            '결과', '좋은 결과', '성공적인 결과',
            '성과', '성과가', '성과가 있다',
            '이익', '이익이', '이익이 있다',
            '가치', '가치가', '가치가 있다',
            '중요', '중요하다', '중요한',
            '필요', '필요하다', '필요한',
            '필수', '필수적', '필수적이다',
            '핵심', '핵심적', '핵심적이다',
            '주요', '주요하다', '주요한',
            '핵심', '핵심적', '핵심적이다',
            '주요', '주요하다', '주요한',
            '핵심', '핵심적', '핵심적이다',
            '주요', '주요하다', '주요한'
        }
        
        self.negative_words = {
            '나쁘다', '나쁜', '나쁘다', '나쁘다',
            '문제', '문제가', '문제점', '문제이다',
            '위험', '위험한', '위험하다', '위험하다',
            '부정적', '부정', '부정적이다',
            '불리한', '불리', '불리하다',
            '단점', '단점이', '단점이다',
            '손실', '손실이', '손실이다',
            '감소', '감소하다', '감소한', '감소했다',
            '하락', '하락하다', '하락한', '하락했다',
            '우려', '우려하다', '우려되는', '우려된다',
            '걱정', '걱정하다', '걱정되는', '걱정된다',
            '비관', '비관적', '비관적이다',
            '실망', '실망하다', '실망스러운', '실망스럽다',
            '어렵다', '어려운', '어렵게',
            '복잡', '복잡하다', '복잡한',
            '불편', '불편하다', '불편한',
            '느리다', '느린', '느리게',
            '부족', '부족하다', '부족한',
            '낮다', '낮은', '낮게',
            '작다', '작은', '작게',
            '적다', '적은', '적게',
            '약하다', '약한', '약하게',
            '나쁘다', '나쁜', '나쁘다',
            '최악', '최악이다', '최악의',
            '실패', '실패하다', '실패한',
            '실패적', '실패적이다',
            '문제', '문제가', '문제점',
            '위험', '위험한', '위험하다',
            '해롭다', '해로운', '해롭게',
            '피해', '피해가', '피해이다',
            '손해', '손해가', '손해이다',
            '비효율', '비효율적', '비효율적이다',
            '비효과', '비효과적', '비효과적이다',
            '무용', '무용하다', '무용한',
            '불필요', '불필요하다', '불필요한',
            '불가능', '불가능하다', '불가능한',
            '불가', '불가하다', '불가한',
            '불가능', '불가능하다', '불가능한',
            '불가', '불가하다', '불가한',
            '불가능', '불가능하다', '불가능한',
            '불가', '불가하다', '불가한',
            '불가능', '불가능하다', '불가능한',
            '불가', '불가하다', '불가한'
        }
    
    def _initialize_english_nlp(self):
        """영어 자연어처리 도구 초기화"""
        self.nlp = None
        self.vader_analyzer = None
        self.english_stopwords = set()
        
        # spaCy 초기화
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                print("✅ spaCy 영어 모델 로드 완료")
            except Exception as e:
                print(f"⚠️ spaCy 영어 모델 로드 실패: {e}")
        
        # VADER 감성 분석기 초기화
        if VADER_AVAILABLE:
            try:
                self.vader_analyzer = VaderAnalyzer()
                print("✅ VADER 감성 분석기 초기화 완료")
            except Exception as e:
                print(f"⚠️ VADER 감성 분석기 초기화 실패: {e}")
        
        # NLTK 불용어 초기화
        if NLTK_AVAILABLE:
            try:
                self.english_stopwords = set(stopwords.words('english'))
                print("✅ NLTK 영어 불용어 로드 완료")
            except Exception as e:
                print(f"⚠️ NLTK 불용어 로드 실패: {e}")
    
    def analyze_english_sentiment(self, text: str) -> Dict[str, float]:
        """
        영어 텍스트 감성 분석 (다중 방법 결합)
        
        Args:
            text: 분석할 영어 텍스트
            
        Returns:
            Dict: 감성 분석 결과 {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
        """
        if not text:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 텍스트 전처리
        text = self._preprocess_english_text(text)
        
        # 다중 방법으로 감성 분석
        results = []
        
        # 1. VADER 감성 분석
        if self.vader_analyzer:
            vader_result = self._analyze_with_vader(text)
            if vader_result:
                results.append(vader_result)
        
        # 2. TextBlob 감성 분석
        if TEXTBLOB_AVAILABLE:
            textblob_result = self._analyze_with_textblob(text)
            if textblob_result:
                results.append(textblob_result)
        
        # 3. spaCy 기반 감성 분석
        if self.nlp:
            spacy_result = self._analyze_with_spacy(text)
            if spacy_result:
                results.append(spacy_result)
        
        # 결과 결합
        if results:
            return self._combine_sentiment_results(results)
        else:
            # 모든 방법 실패 시 기본값
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
    
    def _preprocess_english_text(self, text: str) -> str:
        """영어 텍스트 전처리"""
        # HTML 태그 제거
        text = re.sub(r'<[^>]+>', '', text)
        # 특수문자 정리
        text = re.sub(r'[^\w\s]', ' ', text)
        # 공백 정리
        text = ' '.join(text.split())
        return text.lower()
    
    def _analyze_with_vader(self, text: str) -> Dict[str, float]:
        """VADER 감성 분석"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            return {
                'positive': scores['pos'],
                'neutral': scores['neu'],
                'negative': scores['neg']
            }
        except Exception as e:
            print(f"VADER 분석 오류: {e}")
            return None
    
    def _analyze_with_textblob(self, text: str) -> Dict[str, float]:
        """TextBlob 감성 분석"""
        try:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # polarity를 positive/neutral/negative로 변환
            if polarity > 0.1:
                return {'positive': abs(polarity), 'neutral': 1 - abs(polarity), 'negative': 0.0}
            elif polarity < -0.1:
                return {'positive': 0.0, 'neutral': 1 - abs(polarity), 'negative': abs(polarity)}
            else:
                return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        except Exception as e:
            print(f"TextBlob 분석 오류: {e}")
            return None
    
    def _analyze_with_spacy(self, text: str) -> Dict[str, float]:
        """spaCy 기반 감성 분석"""
        try:
            doc = self.nlp(text)
            
            # 감정 단어 사전 기반 분석
            positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'brilliant', 'outstanding', 'superb', 'outstanding']
            negative_words = ['bad', 'terrible', 'awful', 'horrible', 'disappointing', 'poor', 'worst', 'negative', 'problem', 'issue']
            
            positive_count = 0
            negative_count = 0
            total_words = 0
            
            for token in doc:
                if not token.is_stop and not token.is_punct and token.is_alpha:
                    total_words += 1
                    if token.lemma_.lower() in positive_words:
                        positive_count += 1
                    elif token.lemma_.lower() in negative_words:
                        negative_count += 1
            
            if total_words == 0:
                return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            
            positive_score = positive_count / total_words
            negative_score = negative_count / total_words
            neutral_score = 1.0 - positive_score - negative_score
            
            return {
                'positive': positive_score,
                'neutral': max(0.0, neutral_score),
                'negative': negative_score
            }
        except Exception as e:
            print(f"spaCy 분석 오류: {e}")
            return None
    
    def _combine_sentiment_results(self, results: List[Dict[str, float]]) -> Dict[str, float]:
        """여러 감성 분석 결과 결합"""
        if not results:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 가중 평균 계산
        total_positive = sum(r['positive'] for r in results)
        total_neutral = sum(r['neutral'] for r in results)
        total_negative = sum(r['negative'] for r in results)
        
        count = len(results)
        
        return {
            'positive': total_positive / count,
            'neutral': total_neutral / count,
            'negative': total_negative / count
        }
    
    def analyze_text_sentiment(self, text: str, gemini_analyzer=None) -> Dict[str, float]:
        """
        단일 텍스트 감성 분석 (언어 자동 감지)
        
        Args:
            text: 분석할 텍스트
            gemini_analyzer: Gemini 분석기 객체 (선택사항)
            
        Returns:
            Dict: 감성 분석 결과 {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
        """
        if not text:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 언어 감지 (간단한 휴리스틱)
        is_english = self._is_english_text(text)
        
        if is_english:
            # 영어 텍스트: 영어 전용 감성 분석 사용
            return self.analyze_english_sentiment(text)
        else:
            # 한국어 텍스트: 기존 방법 사용
            if gemini_analyzer:
                return self.analyze_sentiment_combined(text, gemini_analyzer)
            else:
                return self._analyze_sentiment_word_based(text)
    
    def _is_english_text(self, text: str) -> bool:
        """텍스트가 영어인지 간단히 판단"""
        if not text:
            return False
        
        # 영어 문자 비율 계산
        english_chars = sum(1 for c in text if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in text if c.isalpha())
        
        if total_chars == 0:
            return False
        
        english_ratio = english_chars / total_chars
        return english_ratio > 0.7  # 70% 이상이 영어 문자면 영어로 판단
    
    def _analyze_sentiment_word_based(self, text: str) -> Dict[str, float]:
        """
        단어 기반 감성 분석 (fallback)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict: 감성 분석 결과
        """
        # 텍스트 전처리
        text = re.sub(r'[^\w\s]', '', text.lower())
        words = text.split()
        
        if not words:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 긍정/부정 단어 카운트
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        total_words = len(words)
        
        # 감성 점수 계산
        positive_score = positive_count / total_words
        negative_score = negative_count / total_words
        neutral_score = 1.0 - positive_score - negative_score
        
        # 중립 점수가 음수가 되면 조정
        if neutral_score < 0:
            total = positive_score + negative_score
            if total > 0:
                positive_score = positive_score / total
                negative_score = negative_score / total
                neutral_score = 0.0
            else:
                neutral_score = 1.0
                positive_score = 0.0
                negative_score = 0.0
        
        return {
            'positive': positive_score,
            'neutral': neutral_score,
            'negative': negative_score
        }
    
    def analyze_batch_sentiment(self, texts, gemini_analyzer=None) -> Dict[str, float]:
        """
        여러 텍스트의 감성 분석 (Gemini + Hugging Face 결합)
        
        Args:
            texts: 분석할 텍스트 리스트 또는 딕셔너리 리스트
            gemini_analyzer: Gemini 분석기 객체 (선택사항)
            
        Returns:
            Dict: 전체 감성 분석 결과
        """
        if not texts:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
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
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        # 각 텍스트의 감성 분석
        sentiments = [self.analyze_text_sentiment(text, gemini_analyzer) for text in text_list]
        
        # 평균 계산
        total_positive = sum(s['positive'] for s in sentiments)
        total_neutral = sum(s['neutral'] for s in sentiments)
        total_negative = sum(s['negative'] for s in sentiments)
        
        count = len(sentiments)
        
        return {
            'positive': total_positive / count,
            'neutral': total_neutral / count,
            'negative': total_negative / count
        }
    
    def get_sentiment_trend(self, texts_with_dates: List[Dict]) -> List[Dict]:
        """
        시간별 감성 추이 분석
        
        Args:
            texts_with_dates: 날짜가 포함된 텍스트 리스트 [{'text': '...', 'date': '2024-01-01'}, ...]
            
        Returns:
            List[Dict]: 날짜별 감성 분석 결과
        """
        if not texts_with_dates:
            return []
        
        # 날짜별로 그룹화
        from collections import defaultdict
        date_groups = defaultdict(list)
        
        for item in texts_with_dates:
            date = item.get('date', '')
            text = item.get('text', '')
            if date and text:
                date_groups[date].append(text)
        
        # 각 날짜별 감성 분석
        sentiment_trend = []
        for date, texts in date_groups.items():
            sentiment = self.analyze_batch_sentiment(texts)
            sentiment_trend.append({
                'date': date,
                'sentiment': sentiment,
                'text_count': len(texts)
            })
        
        # 날짜순 정렬
        sentiment_trend.sort(key=lambda x: x['date'])
        
        return sentiment_trend
    
    def analyze_with_huggingface(self, text: str) -> Dict[str, float]:
        """
        Hugging Face API를 사용한 감성 분석
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            Dict: 감성 분석 결과 {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
        """
        if not self.huggingface_api_key:
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
        
        try:
            # Hugging Face API 호출
            API_URL = "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment"
            headers = {"Authorization": f"Bearer {self.huggingface_api_key}"}
            
            # 텍스트 길이 제한 (Hugging Face API 제한)
            if len(text) > 512:
                text = text[:512]
            
            response = requests.post(API_URL, headers=headers, json={"inputs": text})
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    # Hugging Face 결과를 positive/neutral/negative로 변환
                    labels = result[0]
                    
                    # 라벨 매핑: 1-2: negative, 3: neutral, 4-5: positive
                    negative_score = 0.0
                    neutral_score = 0.0
                    positive_score = 0.0
                    
                    for label in labels:
                        if label['label'] in ['1 star', '2 stars']:
                            negative_score += label['score']
                        elif label['label'] == '3 stars':
                            neutral_score += label['score']
                        elif label['label'] in ['4 stars', '5 stars']:
                            positive_score += label['score']
                    
                    # 정규화
                    total = negative_score + neutral_score + positive_score
                    if total > 0:
                        return {
                            'positive': positive_score / total,
                            'neutral': neutral_score / total,
                            'negative': negative_score / total
                        }
            
            # API 호출 실패 시 기본값 반환
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
            
        except Exception as e:
            print(f"Hugging Face 감성 분석 오류: {e}")
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
    
    def analyze_with_gemini(self, text: str, gemini_analyzer) -> Dict[str, float]:
        """
        Gemini를 사용한 감성 분석
        
        Args:
            text: 분석할 텍스트
            gemini_analyzer: Gemini 분석기 객체
            
        Returns:
            Dict: 감성 분석 결과
        """
        try:
            return gemini_analyzer.analyze_sentiment([text])
        except Exception as e:
            print(f"Gemini 감성 분석 오류: {e}")
            return {'positive': 0.0, 'neutral': 1.0, 'negative': 0.0}
    
    def analyze_sentiment_combined(self, text: str, gemini_analyzer) -> Dict[str, float]:
        """
        Gemini와 Hugging Face를 결합한 감성 분석
        
        Args:
            text: 분석할 텍스트
            gemini_analyzer: Gemini 분석기 객체
            
        Returns:
            Dict: 감성 분석 결과
        """
        # Gemini 감성 분석
        gemini_result = self.analyze_with_gemini(text, gemini_analyzer)
        
        # Hugging Face 감성 분석
        hf_result = self.analyze_with_huggingface(text)
        
        # 보정 로직: 결과가 다르면 HF 결과를 우선 사용
        if self._sentiment_results_similar(gemini_result, hf_result):
            return gemini_result
        else:
            print("감성 분석 결과 차이 감지 - Hugging Face 결과 사용")
            return hf_result
    
    def _sentiment_results_similar(self, result1: Dict[str, float], result2: Dict[str, float], threshold: float = 0.2) -> bool:
        """
        두 감성 분석 결과가 유사한지 확인
        
        Args:
            result1: 첫 번째 감성 분석 결과
            result2: 두 번째 감성 분석 결과
            threshold: 유사도 임계값
            
        Returns:
            bool: 유사한지 여부
        """
        for key in ['positive', 'neutral', 'negative']:
            if abs(result1.get(key, 0) - result2.get(key, 0)) > threshold:
                return False
        return True
