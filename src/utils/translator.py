"""
번역 유틸리티 모듈
"""
from googletrans import Translator
from typing import List, Dict

class KeywordTranslator:
    """키워드 번역 클래스"""
    
    def __init__(self):
        self.translator = Translator()
    
    def translate_keywords(self, keywords: List[str], target_lang: str = 'en') -> List[str]:
        """
        키워드 리스트를 대상 언어로 번역
        
        Args:
            keywords: 번역할 키워드 리스트
            target_lang: 대상 언어 코드 ('en': 영어, 'ko': 한국어)
            
        Returns:
            List[str]: 번역된 키워드 리스트
        """
        if not keywords:
            return []
        
        try:
            # 키워드들을 하나의 문자열로 결합
            combined_text = ', '.join(keywords)
            
            # 번역 실행
            result = self.translator.translate(combined_text, dest=target_lang)
            
            # 번역된 텍스트를 다시 리스트로 분할
            translated_keywords = [kw.strip() for kw in result.text.split(',')]
            
            return translated_keywords
            
        except Exception as e:
            print(f"번역 오류: {e}")
            # 번역 실패 시 원본 키워드 반환
            return keywords
    
    def translate_to_english(self, keywords: List[str]) -> List[str]:
        """키워드를 영어로 번역"""
        return self.translate_keywords(keywords, 'en')
    
    def translate_to_korean(self, keywords: List[str]) -> List[str]:
        """키워드를 한국어로 번역"""
        return self.translate_keywords(keywords, 'ko')
    
    def detect_language(self, text: str) -> str:
        """텍스트의 언어 감지"""
        try:
            result = self.translator.detect(text)
            return result.lang
        except Exception as e:
            print(f"언어 감지 오류: {e}")
            return 'unknown'
    
    def translate_keywords_bidirectional(self, keywords: List[str]) -> Dict[str, List[str]]:
        """
        키워드를 영어와 한국어로 양방향 번역
        
        Args:
            keywords: 원본 키워드 리스트
            
        Returns:
            Dict: {'original': 원본, 'english': 영어번역, 'korean': 한국어번역}
        """
        # 언어 감지
        sample_text = keywords[0] if keywords else ""
        detected_lang = self.detect_language(sample_text)
        
        result = {
            'original': keywords,
            'english': keywords,
            'korean': keywords
        }
        
        try:
            if detected_lang == 'ko':
                # 한국어 -> 영어
                result['english'] = self.translate_to_english(keywords)
            elif detected_lang == 'en':
                # 영어 -> 한국어
                result['korean'] = self.translate_to_korean(keywords)
            else:
                # 기타 언어 -> 영어, 한국어
                result['english'] = self.translate_to_english(keywords)
                result['korean'] = self.translate_to_korean(keywords)
                
        except Exception as e:
            print(f"양방향 번역 오류: {e}")
        
        return result
