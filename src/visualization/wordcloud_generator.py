"""
워드클라우드 생성 모듈
"""
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from typing import List, Dict
import numpy as np

class WordCloudGenerator:
    """워드클라우드 생성 클래스"""
    
    def __init__(self):
        # 한글 폰트 설정 (Windows 환경)
        import os
        if os.path.exists('C:/Windows/Fonts/malgun.ttf'):
            self.font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
        elif os.path.exists('C:/Windows/Fonts/gulim.ttc'):
            self.font_path = 'C:/Windows/Fonts/gulim.ttc'  # 굴림
        else:
            self.font_path = None  # 기본 폰트 사용
        
    def generate_wordcloud(self, text_data: List[str], max_words: int = 100) -> WordCloud:
        """
        워드클라우드 생성
        
        Args:
            text_data: 텍스트 데이터 리스트
            max_words: 최대 단어 수
            
        Returns:
            WordCloud: 워드클라우드 객체
        """
        # 텍스트 결합
        combined_text = ' '.join(text_data)
        
        # 워드클라우드 생성
        wordcloud_params = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'max_words': max_words,
            'colormap': 'viridis',
            'relative_scaling': 0.5,
            'random_state': 42
        }
        
        # 폰트가 있는 경우에만 추가
        if self.font_path:
            wordcloud_params['font_path'] = self.font_path
        
        wordcloud = WordCloud(**wordcloud_params).generate(combined_text)
        
        return wordcloud
    
    def generate_from_frequency(self, word_freq: Dict[str, int], max_words: int = 100) -> WordCloud:
        """
        단어 빈도 데이터로부터 워드클라우드 생성
        
        Args:
            word_freq: 단어 빈도 딕셔너리 {'단어': 빈도}
            max_words: 최대 단어 수
            
        Returns:
            WordCloud: 워드클라우드 객체
        """
        wordcloud_params = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'max_words': max_words,
            'colormap': 'viridis',
            'relative_scaling': 0.5,
            'random_state': 42
        }
        
        # 폰트가 있는 경우에만 추가
        if self.font_path:
            wordcloud_params['font_path'] = self.font_path
        
        wordcloud = WordCloud(**wordcloud_params).generate_from_frequencies(word_freq)
        
        return wordcloud
    
    def create_wordcloud_figure(self, wordcloud: WordCloud) -> plt.Figure:
        """
        워드클라우드를 matplotlib Figure로 변환
        
        Args:
            wordcloud: 워드클라우드 객체
            
        Returns:
            plt.Figure: matplotlib Figure 객체
        """
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title('주요 키워드 워드클라우드', fontsize=16, pad=20)
        
        return fig
    
    def generate_topic_wordcloud(self, topics: List[Dict]) -> WordCloud:
        """
        토픽 데이터로부터 워드클라우드 생성
        
        Args:
            topics: 토픽 데이터 [{'topic': 'AI', 'count': 50}, ...]
            
        Returns:
            WordCloud: 워드클라우드 객체
        """
        # 토픽을 빈도에 따라 반복하여 텍스트 생성
        text_parts = []
        for topic in topics:
            topic_name = topic['topic']
            count = topic['count']
            # 빈도에 따라 단어 반복
            text_parts.extend([topic_name] * count)
        
        combined_text = ' '.join(text_parts)
        
        wordcloud_params = {
            'width': 800,
            'height': 400,
            'background_color': 'white',
            'max_words': 50,
            'colormap': 'plasma',
            'relative_scaling': 0.5,
            'random_state': 42
        }
        
        # 폰트가 있는 경우에만 추가
        if self.font_path:
            wordcloud_params['font_path'] = self.font_path
        
        wordcloud = WordCloud(**wordcloud_params).generate(combined_text)
        
        return wordcloud
