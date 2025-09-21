"""
차트 생성 모듈
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import List, Dict, Any
import numpy as np

class ChartGenerator:
    """차트 생성 클래스"""
    
    def __init__(self):
        pass
    
    def create_search_volume_chart(self, data: pd.DataFrame, keywords: List[str]) -> go.Figure:
        """
        검색량 추이 차트 생성
        
        Args:
            data: 검색량 데이터 (pandas DataFrame)
            keywords: 키워드 리스트
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        fig = go.Figure()
        
        for keyword in keywords:
            if keyword in data.columns:
                fig.add_trace(go.Scatter(
                    x=data.index,
                    y=data[keyword],
                    mode='lines+markers',
                    name=keyword,
                    line=dict(width=2),
                    marker=dict(size=4)
                ))
        
        fig.update_layout(
            title='검색량 추이',
            xaxis_title='날짜',
            yaxis_title='검색량',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_related_keywords_chart(self, related_data: List[Dict]) -> go.Figure:
        """
        연관 검색어 차트 생성
        
        Args:
            related_data: 연관 검색어 데이터
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not related_data:
            return go.Figure()
        
        # 키워드별로 그룹화
        keyword_groups = {}
        for item in related_data:
            keyword = item.get('keyword', 'Unknown')
            if keyword not in keyword_groups:
                keyword_groups[keyword] = []
            keyword_groups[keyword].append(item)
        
        # 각 키워드별로 상위 5개씩 선택
        fig = go.Figure()
        colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow', 'lightpink']
        
        for i, (keyword, items) in enumerate(keyword_groups.items()):
            # 상위 5개만 선택
            top_items = sorted(items, key=lambda x: x['value'], reverse=True)[:5]
            
            keywords = [item['related'] for item in top_items]
            values = [item['value'] for item in top_items]
            
            fig.add_trace(go.Bar(
                x=values,
                y=keywords,
                orientation='h',
                name=keyword,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{keyword}</b><br>%{{y}}: %{{x}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='키워드별 연관 검색어 Top 5',
            xaxis_title='관련도',
            yaxis_title='검색어',
            template='plotly_white',
            barmode='group'
        )
        
        return fig
    
    def create_news_count_chart(self, news_data: Dict[str, List[Dict]]) -> go.Figure:
        """
        뉴스 건수 추이 차트 생성
        
        Args:
            news_data: 뉴스 데이터 (소스별)
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        fig = go.Figure()
        
        for source, articles in news_data.items():
            if articles is None or (hasattr(articles, 'empty') and articles.empty) or len(articles) == 0:
                continue
                
            # 날짜별 건수 계산
            df = pd.DataFrame(articles)
            if 'pub_date' in df.columns:
                df['date'] = pd.to_datetime(df['pub_date'], errors='coerce')
                daily_counts = df.groupby(df['date'].dt.date).size()
                
                fig.add_trace(go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    mode='lines+markers',
                    name=source,
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='뉴스/블로그 건수 추이',
            xaxis_title='날짜',
            yaxis_title='건수',
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_sentiment_chart(self, sentiment_data: Dict[str, float]) -> go.Figure:
        """
        감성 분석 차트 생성
        
        Args:
            sentiment_data: 감성 분석 결과 {'positive': 0.3, 'neutral': 0.5, 'negative': 0.2}
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        labels = list(sentiment_data.keys())
        values = list(sentiment_data.values())
        colors = ['green', 'gray', 'red']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            marker_colors=colors,
            hole=0.3
        )])
        
        fig.update_layout(
            title='감성 분석 결과',
            template='plotly_white'
        )
        
        return fig
    
    def create_topic_frequency_chart(self, topics: List[Dict]) -> go.Figure:
        """
        토픽 빈도 차트 생성
        
        Args:
            topics: 토픽 데이터 [{'topic': 'AI', 'count': 50}, ...]
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not topics:
            return go.Figure()
        
        # Top 10 토픽만 선택
        top_topics = sorted(topics, key=lambda x: x['count'], reverse=True)[:10]
        
        topic_names = [t['topic'] for t in top_topics]
        counts = [t['count'] for t in top_topics]
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_names,
                y=counts,
                marker_color='lightcoral'
            )
        ])
        
        fig.update_layout(
            title='주요 토픽 빈도',
            xaxis_title='토픽',
            yaxis_title='빈도',
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_topic_cluster_map(self, cluster_data: List[Dict], size_scale: float = 1.0) -> go.Figure:
        """
        토픽 클러스터 맵 생성 (버블 차트 크기 최적화)
        
        Args:
            cluster_data: 클러스터 데이터 [{'topic': 'AI', 'frequency': 50, 'sentiment': 0.7, 'growth': 0.3}, ...]
            size_scale: 크기 스케일 (0.5~2.0)
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not cluster_data:
            return go.Figure()
        
        topics = [d['topic'] for d in cluster_data]
        frequencies = [d['frequency'] for d in cluster_data]
        sentiments = [d['sentiment'] for d in cluster_data]
        growths = [d['growth'] for d in cluster_data]
        
        # 색상 매핑 (감성에 따라)
        colors = []
        for sentiment in sentiments:
            if sentiment > 0.6:
                colors.append('green')
            elif sentiment < 0.4:
                colors.append('red')
            else:
                colors.append('gray')
        
        # 버블 크기 계산 (반지름에 비례, 8px~40px 범위)
        import math
        min_size, max_size = 8, 40
        if frequencies:
            min_freq, max_freq = min(frequencies), max(frequencies)
            if max_freq > min_freq:
                # 선형 스케일링 (sqrt 적용)
                sizes = []
                for freq in frequencies:
                    # sqrt 적용하여 면적이 아닌 반지름에 비례
                    sqrt_freq = math.sqrt(freq)
                    sqrt_min = math.sqrt(min_freq)
                    sqrt_max = math.sqrt(max_freq)
                    
                    if sqrt_max > sqrt_min:
                        # 정규화 후 크기 범위에 매핑
                        normalized = (sqrt_freq - sqrt_min) / (sqrt_max - sqrt_min)
                        size = min_size + (max_size - min_size) * normalized
                    else:
                        size = (min_size + max_size) / 2
                    
                    # 사용자 스케일 적용
                    size = max(min_size, min(max_size, size * size_scale))
                    sizes.append(size)
            else:
                sizes = [min_size * size_scale] * len(frequencies)
        else:
            sizes = [min_size * size_scale] * len(frequencies)
        
        fig = go.Figure(data=[
            go.Scatter(
                x=frequencies,
                y=growths,
                mode='markers',
                marker=dict(
                    size=sizes,  # 최적화된 크기
                    color=colors,
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=topics,
                hovertemplate='<b>%{text}</b><br>' +
                             '빈도: %{x}<br>' +
                             '성장률: %{y}<br>' +
                             '<extra></extra>'
            )
        ])
        
        # 축 범위 설정 (데이터 extents에서 ±10% 여유)
        if frequencies and growths:
            x_min, x_max = min(frequencies), max(frequencies)
            y_min, y_max = min(growths), max(growths)
            
            x_range = x_max - x_min
            y_range = y_max - y_min
            
            x_padding = x_range * 0.1 if x_range > 0 else 1
            y_padding = y_range * 0.1 if y_range > 0 else 0.1
            
            fig.update_layout(
                xaxis=dict(
                    range=[x_min - x_padding, x_max + x_padding],
                    showgrid=True
                ),
                yaxis=dict(
                    range=[y_min - y_padding, y_max + y_padding],
                    showgrid=True
                )
            )
        
        fig.update_layout(
            title='토픽 클러스터 맵',
            xaxis_title='빈도',
            yaxis_title='성장률',
            template='plotly_white',
            # 캔버스 크기 및 여백 설정
            width=800,
            height=600,
            margin=dict(l=60, r=60, t=60, b=60),
            # 반응형 레이아웃
            autosize=True
        )
        
        return fig
    
    def create_news_vs_paper_comparison(self, news_topics: List[str], paper_topics: List[str]) -> go.Figure:
        """
        뉴스 vs 논문 토픽 비교 차트
        
        Args:
            news_topics: 뉴스 토픽 리스트
            paper_topics: 논문 토픽 리스트
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        # 공통 토픽 찾기
        common_topics = set(news_topics) & set(paper_topics)
        news_only = set(news_topics) - set(paper_topics)
        paper_only = set(paper_topics) - set(news_topics)
        
        # 데이터 준비
        categories = ['뉴스만', '공통', '논문만']
        counts = [len(news_only), len(common_topics), len(paper_only)]
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        
        fig = go.Figure(data=[
            go.Bar(
                x=categories,
                y=counts,
                marker_color=colors
            )
        ])
        
        fig.update_layout(
            title='뉴스 vs 논문 토픽 비교',
            xaxis_title='카테고리',
            yaxis_title='토픽 수',
            template='plotly_white'
        )
        
        return fig
    
    def create_news_topics_chart(self, news_topics: List[Dict]) -> go.Figure:
        """
        뉴스 주제 Top 10 차트 생성
        
        Args:
            news_topics: 뉴스 주제 데이터
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not news_topics:
            return go.Figure()
        
        # Top 10 주제 선택
        top_topics = sorted(news_topics, key=lambda x: x.get('count', 0), reverse=True)[:10]
        
        topic_names = [t.get('topic', 'Unknown') for t in top_topics]
        counts = [t.get('count', 0) for t in top_topics]
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_names,
                y=counts,
                marker_color='lightblue',
                hovertemplate='<b>%{x}</b><br>빈도: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='뉴스 주제 Top 10',
            xaxis_title='주제',
            yaxis_title='빈도',
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_paper_topics_chart(self, paper_topics: List[Dict]) -> go.Figure:
        """
        논문 주제 Top 10 차트 생성
        
        Args:
            paper_topics: 논문 주제 데이터
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not paper_topics:
            return go.Figure()
        
        # Top 10 주제 선택
        top_topics = sorted(paper_topics, key=lambda x: x.get('count', 0), reverse=True)[:10]
        
        topic_names = [t.get('topic', 'Unknown') for t in top_topics]
        counts = [t.get('count', 0) for t in top_topics]
        
        fig = go.Figure(data=[
            go.Bar(
                x=topic_names,
                y=counts,
                marker_color='lightcoral',
                hovertemplate='<b>%{x}</b><br>빈도: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='논문 주제 Top 10',
            xaxis_title='주제',
            yaxis_title='빈도',
            template='plotly_white',
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_paper_count_chart(self, papers_data: List[Dict]) -> go.Figure:
        """
        논문 건수 추이 차트 생성
        
        Args:
            papers_data: 논문 데이터
            
        Returns:
            go.Figure: Plotly 차트 객체
        """
        if not papers_data:
            return go.Figure()
        
        # 날짜별 논문 수 계산
        df = pd.DataFrame(papers_data)
        if 'published' in df.columns:
            df['date'] = pd.to_datetime(df['published'], errors='coerce')
            daily_counts = df.groupby(df['date'].dt.date).size()
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    mode='lines+markers',
                    name='논문',
                    line=dict(color='red', width=2),
                    marker=dict(size=4)
                )
            ])
            
            fig.update_layout(
                title='논문 건수 추이',
                xaxis_title='날짜',
                yaxis_title='건수',
                template='plotly_white',
                hovermode='x unified'
            )
            
            return fig
        
        return go.Figure()
