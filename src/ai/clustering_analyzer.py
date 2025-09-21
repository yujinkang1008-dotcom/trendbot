"""
클러스터링 분석 모듈
"""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.express as px

class ClusteringAnalyzer:
    """클러스터링 분석 클래스"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words=None,
            ngram_range=(1, 2)
        )
    
    def cluster_documents(self, documents, method: str = 'kmeans', n_clusters: int = 5) -> Dict[str, Any]:
        """
        문서 클러스터링
        
        Args:
            documents: 클러스터링할 문서 리스트 또는 딕셔너리 리스트
            method: 클러스터링 방법 ('kmeans' 또는 'hdbscan')
            n_clusters: 클러스터 수 (K-means용)
            
        Returns:
            Dict: 클러스터링 결과
        """
        if not documents or len(documents) < 2:
            return {'clusters': [], 'centers': [], 'labels': []}
        
        # 텍스트 추출 (딕셔너리인 경우 'text' 키에서 추출)
        text_list = []
        for item in documents:
            if isinstance(item, dict):
                # 딕셔너리인 경우 'text' 또는 'text_clean' 키에서 텍스트 추출
                text = item.get('text_clean', item.get('text', ''))
                if text:
                    text_list.append(text)
            elif isinstance(item, str):
                text_list.append(item)
        
        if not text_list or len(text_list) < 2:
            return {'clusters': [], 'centers': [], 'labels': []}
        
        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform(text_list)
            feature_names = self.vectorizer.get_feature_names_out()
            
            # 클러스터링 수행
            if method == 'kmeans':
                clusterer = KMeans(n_clusters=min(n_clusters, len(text_list)), random_state=42)
                cluster_labels = clusterer.fit_predict(tfidf_matrix)
                cluster_centers = clusterer.cluster_centers_
            else:  # hdbscan
                clusterer = HDBSCAN(min_cluster_size=2, min_samples=1)
                cluster_labels = clusterer.fit_predict(tfidf_matrix)
                cluster_centers = None
            
            # 클러스터별 문서 그룹화
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'document': text_list[i],
                    'index': i
                })
            
            # 클러스터별 대표 키워드 추출
            cluster_keywords = {}
            for cluster_id, docs in clusters.items():
                if cluster_id == -1:  # HDBSCAN의 노이즈 클러스터
                    continue
                
                # 클러스터 내 문서들의 TF-IDF 점수 평균
                cluster_indices = [doc['index'] for doc in docs]
                cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0).A1
                
                # 상위 키워드 선택
                top_indices = cluster_tfidf.argsort()[-10:][::-1]
                top_keywords = [
                    feature_names[idx] for idx in top_indices 
                    if cluster_tfidf[idx] > 0
                ]
                
                cluster_keywords[cluster_id] = {
                    'keywords': top_keywords,
                    'size': len(docs)
                }
            
            return {
                'clusters': clusters,
                'centers': cluster_centers,
                'labels': cluster_labels,
                'keywords': cluster_keywords,
                'method': method
            }
            
        except Exception as e:
            print(f"클러스터링 오류: {e}")
            return {'clusters': [], 'centers': [], 'labels': []}
    
    def create_topic_map(self, documents: List[str], labels: List[int], keywords: Dict[int, List[str]]) -> go.Figure:
        """
        토픽 맵 생성
        
        Args:
            documents: 문서 리스트
            labels: 클러스터 라벨
            keywords: 클러스터별 키워드
            
        Returns:
            go.Figure: 토픽 맵 차트
        """
        if not documents or len(documents) < 2:
            return go.Figure()
        
        try:
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            
            # 차원 축소 (t-SNE)
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(documents)-1))
            embeddings = tsne.fit_transform(tfidf_matrix.toarray())
            
            # 클러스터별 색상 매핑
            unique_labels = list(set(labels))
            colors = px.colors.qualitative.Set3[:len(unique_labels)]
            color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
            
            # 데이터 준비
            x_coords = embeddings[:, 0]
            y_coords = embeddings[:, 1]
            
            # 클러스터별로 그룹화하여 플롯
            fig = go.Figure()
            
            for label in unique_labels:
                if label == -1:  # 노이즈 클러스터
                    continue
                
                mask = np.array(labels) == label
                cluster_x = x_coords[mask]
                cluster_y = y_coords[mask]
                cluster_docs = [documents[i] for i in range(len(documents)) if mask[i]]
                
                # 클러스터별 대표 키워드
                cluster_keywords = keywords.get(label, {}).get('keywords', [])[:5]
                cluster_name = ', '.join(cluster_keywords) if cluster_keywords else f'Cluster {label}'
                
                fig.add_trace(go.Scatter(
                    x=cluster_x,
                    y=cluster_y,
                    mode='markers',
                    name=cluster_name,
                    marker=dict(
                        size=10,
                        color=color_map[label],
                        opacity=0.7
                    ),
                    text=cluster_docs,
                    hovertemplate='<b>%{text}</b><br><extra></extra>'
                ))
            
            fig.update_layout(
                title='토픽 클러스터 맵',
                xaxis_title='t-SNE 1',
                yaxis_title='t-SNE 2',
                template='plotly_white',
                showlegend=True
            )
            
            return fig
            
        except Exception as e:
            print(f"토픽 맵 생성 오류: {e}")
            return go.Figure()
    
    def analyze_cluster_growth(self, clusters: Dict[int, List[Dict]], time_data: List[str] = None) -> Dict[int, float]:
        """
        클러스터 성장률 분석
        
        Args:
            clusters: 클러스터 데이터
            time_data: 시간 데이터 (선택사항)
            
        Returns:
            Dict: 클러스터별 성장률
        """
        growth_rates = {}
        
        for cluster_id, docs in clusters.items():
            if cluster_id == -1:  # 노이즈 클러스터 제외
                continue
            
            # 시간 데이터가 있는 경우 성장률 계산
            if time_data and len(time_data) == len(docs):
                # 최근 30% vs 이전 70% 비교
                split_point = int(len(docs) * 0.7)
                recent_docs = docs[split_point:]
                older_docs = docs[:split_point]
                
                if len(older_docs) > 0:
                    growth_rate = (len(recent_docs) - len(older_docs)) / len(older_docs)
                else:
                    growth_rate = 0.0
            else:
                # 단순히 클러스터 크기 기반 성장률
                growth_rate = len(docs) / len(clusters) if clusters else 0.0
            
            growth_rates[cluster_id] = growth_rate
        
        return growth_rates
    
    def create_cluster_comparison(self, clusters: Dict[int, List[Dict]], keywords: Dict[int, List[str]]) -> go.Figure:
        """
        클러스터 비교 차트 생성
        
        Args:
            clusters: 클러스터 데이터
            keywords: 클러스터별 키워드
            
        Returns:
            go.Figure: 클러스터 비교 차트
        """
        if not clusters:
            return go.Figure()
        
        # 클러스터별 크기와 키워드 수 계산
        cluster_sizes = []
        cluster_names = []
        keyword_counts = []
        
        for cluster_id, docs in clusters.items():
            if cluster_id == -1:  # 노이즈 클러스터 제외
                continue
            
            cluster_sizes.append(len(docs))
            cluster_keywords = keywords.get(cluster_id, {}).get('keywords', [])
            keyword_counts.append(len(cluster_keywords))
            
            # 클러스터 이름 (상위 3개 키워드)
            top_keywords = cluster_keywords[:3]
            cluster_name = ', '.join(top_keywords) if top_keywords else f'Cluster {cluster_id}'
            cluster_names.append(cluster_name)
        
        # 버블 차트 생성
        fig = go.Figure(data=[
            go.Scatter(
                x=cluster_sizes,
                y=keyword_counts,
                mode='markers',
                marker=dict(
                    size=[size * 2 for size in cluster_sizes],  # 크기 = 문서 수
                    color=keyword_counts,
                    colorscale='Viridis',
                    opacity=0.7,
                    line=dict(width=1, color='black')
                ),
                text=cluster_names,
                hovertemplate='<b>%{text}</b><br>' +
                             '문서 수: %{x}<br>' +
                             '키워드 수: %{y}<br>' +
                             '<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='클러스터 비교 (크기 = 문서 수, 색상 = 키워드 수)',
            xaxis_title='문서 수',
            yaxis_title='키워드 수',
            template='plotly_white'
        )
        
        return fig
