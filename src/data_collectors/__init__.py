"""
데이터 수집기 모듈
"""
from .google_trends_collector import GoogleTrendsCollector
from .naver_collector import NaverCollector
from .arxiv_collector import ArxivCollector

__all__ = [
    "GoogleTrendsCollector",
    "NaverCollector", 
    "ArxivCollector"
]
