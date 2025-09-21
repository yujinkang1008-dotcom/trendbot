"""
텍스트 정제 유틸리티 모듈
"""
import re
import html
from typing import Set

# 한글/영문 공통 불용어 세트
STOPWORDS: Set[str] = {
    # 한글 불용어
    '그', '이', '저', '것', '때', '곳', '일', '번', '가지', '개', '년', '월', '시', '분', '초',
    '하나', '둘', '셋', '네', '다섯', '여섯', '일곱', '여덟', '아홉', '열',
    '그것', '이것', '저것', '여기', '저기', '거기', '이곳', '저곳',
    '나', '너', '우리', '그들', '당신', '자신',
    '는', '은', '이', '가', '을', '를', '에', '의', '로', '와', '과', '도', '만', '부터', '까지',
    '하다', '되다', '있다', '없다', '같다', '다르다', '많다', '적다',
    '크다', '작다', '좋다', '나쁘다', '새로', '오래',
    '또', '그리고', '그러나', '하지만', '그래서', '따라서', '그런데', '그러면',
    '매우', '너무', '정말', '진짜', '아주', '꽤', '상당히', '조금', '약간',
    '항상', '가끔', '자주', '때때로', '언제나', '절대', '결코',
    '모든', '각', '어떤', '모든', '전체', '일부', '대부분',
    # 영문 불용어
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
    'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
    'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
    'this', 'that', 'these', 'those', 'here', 'there', 'where', 'when', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just', 'now',
}

# 쓰레기 토큰 세트 (HTML, URL, 웹 관련) - 확장
GARBAGE_TOKENS: Set[str] = {
    # HTML 엔티티 및 태그
    'nbsp', 'quot', 'amp', 'lt', 'gt', 'font', 'href', 'br', 'span', 'div', 'class', 'id', 'style',
    'script', 'css', 'js', 'jquery', 'ajax', 'json', 'xml', 'html', 'htm', 'php', 'asp', 'jsp',
    
    # URL 및 도메인 관련
    'http', 'https', 'www', 'com', 'net', 'org', 'co', 'kr', 'link', 'url', 'src', 'img',
    
    # 검색 엔진 및 플랫폼
    'google', 'news', 'naver', 'daum', 'yahoo', 'bing', 'search', 'youtube', 'facebook', 'twitter',
    'instagram', 'linkedin', 'github', 'stackoverflow', 'reddit', 'quora',
    
    # 웹 인터페이스 관련
    'click', 'view', 'more', 'read', 'see', 'show', 'hide', 'open', 'close', 'button', 'menu',
    'nav', 'navigation', 'header', 'footer', 'sidebar', 'top', 'bottom', 'left', 'right', 'center',
    'middle', 'first', 'last', 'prev', 'next', 'previous', 'back', 'forward', 'up', 'down',
    
    # 콘텐츠 관련
    'page', 'site', 'web', 'blog', 'post', 'article', 'content', 'text', 'data', 'info',
    'ad', 'ads', 'advertisement', 'banner', 'popup', 'modal', 'dialog', 'window', 'tab',
    
    # 법적/정책 관련
    'copyright', 'reserved', 'rights', 'terms', 'privacy', 'policy', 'cookie', 'gdpr',
    
    # 사용자 인터페이스
    'home', 'about', 'contact', 'help', 'faq', 'support', 'login', 'register', 'signup',
    'signin', 'logout', 'profile', 'account', 'settings', 'preferences', 'options',
    
    # 시간 관련
    'today', 'yesterday', 'tomorrow', 'week', 'month', 'year', 'time', 'date', 'day',
    'am', 'pm', 'morning', 'afternoon', 'evening', 'night', 'hour', 'minute', 'second',
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'may', 'june',
    'july', 'august', 'september', 'october', 'november', 'december',
    
    # 기술 관련
    'api', 'sdk', 'framework', 'library', 'module', 'package', 'version', 'update',
    'download', 'install', 'setup', 'config', 'configuration', 'setting', 'option',
    
    # 일반적인 웹 용어
    'online', 'offline', 'internet', 'web', 'website', 'webpage', 'browser', 'chrome',
    'firefox', 'safari', 'edge', 'mobile', 'desktop', 'tablet', 'phone', 'device',
    
    # 뉴스/미디어 관련
    'media', 'press', 'journal', 'magazine', 'newspaper', 'tv', 'radio', 'podcast',
    'video', 'audio', 'image', 'photo', 'picture', 'graphic', 'chart', 'graph',
    
    # 소셜 미디어
    'social', 'share', 'like', 'comment', 'reply', 'follow', 'unfollow', 'subscribe',
    'unsubscribe', 'notification', 'alert', 'message', 'chat', 'forum', 'community',
    
    # 이메일 관련
    'email', 'mail', 'send', 'receive', 'inbox', 'outbox', 'spam', 'trash', 'draft',
    
    # 파일 관련
    'file', 'folder', 'directory', 'upload', 'download', 'save', 'delete', 'copy',
    'paste', 'cut', 'edit', 'create', 'new', 'old', 'recent', 'latest', 'updated',
    
    # 상태 관련
    'active', 'inactive', 'enabled', 'disabled', 'on', 'off', 'yes', 'no', 'true',
    'false', 'success', 'error', 'warning', 'info', 'debug', 'test', 'demo',
    
    # 크기/수량 관련
    'size', 'small', 'medium', 'large', 'big', 'tiny', 'huge', 'massive', 'mini',
    'micro', 'macro', 'full', 'empty', 'half', 'quarter', 'double', 'triple',
    
    # 색상 관련
    'color', 'colour', 'red', 'green', 'blue', 'yellow', 'orange', 'purple', 'pink',
    'black', 'white', 'gray', 'grey', 'brown', 'dark', 'light', 'bright', 'dim',
    
    # 위치 관련
    'location', 'place', 'position', 'area', 'region', 'country', 'city', 'state',
    'address', 'street', 'road', 'avenue', 'boulevard', 'lane', 'drive', 'way',
    
    # 방향 관련
    'north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest',
    'up', 'down', 'left', 'right', 'front', 'back', 'side', 'corner', 'edge', 'center',
    
    # 숫자 관련 (일반적인 패턴)
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
    'hundred', 'thousand', 'million', 'billion', 'trillion', 'first', 'second', 'third',
    'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth', 'last', 'final',
    
    # 일반적인 접두사/접미사
    'pre', 'post', 'anti', 'pro', 'non', 'un', 're', 'over', 'under', 'out', 'in',
    'up', 'down', 'off', 'on', 'auto', 'self', 'super', 'ultra', 'mega', 'micro',
    
    # 기타 일반적인 웹 용어
    'etc', 'etcetera', 'and', 'or', 'but', 'so', 'if', 'then', 'else', 'when', 'where',
    'how', 'why', 'what', 'who', 'which', 'that', 'this', 'these', 'those', 'here',
    'there', 'everywhere', 'nowhere', 'somewhere', 'anywhere', 'everywhere'
}

def normalize_for_topics(text: str) -> str:
    """
    텍스트를 토픽 분석용으로 정제 (강화된 버전)
    
    Args:
        text: 정제할 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if not text or not isinstance(text, str):
        return ""
    
    # 원본 텍스트 길이 기록
    original_length = len(text)
    
    # HTML 엔티티 디코드
    text = html.unescape(text)
    
    # HTML 태그 제거 (더 강화)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'&[a-zA-Z0-9#]+;', '', text)  # HTML 엔티티 제거
    
    # URL 및 이메일 제거 (더 강화)
    text = re.sub(r'https?://[^\s]+', '', text)
    text = re.sub(r'www\.[^\s]+', '', text)
    text = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '', text)
    
    # 특수 문자 및 기호 제거 (더 강화)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    
    # 숫자만으로 이루어진 토큰 제거
    text = re.sub(r'\b\d+\b', '', text)
    
    # 연속된 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    
    # AI는 의미 있는 단어이므로 그대로 유지
    
    # 토큰화
    tokens = text.lower().split()
    
    # 강화된 필터링
    filtered_tokens = []
    for token in tokens:
        # 기본 길이 체크
        if len(token) <= 1:
            continue
            
        # 숫자 체크
        if token.isdigit():
            continue
            
        # 숫자가 포함된 토큰 제거
        if any(char.isdigit() for char in token):
            continue
            
        # 불용어 체크
        if token in STOPWORDS:
            continue
            
        # 쓰레기 토큰 체크
        if token in GARBAGE_TOKENS:
            continue
            
        # 추가 금지 패턴 체크 (대소문자 구분 없음)
        forbidden_patterns = [
            'rss', 'xml', 'json', 'api', 'http', 'www', 'com', 'net', 'org',
            'nbsp', 'font', 'href', 'src', 'img', 'div', 'span', 'class', 'id',
            'script', 'css', 'js', 'jquery', 'ajax', 'html', 'htm', 'articles',
            'target', 'oc', 'feed', 'atom', 'syndication', 'channel', 'item',
            'link', 'description', 'pubdate', 'guid', 'category', 'enclosure',
            'ios', 'android', 'windows', 'mac', 'linux'  # 추가 금지 단어 (ai 제거)
        ]
        
        if token.lower() in forbidden_patterns:
            continue
            
        # 특수 문자나 기호가 포함된 토큰 제거
        if not re.match(r'^[가-힣a-zA-Z]+$', token):
            continue
            
        filtered_tokens.append(token)
    
    result = ' '.join(filtered_tokens)
    
    # 디버깅 정보 (선택적)
    if original_length > 100:  # 긴 텍스트만 디버깅
        filtered_ratio = len(result) / original_length if original_length > 0 else 0
        if filtered_ratio < 0.1:  # 10% 미만으로 필터링된 경우
            print(f"⚠️ 텍스트 과도하게 필터링됨: {original_length} -> {len(result)} ({filtered_ratio:.2%})")
            print(f"   원본 샘플: {text[:100]}...")
            print(f"   결과 샘플: {result[:100]}...")
    
    return result
