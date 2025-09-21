"""
형태소 분석기 모듈
"""
import re
from typing import List, Dict, Any, Optional
from collections import Counter
import pandas as pd

# KoNLPy 형태소 분석기들
try:
    from konlpy.tag import Okt, Komoran, Mecab, Kkma, Hannanum
    KONLPY_AVAILABLE = True
except ImportError:
    KONLPY_AVAILABLE = False
    Okt = None
    Komoran = None
    Mecab = None
    Kkma = None
    Hannanum = None

class MorphologicalAnalyzer:
    """형태소 분석기 클래스"""
    
    def __init__(self, analyzer_type: str = "komoran"):
        """
        형태소 분석기 초기화
        
        Args:
            analyzer_type: 분석기 타입 ("komoran", "okt", "mecab", "kkma", "hannanum")
        """
        self.analyzer_type = analyzer_type
        self.analyzer = None
        self._initialize_analyzer()
        
        # 불용어 리스트 (확장)
        self.stop_words = {
            # 조사
            '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '도', '는', '은',
            '에게', '한테', '께', '부터', '까지', '만', '도', '조차', '마저', '까지', '부터',
            
            # 어미
            '하다', '있다', '없다', '되다', '이다', '아니다', '그렇다', '이렇다', '저렇다',
            '하', '해', '했', '할', '하니', '하면', '하지만', '하므로', '해서', '하여',
            
            # 대명사
            '그', '이', '저', '그것', '이것', '저것', '그런', '이런', '저런', '그렇게', '이렇게', '저렇게',
            '그때', '이때', '저때', '그곳', '이곳', '저곳',
            
            # 연결어
            '그리고', '하지만', '그러나', '또한', '또', '그래서', '따라서', '그러므로', '그런데',
            '그러면', '그렇다면', '그러니까', '그러므로', '그래서', '따라서',
            
            # 전치사/부사
            '위해', '위한', '대해', '대한', '관해', '관한', '통해', '통한', '대비', '대비해',
            '때문', '때문에', '덕분', '덕분에', '비해', '비해서', '대비', '대비해서',
            
            # 관계사
            '관련', '관련된', '관련해', '관련해서', '관련되다', '관련된다', '관련되다',
            '경우', '경우에', '경우에는', '경우', '경우에', '경우에는',
            
            # 수사/양사
            '수', '수도', '수는', '수만', '수만큼', '수만큼이나', '수만큼이나',
            '것', '것이', '것을', '것에', '것으로', '것으로서', '것으로써',
            
            # 기타 불용어
            '등', '등의', '등이', '등을', '등에', '등으로', '등으로서', '등으로써',
            '및', '그리고', '또한', '또', '그래서', '따라서', '그러므로',
            '하지만', '그러나', '그런데', '그런', '이런', '저런',
            '그렇게', '이렇게', '저렇게', '그때', '이때', '저때',
            '그리고', '하지만', '그러나', '또한', '또', '그래서', '따라서', '그러므로',
            
            # 영어 불용어
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
            'from', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few',
            'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'now', 'also', 'well', 'back', 'even', 'still',
            'way', 'much', 'new', 'old', 'first', 'last', 'long', 'little', 'own', 'other',
            'good', 'great', 'small', 'large', 'big', 'high', 'low', 'right', 'wrong', 'true',
            'false', 'real', 'sure', 'clear', 'open', 'close', 'full', 'empty', 'free', 'busy',
            'easy', 'hard', 'simple', 'complex', 'important', 'necessary', 'possible', 'impossible',
            'likely', 'unlikely', 'certain', 'uncertain', 'sure', 'unsure', 'clear', 'unclear',
            'obvious', 'hidden', 'visible', 'invisible', 'known', 'unknown', 'familiar', 'strange',
            'normal', 'abnormal', 'regular', 'irregular', 'usual', 'unusual', 'common', 'rare',
            'frequent', 'infrequent', 'often', 'seldom', 'always', 'never', 'sometimes', 'usually',
            'generally', 'specifically', 'particularly', 'especially', 'particularly', 'mainly',
            'mostly', 'largely', 'partly', 'completely', 'totally', 'entirely', 'fully', 'partially',
            'almost', 'nearly', 'quite', 'rather', 'very', 'extremely', 'highly', 'deeply',
            'strongly', 'weakly', 'lightly', 'heavily', 'quickly', 'slowly', 'fast', 'slow',
            'early', 'late', 'soon', 'recently', 'lately', 'immediately', 'instantly', 'suddenly',
            'gradually', 'slowly', 'quickly', 'rapidly', 'fast', 'slow', 'early', 'late',
            'before', 'after', 'during', 'while', 'since', 'until', 'till', 'by', 'within',
            'without', 'inside', 'outside', 'above', 'below', 'over', 'under', 'beneath',
            'beside', 'next', 'near', 'far', 'close', 'distant', 'local', 'global', 'national',
            'international', 'regional', 'urban', 'rural', 'domestic', 'foreign', 'internal',
            'external', 'public', 'private', 'personal', 'individual', 'collective', 'group',
            'team', 'organization', 'company', 'business', 'industry', 'sector', 'field',
            'area', 'region', 'country', 'nation', 'state', 'city', 'town', 'village',
            'community', 'society', 'culture', 'tradition', 'custom', 'habit', 'practice',
            'method', 'way', 'approach', 'technique', 'strategy', 'tactic', 'plan', 'scheme',
            'program', 'project', 'initiative', 'campaign', 'movement', 'trend', 'pattern',
            'model', 'framework', 'structure', 'system', 'process', 'procedure', 'operation',
            'activity', 'action', 'behavior', 'conduct', 'performance', 'function', 'role',
            'purpose', 'goal', 'objective', 'target', 'aim', 'intention', 'plan', 'strategy',
            'policy', 'rule', 'regulation', 'law', 'legislation', 'act', 'bill', 'proposal',
            'suggestion', 'recommendation', 'advice', 'guidance', 'direction', 'instruction',
            'order', 'command', 'request', 'demand', 'requirement', 'condition', 'criteria',
            'standard', 'level', 'degree', 'extent', 'scope', 'range', 'limit', 'boundary',
            'border', 'edge', 'margin', 'space', 'room', 'area', 'place', 'location', 'position',
            'point', 'spot', 'site', 'venue', 'setting', 'environment', 'context', 'situation',
            'circumstance', 'condition', 'state', 'status', 'situation', 'case', 'instance',
            'example', 'sample', 'specimen', 'model', 'pattern', 'template', 'format',
            'style', 'type', 'kind', 'sort', 'category', 'class', 'group', 'set', 'collection',
            'series', 'sequence', 'chain', 'link', 'connection', 'relation', 'relationship',
            'association', 'connection', 'bond', 'tie', 'link', 'bridge', 'gap', 'distance',
            'difference', 'similarity', 'comparison', 'contrast', 'distinction', 'separation',
            'division', 'split', 'break', 'cut', 'slice', 'piece', 'part', 'section', 'segment',
            'portion', 'fraction', 'percentage', 'ratio', 'proportion', 'rate', 'speed',
            'velocity', 'acceleration', 'momentum', 'force', 'power', 'energy', 'strength',
            'weakness', 'advantage', 'disadvantage', 'benefit', 'cost', 'price', 'value',
            'worth', 'importance', 'significance', 'meaning', 'purpose', 'reason', 'cause',
            'effect', 'result', 'outcome', 'consequence', 'impact', 'influence', 'change',
            'transformation', 'development', 'growth', 'progress', 'advancement', 'improvement',
            'enhancement', 'upgrade', 'update', 'revision', 'modification', 'adjustment',
            'adaptation', 'accommodation', 'integration', 'coordination', 'cooperation',
            'collaboration', 'partnership', 'alliance', 'union', 'merger', 'acquisition',
            'combination', 'mixture', 'blend', 'fusion', 'synthesis', 'creation', 'production',
            'manufacture', 'construction', 'building', 'development', 'establishment',
            'foundation', 'base', 'basis', 'ground', 'foundation', 'support', 'backing',
            'assistance', 'help', 'aid', 'support', 'service', 'facility', 'resource',
            'asset', 'property', 'possession', 'ownership', 'control', 'management',
            'administration', 'governance', 'leadership', 'direction', 'guidance',
            'supervision', 'oversight', 'monitoring', 'tracking', 'following', 'pursuing',
            'chasing', 'hunting', 'seeking', 'searching', 'looking', 'finding', 'discovering',
            'detecting', 'identifying', 'recognizing', 'understanding', 'comprehending',
            'learning', 'studying', 'researching', 'investigating', 'exploring', 'examining',
            'analyzing', 'evaluating', 'assessing', 'judging', 'deciding', 'choosing',
            'selecting', 'picking', 'opting', 'preferring', 'favoring', 'liking', 'loving',
            'enjoying', 'appreciating', 'valuing', 'treasuring', 'cherishing', 'respecting',
            'admiring', 'praising', 'commending', 'recommending', 'suggesting', 'proposing',
            'offering', 'providing', 'supplying', 'delivering', 'giving', 'presenting',
            'showing', 'displaying', 'exhibiting', 'demonstrating', 'proving', 'confirming',
            'verifying', 'validating', 'authenticating', 'certifying', 'guaranteeing',
            'ensuring', 'securing', 'protecting', 'defending', 'guarding', 'watching',
            'monitoring', 'observing', 'noticing', 'seeing', 'looking', 'viewing',
            'watching', 'listening', 'hearing', 'feeling', 'touching', 'tasting', 'smelling',
            'sensing', 'perceiving', 'experiencing', 'undergoing', 'suffering', 'enduring',
            'bearing', 'tolerating', 'accepting', 'receiving', 'getting', 'obtaining',
            'acquiring', 'gaining', 'earning', 'winning', 'achieving', 'accomplishing',
            'completing', 'finishing', 'ending', 'concluding', 'terminating', 'stopping',
            'ceasing', 'halting', 'pausing', 'waiting', 'staying', 'remaining', 'continuing',
            'persisting', 'lasting', 'enduring', 'surviving', 'living', 'existing',
            'being', 'becoming', 'growing', 'developing', 'changing', 'transforming',
            'evolving', 'progressing', 'advancing', 'moving', 'going', 'coming', 'arriving',
            'reaching', 'getting', 'obtaining', 'acquiring', 'gaining', 'earning',
            'winning', 'achieving', 'accomplishing', 'completing', 'finishing', 'ending',
            'concluding', 'terminating', 'stopping', 'ceasing', 'halting', 'pausing',
            'waiting', 'staying', 'remaining', 'continuing', 'persisting', 'lasting',
            'enduring', 'surviving', 'living', 'existing', 'being', 'becoming', 'growing',
            'developing', 'changing', 'transforming', 'evolving', 'progressing', 'advancing',
            'moving', 'going', 'coming', 'arriving', 'reaching', 'getting', 'obtaining',
            'acquiring', 'gaining', 'earning', 'winning', 'achieving', 'accomplishing',
            'completing', 'finishing', 'ending', 'concluding', 'terminating', 'stopping',
            'ceasing', 'halting', 'pausing', 'waiting', 'staying', 'remaining', 'continuing',
            'persisting', 'lasting', 'enduring', 'surviving', 'living', 'existing', 'being',
            'becoming', 'growing', 'developing', 'changing', 'transforming', 'evolving',
            'progressing', 'advancing', 'moving', 'going', 'coming', 'arriving', 'reaching'
        }
        
        # 품사별 필터링 규칙
        self.pos_filters = {
            'noun': True,      # 명사
            'verb': True,       # 동사
            'adjective': True,  # 형용사
            'adverb': True,     # 부사
            'determiner': False, # 관사
            'preposition': False, # 전치사
            'conjunction': False, # 접속사
            'interjection': False, # 감탄사
            'particle': False,  # 조사
            'auxiliary': False, # 조동사
            'pronoun': False,   # 대명사
            'numeral': False,   # 수사
            'exclamation': False, # 감탄사
            'symbol': False,    # 기호
            'punctuation': False, # 구두점
            'foreign': False,   # 외래어
            'unknown': False    # 미분류
        }
    
    def _initialize_analyzer(self):
        """형태소 분석기 초기화"""
        try:
            if self.analyzer_type == "komoran":
                if not KONLPY_AVAILABLE or not Komoran:
                    raise ImportError("KoNLPy 또는 Komoran을 사용할 수 없습니다")
                self.analyzer = Komoran()
                print("✅ KOMORAN 형태소 분석기 초기화 완료")
            elif self.analyzer_type == "okt":
                if not KONLPY_AVAILABLE or not Okt:
                    raise ImportError("KoNLPy 또는 Okt를 사용할 수 없습니다")
                self.analyzer = Okt()
                print("✅ Okt 형태소 분석기 초기화 완료")
            elif self.analyzer_type == "mecab":
                if not KONLPY_AVAILABLE or not Mecab:
                    raise ImportError("KoNLPy 또는 Mecab을 사용할 수 없습니다")
                self.analyzer = Mecab()
                print("✅ Mecab 형태소 분석기 초기화 완료")
            elif self.analyzer_type == "kkma":
                if not KONLPY_AVAILABLE or not Kkma:
                    raise ImportError("KoNLPy 또는 Kkma를 사용할 수 없습니다")
                self.analyzer = Kkma()
                print("✅ Kkma 형태소 분석기 초기화 완료")
            elif self.analyzer_type == "hannanum":
                if not KONLPY_AVAILABLE or not Hannanum:
                    raise ImportError("KoNLPy 또는 Hannanum을 사용할 수 없습니다")
                self.analyzer = Hannanum()
                print("✅ Hannanum 형태소 분석기 초기화 완료")
            else:
                print(f"⚠️ 지원하지 않는 분석기 타입: {self.analyzer_type}")
                self.analyzer = None
        except ImportError as e:
            print(f"❌ 형태소 분석기 설치 필요: {e}")
            print("설치 명령어: pip install konlpy")
            self.analyzer = None
        except Exception as e:
            print(f"❌ 형태소 분석기 초기화 실패: {e}")
            self.analyzer = None
    
    def analyze_morphology(self, text: str) -> List[Dict[str, str]]:
        """
        형태소 분석 수행
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[Dict]: 형태소 분석 결과 [{'word': '인공지능', 'pos': 'Noun'}, ...]
        """
        if not self.analyzer or not text:
            return []
        
        try:
            # 형태소 분석 수행
            if self.analyzer_type == "komoran":
                morphs = self.analyzer.pos(text)
            elif self.analyzer_type == "okt":
                morphs = self.analyzer.pos(text, stem=True)
            elif self.analyzer_type == "mecab":
                morphs = self.analyzer.pos(text)
            elif self.analyzer_type == "kkma":
                morphs = self.analyzer.pos(text)
            elif self.analyzer_type == "hannanum":
                morphs = self.analyzer.pos(text)
            else:
                return []
            
            # 결과 정리
            results = []
            for word, pos in morphs:
                if word and pos:
                    results.append({
                        'word': word,
                        'pos': pos,
                        'length': len(word)
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ 형태소 분석 실패: {e}")
            return []
    
    def extract_keywords(self, text: str, max_keywords: int = 50) -> List[Dict[str, Any]]:
        """
        키워드 추출 (형태소 분석 기반)
        
        Args:
            text: 분석할 텍스트
            max_keywords: 최대 키워드 수
            
        Returns:
            List[Dict]: 키워드 리스트 [{'keyword': '인공지능', 'count': 10, 'pos': 'Noun'}, ...]
        """
        if not text:
            return []
        
        # 형태소 분석
        morphs = self.analyze_morphology(text)
        if not morphs:
            return []
        
        # 품사별 필터링 및 키워드 추출
        keywords = []
        for morph in morphs:
            word = morph['word']
            pos = morph['pos']
            
            # 품사 필터링
            if not self._is_valid_pos(pos):
                continue
            
            # 기본 불용어만 필터링 (완화)
            basic_stopwords = {
                '이', '가', '을', '를', '에', '에서', '로', '으로', '의', '와', '과', '도', '는', '은',
                '하다', '있다', '없다', '되다', '이다', '아니다', '그', '이', '저', '그것', '이것', '저것',
                '그리고', '하지만', '그러나', '또한', '또', '그래서', '따라서', '그러므로', '그런데'
            }
            
            if word.lower() in basic_stopwords:
                continue
            
            # HTML/웹 관련 불용어만 필터링 (완화)
            html_stopwords = {
                'nbsp', 'font', 'href', 'blank', 'target', 'src', 'img', 'div', 'span', 'class', 'id',
                'script', 'css', 'js', 'jquery', 'ajax', 'json', 'xml', 'html', 'htm', 'php', 'asp',
                'http', 'https', 'www', 'com', 'net', 'org', 'co', 'kr', 'link', 'url',
                'rss', 'feed', 'atom', 'syndication', 'channel', 'item', 'description', 'pubdate',
                'guid', 'category', 'enclosure', 'articles', 'target', 'oc', 'ios', 'android'
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
            
            if word.lower() in html_stopwords:
                continue
            
            # 길이 필터링
            if len(word) < 2:
                continue
            
            # 숫자 필터링
            if word.isdigit():
                continue
            
            # 특수문자 필터링
            if not re.match(r'^[가-힣a-zA-Z]+$', word):
                continue
            
            keywords.append({
                'keyword': word,
                'pos': pos,
                'length': len(word)
            })
        
        # 빈도 계산
        keyword_counts = Counter([kw['keyword'] for kw in keywords])
        
        # 상위 키워드 선택
        top_keywords = keyword_counts.most_common(max_keywords)
        
        # 결과 정리
        results = []
        for keyword, count in top_keywords:
            # 해당 키워드의 품사 정보 찾기
            pos_info = next((kw for kw in keywords if kw['keyword'] == keyword), None)
            pos = pos_info['pos'] if pos_info else 'Unknown'
            
            results.append({
                'keyword': keyword,
                'count': count,
                'pos': pos,
                'length': len(keyword)
            })
        
        return results
    
    def _is_valid_pos(self, pos: str) -> bool:
        """
        품사 유효성 검사
        
        Args:
            pos: 품사 태그
            
        Returns:
            bool: 유효한 품사 여부
        """
        if not pos:
            return False
        
        # 품사 태그 정규화
        pos_lower = pos.lower()
        
        # 명사 (Noun)
        if any(tag in pos_lower for tag in ['noun', 'n', 'nn', 'np', 'nq', 'nr', 'ns', 'nt', 'nv']):
            return True
        
        # 동사 (Verb)
        if any(tag in pos_lower for tag in ['verb', 'v', 'vv', 'va', 'vx', 'vcp', 'vcn']):
            return True
        
        # 형용사 (Adjective)
        if any(tag in pos_lower for tag in ['adjective', 'adj', 'a', 'va', 'vcn']):
            return True
        
        # 부사 (Adverb)
        if any(tag in pos_lower for tag in ['adverb', 'adv', 'mag', 'maj']):
            return True
        
        return False
    
    def extract_nouns(self, text: str) -> List[str]:
        """
        명사만 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[str]: 명사 리스트
        """
        morphs = self.analyze_morphology(text)
        nouns = []
        
        for morph in morphs:
            word = morph['word']
            pos = morph['pos']
            
            # 명사만 필터링
            if (pos and 'noun' in pos.lower() and 
                len(word) > 1 and 
                word not in self.stop_words and
                not word.isdigit() and
                re.match(r'^[가-힣a-zA-Z]+$', word)):
                nouns.append(word)
        
        return nouns
    
    def extract_verbs(self, text: str) -> List[str]:
        """
        동사만 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[str]: 동사 리스트
        """
        morphs = self.analyze_morphology(text)
        verbs = []
        
        for morph in morphs:
            word = morph['word']
            pos = morph['pos']
            
            # 동사만 필터링
            if (pos and 'verb' in pos.lower() and 
                len(word) > 1 and 
                word not in self.stop_words and
                not word.isdigit() and
                re.match(r'^[가-힣a-zA-Z]+$', word)):
                verbs.append(word)
        
        return verbs
    
    def extract_adjectives(self, text: str) -> List[str]:
        """
        형용사만 추출
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            List[str]: 형용사 리스트
        """
        morphs = self.analyze_morphology(text)
        adjectives = []
        
        for morph in morphs:
            word = morph['word']
            pos = morph['pos']
            
            # 형용사만 필터링
            if (pos and 'adj' in pos.lower() and 
                len(word) > 1 and 
                word not in self.stop_words and
                not word.isdigit() and
                re.match(r'^[가-힣a-zA-Z]+$', word)):
                adjectives.append(word)
        
        return adjectives
    
    def batch_analyze(self, texts: List[str]) -> List[List[Dict[str, str]]]:
        """
        배치 형태소 분석
        
        Args:
            texts: 분석할 텍스트 리스트
            
        Returns:
            List[List[Dict]]: 각 텍스트의 형태소 분석 결과
        """
        results = []
        for text in texts:
            morphs = self.analyze_morphology(text)
            results.append(morphs)
        
        return results
    
    def get_word_frequency(self, texts: List[str], pos_filter: str = None) -> Dict[str, int]:
        """
        단어 빈도 분석
        
        Args:
            texts: 분석할 텍스트 리스트
            pos_filter: 품사 필터 ('noun', 'verb', 'adjective', 'adverb')
            
        Returns:
            Dict[str, int]: 단어 빈도 딕셔너리
        """
        all_words = []
        
        for text in texts:
            if pos_filter == 'noun':
                words = self.extract_nouns(text)
            elif pos_filter == 'verb':
                words = self.extract_verbs(text)
            elif pos_filter == 'adjective':
                words = self.extract_adjectives(text)
            else:
                keywords = self.extract_keywords(text)
                words = [kw['keyword'] for kw in keywords]
            
            all_words.extend(words)
        
        return Counter(all_words)
    
    def get_topic_keywords(self, texts: List[str], n_topics: int = 10) -> List[Dict[str, Any]]:
        """
        토픽 키워드 추출 (형태소 분석 기반)
        
        Args:
            texts: 분석할 텍스트 리스트
            n_topics: 추출할 토픽 수
            
        Returns:
            List[Dict]: 토픽 키워드 리스트
        """
        if not texts:
            return []
        
        # 모든 텍스트에서 키워드 추출
        all_keywords = []
        for text in texts:
            keywords = self.extract_keywords(text, max_keywords=100)
            all_keywords.extend(keywords)
        
        # 빈도 계산
        keyword_counts = Counter([kw['keyword'] for kw in all_keywords])
        
        # 상위 키워드 선택
        top_keywords = keyword_counts.most_common(n_topics)
        
        # 결과 정리
        results = []
        for keyword, count in top_keywords:
            # 해당 키워드의 품사 정보 찾기
            pos_info = next((kw for kw in all_keywords if kw['keyword'] == keyword), None)
            pos = pos_info['pos'] if pos_info else 'Unknown'
            
            results.append({
                'keyword': keyword,
                'count': count,
                'pos': pos,
                'frequency': count / len(all_keywords) if all_keywords else 0
            })
        
        return results

