# 📊 Trendbot - 트렌드 분석 서비스

한국어 키워드로 뉴스, 블로그 데이터를 수집하고 KOMORAN 형태소 분석으로 정확한 분석을 제공하는 트렌드 분석 서비스입니다.

## 🚀 주요 기능

- **📰 뉴스 분석**: Naver 뉴스 데이터 수집 및 분석
- **📝 블로그 분석**: Naver 블로그 데이터 수집 및 분석
- **🔤 KOMORAN 형태소 분석**: 정확한 한국어 텍스트 분석
- **😊 감성 분석**: 한국어 텍스트의 긍정/부정/중립 분석
- **📊 토픽 분석**: 주요 키워드와 주제 추출
- **🔗 클러스터링**: 유사한 내용 그룹화
- **📈 시각화**: 차트와 워드클라우드

## 🛠️ 설치 및 실행

### 로컬 실행
```bash
# 의존성 설치
pip install -r requirements.txt

# 애플리케이션 실행
streamlit run src/main.py
```

### 환경 변수 설정
`.env` 파일을 생성하고 다음 환경 변수들을 설정하세요:

```env
# Naver API
NAVER_CLIENT_ID=your_naver_client_id
NAVER_CLIENT_SECRET=your_naver_client_secret

# Hugging Face API
HUGGINGFACE_API_KEY=your_huggingface_api_key

# Google Gemini API
GEMINI_API_KEY=your_gemini_api_key

# Debug 모드
DEBUG=True
```

## 📁 프로젝트 구조

```
trend/
├── src/
│   ├── main.py                 # 메인 애플리케이션
│   ├── common/                 # 공통 설정
│   ├── data_collectors/        # 데이터 수집기
│   ├── ai/                     # AI 분석기
│   ├── nlp/                    # 자연어 처리
│   └── visualization/          # 시각화
├── data/                       # 데이터 저장소
├── requirements.txt            # 의존성
└── README.md                   # 프로젝트 설명
```

## 🔧 기술 스택

- **Frontend**: Streamlit
- **Backend**: Python
- **Data Collection**: Naver API, arXiv API
- **NLP**: KoNLPy (KOMORAN), NLTK, spaCy
- **Visualization**: Plotly, WordCloud, Matplotlib
- **Analysis**: scikit-learn, VADER

## 📊 사용법

1. **키워드 입력**: 분석할 한국어 키워드를 입력하세요
2. **기간 설정**: 분석 기간을 선택하세요 (최대 5년)
3. **데이터 소스**: Naver 뉴스/블로그 선택
4. **분석 시작**: 트렌드 분석 시작 버튼 클릭
5. **결과 확인**: 시각화된 분석 결과 확인

## 🎯 특징

- **한국어 특화**: KOMORAN 형태소 분석으로 정확한 한국어 처리
- **실시간 분석**: 최신 데이터 기반 트렌드 분석
- **직관적 UI**: 사용하기 쉬운 Streamlit 인터페이스
- **다양한 시각화**: 차트, 워드클라우드, 감성 분석 등

## 📝 라이선스

MIT License