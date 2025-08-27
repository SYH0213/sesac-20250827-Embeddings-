🎯 목표

RAG.py의 코드 스타일을 참고하되, 문장 임베딩 유사도 비교용 Gradio 데모를 gradio_text_Embeddings.py에 새로 구현하라.

사용자는 “기준 문장”을 입력한다.

준비된 문장 집합(카테고리 8개, 각 3~4개 + 카테고리당 영어 1문장 포함)과의 코사인 유사도를 계산하고, 유사도 내림차순 정렬 표와 시각화를 제공한다.

여러 임베딩 모델을 토글로 바꿔가며 결과를 비교한다.

📂 파일/입출력 조건

생성 대상 파일: gradio_text_Embeddings.py (빈 파일에 전체 코드를 작성)

외부 키/설정은 환경변수 사용 (키를 하드코딩하지 말 것)

데이터 소스는 다음 중 하나로 선택 가능하게:

내장 샘플(코드 내 하드코딩된 리스트)

PDF 업로드 (PyMuPDF로 텍스트 추출 후 라인/문장 분리)

CSV 업로드 (category,text_kr,text_en 컬럼 지원)

업로드 시 자동 감지: 확장자에 따라 파서 선택(PDF→PyMuPDF, CSV→pandas)

🧪 데이터(샘플 세트 포함)

기본 탑재용 문장 세트는 한글 위주 + 카테고리당 영어 1문장(EN 라벨 없이 자연스럽게 섞임)을 내장해라.

(참고) 내가 제공한 PDF(embedding_test_sentences_mix_noENtag.pdf)와 동일한 구성을 코드 내 리스트로도 포함해라.

문장 컬럼은 하나의 리스트로 합쳐 사용하되, 카테고리 정보도 함께 보관해서 결과 테이블에 표시하라.

⚙️ 임베딩 모델 옵션(토글)

최소 3개 이상 준비(환경에 맞게 가벼운 기본 모델 포함):

OpenAI: text-embedding-3-small 또는 최신 소형 임베딩

Google (Vertex/Gemini Embeddings): 사용 가능 시 포함(키 없으면 비활성)

Hugging Face (로컬/CPU 가능 모델): 예) sentence-transformers/all-MiniLM-L6-v2

모델 선택 드롭다운 제공. 모델 가용성(키/패키지) 확인 후 비활성/경고 표시 지원.

🧮 전처리 & 유사도 산출

입력 텍스트 정규화 옵션(체크박스):

소문자화, 불용어 제거(영/한 선택), 숫자/기호 제거, 중복 공백 정리

벡터 정규화(L2) 후 코사인 유사도 계산(직접 구현 또는 sklearn.metrics.pairwise.cosine_similarity)

Top-K(기본 10)와 유사도 임계치(0~1 슬라이더) 제공

🖥️ Gradio UI 요구사항

좌측:

기준 문장(Textbox)

모델 선택(Dropdown)

Top-K(Slider) / 임계치(Slider)

전처리 옵션(CheckboxGroup)

데이터 소스 선택(Radio: 내장 / PDF 업로드 / CSV 업로드) + 업로더 컴포넌트

실행 버튼(“유사도 계산”)

우측:

결과 테이블(순위, 유사도, 문장, 카테고리)

막대 그래프 시각화(상위 K개 유사도; matplotlib 사용, 색상 지정 X)

요약 섹션: “모델 A vs B 차이 포인트” 자동 요약(간단 규칙/휴리스틱 가능)

상태/오류 메시지 토스트: 키 누락, 모델 불가, 업로드 실패, 파싱 결과 0개 등

🔍 분석/비교 포인트(자동 요약 문구 생성)

같은 카테고리 문장들이 상위에 올랐는지(정확성)

영어 문장이 끼어 있어도 의미가 잘 매칭되는지(언어 혼합 대응력)

모델별 점수 분포(편향/스케일 차이)

임계치 변경 시 결과 민감도

🔒 보안·품질 가이드

API 키는 환경변수에서만 읽기: OPENAI_API_KEY, GOOGLE_API_KEY 등

예외 처리: 키 없으면 해당 모델 옵션 비활성 + 안내

PDF/CSV 파싱 실패 시 사용자에게 원인 메시지 반환

긴 텍스트 업로드 시 청크 분할 옵션(체크박스) 제공: chunk_size, chunk_overlap 노출

재현성: 난수 시드 고정(필요 시)

✅ 완료 기준(수용 테스트)

 python gradio_text_Embeddings.py 실행 시 Gradio 앱 정상 구동

 기준 문장 입력 → 결과 테이블이 유사도 내림차순으로 표시

 모델 토글 시 결과가 갱신되고, 차이점 간단 요약 문구 출력

 PDF/CSV 업로드 후에도 정상 파싱 & 결과 산출

 Top-K/임계치/전처리 옵션이 결과에 반영됨

 코드 내에 내장 샘플 데이터 포함되어 있어 키/업로드 없이도 최소 기능 시연 가능

🧩 함수/구조(권장)

load_sentences(source_mode, file) → List[Dict{category, text}]

preprocess(text, options) → str

get_embedder(provider_name) → (embed_fn, dim, meta)

compute_embeddings(texts, embed_fn) → np.ndarray

cosine_topk(query_vec, doc_vecs, k) → indices, scores

build_plot(top_texts, scores) → matplotlib.figure.Figure

🚫 비목표(Non-goals)

대규모 파인튜닝/학습 구현

RAG 검색/생성(이번 과제는 유사도 비교 실습에 한정)

📝 구현 시 주의 메모

matplotlib 시각화는 단일 플롯만 사용, 색상 지정 금지

한글 폰트 문제로 그래프 라벨이 깨질 수 있음 → 텍스트만 영문/숫자로 처리 권장

매우 짧은 문장(1~2단어)은 임베딩 성능이 불안정 → 경고/필터링 옵션 고려

김치, 마리오에 대해 알려줘, 케이팝 콘서트의 의미는 뭐야?