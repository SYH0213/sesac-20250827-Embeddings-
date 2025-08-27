# 문장 임베딩 기반 유사도 비교 데모

이 문서는 Gradio로 구현한 **문장 임베딩 기반 유사도 비교 데모**의 전체 구성과 모델별 결과를 정리한 보고서입니다. 아래는 UI 흐름 및 모델별 결과 화면 캡처입니다.

---

## 사이트 동작 UI

### 입력칸


![[과제 정리/gradio_data/18_inputdata.png]]

### 모델 선택

![[과제 정리/gradio_data/19_embedding_model_select.png]]

### 출력 문장 개수 및 유사도 임계치 설정

![[과제 정리/gradio_data/20_K_.png]]

### 파일 업로드 및 전처리, 청킹 옵션

![[과제 정리/gradio_data/21_Setting.png]]

### 출력칸

![[과제 정리/gradio_data/22_outputdata.png]]

---

## 모델별 결과 비교

### HuggingFace (multilingual-e5-large-instruct)

- 김치 관련 질의  
    ![[과제 정리/gradio_data/1_huggingface_kimchi.png]]
    
- 마리오 관련 질의  
    ![[과제 정리/gradio_data/2_huggingface_mario.png]]
    
- K-POP 관련 질의  
    ![[과제 정리/gradio_data/3_huggingface_kpop.png]]
    
- 문화 관련 질의  
    ![[과제 정리/gradio_data/4_huggingface_culture.png]]
    

### Ollama (nomic-embed-text)

- 김치 관련 질의  
    ![[과제 정리/gradio_data/5_ollama_kimchi.png]]
    
- 마리오 관련 질의  
    ![[과제 정리/gradio_data/6_ollama_mario.png]]
    
- K-POP 관련 질의  
    ![[과제 정리/gradio_data/7_ollama_kpop.png]]
    
- 문화 관련 질의  
    ![[과제 정리/gradio_data/8_ollama_culture.png]]
    

### OpenAI (text-embedding-3-small)

- 김치 관련 질의 (에러 발생: **유사도 임계치를 높게 설정해서 결과가 출력되지 않음**)  
    ![[과제 정리/gradio_data/9_openai_kimchi_error.png]]
    
- 김치 관련 질의 (재시도 결과)  
    ![[과제 정리/gradio_data/10_openai_kimchi.png]]
    
- 마리오 관련 질의  
    ![[과제 정리/gradio_data/11_openai_mario.png]]
    
- K-POP 관련 질의  
    ![[과제 정리/gradio_data/12_openai_kpop.png]]
    
- 문화 관련 질의  
    ![[과제 정리/gradio_data/13_openai_culture.png]]
    

### Upstage (solar-embedding-1-large)

- 김치 관련 질의  
    ![[과제 정리/gradio_data/14_upstage_kimchi.png]]
    
- 마리오 관련 질의  
    ![[과제 정리/gradio_data/15_upstage_mario.png]]
    
- K-POP 관련 질의  
    ![[과제 정리/gradio_data/16_upstage_kpop.png]]
    
- 문화 관련 질의  
    ![[과제 정리/gradio_data/17_upstage_culture.png]]
    

---

## 요약 분석

- **HuggingFace**: 안정적으로 0.8대의 높은 유사도 점수를 보여주며, 한국어/영어 문장 모두 잘 처리.
    
- **Ollama**: 전반적으로 높은 점수(0.85~0.89대)를 기록, 다만 항목 간 분산이 조금 있음.
    
- **OpenAI**: 점수가 낮고(0.1~0.4대), 특히 임계치 설정이 너무 높으면 결과가 출력되지 않는 문제가 발생. 에러가 뜬 사례도 존재.
    
- **Upstage**: 점수는 낮은 편(0.2~0.4대)이나, 결과 매칭은 비교적 일관적임.
    

---

이 보고서는 Gradio 환경에서 **입력 → 모델 선택 → 유사도 계산 → 결과 시각화**의 전체 흐름을 설명하고, 4종 모델의 결과 차이를 시각적으로 비교할 수 있도록 구성되었습니다.