# 제3회 KRX 금융 언어 모델 경진대회

- 팀명 : overfit-brothers

- 팀원: [유용상](https://github.com/4N3MONE), [이기훈](https://github.com/Liky98), [임형준](https://github.com/lagokun)

- [모델 링크](https://huggingface.co/overfit-brothers/hello_world06)

- 데이터셋 링크 [MCQA](https://huggingface.co/datasets/overfit-brothers/KRX-MCQA), [Instruction](https://huggingface.co/datasets/overfit-brothers/KRX-INST)

---

### 폴더 구성

```
📦 PROJECT_ROOT
├── 📂 configs
│   └── 학습 하이퍼파라미터 및 데이터셋 설정 (yaml)
├── 📂 runs
│   └── 학습 실행 스크립트
├── 📂 data
│   └── 📂 krx_textbook
│       └── KRX e-book PDF 정제 및 문제 생성 코드
├── 📂 preprocess_data
│   └── 데이터 증강 및 정제 코드
├── 📂 scripts
│   └── 모델 학습 및 머지 관련 코드
├── 📂 test
│   └── 로컬 환경 모델 성능 평가 코드
└── 📂 utils
    └── 데이터로더 등 유틸리티 코드
```
