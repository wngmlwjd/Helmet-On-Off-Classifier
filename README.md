# ⛑️ helmet-on-off-classifier

안전모(헬멧) 착용 여부를 분류하는 이미지 분류 모델 학습 및 성능 평가 파이프라인입니다.

---

## 🔍 개요

이미지 데이터를 기반으로 헬멧 착용(on) / 미착용(off) 을 분류하는 머신러닝 모델을 학습하고, 다양한 색상(Color) 설정과 학습 전략(Strategy) 조합으로 성능을 비교·평가합니다. 결과는 정확도(Accuracy)의 평균 및 분산으로 요약됩니다.

---

## 📝 논문

본 프로젝트와 관련된 논문이 출판되면 아래에 추가될 예정입니다.

<!-- ```
저자명. (연도). 논문 제목. 학술지명, 권(호), 페이지. https://doi.org/xxxxx
``` -->

---

## 📁 프로젝트 구조

```
helmet-on-off-classifier/
├── data_prep/                          # 데이터 전처리 관련 스크립트
├── train/                              # 모델 학습 스크립트 및 노트북
├── test/                               # 모델 테스트 스크립트 및 노트북
├── model/                              # 학습된 모델 파일
├── results/                            # 실험 결과 txt 파일 저장
├── main.py                             # 전체 파이프라인 실행 진입점
├── postprogress.py                     # 결과 집계 및 정확도 요약 생성
├── accuracy_summary.txt                # 모델별 정확도 평균/분산 요약
└── use_model.txt                       # 사용 모델 목록 (날짜-번호 매핑)
```

---

## ✨ 주요 기능

- 헬멧 착용 여부 이진 분류 (on / off)
- **색상(Color) 조건** (gray / color) × **학습 전략(Strategy)** (1~4) 총 8가지 조합 실험
- 다수의 실험 결과 파일을 자동으로 집계하여 평균 정확도 및 분산 계산
- 요약 결과를 `accuracy_summary.txt`로 자동 저장

---

<!-- ## 📊 실험 결과 (accuracy_summary.txt)

| Color | Strategy | Accuracy (Mean) | Variance |
|-------|----------|-----------------|----------|
| gray  | 1        | 0.8605          | 0.000031 |
| gray  | 2        | 0.8565          | 0.000048 |
| gray  | 3        | 0.8387          | 0.000024 |
| gray  | 4        | 0.8488          | 0.000042 |
| color | 1        | **0.8744**      | 0.000010 |
| color | 2        | 0.8708          | 0.000040 |
| color | 3        | 0.8575          | 0.000023 |
| color | 4        | 0.8638          | 0.000039 |

> 컬러 이미지 + Strategy 1 조합이 **최고 정확도 87.44%** 달성

--- -->

<!-- ## ⚙️ 설치 방법

**1. 저장소 클론**

```bash
git clone https://github.com/wngmlwjd/helmet-on-off-classifier.git
cd helmet-on-off-classifier
```

**2. 의존 패키지 설치**

```bash
pip install numpy
```

> 학습/테스트에 사용된 추가 패키지는 `train/`, `test/` 폴더 내 노트북을 참고하세요.

--- -->

<!-- ## 🚀 사용 방법

### 전체 파이프라인 실행

```bash
python main.py
```

### 결과 집계 (정확도 요약 생성)

`postprogress.py` 상단의 `input_files` 경로를 실험 결과 파일 경로로 수정한 후 실행합니다.

```python
# postprogress.py

input_files = [
    r"results/20260109_01.txt",
    r"results/20260109_02.txt",
    # ...
]

output_path = r"./accuracy_summary.txt"
```

```bash
python postprogress.py
```

실행하면 `accuracy_summary.txt`에 색상×전략 조합별 정확도 평균과 분산이 저장됩니다.

--- -->

<!-- ## 📂 결과 파일 형식

`results/` 폴더 내 실험 결과 텍스트 파일은 다음 형식을 따릅니다.

```
[COLOR=gray | STRATEGY=1]
Accuracy: 0.861200
...
[COLOR=color | STRATEGY=1]
Accuracy: 0.875100
...
```

`postprogress.py`가 이 파일들을 정규식으로 파싱하여 집계합니다.

--- -->

<!-- ## 📄 실험 보고서

`3가지 데이터 학습 및 성능 평가.pdf` 파일에서 데이터셋 구성, 학습 전략별 상세 실험 내용 및 분석 결과를 확인할 수 있습니다. -->
