# Week6_예습과제_유다연

---

# [YOLO] You Only Look Once: Unified, Real-Time Object Detection (전체 리뷰)

- Redmon et al., CVPR 2016 / https://arxiv.org/pdf/1506.02640

## 1. Introduction

> 💡 논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가
> 논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

### 논문이 다루는 분야
- **객체 탐지(Object Detection)**: 이미지 안에 어떤 물체가 어디에 있는지(bounding box + class) 찾는 task.
- 사람은 이미지를 한 번 보고 즉시 물체와 위치를 파악한다. 빠르고 정확한 탐지가 가능해야 자율주행, 보조 기기, 로보틱스 같은 **실시간 응용**이 가능하다.

### 해당 task에서 기존 연구 한계점
- **DPM**: sliding window로 이미지 전체에 분류기를 반복 적용 → 느림.
- **R-CNN 계열**: region proposal 생성 → 영역별 분류 → box 보정 → 중복 제거의 **복잡한 다단계 파이프라인**.
  - 각 단계를 따로 학습해야 해서 느리고 전체 최적화가 어렵다(R-CNN은 이미지당 40초 이상).
  - proposal 영역만 보기 때문에 **이미지 전체 문맥을 보지 못해** 배경을 물체로 오인하는 경우가 많다.

### 논문의 contributions
- 탐지를 **이미지 픽셀 → bounding box 좌표 + 클래스 확률로 한 번에 가는 단일 회귀 문제**로 재정의했다.
- 하나의 CNN으로 end-to-end 학습·추론 → 기존 한계(느린 속도, 복잡한 파이프라인)에 직접 대응한다.
  1. **매우 빠름**: 45 FPS(YOLO), 155 FPS(Fast YOLO). 다른 실시간 시스템 대비 mAP 2배 이상.
  2. **Global reasoning**: 이미지 전체를 보고 예측 → background 오류가 Fast R-CNN의 절반 이하.
  3. **일반화 능력**: 자연 이미지로 학습해 예술 작품에 적용해도 DPM, R-CNN보다 성능 저하가 작다.
- 단, 최신 탐지기보다 정확도가 낮고 작은 물체의 위치 추정에 약하다는 점을 스스로 밝힌다.

## 2. Related Work

> 💡 Introduction에서 언급한 기존 연구들에 대해 어떻게 서술하는가
> 제안 방법의 차별성을 어떻게 표현하고 있는가

(논문 Section 3 "Comparison to Other Detection Systems")
- 기존 탐지 파이프라인은 Haar, SIFT, HOG 같은 feature 추출 → 분류기/위치 추정기 적용 구조로 서술한다.
- **DPM**: feature 추출, 영역 분류, box 예측이 분리된 파이프라인 ↔ YOLO는 이 모든 부분을 **하나의 CNN으로 통합**해 동시에 수행한다.
- **R-CNN**: Selective Search로 약 2000개 proposal 생성 후 각각 CNN 분류 → 느림 ↔ YOLO도 셀 단위로 후보 box를 제안한다는 점은 비슷하지만, 공간 제약으로 **box를 98개만** 만들고 모든 구성 요소를 **함께 최적화**한다.
- **Fast / Faster R-CNN**: 연산 공유, 신경망 기반 proposal로 속도를 개선했지만 여전히 실시간에 못 미친다.
- **DPM 가속 연구(30Hz DPM 등)**: 기존 파이프라인을 빠르게 만드는 접근 ↔ YOLO는 파이프라인 자체를 버려 **설계상 빠르다**.
- **Deep MultiBox**: 단일 클래스 탐지만 가능하고 추가 분류가 필요 ↔ YOLO는 완전한 탐지 시스템.
- **OverFeat**: sliding window 기반, 지역 정보만 봄 ↔ YOLO는 전역 문맥 활용.
- **MultiGrasp**: grid 회귀 아이디어는 유사하지만 물체 하나의 영역만 예측 ↔ YOLO는 **여러 클래스의 여러 물체**를 동시에 탐지.

## 3. 제안 방법론

> 💡 Introduction에서 언급된 내용과 동일하게 작성되어 있는가
> Introduction에서 언급한 제안 방법이 가지는 장점에 대한 근거가 있는가
> 제안 방법에 대한 설명이 구현 가능하도록 작성되어 있는가

### Main Idea
**Unified Detection**
- 입력 이미지를 **S × S 그리드**로 나누고, 물체 중심이 들어간 셀이 그 물체를 담당한다.
- 각 셀은
  - **B개의 bounding box** (x, y, w, h, confidence)를 예측한다.
    - confidence = Pr(Object) × IOU(pred, truth)
    - (x, y): 셀 기준 상대 좌표, (w, h): 이미지 기준 상대 크기
  - **C개의 조건부 클래스 확률** Pr(Class_i | Object)를 예측한다(셀당 한 세트).
- 테스트 시 class-specific confidence = Pr(Class_i | Object) × Pr(Object) × IOU = Pr(Class_i) × IOU
- VOC 기준 S=7, B=2, C=20 → 출력 **7 × 7 × 30** 텐서.

**Network Design**
- GoogLeNet에서 영감: **24 conv layer + 2 FC layer**, Inception 대신 1×1 reduction + 3×3 conv.
- Fast YOLO: conv 9개, 필터 수 축소.

**Training**
- 앞 20개 conv를 ImageNet(224×224)으로 사전학습 → conv 4개 + FC 2개 추가, 입력 **448×448**로 탐지 학습.
- 마지막 층은 linear, 나머지는 **Leaky ReLU(0.1)**.
- Loss: sum-squared error 기반, 다음과 같이 보완.
  - 물체 없는 셀이 대부분이라 confidence가 0으로 쏠리는 문제 → **λ_coord = 5**, **λ_noobj = 0.5**.
  - 큰/작은 box 오차를 동일하게 다루는 문제 → w, h에 **제곱근**.
  - 셀의 B개 box 중 정답과 **IOU가 가장 높은 하나만 responsible** → predictor가 크기·비율별로 특화.
  - 분류 손실은 물체가 있는 셀에만 부과.
- VOC 2007+2012, 약 135 epoch, batch 64, momentum 0.9, weight decay 0.0005, dropout 0.5, scaling/translation/exposure/saturation 증강.

**Inference**
- 네트워크 1회 통과로 이미지당 98개 box → **NMS**로 중복 제거(+2~3% mAP).

**Limitations**
- 셀당 box 2개·클래스 1개의 **강한 공간 제약** → 작은 물체 무리(새 떼 등)에 약함.
- 처음 보는 **종횡비**에 약하고, 다운샘플링된 coarse feature를 사용.
- 작은 box의 오차를 큰 box와 비슷하게 취급 → **localization 오류**가 주된 오류 원인.

### Contribution
- Introduction의 주장과 방법론이 일관된다: "단일 네트워크·단일 회귀"라는 설계가 그대로 구현되어 있다.
- 장점의 근거
  - 속도: proposal 단계가 없고 네트워크를 한 번만 통과 → 실험에서 FPS로 검증.
  - 배경 오류 감소: 이미지 전체를 입력으로 받는 구조 → 오류 분석 실험으로 검증.
- 그리드 크기, 손실 가중치, 학습 스케줄까지 명시되어 **구현 가능한 수준**으로 서술되어 있다.

## 4. 실험 및 결과

> 💡 Introduction에서 언급한 제안 방법의 장점을 검증하기 위한 실험이 있는가

### Dataset
- **PASCAL VOC 2007, 2012** (학습: VOC 2007+2012)
- 일반화 실험: **Picasso Dataset, People-Art Dataset** (예술 작품 속 사람 탐지)

### Baseline
- 실시간: 100Hz / 30Hz DPM
- 비실시간: Fastest DPM, R-CNN Minus R, R-CNN, Fast R-CNN, Faster R-CNN (ZF, VGG-16)
- 일반화: DPM, Poselets, D&T, R-CNN

### 결과
- **속도/정확도 (장점 1 검증)** — VOC 2007
  - Fast YOLO 52.7 mAP / 155 FPS → 가장 빠른 탐지기
  - YOLO 63.4 mAP / 45 FPS (30Hz DPM 26.1 mAP의 2배 이상)
  - Fast R-CNN 70.0 mAP / 0.5 FPS, Faster R-CNN(VGG-16) 73.2 mAP / 7 FPS
- **오류 분석 (장점 2 검증)** — Fast R-CNN vs YOLO
  - Localization 오류: YOLO 19.0% vs Fast R-CNN 8.6% → YOLO가 위치 추정에 약함
  - Background 오류: YOLO 4.75% vs Fast R-CNN 13.6% → YOLO가 배경 오탐이 훨씬 적음
- **Fast R-CNN + YOLO 결합**
  - 71.8 → **75.0 mAP**(+3.2%p). 다른 Fast R-CNN끼리 결합하면 효과가 미미 → 두 모델의 **오류 유형이 달라** 보완 효과가 크다.
- **VOC 2012**: YOLO 57.9 mAP. bottle, sheep, tv/monitor 등 **작은 물체** 클래스에서 특히 약함.
- **일반화 (장점 3 검증)**: 예술 작품에서 R-CNN은 성능이 크게 떨어지지만 YOLO는 감소 폭이 작아 가장 높은 AP를 기록.
- **Real-time in the wild**: 웹캠 연결 실시간 탐지 데모.

## 5. 결론 (배운점)

> 💡 연구의 의의 및 한계점, 본인이 생각하는 좋았던/아쉬웠던 점 (배운점)

- **의의**: 탐지를 "후보 영역 + 분류"에서 **"하나의 회귀 문제"로 바꾼 발상의 전환**. 진정한 실시간 탐지를 가능하게 했고, SSD와 YOLOv2 이후 one-stage detector 계열의 출발점이 되었다.
- **한계**: 셀당 예측 수 제한, 작은 물체·밀집 물체 탐지 약점, 낮은 localization 정확도.
- **좋았던 점**: Introduction에서 주장한 세 가지 장점(속도, 배경 오류 감소, 일반화)을 각각 대응되는 실험으로 검증해 논리 구조가 깔끔하다. 특히 오류 유형 분석과 Fast R-CNN과의 결합 실험이 인상적이었다.
- **아쉬웠던 점**: 정확도에서는 Fast/Faster R-CNN에 뒤처진다. 손실 함수가 box 크기에 따른 오차를 근본적으로 해결하지 못했고, 이는 후속 버전에서 anchor box와 multi-scale feature로 개선되었다.
- **배운 점**: 문제를 어떻게 정의하느냐(분류 → 회귀)가 모델 구조와 속도를 완전히 바꿀 수 있다는 것을 배웠다.

---

# [LoRA] LoRA: Low-Rank Adaptation of Large Language Models (Introduction 리뷰)

- Hu et al., ICLR 2022 / https://arxiv.org/pdf/2106.09685

## 1. Introduction

> 💡 논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가
> 논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

### 논문이 다루는 분야
- **대규모 사전학습 언어모델의 효율적인 적응(Parameter-Efficient Fine-Tuning)**.
- NLP의 많은 응용은 대규모 사전학습 모델 하나를 여러 다운스트림 task에 맞게 **fine-tuning**하는 방식에 의존한다.
- full fine-tuning은 모든 파라미터를 업데이트하므로 task마다 원래 모델과 같은 크기의 모델이 생긴다. GPT-3(175B)처럼 모델이 커지면 task별 모델을 저장·배포하는 것은 사실상 불가능할 정도로 비용이 크다.

### 해당 task에서 기존 연구 한계점
- 일부 파라미터만 학습하거나 task별 외부 모듈을 추가해, task마다 적은 파라미터만 저장하는 방법이 시도되었다.
- 하지만
  - **Adapter layers**: 모델 깊이가 늘어나 **추론 지연(inference latency)** 발생.
  - **Prefix tuning 등 프롬프트 기반 방법**: 최적화가 어렵고, 프롬프트에 일부를 쓰므로 **사용 가능한 시퀀스 길이가 줄어든다**.
  - 결국 full fine-tuning 성능에 미치지 못하는 경우가 많아 **효율성과 모델 품질 사이의 trade-off**가 생긴다.

### 논문의 contributions
- **핵심 가설**: 과파라미터화된 모델은 실제로 낮은 내재 차원(low intrinsic dimension)에 존재한다는 선행 연구에서 착안해, **적응 과정의 가중치 변화량 ΔW도 낮은 내재 rank를 가진다**고 가정한다.
- **방법**: 사전학습 가중치 W₀는 **freeze**하고, 변화량을 두 저랭크 행렬의 곱으로 학습한다.
  - W₀ + ΔW = W₀ + BA, B ∈ ℝ^{d×r}, A ∈ ℝ^{r×k}, r ≪ min(d, k)
  - GPT-3 175B에서도 full rank(d = 12,288) 대비 **r = 1, 2 정도의 매우 낮은 rank로 충분**함을 보인다.
- **기존 한계에 대한 대응 (LoRA의 장점)**
  1. **저장·task 전환 비용 감소**: 사전학습 모델을 공유하고 task마다 작은 A, B만 교체.
  2. **학습 효율**: 대부분 파라미터의 gradient와 optimizer state가 필요 없어 adaptive optimizer 사용 시 하드웨어 장벽을 **최대 3배** 낮춤.
  3. **추론 지연 없음**: 배포 시 BA를 W₀에 **merge**하면 구조가 fine-tuned 모델과 동일 → adapter의 latency 문제 해결.
  4. **다른 방법과 결합 가능(orthogonal)**: prefix tuning 등과 함께 사용 가능.
- **용어 정리**: d_model(Transformer layer 입출력 차원), W_q / W_k / W_v / W_o(self-attention의 query/key/value/output projection), W₀(사전학습 가중치), ΔW(적응 중 누적 업데이트), r(LoRA 모듈의 rank). 최적화는 Adam, d_ffn = 4 × d_model.

### 💭 Introduction을 읽고
- 기존 방법의 한계(추론 지연, 시퀀스 길이 감소, 성능 저하)를 하나씩 짚고, LoRA의 장점이 각각에 정확히 대응되도록 제시되어 문제 정의와 제안의 연결이 명확하다.
- "큰 모델을 바꾸는 데 필요한 정보는 저차원이다"라는 단순한 가설로 저장·학습·추론 문제를 동시에 해결한다는 점이 인상 깊었다.
