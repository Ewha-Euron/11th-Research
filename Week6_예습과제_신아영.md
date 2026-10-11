# Week6_예습과제_신아영

## 1. LoRA 전체 리뷰
### LoRA: Low-Rank Adaptation of Large Language Models

## 1. Introduction

대규모 언어모델은 보통 하나의 모델을 사전학습한 뒤, 각 downstream task에 맞게 fine-tuning해서 사용한다. 그런데 모델 크기가 커질수록 이 방식이 점점 부담스러워진다. 예를 들어 GPT-3 175B를 task마다 full fine-tuning한다고 하면, 매번 거의 175B개에 해당하는 parameter를 다시 학습하고 task별 checkpoint도 따로 저장해야 한다.

그래서 기존에도 전체 parameter를 다 학습하지 않는 여러 방법이 제안되었다. 대표적으로 adapter layer를 추가하거나 prefix/prompt 쪽 일부 parameter만 학습하는 방식이 있다. 다만 adapter는 원래 모델 사이에 추가 layer가 들어가기 때문에 inference할 때 연산이 늘어날 수 있고, prefix 계열 방법은 입력 sequence 일부를 adaptation용으로 사용한다는 단점이 있다. 또한 parameter 수는 줄어들어도 full fine-tuning만큼의 성능이 항상 보장되는 것은 아니었다.

LoRA는 여기서 **"fine-tuning할 때 생기는 weight 변화량 자체가 사실은 낮은 rank로 표현될 수 있지 않을까?"**라는 아이디어에서 출발한다. 원래 pretrained weight \(W_0\)는 그대로 두고, fine-tuning하면서 필요한 변화량 \(\Delta W\)만 작은 두 행렬의 곱으로 표현한다.

\[
W = W_0 + \Delta W,\qquad \Delta W = BA
\]

즉 \(W_0\) 전체를 다시 학습하는 것이 아니라 \(A\)와 \(B\)만 학습한다. 이때 rank \(r\)을 원래 weight dimension보다 훨씬 작게 두기 때문에 실제 trainable parameter 수가 크게 줄어든다.

이 논문에서 LoRA의 장점은 parameter 수를 줄이는 것뿐만 아니라 실제 배포 상황까지 고려했다는 점이라고 생각했다. 학습 시에는 기존 weight를 freeze하므로 gradient나 optimizer state를 대부분의 parameter에 대해 저장할 필요가 없고, inference 시에는 학습된 \(BA\)를 원래 \(W_0\)에 합쳐서 사용할 수 있다. 그래서 adapter처럼 추가적인 inference latency가 생기지 않는다.

GPT-3 175B 실험에서는 trainable parameter 수를 full fine-tuning보다 최대 약 10,000배 줄이면서도 여러 task에서 비슷하거나 더 좋은 성능을 보였다.

---

## 2. Problem Statement

일반적인 full fine-tuning에서는 pretrained model의 parameter를 \(\Phi_0\)라고 할 때, downstream task에 맞게 전체 parameter update \(\Delta\Phi\)를 학습한다.

\[
\Phi = \Phi_0 + \Delta\Phi
\]

문제는 task가 하나 늘어날 때마다 이 \(\Delta\Phi\)의 크기가 원래 모델과 거의 같다는 점이다. 모델이 작을 때는 큰 문제가 아닐 수 있지만, GPT-3처럼 175B parameter를 갖는 모델에서는 task마다 별도의 fine-tuned model을 저장하고 운영하는 것이 부담이 된다.

LoRA에서는 task-specific parameter를 훨씬 작은 \(\Theta\)로 제한한다.

\[
\Delta\Phi = \Delta\Phi(\Theta), \qquad |\Theta| \ll |\Phi_0|
\]

즉 큰 \(\Delta\Phi\)를 직접 학습하지 않고 더 작은 parameter 집합으로 간접적으로 표현한다. LoRA에서는 이 역할을 low-rank matrix \(A, B\)가 한다.

---

## 3. 기존 방법과 비교

### 3.1 Adapter

Adapter는 Transformer block 사이에 작은 neural network를 추가하는 방식이다. 전체 모델을 fine-tuning하지 않아도 된다는 점에서는 효율적이지만, 원래 없던 layer가 실제 inference 경로에 추가된다.

논문에서도 parameter 수 자체는 적더라도 online inference처럼 batch size가 작은 상황에서는 adapter 때문에 latency가 꽤 증가할 수 있다고 보여준다. GPT-2 medium 실험에서는 조건에 따라 20~30% 정도 slowdown이 발생하기도 했다.

### 3.2 Prefix 계열 방법

Prefix tuning은 모델의 weight를 직접 바꾸기보다 입력이나 각 layer의 activation에 학습 가능한 prefix를 추가한다.

이 방식도 parameter-efficient하지만, prefix가 sequence의 일부를 차지하기 때문에 실제 task 입력에 사용할 수 있는 길이가 줄어들 수 있다. 또한 논문에서는 trainable parameter를 늘린다고 성능이 계속 좋아지는 것도 아니고 optimization이 까다로운 경우가 있다고 지적한다.

LoRA는 이들과 달리 새로운 layer를 넣기보다는 **원래 dense weight의 update 자체를 low-rank로 제한**한다.

---

## 4. LoRA 방법

### 4.1 Low-Rank Update

원래 dense layer의 weight가

\[
W_0 \in \mathbb{R}^{d\times k}
\]

라고 하자. Full fine-tuning에서는 \(W_0\)와 같은 크기의 \(\Delta W\)를 자유롭게 학습한다.

LoRA는 이 \(\Delta W\)를

\[
\Delta W = BA
\]

로 둔다.

\[
B\in\mathbb{R}^{d\times r}, \qquad
A\in\mathbb{R}^{r\times k}
\]

여기서 \(r\)은 아주 작은 값으로 둔다.

forward는

\[
h = W_0x + BAx
\]

가 되고, \(W_0\)는 고정한 채 \(A,B\)만 update된다.

예를 들어 \(W_0\)가 \(4096\times4096\)이라면 full update에는 약 1,677만 개 parameter가 필요하다. 그런데 \(r=8\)이면 LoRA에서 학습되는 parameter는

\[
4096\times8 + 8\times4096 = 65,536
\]

개이다. 숫자로 직접 비교해 보니 왜 parameter-efficient한지 더 명확하게 느껴졌다.

### 4.2 Initialization과 Scaling

논문에서는 \(A\)는 random Gaussian으로 초기화하고, \(B\)는 0으로 초기화한다. 그러면 처음에는

\[
BA = 0
\]

이므로 LoRA를 추가해도 pretrained model의 처음 출력은 그대로 유지된다.

또 update에는

\[
\frac{\alpha}{r}BAx
\]

형태의 scaling을 사용한다. \(r\)을 바꿨을 때 update 크기가 너무 크게 달라지지 않도록 조절하는 역할이다.

### 4.3 Inference에서 추가 latency가 없는 이유

이 부분이 adapter와 가장 크게 다른 점이라고 생각했다.

학습할 때는

\[
W_0x + BAx
\]

처럼 두 항을 따로 계산하지만, 학습이 끝난 뒤에는

\[
W' = W_0 + BA
\]

로 미리 합칠 수 있다.

그러면 inference에서는 그냥

\[
W'x
\]

만 계산하면 된다. 결국 원래 dense layer와 같은 형태가 되기 때문에 추가 layer를 하나 더 통과하지 않아도 된다.

또 task를 바꿀 때도 큰 base model 전체를 새로 불러오는 대신 작은 LoRA weight만 바꾸면 된다.

---

## 5. Transformer에서 LoRA를 어디에 적용하는가

Transformer의 self-attention에는

\[
W_q,\; W_k,\; W_v,\; W_o
\]

네 개의 주요 projection matrix가 있다.

논문의 실험에서는 주로 attention weight에 LoRA를 적용했고, 그중에서도 \(W_q\)와 \(W_v\)를 많이 사용했다. MLP block은 고정했다.

LoRA 자체가 꼭 attention에만 적용 가능한 것은 아니다. Dense layer라면 원칙적으로 어디든 사용할 수 있다. 다만 이 논문에서는 실험을 단순하게 하고 parameter 수를 더 줄이기 위해 attention 쪽에 집중했다.

GPT-3에서 같은 parameter budget으로 비교했을 때, 한 종류의 weight에 큰 rank를 몰아주는 것보다 \(W_q\)와 \(W_v\) 두 군데에 작은 rank를 나눠 적용하는 방식이 더 좋은 결과를 보였다. 이를 보면 rank 크기 자체뿐 아니라 **어떤 weight를 adaptation할 것인가도 중요하다**고 볼 수 있다.

---

## 6. 실험

논문에서는 여러 모델과 task를 이용해 LoRA를 평가한다.

- RoBERTa base / large: GLUE
- DeBERTa XXL: GLUE
- GPT-2 medium / large: E2E NLG, WebNLG, DART
- GPT-3 175B: WikiSQL, MNLI, SAMSum

비교 방법에는 full fine-tuning, BitFit, Adapter, Prefix 계열 방법 등이 포함된다.

### 6.1 RoBERTa / DeBERTa

RoBERTa large에서 full fine-tuning은 355M parameter를 업데이트하지만, LoRA는 약 0.8M만 학습한다. 그런데 GLUE 평균 성능은 거의 비슷한 수준이다.

DeBERTa XXL에서도 full fine-tuning은 1.5B parameter 전체를 학습하는 반면 LoRA는 약 4.7M만 학습하면서 평균 성능이 비슷하거나 조금 더 높았다.

이 결과를 통해 단순히 trainable parameter가 적다고 해서 성능도 크게 떨어지는 것은 아니라는 점을 확인할 수 있었다.

### 6.2 GPT-2

GPT-2에서도 비슷한 경향이 나타난다.

E2E NLG task에서 GPT-2 medium 기준 LoRA는 약 0.35M trainable parameter로 BLEU 70.4를 기록한다. Full fine-tuning은 약 355M parameter를 학습하지만 BLEU는 68.2였다.

즉 일부 task에서는 parameter를 크게 줄였는데도 오히려 full fine-tuning보다 높은 성능이 나왔다.

### 6.3 GPT-3 175B

가장 큰 실험은 GPT-3 175B이다.

LoRA는 설정에 따라 4.7M 또는 37.7M 정도만 학습한다. 175B 전체에 비하면 정말 작은 비율이다.

Table 4의 결과를 보면:

- WikiSQL: Fine-tuning 73.8 / LoRA 최고 74.0
- MNLI-matched: Fine-tuning 89.5 / LoRA 최고 91.7
- SAMSum: Fine-tuning 52.0/28.0/44.5 / LoRA는 53점대/29점대/45점대

정도로 나타난다.

적어도 논문에서 실험한 task들에서는 LoRA가 full fine-tuning보다 훨씬 적은 parameter를 학습하면서도 성능을 유지하거나 오히려 조금 높였다.

또 GPT-3 175B에서 training VRAM이 약 1.2TB에서 350GB로 줄고, \(r=4\)로 query와 value projection만 adaptation할 경우 task-specific checkpoint가 약 35MB 수준까지 줄어든다고 한다.

---

## 7. 왜 이렇게 작은 rank로도 되는가

이 논문에서 개인적으로 흥미로웠던 부분은 성능 비교에서 끝나지 않고, low-rank update가 실제로 어떤 특성을 가지는지 분석한 부분이었다.

### 7.1 어느 weight를 바꾸는 것이 좋은가

같은 parameter budget에서 \(W_q, W_k, W_v, W_o\) 중 어디에 LoRA를 적용할지 비교했을 때 \(W_q\)와 \(W_v\)를 함께 adaptation하는 설정이 전반적으로 좋은 결과를 보였다.

한 matrix 하나만 크게 바꾸는 것보다 여러 중요한 projection에 작은 rank를 나눠 쓰는 것이 나을 수 있다는 결과다.

### 7.2 rank는 얼마나 커야 하는가

처음에는 rank가 너무 작으면 표현력이 부족할 것이라고 생각했다. 그런데 논문에서는 \(r=1,2,4\)처럼 매우 작은 값에서도 성능이 잘 유지되는 경우가 많았다.

특히 \(W_q,W_v\)를 함께 adaptation하면 \(r=1\)도 꽤 좋은 결과를 냈다.

저자들은 이 결과를 fine-tuning에 필요한 update가 실제로 매우 낮은 intrinsic rank를 가질 수 있다는 근거로 본다.

물론 이것이 모든 task에서 \(r=1\)이면 충분하다는 뜻은 아니다. 논문에서도 pre-training과 매우 다른 task라면 더 큰 rank가 필요할 수 있다고 언급한다.

### 7.3 \(\Delta W\)와 원래 \(W\)의 관계

저자들은 학습된 \(\Delta W\)의 singular direction도 분석했다.

그 결과 \(\Delta W\)는 random matrix보다는 원래 \(W\)와 관련이 있었지만, 단순히 \(W\)의 가장 큰 방향을 그대로 따라가는 것은 아니었다. 저자들은 pretrained model에 이미 들어 있지만 상대적으로 덜 강조된 방향을 downstream task에 맞게 증폭하는 것으로 해석한다.

이 부분을 보면서 LoRA는 단순히 parameter를 줄이는 압축 기법이라기보다, 이미 배운 representation 중 필요한 부분만 작게 조정하는 방식에 더 가깝다고 느꼈다.

---

## 8. 한계

LoRA가 항상 같은 작은 rank로 잘 동작한다고 보장할 수는 없다. 논문에서 좋은 결과가 나온 것은 특정 모델과 task에 대한 실험 결과이고, pre-training과 많이 다른 task에서는 더 큰 adaptation이 필요할 수 있다.

또 여러 task의 LoRA를 한 batch 안에서 동시에 써야 하는 경우에는, weight를 base model에 미리 merge해 두는 방식이 불편하다.

그리고 어떤 layer에 LoRA를 적용할지, rank를 얼마로 둘지에 대해서도 완전히 정해진 원칙이 있는 것은 아니다. 논문에서도 이 부분은 heuristic에 의존한다고 보고 future work로 남긴다.

---

## 9. Conclusion / 배운 점

처음 LoRA를 봤을 때는 큰 weight matrix 자체를 low-rank로 압축하는 방법이라고 생각했다. 그런데 실제 핵심은 **원래 pretrained weight가 아니라 fine-tuning하면서 생기는 변화량 \(\Delta W\)를 low-rank로 표현한다는 것**이었다.

\[
W_0 \text{는 고정하고 } \Delta W = BA \text{만 학습}
\]

이라고 놓고 보니 구조가 훨씬 단순하게 이해됐다.

또 trainable parameter가 적다는 것과 inference가 빠르다는 것이 항상 같은 말은 아니라는 점도 이번 논문에서 처음 확실하게 구분해서 이해했다. Adapter도 parameter는 적지만 새 layer를 통과해야 하므로 latency가 생길 수 있다. LoRA는 반대로 학습한 weight를 원래 weight에 합칠 수 있기 때문에 inference 구조 자체를 바꾸지 않는다.

그리고 \(r=1\)이나 \(r=4\)처럼 생각보다 작은 rank에서도 성능이 잘 나오는 결과가 가장 인상적이었다. 대형 언어모델을 특정 task에 맞게 바꾸기 위해 필요한 변화가 전체 parameter 공간에 넓게 퍼져 있지 않고 작은 subspace에 집중되어 있을 수도 있다는 의미로 이해했다.

다만 논문 결과만 보고 LoRA가 항상 full fine-tuning보다 좋다고 일반화하면 안 될 것 같다. task가 pre-training distribution과 많이 다르거나 더 큰 변화가 필요하면 작은 rank가 충분하지 않을 수 있고, 어떤 weight에 LoRA를 적용할지도 아직 선택이 필요하다.

---

# 2. YOLO Introduction 리뷰
### You Only Look Once: Unified, Real-Time Object Detection

## 1. 기존 Object Detection 방식의 문제

Object detection은 이미지에 어떤 object가 있는지만 맞히는 classification과 달리, **무엇이 있는지와 어디에 있는지를 동시에 찾아야 하는 문제**다.

YOLO 이전의 대표적인 detection 방법은 이 과정을 여러 단계로 나누는 경우가 많았다. DPM은 sliding window 방식으로 이미지 여러 위치와 scale에서 classifier를 반복해서 실행했고, R-CNN은 먼저 object가 있을 것 같은 region proposal을 만든 뒤 각 region을 CNN으로 분류했다.

R-CNN 계열을 보면 region proposal, feature extraction, classification, bounding box 보정, NMS처럼 과정이 여러 개로 나뉜다. 각 component를 따로 조정해야 하고 전체 pipeline도 느리다는 문제가 있었다.

---

## 2. YOLO의 핵심 아이디어

YOLO는 이 문제를 아예 **하나의 regression problem**으로 본다.

이미지 전체를 입력으로 넣고 network가 바로

- bounding box 위치
- object가 있을 confidence
- class probability

를 예측한다.

즉 object 후보 영역을 먼저 찾고 그 부분을 다시 분류하는 것이 아니라, **이미지 전체를 한 번 보고 여러 object의 위치와 class를 동시에 예측**한다.

그래서 이름이 You Only Look Once이다.

논문 Figure 1의 과정도 단순하다.

1. 이미지를 \(448\times448\)로 resize
2. convolutional network 한 번 실행
3. confidence를 기준으로 최종 detection 선택

처음에는 "한 번만 본다"는 게 단순히 빠르다는 의미라고 생각했는데, 논문을 읽어보니 더 중요한 부분은 기존 detection pipeline을 여러 module로 나누지 않고 **single network로 통합해서 end-to-end로 학습한다는 점**이었다.

---

## 3. 장점

### 3.1 속도가 매우 빠르다

YOLO는 test image 하나에 대해 network evaluation을 한 번만 수행한다.

논문에서 base YOLO는 45 FPS, Fast YOLO는 155 FPS로 동작했다. 당시 real-time 기준인 30 FPS를 넘기기 때문에 실제 video stream에서도 사용할 수 있는 수준이다.

### 3.2 이미지 전체 context를 이용한다

Sliding window나 region proposal 방식은 local region을 중심으로 판단하는 경우가 많다. 반면 YOLO는 training과 inference 모두에서 전체 이미지를 입력으로 본다.

그래서 object의 모양만 보는 것이 아니라 주변 context도 같이 사용할 수 있다. 논문에서는 Fast R-CNN이 background patch를 object라고 잘못 판단하는 경우가 YOLO보다 많았다고 설명한다.

### 3.3 다른 domain으로도 어느 정도 일반화된다

저자들은 자연 이미지로 학습한 detector를 artwork에 적용하는 실험도 진행했다.

R-CNN은 사진에서 artwork로 domain이 바뀌었을 때 성능이 크게 떨어졌지만, YOLO는 상대적으로 성능 저하가 작았다. 저자들은 YOLO가 이미지 전체를 보면서 object의 모양과 주변 관계까지 학습하기 때문이라고 해석한다.

---

## 4. 한계

물론 속도가 빠른 대신 정확도 측면에서는 당시 state-of-the-art detector보다 부족했다.

가장 큰 문제는 localization error였다. object가 있다는 것은 찾더라도 bounding box를 정확하게 맞추지 못하는 경우가 많았다.

또 작은 object가 여러 개 가까이 모여 있는 경우에도 약했다. 뒤의 방법론을 보면 YOLO는 이미지를 grid로 나누고 각 grid cell이 제한된 수의 bounding box를 담당하도록 하는데, 이런 구조가 작은 object가 많이 모여 있는 장면에서는 제약이 된다.

그래서 YOLO의 trade-off는 **속도와 단순함은 크게 얻었지만 정밀한 localization은 어느 정도 희생했다**고 정리할 수 있다.

---

## 5. Intro를 읽고 이해한 점

YOLO를 읽기 전에는 그냥 "빠른 object detection 모델" 정도로 알고 있었다. 그런데 Intro를 읽고 보니 핵심은 기존 모델을 조금 더 빠르게 만든 것이 아니라 **object detection 문제 자체를 다른 방식으로 정의했다는 점**이었다.

기존 방식은

> object가 있을 만한 곳을 찾고 → 그 부분을 분류하고 → bounding box를 조정한다

는 흐름이었다면, YOLO는

> 이미지 → 바로 bounding box + class probability

로 간다.

이렇게 보니 왜 모델 구조가 단순해지고 속도가 빨라졌는지도 이해가 됐다.

또 "You Only Look Once"라고 해서 후처리가 아예 없는 것은 아니다. 논문에서도 중복 detection을 줄이기 위해 NMS를 사용할 수 있다. 여기서 one look은 region proposal과 classifier를 여러 번 반복하지 않고, **main prediction이 한 번의 network forward pass에서 나온다**는 의미로 이해하는 게 맞다고 생각했다.
