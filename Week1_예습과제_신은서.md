# ResNet

## 1. Introduction

### 논문이 다루는 분야
딥러닝 기반 이미지 인식(image classification, object detection, localization, segmentation) 분야를 다루며, 특히 매우 깊은 CNN을 안정적으로 학습시키는 최적화(optimization) 문제에 초점을 맞춘다.

### 해당 task에서 기존 연구 한계점
- 네트워크 깊이가 증가할 때 발생하는 vanishing/exploding gradient 문제는 정규화된 초기화(normalized initialization)와 배치 정규화(Batch Normalization) 등으로 어느 정도 해결되었다.
- 그러나 깊이가 깊어질수록 정확도가 포화(saturate)된 후 오히려 급격히 저하되는 **degradation problem**이 새롭게 관찰되었다. 이는 overfitting 때문이 아니라 학습(optimization) 자체의 어려움 때문이며, 저자들은 이를 실험적으로 검증하였다.
- 이론적으로는 얕은 네트워크에 identity mapping 층을 추가하여 구성한 더 깊은 네트워크는 얕은 네트워크보다 training error가 높을 수 없어야 하지만, 실제 SGD 기반 solver는 이러한 해(solution)를 합리적인 시간 내에 찾지 못한다.

### 논문의 contributions
- 기존의 unreferenced mapping H(x)를 직접 학습하는 대신, residual mapping F(x) := H(x) − x를 학습하고 F(x) + x로 원래 mapping을 복원하는 **residual learning framework**를 제안하였다.
- Shortcut(identity) connection을 통해 추가 파라미터와 연산량 없이 residual 학습이 가능함을 보였다.
- 최대 152층에 이르는 극도로 깊은 residual network를 ImageNet에서 성공적으로 학습시켜, 동일 깊이의 plain network 대비 최적화 용이성과 정확도 향상을 입증하였다.
- CIFAR-10에서는 100층, 1000층 이상까지 깊이를 확장한 실험을 추가로 제시하였다.
- ImageNet 분류에서 top-5 error 3.57%를 달성하여 ILSVRC 2015 classification task 1위를 차지하였고, ImageNet detection·localization, COCO detection·segmentation에서도 모두 1위를 기록하였다.

## 2. Related Work
Introduction에서 언급한 degradation problem과 관련하여, 논문은 residual 개념과 shortcut connection을 활용한 기존 연구들을 폭넓게 다룬다. 이미지 검색·분류를 위한 shallow residual representation인 VLAD와 Fisher Vector, PDE를 다중 스케일 하위 문제로 재구성하는 Multigrid 기법, hierarchical basis preconditioning 등은 모두 "적절한 재구성(reformulation)이나 전처리(preconditioning)가 최적화를 쉽게 만든다"는 공통된 통찰을 제공하며, 이는 본 논문의 동기와 맞닿아 있다.

Shortcut connection 계열 연구로는 MLP 입력을 출력에 직접 연결하는 초기 기법, 중간 층을 auxiliary classifier에 연결하는 방식, gradient·response를 centering하는 기법, inception 구조의 shortcut branch 등을 소개한다. 특히 동시대 연구인 Highway Networks와의 차별성을 명확히 제시하는데, Highway Networks는 데이터에 종속적이고 학습 가능한 파라미터를 갖는 gating 함수로 shortcut을 열고 닫는 반면, 본 논문의 identity shortcut은 파라미터가 전혀 없고 항상 열려 있어 모든 정보가 그대로 전달되며 residual function이 추가로 학습된다는 점에서 근본적으로 다르다. 또한 Highway Networks는 100층 이상의 깊이에서 정확도 향상을 보이지 못했다는 한계를 지적하며 제안 방법의 우수성을 부각시킨다.

## 3. 제안 방법론

### Main Idea
몇 개의 stacked layer가 목표로 하는 underlying mapping을 H(x)라 할 때, 기존 방식은 이 층들이 H(x)를 직접 근사하도록 학습한다. 본 논문은 대신 동일한 층들이 residual mapping F(x) := H(x) − x를 근사하도록 하고, 최종 출력은 shortcut을 통해 더해진 F(x) + x로 복원한다(y = F(x, {Wi}) + x). Identity shortcut connection은 추가 파라미터나 연산량을 거의 발생시키지 않는다. 만약 identity mapping이 최적해라면, 다수의 비선형 층으로 identity를 직접 근사하는 것보다 residual을 0으로 수렴시키는 편이 훨씬 쉬울 것이라는 가설에 근거한다. 입력과 출력의 차원이 다를 경우에는 1×1 convolution 기반 projection shortcut(Ws)으로 차원을 맞추거나(option B), zero-padding으로 차원을 늘리는 방식(option A)을 사용한다. 더 깊은 네트워크(50/101/152층)에서는 연산 비용 절감을 위해 1×1–3×3–1×1 구조의 bottleneck building block을 도입한다.

### Contribution
Shortcut이 추가 파라미터를 유발하지 않으므로 plain network와 residual network를 동일한 depth·width·parameter 수 조건에서 공정하게 비교할 수 있다는 실험 설계상의 장점이 있다. 이를 바탕으로 identity shortcut이 degradation problem을 실제로 완화한다는 것을 다양한 깊이와 두 데이터셋(ImageNet, CIFAR-10)에서 일관되게 입증하였으며, 1000층 이상의 극단적으로 깊은 네트워크에서도 최적화 자체는 어려움 없이 가능함을 보였다(다만 성능은 overfitting으로 인해 저하될 수 있음을 함께 확인함).

## 4. 실험 및 결과

### Dataset
- ImageNet 2012 classification: 128만 장의 학습 이미지, 5만 장의 validation, 10만 장의 test 이미지, 1000개 클래스
- CIFAR-10: 5만 장 학습, 1만 장 테스트, 10개 클래스
- PASCAL VOC 2007/2012, MS COCO: object detection·segmentation 평가에 사용 (Faster R-CNN 프레임워크 기반)

### Baseline
- Plain network(shortcut 없이 층을 단순히 쌓은 VGG 스타일 구조): 18-layer, 34-layer
- 비교 대상 기존 SOTA 모델: VGG-16/19, GoogLeNet, PReLU-net, BN-inception, Highway Networks, FitNet, Maxout, NIN, DSN 등

### 결과
- ImageNet에서 plain-34는 plain-18보다 오히려 validation error가 높아 degradation problem을 재확인하였다. 반면 ResNet-34는 ResNet-18보다 우수했으며(2.8%p 개선), plain-34 대비 top-1 error를 3.5%p 낮추었다.
- Identity shortcut(A) vs projection shortcut(B, C) 비교 실험에서 세 옵션 모두 plain network보다 우수했고 옵션 간 차이는 미미하여, 파라미터가 없는 identity shortcut만으로도 충분함을 확인하였다.
- Bottleneck 구조 기반 ResNet-50/101/152는 34-layer보다도 더 우수한 성능을 보였으며, 152-layer 단일 모델은 top-5 validation error 4.49%를 기록하였고, 여러 모델의 ensemble로 ImageNet test set에서 top-5 error 3.57%를 달성해 ILSVRC 2015 classification 1위를 차지하였다.
- CIFAR-10 실험에서 plain network는 110층 이상에서 학습 자체가 어려워졌지만, ResNet은 1202층까지도 training error 0.1% 미만으로 학습이 가능했다. 다만 1202층 모델은 110층 모델보다 test error가 높아(오버피팅으로 추정) 무조건적인 깊이 증가가 능사는 아님을 보여주었다.
- Object detection(PASCAL VOC, COCO)에서는 동일한 Faster R-CNN 구조에서 backbone만 VGG-16에서 ResNet-101로 교체했을 때 COCO의 mAP@[.5, .95] 기준 28%의 상대적 성능 향상을 얻었으며, PASCAL VOC 2007/2012 detection, COCO detection·segmentation, ImageNet detection·localization 등 ILSVRC & COCO 2015의 여러 트랙에서 모두 1위를 차지하였다.

## 5. 결론(배운점)
이 연구의 의의는 F(x) + x라는 매우 단순한 아이디어만으로 극도로 깊은 신경망의 고질적인 최적화 문제(degradation problem)를 해결했다는 데 있다. Identity shortcut은 추가 파라미터나 연산 비용 없이도 학습을 용이하게 만들며, classification뿐 아니라 detection·localization·segmentation 등 다양한 task와 다양한 depth에서 일관되게 효과가 검증되었다. 이후 residual connection은 CNN을 넘어 Transformer를 포함한 대부분의 현대 딥러닝 아키텍처에서 표준적인 구성 요소로 자리 잡았다는 점에서 파급력이 매우 크다.

다만 한계점으로, 왜 degradation problem이 발생하는지, 그리고 residual learning이 왜 이 문제를 효과적으로 완화하는지에 대한 이론적 설명(예: loss landscape 관점의 분석)은 충분히 제시되지 않았으며 향후 연구 과제로 남겨두었다. 또한 1202층과 같이 지나치게 깊은 모델에서는 여전히 overfitting으로 인한 성능 저하가 관찰되어, 단순히 깊이를 늘리는 것이 항상 정답은 아님을 시사한다.

배운 점으로는, 아이디어 자체는 단순하지만 이를 뒷받침하기 위해 plain network와 residual network를 동일 조건에서 비교하고, ImageNet과 CIFAR-10 양쪽에서 검증하며, shortcut 옵션(A/B/C)을 세밀하게 비교하고, layer response의 표준편차 분석까지 곁들이는 등 매우 체계적이고 설득력 있는 실험 설계를 갖추었다는 점이 인상 깊었다. 아쉬운 점은 앞서 언급했듯 residual learning의 효과에 대한 근본적인 원리 규명이 실험적 관찰 수준에 머물러 있어, 이후 연구에서 이를 이론적으로 규명할 여지가 크게 남아 있다는 것이다.
