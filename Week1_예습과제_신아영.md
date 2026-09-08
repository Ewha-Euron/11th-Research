# Week1_예습과제_신아영

## 논문 정보
- **Title**: Deep Residual Learning for Image Recognition
- **Authors**: Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
- **Conference**: CVPR 2016
- **Paper**: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf

---

## 1. 연구 배경 및 문제 정의

CNN은 일반적으로 층이 깊어질수록 더 복잡하고 추상적인 특징을 학습할 수 있다. 그러나 단순히 층을 계속 쌓는다고 해서 항상 성능이 좋아지는 것은 아니다.

기존의 깊은 네트워크에서는 vanishing/exploding gradient 문제가 알려져 있었고, 이를 weight initialization이나 Batch Normalization(BN) 등의 방법으로 어느 정도 완화할 수 있었다. 하지만 이러한 문제를 해결한 뒤에도 네트워크가 지나치게 깊어지면 **더 깊은 모델의 training error가 오히려 더 높아지는 degradation problem**이 나타났다.

이 현상은 단순한 overfitting으로 설명하기 어렵다. 만약 더 깊은 네트워크의 추가 층들이 단순히 identity mapping을 수행한다면, 최소한 얕은 네트워크와 비슷한 training error는 얻을 수 있어야 하기 때문이다. 즉, 문제의 핵심은 깊어진 네트워크를 실제로 최적화하는 것이 어렵다는 데 있다.

---

## 2. 핵심 아이디어: Residual Learning

기존 네트워크가 원하는 mapping을 직접

\[
H(x)
\]

로 학습한다고 하자.

ResNet에서는 이를 직접 학습하는 대신, 입력 \(x\)와 목표 mapping의 차이인 residual function

\[
F(x) = H(x) - x
\]

를 학습하도록 문제를 재구성한다.

따라서 최종 출력은

\[
H(x) = F(x) + x
\]

가 된다.

즉, 여러 층이 전체 변환 \(H(x)\)를 처음부터 직접 학습하게 하는 대신 **입력에서 얼마나 변화해야 하는지 \(F(x)\)** 만 학습하게 한다.

저자들은 원하는 mapping이 identity mapping에 가까운 경우, 여러 nonlinear layer가 identity 자체를 새로 학습하는 것보다 residual을 0에 가깝게 만드는 것이 더 쉬울 수 있다고 설명한다.

---

## 3. Shortcut Connection

Residual learning은 **shortcut connection(skip connection)** 을 사용해 구현된다.

기본 residual block은 다음과 같이 표현된다.

\[
y = F(x, \{W_i\}) + x
\]

여기서
- \(x\): block의 입력
- \(F(x)\): convolution 등의 layer들이 학습한 residual
- \(x\)를 그대로 전달하는 경로: shortcut connection

이다.

입력과 출력의 차원이 같다면 \(x\)를 그대로 더하는 identity shortcut을 사용할 수 있다. 이 방식은 추가적인 parameter를 거의 요구하지 않고 계산량도 거의 증가시키지 않는다.

입력과 출력의 차원이 다른 경우에는 projection shortcut을 사용하여

\[
y = F(x, \{W_i\}) + W_sx
\]

와 같이 차원을 맞출 수 있다.

---

## 4. ResNet Architecture

논문에서는 ImageNet 실험을 위해 ResNet-18, 34, 50, 101, 152 등을 구성했다.

### ResNet-18 / ResNet-34
주로 두 개의 3×3 convolution으로 이루어진 residual block을 사용한다.

### ResNet-50 / ResNet-101 / ResNet-152
네트워크가 매우 깊어지면서 계산량을 줄이기 위해 **bottleneck block**을 사용한다.

Bottleneck block은 대략

1×1 convolution → 3×3 convolution → 1×1 convolution

구조를 사용한다.

1×1 convolution으로 channel 수를 줄인 뒤 3×3 convolution을 수행하고, 다시 1×1 convolution으로 channel 수를 늘려 계산 효율을 높인다.

특히 ResNet-152는 VGG-19보다 훨씬 깊지만 계산 복잡도는 더 낮게 설계되었다.

---

## 5. 실험 결과

### 5.1 Plain Network vs ResNet

ImageNet에서 18-layer와 34-layer plain network를 비교했을 때, 더 깊은 34-layer plain network가 오히려 더 높은 training/validation error를 보였다.

반면 residual connection을 적용하면 결과가 반대로 나타났다.

| Model | Top-1 error (%) |
|---|---:|
| Plain-18 | 27.94 |
| Plain-34 | 28.54 |
| ResNet-18 | 27.88 |
| ResNet-34 | 25.03 |

Plain network에서는 깊이를 늘렸을 때 성능이 악화되었지만, ResNet에서는 34-layer가 18-layer보다 더 좋은 성능을 보였다.

이는 residual learning이 깊은 네트워크의 optimization을 쉽게 만들고, 더 깊은 모델이 depth 증가의 이점을 실제로 활용하도록 해준다는 것을 보여준다.

### 5.2 매우 깊은 ResNet

저자들은 50, 101, 152-layer ResNet까지 확장했다. 깊이를 크게 늘렸음에도 degradation problem이 관찰되지 않았고 성능이 지속적으로 향상되었다.

특히 ResNet-152는 ImageNet validation에서 매우 높은 성능을 기록했고, 여러 ResNet을 ensemble한 모델은 ImageNet test set에서 **3.57% top-5 error**를 기록하여 ILSVRC 2015 classification task에서 1위를 차지했다.

### 5.3 CIFAR-10

CIFAR-10에서도 100개 이상의 layer를 가진 ResNet을 성공적으로 학습했고, 1000-layer 이상의 매우 깊은 모델도 실험했다.

따라서 residual learning의 효과가 특정 데이터셋에만 국한되지 않는다는 점을 보였다.

---

## 6. 논문의 핵심 기여

1. **Degradation problem을 명확하게 제시**
   - 네트워크를 단순히 깊게 만드는 것이 항상 성능 향상으로 이어지지 않으며, 깊은 plain network에서 optimization 문제가 발생함을 실험적으로 확인했다.

2. **Residual learning framework 제안**
   - 전체 mapping을 직접 학습하는 대신 residual \(F(x)=H(x)-x\)를 학습하도록 문제를 재구성했다.

3. **Shortcut connection을 이용한 간단한 구현**
   - identity shortcut은 별도의 parameter와 큰 계산 비용을 거의 추가하지 않으면서 깊은 네트워크의 학습을 쉽게 만든다.

4. **매우 깊은 네트워크 학습 가능성 입증**
   - ResNet-152까지 성공적으로 학습하여 기존보다 훨씬 깊은 신경망을 실용적으로 사용할 수 있음을 보였다.

5. **다양한 vision task로의 확장 가능성 확인**
   - ImageNet classification뿐 아니라 detection, localization, COCO detection/segmentation에서도 강력한 성능을 보였다.

---

## 7. 느낀 점 및 의문점

이 논문에서 가장 인상적이었던 부분은 단순히 새로운 layer를 추가한 것이 아니라, **학습해야 하는 문제 자체를 더 쉬운 형태로 바꿨다**는 점이다.

처음에는 ResNet이 vanishing gradient 문제를 해결하기 위한 방법이라고 생각했지만, 논문에서는 Batch Normalization을 사용한 plain network에서도 gradient가 정상적으로 전달됨에도 degradation problem이 발생한다고 설명한다. 따라서 ResNet의 핵심을 단순히 “gradient vanishing 해결”이라고만 보는 것은 부족하며, **deep network의 optimization을 쉽게 만드는 구조**라는 점이 더 중요하다고 이해했다.

또한 입력을 그대로 다음 layer로 전달하는 매우 단순한 shortcut connection이 이후의 매우 깊은 neural network 설계에 큰 영향을 주었다는 점이 흥미로웠다.

---

## 8. 토의해보고 싶은 점

**Residual connection의 효과는 주로 gradient가 더 잘 전달되기 때문일까, 아니면 \(H(x)\) 대신 \(F(x)=H(x)-x\)를 학습하게 함으로써 optimization landscape 자체가 쉬워지는 것이 더 핵심일까?**

논문에서는 vanishing gradient만으로 degradation problem을 설명하기 어렵다고 주장한다. 그렇다면 skip connection이 실제 학습 과정에서 어떤 메커니즘으로 optimization을 쉽게 만드는지 추가적으로 생각해볼 수 있을 것 같다.
