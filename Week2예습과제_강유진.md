# 논문리뷰 템플릿

**논문:** *Attention Is All You Need*  
**저자:** Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, Illia Polosukhin  
**발표:** NeurIPS 2017

## 1. Introduction

<aside>
💡

논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가

논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

</aside>

### 논문이 다루는 분야

본 논문은 **sequence transduction**, 특히 **Neural Machine Translation(NMT)** 문제를 다룬다. 당시 대표적인 sequence transduction 모델은 RNN, LSTM, GRU와 같은 recurrent neural network를 기반으로 한 encoder-decoder 구조였으며, 좋은 성능을 내는 모델은 여기에 attention mechanism을 결합해 사용했다.

논문은 이러한 기존 구조에서 **recurrence와 convolution을 모두 제거하고 attention mechanism만을 사용하는 새로운 architecture인 Transformer**를 제안한다. 목적은 번역 성능을 높이는 동시에, 기존 sequence model의 순차 계산이 만드는 학습 병목을 해결하는 것이다.

### 해당 task에서 기존 연구 한계점

RNN 기반 sequence model은 일반적으로 다음과 같이 이전 hidden state에 의존한다.

\[
h_t=f(h_{t-1},x_t)
\]

현재 위치 \(t\)의 hidden state를 계산하려면 이전 위치 \(t-1\)의 계산이 끝나야 한다. 이 **sequential dependency** 때문에 하나의 sequence 안에서 위치별 계산을 병렬화하기 어렵다. 문장이 길어질수록 이 제약이 커지고, memory constraint 때문에 example 단위의 batching에도 한계가 생긴다.

CNN 기반 sequence model은 모든 위치의 representation을 병렬로 계산할 수 있지만, 멀리 떨어진 두 위치를 연결하는 데 필요한 연산 수가 거리에 따라 증가한다. ConvS2S에서는 그 수가 선형적으로, ByteNet에서는 logarithmic하게 증가한다. 이렇게 정보 전달 경로가 길어지면 먼 위치 사이의 dependency를 학습하기가 더 어려워질 수 있다.

Attention mechanism은 long-range dependency를 직접 다룰 수 있었지만, 기존의 대부분의 sequence transduction model에서는 recurrent network와 함께 사용하는 보조 요소였다.

- RNN: sequence 내부 계산이 순차적이어서 병렬화가 어려움
- CNN: distant position 사이의 정보 전달에 여러 연산 또는 layer가 필요함
- 기존 attention model: 대부분 attention을 recurrence와 결합해 사용함

### 논문의 contributions

본 논문의 가장 큰 contribution은 **recurrence와 convolution 없이 attention만으로 구성한 Transformer architecture**를 제안했다는 것이다.

1. 기존 encoder-decoder 틀을 유지하면서 encoder와 decoder를 self-attention, encoder-decoder attention, position-wise feed-forward network로 구성하였다.
2. 모든 위치의 representation을 동시에 계산할 수 있어 sequence 내부 연산의 병렬성을 높였다.
3. Self-attention layer에서는 임의의 두 위치가 직접 상호작용할 수 있어 maximum path length를 \(O(1)\)로 줄였다.
4. WMT 2014 English-to-German 및 English-to-French translation에서 기존 모델보다 적은 training cost로 우수한 BLEU 성능을 기록했다.
5. English constituency parsing에도 적용하여 translation 이외의 task로 일반화될 가능성을 보였다.

## 2. Related Work

<aside>
💡

Introduction에서 언급한 기존 연구들에 대해 어떻게 서술하는가

제안 방법의 차별성을 어떻게 표현하고 있는가

</aside>

논문에서는 별도의 `Related Work`라는 제목 대신 **Section 2: Background**에서 기존 연구와의 관계를 설명한다.

RNN, LSTM, GRU는 language modeling과 machine translation 같은 sequence modeling 및 transduction 문제에서 확립된 방법이었다. 하지만 recurrent computation은 이전 hidden state와 현재 입력으로 다음 hidden state를 생성하기 때문에 sequence position 사이의 계산을 병렬화할 수 없다.

이 한계를 줄이기 위해 Extended Neural GPU, ByteNet, ConvS2S처럼 convolution을 이용해 모든 위치의 hidden representation을 병렬로 계산하는 모델이 제안되었다. 그러나 convolution 기반 모델에서 입력과 출력의 임의 위치를 연결하는 데 필요한 연산 수는 위치 간 거리에 따라 증가한다. 논문은 ConvS2S에서는 선형적으로, ByteNet에서는 logarithmic하게 증가한다고 설명한다.

Self-attention은 하나의 sequence representation을 계산할 때 그 sequence의 서로 다른 위치를 연결하는 방법이다. Transformer 이전에도 reading comprehension, abstractive summarization, textual entailment, task-independent sentence representation 등에 사용되었다. 다만 기존 연구의 대다수는 self-attention을 recurrent network와 함께 사용했다.

Transformer의 차별점은 다음과 같이 정리할 수 있다.

\[
\text{기존: RNN 또는 CNN + Attention}
\]

\[
\text{Transformer: Attention 중심 구조}
\]

저자들은 Transformer를 input과 output representation을 계산할 때 sequence-aligned RNN이나 convolution을 사용하지 않고 전적으로 self-attention에 의존하는 최초의 transduction model로 제시한다.

## 3. 제안 방법론

<aside>
💡

Introduction에서 언급된 내용과 동일하게 작성되어 있는가

Introduction에서 언급한 제안 방법이 가지는 장점에 대한 근거가 있는가

제안 방법에 대한 설명이 구현 가능하도록 작성되어 있는가

</aside>

### Main Idea

Transformer는 기존 sequence-to-sequence model처럼 **encoder와 decoder**로 구성된다.

\[
\text{Input}
\rightarrow \text{Embedding + Positional Encoding}
\rightarrow \text{Encoder}
\rightarrow \text{Decoder}
\rightarrow \text{Linear + Softmax}
\rightarrow \text{Output Probabilities}
\]

#### Encoder

Encoder는 동일한 구조의 layer \(N=6\)개를 쌓는다. 각 layer에는 다음 두 sub-layer가 있다.

1. Multi-head self-attention
2. Position-wise fully connected feed-forward network

각 sub-layer에는 residual connection과 layer normalization을 적용한다.

\[
\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))
\]

Residual connection을 사용할 수 있도록 embedding layer와 모든 sub-layer의 출력 차원을 \(d_{model}=512\)로 통일한다.

#### Decoder

Decoder도 동일한 구조의 layer \(N=6\)개를 쌓지만, encoder layer의 output에 multi-head attention을 수행하는 sub-layer가 하나 더 있다.

1. Masked multi-head self-attention
2. Encoder-decoder multi-head attention
3. Position-wise fully connected feed-forward network

Decoder self-attention에는 masking을 적용한다. Output embedding을 한 position만큼 오른쪽으로 이동시키는 방식과 결합하여, position \(i\)의 prediction이 \(i\)보다 앞선 위치의 known output에만 의존하도록 한다. 이를 통해 autoregressive generation을 유지한다.

#### Scaled Dot-Product Attention

Attention의 입력은 query, key, value로 이루어진다. Query와 key의 차원을 \(d_k\), value의 차원을 \(d_v\)라고 하면 다음과 같이 계산한다.

\[
\mathrm{Attention}(Q,K,V)
=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

직관적으로 \(QK^T\)는 각 query와 key의 compatibility를 계산하고, softmax는 이를 value에 적용할 가중치로 바꾼다.

Additive attention과 dot-product attention은 이론적 complexity가 비슷하지만, dot-product attention은 matrix multiplication code로 구현할 수 있어 실제로 더 빠르고 공간 효율적이다. 다만 \(d_k\)가 크면 dot product의 크기도 커져 softmax가 gradient가 매우 작은 영역으로 들어갈 수 있다. 논문은 이를 완화하기 위해 dot product를 \(\sqrt{d_k}\)로 나눈다.

#### Multi-Head Attention

하나의 attention을 \(d_{model}\) 차원에서 수행하는 대신, query, key, value를 서로 다른 learned linear projection으로 \(h\)번 투영하고 각 attention을 병렬로 계산한다.

\[
\mathrm{MultiHead}(Q,K,V)
=\mathrm{Concat}(head_1,\ldots,head_h)W^O
\]

\[
head_i=\mathrm{Attention}(QW_i^Q,KW_i^K,VW_i^V)
\]

논문의 기본 설정은 \(h=8\), \(d_k=d_v=d_{model}/h=64\)이다. 각 head의 차원을 줄였기 때문에 전체 계산량은 full-dimensional single-head attention과 비슷하다. Multi-head attention은 서로 다른 position의 정보와 서로 다른 representation subspace의 정보를 함께 참고할 수 있게 한다.

#### Transformer에서 사용하는 세 가지 attention

1. **Encoder-decoder attention:** Query는 이전 decoder layer에서, key와 value는 encoder output에서 온다. Decoder의 각 position이 input sequence의 모든 position을 참고할 수 있다.
2. **Encoder self-attention:** Query, key, value가 모두 이전 encoder layer의 output에서 온다. Encoder의 각 position이 이전 encoder layer의 모든 position을 참고할 수 있다.
3. **Masked decoder self-attention:** Query, key, value가 decoder 내부에서 오지만, illegal connection을 mask하여 각 position이 현재 위치까지의 정보만 참고하도록 제한한다.

#### Position-wise Feed-Forward Network

각 encoder와 decoder layer에는 fully connected feed-forward network가 포함된다. 두 linear transformation 사이에 ReLU를 사용한다.

\[
\mathrm{FFN}(x)=\max(0,xW_1+b_1)W_2+b_2
\]

이 연산은 각 position에 독립적으로 동일하게 적용된다. 논문의 기본 모델에서 입력과 출력 차원은 \(d_{model}=512\), inner-layer 차원은 \(d_{ff}=2048\)이다.

#### Embedding과 Softmax

Input token과 output token은 learned embedding을 통해 \(d_{model}\) 차원의 vector로 변환한다. Decoder output은 learned linear transformation과 softmax를 거쳐 next-token probability로 변환된다. 논문은 두 embedding layer와 pre-softmax linear transformation 사이에 동일한 weight matrix를 공유하고, embedding weight에 \(\sqrt{d_{model}}\)을 곱한다.

#### Positional Encoding

Transformer에는 recurrence와 convolution이 없으므로 architecture만으로 token의 순서를 표현하지 못한다. 따라서 encoder와 decoder stack의 가장 아래에서 embedding에 positional encoding을 더한다. Positional encoding의 차원은 embedding과 같은 \(d_{model}\)이다.

\[
PE_{(pos,2i)}=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE_{(pos,2i+1)}=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

서로 다른 frequency의 sine과 cosine을 사용하는 방식이다. 저자들은 고정된 offset \(k\)에 대해 \(PE_{pos+k}\)를 \(PE_{pos}\)의 linear function으로 나타낼 수 있으므로 상대 위치를 학습하기 쉬울 것이라고 가정했다. Learned positional embedding도 실험했으며 두 방식은 거의 같은 결과를 보였다. 저자들은 training에서 본 길이보다 긴 sequence에도 extrapolate할 가능성을 고려해 sinusoidal version을 선택했다.

#### Training 설정

논문의 base model은 8개의 NVIDIA P100 GPU에서 약 12시간 동안 100,000 step 학습했고, big model은 300,000 step에 약 3.5일이 걸렸다. Optimizer는 Adam을 사용한다.

\[
\beta_1=0.9,\qquad \beta_2=0.98,\qquad \epsilon=10^{-9}
\]

Learning rate는 다음 schedule을 따른다.

\[
lrate=d_{model}^{-0.5}\cdot
\min(step\_num^{-0.5},\;step\_num\cdot warmup\_steps^{-1.5})
\]

\(warmup\_steps=4000\)으로 두어 초기에는 learning rate를 선형적으로 증가시키고 이후에는 step number의 inverse square root에 비례해 감소시킨다. Regularization으로 residual dropout, attention weight dropout, embedding과 positional encoding 합에 대한 dropout을 적용하고, label smoothing \(\epsilon_{ls}=0.1\)을 사용한다.

### Contribution

제안 방법의 핵심 장점은 **self-attention으로 병렬성과 long-range dependency modeling을 동시에 개선했다는 것**이다. 논문의 Section 4는 self-attention, recurrent layer, convolutional layer를 세 관점에서 비교한다.

| Layer Type | Complexity per Layer | Sequential Operations | Maximum Path Length |
|---|---:|---:|---:|
| Self-Attention | \(O(n^2d)\) | \(O(1)\) | \(O(1)\) |
| Recurrent | \(O(nd^2)\) | \(O(n)\) | \(O(n)\) |
| Convolutional | \(O(knd^2)\) | \(O(1)\) | \(O(\log_k n)\) |

Self-attention은 모든 position을 동시에 계산할 수 있으므로 필요한 sequential operation 수가 constant이고, 임의의 두 position이 직접 연결되므로 maximum path length도 constant이다. 또한 논문이 다룬 sentence representation의 일반적인 조건인 \(n<d\)에서는 self-attention layer가 recurrent layer보다 빠르다.

반면 self-attention의 complexity는 sequence length에 대해 quadratic인 \(O(n^2d)\)이다. 논문은 매우 긴 sequence를 효율적으로 다루기 위한 방법으로 각 output position이 input의 크기 \(r\)인 neighborhood만 보게 하는 restricted self-attention을 향후 연구 방향으로 제시한다. 이 경우 complexity는 \(O(rnd)\)로 줄지만 maximum path length는 \(O(n/r)\)로 늘어난다.

## 4. 실험 및 결과

<aside>
💡

Introduction에서 언급한 제안 방법의 장점을 검증하기 위한 실험이 있는가

</aside>

### Dataset

#### WMT 2014 English-to-German

- 약 450만 개의 sentence pair
- Byte-Pair Encoding 사용
- Source와 target이 공유하는 약 37,000 token vocabulary 사용

#### WMT 2014 English-to-French

- 약 3,600만 개의 sentence pair
- 32,000 word-piece vocabulary 사용

#### English Constituency Parsing

Transformer가 translation 외의 task에도 일반화되는지 평가하기 위해 Penn Treebank의 Wall Street Journal portion을 사용했다.

- WSJ-only setting: 약 40,000개의 training sentence
- Semi-supervised setting: 약 1,700만 개의 sentence를 사용한 high-confidence 및 BerkleyParser corpus

### Baseline

Machine translation에서는 당시의 주요 recurrent, convolutional, attention 기반 sequence model과 비교하였다.

- ByteNet
- Deep-Att + PosUnk
- GNMT + RL
- ConvS2S
- MoE
- 위 모델들의 ensemble 결과

Constituency parsing에서는 WSJ-only 및 semi-supervised setting의 기존 parser들과 비교했다. 논문은 task-specific tuning을 많이 하지 않은 Transformer 결과를 함께 제시한다.

### 결과

#### Machine Translation

WMT 2014 English-to-German에서 Transformer (base)는 27.3 BLEU, Transformer (big)는 28.4 BLEU를 기록했다. Big model은 당시 표에 제시된 기존 model과 ensemble의 성능을 넘어섰으며, 기존 best model보다 2 BLEU 이상 높은 결과였다.

WMT 2014 English-to-French에서는 Transformer (big)가 Table 2와 abstract 기준 41.8 BLEU를 기록했다. 기존 single model을 넘어섰고, training cost는 당시 state-of-the-art model의 일부 수준이었다.

| Model | EN-DE BLEU | EN-FR BLEU | EN-DE Training Cost (FLOPs) | EN-FR Training Cost (FLOPs) |
|---|---:|---:|---:|---:|
| Transformer (base) | 27.3 | 38.1 | \(3.3\times10^{18}\) | — |
| Transformer (big) | **28.4** | **41.8** | \(2.3\times10^{19}\) | \(1.4\times10^{20}\) |

원문에는 English-to-French 수치의 표기 차이가 있다. Abstract와 Table 2에는 Transformer (big)가 **41.8 BLEU**로 제시되지만, Section 6.1 본문에는 **41.0 BLEU**로 적혀 있다. 이 리뷰의 표는 Abstract와 Table 2의 수치인 41.8을 따른다.

#### Model Variation 및 Ablation

Table 3은 Transformer base의 여러 구성 요소를 바꾸어 English-to-German development set에서 효과를 비교한다.

- Single-head attention은 가장 좋은 head 설정보다 0.9 BLEU 낮았다.
- Head 수를 지나치게 크게 늘려도 성능이 감소했다.
- Attention key dimension \(d_k\)를 줄이면 model quality가 낮아졌다.
- Model을 크게 만들면 전반적으로 성능이 향상되었다.
- Dropout을 제거하면 성능이 저하되었다.
- Sinusoidal positional encoding을 learned positional embedding으로 바꾸었을 때 BLEU는 25.8에서 25.7로 거의 변하지 않았다.

이 결과는 성능이 단순히 head 수에 비례하지 않으며, representation dimension과 regularization이 중요하다는 것을 보여준다. Positional encoding 실험은 Transformer의 성능이 sinusoidal 방식 하나에만 의존하지 않음을 보여준다.

#### Constituency Parsing

Transformer는 WSJ-only setting에서 91.3 F1, semi-supervised setting에서 92.7 F1을 기록했다. Recurrent network를 사용하지 않고 task-specific tuning을 많이 하지 않았음에도 기존 모델과 경쟁력 있는 결과를 보였다. 이는 논문이 제안한 architecture가 machine translation 이외의 sequence task에도 적용될 수 있다는 근거다.

## 5. 결론 (배운점)

<aside>
💡

연구의 의의 및 한계점, 본인이 생각하는 좋았던/아쉬웠던 점 (배운점)

</aside>

본 논문의 핵심 의의는 sequence modeling에서 recurrence가 필수라는 전제를 벗어나, **attention을 sequence transduction model의 중심 연산으로 사용했다는 점**이다. 기존 RNN이 위치를 순서대로 처리했다면 Transformer는 각 위치가 같은 layer 안에서 다른 모든 위치를 직접 참고하도록 바꾸었다. 그 결과 sequence 안의 위치별 계산을 병렬화하면서 long-range dependency의 정보 전달 경로도 짧게 만들었다.

좋았던 점은 architecture의 장점을 최종 BLEU만으로 주장하지 않고, Section 4에서 computational complexity, sequential operation 수, maximum path length를 recurrent 및 convolutional layer와 비교해 설계의 근거를 제시한 것이다. Table 3의 variation 실험을 통해 attention head 수, attention dimension, model size, dropout, positional encoding 같은 구성 요소가 성능에 미치는 영향도 확인했다.

한계는 global self-attention의 연산량과 memory 사용량이 sequence length에 대해 quadratic하게 증가한다는 점이다. 논문도 매우 긴 sequence에서는 restricted self-attention이 필요할 수 있다고 보고 이를 향후 연구 과제로 남겼다. 또한 recurrence와 convolution을 제거했기 때문에 순서 정보를 positional encoding으로 별도로 주입해야 한다.

정리하면 이 논문에서 배울 수 있는 가장 중요한 관점은 다음과 같다.

\[
\boxed{\text{Transformer의 혁신은 attention을 보조 장치가 아니라 sequence model의 중심 연산으로 만든 것이다.}}
\]

RNN의 recurrence를 개선하는 데 머무르지 않고 **“sequence를 반드시 순차적으로 계산해야 하는가?”**라는 구조적 전제를 다시 질문했고, translation과 parsing 실험으로 그 대안이 실제로 작동함을 보였다.
