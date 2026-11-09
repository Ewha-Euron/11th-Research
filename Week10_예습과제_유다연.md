# Week10_예습과제_유다연

## 📄 [GPT-3] Language Models are Few-Shot Learners (Brown et al., 2020)

### 1. Introduction

<aside>
💡

논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가

논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

</aside>

- 논문: [arXiv 2005.14165](https://arxiv.org/pdf/2005.14165) (Brown et al., OpenAI, NeurIPS 2020)
- 한 줄 요약: LM을 175B까지 키우면 fine-tuning 없이 prompt에 예시 몇 개만 넣어도(in-context learning) 여러 NLP task를 꽤 잘 푼다.

#### 논문이 다루는 분야

- NLP의 pre-trained language model, 그중에서도 gradient update 없이 task를 푸는 **in-context learning (zero / one / few-shot)**
- 지금까지 흐름: word vector(word2vec, GloVe) → 문맥을 반영하는 RNN 표현 → Transformer LM을 pre-train하고 task별로 fine-tuning (GPT-1, BERT)
- 구조는 task-agnostic해졌지만 여전히 **task마다 labeled dataset과 fine-tuning이 필요하다**는 게 문제의식

#### 해당 task에서 기존 연구 한계점

1. **task마다 큰 라벨 데이터가 필요함.** 새 task마다 수천~수십만 개 예시가 필요해서 실용성이 떨어진다. 문법 교정이나 추상 개념 예시 만들기처럼 데이터를 모으기 애매한 task도 많음.
2. **좁은 분포에 과적합.** 큰 모델을 좁은 학습 분포로 fine-tuning하면 spurious correlation을 배우기 쉬워서, benchmark 점수가 실제 능력보다 과장되고 분포 밖 일반화가 약해진다.
3. **사람이 배우는 방식과 다름.** 사람은 짧은 지시나 예시 몇 개만 보고도 새 task를 하는데 지금 시스템은 그런 유연함이 없다.

- GPT-2 때도 자연어 지시 + 예시로 푸는 시도가 있었지만 성능이 fine-tuning에 한참 못 미쳤음 (Natural Questions 4%, CoQA 55 F1로 SOTA보다 35점 이상 낮음)

#### 논문의 contributions

- **175B 파라미터 autoregressive LM, GPT-3** 학습 (기존 non-sparse LM보다 10배 큼)
- fine-tuning 없이 **zero-shot / one-shot / few-shot** 세 설정으로 20개 이상의 NLP 데이터셋과 새로 만든 task들에서 평가
- 모델이 클수록 context 안의 예시를 더 잘 활용함 (zero-shot과 few-shot 성능 차이가 모델 크기에 따라 벌어짐) → **큰 모델일수록 더 좋은 meta-learner**
- TriviaQA, LAMBADA 등은 few-shot만으로 fine-tuning SOTA에 근접하거나 넘음. 반대로 ANLI 같은 NLI나 RACE, QuAC 같은 일부 독해는 여전히 약하다는 것도 같이 보고함
- benchmark data contamination 분석, 사회적 영향(오용, bias, 에너지) 논의

#### 기존 문제점에 대응되는가 (내 생각)

- task마다 데이터와 fine-tuning이 필요하다 → fine-tuning을 아예 안 하고 prompt로 task를 정의하니까 직접 대응함
- 좁은 분포 과적합 → weight를 안 건드리니 task별 spurious feature를 학습할 여지 자체가 줄어듦
- 다만 성능은 아직 fine-tuning을 대체한다기보다 "scale을 키우면 가능성이 보인다"에 가깝고, 논문도 이걸 인정함
- Figure 1.1의 meta-learning 그림이 핵심: outer loop = 비지도 pre-training(SGD로 weight 학습), inner loop = in-context learning(한 시퀀스 안에서 forward pass만으로 task에 적응)

### 2. Related Work

<aside>
💡

Introduction에서 언급한 기존 연구들에 대해 어떻게 서술하는가

제안 방법의 차별성을 어떻게 표현하고 있는가

</aside>

Related Work는 7장(맨 뒤)에 있고, 기존 연구를 7갈래로 나눠서 각각 GPT-3와 어떤 관계인지 한두 문장씩 정리하는 방식이다.

| 분야 | 기존 연구 | GPT-3와의 관계 |
| --- | --- | --- |
| 모델 scale 키우기 | Transformer 213M → BERT 300M → GPT-2 1.5B → Megatron 8B → T5 11B → Turing-NLG 17B | 파라미터와 연산을 같이 키우는 방식을 그대로 따르고 10배 더 키움 (MoE처럼 파라미터만 늘리는 방식, Universal Transformer처럼 연산만 늘리는 방식도 소개) |
| Scaling law | Kaplan et al. (2020) 등: 모델이 커질수록 loss가 power-law로 감소 | 175B에서도 이 경향이 대체로 이어지고, downstream task도 (전부는 아니지만) 부드럽게 오른다고 확인 |
| 작게 만들기 | ALBERT, knowledge distillation | 반대 방향이지만 상호보완적 → 거대 모델의 latency와 메모리를 줄이는 데 쓸 수 있다고 봄 |
| Meta-learning | Matching Networks, RL², MAML 등 | context에 예시를 넣는 방식은 RL²와 구조적으로 가장 비슷 (inner loop는 weight 업데이트 없이 activation으로 적응, outer loop는 LM pre-training) |
| 자연어로 task 지시 | MQAN (McCann et al.), GPT-2, T5의 text-to-text | T5는 자연어 지시를 multi-task **fine-tuning**에 썼고, GPT-3는 weight 업데이트 없는 in-context learning에 씀 |
| Multi-task learning | 여러 task를 섞어서 같이 fine-tuning | 데이터셋을 직접 모으고 커리큘럼을 짜야 함. GPT-3는 대규모 텍스트 예측 안에 넓은 task 분포가 자연스럽게 들어 있다고 봄 |
| 알고리즘 개선 | BERT(denoising 양방향), prefixLM, encoder-decoder(BART, T5), XLNet(permutation), ALBERT 등 | fine-tuning 성능은 많이 올렸지만, GPT-3는 in-context learning에 집중하고 구현을 단순하게 하려고 **순수 autoregressive LM**을 유지. 둘을 합치는 건 future work |

- **차별성 표현**: 기존 연구 = "구조·학습 기법 개선 + task별 fine-tuning", GPT-3 = "구조는 GPT-2 그대로, scale만 키워서 in-context learning을 체계적으로 측정". 방법의 새로움이 아니라 **scale과 평가 방식**으로 차별성을 잡는다.
- Intro에서 말한 GPT-2의 in-context learning 한계를 Related Work에서 meta-learning 관점으로 다시 정리해 줘서, 왜 scale을 키우면 inner loop가 좋아질 거라 기대하는지 연결이 된다.
- 아쉬운 점: 맨 뒤에 있고 대부분 한 문장씩 나열이라, 기존 방법과의 실질적인 비교는 실험의 baseline 숫자로만 드러남.

### 3. 제안 방법론

<aside>
💡

Introduction에서 언급된 내용과 동일하게 작성되어 있는가

Introduction에서 언급한 제안 방법이 가지는 장점에 대한 근거가 있는가

제안 방법에 대한 설명이 구현 가능하도록 작성되어 있는가

</aside>

방법 자체는 새 구조가 아니라 "GPT-2 구조 + 훨씬 큰 scale + 더 좋은 데이터 + zero/one/few-shot 평가 틀"이다.

#### Main Idea

**(1) Fine-tuning 대신 in-context learning**

weight는 고정하고 입력(prompt)에 task 설명과 예시를 넣어서 task를 정의한다. (Figure 2.1의 영어→프랑스어 번역 예시)

| 설정 | 입력 | 특징 (논문 2장) |
| --- | --- | --- |
| Fine-tuning (FT) | task별 라벨 데이터로 weight 업데이트 | 많은 benchmark에서 가장 강하지만 task마다 큰 데이터셋이 필요하고, 분포 밖 일반화가 약하며 spurious feature를 배울 수 있음 → **이 논문에선 사용 안 함** |
| Few-shot (FS) | task 설명 + 예시 K개 (보통 10~100개, context 2048 토큰에 들어가는 만큼) | task별 데이터가 훨씬 적게 필요. 대신 아직 fine-tuned SOTA보다 낮은 경우가 많음 |
| One-shot (1S) | task 설명 + 예시 1개 | 사람한테 task를 알려줄 때와 가장 비슷한 방식 |
| Zero-shot (0S) | task 설명만 | 가장 편하고 spurious correlation 위험이 적지만 가장 어려움 (사람도 예시 없이는 헷갈리는 task가 있음) |

**(2) 모델 구조**

- GPT-2와 동일 (modified initialization, pre-normalization, reversible tokenization). 차이는 layer마다 **dense attention과 locally banded sparse attention을 번갈아** 쓴 것 (Sparse Transformer와 비슷)
- scale 효과를 보려고 125M~175B까지 8개 크기를 학습. context window는 전부 2048 토큰이고 전부 300B 토큰으로 학습

| 모델 | 파라미터 | layer | d_model | head | d_head | batch (토큰) | LR |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GPT-3 Small | 125M | 12 | 768 | 12 | 64 | 0.5M | 6.0e-4 |
| GPT-3 Medium | 350M | 24 | 1024 | 16 | 64 | 0.5M | 3.0e-4 |
| GPT-3 Large | 760M | 24 | 1536 | 16 | 96 | 0.5M | 2.5e-4 |
| GPT-3 XL | 1.3B | 24 | 2048 | 24 | 128 | 1M | 2.0e-4 |
| GPT-3 2.7B | 2.7B | 32 | 2560 | 32 | 80 | 1M | 1.6e-4 |
| GPT-3 6.7B | 6.7B | 32 | 4096 | 32 | 128 | 2M | 1.2e-4 |
| GPT-3 13B | 13B | 40 | 5140 | 40 | 128 | 2M | 1.0e-4 |
| **GPT-3 175B** | **175B** | **96** | **12288** | **96** | **128** | **3.2M** | **0.6e-4** |

- 모델이 클수록 batch는 크게, LR은 작게 (gradient noise scale을 보고 batch 크기 결정)
- GPT-3 Small(12층, 768차원)은 지난주 BERT-BASE와 크기가 거의 같아서 비교하기 좋음
- 모델을 depth, width 양쪽으로 쪼개서(model parallelism) Microsoft의 V100 GPU 클러스터에서 학습

**(3) 학습 데이터**

Common Crawl을 그대로 쓰지 않고 품질을 올리는 3단계를 거친다.

1. 고품질 reference corpus(WebText 등)와 비슷한 문서만 남도록 분류기로 필터링
2. 문서 단위 fuzzy deduplication (중복 제거, validation set 오염 방지)
3. 고품질 데이터셋(WebText2, Books1, Books2, Wikipedia)을 섞음

| 데이터셋 | 토큰 수 | 학습 mix 비중 | 300B 토큰 학습 시 epoch |
| --- | --- | --- | --- |
| Common Crawl (filtered) | 410B | 60% | 0.44 |
| WebText2 | 19B | 22% | 2.9 |
| Books1 | 12B | 8% | 1.9 |
| Books2 | 55B | 8% | 0.43 |
| Wikipedia | 3B | 3% | 3.4 |

- Common Crawl은 압축 텍스트 45TB → 필터링 후 570GB
- 크기에 비례해서 뽑지 않고 품질 높은 데이터를 더 자주 뽑음 (Wikipedia 3.4 epoch vs Common Crawl 0.44 epoch) → 약간의 overfitting을 감수하고 품질을 택함

**(4) 학습·평가 세부**

- Adam(β1 = 0.9, β2 = 0.95, ε = 1e-8), gradient norm clipping 1.0, 260B 토큰에 걸쳐 cosine decay(최대값의 10%까지), 처음 375M 토큰 linear warmup, weight decay 0.1
- few-shot 예시는 각 task의 training set에서 랜덤으로 K개 뽑아서 평가 예시 앞에 붙임
- 객관식: 보기마다 LM likelihood를 비교 (토큰 수로 정규화. ARC, OpenBookQA, RACE는 "Answer:" 뒤 무조건부 확률로 나눠서 정규화)
- 이진 분류: "True" / "False"처럼 의미 있는 단어를 보기로 줘서 객관식처럼 처리
- 자유 생성: beam search (beam 4, length penalty α = 0.6) 후 F1, BLEU, EM으로 채점

#### Contribution

- 기여는 방법론보다는 **"scale만 키워도 in-context learning 능력이 생기고 커진다"는 실증**
- zero/one/few-shot을 같은 모델로 비교하는 평가 틀을 정리함
- 체크리스트로 보면
    - Intro와 일치하는가 → O. Intro의 가설(fine-tuning 없이 context로 task 수행, scale이 커지면 in-context learning도 좋아짐)을 "8개 크기 × 3개 설정" 실험 설계로 그대로 옮김
    - 장점의 근거가 있는가 → 방법론 장에서는 설정별 장단점을 정성적으로만 정리하고, 실제 근거는 실험(Figure 1.2, 1.3과 3장 결과)에서 보여줌
    - 구현 가능한가 → 구조, 하이퍼파라미터, 데이터 mix, 평가 방식은 표로 꽤 자세히 공개함. 하지만 sparse attention 패턴 세부, Books1/Books2의 정체, 필터링 분류기 세부는 불명확하고, 학습 compute가 약 3.14×10²³ FLOPs(3,640 petaflop/s-days)라 사실상 재현은 불가능

### 4. 실험 및 결과

<aside>
💡

Introduction에서 언급한 제안 방법의 장점을 검증하기 위한 실험이 있는가

</aside>

9개 카테고리, 20개 이상 데이터셋에서 8개 크기 모델을 zero/one/few-shot으로 평가했고, 일부 task는 few-shot만으로 fine-tuned SOTA에 근접하거나 넘었다.

#### Dataset

| 카테고리 | 데이터셋 |
| --- | --- |
| Language modeling · cloze · completion | PTB, LAMBADA, HellaSwag, StoryCloze |
| Closed-book QA | Natural Questions, WebQuestions, TriviaQA |
| 번역 | WMT'14 En↔Fr, WMT'16 En↔De, WMT'16 En↔Ro |
| Winograd 류 | WSC273, WinoGrande |
| 상식 추론 | PIQA, ARC (Easy / Challenge), OpenBookQA |
| 독해 | CoQA, DROP, QuAC, SQuAD 2.0, RACE |
| SuperGLUE | BoolQ, CB, COPA, RTE, WiC, WSC, MultiRC, ReCoRD |
| NLI | RTE, ANLI (R1~R3) |
| 합성·정성 task | 산수, 단어 섞기/복원, SAT 유추, 뉴스 기사 생성, 새 단어 사용, 영어 문법 교정 |

#### Baseline

- 각 task의 **fine-tuned SOTA** (BERT-Large, RoBERTa, T5-11B 등)
- closed-book QA: 같은 closed-book 조건의 fine-tuned T5-11B(+SSM), 그리고 retrieval을 쓰는 open-domain 모델(RAG)
- 번역: supervised SOTA와 unsupervised NMT (XLM, MASS, mBART)
- GPT-3 자신의 작은 버전들(125M~13B) → scale 효과 비교

#### 결과

**잘 된 task**

| Task (지표) | Zero-shot | One-shot | Few-shot | 비교 대상 |
| --- | --- | --- | --- | --- |
| LAMBADA (acc) | 76.2 | 72.5 | 86.4 | 기존 SOTA 68.0 |
| TriviaQA (acc) | 64.3 | 68.0 | 71.2 | closed-book fine-tuned T5-11B+SSM 60.5, open-domain RAG 68.0 |
| CoQA (F1) | 81.5 | 84.0 | 85.0 | fine-tuned SOTA 90.7 |

- PTB zero-shot perplexity 20.5로 기존 SOTA 35.8보다 크게 낮음
- SuperGLUE: task당 예시 32개 few-shot으로 test 평균 71.8 → fine-tuned BERT-Large(69.0)보다는 높고 SOTA(89.0)보다는 낮음
- 산수 (175B few-shot): 2자리 덧셈 100%, 2자리 뺄셈 98.9%, 3자리 덧셈 80.4%, 3자리 뺄셈 94.2%. 4자리는 25~26%, 5자리는 9~10%로 자릿수가 늘수록 급격히 떨어짐
- 뉴스 기사 생성: 175B가 쓴 약 200단어 기사를 사람이 구별한 정확도 52% (거의 찍는 수준). 일부러 품질을 낮춘 control 모델은 86%
- Figure 1.3: 42개 accuracy 기반 benchmark 평균이 모델 크기에 따라 zero/one/few-shot 모두 오르고, few-shot이 가장 가파르게 오름 → Intro의 "큰 모델이 더 좋은 meta-learner" 주장의 근거

**약한 task**

- NLI: ANLI는 13B 이하에서는 거의 찍는 수준이고, 175B few-shot에서도 R3 기준 40% 정도 (3지선다라 chance가 33%)
- WiC(두 문장에서 같은 단어가 같은 뜻인지): few-shot에서도 chance 수준(50% 안팎)
- 독해: DROP은 few-shot F1 36.5 vs fine-tuned SOTA 89.1처럼 차이가 크고, RACE, QuAC도 약함
- 번역: 다른 언어 → 영어는 기존 unsupervised NMT보다 좋지만 영어 → 다른 언어는 약함 (학습 데이터의 약 93%가 영어)

**Data contamination (4장)**

- test set이 웹 학습 데이터에 섞였을 수 있어서 n-gram(최대 13) overlap 기준으로 "깨끗한" 부분집합을 따로 만들어 성능을 비교
- 대부분은 차이가 거의 없었고, PIQA와 Winograd만 영향 가능성이 있어서 결과에 별표(*) 표시. 일부 language modeling benchmark와 Children's Book Test는 아예 결과에서 뺌
- 필터링 버그 때문에 책처럼 긴 문서에서 겹침을 제대로 못 지웠는데, 비용 때문에 재학습은 못 했다고 밝힘

**장점을 검증하는 실험이 있는가 (내 생각)**

- O. (1) 8개 크기 × zero/one/few-shot으로 scale과 in-context learning의 관계를 보여주고, (2) fine-tuning 없이도 일부 task에서 SOTA에 근접함을 보임
- 다만 같은 175B 모델을 fine-tuning한 결과는 없어서, fine-tuning 대비 얼마나 좋은지/나쁜지는 다른 모델의 SOTA와 비교해서만 판단할 수 있음

### 5. 결론 (배운점)

<aside>
💡

연구의 의의 및 한계점, 본인이 생각하는 좋았던/아쉬웠던 점 (배운점)

</aside>

scale만으로 fine-tuning 없는 범용 LM의 가능성을 보여준 논문이지만, 왜 되는지와 재현성은 숙제로 남았다.

#### 의의

- "모델을 키우면 fine-tuning 없이 prompt만으로 여러 task를 푼다"는 걸 처음 대규모로 보여줌 → 지금처럼 prompt로 LLM을 쓰는 방식(prompt engineering, in-context learning)의 출발점
- scale에 따라 성능이 꽤 예측 가능하게 오른다는 걸 loss뿐 아니라 downstream task에서도 확인

#### 한계 (논문 5·6장)

- 긴 글을 쓰면 반복, 일관성 붕괴, 자기모순, 뜬금없는 문장이 나옴
- 상식 물리에 약함 ("치즈를 냉장고에 넣으면 녹을까?" 같은 질문)
- autoregressive라 **양방향성이 없음** → WiC, ANLI처럼 두 문장을 비교하거나 다시 읽어야 하는 task에 약함
- 모든 토큰을 같은 가중치로 예측하는 학습 목표라 뭐가 중요한지 모르고, 텍스트만 봐서 실제 세계와 연결(grounding)이 없음
- pre-training의 sample efficiency가 낮음 (사람이 평생 보는 것보다 훨씬 많은 텍스트로 학습)
- few-shot이 inference 때 정말 "새 task를 배우는" 건지, 학습 때 본 task를 "알아보는" 건지 애매함 → 논문도 불확실하다고 인정
- 추론 비용이 크고(distillation 제안), 해석이 어렵고 calibration이 안 되며, 학습 데이터의 bias를 그대로 가짐
- 사회적 영향: 가짜 뉴스·스팸·피싱 같은 오용 가능성, bias (직업 388개 중 83%가 남성 지칭어와 더 자주 이어지고, 종교 중 Islam이 "violent", "terrorism" 같은 단어와 더 자주 같이 나옴), 학습 에너지 소비 (대신 학습 후 100페이지 생성은 약 0.4kWh)

#### 좋았던 점

- 8개 크기로 scale 효과를 체계적으로 보여준 실험 설계. 같은 조건에서 크기만 바꾼 곡선이라 설득력이 큼
- 잘 된 결과뿐 아니라 안 되는 task, 데이터 오염(버그까지), bias, 에너지 문제를 숨기지 않고 정리함

#### 아쉬웠던 점

- 방법론적인 새로움은 거의 없고, 3.14×10²³ FLOPs 규모라 다른 연구자가 재현하거나 검증하기 어려움
- few-shot 성능이 예시 선택, 순서, prompt 형식에 얼마나 민감한지 분석이 부족함. 같은 task라도 prompt를 어떻게 쓰느냐에 따라 결과가 달라질 것 같은데 이 부분은 거의 안 다룸
- in-context learning이 "왜" 되는지 메커니즘 설명이 없음

#### 배운점

- 지난주 BERT는 "양방향 + task마다 출력층 붙여서 fine-tuning"이었고, GPT-3는 "단방향 + 엄청난 scale + fine-tuning 없음"으로 범용성을 택함. 같은 Transformer인데 쓰는 방식이 완전히 다르다는 게 인상적이었음
- GPT-3 한계 섹션에 "양방향성이 없어서 WiC, ANLI에 약하다"는 말이 직접 나오는 게 재밌었다. BERT 복습과제 Table 5에서 봤던 양방향성의 효과가 여기서 다시 나옴
- 같이 본 Knowledge Conflict 논문도 결국 GPT-3 이후 "context로 정보를 넣어 주는" 사용 방식이 일반화되면서 생긴 문제(context 정보 vs 파라미터 지식 충돌)라서 흐름이 이어진다고 느낌

---

## 📄 [Knowledge Conflict] Which Knowledge Should LLMs Trust? Uncovering a Generalizable Steering Direction Across Knowledge Conflicts (Lou et al., 2026)

### 1. Introduction

<aside>
💡

논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가

논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

</aside>

- 논문: [arXiv 2610.20067](https://arxiv.org/pdf/2610.20067) (Yibin Lou, Yi Hu, Jianhui Chen, Chenpeng Wang, Muhan Zhang, Liangming Pan)
- ※ 이 논문은 Introduction만 리뷰
- 한 줄 요약: LLM이 "파라미터 지식 vs context 지식" 중 뭘 따를지는 residual stream의 방향 하나로 표현되고, 이 방향으로 steering하면 여러 task에서 지식 선택을 조절할 수 있다.

#### 논문이 다루는 분야

- LLM의 **knowledge conflict**: 학습 때 파라미터에 저장된 지식(parametric knowledge)과 prompt로 주어진 외부 지식(contextual knowledge)이 다를 때 어느 쪽을 따를지 고르는 문제 (knowledge selection)
- 필요성: RAG(검색 문서), 여러 턴의 대화 기록(memory), 자연어로 준 규칙(rule-following)처럼 실제 응용에서는 외부 context를 써야 하는데, 충돌이 생기면 모델이 엉뚱한 쪽을 믿는 경우가 많음
- 방법론으로는 activation steering 분야: 모델 내부 표현(residual stream)에 방향 벡터를 더해서 행동을 조절

#### 해당 task에서 기존 연구 한계점

| 접근 | 예시 | 한계 |
| --- | --- | --- |
| 입력 수준 | prompt·instruction 설계 (Choi et al., 2025; Merth et al., 2024), 검색 문서를 구조화·라벨링하는 RAG 템플릿 (Cattan et al., 2025) | 모델 내부에서 두 지식이 어떻게 경합하는지 안 보이고, context와 파라미터 의존도를 간접적으로만 조절 |
| 내부 구성요소 | attention 조작 (Jin et al., 2024; Li et al., 2025), SAE feature steering (Zhao et al., 2025a), decoding 시 토큰 분포 조정 (Bi et al., 2025; Wang et al., 2025a) | 대부분 단순한 factual 데이터셋에서만 검증. task마다 attribution, feature 탐색, decoding 수정이 필요해서 옮기는 비용이 크고, 검색 문서·긴 대화 기록·명시적 규칙 같은 복잡한 설정 평가가 부족 |

- 정리하면 기존 방법은 (1) 내부 작동이 잘 안 보이고 제어가 제한적이고, (2) 단순 factual setting 위주이며, (3) task마다 다시 맞춰야 해서 비쌈

#### 논문의 contributions

- 출발점: truthfulness (Li et al., 2024), refusal (Arditi et al., 2024), safety, emotion 같은 고수준 개념이 residual stream의 **선형 방향**으로 표현된다는 선행 연구 → "지식 선택도 저차원 축 하나로 잡을 수 있고, 그게 task를 넘어 일반화될까?"라는 질문
- 방법 개요: clean prompt(충돌 없음)와 conflict prompt(충돌 있음) 쌍에서 방향을 추출 → 두 표현의 cosine similarity(각도 분리)로 개입할 layer 선택 → 부호가 있는 계수(signed coefficient)로 steering
- Contribution 3가지
    1. 통제된 factual-conflict 벤치마크에서 residual-stream 방향 하나를 찾고, 이 방향으로 parametric ↔ contextual 지식 사이를 **양방향·연속적으로** 조절할 수 있음을 4개 모델 계열에서 보임 (Figure 1: 계수에 따라 답이 파라미터 쪽 ↔ context 쪽으로 부드럽게 바뀌고, 중간에서는 섞이거나 망설이는 답이 나옴)
    2. 같은 방향이 context 형식, 지식 카테고리, 데이터셋이 달라져도 작동함
    3. 재학습 없이 복잡한 RAG, memory, rule-following task로 옮겨도 성능이 오름 → 외부 지식을 받아들이는 데 공통으로 쓰이는 방향이 있다는 시사점
- 참고로 Intro에는 이름이 안 나오지만 3.2절 기준 모델은 Llama-3.1-8B-Instruct, Qwen3-8B, Gemma-2-9B-IT, Mistral-7B-Instruct-v0.3이고, 통제 실험은 NQ-Swap, MacNoise, ParaConflict, CounterFact, MemoTrap, 응용 task는 RGB, CRAG, LoCoMo(-Plus), LongMemEval, IFEval, FollowBench

#### 기존 문제점에 대응되는가 (내 생각)

- 내부가 안 보인다 → 내부 표현(residual stream)의 방향을 직접 뽑아서 씀 → 대응 O
- 단순 factual 위주 평가 → RAG, memory, rule-following까지 평가 → 대응 O
- task별 적응 비용 → 단순 충돌 데이터에서 한 번 뽑은 방향을 재학습 없이 전이 → 대응 O
- 궁금한 점: 제목은 "어떤 지식을 믿어야 하나"인데 Intro에서 강조하는 결과는 주로 context를 더 잘 쓰게 만드는 쪽. context가 틀렸을 때(잘못된 검색 결과 등) 언제, 어느 방향으로, 얼마나 steering할지 정하는 방법은 Intro에 없어서 본문에서 확인해 봐야 할 것 같음
- GPT-3에서 시작된 "prompt(context)로 task와 정보를 준다"는 사용 방식이 RAG, 메모리, 에이전트로 커지면서 생긴 문제라 이번 주 두 논문이 자연스럽게 이어짐
