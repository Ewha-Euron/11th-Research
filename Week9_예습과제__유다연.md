# Week9_예습과제__유다연

---

# [BERT] BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (전체 리뷰)

- Devlin, Chang, Lee, Toutanova (Google AI Language), NAACL 2019 / https://arxiv.org/pdf/1810.04805

## 1. Introduction

> 💡 논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가
> 논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

### 논문이 다루는 분야
- **언어 모델 사전학습(Pre-training)** 과 이를 다운스트림 NLP task(문장 수준: NLI, paraphrase / 토큰 수준: NER, QA)에 적용하는 방법.
- 사전학습 표현을 적용하는 방식은 두 가지다.
  - **Feature-based**: 사전학습 표현을 추가 feature로 사용 (ELMo)
  - **Fine-tuning**: 최소한의 task별 파라미터만 추가하고 전체를 미세 조정 (OpenAI GPT)

### 해당 task에서 기존 연구 한계점
- 두 방식 모두 **단방향(unidirectional) 언어 모델**을 사용한다.
  - GPT: left-to-right 구조라 각 토큰이 이전 토큰에만 attention 가능.
  - ELMo: left-to-right와 right-to-left를 따로 학습해 이어 붙인 **얕은(shallow) 양방향**.
- 단방향 제약은 문장 수준 task에서 sub-optimal이고, 양쪽 문맥이 모두 중요한 **토큰 수준 task(QA 등)에서는 특히 치명적**이다.

### 논문의 contributions
- **Masked Language Model(MLM)**: 입력 토큰 일부를 가리고 원래 단어를 맞추게 해, 왼쪽·오른쪽 문맥을 함께 보는 **깊은 양방향 Transformer**를 사전학습할 수 있게 했다.
- **Next Sentence Prediction(NSP)**: 문장 쌍 관계 표현을 함께 학습한다.
- 사전학습 표현이 무거운 task별 구조 설계의 필요성을 줄인다는 것을 보였다. BERT는 fine-tuning 기반으로 문장·토큰 수준 task 모두에서 SOTA를 달성한 첫 모델이다.
- **11개 NLP task에서 SOTA** 달성 (코드와 모델 공개).

## 2. Related Work

> 💡 Introduction에서 언급한 기존 연구들에 대해 어떻게 서술하는가
> 제안 방법의 차별성을 어떻게 표현하고 있는가

- **Unsupervised Feature-based Approaches**
  - word2vec, GloVe 등 단어 임베딩 → 문장·문단 임베딩으로 확장.
  - **ELMo**: left-to-right / right-to-left LM의 문맥 표현을 결합한 문맥 의존 임베딩. 하지만 깊은 양방향이 아니라 **feature-based** 방식이다.
- **Unsupervised Fine-tuning Approaches**
  - 레이블 없는 텍스트로 문장/문서 인코더를 사전학습한 뒤 fine-tuning한다. 처음부터 학습할 파라미터가 적다는 장점이 있다.
  - **OpenAI GPT**: GLUE에서 SOTA를 냈지만 **left-to-right** Transformer.
- **Transfer Learning from Supervised Data**: NLI, 기계번역 등 대규모 지도학습 데이터에서의 전이. CV에서 ImageNet 사전학습 모델을 fine-tuning하는 것과 같은 맥락이다.
- **차별성**: BERT는 **fine-tuning 방식이면서 모든 층에서 양방향 문맥을 함께 보는 유일한 구조**임을 그림 3(BERT vs GPT vs ELMo)으로 명확히 대비한다.

## 3. 제안 방법론

> 💡 Introduction에서 언급된 내용과 동일하게 작성되어 있는가
> Introduction에서 언급한 제안 방법이 가지는 장점에 대한 근거가 있는가
> 제안 방법에 대한 설명이 구현 가능하도록 작성되어 있는가

### Main Idea
**프레임워크: Pre-training → Fine-tuning 두 단계**
- 같은 구조로 사전학습한 뒤, 모든 파라미터를 각 task 데이터로 fine-tuning한다. task별로 바뀌는 것은 출력층뿐이다.

**모델 구조**
- 다층 양방향 **Transformer encoder**.
  - BERT_BASE: L=12, H=768, A=12, 110M 파라미터 (GPT와 비교를 위해 같은 크기)
  - BERT_LARGE: L=24, H=1024, A=16, 340M 파라미터

**입력 표현**
- WordPiece 임베딩(30,000 vocab).
- 모든 입력의 첫 토큰은 **[CLS]**. 마지막 hidden state를 분류 task의 문장 표현으로 사용한다.
- 문장 쌍은 **[SEP]** 으로 구분하고 segment embedding(A/B)을 더한다.
- 입력 = Token embedding + Segment embedding + Position embedding

**Pre-training Task 1: Masked LM**
- 입력 토큰의 **15%** 를 선택해 원래 토큰을 예측한다.
- fine-tuning 시에는 [MASK]가 없으므로 불일치를 줄이기 위해 선택된 토큰 중
  - 80% → [MASK], 10% → 임의 토큰, 10% → 원래 토큰 유지
- 마지막 hidden vector에 softmax를 붙여 cross-entropy로 학습한다.

**Pre-training Task 2: Next Sentence Prediction**
- 문장 A, B에 대해 50%는 실제 다음 문장(IsNext), 50%는 임의 문장(NotNext).
- [CLS] 표현 C로 이진 분류한다. QA, NLI처럼 문장 관계가 중요한 task에 도움이 된다.

**사전학습 데이터**
- BooksCorpus(800M words) + English Wikipedia(2,500M words).
- 긴 연속 시퀀스를 위해 섞인 문장 단위 코퍼스가 아닌 **문서 단위 코퍼스**를 사용한다.

**Fine-tuning**
- 입력·출력만 task에 맞게 연결: 문장 쌍(paraphrase, entailment, QA), 단일 문장(분류, 태깅).
- 토큰 표현 → 토큰 수준 task, [CLS] 표현 → 분류 task.
- 사전학습에 비해 저렴하다(TPU 1시간, GPU 몇 시간이면 재현 가능).

### Contribution
- Introduction에서 지적한 "단방향 제약"을 MLM으로 정면으로 해결한다. 방법론이 문제 정의와 일관된다.
- 장점(깊은 양방향, task 구조 단순화)의 근거는 ablation(NSP 제거, LTR 모델 비교)으로 뒷받침된다.
- 모델 크기, masking 비율, 데이터, 학습 하이퍼파라미터가 모두 명시되어 있고 코드가 공개되어 **재현 가능성**이 높다.

## 4. 실험 및 결과

> 💡 Introduction에서 언급한 제안 방법의 장점을 검증하기 위한 실험이 있는가

### Dataset
- **GLUE** (MNLI, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE)
- **SQuAD v1.1 / v2.0** (질의응답, 답변 span 예측 / 답 없음 포함)
- **SWAG** (상식 추론, 4개 후보 중 이어질 문장 선택)
- **CoNLL-2003 NER** (feature-based 방식 실험)

### Baseline
- OpenAI GPT, ELMo 기반 모델(BiLSTM+ELMo+Attn), 기존 각 task별 SOTA, 사람 성능(SQuAD, SWAG).

### 결과
- **GLUE**: BERT_LARGE 평균 **80.5**(+7.7%p). MNLI 86.7%(+4.6%p). 동일 크기 GPT보다 BERT_BASE가 높아 구조 차이(양방향)의 효과를 보여준다.
- **SQuAD v1.1**: Test F1 **93.2**(앙상블+TriviaQA), 단일 모델도 기존 최고 앙상블을 넘는다.
- **SQuAD v2.0**: F1 **83.1** (+5.1%p).
- **SWAG**: 86.3% — ESIM+ELMo보다 +27.1%p, GPT보다 +8.3%p.
- **Ablation – Pre-training task**
  - NSP 제거 → QNLI, MNLI, SQuAD 성능 하락 → NSP의 효과.
  - LTR(단방향) + NSP 제거 → 전반적으로 크게 하락, 특히 SQuAD와 MRPC → **양방향성의 효과** 검증.
  - LTR 위에 BiLSTM을 얹어도 SQuAD는 개선되지만 GLUE는 오히려 악화.
- **Ablation – 모델 크기**: 레이어·hidden·head를 키울수록 MRPC(3,600개)처럼 **작은 데이터셋에서도** 성능이 꾸준히 향상된다.
- **Feature-based 접근 (CoNLL-2003 NER)**: fine-tuning 없이 마지막 4개 층을 concat하면 F1 96.1로 fine-tuning(96.4)과 0.3 차이 → feature-based로도 효과적이다.

## 5. 결론 (배운점)

> 💡 연구의 의의 및 한계점, 본인이 생각하는 좋았던/아쉬웠던 점 (배운점)

- **의의**: 비지도 사전학습이 언어 이해의 핵심임을 보였고, 이를 **깊은 양방향 구조**로 일반화했다. 하나의 사전학습 모델로 다양한 NLP task를 해결하는 "pre-train → fine-tune" 패러다임을 정착시켰다.
- **한계**
  - [MASK] 토큰이 fine-tuning에는 없어 pre-train/fine-tune 불일치가 남는다.
  - 배치당 15% 토큰만 예측하므로 LTR보다 수렴이 느리다.
  - 인코더 전용 구조라 텍스트 생성에는 부적합하다.
  - NSP의 효과는 후속 연구(RoBERTa)에서 의문이 제기되었다.
- **좋았던 점**: 문제(단방향 제약) → 해결책(MLM) → 검증(ablation)의 흐름이 매우 명확하다. 그림 3으로 GPT, ELMo와의 차이를 직관적으로 보여준다.
- **아쉬웠던 점**: 대규모 연산 자원에 의존하고, masking 비율(15%)이나 80/10/10 비율의 선택 근거가 충분히 탐색되지 않았다.
- **배운 점**: 학습 목표(objective) 설계만 바꿔도 같은 Transformer 구조로 전혀 다른 표현을 얻을 수 있다는 것을 배웠다.

---

# [Reasoning Structure] LLMs Can Easily Learn to Reason from Demonstrations: Structure, not content, is what matters! (Introduction 리뷰)

- Dacheng Li, Shiyi Cao, Tyler Griggs, Shu Liu 외 (UC Berkeley, Anyscale), Findings of EMNLP 2025 / https://aclanthology.org/2025.findings-emnlp.866.pdf

## 1. Introduction

> 💡 논문에서 다루고 있는 주제가 무엇인지와 해당 주제의 필요성이 무엇인가
> 논문에서 제안하는 방법이 기존 방법의 문제점에 대응되도록 제안 되었는가

### 논문이 다루는 분야
- **대규모 추론 모델(Large Reasoning Model, LRM)** 의 **Long Chain-of-Thought(Long CoT)** 추론 능력 학습.
- o1, DeepSeek-R1 같은 LRM은 반성(reflection), 되돌아가기(backtracking), 자기 검증(self-validation)을 포함한 긴 CoT로 어려운 수학·코딩 문제를 푼다.
- 그러나 **Long CoT를 끌어내는 데 필요한 학습 방법과 데이터 조건은 잘 알려져 있지 않다**.

### 해당 task에서 기존 연구 한계점
- 기존 접근은 **비공개(closed-source)** 이거나(OpenAI o1, 일부 Qwen 모델), 공개되었더라도 대규모 RL 등 **재현 비용이 매우 크다**(DeepSeek-R1).
- 따라서 "얼마나 적은 데이터와 파라미터로 Long CoT를 학습할 수 있는가", "distillation 데이터의 어떤 요소가 중요한가"에 대한 이해가 부족하다.

### 논문의 contributions
1. **데이터·파라미터 효율적인 Long CoT 학습**
   - Qwen2.5-32B-Instruct를 DeepSeek-R1이 생성한 **17k개 Long CoT 샘플**로 SFT하는 것만으로 o1-preview에 견줄 만한 성능을 얻었다.
   - Math-500 90.8%(+6.0), **AIME 2024 56.7%(+40.0)**, AMC 2023 85.0%(+17.5), OlympiadBench 60.3%(+12.7), LiveCodeBench 57.0%(+8.1)
   - **LoRA로 파라미터의 5% 미만만 업데이트**해도 o1-preview 수준에 도달한다.
   - 학습된 모델은 "Alternatively", "Wait, but" 같은 표현을 쓰며 스스로 반성하고 수정하는 법을 배운다.
2. **내용(content)이 아니라 구조(structure)가 핵심**
   - **내용 교란**: 학습 샘플의 숫자 50%를 무작위로 바꿔도(오답 포함) AIME 2024 정확도는 3.3%p만 하락.
   - **구조 교란**: 67% 샘플에서 추론 단계를 섞으면(shuffle) AIME 2024 정확도가 13.3%p 하락.
   - 결론: distillation 데이터에서 중요한 것은 개별 단계의 정확성이 아니라 **Long CoT의 전체적인 논리 구조**다.
3. **광범위한 ablation**
   - 모델 크기·아키텍처, 데이터 크기, teacher 모델(DeepSeek-R1, QwQ-32B-Preview), 5개 수학·코딩 benchmark에 걸쳐 결과를 검증한다.

### 💭 Introduction을 읽고
- "추론 능력은 비싼 RL로만 얻는다"는 통념과 달리, 적은 데이터의 SFT(+LoRA)만으로 Long CoT 능력이 생긴다는 점이 인상적이었다. 결국 모델은 이미 가진 지식을 **어떻게 펼칠지(구조)** 를 배우는 것이라는 해석이 설득력 있다.
- 내용과 구조를 분리해 교란하는 실험 설계가 주장을 직접 검증하도록 짜여 있어, 문제 제기와 제안의 연결이 명확하다.
- 한편 "내용이 틀려도 괜찮다"는 결과가 어디까지 일반화되는지(더 작은 모델, 다른 도메인)는 본문에서 확인해볼 부분이다.
