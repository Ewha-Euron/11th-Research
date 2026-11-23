# Week11\_예습과제\_유다연

# Paper 1. Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (EMNLP 2024 Industry)

| 항목 | 내용 |
| --- | --- |
| 저자 | Zhuowan Li, Cheng Li, Mingyang Zhang, Qiaozhu Mei, Michael Bendersky (Google DeepMind, Univ. of Michigan) |
| 학회 | EMNLP 2024 Industry Track, pp. 881–893 |
| 링크 | [ACL Anthology](https://aclanthology.org/2024.emnlp-industry.66/) |
| 한 줄 요약 | 성능은 Long-Context(LC)가, 비용은 RAG가 우세하다. 모델이 스스로 "RAG로 답할 수 있는지" 판단해 라우팅하는 Self-Route로 LC급 성능을 훨씬 적은 토큰으로 얻는다. |

## 1. Introduction

### 논문이 다루는 분야

긴 외부 문서를 LLM에 활용하는 방법, 즉 **RAG(Retrieval Augmented Generation)와 Long-Context LLM(LC)의 비교**를 다룬다. RAG는 질문과 관련된 chunk만 검색해 모델에 넣는 방식이고, LC는 문서 전체를 컨텍스트 창에 직접 넣는 방식이다.

주제의 필요성: Gemini-1.5-Pro(1M 토큰), GPT-4o(128k)처럼 컨텍스트 창이 급격히 커지면서 "RAG가 아직 필요한가?"라는 질문이 생겼다. 반면 LC는 transformer의 연산이 입력 길이에 따라 크게 늘어나 비용이 비싸다. 따라서 두 방식의 성능과 비용을 최신 모델로 다시 따져볼 필요가 있다.

### 해당 task에서 기존 연구 한계점

- **비교 결과가 낡았다:** Xu et al.(2023)은 RAG가 LC보다 낫다고 보고했지만, 당시엔 약한 LLM과 짧은 컨텍스트를 썼다. 최신 LC 모델에서도 그 결론이 유지되는지 검증되지 않았다.
- **둘 중 하나만 택하는 구도:** 기존 연구는 RAG 개선 또는 LC 확장 중 한쪽에 집중했고, 두 방식의 장점을 쿼리 단위로 결합하는 방법은 부족했다.
- **합성 벤치마크 의존:** needle-in-a-haystack 같은 합성 데이터는 구성 방식의 artifact 때문에 결론이 왜곡될 수 있다.

### 논문의 contributions

1. 9개 공개 데이터셋, 3개 최신 LLM으로 RAG vs LC를 체계적으로 벤치마크했다. 자원이 충분하면 LC가 평균적으로 우세하지만 RAG가 훨씬 저렴함을 보였다.
2. 두 방식의 예측이 63% 쿼리에서 완전히 같다는 관찰에서 출발해, 모델의 self-reflection으로 쿼리를 라우팅하는 **Self-Route**를 제안했다. Gemini-1.5-Pro 기준 비용 65%, GPT-4o 기준 39% 절감.
3. RAG 실패 원인(multi-step, general, long/complex, implicit 쿼리), k에 따른 비용-성능 trade-off, 합성 데이터의 artifact를 분석했다.

### 리뷰 포인트: 제안 방법이 기존 문제점에 대응하는가

대체로 대응한다. "LC는 성능, RAG는 비용"이라는 trade-off를 벤치마크로 먼저 확인하고, 그 trade-off를 줄이는 방법으로 Self-Route를 내놓는 흐름이 자연스럽다. 다만 Self-Route는 Xu et al.의 결론이 뒤집힌 이유(모델·컨텍스트 길이)를 직접 다루지는 않는다. 벤치마크 기여와 방법 기여가 다소 따로 노는 구조다.

## 2. Related Work

Related Work는 세 갈래로 정리되며, 각 갈래에서 본 논문이 채우는 빈자리를 짧게 짚는 방식으로 서술된다.

| 갈래 | 기존 연구 | 본 논문의 차별점 |
| --- | --- | --- |
| Long-context LLM | Longformer, LongT5, positional interpolation 등 컨텍스트 확장. 비용 절감으로 prompt compression, distillation, LLM cascading | 길이 확장이 아니라, 이미 긴 모델을 언제 쓸지 라우팅으로 비용을 줄임 |
| RAG | 낮은 비용의 QA·LM. correction, critique, verification, adaptive retrieval 등 변형 | RAG 자체를 개선하지 않고, 최신 LC 모델과 직접 비교하고 결합 |
| Long-context 평가 | 합성(needle-in-a-haystack, RULER, Counting Stars), 실제(LongBench, ∞Bench, L-Eval), lost-in-the-middle | 합성 데이터의 artifact를 지적하고 실제 데이터셋 위주로 평가 |

### 리뷰 포인트

- **Intro에서 언급한 기존 연구 서술:** Intro의 핵심 비교 대상인 Xu et al.(2023)과 이어지며, LC의 비용 문제와 합성 벤치마크 문제도 여기서 근거를 보강한다.
- **차별성 표현:** "최신 모델로의 직접 비교"와 "실제 데이터 중심"이라는 두 축으로 차별성을 두드러지게 보여준다.
- **아쉬운 점:** Adaptive retrieval이나 LLM cascading은 Self-Route와 발상이 매우 가까운데도, 어떤 점이 다른지 구체적인 비교가 없다. Self-Route가 "LLM cascading의 한 형태"에 가깝다는 점을 더 분명히 다뤘다면 좋았을 것이다.

## 3. 제안 방법론: Self-Route

### Main Idea

대부분의 쿼리는 검색된 chunk만으로 답할 수 있고(RAG와 LC 예측이 63% 동일), 모델은 "이 정보로 답할 수 있는지"를 꽤 잘 판단한다는 가정에서 출발한다. 그래서 별도 라우터 모델 없이, **LLM 자신이 라우터 역할**을 한다.

1. **RAG-and-Route:** 쿼리와 top-k chunk(약 1.5k 토큰)를 넣고, 답할 수 없으면 "unanswerable"이라고 쓰도록 프롬프트로 지시한다. 답이 나오면 그 RAG 답을 최종 답으로 쓴다.
2. **Long-context prediction:** "unanswerable"로 판정된 쿼리만 전체 컨텍스트를 넣어 LC로 다시 푼다.

API 비용은 토큰 수에 비례하므로, 1단계에서 처리되는 비율이 높을수록 비용이 크게 줄어든다. 전체 비용은 대략 다음과 같다(r = 2단계로 넘어가는 비율).

```latex
\text{Cost}_{\text{Self-Route}} \approx \text{Cost}_{\text{RAG}} + r \cdot \text{Cost}_{\text{LC}}
```

### Contribution

- 추가 학습·별도 분류기 없이 프롬프트 하나로 동작하는 단순한 라우팅 기법
- LC 성능에 근접하면서 토큰 비용을 크게 절감(Gemini 기준 LC 토큰의 약 38%만 사용)

### 리뷰 포인트

- **Intro와 일관성:** Intro에서 말한 "63% 동일 예측 → self-reflection 라우팅"이 그대로 구현되어 있다.
- **장점의 근거:** 비용 절감은 수식적으로 자명하지만, "LLM이 answerability를 잘 보정(calibrate)한다"는 핵심 가정은 별도로 정량 검증되지 않는다. "unanswerable"의 precision/recall 분석이 있었다면 더 설득력 있었을 것이다.
- **구현 가능성:** chunk 300 단어, Contriever, top-5, cosine similarity 정렬, 프롬프트까지 공개되어 있어 재현이 쉽다.

## 4. 실험 및 결과

### Dataset

실제 데이터 기반 query-based 영어 데이터셋 9개를 사용한다.

- **LongBench (7개):** NarrativeQA, Qasper, MultiFieldQA, HotpotQA, 2WikiMultihopQA, MuSiQue, QMSum
- **∞Bench (2개):** En.QA(평균 약 150k 단어), En.MC(평균 약 143k 단어)
- **Metric:** 주관식 QA는 F1, 객관식은 accuracy, 요약은 ROUGE

### Baseline

- 모델: Gemini-1.5-Pro(1M), GPT-4o(128k), GPT-3.5-Turbo(16k)
- 비교 방식: RAG(Contriever, 300단어 chunk, top-5) vs LC(전체 컨텍스트) vs Self-Route
- 두 방식 모두 "based only on the provided passage" 프롬프트로 모델 내부 지식 사용을 억제

### 결과

Self-Route는 세 모델 모두에서 RAG를 5점 이상 앞서고, LC 대비 토큰을 39\~62%만 쓴다(Table 1, 9개 데이터셋 평균).

| 모델 | LC | RAG | Self-Route | RAG로 처리된 비율 | LC 대비 토큰 |
| --- | --- | --- | --- | --- | --- |
| Gemini-1.5-Pro | **49.70** | 37.33 | 46.41 | 76.78% | 38.39% |
| GPT-4o | 48.67 | 32.60 | **48.89** | 57.36% | 61.40% |
| GPT-3.5-Turbo | 32.07 | 30.33 | **35.32** | 74.10% | 38.85% |

- **LC vs RAG:** 자원이 충분하면 LC가 우세하다. 예외는 GPT-3.5-Turbo의 ∞Bench로, 16k 창이 입력(약 147k 단어)보다 훨씬 작아 RAG가 이긴다.
- **Self-Route:** GPT 계열은 LC와 동등 이상, Gemini는 LC보다 약 3.3점 낮지만 토큰은 약 62% 절감한다. GPT-4o는 "unanswerable"을 더 자주 말해 LC로 더 많이 넘기고, 그만큼 성능은 높고 비용 절감은 작다.
- **Top-k ablation:** k가 커질수록 두 방식 모두 성능이 오르고, k ≥ 50에서는 LC에 수렴한다. Self-Route의 토큰 비용은 k=5에서 최소(LC의 38.63%)인 비단조 형태다. k가 작으면 LC로 넘어가는 쿼리가 많아지고, 크면 RAG 단계 자체가 비싸지기 때문이다.
- **다른 retriever:** Dragon으로 바꿔도(Gemini) RAG 38.09, Self-Route 46.81로 결론이 유지된다.

### RAG 실패 원인 분석

| 유형 | 설명 | 많이 나오는 데이터셋 |
| --- | --- | --- |
| Multi-step | 앞 단계 답에 따라 다음 검색이 달라짐 | HotpotQA, 2WikiMQA, MuSiQue |
| General | 모호해서 검색 쿼리로 바꾸기 어려움 | QMSum |
| Long/complex | retriever가 인코딩하기 어려운 긴 질문 | - |
| Implicit | 전체 맥락 이해가 필요 | NarrativeQA |

또한 ∞Bench PassKey를 살짝 변형하면 RAG 점수가 80.34에서 4.58로 무너지는 반면 LC는 65\~69로 안정적이다. 합성 데이터의 구성 방식이 결론을 뒤집을 수 있다는 근거다.

### 리뷰 포인트: Intro의 장점을 검증하는 실험이 있는가

있다. "성능 유지 + 비용 절감"은 Table 1의 성능·토큰 비율로, 일반성은 retriever 교체와 k ablation으로 확인한다. 다만 본문과 표의 수치가 일부 어긋난다. Gemini의 answerable 비율은 본문 82%/81.74%, Table 1은 76.78%로 다르다. 발표 때 표 기준으로 인용하는 게 안전하다.

## 5. 결론 (배운점)

### 연구의 의의

"RAG냐 LC냐"를 양자택일이 아니라 **쿼리별 선택 문제**로 바꾼 점이 핵심이다. 대부분은 RAG로 처리하고 어려운 쿼리만 LC로 넘기는 실용적인 가이드라인을 제시했고, Industry Track 논문답게 바로 서비스에 적용할 수 있는 수준으로 단순하다.

### 한계점

- **데이터 누수:** Wikipedia 기반 데이터셋은 모델이 이미 알고 있을 수 있다. "based only on" 프롬프트는 경험적 완화책일 뿐이다.
- **모델 의존성:** 라우팅 성향이 모델마다 달라(GPT-4o는 RAG 처리 57%, Gemini는 77%) 비용-성능 균형을 사용자가 조절하기 어렵다.
- **지연 시간 미분석:** LC로 넘어가는 쿼리는 두 번 호출되는데, latency는 다루지 않고 토큰 비용만 측정했다.
- **수치 불일치:** 본문과 표의 일부 수치(answerable 비율, k=1 성능)가 다르다.

### 좋았던 점

- "63% 동일 예측"이라는 간단한 관찰에서 방법을 도출하는 흐름이 깔끔했다. 분석이 방법의 동기가 되는 좋은 예시다.
- RAG 실패 유형을 분류하고 대응책(CoT, query rewriting)까지 제시해 후속 연구 방향이 보인다.
- 합성 벤치마크의 함정을 PassKey 변형 실험으로 직접 보여준 점이 인상적이었다.

### 아쉬웠던 점

- 핵심 가정인 "LLM의 answerability 판단이 정확하다"를 직접 측정하지 않았다. 잘못 "answerable"로 판정해 틀린 RAG 답을 내놓는 비율이 궁금하다.
- 라우팅 기준이 이진(답할 수 있다/없다)이라, confidence threshold로 비용-성능을 연속적으로 조절하는 실험이 있었다면 더 실용적이었을 것이다.
- Adaptive RAG나 LLM cascading 같은 유사 기법과의 직접 비교가 없다.

### 배운점

컨텍스트 창이 커져도 RAG는 비용 측면에서 여전히 의미가 있고, 실제 시스템에서는 "최고 성능"보다 "비용 대비 성능"이 기준이 된다는 점을 배웠다. 또 벤치마크 결과를 볼 때 데이터셋 구성 방식을 함께 확인해야 한다는 점도 기억할 만하다.

# Paper 2. Visual Instruction Tuning (LLaVA, NeurIPS 2023) — Introduction 리뷰

| 항목 | 내용 |
| --- | --- |
| 저자 | Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee |
| 학회 | NeurIPS 2023 |
| 링크 | [arXiv 2304.08485](https://arxiv.org/abs/2304.08485) |

## 1. Introduction

### 논문이 다루는 분야

이미지와 텍스트 지시를 함께 이해하는 \*\*범용 멀티모달 어시스턴트(Large Multimodal Model, LMM)\*\*와, 이를 만들기 위한 **visual instruction tuning**을 다룬다.

주제의 필요성: LLM 분야에서는 ChatGPT, GPT-4가 "언어를 범용 인터페이스"로 삼아 다양한 작업을 지시만으로 수행한다. Alpaca, Vicuna처럼 기계 생성 데이터로 instruction tuning하는 오픈소스 연구도 활발하다. 반면 비전 쪽에서는 이런 범용 어시스턴트가 아직 없었다.

### 해당 task에서 기존 연구 한계점

- **비전 모델의 고정된 인터페이스:** classification, detection, segmentation, captioning 등 각 모델이 하나의 task만 푼다. 언어는 이미지 내용을 설명하는 데만 쓰여, 사용자 지시에 맞추는 상호작용이 부족하다.
- **instruction tuning은 텍스트 전용:** LLM의 instruction tuning 연구는 시각 입력을 다루지 않는다.
- **기존 멀티모달 LLM:** Flamingo, BLIP-2, OpenFlamingo, LLaMA-Adapter는 vision-language instruction 데이터로 명시적으로 튜닝되지 않아, 멀티모달 task 성능이 언어 task보다 떨어진다.
- **데이터 부족(암묵적 문제):** 사람이 직접 만든 멀티모달 instruction 데이터는 구축 비용이 크고 양이 적다.

### 논문의 contributions

1. **멀티모달 instruction-following 데이터:** 텍스트 전용 ChatGPT/GPT-4에 이미지의 caption과 bounding box를 텍스트로 주고, image-text pair를 instruction 데이터(대화, 상세 설명, 복잡한 추론)로 변환하는 파이프라인을 제안했다.
2. **LMM(LLaVA):** CLIP 비전 인코더와 Vicuna 언어 디코더를 연결해 생성 데이터로 end-to-end fine-tuning했다. GPT-4와 결합해 ScienceQA에서 92.53%로 SOTA를 달성했다.
3. **벤치마크(LLaVA-Bench):** 다양한 이미지·지시·상세 어노테이션으로 구성된 멀티모달 instruction-following 평가셋을 만들었다.
4. **오픈소스:** 생성 데이터, 코드, 체크포인트, 데모를 공개했다.

### 리뷰 포인트: 제안 방법이 기존 문제점에 대응하는가

잘 대응한다. "instruction tuning이 텍스트에만 있다"는 한계에는 visual instruction tuning이라는 개념 자체로, "데이터가 없다"는 문제에는 GPT-4 기반 데이터 생성으로, "평가 기준이 없다"는 문제에는 LLaVA-Bench로 각각 답한다. 한계와 contribution이 1:1로 짝지어져 있는 구조가 인상적이다.

다만 GPT-4가 이미지를 직접 보지 않고 caption·box만 보고 데이터를 만든다는 점에서, 생성 데이터의 품질이나 hallucination 가능성은 Intro에서 언급되지 않는다. 또 "Flamingo, BLIP-2가 멀티모달 task에서 약하다"는 주장은 Intro에서는 정량 근거 없이 제시되어, 실험 파트에서 확인해볼 필요가 있다.
