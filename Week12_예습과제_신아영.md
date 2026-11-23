# Week12예습과제_신아영

---

# [RAG] Retrieval Augmented Generation or Long-Context LLMs? A Comprehensive Study and Hybrid Approach (전체 리뷰)

- Li et al., EMNLP 2024 Industry Track / https://aclanthology.org/2024.emnlp-industry.66/

## 1. Introduction

> 논문에서 다루는 주제와 필요성, 기존 방법의 한계, 논문의 기여

### 논문이 다루는 분야
- **Retrieval-Augmented Generation(RAG)**: 질문과 관련된 외부 정보를 검색한 뒤, 검색 결과를 LLM의 입력에 포함하여 답변을 생성하는 방식이다. 전체 문서를 입력하지 않아도 필요한 정보를 활용할 수 있다는 장점이 있다.
- **Long-Context(LC) LLM**: 긴 문서를 모델에 직접 입력해 전체 문맥을 바탕으로 답변을 생성하는 방식이다.
- 최근(논문 발표 시점 기준) Gemini-1.5-Pro, GPT-4o 등의 context window가 크게 확장되면서, 긴 문서를 굳이 검색을 통해 일부만 전달해야 하는지에 대한 의문이 생겼다. 이에 RAG와 LC의 성능뿐 아니라 비용까지 비교할 필요가 있었다.

### 해당 task에서 기존 연구의 한계점
- RAG는 입력 길이를 줄이고 관련 정보에 집중할 수 있지만, 검색 단계에서 필요한 정보가 누락되면 LLM에 충분한 근거가 전달되지 않는다.
- LC는 전체 문맥을 활용할 수 있지만 긴 입력을 처리하기 위한 연산량과 비용이 크다. 또한 문서 길이가 모델의 context window보다 길면 전체 내용을 한 번에 입력할 수 없다.
- Xu et al.(2023)은 LC의 성능이 RAG에 미치지 못한다는 결과를 제시했지만, 더 긴 context window를 지원하는 모델들에서도 동일한 결과가 나타나는지는 추가로 확인할 필요가 있었다.

### 논문의 contributions
- 3종의 LLM과 9개 데이터셋을 이용해 RAG와 LC의 성능을 비교했다. 충분한 context window와 자원이 주어진 조건에서는 LC가 세 모델 모두에서 평균적으로 RAG보다 높은 성능을 보였다.
- RAG와 LC가 약 63%의 질문에서 동일한 답변을 생성한다는 점에 착안해, 두 방식의 장점을 결합한 **Self-Route**를 제안했다.
- Self-Route는 모델이 검색 결과만으로 답변할 수 있는지를 판단하게 하고, 필요할 때만 LC를 사용한다. 이를 통해 LC와 비슷한 성능을 유지하면서 입력 토큰 사용량을 줄였다.
- 검색 chunk 수, RAG의 실패 유형, retriever 종류, 합성 데이터의 질문 구성 및 모델의 내부 지식이 비교 결과에 미치는 영향도 분석했다.

## 2. Related Work

> 기존 연구와 제안 방법의 차이

- **Long-context LLMs**: Longformer 등 긴 시퀀스를 처리하는 구조와 context window를 확장하는 연구들이 진행되었다. 하지만 긴 입력의 처리 비용은 여전히 문제이며, prompt compression, distillation, LLM cascading 등 이를 줄이려는 연구도 있었다. 이 논문은 입력 자체를 압축하기보다 질문에 따라 RAG와 LC를 선택한다.
- **Retrieval-Augmented Generation**: Lewis et al.(2020) 등을 비롯한 RAG 연구는 모델의 내부 지식을 외부 검색 결과로 보완했다. 이후 Self-RAG, Corrective RAG, Adaptive RAG 등 검색 품질이나 검색 전략을 개선하는 연구도 등장했다. 이 논문은 검색의 품질 개선 자체보다 **언제 RAG를 사용하고 언제 전체 문맥을 입력할 것인지**에 초점을 맞춘다.
- **Long-context evaluation**: LongBench, ∞Bench 등의 벤치마크와 needle-in-a-haystack 같은 합성 테스트를 통해 긴 문맥 이해 능력을 평가해 왔다. 저자들은 실제 문서를 기반으로 한 다양한 질문을 중심으로 RAG와 LC를 비교하고, 합성 테스트의 구성에 따른 결과 차이도 살펴봤다.

## 3. 제안 방법론

> Main Idea, 구체적인 방법, 장점의 근거

### Main Idea
- 모든 질문에 전체 문서를 입력할 필요는 없다. RAG에서 검색한 정보만으로도 충분히 답할 수 있는 질문에는 RAG를 사용하고, 그렇지 않은 질문에만 LC를 사용하자는 것이 Self-Route의 핵심이다.
- Figure 2에서는 약 63%의 질문에서 두 방식의 예측이 동일했고, 70%의 질문에서 평가 점수 차이의 절댓값이 10 미만이었다(100점 척도).
- 다만 동일한 답변이 모두 정답이라는 뜻은 아니다. RAG와 LC가 동일한 오답을 생성하는 경우도 포함된다.

### 구체적 방법

**(1) RAG: 검색 및 답변 생성**
- 긴 문서를 300 **단어(words)** 단위의 chunk로 나눈다.
- 질문과 각 chunk를 각각 embedding으로 변환하고, cosine similarity를 계산한다.
- 유사도가 높은 상위 k개 chunk를 선택한다(기본값 k=5). 선택한 chunk를 유사도 순서로 정렬하고 chunk index를 붙인 뒤 질문과 함께 LLM에 입력한다.
- 검색에는 dense retriever인 Contriever 또는 Dragon을 사용한다.

**(2) LC: 전체 문맥 활용**
- 검색으로 일부 chunk를 선택하는 대신 질문과 긴 문서 전체를 모델에 입력하는 방식이다. 모델의 context window에 의해 실제로 처리할 수 있는 길이가 제한될 수 있다.

**(3) Self-Route: 두 방법을 결합하는 방식 (§4.2)**

```text
질문 + 긴 문서
        ↓
Retriever: Top-k chunk 검색
        ↓
LLM: 제공된 chunk만으로 답변 가능한가?
        ├─ 가능 → RAG 답변을 최종 답변으로 사용
        └─ 불가능(unanswerable) → 전체 문맥을 입력하여 LC 답변 생성
```

- **RAG-and-Route:** LLM에 질문과 검색 결과를 제공하고, 해당 내용만으로 답할 수 없으면 `unanswerable`을 출력하도록 지시한다. 답할 수 있다고 판단하면 이 단계에서 바로 답변을 생성한다.
- **Long-context prediction:** 첫 단계에서 `unanswerable`을 출력한 경우 전체 문맥을 제공해 답변을 다시 생성한다.
- 별도의 routing 모델을 학습하는 것이 아니라 기존 LLM의 **self-reflection(답변 가능성에 대한 자기 판단)**을 이용한다.
- 모든 질문이 첫 번째 RAG 단계를 거치기 때문에 LC로 전환되는 질문에는 추가적인 호출 비용이 발생하지만, 대부분의 질문을 RAG에서 처리한다면 전체 입력 토큰을 줄일 수 있다.

### Contribution
- 새로운 LLM 구조나 학습 방법을 제안하기보다는 기존 RAG와 LC를 질문별로 선택하는 inference 전략을 제안했다.
- Self-Route가 잘 작동하려면 LLM이 검색된 근거만으로 답할 수 있는지를 정확하게 판단해야 한다. 즉, 모델의 answerability 판단이 잘 보정(calibrated)되어 있다는 가정에 의존한다.

## 4. 실험 및 결과

> Introduction의 주장을 어떤 실험으로 검증했는가

### Dataset (§3.1, Appendix A)
- LongBench의 7개 데이터셋과 ∞Bench의 2개 데이터셋을 사용했다. 영어, 실제 문맥, 질문 기반 과제를 중심으로 선정했다.

| Benchmark | Dataset | Task |
|---|---|---|
| LongBench | NarrativeQA | 소설·대본에 대한 질의응답 |
| LongBench | Qasper | NLP 논문에 대한 질의응답 |
| LongBench | MultiFieldQA | 여러 분야의 문서 기반 질의응답 |
| LongBench | HotpotQA, 2WikiMultihopQA, MuSiQue | 여러 근거를 연결하는 multi-hop QA |
| LongBench | QMSum | 질문 기반 회의록 요약 |
| ∞Bench | En.QA | 긴 소설에 대한 주관식 QA |
| ∞Bench | En.MC | 긴 소설에 대한 객관식 QA |

- Appendix Table 6 기준 En.QA와 En.MC의 평균 문서 길이는 각각 150,374단어와 142,622단어로 LongBench보다 훨씬 길다.
- 평가지표: 주관식 QA는 F1, 객관식 QA는 Accuracy, 요약 과제는 ROUGE를 사용했다. 따라서 Table 1의 Avg는 서로 다른 과제별 평가 점수를 평균한 값이지 단일 정확도가 아니다.

### Models 및 Baseline

| LLM | Context window (논문 실험 시점) |
|---|---:|
| Gemini-1.5-Pro | 1M tokens |
| GPT-4o | 128K tokens |
| GPT-3.5-Turbo | 16K tokens |

- 비교 방법: RAG, LC, Self-Route
- 기본 retriever: Contriever / 추가 비교: Dragon
- Chunk size: 300 words, Top-k 기본값: 5
- 모델이 사전학습 중 평가 데이터를 접했을 가능성을 줄이기 위해 제공된 문맥만을 근거로 답하도록 지시했다.

### 결과 ① RAG vs. LC vs. Self-Route — Table 1

| Model | LC 평균 점수 | RAG 평균 점수 | Self-Route 평균 점수 | Answerable % | Token % |
|---|---:|---:|---:|---:|---:|
| Gemini-1.5-Pro | 49.70 | 37.33 | 46.41 | 76.78 | 38.39 |
| GPT-4o | 48.67 | 32.60 | 48.89 | 57.36 | 61.40 |
| GPT-3.5-Turbo | 32.07 | 30.33 | 35.32 | 74.10 | 38.85 |

- LC가 세 모델 모두에서 RAG보다 높은 평균 점수를 기록했다. 특히 GPT-4o와 Gemini-1.5-Pro에서 차이가 컸다.
- 예외적으로 GPT-3.5-Turbo에서는 ∞Bench의 En.QA(LC 14.73 / RAG 15.39)와 En.MC(LC 34.50 / RAG 43.67)에서 RAG가 더 높은 점수를 기록했다. 문서 길이가 16K context window를 크게 초과하기 때문으로 저자들은 해석한다.
- Self-Route를 적용하면 세 모델 모두 RAG보다 평균 점수가 높아졌다. GPT-4o는 LC와 거의 같은 수준이고, GPT-3.5-Turbo는 LC보다 높았으며, Gemini-1.5-Pro는 LC보다 낮았지만 차이를 줄였다.
- **Answerable %**는 RAG 단계에서 모델 스스로 답할 수 있다고 판단한 질문의 비율이다. 실제 정답률과는 다르다.
- **Token %**는 LC를 사용할 때의 입력 토큰 수를 100%로 두고 비교한 비율이다. Table 1 기준 Gemini는 38.39%, GPT-4o는 61.40%, GPT-3.5는 38.85%만 사용했다. 이는 각각 약 62%, 39%, 61%의 입력 토큰 절감에 해당한다. 검색·출력·호출 지연 등을 모두 합산한 실제 서비스 비용 절감률과는 구별해야 한다.

### 결과 ② Top-k에 따른 성능과 비용 — Figure 3 / Appendix Table 7

| k | RAG 점수 | Self-Route 점수 | RAG Token % | Self-Route Token % |
|---:|---:|---:|---:|---:|
| 1 | 20.24 | 41.35 | 5.26 | 39.64 |
| 5 | 37.92 | 43.33 | 17.02 | 38.63 |
| 10 | 41.20 | 44.38 | 42.42 | 53.66 |
| 50 | 44.06 | 45.19 | 95.29 | 102.97 |
| 100 | 44.12 | 45.23 | 100.32 | 106.59 |

- k가 증가하면 검색 결과에 더 많은 정보가 포함되기 때문에 RAG와 Self-Route의 성능이 대체로 향상된다.
- Self-Route는 k=1에서 k=5로 늘렸을 때 오히려 token ratio가 39.64%에서 38.63%로 감소했다. 검색 결과가 많아지면서 LC로 전환해야 하는 질문이 줄어들 수 있기 때문이다.
- 비교한 k 값 중 k=5에서 Self-Route의 토큰 사용 비율이 가장 낮았다. 반면 k=50이나 100에서는 LC만 쓰는 것보다 토큰을 더 많이 사용했다.
- 따라서 무조건 많은 chunk를 검색하는 것이 좋은 것은 아니다. 과제의 성격에 따라 최적 k가 달라질 수 있다.

### 결과 ③ RAG가 실패하는 이유 — §5.2 / Figure 4

연구진은 Self-Route의 첫 단계에서 `unanswerable`로 판단된 사례를 일부 직접 검토한 후, LLM을 이용해 실패 유형을 분류했다.

| 실패 유형 | 설명 |
|---|---|
| Multi-step | 앞 단계에서 얻은 정보를 바탕으로 다음 정보를 다시 찾아야 하는 질문 |
| General | 질문이 포괄적이라 특정 관련 chunk를 찾기 어려운 경우 |
| Complex | 질문이 길고 복잡하여 retriever가 의미를 제대로 파악하기 어려운 경우 |
| Implicit | 문서 전체의 흐름이나 여러 문맥을 종합해야 답할 수 있는 질문 |

- HotpotQA, 2WikiMultihopQA, MuSiQue에서는 multi-step 유형이 많았고, NarrativeQA에서는 implicit 유형, QMSum에서는 general 유형이 많이 나타났다.
- 예를 들어 특정 노래를 부른 사람의 국적을 묻는 질문은 먼저 가수를 알아낸 뒤 그 사람의 국적을 찾아야 한다. 처음 질문에 대한 단일 검색으로 두 정보를 모두 찾지 못할 수 있다.
- 저자들은 multi-step 질문에는 CoT를 결합한 검색 방식이, general이나 complex 질문에는 query expansion 등의 방법이 도움이 될 수 있다고 제안한다. 이는 후속 개선 방향이지 이 논문에서 직접 검증한 성능 개선은 아니다.

### 결과 ④ Retriever 변경 — Table 2
- Gemini-1.5-Pro에서 Contriever 대신 Dragon을 사용한 결과, LC 49.70, RAG 38.09, Self-Route 46.81의 평균 점수를 기록했다.
- Self-Route의 token ratio는 37.87%로, retriever를 바꿔도 주요 경향이 유지됐다. 다만 다른 모든 검색 방식에서도 동일하게 나타난다고 일반화할 수는 없다.

### 결과 ⑤ 합성 데이터 평가 — Table 3

| PassKey 질문 유형 | RAG (%) | LC (%) |
|---|---:|---:|
| 기본 질문 (passkey 찾기) | 80.34 | 65.25 |
| Variant 1 (special token으로 바꿔 질문) | 4.58 | 69.32 |
| Variant 2 (두 passkey 중 더 큰 값 찾기) | 47.63 | 64.24 |

- 기본 PassKey 질문에서는 RAG가 LC보다 우수했지만, 같은 정보를 다른 표현으로 질문하자 RAG의 정확도가 크게 떨어졌다.
- 질의 표현과 문서 속 키워드가 잘 맞는지에 따라 검색 난도가 달라질 수 있다는 점을 보여준다. 합성 데이터셋의 구성 방식이 두 방법의 성능 비교에 영향을 미칠 수 있음을 지적한 실험이다.

### 결과 ⑥ 모델의 내부 지식 통제 — §5.5 / Table 4, 5
- Table 4에서 제공된 문맥만을 근거로 답하라는 지시를 추가한 뒤 Gemini-1.5-Pro의 LC 평균 점수(표에 기재된 Avg)가 50.57에서 45.53으로 감소했다. 저자들은 이 지시가 모델 내부 지식의 활용을 제한하는 데 도움이 된다고 해석한다.
- Table 5에서는 MuSiQue에서 문맥 없이도 모델이 정답을 맞힐 수 있는 질문을 제외하고 평가했다. Gemini-1.5-Pro의 LC 31.76, RAG 15.51, Self-Route 31.32로 주요 경향이 유지됐다. GPT-3.5-Turbo에서도 Self-Route가 가장 높은 점수를 기록했다.
- 다만 프롬프트만으로 내부 지식의 사용이나 데이터 누출을 완전히 막았다고 볼 수는 없다.

> 수치 참고: 원문 본문에는 Table 1의 Avg, Answerable %, Token % 및 Appendix Table 7의 일부 수치와 일치하지 않는 서술이 있다. 여기서는 비교표에 직접 기재된 값을 기준으로 정리했다. 또한 Table 4의 Avg는 해당 표의 보고값이며 Table 1의 Avg와 동일하게 취급하지 않았다.

## 5. 결론 (배운 점)

### 논문의 의의
- LC가 평균 성능 측면에서 우수하더라도 RAG가 여전히 유용한 이유를 비용과 context window의 제약을 통해 설명했다.
- Self-Route는 상대적으로 간단한 방법으로 두 접근법의 차이를 활용한다. 질문마다 필요한 문맥의 양이 다르다는 점에 주목한 것이 핵심이다.

### 좋았던 점
- 특히 Figure 2를 통해 평균 점수만 비교했을 때는 알기 어려웠던 사실을 확인한 점이 인상적이었다. LC가 평균적으로 더 좋더라도 RAG와 답변이 같은 질문이 많다는 점에서 모든 질문에 LC를 적용하는 것은 효율적이지 않을 수 있다고 생각했다.
- k가 커지면 비용도 무조건 증가할 것이라고 생각했는데, Self-Route에서는 적절히 k를 늘리는 것이 오히려 LC 호출을 줄여 전체 토큰 사용량을 낮출 수 있다는 결과가 흥미로웠다.

### 아쉬웠던 점
- Self-Route는 LLM이 스스로 답변 가능 여부를 잘 판단한다는 가정에 의존한다. 검색 결과가 불충분한데도 모델이 답할 수 있다고 판단하면 LC로 전환할 기회가 없다. Answerable %가 높다는 사실만으로 routing이 잘 이루어졌다고 보기 어려우므로 판단 자체의 정확도도 평가했다면 좋았을 것 같다.
- 비용 비교가 주로 입력 토큰 수를 기준으로 이루어졌다. 실제 서비스에서는 문서를 embedding하는 비용, 검색 시간, 추가 LLM 호출로 인한 응답 지연 등도 고려해야 한다고 생각한다.
- 기본적인 single-pass top-k 검색과 LC를 비교했으므로, iterative retrieval이나 query rewriting 등 개선된 RAG 방식과 비교하면 결과가 어떻게 달라질지도 궁금하다.

### 배운 점
- 이전 CoT 논문에서는 복잡한 질문을 중간 추론 단계로 나누는 것이 중요하다는 점을 배웠는데, 이번에는 이런 아이디어가 RAG의 검색 단계에도 연결될 수 있다는 것을 알게 됐다. 여러 정보를 순차적으로 찾아야 하는 질문에서는 답변을 생성하기 전에 검색 자체를 여러 단계로 나눌 필요도 있을 것 같다.
- RAG와 LC 중 하나를 모든 질문에 사용하는 것보다, 질문 유형과 필요한 문맥의 양, 비용을 함께 고려해 선택하는 것이 중요하다고 느꼈다.

## 6. 토의 질문

1. **Self-Route가 검색 결과만으로 답할 수 있다고 잘못 판단하는 경우를 어떻게 줄일 수 있을까?**
   - RAG 단계에서 답변에 필요한 근거를 명시하도록 하고, 별도의 검증 단계를 추가하는 방법을 생각해 볼 수 있다. 다만 검증에 드는 비용까지 비교해야 한다.
2. **모든 질문에 동일한 k=5를 사용할 필요가 있을까?**
   - 단순한 사실 질문은 작은 k를, 여러 근거가 필요한 질문은 큰 k 또는 반복 검색을 사용하는 adaptive 방식이 더 효율적일 수 있다. 질문별 검색 비용과 LC 전환 비율을 함께 비교해 보고 싶다.
3. **LC보다 토큰 사용량이 적다고 해서 실제 서비스에서도 항상 더 효율적일까?**
   - RAG의 검색 비용과 LC로 전환할 때 추가 호출이 발생하는 점을 고려해야 한다. 실제 비용과 latency까지 포함하여 비교하면 다른 결과가 나올 가능성도 있다.

위 토의 답변은 논문에서 검증한 결과가 아니라 실험을 읽고 제안한 추가 연구 아이디어이다.

---

# [LLaVA] Visual Instruction Tuning (Introduction 리뷰)

- Liu et al., NeurIPS 2023 / https://arxiv.org/abs/2304.08485

## 1. Introduction

> 논문에서 다루는 주제와 필요성, 기존 연구의 한계 및 contributions

### 논문이 다루는 분야
- **Vision-Language Multimodal Learning**: 이미지와 자연어 정보를 함께 활용하여 사용자의 질문이나 지시를 처리하는 분야이다.
- **Instruction Tuning**: 모델이 자연어로 주어진 지시를 이해하고 그에 맞는 응답을 생성하도록 지시-응답 데이터로 미세조정하는 방법이다.
- 기존 LLM에서는 Alpaca, Vicuna, GPT-4-LLM 등의 연구가 모델이 생성한 instruction-following 데이터를 활용해 지시 수행 능력을 높였다. 이 논문은 이러한 접근을 이미지와 언어를 함께 사용하는 분야로 확장하고자 한다.

### 기존 연구의 한계
- 기존 vision foundation model은 이미지 분류, 객체 탐지, segmentation, captioning 등의 과제에서 좋은 성능을 보였지만, 일반적으로 각각의 과제에 맞춰 모델이 설계되었다.
- 자연어를 주로 이미지 내용을 표현하는 데 사용했기 때문에, 사용자가 자연어로 다양한 지시를 내리고 그에 따라 과제를 바꾸는 데에는 한계가 있었다.
- 반면 ChatGPT와 GPT-4 같은 LLM은 자연어를 여러 과제를 수행하기 위한 공통 인터페이스로 활용할 수 있다는 가능성을 보여주었다. 그러나 Alpaca, Vicuna 등의 instruction tuning 연구는 텍스트 중심이었다.
- Image-text pair는 많이 존재하지만, 이미지에 관한 다양한 지시와 그에 적합한 응답으로 구성된 multimodal instruction-following 데이터는 부족했다.

### 논문의 contributions
- **Multimodal instruction-following data:** ChatGPT/GPT-4를 활용해 기존 image-text pair를 이미지와 관련된 지시 및 응답 형식의 데이터로 변환하는 방법을 제안했다. Abstract에서는 이미지 입력을 직접 처리하지 않는 language-only GPT-4를 활용했다는 점을 강조한다.
- **Large Multimodal Model (LMM):** CLIP의 visual encoder와 Vicuna의 language decoder를 연결한 LLaVA(Large Language and Vision Assistant)를 제안했다. 생성한 vision-language instruction 데이터로 모델을 end-to-end fine-tuning하여 사용자의 시각적 지시를 수행하도록 했다.
- **Multimodal instruction-following benchmark:** 다양한 이미지, 지시문, 상세한 annotation으로 구성된 두 가지 LLaVA-Bench 평가 벤치마크를 제안했다.
- **Open-source:** 생성한 instruction 데이터, 코드, 모델 checkpoint 및 visual chat demo를 공개했다.
- Abstract에서 보고한 주요 결과로, 합성 multimodal instruction-following 평가에서 GPT-4 대비 **85.1%의 상대 점수**를 얻었으며, ScienceQA에서는 LLaVA와 GPT-4를 결합해 당시 **92.53%의 정확도**를 달성했다. 85.1%는 일반적인 정답률이 아니며, 92.53%도 LLaVA 단독의 결과는 아니다.

### 💭 Introduction을 읽고
- 기존 이미지 모델은 사진에 무엇이 있는지를 알아내는 데 집중했다면, LLaVA는 같은 사진이라도 사용자가 무엇을 묻느냐에 따라 다른 답변을 생성하도록 학습한다는 점이 흥미로웠다. 이미지를 인식하는 능력과 사용자의 의도에 맞게 시각 정보를 활용하는 능력은 구분해서 생각할 필요가 있다는 것을 알게 됐다.
- 특히 이미지를 직접 입력받지 않는 GPT-4를 이용해 시각적 지시 학습에 필요한 데이터를 만들었다는 점이 인상적이었다. 이전 Long CoT 논문에서도 Teacher 모델이 생성한 데이터를 이용해 Student 모델을 학습시키는 방법을 살펴봤는데, 이번에는 이러한 데이터 생성 방식이 멀티모달 분야에도 활용될 수 있다는 점이 흥미로웠다.
- 다만 이미지에 대한 설명이 실제 이미지의 모든 정보를 담을 수는 없을 것 같다. 생성된 지시-응답 데이터에 이미지에서는 확인할 수 없는 정보가 들어간다면 모델의 학습에 어떤 영향을 미치는지, 데이터의 정확성을 어떻게 확보하는지 궁금했다.

