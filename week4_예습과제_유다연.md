# Week4 예습과제 - 비전 트랜스포머(ViT)

- 작성자: 유다연
- 교재: 『파이토치 트랜스포머를 활용한 자연어 처리와 컴퓨터비전 심층학습』 10장 비전 트랜스포머
- 구성
  - (1) 개념정리: ViT (p.600~608)
  - (2) ViT 모델 실습 코드 필사 (p.609~624)

---

## (1) 개념정리 - ViT (p.600~608)

### 1. ViT 개요

- **ViT(Vision Transformer)**: 2020년 Google 논문 「An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale」에서 소개된 이미지 인식 모델
- 핵심 아이디어: 이미지를 **자연어 문장처럼** 처리 → 이미지를 작은 패치(=단어 토큰)로 나눈 뒤 트랜스포머에 입력
- CNN의 합성곱 계층 없이 **셀프 어텐션**만으로 이미지 전체를 한 번에 처리
- 트랜스포머 구조 자체를 컴퓨터비전에 그대로 적용한 첫 연구 (이전 연구는 CNN에 셀프 어텐션 모듈을 일부 결합하는 방식)
- 한계: 패치를 왼쪽→오른쪽, 위→아래 순서의 1차원 시퀀스로 입력 → 2차원 이미지 구조를 온전히 반영하지 못함
  - 이를 보완한 후속 모델: **Swin Transformer**(로컬 윈도 기반 계층적 학습), **CvT**(합성곱 연산을 결합해 저수준·고수준 특징을 계층적으로 반영)

### 2. BERT vs ViT: 입력 데이터 구성 차이

| 구분 | BERT | ViT |
|---|---|---|
| 모델 구조 | 트랜스포머 | 트랜스포머 (동일) |
| 입력 단위 | 문장을 쪼갠 **토큰** | 이미지를 쪼갠 **패치** |
| 입력 순서 | 문장 내 순서 | 왼쪽→오른쪽, 위→아래 순서의 시퀀스 |
| 맨 앞 토큰 | [CLS] | [CLS] |

→ 모델 구조는 같고, **입력 데이터를 만드는 과정만 다름**

### 3. CNN vs ViT 비교

- 공통 목적: 이미지 특징을 잘 표현하는 임베딩 생성
- **CNN**
  - 커널이 이미지 일부 영역만 보며 지역 특징 추출 → 층을 쌓아 전체 특징 표현
  - 예: 고양이 오른쪽 눈 표현 시 주변 패치(1, 2, 4, 5)만 학습에 관여
  - 수용 영역(Receptive Field)이 좁아 전체 정보를 표현하려면 많은 계층 필요
- **ViT**
  - 셀프 어텐션으로 **모든 패치 간 상관관계** 학습 → 모든 패치가 학습에 관여
  - **어텐션 거리(Attention Distance)**(쿼리·키 벡터의 내적 기반 유사도)로 한 개 레이어만으로도 전체 이미지 정보 표현 가능
  - 픽셀 단위가 아닌 패치 단위 처리 → 작은 모델로도 높은 성능
- ViT의 단점
  - 입력 이미지 크기 고정 → 크기가 다른 이미지는 전처리(크기 조정) 필요
  - 패치 간 **상대적 위치 정보**만 고려 → 이미지 변환(회전·이동 등)에 취약

### 4. 귀납적 편향(Inductive Bias)

- 정의: 일반화 성능 향상을 위해 모델이 갖는 **가정(Assumption)**
- 가정이 데이터에 맞으면 적은 데이터로도 높은 일반화 성능 / 너무 강하면 다른 유형의 관계 표현이 어려움

| 모델 | 귀납적 편향 | 이유 |
|---|---|---|
| CNN | 지역적(Local) 편향 | 이미지의 공간적 관계 표현에 특화 |
| RNN | 순차적(Sequential) 편향 | 시계열의 시간적 관계 표현에 특화 |
| ViT | 거의 없음 | Q·K·V 임베딩으로 일반화된 관계를 학습 |

- ViT는 귀납적 편향이 약한 대신 **대용량 데이터**에서 패치 간 관계를 잘 학습하는 모델

### 5. ViT 모델 구조

- 구성: **패치 임베딩(Patch Embedding)** + **인코더(Encoder) 계층**
- 흐름: 이미지 → 패치 분할 → 패치 임베딩 + [CLS] + 위치 임베딩 → 트랜스포머 인코더 × N → [CLS] 벡터 → 분류기(FFN) → 클래스 예측

#### 5-1. 패치 임베딩

- 입력 이미지를 작은 패치로 분할해 벡터로 만드는 과정
- 전처리: 입력 크기를 정사각형으로 통일 (예: 640×480 → 224×224)
- 패치 분할에 **합성곱 계층(Conv2d)** 활용 → 하이퍼파라미터: 커널 크기(패치 크기), 간격(stride, 패치 이동 폭)
- 패치 개수 계산 (가로·세로 각각)

$$\text{patch size} = \frac{\text{image size} - \text{kernel size}}{\text{stride}} + 1$$

- 예시 (이미지 9×9, 커널 3×3)
  - stride 3: (9−3)/3 + 1 = 3 → 3×3 = **9개** 패치 (겹침 없음)
  - stride 1: (9−3+1)×(9−3+1) = **49개** 패치 (겹침 있음)
- 분할된 패치를 왼쪽→오른쪽, 위→아래 순서로 나열
- **[CLS] 토큰(Special Classification Token)**: 시퀀스 맨 앞에 추가, 이미지 전체를 대표하는 벡터로 최종 예측에 사용
- **위치 임베딩(Position Embedding)**: 패치의 위치 정보를 벡터로 변환해 패치 벡터에 더함 → 인접 패치 간 관계 학습
- 마지막으로 **계층 정규화(Layer Normalization)** 적용
- 장점: 이미지를 패치 단위로 처리 → GPU 메모리 한계를 줄이고 더 큰 이미지 처리 가능

#### 5-2. 인코더 계층

- 패치 벡터들을 인코더에 입력 → Q·K·V 임베딩으로 패치 간 관계 학습 (7장 **멀티 헤드 어텐션**과 동일)
- 인코더 블록 구성: 셀프 어텐션 → 벡터 더하기 & 정규화 → 순방향 신경망 → 벡터 더하기 & 정규화, 이를 **N번 반복**
- 마지막 레이어의 **[CLS] 토큰 벡터**를 이미지 전체 특징 벡터로 사용
  - 모든 패치 출력(시퀀셜 산출물)이 아닌 **분류 토큰 벡터 하나만** 사용해 분류
  - [CLS] 벡터 → 순방향 신경망(완전 연결 계층) → 클래스 분류 (예: 고양이 O / X)

### 6. 한 줄 요약

- ViT = 이미지를 16×16 패치 "단어"로 바꿔 트랜스포머 인코더에 넣고, [CLS] 토큰으로 분류하는 모델
- CNN 대비 귀납적 편향이 약해 **데이터가 많을수록 유리**, 전역적 관계 학습에 강점

---

## (2) ViT 모델 실습 코드 필사 (p.609~624)

- 실습 내용: Hugging Face `transformers` 라이브러리의 사전 학습 ViT(`google/vit-base-patch16-224-in21k`)를 **FashionMNIST**로 미세 조정
- FashionMNIST: 의류 이미지 10개 클래스, 훈련 60,000개 / 테스트 10,000개 → 실습에서는 훈련 10,000개 / 테스트 1,000개로 샘플링
- 출력 결과는 교재 기준으로 기재

### 예제 10.1 FashionMNIST 다운로드

```python
from itertools import chain
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets

# 클래스별로 max_len개씩 균등 샘플링하는 함수
def subset_sampler(dataset, classes, max_len):
    target_idx = defaultdict(list)                 # {클래스 번호: [인덱스 목록]}
    for idx, label in enumerate(dataset.train_labels):
        target_idx[int(label)].append(idx)

    indices = list(
        chain.from_iterable(                       # 클래스별 인덱스 리스트를 1차원으로 펼침
            [target_idx[idx][:max_len] for idx in range(len(classes))]
        )
    )
    return Subset(dataset, indices)                # 해당 인덱스의 데이터만 추출


train_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=True)
test_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=False)

classes = train_dataset.classes                    # 클래스 이름 목록
class_to_idx = train_dataset.class_to_idx          # {클래스 이름: 클래스 ID}

print(classes)
print(class_to_idx)

subset_train_dataset = subset_sampler(
    dataset=train_dataset, classes=train_dataset.classes, max_len=1000   # 10개 클래스 × 1,000 = 10,000개
)
subset_test_dataset = subset_sampler(
    dataset=test_dataset, classes=test_dataset.classes, max_len=100      # 10개 클래스 × 100 = 1,000개
)

print(f"Training Data Size : {len(subset_train_dataset)}")
print(f"Testing Data Size : {len(subset_test_dataset)}")
print(train_dataset[0])
```

```
['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
{'T-shirt/top': 0, 'Trouser': 1, 'Pullover': 2, 'Dress': 3, 'Coat': 4, 'Sandal': 5, 'Shirt': 6, 'Sneaker': 7, 'Bag': 8, 'Ankle boot': 9}
Training Data Size : 10000
Testing Data Size : 1000
(<PIL.Image.Image image mode=L size=28x28 at 0x1661A976310>, 9)
```

- 개별 데이터 = (PIL 이미지, 클래스 번호) → 학습을 위해 Tensor로 변환하는 전처리 필요

### 예제 10.2 이미지 전처리

```python
import torch
from torchvision import transforms
from transformers import AutoImageProcessor

# 사전 학습된 ViT와 동일한 전처리 설정(크기, 평균, 표준편차) 불러오기
image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k"
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),                              # PIL → Tensor
        transforms.Resize(
            size=(
                image_processor.size["height"],             # 28×28 → 224×224
                image_processor.size["width"]
            )
        ),
        transforms.Lambda(
            lambda x: torch.cat([x, x, x], 0)               # 단일 채널(흑백) → 3채널로 복제
        ),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ]
)

print(f"size : {image_processor.size}")
print(f"mean : {image_processor.image_mean}")
print(f"std : {image_processor.image_std}")
```

```
size : {'height': 224, 'width': 224}
mean : [0.5, 0.5, 0.5]
std : [0.5, 0.5, 0.5]
```

- `google/vit-base-patch16-224-in21k`: ImageNet-21k로 학습, 224×224 입력, 16×16 패치 기반

### 예제 10.3 ViT 데이터로더 적용

```python
from torch.utils.data import DataLoader

# ViT 입력 형식 {"pixel_values": ..., "labels": ...}에 맞추는 집합 함수(collate_fn)
def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])   # (배치, 3, 224, 224)
    labels = torch.tensor([label for label in labels])
    return {"pixel_values": pixel_values, "labels": labels}

train_dataloader = DataLoader(
    subset_train_dataset,
    batch_size=32,
    shuffle=True,
    collate_fn=lambda x: collator(x, transform),
    drop_last=True
)
valid_dataloader = DataLoader(
    subset_test_dataset,
    batch_size=4,
    shuffle=True,
    collate_fn=lambda x: collator(x, transform),
    drop_last=True
)

batch = next(iter(train_dataloader))
for key, value in batch.items():
    print(f"{key} : {value.shape}")
```

```
pixel_values : torch.Size([32, 3, 224, 224])
labels : torch.Size([32])
```

### 예제 10.4 사전 학습된 ViT 모델

```python
from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
    num_labels=len(classes),                                        # 분류 클래스 수 10개로 변경
    id2label={idx: label for label, idx in class_to_idx.items()},  # ID → 레이블
    label2id=class_to_idx,                                          # 레이블 → ID
    ignore_mismatched_sizes=True                                    # 분류기 크기 불일치 무시(새로 초기화)
)

print(model.classifier)
```

```
Linear(in_features=768, out_features=10, bias=True)
```

- 원래 21,841개 클래스로 학습된 모델 → 분류기 출력 차원을 10으로 바꿔 미세 조정

### 예제 10.5 패치 임베딩 확인

```python
print(model.vit.embeddings)

batch = next(iter(train_dataloader))
print("image shape :", batch["pixel_values"].shape)
print("patch embeddings shape :",
    model.vit.embeddings.patch_embeddings(batch["pixel_values"]).shape
)
print("[CLS] + patch embeddings shape :",
    model.vit.embeddings(batch["pixel_values"]).shape
)
```

```
ViTEmbeddings(
  (patch_embeddings): ViTPatchEmbeddings(
    (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
  )
  (dropout): Dropout(p=0.0, inplace=False)
)
image shape : torch.Size([32, 3, 224, 224])
patch embeddings shape : torch.Size([32, 196, 768])
[CLS] + patch embeddings shape : torch.Size([32, 197, 768])
```

- 패치 임베딩 = 16×16 커널, stride 16의 Conv2d → (224−16)/16 + 1 = 14 → 14×14 = **196개 패치**
- [CLS] 토큰 1개 추가 → **197개 벡터** (각 768차원)

### 예제 10.6 하이퍼파라미터 설정

```python
from transformers import TrainingArguments

args = TrainingArguments(
    output_dir="../models/ViT-FashionMNIST",   # 체크포인트 저장 경로
    save_strategy="epoch",                     # 에폭마다 저장
    evaluation_strategy="epoch",               # 에폭마다 평가
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.001,                        # 가중치 감쇠
    load_best_model_at_end=True,               # 학습 종료 시 최상의 모델 불러오기
    metric_for_best_model="f1",                # 최상의 모델 선정 기준
    logging_dir="logs",
    logging_steps=125,
    remove_unused_columns=False,               # 사용자 정의 collator 사용 시 입력 컬럼 유지
    seed=7
)
```

- 참고: 최신 `transformers` 버전에서는 `evaluation_strategy` → `eval_strategy`로 이름 변경

### 예제 10.7 매크로 평균 F1 점수

```python
import evaluate
import numpy as np

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)       # 가장 확률이 높은 클래스 선택
    macro_f1 = metric.compute(
        predictions=predictions, references=labels, average="macro"   # 클래스별 F1의 평균
    )
    return macro_f1
```

### 예제 10.8 ViT 모델 학습

```python
import torch
import evaluate
import numpy as np
from itertools import chain
from collections import defaultdict
from torch.utils.data import Subset
from torchvision import datasets
from torchvision import transforms
from transformers import AutoImageProcessor
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer


def subset_sampler(dataset, classes, max_len):
    target_idx = defaultdict(list)
    for idx, label in enumerate(dataset.train_labels):
        target_idx[int(label)].append(idx)

    indices = list(
        chain.from_iterable(
            [target_idx[idx][:max_len] for idx in range(len(classes))]
        )
    )
    return Subset(dataset, indices)

# Trainer가 모델을 함수 형태로 생성하도록 model_init 정의
def model_init(classes, class_to_idx):
    model = ViTForImageClassification.from_pretrained(
        pretrained_model_name_or_path="google/vit-base-patch16-224-in21k",
        num_labels=len(classes),
        id2label={idx: label for label, idx in class_to_idx.items()},
        label2id=class_to_idx,
    )
    return model

def collator(data, transform):
    images, labels = zip(*data)
    pixel_values = torch.stack([transform(image) for image in images])
    labels = torch.tensor([label for label in labels])
    return {"pixel_values": pixel_values, "labels": labels}

def compute_metrics(eval_pred):
    metric = evaluate.load("f1")
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    macro_f1 = metric.compute(
        predictions=predictions, references=labels, average="macro"
    )
    return macro_f1

train_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=True)
test_dataset = datasets.FashionMNIST(root="../datasets", download=True, train=False)

classes = train_dataset.classes
class_to_idx = train_dataset.class_to_idx

subset_train_dataset = subset_sampler(
    dataset=train_dataset, classes=train_dataset.classes, max_len=1000
)
subset_test_dataset = subset_sampler(
    dataset=test_dataset, classes=test_dataset.classes, max_len=100
)

image_processor = AutoImageProcessor.from_pretrained(
    pretrained_model_name_or_path="google/vit-base-patch16-224-in21k"
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(
            size=(
                image_processor.size["height"],
                image_processor.size["width"]
            )
        ),
        transforms.Lambda(
            lambda x: torch.cat([x, x, x], 0)
        ),
        transforms.Normalize(
            mean=image_processor.image_mean,
            std=image_processor.image_std
        )
    ]
)

args = TrainingArguments(
    output_dir="../models/ViT-FashionMNIST",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.001,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="logs",
    logging_steps=125,
    remove_unused_columns=False,
    seed=7
)

# 훈련자 클래스: 데이터로더 구성·학습·평가를 내부적으로 수행
trainer = Trainer(
    model_init=lambda x: model_init(classes, class_to_idx),
    args=args,
    train_dataset=subset_train_dataset,
    eval_dataset=subset_test_dataset,
    data_collator=lambda x: collator(x, transform),
    compute_metrics=compute_metrics,
    tokenizer=image_processor,       # 이미지 프로세서를 토크나이저 자리에 전달
)
trainer.train()
```

- 교재 학습 결과 (표 10.3)

| Epoch | Training Loss | Validation Loss | F1-Score |
|---|---|---|---|
| 1 | 0.7062 | 0.6377 | 0.8905 |
| 2 | 0.4954 | 0.4781 | 0.9166 |
| 3 | 0.4078 | 0.4357 | 0.9231 |

- 저장된 체크포인트 사용 시 `pretrained_model_name_or_path="../models/ViT-FashionMNIST/checkpoint-1875"` 형태로 지정

### 예제 10.9 ViT 모델 성능 평가

```python
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

outputs = trainer.predict(subset_test_dataset)
print(outputs)

y_true = outputs.label_ids
y_pred = outputs.predictions.argmax(1)

labels = list(classes)
matrix = confusion_matrix(y_true, y_pred)                  # 혼동 행렬
display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=labels)
_, ax = plt.subplots(figsize=(10, 10))
display.plot(xticks_rotation=45, ax=ax)
plt.show()
```

```
# 생략
{
    'test_loss': 0.43576720356941223,
    'test_f1': 0.9231955290222554,
    'test_runtime': 8.4356,
    'test_samples_per_second': 29.636,
    'test_steps_per_second': 4.283
}
# 생략
```

- 테스트 정확도 약 92%
- 혼동 행렬 분석: 'Shirt'를 'T-shirt/top'(10개)·'Coat'(9개)로 오분류한 경우가 가장 많음 (Shirt 정답 72/100)
- 개선 방향: 하이퍼파라미터 조정, 모델 구조 변경, 전처리 개선, 데이터 증강
