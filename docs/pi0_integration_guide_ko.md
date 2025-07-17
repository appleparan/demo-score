# LeRobot PI0 모델과 Demo-SCORE 통합 가이드

## 개요

이 가이드는 LeRobot PI0(Policy Influence 0) 모델을 Demo-SCORE 데이터셋 큐레이션 시스템과 통합하는 방법을 설명합니다. 주요 과제는 PI0 모델의 `embed_prefix` 메소드 출력(`prefix_embs`)을 Demo-SCORE의 입력 형식에 맞게 적응시키는 것입니다.

## 현재 Demo-SCORE 아키텍처

Demo-SCORE는 분류기 기반 접근법을 사용하여 시연 데이터셋을 필터링합니다:

1. **데이터셋 형식**: `(sequence_length, state_dim)` 형태의 `observation.state` 텐서 사용
2. **분류기 모델**: MLPPoolMLP, TransformerClassifier, 또는 StepwiseMLPClassifier
3. **필터링 프로세스**: 시연을 성공/실패로 분류하고 이에 따라 필터링

### 지원되는 데이터셋 형식

- **lerobot**: LeRobotDataset의 `observation.state` 사용
- **robodiff**: HDF5 파일의 액션 시퀀스 사용
- **aloha**: 관절 위치와 속도 사용

## PI0 모델 통합 전략

### 1. PI0 Prefix Embeddings 이해

PI0 모델의 `embed_prefix` 메소드는 다음을 나타내는 `prefix_embs`를 생성합니다:
- 시연 접두사의 컨텍스트 정보
- 인코딩된 행동 패턴
- 정책 관련 상태 표현

### 2. Demo-SCORE용 PI0 적응

이 저장소의 lerobot 구현에는 PI0가 포함되어 있지 않으므로, 다음을 수행해야 합니다:

#### 1단계: 저장소에 PI0 모델 추가

```bash
# lerobot 정책에 PI0 구현 추가
mkdir -p lerobot/lerobot/common/policies/pi0
```

다음 파일들을 생성하세요:
- `lerobot/lerobot/common/policies/pi0/configuration_pi0.py`
- `lerobot/lerobot/common/policies/pi0/modeling_pi0.py`

#### 2단계: 데이터셋 클래스 수정

PI0 임베딩을 처리하기 위해 `ClassifierDataset` 클래스를 확장하세요:

```python
# demo_score/demo_score/dataset.py에 추가

class PI0ClassifierDataset(ClassifierDataset):
    def __init__(self, data_dir, pi0_model, data_root='./data', format='lerobot', **kwargs):
        super().__init__(data_dir, data_root, format=format, **kwargs)
        self.pi0_model = pi0_model
        self.pi0_model.eval()
        
    def _get_pi0_embeddings(self, observations):
        """
        PI0 모델에서 prefix 임베딩 추출
        
        Args:
            observations: 입력 관찰 텐서
            
        Returns:
            prefix_embs: PI0에서 임베딩된 표현
        """
        with torch.no_grad():
            # PI0FlowMatching에 embed_prefix 메소드가 있다고 가정
            prefix_embs = self.pi0_model.embed_prefix(observations)
        return prefix_embs
    
    def __getitem__(self, index):
        if self.format == 'lerobot':
            # 표준 관찰 가져오기
            obs, label = super().__getitem__(index)
            
            # PI0 임베딩 가져오기
            prefix_embs = self._get_pi0_embeddings(obs)
            
            # PI0 임베딩과 결합하거나 대체
            return prefix_embs, label
        else:
            return super().__getitem__(index)
```

#### 3단계: PI0 호환 분류기 생성

```python
# demo_score/demo_score/models.py에 추가

class PI0Classifier(nn.Module):
    def __init__(self, pi0_embed_dim, hidden_sizes=[128, 64], dropout_prob=0.3):
        super(PI0Classifier, self).__init__()
        
        layers = []
        input_size = pi0_embed_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_prob)
            ])
            input_size = hidden_size
        
        layers.append(nn.Linear(input_size, 1))
        self.model = nn.Sequential(*layers)
        
    def forward(self, prefix_embs):
        """
        Args:
            prefix_embs: PI0 embed_prefix 메소드의 출력
        """
        # 다양한 prefix_embs 형태 처리
        if len(prefix_embs.shape) == 3:  # (batch, seq, embed_dim)
            prefix_embs = prefix_embs.mean(dim=1)  # 시퀀스에 대한 풀링
        
        return torch.sigmoid(self.model(prefix_embs))
```

#### 4단계: PI0 필터 생성

```python
# demo_score/demo_score/filters/classifier_filter_pi0.py 생성

import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching
from pathlib import Path
from safetensors.torch import load_file
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def classifier_filter_pi0(orig_datasets, new_datasets_file, pi0_model, classifier_model, 
                         rew_thresh=3.99, classifier_thresh=0.5):
    """
    PI0 임베딩과 분류기를 사용하여 데이터셋 필터링
    
    Args:
        orig_datasets: 원본 데이터셋 경로 목록
        new_datasets_file: 필터링된 데이터셋 저장 경로
        pi0_model: 훈련된 PI0FlowMatching 모델
        classifier_model: PI0 임베딩용 훈련된 분류기
        rew_thresh: 성공을 위한 보상 임계값
        classifier_thresh: 필터링을 위한 분류기 임계값
    """
    
    pi0_model = pi0_model.to(device)
    classifier_model = classifier_model.to(device)
    
    pi0_model.eval()
    classifier_model.eval()
    
    orig_datasets_lines = []
    
    # 원본 데이터셋 파싱
    for orig_datasets_file in orig_datasets:
        if '.txt' in orig_datasets_file:
            with open(orig_datasets_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    orig_datasets_lines.append(line)
        else:
            path = Path('/') / orig_datasets_file / "meta_data" / "episode_data_index.safetensors"
            ep_data_index = load_file(path)
            num_episodes = int(ep_data_index['to'].shape[0])
            line = orig_datasets_file + " " + ",".join([str(el) for el in list(range(num_episodes))])
            orig_datasets_lines.append(line)
    
    new_lines = []
    for line in orig_datasets_lines:
        fname = line.split(' ')[0]
        path = Path('/') / fname / "meta_data" / "episode_data_index.safetensors"
        ep_data_index = load_file(path)
        take_eps = []
        
        for episode_idx in line.split(' ')[1].split(','):
            episode_idx = int(episode_idx)
            dataset_repo_root = '/'
            dataset_repo_id = fname
            split = f"train[{int(ep_data_index['from'][episode_idx])}:{int(ep_data_index['to'][episode_idx])}]"
            
            old_ep_dataset = LeRobotDataset(dataset_repo_id, dataset_repo_root, split=split)
            
            # 성공 확인
            success = False
            if 'next.reward' in old_ep_dataset.hf_dataset.features:
                final_rew = old_ep_dataset.hf_dataset['next.reward'][-1]
                if final_rew > rew_thresh:
                    success = True
            else:
                success = True
            
            if success:
                # 관찰 가져오기
                inputs = old_ep_dataset.hf_dataset['observation.state'][:]
                inputs = torch.stack(inputs).to(device)
                
                with torch.inference_mode():
                    # PI0 임베딩 가져오기
                    prefix_embs = pi0_model.embed_prefix(inputs.unsqueeze(0))
                    
                    # PI0 임베딩을 사용하여 분류
                    pred = classifier_model(prefix_embs).squeeze(0)
                    
                    if pred.shape[0] > 1:
                        pred = pred.mean()
                    
                    if pred.item() > classifier_thresh:
                        take_eps.append(episode_idx)
        
        if len(take_eps) > 0:
            new_line = fname + " " + ",".join([str(el) for el in take_eps])
            new_lines.append(new_line)
    
    # 필터링된 데이터셋 저장
    os.makedirs(os.path.dirname(new_datasets_file), exist_ok=True)
    with open(new_datasets_file, 'w') as f:
        for nline in new_lines:
            f.write(nline + "\n")
```

## 사용 예시

### 1. PI0 기반 분류기 훈련

```python
from demo_score.dataset import PI0ClassifierDataset
from demo_score.models import PI0Classifier
from lerobot.common.policies.pi0.modeling_pi0 import PI0FlowMatching

# 사전 훈련된 PI0 모델 로드
pi0_model = PI0FlowMatching.from_pretrained("path/to/pi0/model")

# PI0 임베딩을 사용한 데이터셋 생성
dataset = PI0ClassifierDataset(
    data_dir="path/to/dataset",
    pi0_model=pi0_model,
    format='lerobot'
)

# PI0 임베딩용 분류기 생성
classifier = PI0Classifier(
    pi0_embed_dim=pi0_model.embed_dim,  # PI0 모델에서 가져오기
    hidden_sizes=[128, 64]
)

# 분류기 훈련
# ... 훈련 코드 ...
```

### 2. 데이터셋 필터링

```python
from demo_score.filters.classifier_filter_pi0 import classifier_filter_pi0

# PI0 임베딩을 사용하여 데이터셋 필터링
classifier_filter_pi0(
    orig_datasets=["path/to/original/dataset"],
    new_datasets_file="path/to/filtered/dataset.txt",
    pi0_model=pi0_model,
    classifier_model=trained_classifier,
    classifier_thresh=0.7
)
```

## 통합 단계 요약

1. **PI0 모델 추가**: lerobot 정책 디렉토리에 PI0FlowMatching 구현 포함
2. **데이터셋 클래스 확장**: PI0 임베딩을 처리하는 PI0ClassifierDataset 생성
3. **PI0 분류기 생성**: PI0 임베딩과 함께 작동하는 분류기 구축
4. **필터 구현**: 데이터셋 큐레이션을 위한 PI0 전용 필터 생성
5. **훈련 파이프라인**: PI0 임베딩에 대한 분류기 훈련
6. **필터링 적용**: 훈련된 분류기를 사용하여 데이터셋 필터링

## 주요 고려사항

### 임베딩 차원
- PI0 `prefix_embs` 차원은 분류기 입력과 일치해야 함
- PI0 출력 형식에 따라 풀링 또는 리셰이핑이 필요할 수 있음

### 훈련 데이터
- 분류기 훈련을 위한 성공/실패 시연 필요
- 보상 임계값 또는 수동 라벨링 사용 고려

### 성능 최적화
- 재계산을 피하기 위한 PI0 임베딩 캐싱
- 대규모 데이터셋에 대한 배치 처리 사용
- 대용량 임베딩에 대한 GPU 메모리 사용량 고려

## 문제 해결

### 일반적인 문제

1. **PI0 모델을 찾을 수 없음**: PI0 구현이 lerobot 정책에 올바르게 추가되었는지 확인
2. **차원 불일치**: PI0 임베딩 차원을 확인하고 그에 따라 분류기 조정
3. **메모리 문제**: 더 작은 배치 크기 사용 또는 임베딩 캐싱 구현
4. **성능**: 실시간 필터링을 위한 더 가벼운 분류기 아키텍처 고려

### 디버깅 팁

- 호환성 확인을 위한 임베딩 형태 출력
- 먼저 작은 데이터셋에서 테스트
- 임베딩 품질 이해를 위한 시각화 도구 사용
- 분류기 훈련 수렴 모니터링

## 미래 개선사항

1. **멀티모달 통합**: PI0 임베딩을 다른 모달리티와 결합
2. **적응형 임계값**: 데이터셋 특성에 기반한 동적 임계값 조정
3. **앙상블 방법**: 더 나은 필터링을 위한 여러 PI0 모델 결합
4. **온라인 학습**: 새로운 시연에 기반한 분류기 업데이트

## 질문에 대한 직접적인 답변

**질문**: lerobot_hf의 `PI0FlowMatching` 클래스의 `embed_prefix` 메소드에서 생성된 `prefix_embs`를 demo-score 데이터셋 입력에 어떻게 사용할 수 있는가?

**답변**: 

1. **현재 상황**: 이 저장소에는 PI0 모델이 포함되어 있지 않습니다.

2. **통합 방법**: 
   - PI0 모델을 lerobot 정책에 추가
   - `prefix_embs`를 demo-score의 `observation.state` 형식으로 변환
   - PI0 임베딩용 전용 분류기 생성
   - 새로운 필터링 파이프라인 구현

3. **핵심 아이디어**: `prefix_embs`를 직접 분류기 입력으로 사용하거나, 기존 상태 정보와 연결하여 사용할 수 있습니다.

이 문서는 PI0 모델의 `prefix_embs`를 Demo-SCORE 시스템과 통합하는 완전한 가이드를 제공합니다.