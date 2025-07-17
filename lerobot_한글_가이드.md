# LeRobot 한글 사용 가이드

이 문서는 LeRobot의 실험 스크립트와 설정 시스템을 처음 사용하는 분들을 위한 가이드입니다.

## 1. `lerobot/experiments/example` 스크립트 파라미터 가이드

### 개요
`lerobot/experiments/example` 폴더에는 로봇 학습을 위한 5단계 실험 과정이 스크립트로 정리되어 있습니다. 각 단계는 순서대로 실행되며, 다음과 같은 과정을 거칩니다:

```
1단계: 기본 정책 학습 → 2단계: 롤아웃 수집 → 3단계: 분류기 학습 → 4단계: 필터 적용 → 5단계: 최종 정책 학습
```

### 1.1 step1_train_base_policy.sh - 기본 정책 학습

**이 스크립트가 하는 일:**
- 전체 시연 데이터셋을 사용하여 기본 로봇 정책을 학습합니다
- 학습된 모델은 이후 단계에서 롤아웃 데이터를 생성하는 데 사용됩니다

**주요 파라미터 설명:**

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `policy=act` | 사용할 정책 알고리즘 | `act` (Action Chunking with Transformers) |
| `env=aloha` | 실험 환경 설정 | `aloha` (Aloha 로봇 환경) |
| `wandb.enable=True` | 실험 결과 추적 도구 활성화 | `True` (활성화) / `False` (비활성화) |
| `hydra.job.name=ex1_seed10000_base_policy` | 실험 이름 | 원하는 실험 이름으로 변경 가능 |
| `training.offline_steps=300000` | 학습 반복 횟수 | `300000` (30만 번 학습) |
| `training.eval_freq=10000` | 평가 주기 | `10000` (1만 번마다 평가) |
| `training.save_freq=10000` | 모델 저장 주기 | `10000` (1만 번마다 저장) |
| `seed=10000` | 랜덤 시드 값 | `10000` (결과 재현을 위한 난수 고정) |
| `training.log_freq=100` | 로그 출력 주기 | `100` (100번마다 로그 출력) |
| `eval.n_episodes=256` | 평가 에피소드 수 | `256` (256개 에피소드로 평가) |
| `eval.batch_size=64` | 평가 배치 크기 | `64` (한 번에 64개씩 처리) |
| `+datasets_file=` | 데이터셋 파일 경로 | 사용할 데이터셋 파일 위치 |

**초보자를 위한 팁:**
- `seed` 값을 같게 하면 동일한 결과를 얻을 수 있습니다
- `training.offline_steps` 값이 클수록 학습이 오래 걸리지만 성능이 향상될 수 있습니다
- `MUJOCO_GL=egl`은 그래픽 설정으로, 서버 환경에서 필요한 설정입니다

### 1.2 step2_collect_rollouts.sh - 롤아웃 데이터 수집

**이 스크립트가 하는 일:**
- 1단계에서 학습한 정책의 여러 체크포인트를 사용하여 롤아웃 데이터를 수집합니다
- 수집된 데이터는 나중에 분류기 학습에 사용됩니다

**주요 파라미터 설명:**

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `eps=("070000" "150000" "220000" "300000")` | 사용할 체크포인트 목록 | 학습 단계별 모델 저장 시점 |
| `eval.n_episodes=100` | 수집할 에피소드 수 | `100` (100개 에피소드 수집) |
| `eval.batch_size=100` | 배치 크기 | `100` (한 번에 100개씩 처리) |
| `seed=12000` | 랜덤 시드 | `12000` (학습 시와 다른 시드 사용) |
| `-p` | 정책 모델 경로 | 1단계에서 저장된 모델 위치 |
| `--out-dir` | 출력 디렉토리 | 수집된 데이터를 저장할 위치 |

**초보자를 위한 팁:**
- `eps` 배열의 숫자들은 1단계에서 저장된 체크포인트 번호입니다
- `seed=12000`을 사용하는 이유는 학습 시(`seed=10000`)와 다른 환경을 만들기 위함입니다
- 각 체크포인트마다 별도의 폴더에 롤아웃 데이터가 저장됩니다

### 1.3 step3_classifier_sweep.py - 분류기 학습 스위핑

**이 스크립트가 하는 일:**
- 다양한 분류기 모델을 학습시켜 어떤 롤아웃 데이터가 좋은 품질인지 판단하는 분류기를 만듭니다
- 여러 설정으로 실험하여 최적의 분류기를 찾습니다

**주요 파라미터 설명:**

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `model_dicts` | 분류기 모델 구조 설정 | 다양한 신경망 구조 옵션 |
| `hidden_sizes: [8, 8]` | 신경망 레이어 크기 | `[8, 8]` (8개 뉴런 2개 레이어) |
| `eps_dict` | 사용할 에피소드 설정 | 학습/검증용 에피소드 지정 |
| `train: ["070000", "150000", "220000"]` | 학습용 체크포인트 | 분류기 학습에 사용할 데이터 |
| `cross_val: ["300000"]` | 검증용 체크포인트 | 분류기 성능 평가용 데이터 |
| `tr_dataset_size = 100` | 학습 데이터 크기 | `100` (100개 샘플로 학습) |
| `cross_val_dataset_size = 100` | 검증 데이터 크기 | `100` (100개 샘플로 검증) |
| `weight_decay = 0.1` | 가중치 감소 | `0.1` (과적합 방지) |
| `lr = 1e-4` | 학습률 | `0.0001` (학습 속도 조절) |
| `steps: 100` | 학습 반복 횟수 | `100` (100번 학습) |

**초보자를 위한 팁:**
- `hidden_sizes`에서 숫자가 클수록 복잡한 분류기가 됩니다
- `lr` (학습률)이 너무 크면 학습이 불안정해지고, 너무 작으면 학습이 느려집니다
- 여러 설정으로 실험하여 가장 좋은 성능을 내는 분류기를 찾습니다

### 1.4 step4_apply_filter.py - 필터 적용

**이 스크립트가 하는 일:**
- 3단계에서 학습한 분류기를 사용하여 롤아웃 데이터를 필터링합니다
- 품질이 좋은 데이터만 선별하여 새로운 데이터셋을 만듭니다

**주요 파라미터 설명:**

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `expt_dir='data/example'` | 실험 디렉토리 경로 | 데이터가 저장된 위치 |
| `run_name='ex1_seed10000'` | 실험 이름 | 1단계에서 사용한 실험 이름 |
| `eps_dict` | 에피소드 설정 | 3단계에서 사용한 설정 재사용 |
| `dataset_type='lerobot'` | 데이터셋 형식 | `lerobot` 형식 사용 |
| `model_arch='small_stepwise'` | 사용할 분류기 모델 | 3단계에서 학습한 모델 선택 |

**초보자를 위한 팁:**
- 이 단계에서는 품질이 좋은 데이터만 선별됩니다
- 필터링 결과는 `demo_score_datasets` 폴더에 저장됩니다
- 필터링된 데이터는 마지막 단계에서 더 나은 정책 학습에 사용됩니다

### 1.5 step5_train_demo_score_policy.sh - 최종 정책 학습

**이 스크립트가 하는 일:**
- 4단계에서 필터링된 고품질 데이터를 사용하여 최종 정책을 학습합니다
- 두 가지 버전을 제공합니다: 시연 데이터만 사용하는 버전과 시연+롤아웃 데이터를 함께 사용하는 버전

**주요 파라미터 설명:**

| 파라미터 | 설명 | 예시 값 |
|---------|------|---------|
| `hydra.job.name` | 실험 이름 | `ex1_seed10000_demo_score_only_demos` |
| `seed=14000` | 랜덤 시드 | `14000` (이전 단계와 다른 시드) |
| `+datasets_file` | 사용할 데이터셋 파일 | 필터링된 데이터 파일 경로 |
| `demo_score_demos_only.txt` | 시연 데이터만 사용 | 원본 시연 데이터에서 필터링된 것 |
| `demo_score_rollouts_plus_demos.txt` | 시연+롤아웃 데이터 사용 | 시연 데이터 + 롤아웃 데이터 |

**초보자를 위한 팁:**
- 첫 번째 명령어는 원본 시연 데이터 중 좋은 것만 사용합니다
- 두 번째 명령어는 시연 데이터와 롤아웃 데이터를 모두 사용합니다
- 보통 두 번째 방법이 더 많은 데이터를 사용하므로 성능이 더 좋을 수 있습니다

## 2. `lerobot/lerobot/configs` 설정 시스템 가이드

### 2.1 Hydra 프레임워크 소개

LeRobot은 **Hydra**라는 설정 관리 프레임워크를 사용합니다. Hydra는 설정 파일을 체계적으로 관리할 수 있게 해주는 도구입니다.

**Hydra의 주요 특징:**
- **계층적 설정**: 설정을 여러 파일로 나누어 관리
- **조합 가능**: 다양한 설정을 조합하여 사용
- **명령어 오버라이드**: 실행 시 설정 값을 변경 가능

### 2.2 설정 파일 구조

```
lerobot/lerobot/configs/
├── default.yaml          # 기본 설정
├── env/                  # 환경별 설정
│   ├── aloha.yaml       # Aloha 로봇 환경
│   ├── pusht.yaml       # PushT 환경
│   └── ...
└── policy/               # 정책별 설정
    ├── act.yaml         # ACT 정책
    ├── diffusion.yaml   # Diffusion 정책
    └── ...
```

### 2.3 default.yaml - 기본 설정

이 파일은 모든 실험의 기본 설정을 담고 있습니다.

**주요 설정 섹션:**

#### 2.3.1 기본 설정
```yaml
defaults:
  - _self_
  - env: pusht          # 기본 환경은 pusht
  - policy: diffusion   # 기본 정책은 diffusion
```

**설명:**
- `defaults` 섹션은 기본으로 사용할 설정을 지정합니다
- `env: pusht`는 환경 설정으로 `pusht.yaml`을 사용한다는 의미입니다
- `policy: diffusion`은 정책 설정으로 `diffusion.yaml`을 사용한다는 의미입니다

#### 2.3.2 실행 설정
```yaml
hydra:
  run:
    dir: outputs/train/${now:%Y-%m-%d}/${now:%H-%M-%S}_${env.name}_${policy.name}_${hydra.job.name}
  job:
    name: default
```

**설명:**
- `hydra.run.dir`: 실험 결과를 저장할 디렉토리 경로
- `${now:%Y-%m-%d}`: 현재 날짜 (예: 2024-01-15)
- `${now:%H-%M-%S}`: 현재 시간 (예: 14-30-25)
- `${env.name}`: 환경 이름
- `${policy.name}`: 정책 이름

#### 2.3.3 하드웨어 설정
```yaml
device: cuda     # GPU 사용
use_amp: false   # 자동 혼합 정밀도 사용 안 함
seed: ???        # 시드 값은 실행 시 지정 필요
```

**설명:**
- `device: cuda`: GPU를 사용하여 학습 (CPU 사용 시 `cpu`로 변경)
- `use_amp: false`: 메모리 절약을 위한 혼합 정밀도 사용 여부
- `seed: ???`: 반드시 실행 시 지정해야 하는 값

#### 2.3.4 데이터 설정
```yaml
dataset_repo_id: lerobot/pusht  # 사용할 데이터셋
video_backend: pyav             # 비디오 처리 백엔드
```

#### 2.3.5 학습 설정
```yaml
training:
  offline_steps: ???      # 학습 반복 횟수 (필수 지정)
  eval_freq: ???          # 평가 주기 (필수 지정)
  log_freq: 200           # 로그 출력 주기
  save_freq: ???          # 모델 저장 주기 (필수 지정)
  batch_size: ???         # 배치 크기 (필수 지정)
  num_workers: 4          # 데이터 로딩 워커 수
```

**초보자를 위한 팁:**
- `???`가 있는 항목은 실행 시 반드시 값을 지정해야 합니다
- `offline_steps`가 클수록 학습이 오래 걸리지만 성능이 향상될 수 있습니다
- `batch_size`는 GPU 메모리에 따라 조절해야 합니다

#### 2.3.6 이미지 변환 설정
```yaml
training:
  image_transforms:
    enable: false                    # 이미지 변환 사용 안 함
    max_num_transforms: 3            # 최대 변환 수
    brightness:
      weight: 1
      min_max: [0.8, 1.2]          # 밝기 변환 범위
    contrast:
      weight: 1
      min_max: [0.8, 1.2]          # 대비 변환 범위
```

**설명:**
- 이미지 변환은 데이터 증강을 위한 기능입니다
- `enable: true`로 설정하면 학습 시 이미지에 다양한 변환을 적용합니다
- 각 변환은 `weight` (확률)과 `min_max` (범위)로 설정됩니다

#### 2.3.7 평가 설정
```yaml
eval:
  n_episodes: 1           # 평가 에피소드 수
  batch_size: 1           # 평가 배치 크기
  use_async_envs: false   # 비동기 환경 사용 안 함
```

#### 2.3.8 실험 추적 설정
```yaml
wandb:
  enable: false           # Weights & Biases 사용 안 함
  project: lerobot        # 프로젝트 이름
  notes: ""               # 실험 메모
```

**설명:**
- Weights & Biases는 실험 결과를 추적하고 시각화하는 도구입니다
- `enable: true`로 설정하면 실험 과정과 결과를 자동으로 기록합니다

### 2.4 환경 설정 (env/)

#### 2.4.1 aloha.yaml - Aloha 로봇 환경
```yaml
fps: 50                    # 초당 프레임 수

env:
  name: aloha              # 환경 이름
  task: AlohaInsertion-v0  # 작업 이름
  state_dim: 14            # 상태 차원
  action_dim: 14           # 액션 차원
  fps: ${fps}              # 프레임 속도
  episode_length: 400      # 에피소드 길이
```

**설명:**
- `fps: 50`: 1초에 50프레임으로 실행
- `state_dim: 14`: 로봇의 상태는 14차원 벡터
- `action_dim: 14`: 로봇의 액션은 14차원 벡터
- `episode_length: 400`: 한 에피소드는 400스텝

#### 2.4.2 pusht.yaml - PushT 환경
```yaml
fps: 10                    # 초당 프레임 수

env:
  name: pusht              # 환경 이름
  task: PushT-v0           # 작업 이름
  image_size: 96           # 이미지 크기
  state_dim: 2             # 상태 차원
  action_dim: 2            # 액션 차원
  episode_length: 300      # 에피소드 길이
```

**설명:**
- PushT는 2D 환경이므로 상태와 액션이 각각 2차원입니다
- `image_size: 96`: 96x96 픽셀 이미지 사용
- Aloha보다 단순한 환경이므로 fps가 10으로 낮습니다

### 2.5 정책 설정 (policy/)

#### 2.5.1 act.yaml - ACT 정책
```yaml
seed: 1000                                    # 시드 값
dataset_repo_id: lerobot/aloha_sim_insertion_human  # 데이터셋

training:
  offline_steps: 100000    # 학습 반복 횟수
  batch_size: 8           # 배치 크기
  lr: 1e-5                # 학습률
  weight_decay: 1e-4      # 가중치 감소

policy:
  name: act               # 정책 이름
  n_obs_steps: 1          # 관찰 스텝 수
  chunk_size: 100         # 청크 크기
  n_action_steps: 100     # 액션 스텝 수
  
  # 신경망 구조
  vision_backbone: resnet18        # 비전 백본
  dim_model: 512                   # 모델 차원
  n_heads: 8                       # 어텐션 헤드 수
  n_encoder_layers: 4              # 인코더 레이어 수
  n_decoder_layers: 1              # 디코더 레이어 수
  
  # VAE 설정
  use_vae: true                    # VAE 사용
  latent_dim: 32                   # 잠재 차원
```

**초보자를 위한 팁:**
- `chunk_size: 100`: 한 번에 100스텝의 액션을 예측합니다
- `vision_backbone: resnet18`: 이미지 처리를 위해 ResNet18을 사용합니다
- `use_vae: true`: 변분 자동 인코더를 사용하여 액션을 생성합니다

### 2.6 설정 커스터마이징 방법

#### 2.6.1 명령어에서 설정 오버라이드
```bash
# 기본 명령어
python train.py

# 환경 변경
python train.py env=aloha

# 정책 변경
python train.py policy=act

# 여러 설정 동시 변경
python train.py env=aloha policy=act training.batch_size=16

# 새로운 설정 추가
python train.py +my_setting=value
```

#### 2.6.2 새로운 설정 파일 만들기

**1. 새로운 환경 설정 만들기:**
```yaml
# configs/env/my_env.yaml
fps: 25

env:
  name: my_env
  task: MyTask-v0
  state_dim: 10
  action_dim: 5
  fps: ${fps}
  episode_length: 200
```

**2. 새로운 정책 설정 만들기:**
```yaml
# configs/policy/my_policy.yaml
policy:
  name: my_policy
  # 정책 관련 설정들...
```

**3. 사용하기:**
```bash
python train.py env=my_env policy=my_policy
```

#### 2.6.3 실험별 설정 파일 만들기

복잡한 실험을 위해 전용 설정 파일을 만들 수 있습니다:

```yaml
# configs/experiment/my_experiment.yaml
# @package _global_
defaults:
  - /env: aloha
  - /policy: act
  - _self_

seed: 42
dataset_repo_id: my_dataset

training:
  offline_steps: 200000
  batch_size: 16
  lr: 5e-5

eval:
  n_episodes: 100
```

**사용하기:**
```bash
python train.py --config-path configs/experiment --config-name my_experiment
```

## 3. 실제 사용 예시

### 3.1 기본 실험 실행
```bash
# 기본 설정으로 실행
python lerobot/scripts/train.py

# Aloha 환경에서 ACT 정책 사용
python lerobot/scripts/train.py env=aloha policy=act

# 학습 파라미터 조정
python lerobot/scripts/train.py env=aloha policy=act \
  training.offline_steps=100000 \
  training.batch_size=8 \
  seed=1000
```

### 3.2 실험 추적 활성화
```bash
python lerobot/scripts/train.py env=aloha policy=act \
  wandb.enable=true \
  wandb.project=my_project \
  wandb.notes="첫 번째 실험"
```

### 3.3 설정 확인
```bash
# 현재 설정 확인
python lerobot/scripts/train.py --cfg job

# 설정 파일 구조 확인
python lerobot/scripts/train.py --help
```

## 4. 문제 해결 팁

### 4.1 자주 발생하는 오류

**1. `???` 값 오류:**
```
Error: Missing @package directive default.yaml
```
**해결:** 필수 파라미터를 명령어에서 지정하세요.
```bash
python train.py seed=1000 training.offline_steps=100000
```

**2. GPU 메모리 부족:**
```
CUDA out of memory
```
**해결:** 배치 크기를 줄이세요.
```bash
python train.py training.batch_size=4
```

**3. 데이터셋 오류:**
```
Dataset not found
```
**해결:** 올바른 데이터셋 경로를 지정하세요.
```bash
python train.py dataset_repo_id=lerobot/pusht
```

### 4.2 성능 최적화 팁

**1. GPU 사용:**
```bash
python train.py device=cuda
```

**2. 멀티프로세싱:**
```bash
python train.py training.num_workers=8
```

**3. 혼합 정밀도:**
```bash
python train.py use_amp=true
```

## 5. 요약

이 가이드에서는 LeRobot의 두 가지 주요 구성 요소를 다뤘습니다:

1. **실험 스크립트** (`lerobot/experiments/example`): 5단계 실험 워크플로우
2. **설정 시스템** (`lerobot/lerobot/configs`): Hydra 기반 계층적 설정 관리

각 구성 요소는 로봇 학습의 다른 측면을 담당하며, 함께 사용하면 체계적이고 재현 가능한 실험을 수행할 수 있습니다.

초보자는 다음 순서로 학습하는 것을 권장합니다:
1. 기본 설정 이해
2. 간단한 실험 실행
3. 설정 커스터마이징
4. 전체 워크플로우 실행

궁금한 점이 있으시면 각 설정 파일을 직접 열어보시거나, 작은 실험부터 시작해보세요!