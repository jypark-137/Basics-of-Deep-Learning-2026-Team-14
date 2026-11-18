# NuWa 기반 ViTPose Vision Transformer 경량화 및 Ablation Study

> **팀 프로젝트 실행 계획서 | 2026-11-16**  
> **데이터셋:** AP-10K | **기본 모델:** ViTPose-S (일반 ViT 구조) | **환경:** 단일 A100 MIG GPU, 할당 VRAM 약 9.5 GiB  
> **핵심 목표:** 특정 동물 종 집합에 특화한 자세 추정 모델을 NuWa 방식으로 구조적 압축하고, MLP 뉴런 선택 기준을 바꾼 두 개선안의 **정확도–압축률–모델 생성 비용**을 비교한다.

---

## 0. 프로젝트 한눈에 보기

### 0.1 연구 목표

1. AP-10K의 다양한 종을 추정할 수 있도록 사전학습된 ViTPose 계열 모델에서 **특정 종 집합**의 17개 관절을 추정하는 모델을 파생한다.
2. NuWa의 **SKP(Self-Knowledge Purification) + OFP(Optimization-based Fast Pruning)**를 일반 ViTPose의 Transformer Backbone에 적용하도록 확장한다.
3. 기존 NuWa의 MLP 뉴런 우선순위 산정과 다음 두 개선안을 동일한 압축 조건에서 비교한다.
   - **개선안 A:** Post-activation의 Mean + Variance 등 통계 기반 우선순위.
   - **개선안 B:** Pose Loss를 이용하는 경량 Neuron-wise Mask Learning 기반 우선순위.
4. Keypoint AP, 모델 크기, MACs/FLOPs, 추론 지연, SKP/OFP/추가 마스크 학습 시간을 함께 보고한다.

### 0.2 연구 질문

- **RQ1 — 적용 가능성:** Classification용으로 제안된 NuWa를 ViTPose의 Animal Pose Estimation에 확장할 때, 자세 추정 정확도를 크게 훼손하지 않고 실질적인 구조적 압축이 가능한가?
- **RQ2 — 학습 없는 개선:** 추가 역전파 없이 Post-activation의 평균 외 통계량을 사용하면 뉴런 선택 품질이 개선되는가?
- **RQ3 — 학습 기반 개선:** 경량 Mask Learning이 통계 기반 Ranking보다 좋은 정확도를 제공하는가? 그 정확도 개선이 추가 모델 생성 시간을 정당화하는가?
- **RQ4 — 종 특화 효과:** 유사한 종 집합과 이질적인 종 집합에서 동일 압축률의 정확도 손실이 달라지는가? (확장 분석)

### 0.3 비교 기준선의 정의

반드시 다음 세 대상을 구분한다.

| 명칭 | 모델 | 역할 |
|---|---|---|
| **Unpruned baseline** | AP-10K용 일반 ViTPose 모델, Pruning 전 | 정확도·연산량의 기준점 |
| **NuWa baseline** | 동일 ViTPose에 NuWa 방식의 SKP+OFP를 적용한 모델 | **개선안 A/B의 직접 비교 대상** |
| **Proposed A/B** | 같은 NuWa Anchor와 압축 예산에서 MLP 뉴런 선택 점수만 변경 | 제안 방식의 효과 검증 |

**주의:** MLP-only OFP만 구현했다면 이를 **`NuWa-inspired MLP OFP`**라고 명시한다. SKP와 MHA OFP까지 실제로 이식·검증한 뒤에야 **`NuWa-adapted ViTPose`** 또는 이에 준하는 이름을 사용한다. 원논문의 수치나 성능을 그대로 재현했다고 주장해서는 안 된다.

---

## 1. 모델·데이터·환경 확정

### 1.1 데이터와 태스크

- **AP-10K:** 54개 동물 종, 17개 공통 Keypoint. 자세 추정 데이터이며 종 분류 자체가 목적이 아니다.
- **입력:** AP-10K의 동물 Bounding Box를 이용해 잘라낸 Crop(우선 GT Box), ViTPose의 **256×192** 입력·16×16 Patch를 유지한다.
- **출력:** 대상 종이 무엇이든 동일한 **17개 관절 Heatmap**을 예측한다.
- **종별 특화:** 대상 종 집합 `S`를 *모델 생성 전에* 확정한다. Pruning/Calibration/Mask Learning에는 `S`의 **train subset**만 사용하며, 평가에는 `S`의 held-out val/test만 사용한다.
- **추론 제약:** 모델은 특정 종 집합 전용이지만, 이미지별 **정답 종 레이블을 추론 시 사용하지 않는다**. 하나의 고정된 압축 모델로 `S`의 모든 종을 처리한다.

### 1.2 권장 모델 확보 경로

**주 경로(권장):** ViTPose 공식 저장소에서 **ViTPose++-S 학습 완료 체크포인트** 확보 → `tools/model_split.py`로 `ap10k.pth` 추출 → **일반 ViTPose-S 구조**에 로드 → 출력 일치 검증.

- 일반 ViTPose-S: `embed_dim=384`, `depth=12`, `num_heads=12`, `mlp_ratio=4`, FFN hidden=1536.
- AP-10K용 모델 구성: `TopDown + ViT + TopdownHeatmapSimpleHead`, 17 channels.
- `model_split.py`는 **S 전용이 아니므로** Base 체크포인트를 넣으면 Base 구조에 대응하는 가중치가 나온다. `S↔S`, `B↔B`를 엄격히 맞춘다.
- 원본 ViTPose++에는 MoE 기반 Task-specific Expert가 있지만, Split 이후에는 AP-10K 경로의 공유·전용 FC2를 병합한 **보통의 `fc1`/`fc2` MLP**를 사용한다.
- **검증 필요:** `model_split.py`의 Expert key 정리 로직에는 수정이 필요한 부분이 있으므로 생성 파일을 그대로 신뢰하지 않는다. 로딩 시 missing/unexpected keys, 원본·분리 모델의 Heatmap 차이를 검사한다.

**대체 경로:** APTv2 저장소의 ViTPose-B 체크포인트 검토. 단, **APTv2 결과표의 `pretrain=AP-10K`는 APTv2 학습 전 초기화 출처를 뜻할 수 있으며, 제공된 체크포인트 자체가 AP-10K 평가용으로 학습된 모델임을 뜻하지 않는다.** 따라서 APTv2용 추가 학습 여부·17-keypoint 정의·입력 해상도·AP-10K Validation 성능을 검증하고 채택한다. APTv2 저장소 자체는 AP-10K가 아닌 별도 동영상 기반 벤치마크다.

**결정 게이트 G1:** 모델 체크포인트 로딩·Baseline AP 평가가 가능한 경로를 최종 선택한다. 같은 실험표에 서로 다른 출처의 체크포인트를 혼합하지 않는다. 주 실험은 ViTPose-S로 수행하고 B는 시간·VRAM이 허용될 때 확장한다.

### 1.3 실제 서버 환경

| 항목 | 확인 내용 / 운영 원칙 |
|---|---|
| GPU | `NVIDIA A100-SXM4-80GB`의 **MIG 인스턴스**, 실제 할당 **9,728 MiB ≈ 9.5 GiB** |
| 리소스 | 14 SM; 물리 A100 80GB 전체가 아님 |
| 할당 | `ssai_agpu -g 1`; GPU 세션 최대 48시간(제공 가이드 기준) |
| PyTorch | 팀의 `dk14_sjk_torch` 환경에서 GPU CUDA 실행 검증 완료. **이 환경의 기존 버전을 임의로 업그레이드하지 않는다.** |
| 초기 Batch | Pose Forward/Calibration 1–4, Mask Learning 1–2부터 실측하고 조정 |
| 메모리 대응 | AMP는 Forward/Backward용; SVD/Least-squares는 FP32 또는 필요 시 FP64·CPU 처리 |
| 재현성 | Git commit, 환경 잠금 파일, config YAML, seed, 입력 크기, GPU 이름/VRAM 기록 |

**환경 리스크:** 공식 ViTPose는 구형 MMPose/MMCV 및 `timm` 내부 API에 의존한다. 현재 PyTorch 2.6 환경에서 곧바로 실행되는 것은 확인되지 않았다. **별도 Conda 환경에서 Baseline 추론 성공 → 환경 버전 고정**을 먼저 수행하고, 팀 공용 환경의 패키지를 함부로 덮어쓰지 않는다.

---

## 2. 전체 프로젝트 흐름과 통과 기준

```text
[0] 환경 및 코드 재현성 확보
       │
       ▼
[1] AP-10K 종별 데이터 분석·Split 고정 ──────────────┐
       │                                              │
       ▼                                              ▼
[2] ViTPose++ → AP-10K용 일반 ViTPose 기준모델    [평가 API/Metric 검증]
       │                                              │
       └─────────────── Baseline AP·출력 검증 ───────────┘
                              │
                              ▼
[3] NuWa 이식: MLP OFP → (SKP) → (MHA OFP)
                              │
                       Anchor/예산/보정 고정
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
[4A] Mean+Variance Ranking       [4B] Neuron-wise Mask Learning
                 │                         │
                 └────────────┬────────────┘
                              ▼
[5] 동일 조건 Accuracy–MACs–Latency–Derivation Time 비교
                              │
                              ▼
[6] Ablation, 실패 원인 분석, 그림, 최종 보고서·발표
```

### Phase / Gate 계획

| Phase | 작업 | 주요 담당 | 완료 증거(다음 Phase 진입 조건) |
|---|---|---|---|
| P0 | GPU·환경·코드·AP-10K 경로 준비 | **전원**, 통합 박재영 | `environment.md`, import/Forward 통과 |
| P1 | AP-10K 종 분석 및 공통 평가 파이프라인 | **최준혁** | 고정된 `species_sets.json`, 원본 Baseline AP, 평가 검증 테스트 |
| P2 | ViTPose 체크포인트 분리 및 원본 재현 | **박재영** + 최준혁 | 원본·분리 출력 일치, Checkpoint/Config 고정 |
| P3a | MLP-only NuWa OFP 이식 | **박재영** | 실제 FC1/FC2 크기 감소, 정상 AP·MACs 산출 |
| P3b | NuWa SKP·MHA OFP 확장 | **박재영**, 지원 서준교·백지승 | 각 단계 독립 테스트, 원본 방법의 범위 기록 |
| P4a | 통계 기반 Ranking 개발 | **서준교** | 통계 재현, Ranking 추출, 전체 파이프라인 연동 |
| P4b | 학습 기반 Ranking 개발 | **백지승** | Gate 학습 수렴, 공통 Anchor 및 Hard Pruning 연동 |
| P5 | 압축률·종 집합별 실험 | **전원**, 평가 집계 최준혁 | 반복 가능한 Result CSV + 로그 + 저장된 모델 |
| P6 | Ablation, 그래프, 보고서, 발표 | **전원** | 그림·표·방법론·한계 정리 |

**역할 배정 주석:** 사용자 제안에 따라 서준교·백지승 두 사람이 개선안을 맡되, **서준교=A / 백지승=B**는 실행 편의를 위한 제안 배분이다. 두 사람이 작업량을 고려해 바꿔도 인터페이스를 동일하게 유지한다.

---

## 3. 역할별 상세 지침

## 3.1 최준혁 — Evaluation / Validation / Experimental Protocol Lead

### 책임 범위

**(A) AP-10K 데이터 분석과 종 집합 선정**

1. `ap10k-train/val/test-split1.json`의 이미지, annotation, category, bbox, keypoint, visibility 필드 구조 분석.
2. 종별 `#images`, `#animal_instances`, `#visible_keypoints`, bbox 크기·비율, 자세 다양성(가능하면) 요약. **종별 샘플 분포를 확인하기 전 동물 종 이름을 임의로 확정하지 않는다.**
3. 메인 `species_set` 하나 선정: 같은 과(Family)의 2–5개 종 중 학습·평가 샘플이 충분한 조합. 검증용 비교 조합: 가능한 범위에서 같은 종 수의 이질적 종 집합.
4. 원본 AP-10K **공식 Split을 보존**하고, Training 내부에서 `calibration_train` / `mask_train` / `dev_validation`을 분리. 공식 Test는 최종 보고용으로만 사용.
5. `species_sets.json`에 `species_id`, `species_name`, family, split별 수량, 고정 random seed와 선정 이유 저장.

**(B) Pose 평가·검증 코드**

1. `load_pose_model(config, checkpoint)` 구현. 이미지 전처리, GT bbox crop, affine transform, heatmap decode, 좌표 역변환을 **원래 ViTPose 설정**과 동일하게 유지.
2. 주 지표는 **AP-10K 키포인트 OKS 기반 AP**. MMPose/AP-10K가 사용하는 `sigmas`, annotation convention, visible/ignore 처리, matching 규칙을 재사용한다. 사람용 COCO sigmas를 기계적으로 적용하지 않는다.
3. 전체 대상 종 집합의 pooled AP + 종별 AP를 기록한다. *종별 AP의 단순 평균(Macro AP)은 전체 pooled AP와 다른 지표*로 별도 제공.
4. 필요 시 보조 지표 PCK, Keypoint별 AP, Mean Heatmap MSE를 제공하되 공식 AP 대신 주 지표로 오용하지 않는다.
5. GT bbox 평가를 메인으로 통일하여 Detector의 변화가 Pose 모델 결과에 섞이지 않도록 한다. Detector-based 평가는 별도 확장 실험으로만 수행.
6. Batch 1, 동일 입력, 모델 `eval()`, 동일 AMP/FP32, Flip Test on/off를 통일하여 AP 측정. 처음에는 Flip Test off로 디버깅하고 보고용 설정을 고정.
7. `torch.cuda.Event`, 사전 Warmup, `torch.cuda.synchronize()`를 사용하여 추론 지연 측정. GPU 공유·MIG 노이즈 방지를 위해 순차 측정하고 반복 결과(중앙값/분위수)를 기록.
8. Params는 저장된 **실제 Dense Parameter** 수로 기록. Nonzero count는 필요 시 별도 열로 두고, MACs와 FLOPs 정의를 구분. Pose Head를 포함한 Full-model 비용과 Backbone-only 비용을 모두 보고한다.
9. 평가 함수의 테스트 작성: identity 변환(원본 checkpoint 비교), 동일 prediction JSON의 재평가, species filtering, empty subset, coordinate inverse transform 등을 검증.

### 코드/API 산출물

- `src/data/ap10k_species.py` : 종별 로더 및 Subset 필터.
- `src/eval/evaluate_pose.py` : `evaluate_pose(model, split, species_set, config) -> dict`.
- `src/eval/profile_model.py` : Params/MACs/Latency/VRAM 측정.
- `src/eval/metrics.py` : AP/PCK 및 종별 분해 결과.
- `configs/species_sets.json`, `reports/dataset_audit.md`, `results/baseline.json`.

### 완료 기준 및 인수인계

- Unpruned ViTPose의 AP를 **고정된 Validation Split**에서 재현하고, 공식 코드 결과와 차이의 원인을 설명할 수 있다(평가 집합·BBox·후처리·체크포인트 차이 포함).
- 모든 팀원이 **같은 명령과 동일 모델**로 평가했을 때 수치가 허용 오차 안에서 일치한다.
- 박재영에게 모델 forward/evaluate 인터페이스 제공; 서준교·백지승에게 Calibration/Mask Train/Val DataLoader 제공.

---

## 3.2 박재영 — Baseline ViTPose / NuWa Adaptation Lead

### 책임 범위

**(A) 모델 준비 및 공식 체크포인트 검증**

1. 공식 ViTPose 저장소를 clone하고, 실행 가능한 환경(별도 Conda 권장)을 확정한다.
2. ViTPose++-S 체크포인트를 확보해 `tools/model_split.py`로 `ap10k.pth` 생성. 원본 스크립트의 불필요한 Expert Key 제거 동작을 수정/점검한다.
3. 일반 ViTPose-S `TopDown + ViT + 17-Keypoint Heatmap Head`를 구성하고 `load_state_dict` 결과를 확인한다. 예상하지 못한 Missing/Unexpected Keys가 있으면 실패로 처리한다.
4. 검증용 샘플 5–20장에서 원본 ViTPose++의 **AP-10K Expert 경로**와 분리된 일반 ViTPose의 Heatmap/좌표를 대조한다. `eval()` 및 동일 후처리 기준에서 수치 오차가 충분히 작은지 검사한다.
5. 성공한 모델과 원본 checkpoint SHA256/모델 설정을 기록하고 **모든 실험의 출발 체크포인트**로 고정한다.

**(B) NuWa OFP의 MLP 구조적 압축**

1. `backbone.blocks[l].mlp.fc1/act/fc2`에 Hook을 설치해 Post-activation을 수집한다(정확한 Hook 위치를 NuWa와 일치시킨다).
2. NuWa의 실제 코드가 사용하는 Activation Ranking을 **코드 수준에서 정확하게 재현**한다. 특히 코드의 `abs(sum(batch activation))`와 일반 `mean(abs(activation))`은 다르므로 후자를 원본으로 잘못 표기하지 않는다.
3. 고정된 압축 예산에서 층별 보존 뉴런 수 `e'_l`을 결정한다. 동일 실험군에서 `e'_l`은 고정.
4. 보존 인덱스 `J_l`에 대해 `fc1.weight[J_l, :]`, `fc1.bias[J_l]` 및 `fc2.weight[:, J_l]`로 **실제 작은 Linear 모듈을 새로 구성**한다. 단순히 0을 곱하는 가짜 Pruning은 금지.
5. NuWa의 OFP Least-squares 출력 보정 이식. 선택 뉴런 출력 `H_J`로 원래 MLP 출력 `Y`를 근사: `min_B ||H_J B - Y||²`; Bias를 명시적으로 처리한다.
6. 9.5 GiB VRAM 대응: Calibration Feature를 한꺼번에 GPU에 쌓지 말고 Batch별 `H_J^T H_J` 및 `H_J^T Y`를 누적해 CPU FP32/필요 시 FP64에서 Ridge 정규화 선형계 풀이를 검토한다.
7. 원본과 Pruning 후의 차원·Weight Shape·Parameter Count·MACs가 실제 감소했는지 검증한다.
8. 먼저 MLP-only 압축(예: Backbone MACs 목표 감소율이 아니라 **MLP 내부 10/20/30% 제거**)로 Smoke Test를 하고, 이후 Full-model 목표 MACs 예산으로 맞춘다.

**(C) NuWa SKP 및 MHA OFP 확장 — 중요하지만 별도의 Gate**

1. **SKP:** 원논문의 분류 중심 목적·Mask 학습을 Pose Loss/종별 특화 데이터에 맞게 변경한다. 기존 ViTPose Backbone·Head Weight를 Freeze하고 학습되는 Mask 범위를 명시한다.
2. **MHA OFP:** ViTPose Attention의 `qkv`·`proj` 형태에 맞춰 NuWa의 QK/VO SVD 압축을 이식한다. Head별 Q/K/V 차원, `scale`, `reshape`, 출력 `proj` 입력 차원과 Bias의 일관성을 테스트한다.
3. **Pose Head 유지:** 가장 먼저 Backbone의 MLP/MHA만 압축하고 동일한 Decoder를 유지한다. 필요 시 Decoder 압축은 독립 실험으로 분리.
4. NuWa 전체(SKP+OFP) 구현의 범위와 원논문 대비 변경 사항을 명시한다. **SKP 미완료 상태에서 MLP-only 결과를 Full NuWa라 주장하지 않는다.**

### 코드/API 산출물

- `src/models/vitpose_loader.py`, `src/models/pruned_mlp.py`, `src/models/pruned_attention.py`.
- `src/pruning/nuwa_base.py`, `src/pruning/activation_hooks.py`, `src/pruning/least_squares.py`.
- `src/pruning/skp_pose.py` (Phase 3b), `src/pruning/mha_ofp.py` (Phase 3b).
- `tests/test_split_equivalence.py`, `tests/test_pruned_shapes.py`, `tests/test_mlp_ls.py`.
- `checkpoints/base_vitpose_ap10k.pth`, `checkpoints/anchor/<species_set>.pth`, `results/nuwa_base.csv`.

### 반드시 제공할 공통 API

```python
# 설계 인터페이스 예시(현존하는 공식 NuWa 함수명이 아님)
anchor = prepare_anchor(base_model, species_loader, cfg)
importance = score_neurons(anchor, calibration_loader, cfg)  # [layer] -> [neuron]
pruned = prune_mlp_with_scores(anchor, importance, layer_widths, calibration_loader, cfg)
result = evaluate_pose(pruned, split="val", species_set=cfg.species_set, config=cfg)
```

서준교·백지승가 수정할 지점은 **`score_neurons(...)` 하나**가 되도록 의존성을 최소화한다.

### 완료 기준 및 인수인계

- Base → (SKP Anchor) → OFP 모델이 지정 종 집합에서 Pose Heatmap을 정상 출력.
- MLP 차원·MACs가 목표대로 감소하고 모듈이 다시 저장/로드 가능.
- `score_neurons` 교체만으로 두 개선안이 실행 가능하며, 동일 `e'_l`과 동일 Least-squares 적용.
- 체크포인트 생성 시간의 각 부분(`T_SKP`, `T_MHA`, `T_MLP`, `T_LS`) 분리 기록.

---

## 3.3 서준교 — 개선안 A: Post-activation Mean / Variance Ranking Lead

### 핵심 가설

> 원래 NuWa의 MLP Activation 선택 점수보다 평균 외 분산·종별 통계를 함께 활용하면, **추가 학습 없이** 같은 압축률에서 AP 손실을 줄일 수 있다.

### 구현해야 할 통계량

각 MLP의 GELU 이후 뉴런 `j` 출력 `h_{i,t,j}`를 이미지 `i`, Token `t`에 대해 계산한다. Calibration 이미지 수 `N`, Token 수 `T`, 대상 종 집합 `S`로 두자.

1. **NuWa-code baseline score** — 박재영과 정확한 원본 코드 동작을 공유하여 우선 재현.
2. **전체 평균** `mu_j = (1/NT) Σ_(i,t) h_(i,t,j)`; 부호를 가진 평균과 `mean(|h|)`를 혼동하지 않고 각각 별도 비교 가능.
3. **전체 분산** `V_total,j = (1/NT) Σ_(i,t) (h_(i,t,j)-mu_j)^2`.
4. **종별 평균** `mu_(j,c) = E[h_j | species=c]`.
5. **종간 분산** `V_between,j = (1/|S|) Σ_(c∈S) (mu_(j,c)-mean_c mu_(j,c))^2`.
6. **선택 확장:** 종내 분산 `V_within,j`; Fisher-style `V_between/(V_within+eps)`.

같은 층에서 서로 단위가 다른 통계량의 정규화 방식을 명시한 뒤, 예를 들어

`I_j(lambda) = (1-lambda) * norm(mu_j) + lambda * norm(V_between,j)`

를 사용한다. **큰 분산이 Pose에 중요하다는 보장은 없으며** 이를 검증할 가설로 취급한다. 부호 있는 평균이 0에 가까워도 중요한 뉴런이 존재할 수 있으므로 `abs/positive/rank-normalization` 선택도 명시한다.

### 구현 및 Ablation

1. Hook 출력 텐서 Shape `[B,T,D_ff]` 확인.
2. 메모리 효율적인 누적 합 `sum`, `sum_sq`, 종별 `sum_c`, `count_c`를 계산한다. 전체 Activation 저장 금지.
3. 종별 샘플 수가 다르면 **sample-weighted mean**과 **species-balanced mean**을 구분한다. 전자가 원본 분포에 대응하고 후자는 소수 종에 동일 가중치를 부여한다.
4. 최소 실험: 원본 점수, Mean, Mean+Total Variance, Mean+Between-species Variance. 확장은 Fisher-style.
5. `lambda ∈ {0, 0.25, 0.5, 0.75, 1}`을 **개발 Validation**에서 탐색하고 최종 평가용 Test에는 재사용하지 않는다.
6. 같은 Anchor, 같은 `e'_l`, 같은 MHA·LS 설정에서 Ranking만 변경한다.
7. 뉴런 Rank Correlation(Spearman), 원본과의 Top-k Overlap, 층별 분산 분포를 시각화하여 **왜 다른 뉴런을 고르는지** 설명한다.
8. 통계 집계에 드는 부가 시간/메모리 `T_stats - T_baseline_activation`을 측정한다.

### 코드/API 산출물

- `src/importance/statistical.py`: `score_neurons_statistical(model, loader, cfg)`.
- `tests/test_online_statistics.py`: 작은 텐서에서 NumPy 직접 계산 결과와 대조, imbalance test 포함.
- `results/stat_ablation.csv`, `figures/mean_variance_ranking.png`, `reports/stat_method.md`.

### 완료 기준

- 다른 팀원의 Pruning 본체를 수정하지 않고 Importance Score만 교체 가능.
- 같은 입력에 대한 온라인 통계와 일괄 계산 통계가 수치 오차 이내로 일치.
- 모든 결과에 *추가 학습 없음*을 명확히 기록하고, 성공/실패 여부 모두 결과에 포함.

---

## 3.4 백지승 — 개선안 B: Pose-/Species-aware Mask Learning Lead

### 핵심 가설

> 기존 ViTPose Weight를 고정하고 **MLP 중간 뉴런에만 가벼운 Gate**를 학습하면, 추가 비용을 허용한 만큼 평균 Activation 기반 선택보다 Pose AP를 잘 보존할 수 있다.

### 최소 구현

ViTPose-S는 12 Blocks, Block당 FFN 1536이므로 대상 종 수 `K`일 때 뉴런 단위 Mask 수는 `12 × 1536 × K`개(SKP 후에는 층마다 달라질 수 있음). **이 수치는 MLP Weight-wise Mask보다 훨씬 적지만, 역전파 비용은 여전히 존재**한다.

- 각 층 `l`과 대상 종 `c`에 대한 Learnable Logit `M[l,c,j]` 준비.
- 학습 시 입력의 정답 종 `c`에 대한 Gate `g[l,c,j] = sigmoid(M[l,c,j])`를 MLP GELU 출력에 곱한다.
- **모델 본체(Backbone/Decoder)는 Freeze**하고 Mask 파라미터만 Optimize. 각 중간층 Gate의 Gradient 계산을 위해 Backbone의 연산 그래프는 유지해야 한다.
- Pose Loss는 GT Heatmap과 Predicted Heatmap의 가시성 가중 MSE를 기반으로 구현한다. 정확한 정규화는 ViTPose의 `JointsMSELoss`와 일치시킨다.
- Sparsity/Budget Regularization을 추가할 수 있으나, 너무 강한 Penalty로 모든 Gate가 0 또는 1로 포화되지 않는지 모니터링한다.
- 결과는 `mean_c(sigmoid(M[l,c,j]))`, `max_c` 또는 보조 Hybrid Aggregation으로 하나의 Ranking 점수를 만든다.
- **추론 시 종 레이블에 따라 Gate를 바꾸지 않는다.** 학습된 Score만 사용해 상위 `e'_l`를 남기고 하나의 고정형 Pruned MLP를 만든다.

### 권장 손실 및 안정성

`L = L_pose(heatmap_pred, target, target_weight) + beta * L_sparse(gates)`

- 초기 Gate를 1에 가깝게 설정하여 처음부터 Pretrained 모델의 Heatmap을 크게 훼손하지 않도록 한다(예: `sigmoid` logit 양수 초기화).
- Mask Learning 중 **GT Pose Keypoint는 Loss에만** 사용; 모델의 최종 Inference에서는 종 레이블·GT Keypoint가 모두 없다.
- `mask_train`과 `calibration`을 분리하거나 고정된 protocol로 공유하고, 모델 선택은 개발 Validation만 이용한다.
- 최종 AP는 **Mask 적용 상태의 Soft Model이 아니라, 뉴런을 물리적으로 제거하고 LS 보정까지 끝낸 Hard-Pruned Model**에서 측정한다.

### 실험 계획

1. 한 층 Gate만 학습하는 Unit Test: Gradient가 Gate에만 생기는지 확인.
2. 12층 전체 Gate 적용 후 과도한 Memory 사용 여부 확인. AMP, Microbatch 1–2, Gradient Accumulation 우선.
3. `steps ∈ {100, 300, 1000}` 수준의 작은 탐색 → `L_pose` 추이·Gate 분포·추론 결과 확인. 수렴은 실제 로그로 판단하며 특정 Step에서 보장되지 않는다.
4. `beta`에 따른 희소성·정확도 민감도 조사. 최종 구조의 보존 뉴런 수는 동일 `e'_l`로 맞춰 **하이퍼파라미터가 실질적 모델 크기를 바꾸지 않게 한다**.
5. 적절한 `mean` 및 `max` 집계 실험(선택). 모델 선정 시 과도한 탐색으로 Test를 오염시키지 않는다.
6. 학습 시간 `T_mask`, Peak VRAM, GPU 동기화된 Step Time, Gate 파라미터 수 기록.
7. 기존 통계 Ranking과 Top-k Overlap, AP 개선량, 생성시간 증가량을 비교한다.

### 코드/API 산출물

- `src/importance/mask_gates.py`: `NeuronGate`, `attach_gates` / `remove_gates`.
- `src/importance/train_mask.py`: `train_importance_mask(anchor, loader, cfg) -> masks`.
- `src/importance/mask_rank.py`: `score_neurons_learned(masks, cfg) -> scores`.
- `tests/test_mask_gradients.py`, `tests/test_hard_prune_from_mask.py`.
- `results/mask_ablation.csv`, `figures/mask_training_curve.png`, `reports/mask_method.md`.

### 완료 기준

- 기존 ViTPose Weight Gradient는 없음; Gate Gradient는 유한하고 학습이 진행됨.
- 동일 Pruning API의 `score_neurons` 자리에 출력값 삽입 가능.
- Hard Pruning 모델의 구조·AP·실제 메모리·총 생성시간이 측정됨.
- 원래 X-Pruner의 Weight-wise Mask를 정확히 재현한 것이 아니라 **X-Pruner-inspired Neuron-wise Gate**라고 명시한다.

---

## 4. 팀 전체의 공통 실험 규약 (가장 중요)

### 4.1 비교 시 반드시 고정할 항목

| 항목 | 공통 규칙 |
|---|---|
| Backbone 기준점 | 동일 모델·체크포인트·입력해상도 256×192 |
| 종 집합 | 같은 `species_set_id`, 같은 train/val/test Split |
| Anchor | **동일 SKP 결과를 재사용**하여 제안 방법마다 Anchor를 재학습하지 않음 |
| MHA | MLP Ranking 비교 시 동일한 MHA OFP 또는 동일한 MHA 미적용 상태 |
| Layer Budget | 층별 보존 뉴런 수 `e'_l`을 모든 방법에서 동일하게 고정 |
| Pose Decoder | 동일 가중치·구조 유지. Decoder Fine-tuning 금지(별도 실험 제외) |
| Calibration | 샘플 ID·샘플 수·Augmentation/전처리·Batch 구성 고정 |
| LS 보정 | 같은 Ridge 계수·수치정밀도·목표 출력·Calibration Sample 사용 |
| Post-pruning retrain | **기본 실험은 없음**. 허용하면 전 방법에 같은 예산을 적용한 별도 표 |
| 평가 | 동일 AP-10K OKS, Bounding Box, Heatmap Decode, Flip Test, Precision |
| 랜덤성 | Seed 기록. GPU 시간이 허용되면 복수 Seed/종 집합 반복 |

**기준 결과를 만들 때 `NuWa`의 보존 뉴런 수가 층별로 다르면 그 층별 구조를 그대로 공유한다.** 단순히 파라미터 감소율만 비슷하게 만드는 것보다 **동일한 정확한 구조 크기**가 Ranking 차이를 분리하기 쉽다.

### 4.2 권장 압축 예산

- **P0 Smoke Test:** 특정 층 10% MLP 중간 뉴런 제거 → 전 모델 12층에 확장.
- **주 실험:** 사전에 정의한 **전체 Backbone MACs 감소율** 또는 **전체 모델 MACs 감소율** 목표(예: 10%, 20%, 30%)에서 서로 비교. 실제 달성량을 독립 측정.
- **보조 실험:** FFN 단독 뉴런 제거율(예: hidden 1536→1382/1229 등)은 Debug용; 이를 전체 ViTPose 압축률로 부르지 않는다.
- **압축률 표기:** `Reduction = 1 - C_pruned/C_base`. MACs/FLOPs 기준과 Params 기준을 따로 기록한다. 압축 계수 `C_base/C_pruned`와 혼동하지 않는다.

### 4.3 평가 지표 공식 및 비용 분해

**정확도:** AP-10K 공식 OKS 기반 Keypoint `AP@[.50:.95]`(가능한 원본 평가 코드 사용); 종별 `AP_c`; `Macro-AP = mean_c AP_c`(별도). 결과 보고 시 샘플 수를 함께 표시한다.

**연산량 및 크기:**

- `Parameter reduction = 1 - P_pruned/P_base` (실제 dense weights).
- `MAC reduction = 1 - MAC_pruned/MAC_base` (동일 입력 한 Crop 기준, Pose Head 포함 여부 명시).
- `Latency`: 동일 GPU MIG 인스턴스, batch 1 기준 `ms/image`, Warmup 후 다회 반복; 별도 batch 4 throughput(선택).
- `Peak VRAM`: `torch.cuda.max_memory_allocated()` 및 해당 단계의 메모리 로그.

**모델 파생 비용:**

- `T_total = T_checkpoint_load + T_SKP + T_calibration + T_MHA + T_MLP_rank + T_LS + T_mask_extra`.
- SKP Anchor를 공통으로 재사용한 비교에서는 `T_SKP`를 중복 계상하지 않는다.
- **Cold-start 모델 생성 시간**(새 종 집합을 처음 처리)과 **Warm-start 추가 압축 시간**(Anchor/점수 캐시 재활용)을 구분한다.
- Learned Mask의 비용은 `T_mask_extra`로 별도 표기하고, 정확도 개선 `ΔAP` 대비 시간 증가율을 함께 분석한다.

### 4.4 데이터 누수 금지

- 공식 Test Annotation은 **모델 선택, Mask 학습, Hyperparameter 튜닝, LS Calibration에 사용하지 않는다**.
- 종 선택, `lambda`, `beta`, Training Step은 Train/개발 Validation 기준으로 확정한 뒤 최종 Test 1회 평가.
- 원본·개선 모델은 **동일한 종 Subset**에서 평가한다. 원본 AP는 54종 전체에서, Pruned AP는 일부 종에서 평가하면 안 된다.
- 종별 Calibration Sample Count가 작을 수 있으므로 해당 종의 학습/검증 샘플 수 및 Keypoint 가시성을 표에 명시한다.

### 4.5 성능 기록 파일 예시

```csv
run_id,seed,model_source,species_set,method,stage,mlp_budget,mha_budget,ap,macro_ap,worst_species_ap,params,macs,latency_ms,peak_vram_mib,t_skp_sec,t_rank_sec,t_mask_sec,t_ls_sec,t_total_sec,commit
```

| 모델 ID | AP↑ | Macro AP↑ | Params↓ | MACs↓ | Latency↓ | 생성시간↓ |
|---|---:|---:|---:|---:|---:|---:|
| Unpruned ViTPose | TBD | TBD | TBD | TBD | TBD | — |
| NuWa-adapted / MLP OFP | TBD | TBD | TBD | TBD | TBD | TBD |
| A: Mean + Variance | TBD | TBD | TBD | TBD | TBD | TBD |
| B: Learned Mask | TBD | TBD | TBD | TBD | TBD | TBD |

모델 종류에 따라 구조가 완전히 같으면 Params/MACs가 동일한 것이 정상이며, 이는 **선택된 뉴런의 차이가 AP에 미친 영향**을 비교하는 장점이다.

---

## 5. 공유 저장소·협업 규칙

### 5.1 권장 디렉터리

```text
NuWa-ViTPose/
├── README.md
├── ENVIRONMENT.md
├── requirements-lock.txt
├── configs/
│   ├── base.yaml
│   ├── species_sets.json
│   ├── nuwa_baseline.yaml
│   ├── statistical.yaml
│   └── learned_mask.yaml
├── src/
│   ├── data/
│   │   └── ap10k_species.py
│   ├── models/
│   │   ├── vitpose_loader.py
│   │   ├── pruned_mlp.py
│   │   └── pruned_attention.py
│   ├── pruning/
│   │   ├── nuwa_base.py
│   │   ├── activation_hooks.py
│   │   ├── least_squares.py
│   │   ├── skp_pose.py
│   │   └── mha_ofp.py
│   ├── importance/
│   │   ├── statistical.py
│   │   ├── mask_gates.py
│   │   ├── train_mask.py
│   │   └── mask_rank.py
│   └── eval/
│       ├── metrics.py
│       ├── evaluate_pose.py
│       └── profile_model.py
├── scripts/
│   ├── prepare_model.py
│   ├── evaluate_baseline.py
│   ├── run_nuwa.py
│   ├── run_statistical.py
│   ├── run_learned_mask.py
│   └── aggregate_results.py
├── tests/
├── checkpoints/      # Git에 대용량 모델 파일을 직접 commit하지 않음
├── data/             # Git에 데이터셋 원본 commit 금지
├── results/
├── figures/
└── reports/
```

### 5.2 브랜치·통합·인수인계

| Branch | 담당 | Merge 조건 |
|---|---|---|
| `feature/evaluation` | 최준혁 | 같은 모델의 AP 재현, Test 통과 |
| `feature/nuwa-baseline` | 박재영 | 최소 MLP OFP 테스트, Shape·평가 성공 |
| `feature/importance-stat` | 서준교 | 공통 `score_neurons` API 준수, 통계 단위 테스트 |
| `feature/importance-mask` | 백지승 | Gate Gradient 테스트, Hard Pruning 성공 |
| `main` | 전원 | 실행 결과·설정·파일 경로가 재현 가능한 상태만 Merge |

- **큰 모델 파일은 공용 파일시스템 또는 별도 아티팩트 저장소에 보관**하고 Git에는 config·hash·경로 규약만 기록.
- 실험 결과물은 `method/species_set/budget/seed` 기준으로 디렉터리 분리. 동일 파일명 덮어쓰기 금지.
- 환경 변경은 한 명이 조율하고 `requirements-lock.txt` 및 변경 내역을 업데이트.
- 최소 매일/작업 종료 시 공통 상태표에 `(현재 통과한 Gate, 막힌 문제, 다음 인수인계)`를 남긴다.
- GPU 할당은 한 번에 한 사람/한 작업 중심으로 운영. CPU만 필요한 코드·보고서·데이터 분석을 병렬로 진행.

### 5.3 통합 테스트 체크리스트

- [ ] 같은 종 집합의 Train/Validation/Test 이미지 ID에 중복 없음.
- [ ] ViTPose++ 원본과 Split ViTPose의 AP-10K Heatmap이 같은 입력에서 일치.
- [ ] Pruning 전후 MLP Shape, Parameter Count, MACs가 예상대로 변함.
- [ ] Mask 방식의 Soft-gate 결과와 Hard-pruned 결과를 구분하여 기록.
- [ ] 최종 모델은 종 ID 없이 `pose_model(image)`만으로 동작.
- [ ] 두 개선안이 NuWa Baseline과 동일 Anchor 및 `e'_l` 사용.
- [ ] Validation 전처리, BBox 조건, AP-10K OKS sigmas가 방법 간 동일.
- [ ] Baseline/제안 모델의 수치 비교에서 CPU/GPU, FP16/FP32 차이가 통제됨.
- [ ] 실험 표의 모든 행에 모델 소스·Git commit·Config·Seed 정보 존재.

---

## 6. 실행 우선순위와 실패 대비 계획

### 6.1 최소 성공 범위 (Must-have)

1. **공식 ViTPose의 AP-10K 추론·평가 재현** 및 대상 종 집합 고정.
2. **MLP-only NuWa OFP 이식**: 원본 Activation Ranking + Structured Pruning + Least-squares.
3. 두 개선안의 Ranking을 **동일 MLP 구조·동일 예산**에서 비교.
4. Keypoint AP–MACs–Latency–모델 생성시간 결과표와 적어도 하나의 압축률 곡선.

이 단계까지 진행되면 **`NuWa-inspired ViTPose Compression and Ablation`**으로 충분한 결과를 제시할 수 있다. 단, Full NuWa 재현이라는 표현은 사용하지 않는다.

### 6.2 확장 범위 (Should/Could-have)

- **Should-have:** Pose-aware SKP 이식, 동일 Anchor 기반 전체 실험. 성공하면 NuWa Class-specific Adaptation의 의미가 강화됨.
- **Could-have:** MHA QK/VO SVD 압축, 다른 압축 예산, 동일/이질 종 비교, ViTPose-B 복수 모델 검증, 복수 Seed.
- **마지막 수단:** Full NuWa가 어려우면 MLP-only Pruning의 재현성·평가 품질·두 개선안 비교를 완성하는 데 집중.

### 6.3 단계별 중단/전환 기준

| 막힌 문제 | 우선 조치 | 전환 기준 |
|---|---|---|
| MMPose 구형 패키지 설치 실패 | 호환 환경 분리, 공식 체크포인트 로드 경로 확인 | Baseline Pose 추론 불가하면 개선 실험 진행 중단 |
| Split 체크포인트 불일치 | MLP Expert 병합·Decoder State Key 점검 | Heatmap 동등성 미확보 시 모델 소스 재선정 |
| 9.5 GiB OOM | batch=1, AMP, CPU 통계 누적, LS 별도 CPU, 불필요한 Autograd 해제 | 그래도 실패하면 MLP-only / 적은 Calibration Sample |
| SKP 불안정 | Loss·Mask 초기화·Freeze·Grad 확인 | OFP-only 논문 범위로 축소 및 명시 |
| Learned Mask 비용 과다 | 학습 Step 축소·한 층만 Gate 디버깅·Gradient Checkpointing | Statistical과 계산 비용 비교 중심으로 분석 |
| AP 향상 없음 | 동일 Anchor/예산 검증, Rank Overlap 및 종별 오류 분석 | 부정적인 결과를 선택적으로 누락하지 않고 보고 |

---

## 7. 최종 보고서 및 발표 구성

1. **Problem & Motivation:** 범용 동물 Pose Estimation 모델을 특정 종 집합에 특화하는 이유, Edge 장치의 제약.
2. **Related Work:** NuWa(SKP/OFP), ViTPose/ViTPose++, AP-10K, 관련 Structured Pruning 및 중요도 추정.
3. **Problem Definition:** 종 집합 `S`, 17개 Keypoint, 고정 모델 파생, 압축 예산, GT BBox.
4. **Implementation:** ViTPose++ Split 경로/검증, NuWa를 Pose Loss와 ViTPose 구조에 맞춘 수정, MLP·MHA·SKP 구현 범위.
5. **Proposed Methods:** Statistical Ranking과 Learned Mask의 수식, 공통 LS/Hard Pruning.
6. **Experimental Setup:** GPU MIG 용량, Dataset Split, Calibration 수, 환경·Seed, 비용 측정법.
7. **Main Results:** Unpruned / NuWa / Proposed A / Proposed B의 AP, MACs, Latency, Derivation Time.
8. **Ablation:** Mean vs Variance 종류, `lambda`; Learned Mask Step·`beta`; 종 집합·압축률 변화.
9. **Interpretation:** 정확도–효율 Pareto, 종별 성능 편차, Rank Overlap, 실패 사례 시각화.
10. **Limitations & Conclusion:** 구형 종속성, 모델 크기와 GPU 한계, NuWa 전체 구현 여부, 향후 확장.

### 발표용 필수 그림/표 (권장)

- AP-10K의 대상 종 이미지와 17 Keypoint 시각화.
- `ViTPose++ → ViTPose → NuWa → A/B` 실행 흐름도.
- **AP vs MAC Reduction** 곡선 (모든 방법 동일 축).
- **AP vs Model Derivation Time** 산점도 (빠른 NuWa 장점 검증).
- 대상 종별 AP 변화 막대그래프 또는 Heatmap.
- Mean/Variance/Mask 기반 뉴런 중요도 분포 및 Top-k 중복률.
- 모델의 Pruning 전후 실제 FC1/FC2 Shape/Parameter Count 표.

---

## 8. 별도 보완 의견 (Assistant Recommendations)

이 섹션은 **팀이 제시한 기본 계획의 변경 제안**이며, 이미 확정된 요구사항으로 간주하지 않는다.

### 8.1 'AP-10K Pre-training'보다 정확한 표현

가능하면 **"AP-10K를 포함한 다중 데이터셋에서 학습한 ViTPose++로부터 AP-10K 전용 ViTPose를 추출"**이라고 쓰는 편이 정확하다. ViTPose++ 원본은 여러 데이터셋의 MoE 학습 결과이며 단순히 AP-10K만으로 Pretraining된 모델이라고 단정할 수 없다. 출처와 Split 변환 과정을 기록한다.

### 8.2 ViTPose++ Split은 NuWa Pruning이 아니다

Task Expert 제거/병합 및 체크포인트 변환은 **별도의 모델 구조 정리 작업**이다. Split 전 ViTPose++와 Split 후 일반 ViTPose의 크기 차이를 NuWa의 압축 성과로 합산하지 않는다. **공통 Unpruned ViTPose부터** Params/MACs/AP를 비교한다.

### 8.3 NuWa의 Pose Adaptation과 Full Replication 구분

NuWa는 원래 **클래스 분류**에 맞게 설계됐다. 본 프로젝트에서는 대상 종 집합이 Class 역할을 하지만, 출력은 **17관절 위치**이고 Loss는 Heatmap Regression이다. SKP를 수정하면 이미 방법론이 변형되므로 **"NuWa-adapted"**라고 기술하며 원논문의 성능과 직접 동등 비교하지 않는다.

### 8.4 역할 분담에 공통 환경 담당과 종 선택 책임이 빠져 있음

현재 분담은 평가/압축/개선으로 구성되어 있지만 **환경 재현성, 공식 체크포인트 다운로드, AP-10K 종별 샘플 분석**이 선행되지 않으면 모두 멈춘다. 따라서 **박재영(환경+모델 기준점)**, **최준혁(종 선택+데이터 Split+평가)**을 공식 초기 담당으로 추가하는 것을 권장한다.

### 8.5 Learned Mask의 현실적인 난이도

Neuron-wise Mask 수는 작지만 **Gradient가 12개 ViT Block을 지나므로 역전파 비용이 사라지지 않는다**. 기존 NuWa의 속도를 개선했다고 단정할 수 없으므로 추가 학습 시간을 반드시 측정한다. SKP Mask 재활용을 세 번째 저비용 Baseline으로 비교하면 연구가 더 강해질 수 있다.

### 8.6 Mean + Variance의 정의를 사전에 고정

- `Var(h)`(샘플·Token 전체 분산), `Var_c(E[h|c])`(종간 평균 분산), `E_c[Var(h|c)]`(종내 분산)는 서로 다른 통계량이다.
- 큰 분산은 Pose 구분 능력이 아니라 배경·종 외형·조명 변화도 반영할 수 있다.
- 원본 NuWa **코드의 점수 계산**과 일반 수학적 Mean이 완전히 같은지 점검하고, 원본 코드 기준 Baseline을 별도 명시한다.
- Mean + Variance의 결합은 알려진 일반적 휴리스틱에 가까울 수 있으므로 **"신규 수식"**이라고 강조하기보다 실험적 검증을 기여로 제시한다.

### 8.7 종 집합과 평가 공정성

종 유사성 가설은 매력적이지만 AP-10K의 종별 샘플 수가 불균형할 수 있다. 적어도 **종 수·평가 샘플 수·가시성**을 기록하고 동일 집합에서 Unpruned와 Pruned 모델을 비교한다. 종간 비교 시 Dataset 구성 차이를 명시한다.

### 8.8 일정 위험을 반영한 최우선 순서

**첫 번째 목표는 모델 평가 코드와 MLP-only 압축이 돌아가는 상태를 만드는 것**이다. 이 단계가 끝나지 않은 상태에서 SKP/MHA 전체 이식과 다양한 Ablation을 동시에 확장하지 않는다. Full NuWa는 후속 Gate로 두되, 교수/조교에게 보고할 계획서에는 Must-have와 Stretch Goal을 명확히 구분한다.

### 8.9 권장 추가 대조군

같은 층별 뉴런 수에서 **Random Ranking**, **Weight Magnitude Ranking**, **Original Activation Ranking**을 넣으면 Mean/Variance/Mask 개선안의 가치를 해석하기 쉬워진다. Random은 Seed 여러 개가 필요하다. 재학습 여부·LS 보정 조건은 모두 동일하게 한다.

### 8.10 최종 논문의 주장 수준

실험 결과가 나오기 전에는 "AP 증가", "모델 생성 시간 단축", "더 좋은 NuWa"를 사실처럼 쓰지 않는다. 정확도가 더 높지만 시간도 증가하면 정확도–시간 절충으로 보고, 성능이 낮아도 **왜 그런지(종별 성능, 활성화 분포, 선택 뉴런 겹침)** 분석해 연구 결과로 제시한다.

---

## 9. 최우선 할 일 (업무 시작 순서)

| 순서 | 담당 | 즉시 착수할 작업 | 구체적 산출물 |
|---|---|---|---|
| 1 | 박재영 | ViTPose 공식 코드 환경 구성, ViTPose++-S Checkpoint 확보, `model_split.py` 점검 | 실행 환경/`ap10k.pth`/모델 추론 로그 |
| 2 | 최준혁 | AP-10K Annotation 분석, 후보 종별 수량표, OKS AP 평가 함수 | `species_sets.json`, `dataset_audit.md`, baseline AP |
| 3 | 서준교 | NuWa 코드의 정확한 Activation Score 파악, Mean/Variance 온라인 통계 Prototype | `statistical.py`, 단위 테스트 |
| 4 | 백지승 | MLP Neuron Gate Module 작성, Frozen Model Gradient/Mask 학습 Smoke Test | `mask_gates.py`, `train_mask.py`, Gradient 테스트 |
| 5 | 전원 | AP 평가 + NuWa MLP OFP + Ranking A/B 하나의 인터페이스 통합 | 최초 공유 결과 CSV |
| 6 | 전원 | 공정한 동일 예산 실험, 반복 측정, 그림·보고서 작성 | 최종 결과표·발표자료 |

**완료 우선순위:** *동작하는 Baseline → 재현 가능한 MLP-only NuWa OFP → 개선안 A/B 비교 → SKP/MHA 확장.*

---

## 10. 코드 및 문헌 확인 출처

> 아래는 **공식 코드에서 확인된 사실**과 본 문서의 **제안 실험 설계**를 구분하기 위한 참고 링크다.

1. [NuWa 공식 코드 — CGCL-codes/NuWa](https://github.com/CGCL-codes/NuWa): SKP/OFP 및 ViT Pruning 구현 참고. 특히 `method/pruning.py`, `method/get_anchor_model.py` 및 calibration 데이터 수집 구현.
2. [ViTPose/ViTPose++ 공식 코드](https://github.com/ViTAE-Transformer/ViTPose): 일반 ViT 및 ViTMoE Backbone, AP-10K Config, 사전학습 모델 제공.
3. [ViTPose `tools/model_split.py`](https://github.com/ViTAE-Transformer/ViTPose/blob/main/tools/model_split.py): Multi-task Expert를 태스크별 일반 MLP 출력으로 병합하는 코드. 키 정리 버그 가능성 및 출력 동등성 테스트 필요.
4. [ViTPose-S AP-10K Config](https://github.com/ViTAE-Transformer/ViTPose/blob/main/configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/ap10k/ViTPose_small_ap10k_256x192.py): 256×192, 17 Keypoint, 일반 ViT Backbone 설정.
5. [AP-10K 공식 저장소](https://github.com/AlexTheBad/AP-10K): 54종, 17개 Keypoint, 데이터 구조 및 Annotation.
6. [APTv2 공식 저장소](https://github.com/ViTAE-Transformer/APTv2): ViTPose-B의 AP-10K **사전학습 초기화**를 사용한 APTv2 실험 체크포인트를 별도로 제공. AP-10K용 모델과 혼동하지 않도록 검증.

**검증 전제:** 공식 체크포인트의 실제 다운로드 가능성, 서버의 ViTPose 구형 종속성 설치 성공, AP-10K 종별 실제 샘플 수, NuWa 전체 이식 성공 여부는 **아직 실측/확인되지 않은 항목**이다. 본 문서의 추천 배치·Step·예산은 실험 제안값이며 보장된 설정이 아니다.
