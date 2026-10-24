# 실행 담당 AI 에이전트 인계서 — CoverMem A/B 및 세 번째 실험

2026-10-24. 이 문서는 두 저장소에 동일하게 제공하는 통합 인계서다. 과거 대화 전체를 읽지 않고 이 문서와 연결된 설정/설명서만으로 자료 준비, 설치, 실행, 복구, 결과 보고를 할 수 있도록 작성했다. 대상 서버는 A100 40GB ×16이며 **실험당 GPU 1장**을 사용한다.

## 1. 먼저 읽을 파일과 현재 상태

1. 현재 checkout 루트의 `AGENTS.md`와 이 문서.
2. [다운로드·분할 복원 안내](DOWNLOAD_AND_RESTORE_KO.md).
3. 각 저장소의 `docs/SERVER_RUNBOOK_KO.md`와 실행할 `configs/*.json`.
4. A/B의 [METHOD_AND_CONFIG_KO.md](https://github.com/jiho-98/covermem-selector-grpo/blob/main/METHOD_AND_CONFIG_KO.md), 세 번째의 [METHOD_AND_DECISIONS_KO.md](https://github.com/jiho-98/covermem-branch-grpo/blob/main/docs/METHOD_AND_DECISIONS_KO.md).
5. 각 저장소의 `validation/VALIDATION_KO.md`와 `environment.lock.json`.

코드·입력 검증과 CPU 테스트는 완료했지만 실제 4B 모델의 A100 GPU smoke·학습·평가 결과는 아직 없다. 로컬 최신 CPU 검증은 torch 2.12에서 A/B 25개, 세 번째 28개다. supervisor 2개도 통과했으나 실제 서버의 고정 torch 2.11 환경에서 다시 확인해야 한다. CPU 통과를 GPU 메모리·속도·성능 검증으로 보고하지 않는다.

사용자가 설치/검토만 요청한 경우 그 범위까지 진행한다. 실행을 요청한 경우 아래 기본 조건과 제공 큐를 사용한다. 이미 허용된 작업을 단계마다 다시 승인받을 필요는 없으며, GPU 점유권·계정 접근·서버별 Slurm 설정처럼 실제로 빠진 정보만 확인한다. 이 인계 문서 자체를 새로운 GPU 실행 지시로 해석하지 않는다.

## 2. 자료와 코드의 위치

| 용도 | 위치 |
|---|---|
| A/B 저장소 | [jiho-98/covermem-selector-grpo](https://github.com/jiho-98/covermem-selector-grpo) |
| 세 번째 저장소 | [jiho-98/covermem-branch-grpo](https://github.com/jiho-98/covermem-branch-grpo) |
| 어댑터·데이터 포함 full ZIP의 분할 파일 | [Google Drive 전달 폴더](https://drive.google.com/drive/folders/1C2iI8EUexJFjcIui8Xp5uAO-_QOxMdGF) |
| A/B 입력 검증 목록 | 해당 저장소 `INPUTS.sha256.json`: 19파일 |
| 세 번째 입력 검증 목록 | 해당 저장소 `INPUTS.sha256.json`: 13파일 |

권장 방식은 Git 최신 checkout에 Drive ZIP의 입력만 import하는 것이다. Drive ZIP은 기존 배포 스냅샷이며 이 에이전트 문서가 들어 있지 않을 수 있다. 복원 순서·파일 수·원본 SHA256·실제 명령은 [복원 안내](DOWNLOAD_AND_RESTORE_KO.md)에 있다. 다운로드 토큰이나 접근 자격증명을 로그/결과/Git에 기록하지 않는다.

## 3. 기본 실행 행렬과 비교 관계

전체 실험 실행을 의뢰받았을 때의 기본 행렬은 **주 실험 3개 + 별도 학습 기준선 3개 = 6개 실행**이다. A의 시작 매니저 평가는 A 큐에 포함되며 별도 학습 조건으로 세지 않는다. 사용자가 일부만 지정하면 그 조건과 필요한 비교군을 명확히 구분하여 보고한다. 선택 사항을 자동 추가하지 않는다.

| ID | 저장소 | config | 기본 출력 디렉토리 | 비교 |
|---|---|---|---|---|
| A | selector | `configs/A_loo.json` | `runs/allq_k10/A_loo` | 같은 A 큐의 `evaluation_initial_manager` |
| B | selector | `configs/B_joint.json` | `runs/allq_k10/B_joint` | B-control |
| B-control | selector | `configs/vanilla_control.json` | `runs/allq_k10/vanilla_control` | 선택기 없는 같은 엔진 LOO |
| Third | branch | `configs/branch_p025.json` | `runs/branch_v1/branch_p025` | Third-matched, Third-original |
| Third-matched | branch | `configs/vanilla_matched.json` | `runs/branch_v1/vanilla_matched` | 액션 제한·전환·정규화를 맞춘 LOO |
| Third-original | branch | `configs/vanilla_original.json` | `runs/branch_v1/vanilla_original` | 원래 동작 LOO, 추가 세션 std 없음 |

선택 사항은 `A_h1.json`, `branch_p050.json`, `branch_p100.json`이다. `A_h1`은 완료 H1을 고정한 선택기 실험이며 원래 H1 알고리즘 재학습 명령이 아니다. `legacy/`는 역사적 소스/설정 보존용이므로 옛 절대 경로의 런처를 실행하지 않는다. 두 저장소의 vanilla를 임의로 하나로 합치거나 과거 vLLM 결과로 대체하지 않는다.

## 4. 변경하면 안 되는 실험 정의

### 공통

- 베이스 모델은 `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`; 베이스는 동결한다.
- 학습은 대화당 독립 bank 4개, 예산 43,500 content characters다. 네 경로는 GPU 4개를 뜻하지 않는다.
- 대상은 `memory_id`다. UPDATE로 여러 원본 턴이 묶인 항목은 그대로 유지하고 REPLACE 시 해당 항목 전체를 제거한다.
- bank utility는 `U = 0.7 × evidence coverage + 0.3 × evidence recall`이다. 이 0.7/0.3은 현재/미래 질문 비율이 아니다.
- 후보 보상은 행동 직후 bank와 행동 전 bank의 utility 차이다. 미래 턴을 이어 생성하는 긴 rollout 분기가 아니다.
- 훈련 대화의 미래 생성 질문은 CPU 채점에만 사용할 수 있다. 공식 평가 질문을 훈련에 쓰거나 미래 질문·정답·근거 라벨을 모델 입력에 넣지 않는다.
- K는 정책 출력의 샘플 수다. 서로 다른 ID/행동 K개를 보장하지 않으며 중복 샘플을 강제로 제거하지 않는다. 실제 실행은 사전 지정한 index 0이며 reward argmax로 선택하지 않는다.
- 기본 seed 42, BF16, SDPA, gradient checkpointing, logits chunk 128, 학습률 5e-6, KL 0.02, PPO epoch 1을 제공 config대로 유지한다. A/B 기본 수집 배치는 한 세션, 세 번째도 세션당 최대 한 업데이트다.

### A/B: REPLACE 대상 선택기의 GRPO

매니저의 원래 executor를 통과한 **유효 REPLACE**에서만 선택기로 전환한다. 두 LoRA를 더하는 방식이 아니라 같은 동결 베이스에서 활성 어댑터를 전환한다. 선택기 입력은 전체 bank 항목 내용, 현재 턴, 실행 가능한 ID 목록이며 매니저의 기존 rationale/대상 ID나 채점 결과를 보여주지 않는다.

A는 완료 LOO 매니저 고정 + SFT 초기 선택기 학습이다. 훈련 대화를 빈 bank에서 다시 진행하고 ADD/SKIP/UPDATE/DELETE도 실행한다. 매니저를 고정해도 선택기로 바뀐 bank 때문에 이후 출력은 달라질 수 있다.

B는 매니저와 선택기를 SFT의 독립 복사본으로 시작한다. 수집 중 두 어댑터를 고정하고 배치 뒤 각각 업데이트한다. 매니저는 원래 LOO continuation gain/fixed scale, 선택기는 각 REPLACE의 K=10 local reward/group std를 사용한다. 네 경로에서 각각 10개가 나와도 40개 전체를 한 그룹으로 비교하지 않는다. 매니저 보상과 선택기 보상을 직접 혼합하지 않는다.

선택기가 관여한 B의 REPLACE에서는 매니저가 실제 생성한 토큰/old logprob를 보존하고 action 값 구간만 직접 loss를 받는다. rationale/target/suffix는 직접 loss에서 제외하고 기존 전체 응답 토큰 분모를 유지한다. 다른 액션은 기존 전체 응답 loss다. 선택기로 생긴 bank는 이후 매니저의 입력/보상에 영향을 주므로 완전한 학습 독립성을 주장하지 않는다.

후보 채점은 해당 훈련 대화의 생성 질문 전체 675/658/703/590/712개를 동일 가중치로 사용한다. 현재/미래 시간 비율이나 세 번째 실험의 중복 제거 규칙을 가져오지 않는다. 샘플0이 무효이면 선택기 reward −2, 실제 실행은 이미 유효한 원래 매니저 대상 fallback이다. 다른 후보 중 최선의 유효 출력을 선택하지 않는다. 모두 동점인 group은 선택기 업데이트에서 제외한다.

### 세 번째: 단일 매니저의 두 단계 GRPO

SFT 초기 매니저 LoRA 하나만 학습한다. 별도 동결 SFT reference는 KL 계산용이며 선택기가 아니다. 매 대화는 빈 bank/1단계로 시작하되 학습된 매니저 파라미터는 이어간다.

1단계는 원래 5액션, 학습 최대 3시도, 중간/끝 누적 QG의 action-aligned LOO continuation gain을 사용한다. 같은 원문 턴의 다른 세 경로 첫 시도를 baseline으로 하며 fixed scale은 `0.008103060640792133`이다. 경로별 endpoint advantage 하나를 모든 액션에 복사하는 방식이 아니다. 분기 실험과 matched에는 그 계수 뒤 추가 세션 std가 있으므로 원래 학습과 수치적으로 완전히 같다고 설명하지 않는다.

**같은 세션 종료 시점의 네 bank가 모두 95% 이상**이면 다음 세션부터 2단계다. 각 경로가 과거에 한 번씩 도달한 것으로는 전환하지 않는다. 전환 세션 도중에는 바뀌지 않고, 같은 대화에서 bank가 줄어도 1단계로 돌아가지 않는다.

2단계의 각 경로/턴에서 출력 전에 독립적으로 p=0.25 분기 추첨을 한다. 선택된 상태는 같은 입력에서 K=4 전체 응답을 생성·가상 실행·채점하고, 비분기 상태는 한 응답을 실행하되 PG/KL 양쪽 loss에서 제외한다. 허용 액션은 ADD/SKIP/REPLACE이며 UPDATE/DELETE는 무효다. 샘플0이 무효이면 −2/no-op/재시도 없음이다. A/B의 원래 대상 fallback을 적용하지 않는다.

질문풀은 같은 대화의 전체 QG 중 충분 근거 표현을 OR/AND 구조 그대로 중복 제거한 496/497/509/411/500개, 총 2,413개다. 1단계 누적 질문 3,338개를 이 수로 바꾸지 않는다. `A_raw = r_k − mean_K(r)`는 같은 상태의 K개 안에서만 계산한다.

세션 내 학습 출력별 advantage의 population std로 나누며 평균을 다시 빼지 않는다. std ≤1e-8이면 1로 나눈다. 비분기 출력은 정규화에서도 제외한다. 전체 응답 토큰에 loss를 적용하고 최대 한 번 업데이트한다. 학습 출력이 없거나 모든 advantage가 0인 세션은 KL-only 업데이트도 하지 않고 bank/cursor만 진행한다. 다른 유효 신호가 있는 세션 안의 동점 출력에는 KL 항이 남을 수 있다. 2단계에 기존 경로 LOO를 추가로 더하지 않는다.

matched는 같은 단계 전환·3액션·무효 no-op·세션 std를 유지하되 한 출력/턴과 원래 누적 질문 LOO를 사용한다. 분기 여부·학습 출력 선택·보상풀/credit이 함께 달라지므로 한 요소만의 인과 실험이라고 주장하지 않는다. original은 전 구간 5액션과 원래 재시도/fixed scale이며 추가 세션 std가 없다.

## 5. 서버 준비와 CPU 검증 순서

먼저 서버/작업 상태를 읽는다. 모델을 로드하기 전에 현재 GPU 종류·VRAM·드라이버, 디스크/RAM, Slurm 사용 여부, 사용자의 기존 할당 및 기존 CoverMem 출력/큐가 있는지 확인한다. 다른 사용자의 프로세스를 중단하지 않는다. 처음 받은 서버에서 GPU 0~15를 모두 사용 가능하다고 가정하지 않는다.

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .
free -h
command -v sbatch || true
# Slurm이 있는 서버에서만 실행
# squeue -u "$USER"
```

[복원 안내](DOWNLOAD_AND_RESTORE_KO.md)대로 입력을 가져온 뒤 각 저장소 루트에서 설치한다. 각 repo의 `.venv`는 별개다. 공용 Python/site-packages를 수정하지 않는다. Python은 3.10, torch 2.11.0, Transformers 5.8.0, PEFT 0.19.1, Accelerate 1.14.0이다. 전체 버전은 lock 파일을 따른다.

```bash
# 각 저장소 루트에서 한 번씩; GPU를 초기화하지 않는 준비 단계
bash scripts/setup_env.sh
bash scripts/download_model.sh
git rev-parse HEAD
git status --short
```

기본 torch wheel은 cu130이다. 시스템 nvcc 대신 드라이버와 실제 wheel 호환성을 확인한다. 다른 공식 torch 2.11 wheel이 서버에 필요하면 설치 전 `COVERMEM_TORCH_INDEX_URL`을 사용하고 실제 환경을 기록한다. 모델 revision/패키지 버전을 임의로 최신 버전으로 바꾸거나 runtime가 달라진 checkpoint를 재개하지 않는다. 설치/오프라인 모델 경로 상세는 각 [서버 안내서](SERVER_RUNBOOK_KO.md)를 따른다.

A/B 저장소 루트:

```bash
(
  set -e
  bash scripts/check_cpu.sh configs/A_loo.json
  for cm_case in A_loo B_joint vanilla_control; do
    bash scripts/preflight.sh "configs/${cm_case}.json"
    cp validation/preflight.json "validation/preflight_${cm_case}.json"
  done
)
```

세 번째 저장소 루트:

```bash
(
  set -e
  bash scripts/check_cpu.sh configs/branch_p025.json
  for cm_case in branch_p025 vanilla_matched vanilla_original; do
    bash scripts/preflight.sh "configs/${cm_case}.json"
    cp validation/preflight.json "validation/preflight_${cm_case}.json"
  done
)
```

**어느 하나라도 실패하면 GPU 단계로 넘어가지 않는다.** 각 대상의 preflight 파일이 새로 생성되었고 `prepared`와 fingerprint가 있는지 확인한다. 입력만 검사한 `assets_validated`는 모델/환경 full preflight 통과와 다르다. 5개 훈련 대화/142세션/원래 생성 질문 3,338개 및 입력 파일 수를 확인한다. smoke 전에 `validation/installed.freeze.txt`, preflight JSON, commit과 수정 내역을 보존한다. 복사한 옛 `validation/` 로그를 새 서버의 통과 기록으로 사용하지 않는다.

## 6. GPU 실행 — 기존 supervisor 사용

실험당 1GPU·1프로세스의 HF/PEFT 순차 생성 구현이다. DDP/FSDP/텐서 병렬/공유 prefix cache는 없다. K나 네 경로를 GPU 개수로 해석하지 않는다. `torchrun`으로 16프로세스를 띄우지 않는다. 40GB에 들어가는지는 GPU smoke와 실제 실행에서 확인한다.

아래는 **Slurm 없는 전용 서버에서 사용 권한을 확보한 GPU 6장이 있을 때의 배치 예시**다. 각 줄은 해당 저장소 루트의 별도 터미널/tmux에서 실행한다. 번호는 실제 확보한 GPU에 맞춘다. 모든 명령을 한 터미널에 붙이면 순차 실행되며 자동 병렬화되지 않는다.

```bash
# selector 저장소, 각각 별도 터미널
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh configs/A_loo.json orchestration/allq_k10/A_loo
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh configs/B_joint.json orchestration/allq_k10/B_joint
CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh configs/vanilla_control.json orchestration/allq_k10/vanilla
```

```bash
# branch 저장소, 각각 별도 터미널
CUDA_VISIBLE_DEVICES=3 bash scripts/run.sh configs/branch_p025.json orchestration/branch_p025
CUDA_VISIBLE_DEVICES=4 bash scripts/run.sh configs/vanilla_matched.json orchestration/vanilla_matched
CUDA_VISIBLE_DEVICES=5 bash scripts/run.sh configs/vanilla_original.json orchestration/vanilla_original
```

GPU가 적으면 같은 repo의 후속 조건을 `queue.py append`로 기존 큐에 넣어 순차 실행할 수 있다. 별도 실행과 append로 같은 output을 중복 사용하지 않는다. 서로 다른 repo의 config를 한 repo의 queue에 넣지 않는다. 각 launcher/module은 자기 repo를 기준으로 해석한다.

| 조건 | 자동 실행 순서 |
|---|---|
| A | smoke → 시작 매니저 selector-off 평가 → 선택기 학습 → 최종 평가 |
| B, B-control | smoke → 학습 → 최종 평가 |
| 세 번째와 그 기준선 | smoke → 학습 → 최종 평가 → CPU 분석 |

A 초기 평가는 전체 held-out 대화를 진행하므로 학습 로그가 늦게 생겨도 즉시 정지로 판단하지 않는다. smoke의 합성 advantage는 연결 검사이며 과학적 성능 실험이 아니다. 자동 평가를 별도로 중복 실행하거나 partial checkpoint로 대체하지 않는다.

Slurm에서는 기존 호환 allocation을 먼저 재사용하고 스케줄러가 정한 GPU mask를 유지한다. 이미 다중 GPU allocation이면 사이트 정책에 맞는 step/binding으로 실험마다 1장만 노출한다. 새 allocation이 필요할 때만 실제 partition/account/GRES/CPU/RAM/허용 최대 wall time을 확인하여 `scripts/slurm_entry.sh`를 사용한다. 값은 각 서버 안내서에 있으며 임의의 서버 설정을 만들어 제출하지 않는다.

## 7. 실패 처리와 allocation 보존

outer supervisor가 allocation을 소유하고 smoke/train/eval/analyze는 자식 stage다. 단계 실패/수정 때문에 routine으로 `scancel` 후 재제출하지 않는다. 큐는 원자적으로 갱신되고 stage마다 다시 읽힌다. 기본 수리 대기는 1,800초이며 무한 재시도/무기한 유휴 점유는 없다. 승인된 작업이 모두 끝나면 자원을 반환한다. wall time·preemption·노드 장애를 우회할 수는 없다.

```bash
# selector 저장소의 A 예시; 실제 stage ID는 status에서 확인
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/A_loo
.venv/bin/python scripts/queue.py cancel --root orchestration/allq_k10/A_loo --id A_loo_train
# 해당 원인 해결 후
.venv/bin/python scripts/queue.py resume --root orchestration/allq_k10/A_loo --id A_loo_train
```

cancel은 해당 실험의 검증된 PID·시작 시각·소유 표식으로 process tree만 정지한다. 광범위한 `pkill`, GPU 전체 reset, supervisor/sibling을 죽이는 process-group signal을 쓰지 않는다. 이전 worker가 종료되어 VRAM이 해제됐는지 확인한 뒤 대체 stage를 실행한다. supervisor가 살아 있으면 run 명령을 중복 호출하지 않는다. 죽은 경우 기존 allocation 상태를 먼저 확인하고 동일 run 명령과 해당 stage resume 요청을 사용한다.

학습 재개는 마지막 `COMMITTED.json` checkpoint 경계다. LoRA뿐 아니라 optimizer, RNG, bank, 대화/세션 cursor와 세 번째의 phase/95% 이력이 필요하다. 미확정 배치는 재수집하고, 중단된 실행을 완료 결과로 보고하지 않는다. 최신 2 checkpoint와 모든 배치 trace/metrics를 보존한다.

OOM/시간 초과 해결을 위해 K, p, bank 예산, 질문풀, dtype, 모델 길이, reward, loss 가중치, 학습량을 자동 축소하지 않는다. 먼저 중복 worker·잘못된 GPU mask·드라이버/환경·남은 자원을 확인한다. 조건 변경이 필요한 경우 바뀌는 조건과 근거를 보고하고 새로운 run/fingerprint로 구분한다. 코드 수정은 이미 로드된 interpreter에 반영되지 않으며 source/config/runtime fingerprint가 다르면 기존 checkpoint를 그대로 재개할 수 없다. 실행 중인 checkout에 `git pull`하지 않는다.

## 8. 정상 진행과 완료 판정

| 단계 | 확인할 증거 |
|---|---|
| 준비 | 실제 서버의 CPU 검사, 입력 hash, full preflight, commit·환경 |
| smoke | `smoke/SMOKE_COMPLETE.json`, status complete, GPU명/CUDA/peak_allocated_bytes |
| 학습 중 | queue 상태, 수집 로그, `progress.json`, 배치 trace/metrics 증가 |
| 학습 완료 | `TRAIN_COMPLETE.json`: complete, 기본 142 completed sessions, 유효 checkpoint |
| 평가 완료 | `evaluation/EVAL_COMPLETE.json`: complete, questions=729, pooled metrics |
| A 비교 평가 | `evaluation_initial_manager/EVAL_COMPLETE.json` |
| 세 번째 분석 | `ANALYSIS_COMPLETE.json`, 필요 시 조건 간 `analysis/comparison.json` |
| 큐 완료 | 큐 `ALL_COMPLETE.json`과 status `complete_allocation_releasing` |

파일 존재만으로 완료를 판단하지 않고 status/fingerprint/필수 개수를 확인한다. 학습 완료와 평가 완료, allocation 보유와 stage 진행 상태를 별개로 보고한다. 긴 수집 중에는 progress가 갱신되지 않을 수 있다. A의 manager 업데이트 수는 0이어야 한다. B의 두 optimizer나 세 번째 manager가 매 세션 모두 업데이트되어야 하는 것은 아니다. 동점/무신호/무분기 세션은 정상적으로 skip할 수 있다.

세 번째에서는 전환 세션의 `phase_used=1`, `next_phase=2`, 네 `end_occupancy>=0.95`를 확인한다. 다음 세션부터 2단계여야 한다. 모든 대화가 반드시 2단계에 도달해야 하는 것은 아니다. `branch_groups`, `signal_groups`, `invalid_executed`, 학습 출력 수/정규화 계수도 확인한다.

## 9. 평가 해석과 결과 반환

held-out 5대화/729문항의 근거 보존 평가다. `pooled.complete`는 근거 커버리지, `recall`은 근거 재현율, `hit`는 근거 적중률이며 값은 0~1이다. 백분율은 ×100, 두 비율의 차이 ×100은 percentage points로 표시한다. QA reader를 실행하지 않으므로 EM/F1이나 답변 정답률로 바꾸어 부르지 않는다.

평가에는 학습/오라클/K분기 채점이 없다. A/B는 REPLACE에서 선택기를 한 번 사용한다. 세 번째와 matched는 단일 greedy bank의 자기 세션끝95%로 다음 세션의 phase를 전환하며 훈련의 all-four 조건을 복제하지 않는다. original은 5액션을 유지한다. 세 번째의 1단계 평가 최대 시도는 6, 학습은 3이며 2단계는 1/no-op이다. 공식 근거 라벨로 정책을 선택하거나 평가 조건을 유리하게 고르지 않는다.

세 번째 조건 비교는 branch 저장소에서 다음 CPU 명령을 사용한다.

```bash
.venv/bin/python -m cmbranch.analyze \
  --run runs/branch_v1/branch_p025 \
  --run runs/branch_v1/vanilla_matched \
  --run runs/branch_v1/vanilla_original \
  --report analysis/comparison.json
```

A/B 비교 집계 명령은 selector 서버 안내서의 결과 확인 절에 있다. 과거 vLLM 수치와 새 HF 엔진 결과를 같은 조건으로 간주하지 않는다. 세 번째의 K4,p0.25에서 기대 생성 출력 1.75개/상태, 기대 학습 출력 1개/상태라는 것은 실측 시간/FLOPs 동등성의 증명이 아니다. 생성/학습 토큰, 수집/업데이트 시간, peak VRAM을 함께 보고한다.

세 번째 학습의 scored REPLACE 손실은 채점한 분기에서 실제 실행한 경우만 포함한다. 평가의 post-hoc REPLACE 이벤트는 실제 교체 전체를 포함하지만 같은 문항의 반복 상실을 중복 집계할 수 있다. 이벤트 합을 최종 coverage 차이 또는 QA 오답 수로 해석하지 않는다. 95% 미만 근거/비근거 ADD율은 실제 방문 상태의 관측 통계다.

반환할 최소 자료는 각 run의 smoke/train/eval 완료 JSON, A 초기 평가 JSON, 세 번째 분석 JSON, 각 큐의 state/queue/logs/ALL_COMPLETE, preflight와 freeze, GPU/드라이버·실행 commit/변경 상태다. 최종 checkpoint 전체와 `batches/`도 보존한다. 대용량 trace/가중치/원문 데이터는 Git에 추가하지 않는다.

에이전트의 최종 보고는 다음 표를 채운다. 수행하지 않은 항목은 `미실행`, 중간 중단은 `중단`, 평가가 없으면 `평가 미완료`로 표시한다.

| 조건 | 실제 상태 | commit/fingerprint | GPU/할당 | 재개 checkpoint | Cov/Rec/Hit | 시간·peak VRAM | 결과 경로 |
|---|---|---|---|---|---|---|---|
| A / 초기 매니저 | | | | | | | |
| B / B-control | | | | | | | |
| Third / matched / original | | | | | | | |

실패 시에는 실패 stage/원인/마지막 확정 상태/할당 유지 여부/실제로 바꾼 파일/다음 조치를 추가한다. 예상 성능·역사적 오라클 수치·CPU smoke 수치를 이번 학습 결과로 채워 넣지 않는다.

## 10. 박사님이 AI 에이전트에 전달할 짧은 지시문

> CoverMem 두 저장소의 AGENTS.md와 docs/AI_AGENT_HANDOFF_KO.md를 먼저 읽고, Drive 자료 복원부터 입력·환경 검증을 진행해줘. A100 40GB 서버에서 A/B와 세 번째 실험 및 지정 비교군을 제공 설정과 supervisor로 실행·평가하고, 완료 기준과 결과 반환 형식은 인계서를 따라줘. 기존 GPU 할당을 우선 활용하고 실험 조건은 임의로 변경하지 마. 권한이나 서버 설정이 실제로 누락된 경우에만 해당 항목을 물어봐.
