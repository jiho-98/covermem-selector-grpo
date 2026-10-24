# CoverMem REPLACE Selector GRPO

REPLACE 실행 시 삭제 대상을 고르는 별도 LoRA를 학습하는 A/B 실험 코드다. **매 REPLACE에서 해당 훈련 대화의 전체 생성 질문으로 교체 후−전 점수를 계산하고, K=10개 출력을 같은 상태 안에서 비교한다.** 현재/미래 질문의 7:3 가중치는 사용하지 않는다. 기존 점수 정의인 `0.7 × 근거 커버리지 + 0.3 × 근거 재현율`은 유지한다.

**대상 서버: A100 40GB × 16장. 현재 실행기는 실험당 1 GPU/1 프로세스이며, 16 GPU 분산 학습은 구현하지 않았다.** A/B를 두 GPU에서 독립 실행할 수 있다. 실제 4B 모델의 A100 메모리·속도·성능은 아직 검증하지 않았으며, 제공한 실행 큐가 GPU smoke를 먼저 수행한다. 로컬에서 GPU 작업을 시작하지 않았다.

| 실험 | 학습 방식 | config | 비교 대상 |
|---|---|---|---|
| A | 완료된 LOO 매니저를 고정하고 훈련 대화를 재생하며 선택기만 학습 | `configs/A_loo.json` | 같은 매니저의 selector-off 평가, 큐에서 자동 수행 |
| B | SFT부터 매니저와 선택기를 독립 LoRA/optimizer로 학습 | `configs/B_joint.json` | `configs/vanilla_control.json`, 같은 실행 엔진의 기준선 |
| A-H1, 선택 사항 | A의 시작 매니저만 완료 H1으로 변경 | `configs/A_h1.json` | 해당 H1의 selector-off 평가 |

선택기 local advantage와 매니저 기존 LOO advantage는 별개다. 같은 REPLACE의 10개만 상대 비교하며, 다른 경로의 후보를 한 group에 섞지 않는다. 실제 경로는 미리 지정한 샘플 0으로 계속 진행한다. K=10은 생성 수이며 고유 ID 10개를 보장하지 않는다. 세 번째 연구안인 단일 매니저 95% 전환 분기 GRPO는 [별도 저장소](https://github.com/jiho-98/covermem-branch-grpo)에 구현했다. 그 기본값 K=4,p=0.25는 이 저장소의 A/B K=10과 구분한다.

## 먼저 읽기

- **[설치부터 A/B 실행·확인까지: 서버 실행 안내](docs/SERVER_RUNBOOK_KO.md)** — 처음 실행하는 분을 위한 명령 모음.
- **[박사님께 전달할 메시지](docs/HANDOFF_MESSAGE_KO.md)** — 저장소/ZIP과 함께 전달할 본문.
- [상세 운영 설명](README_KO.md) — 자원 할당 유지, 재개, 평가 범위.
- [방법·설정·비용](METHOD_AND_CONFIG_KO.md) — A/B 차이, reward, loss 마스크, 비교 조건.
- [검증 범위와 미검증 사항](validation/VALIDATION_KO.md) — CPU 검증을 GPU 검증과 구분.

## GitHub 코드 + 전달 ZIP으로 준비

이 저장소는 비공개다. 저장소 소유자가 박사님 GitHub 계정에 접근 권한을 부여하고 초대를 수락해야 clone할 수 있다. GitHub 계정 없이도 전체 ZIP을 풀어 동일 코드로 실행할 수 있다.

Git에는 코드·config·환경 고정 파일·문서·원본 코드 스냅샷이 포함된다. **`assets/`, `adapters/`, 베이스 모델, 실행 결과는 포함하지 않는다.** 데이터와 SFT/LOO/H1/배치 LOO 어댑터는 별도 `CoverMem_selector_GRPO_AB_full.zip`에서 가져온다. 베이스 모델은 고정 revision으로 다운로드한다.

```bash
git clone https://github.com/jiho-98/covermem-selector-grpo.git
cd covermem-selector-grpo

# Python 3.10 준비. 이미 python3.10이 있으면 conda 두 줄은 생략.
conda env create -f environment.yml
conda activate covermem-selector

# ZIP 위치는 받은 파일의 실제 경로로 바꾼다. 코드/문서는 덮어쓰지 않는다.
python3.10 scripts/import_assets.py --archive /path/to/CoverMem_selector_GRPO_AB_full.zip
bash scripts/setup_env.sh
bash scripts/download_model.sh
bash scripts/check_cpu.sh configs/A_loo.json
bash scripts/preflight.sh configs/A_loo.json
bash scripts/preflight.sh configs/B_joint.json
```

설치 Python은 3.10, PyTorch는 2.11.0이다. 기본 CUDA wheel은 cu130이며 서버 드라이버가 이를 지원해야 한다. 환경 선택과 오프라인 모델 준비는 [서버 안내](docs/SERVER_RUNBOOK_KO.md)를 따른다. 설치/CPU 검사/모델 다운로드는 GPU 학습을 실행하지 않는다.

## A/B 실행 — 일반 전용 서버

먼저 서버에서 사용 가능한 GPU를 확인한다. 아래 0/1은 사용 권한이 확보된 GPU의 예시다. 각 명령은 별도 터미널 또는 tmux 창에서 실행한다.

```bash
# A: smoke → 시작 매니저 평가 → 선택기 학습 → 최종 평가
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh configs/A_loo.json orchestration/allq_k10/A_loo

# B: smoke → 매니저/선택기 학습 → 최종 평가
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh configs/B_joint.json orchestration/allq_k10/B_joint
```

Slurm 환경에서는 위 GPU 번호를 직접 지정하지 않고 스케줄러의 할당을 사용한다. [Slurm 명령](docs/SERVER_RUNBOOK_KO.md#5-slurm-서버)을 참고한다. `torchrun`으로 16개 프로세스를 실행하지 않는다.

```bash
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/A_loo
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/B_joint
tail -F orchestration/allq_k10/A_loo/logs/A_loo_train.log
```

학습 결과는 `runs/allq_k10/A_loo/`와 `runs/allq_k10/B_joint/`에 저장한다. 학습 `TRAIN_COMPLETE.json`, 평가 `evaluation/EVAL_COMPLETE.json`, 큐 `ALL_COMPLETE.json`까지 확인해야 전체 완료다. 평가는 5대화/729문항의 근거 보존 지표이며 QA 정답률·EM/F1은 아니다.

## 코드 위치

| 경로 | 역할 |
|---|---|
| `cmselect/train.py`, `rollout.py` | A/B 수집·업데이트와 실제 뱅크 진행 |
| `cmselect/selection.py` | 전체 질문 교체 전후 보상, K개 local advantage |
| `cmselect/backend.py`, `masking.py` | LoRA 전환, GRPO loss, 매니저 REPLACE loss 마스크 |
| `cmselect/evaluate.py`, `checkpoint.py` | 최종 평가와 재개 가능한 checkpoint |
| `scripts/`, `supervision/` | 설치·입력 가져오기·실행 큐·지속 supervisor |
| `configs/`, `tests/` | 실험 config 및 CPU 테스트 |
| `legacy/` | H1/LOO/배치 LOO 원본 코드와 완료 기록; 보존용이며 새 서버 실행 진입점은 `scripts/` |

실험 중인 checkout에서 `git pull`하거나 config를 바꾸지 않는다. 과학적 조건·소스·런타임 변경은 fingerprint가 달라지므로 새로운 checkout과 출력 경로를 사용한다. 결과를 보낼 때 실행 commit, fingerprint, 환경 버전도 함께 보존한다.
