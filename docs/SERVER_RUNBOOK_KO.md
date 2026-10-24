# A100 40GB × 16장 서버 실행 안내

AI 에이전트는 먼저 [통합 인계서](AI_AGENT_HANDOFF_KO.md)를 읽는다. 자료는 [Google Drive 다운로드·분할 복원 안내](DOWNLOAD_AND_RESTORE_KO.md)에 따라 준비한다. 최신 코드는 Git checkout을 사용하고 복원 ZIP에서는 입력만 가져오는 방법을 권장한다.

이 안내는 **전체 질문 보상·K=10·schema v2** 코드 기준이다. 설치와 실행 명령은 저장소 루트에서 수행한다. 과거 7:3 시간 가중치 전달본은 사용하지 않는다. GPU 작업은 이 안내 작성 과정에서 실행하지 않았다.

## 1. 실행 구성과 먼저 확인할 것

| 항목 | 조건 |
|---|---|
| OS / Python | Linux x86_64 / Python 3.10 |
| GPU | A100 40GB. 실험당 1장, 프로세스 1개 |
| 모델 | Qwen3-4B-Instruct-2507, BF16, gradient checkpointing |
| 학습 경로 / 후보 | 대화당 뱅크 4개 / 유효 REPLACE당 K=10 |
| 기본 update batch | 한 세션, `session_batch_size=1` |
| 보상 | 해당 훈련 대화 전체 생성 질문의 교체 후−전 utility |
| A | 완료 LOO 매니저 고정 + 선택기 학습 |
| B | SFT에서 매니저와 선택기를 별도 LoRA/optimizer로 학습 |
| B 기준선 | 같은 엔진 `vanilla_control.json` |

16장 중 A/B에 1장씩, 기준선에 1장을 독립 배정할 수 있다. 나머지 GPU를 자동 점유하지 않는다. 네 경로와 K=10은 GPU 개수 또는 동시 프로세스 수가 아니다. A100 한 장의 메모리에 들어가는지는 첫 GPU smoke에서 확인한다. smoke 성공도 모든 응답 길이의 메모리 상한을 보장하지 않는다.

서버에서 확인한다:

```bash
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
df -h .
free -h
command -v sbatch || true
```

`nvidia-smi` 결과와 서버 운영 정책으로 사용 가능한 GPU를 확인한다. 모델 파일 외에 입력 약 0.6GB, Python/CUDA wheel, checkpoint와 배치별 긴 trace를 저장할 공간이 필요하다. 전체 실행의 실제 디스크·RAM·시간 사용량은 미측정이며, 운영 중 여유 공간을 확인한다. checkpoint는 최신 2개, 배치 trace/metrics는 모두 보존한다.

## 2. 코드와 입력 준비

### 방법 1: GitHub clone + 전달 ZIP에서 입력 가져오기

저장소 소유자가 박사님 계정에 초대하고 초대가 수락되어 있어야 한다. 비공개 repo에 대한 평소 GitHub HTTPS/SSH 인증을 사용한다.

```bash
git clone https://github.com/jiho-98/covermem-selector-grpo.git
cd covermem-selector-grpo

# Python 3.10이 이미 있으면 아래 conda 두 줄을 생략한다.
conda env create -f environment.yml
conda activate covermem-selector

# 실제 받은 ZIP 경로로 바꾼다.
python3.10 scripts/import_assets.py --archive /path/to/CoverMem_selector_GRPO_AB_full.zip
python3.10 scripts/import_assets.py --check-only
```

정상 출력은 `status: inputs_verified`, `files: 19`, `gpu_used: false`다. 이미 같은 파일이 있으면 재사용한다. ZIP의 코드/설명서가 checkout을 덮어쓰지 않으며 `INPUTS.sha256.json`에 고정된 입력만 가져온다. 해시가 다른 기존 파일은 덮어쓰지 않고 실패한다. 약 0.6GB의 임시 추가 공간을 사용한다.

### 방법 2: GitHub 없이 전체 ZIP만 실행

ZIP과 `SHA256SUMS.txt`를 같은 폴더에 놓는다. SHA 파일에는 code-only ZIP도 포함되므로 전체 ZIP 행만 검사한다.

```bash
python3 - <<'PY'
from pathlib import Path
import hashlib
name = 'CoverMem_selector_GRPO_AB_full.zip'
expected = next(line.split()[0] for line in Path('SHA256SUMS.txt').read_text().splitlines() if line.split()[-1] == name)
h = hashlib.sha256()
with Path(name).open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
assert h.hexdigest() == expected, 'ZIP SHA256 mismatch'
print('ZIP SHA256 OK')
PY
unzip CoverMem_selector_GRPO_AB_full.zip
cd covermem_selector
conda env create -f environment.yml
conda activate covermem-selector
python3.10 scripts/import_assets.py --check-only
```

기존 실행 폴더 위에 덮어 풀지 않고 새 폴더에서 시작한다. code-only ZIP에는 데이터/어댑터가 없어 단독 실행할 수 없다.

## 3. 환경 설치·모델 다운로드·CPU 확인

```bash
bash scripts/setup_env.sh
bash scripts/download_model.sh
bash scripts/check_cpu.sh configs/A_loo.json
bash scripts/preflight.sh configs/A_loo.json
cp validation/preflight.json validation/preflight_A_loo.json
bash scripts/preflight.sh configs/B_joint.json
cp validation/preflight.json validation/preflight_B_joint.json
```

`setup_env.sh`는 루트의 `.venv`를 만들고 여기에 설치한다. 이후 명령은 `.venv/bin/python`을 명시하므로 conda 환경을 바꿔도 실행 Python이 섞이지 않는다. Python을 다른 경로로 지정하려면 설치할 때 `COVERMEM_PYTHON=/path/to/python3.10 bash scripts/setup_env.sh`를 사용한다.

| 패키지 | 고정 버전 |
|---|---|
| PyTorch | 2.11.0 |
| Transformers | 5.8.0 |
| PEFT | 0.19.1 |
| Accelerate | 1.14.0 |
| 나머지 | `requirements.txt`, `environment.lock.json` |

기본 torch wheel은 `https://download.pytorch.org/whl/cu130`에서 받는다. 서버 드라이버가 해당 CUDA 빌드를 지원하는지 확인한다. 시스템 `nvcc` 버전만으로 판단하지 않는다. 관리자가 다른 호환 공식 PyTorch 2.11.0 wheel index를 지정한 경우 설치 전에 `COVERMEM_TORCH_INDEX_URL`을 설정한다. 드라이버를 이 코드가 설치하지 않으며, wheel/CUDA 빌드 변경 후 기존 checkpoint를 재개하지 않는다. vLLM/verl은 새 실행기에 필요하지 않다.

모델 revision은 `cdbee75f17c01a7cc42f958dc650907174af0554`, 저장 경로는 `models/Qwen3-4B-Instruct-2507`이다. 외부 인터넷이 없으면 다른 컴퓨터에서 이 revision의 전체 모델 파일을 받아 해당 위치에 복사한다. 이미 같은 모델이 있으면 config의 `paths.base_model`에 경로를 지정할 수 있다. 상대 경로는 패키지 루트 기준이며 어댑터 안의 과거 절대 경로를 사용하지 않는다.

다음을 확인한 뒤 실행한다:

- CPU 테스트: `tests` 전체 `OK`, supervisor 2개 `OK`. 입력 가져오기 검사 4개가 추가되어 현재 `tests`는 25개다.
- 입력 preflight: `assets_validated`, 142세션/3,338질문, GPU 사용 없음.
- full preflight: `prepared`, fingerprint가 비어 있지 않음. 모델 shard/설치 버전/입력 hash 오류 없음.
- `validation/preflight_A_loo.json`, `preflight_B_joint.json`, `installed.freeze.txt` 보존.

full preflight는 모델 shard를 CPU로 해시하므로 잠시 출력이 없어도 GPU를 사용하는 것이 아니다. 실제 GPU 메모리 검증은 다음 단계의 smoke다. 현재 수정본의 사전 검증은 로컬 torch2.12에서 수행했으며, 서버의 고정 환경 torch2.11에서 위 검사를 다시 통과해야 한다.

## 4. 일반 전용 서버에서 A/B 실행

Slurm 서버면 이 절 대신 다음 절을 따른다. 아래 GPU 0/1/2는 사용 권한이 확보된 장치 번호로 바꾼다. SSH 연결 종료로 작업이 중단되지 않도록 tmux 또는 운영 중인 작업 관리 세션 안에서 수행한다. `tmux`가 설치되어 있다면 `tmux new -s covermem-A`로 시작하고, Ctrl-b를 누른 뒤 d로 빠져나올 수 있다.

터미널 A, 저장소 루트:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh configs/A_loo.json orchestration/allq_k10/A_loo
```

터미널 B, 같은 저장소 루트:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh configs/B_joint.json orchestration/allq_k10/B_joint
```

비교용 vanilla, 권한이 확보된 별도 GPU에서:

```bash
bash scripts/preflight.sh configs/vanilla_control.json
cp validation/preflight.json validation/preflight_vanilla_control.json
CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh configs/vanilla_control.json orchestration/allq_k10/vanilla
```

| 실행 | 자동 진행 순서 | 출력 |
|---|---|---|
| A | smoke → 고정 매니저 selector-off 평가 → 선택기 학습 → 최종 평가 | `runs/allq_k10/A_loo/` |
| B | smoke → 매니저/선택기 학습 → 최종 평가 | `runs/allq_k10/B_joint/` |
| vanilla | smoke → 매니저 학습 → 최종 평가 | `runs/allq_k10/vanilla_control/` |

A 초기 평가는 5대화를 처음부터 진행하므로, 학습 로그가 아직 없더라도 초기 평가가 정상 진행 중일 수 있다. 각 큐는 실험 완료 후 자동 종료한다. 한 실험에 GPU가 정확히 한 장만 보이도록 해야 하며 `torchrun --nproc_per_node=16`은 사용하지 않는다.

H1 시작 A는 추가로 필요할 때 `configs/A_h1.json`과 `orchestration/allq_k10/A_h1`을 사용한다. 원래 H1 알고리즘을 새로 학습하는 명령은 아니다.

## 5. Slurm 서버

먼저 기존 할당을 확인한다. **호환되는 기존 1-GPU 할당이 있으면 이를 재사용**하고, GPU 번호는 Slurm이 제공한 `CUDA_VISIBLE_DEVICES`를 유지한다. 서로 다른 실험을 같은 출력/큐에 중복 실행하지 않는다.

```bash
squeue -u "$USER"
# 기존 할당 안에서, 사용 가능한 GPU 한 장만 노출된 경우:
bash scripts/run.sh configs/A_loo.json orchestration/allq_k10/A_loo
```

새 할당이 필요한 경우 아래 변수의 실제 값만 서버 규칙에 맞게 설정한다. partition/account/GRES 명칭/허용 wall time은 아직 제공받지 않았으므로 임의로 고정하지 않았다. 16장 전체를 요청할 필요는 없다.

```bash
# 이 다섯 값을 서버 관리자가 제공한 값으로 수정한다.
CM_PARTITION='실제_partition'
CM_GRES='gpu:a100:1'
CM_CPUS='허용_CPU수'
CM_MEM='허용_RAM_예_64G'
CM_WALLTIME='허용_최대시간_예_2-00:00:00'
mkdir -p orchestration/slurm_logs

# A: 한 장짜리 allocation의 outer job이 supervisor를 소유한다.
sbatch --job-name=cm-A --partition="$CM_PARTITION" --gres="$CM_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task="$CM_CPUS" --mem="$CM_MEM" --time="$CM_WALLTIME" \
  --output=orchestration/slurm_logs/A_%j.log \
  scripts/slurm_entry.sh configs/A_loo.json orchestration/allq_k10/A_loo

# B: 별도의 한 장짜리 allocation
sbatch --job-name=cm-B --partition="$CM_PARTITION" --gres="$CM_GRES" \
  --nodes=1 --ntasks=1 --cpus-per-task="$CM_CPUS" --mem="$CM_MEM" --time="$CM_WALLTIME" \
  --output=orchestration/slurm_logs/B_%j.log \
  scripts/slurm_entry.sh configs/B_joint.json orchestration/allq_k10/B_joint
```

account가 필요한 클러스터는 각 `sbatch`에 `--account=실제_account`를 추가한다. CPU는 기본 config가 4 thread를 사용하므로 4개 이상을 할당한다. 이미 16장짜리 할당을 받은 경우 운영 정책에 맞게 `srun` 등으로 **각 작업에 한 장을 노출하는 독립 step**을 만들거나 관리자에게 배치를 확인한다. 이 코드가 기존 할당을 해제하거나 임의로 재구성하지 않는다.

다음 승인된 실험을 같은 할당에서 이어 실행하려면 현재 큐가 끝나기 전에, 남은 시간과 자원을 확인하고 append한다:

```bash
.venv/bin/python scripts/queue.py append --root orchestration/allq_k10/B_joint --config configs/vanilla_control.json
```

이 경우 vanilla의 큐 로그/상태는 B의 orchestration 아래에 있고 결과는 여전히 `runs/allq_k10/vanilla_control/`이다. 별도 vanilla 큐와 동시에 시작하지 않는다. 단계 완료/실패가 outer allocation을 무조건 끝내지는 않으며, 실패 시 기본 30분 수리 대기 후 종료한다. 모든 승인된 단계 완료 후에는 자동 반환한다. wall time 만료·노드 장애를 넘어 할당을 유지하는 것은 아니다.

## 6. 진행 확인과 정상 판정

다른 터미널에서:

```bash
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/A_loo
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/B_joint

# 파일이 아직 없어도 생성될 때까지 기다리는 tail -F
tail -F orchestration/allq_k10/A_loo/logs/A_loo_smoke.log
tail -F orchestration/allq_k10/A_loo/logs/A_loo_initial_control.log
tail -F orchestration/allq_k10/A_loo/logs/A_loo_train.log
tail -F orchestration/allq_k10/B_joint/logs/B_joint_train.log
```

tail은 하나씩 실행하고 Ctrl-C로 관찰을 끝낸다. 이는 학습 중단 명령이 아니다. 실행 로그는 각 단계의 `*.log`에 있으며, `progress.json`은 배치를 확정할 때 갱신되므로 긴 수집 중에는 값이 그대로일 수 있다.

| 시점 | 확인 파일/값 |
|---|---|
| GPU smoke | `runs/allq_k10/<실험>/smoke/SMOKE_COMPLETE.json`: `status=complete`, device, CUDA 빌드, `peak_allocated_bytes` |
| 수집/학습 중 | 로그의 `collect_batch`, `batch_committed`; `progress.json`의 `next_batch`, `updates` 증가 |
| A 업데이트 | manager 업데이트 수는 0 유지. selector는 유효한 비동점 신호가 있는 배치에서 증가 |
| B 업데이트 | manager/selector 별도 업데이트 수. 모든 배치에서 둘 다 증가해야 하는 것은 아님 |
| 학습 완료 | `TRAIN_COMPLETE.json`: `status=complete`, 기본 142 completed sessions, 완료 checkpoint |
| 평가 완료 | `evaluation/EVAL_COMPLETE.json`: `status=complete`, `questions=729`, pooled metrics |
| 큐 전체 완료 | `orchestration/allq_k10/<큐>/ALL_COMPLETE.json`, 상태 `complete_allocation_releasing` |

선택기가 같은 ID만 반복하거나 점수가 모두 동점이면 해당 group의 advantage는 0이다. 학습 오류와 구분하여 trace의 `distinct_sampled_targets`, `has_signal`, `fallback`, 각 후보의 `reward`/`advantage`를 확인한다. smoke는 합성 advantage로 optimizer 연결을 검사하며, K=10 실제 학습이나 선택 성능을 측정하는 실험은 아니다.

## 7. 결과 확인과 비교

완료 후, 저장소 루트에서:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
for name, sub in [('A_loo','evaluation_initial_manager'), ('A_loo','evaluation'),
                  ('B_joint','evaluation'), ('vanilla_control','evaluation')]:
    p = Path('runs/allq_k10') / name / sub / 'EVAL_COMPLETE.json'
    if not p.exists():
        print(name, sub, '아직 평가 미완료')
        continue
    r = json.loads(p.read_text())
    assert r['status'] == 'complete' and r['questions'] == 729
    print(name, sub, {k: round(100*v, 2) for k,v in r['pooled'].items()})
PY
```

`complete`는 **근거 커버리지**, `recall`은 근거 재현율, `hit`는 근거 적중률이다. 위 출력은 백분율이다. QA reader를 실행하지 않으므로 EM/F1·실제 정답률은 없다. A는 자기 시작 매니저와, B는 같은 실행기의 vanilla와 비교한다. 과거 vLLM 평가 수치와 실행 엔진이 달라 새 결과만 직접 붙여 비교하지 않는다.

학습 분석은 `runs/allq_k10/<실험>/batches/batch_XXXXXX/metrics.json` 및 `traces.json`에서 한다. traces는 대화별 `streams[].selector_groups[]`에 REPLACE group을 저장한다. 각 group은 전체 질문 수 `all_questions`, 교체 전 `before_all`, 후보 `rows[]` 안에 `after_all`, `delta_all`, `gained_complete`, `lost_complete`, `net_complete`, `utility_gain`, `utility_loss`, `advantage`를 갖는다. gained/lost는 실제 QA 정답 수가 아니라 충분 근거 확보/상실 문항 수다.

## 8. 실패·중단·재개

OOM/형식 오류/확률 일치 오류가 나면 먼저 실패한 stage 로그를 확인한다. 완료 marker가 없는 단계를 완료로 간주하지 않는다. supervisor는 기본 1,800초 수리 대기 중 할당을 유지하며 자동 무한 재시도하지 않는다.

```bash
# 실제 stage ID는 queue status로 확인. A 학습만 중단하는 예시.
.venv/bin/python scripts/queue.py cancel --root orchestration/allq_k10/A_loo --id A_loo_train

# 원인을 해결한 뒤 그 단계만 재개 요청
.venv/bin/python scripts/queue.py resume --root orchestration/allq_k10/A_loo --id A_loo_train
```

B의 stage ID는 `B_joint_smoke`, `B_joint_train`, `B_joint_eval`이다. smoke가 실패했다면 train 대신 smoke ID를 resume한다. supervisor가 살아 있는 동안은 run 명령을 중복 호출하지 않는다. supervisor가 이미 종료된 경우 기존 자원 할당을 확인한 뒤 동일 run 명령으로 시작하고 해당 stage의 resume 요청을 사용한다. Slurm 할당이 끝났으면 기존 할당 재사용 가능 여부를 먼저 확인하고 필요할 때 새 할당을 받는다.

학습 재개는 마지막 **COMMITTED checkpoint의 배치 경계**부터다. 두 LoRA/optimizer, RNG, 4개 뱅크, 대화 cursor를 복원하며, 끊긴 배치의 미확정 작업은 재수집한다. 평가에는 턴별 진행 저장이 있다. source/config/환경 fingerprint가 다르면 기존 결과 폴더로 이어 학습하지 않는다. GPU 실행 중 파일을 수정해도 이미 로드된 Python이 갱신되지 않는다.

| 증상 | 확인/조치 |
|---|---|
| `Repository not found` | 비공개 저장소 초대 수락과 Git 인증 확인. 전체 ZIP으로도 실행 가능 |
| 입력 SHA256 mismatch | 받은 ZIP/기존 입력 확인. 변경된 입력 위에 덮어쓰지 말고 새 checkout 사용 |
| runtime version mismatch | Python3.10과 `.venv`, `environment.lock.json` 대조 |
| `Expose exactly one GPU` | 한 작업에 GPU 한 장만 노출. Slurm mask/step 확인 |
| CUDA/driver 오류 | 서버 드라이버와 설치 wheel 호환성을 관리자와 확인 |
| CUDA OOM | stage 로그·smoke peak·GPU 사용 상태 보존. K/예산/정밀도를 임의 변경하지 않고 공유 |
| `Previous stage remains alive` | 기록된 owner/child PID가 남았는지 확인. 중복 실행이나 전체 `pkill` 금지 |
| changed fingerprint | 동일 조건/버전인지 확인. 새 연구 조건이면 새 checkout/출력 디렉토리 |

## 9. 박사님께 요청할 반환 파일

우선 각 실험의 아래 파일을 전달하면 실행 상태와 비교 결과를 확인할 수 있다:

- `smoke/SMOKE_COMPLETE.json`, `TRAIN_COMPLETE.json`, `evaluation/EVAL_COMPLETE.json`.
- A의 `evaluation_initial_manager/EVAL_COMPLETE.json`.
- 해당 orchestration의 `state.json`, `queue.json`, `logs/`, 있으면 `ALL_COMPLETE.json`.
- `validation/preflight_*.json`, `validation/installed.freeze.txt`, GPU/드라이버 정보.
- Git 실행 commit: `git rev-parse HEAD`, 변경 여부: `git status --short` (ZIP 실행이면 전달 ZIP SHA256).

최종 checkpoint는 `TRAIN_COMPLETE.json`의 `checkpoint` 경로 전체를 보존한다. `adapters/manager`, `adapters/selector`만으로는 optimizer/RNG/뱅크를 복구할 수 없다. 후속 분석용 `batches/`도 함께 보존한다. 큰 trace와 모델 가중치는 Git에 추가하지 않는다.
