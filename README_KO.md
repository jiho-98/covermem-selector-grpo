# CoverMem REPLACE 선택기 GRPO — 박사님 서버 전달본

AI 에이전트는 [AGENTS.md](AGENTS.md)와 [통합 인계서](docs/AI_AGENT_HANDOFF_KO.md)부터 읽는다. [Drive 자료 다운로드·분할 복원](docs/DOWNLOAD_AND_RESTORE_KO.md)을 마친 뒤 아래 설치 절차를 따른다.

처음 실행할 때는 [서버 실행 안내](docs/SERVER_RUNBOOK_KO.md)를 순서대로 따른다. GitHub 저장소는 [jiho-98/covermem-selector-grpo](https://github.com/jiho-98/covermem-selector-grpo)이며 코드/설정/문서를 제공한다. 아래 파일 목록의 데이터·어댑터·검증 로그는 **전체 ZIP에 포함**된다. Git clone에서는 `python3.10 scripts/import_assets.py --archive /실제/경로/CoverMem_selector_GRPO_AB_full.zip`으로 데이터와 어댑터를 가져온다.

2026-10-24. 대상 서버는 **A100 40GB × 16장**이다. 이 전달본은 **실험당 GPU 1장·프로세스 1개**로 실행한다. 여러 GPU를 묶는 DDP/FSDP/텐서 병렬 코드는 포함하지 않는다. A, B, 기준선을 서로 다른 GPU에 배치할 수 있다.

**A·B 구현과 CPU 검증을 완료했다. 실제 4B 모델의 A100 GPU 검증·학습·평가는 아직 실행하지 않았다.** GPU smoke가 통과해야 본 학습을 시작하는 실행 큐를 제공한다. 사용자는 이 전달본을 작성한 환경에서 GPU 작업을 시작하지 않도록 요청했다.

**현재 조건: 각 REPLACE에서 해당 훈련 대화의 전체 생성 질문으로 교체 후−전 utility를 계산하고, K=10개 출력을 생성한다.** 현재/미래 질문 비율은 사용하지 않는다. 새 schema는 `covermem.selector_grpo.v2`, 출력 경로는 `runs/allq_k10/<실험>`이며 이전 전달본의 checkpoint/큐와 분리한다. 근거 커버리지 0.7 + 근거 재현율 0.3이라는 bank utility 정의는 유지한다.

## 포함된 실험과 파일

| 실행 config | 동작 | 시작 매니저 |
|---|---|---|
| `configs/A_loo.json` | 완료된 GRPO 매니저 고정 → 훈련 대화를 처음부터 재생 → 선택기만 GRPO | 원래 LOO 최종 어댑터 |
| `configs/A_h1.json` | A와 동일한 학습 구조, 시작 매니저만 H1 | H1 최종 어댑터; 추가 선택 사항 |
| `configs/B_joint.json` | SFT에서 매니저 GRPO + 독립 선택기 GRPO를 같은 수집 배치 뒤에 각각 업데이트 | SFT 어댑터 |
| `configs/vanilla_control.json` | 같은 이식용 실행 엔진에서 원래 LOO를 학습하는 비교 기준선 | SFT 어댑터 |

**세 번째 연구 실험은 [별도 저장소](https://github.com/jiho-98/covermem-branch-grpo)에 구현했다.** 단일 매니저가 네 경로의 세션끝95% 조건을 충족한 다음 세션부터 분기 GRPO로 전환하며 기본 K=4,p=0.25다. 이 저장소의 A/B 및 H1 초기화 선택지와 다른 실험이다.

- `cmselect/`: 새 학습·평가 구현. `train.py`, `rollout.py`, `selection.py`, `backend.py`가 핵심이다.
- `adapters/{sft,loo,h1,loo_batch5}/`: 실제 완료된 어댑터 가중치와 PEFT config. 어댑터만으로 모델을 실행할 수 없으므로 아래의 베이스 모델이 필요하다.
- `ADAPTER_MANIFEST.json`: 가중치 SHA256, 완료 여부, 원래 실험 fingerprint.
- `assets/`: 훈련 원문 5대화·기존 생성 질문 3,338개·split·토크나이저·공식 평가 자료. 공식 평가는 학습 보상에 사용하지 않는다.
- `legacy/`: 기존 LOO·배치 LOO·H1 코드와 원래 환경/config/완료 기록. **보존용 원본**으로, 예전 절대 경로가 남아 있다. 이 폴더의 옛 `.sh`를 새 서버에서 그대로 실행하지 않는다. 이식하여 실행하는 진입점은 `scripts/`다. H1의 원래 lookahead 재학습 런처를 새 서버에 이식한 패키지는 아니다.
- `environment.lock.json`, `requirements.txt`, `environment.yml`, `scripts/`: 설치·검증·실행·재개 파일.
- `METHOD_AND_CONFIG_KO.md`: 학습 신호, 마스크, 기본값, 비교 조건 및 비용.
- `validation/`: CPU 검증 기록. GPU 결과를 의미하지 않는다.

## 1. 환경 설치

Linux x86_64, Python 3.10을 사용한다. conda가 있으면 다음과 같이 Python을 준비할 수 있다. 이미 Python 3.10이 있으면 conda 단계는 생략한다.

```bash
cd /원하는/경로/covermem_selector
conda env create -f environment.yml
conda activate covermem-selector
bash scripts/setup_env.sh
```

설치 스크립트는 이 폴더 안의 `.venv`만 사용한다. 핵심 버전은 기존 실행 서버에서 확인한 PyTorch 2.11.0, Transformers 5.8.0, PEFT 0.19.1, Accelerate 1.14.0이다. 나머지는 `environment.lock.json`에 고정했다. 기존 vLLM 0.22.0/verl v0.7.1 정보도 보존했지만 **새 실행 엔진은 Transformers/PEFT이며 vLLM/verl 설치가 필요하지 않다.**

기본 PyTorch 설치는 공식 `cu130` wheel을 사용한다. 서버에서 먼저 `nvidia-smi`로 GPU와 드라이버 상태를 확인한다. 드라이버가 이 CUDA 빌드를 지원해야 한다. 시스템의 `nvcc` 버전만으로 판단하지 않는다. 서버 관리자 정책에 맞는 공식 PyTorch 2.11.0 wheel index가 필요하면 설치 전에 `COVERMEM_TORCH_INDEX_URL`을 지정한다. 코드가 드라이버를 설치하거나 변경하지 않는다. 실제 torch/CUDA 빌드가 실험 fingerprint에 기록되므로 빌드를 바꾼 채 기존 체크포인트를 재개할 수 없다.

## 2. 베이스 모델 준비

```bash
bash scripts/download_model.sh
```

- 모델: `Qwen/Qwen3-4B-Instruct-2507`
- 고정 revision: `cdbee75f17c01a7cc42f958dc650907174af0554`
- 기본 위치: `models/Qwen3-4B-Instruct-2507`

베이스 모델은 전송 용량을 줄이기 위해 압축 파일에 포함하지 않았다. 인터넷이 없는 서버라면 위 revision을 다른 곳에서 받아 해당 폴더로 복사한다. 이미 있는 모델을 쓰려면 각 config의 `paths.base_model`만 실제 경로로 지정한다. **어댑터 config에 남은 원래 서버의 `base_model_name_or_path`는 읽어 모델을 찾는 데 사용하지 않는다.** 새 loader는 config의 베이스 모델을 먼저 로드하고 그 위에 어댑터를 연결한다.

JSON 내부의 상대 경로는 모두 이 패키지 루트를 기준으로 해석한다. 작업자 이름이나 `/home/kimjh` 경로는 새 실행 코드에 필요하지 않다. 각 실험의 `paths.output`은 반드시 서로 달라야 한다.

## 3. GPU 없이 먼저 확인

```bash
bash scripts/check_cpu.sh configs/A_loo.json
bash scripts/preflight.sh configs/A_loo.json
bash scripts/preflight.sh configs/B_joint.json
```

`check_cpu.sh`는 작은 무작위 Qwen 모델의 CPU 테스트와 데이터/어댑터 무결성을 확인한다. `preflight.sh`는 실제 베이스 모델 파일, 설치 버전, 토크나이저, 데이터와 어댑터를 확인하고 fingerprint를 만든다. 둘 다 GPU를 숨긴 상태에서 실행한다.

이번 전체 질문 보상 버전은 로컬 PyTorch 2.12 + Transformers 5.8 + PEFT 0.19.1에서 학습 관련 CPU 테스트 21개와 GitHub 입력 가져오기 테스트 4개, 총 25개를 통과했다. 최초 버전은 기존 서버의 PyTorch 2.11 환경에서도 검증했으며, 그 기록은 이전 전달본과 함께 보존했다. 두 버전의 검증 기록을 구분하며, 세부 결과는 `validation/VALIDATION_KO.md`를 참고한다.

## 4. A100 40GB에서 실행

40GB 메모리를 고려하여 BF16, gradient checkpointing, 응답 logits 128토큰 단위 계산, 순차 마이크로배치를 사용한다. 이 설정으로 실제로 메모리에 들어가는지는 **박사님 서버의 첫 GPU smoke에서 측정해야 한다.** OOM이면 자동으로 뱅크 예산이나 모델 정밀도, 질문/후보 수를 줄이지 않고 해당 단계를 실패 처리한다.

스케줄러가 없는 전용 서버에서, 각 GPU 사용 권한을 확보한 뒤 별도 터미널에서 실행한다.

```bash
# 터미널 1: GPU 0 — A
CUDA_VISIBLE_DEVICES=0 bash scripts/run.sh configs/A_loo.json orchestration/allq_k10/A_loo

# 터미널 2: GPU 1 — B
CUDA_VISIBLE_DEVICES=1 bash scripts/run.sh configs/B_joint.json orchestration/allq_k10/B_joint

# 터미널 3: GPU 2 — B의 같은 엔진 기준선
CUDA_VISIBLE_DEVICES=2 bash scripts/run.sh configs/vanilla_control.json orchestration/allq_k10/vanilla
```

위 GPU 번호는 전용 서버 예시다. Slurm이 GPU를 할당했다면 그 `CUDA_VISIBLE_DEVICES`를 그대로 사용한다. 각 실행에 한 장만 노출해야 하며 `torchrun --nproc_per_node=16`으로 실행하지 않는다. 16장을 자동으로 점유하는 스크립트는 없다.

실행 순서는 다음과 같다.

- **A:** GPU smoke → 고정된 시작 매니저의 동일 엔진 평가 → 선택기 학습 → 완료 체크포인트 평가.
- **B/vanilla:** GPU smoke → 학습 → 완료 체크포인트 평가.

GPU smoke는 실제 크기의 거의 찬 뱅크를 사용해 생성, backward, 어댑터 분리, 저장/재로드를 검사한다. optimizer 연결 검증에는 합성 advantage를 사용하며, 생성된 진단 어댑터는 본 학습에 사용하지 않는다. `runs/allq_k10/<실험>/smoke/SMOKE_COMPLETE.json`에 GPU 모델·CUDA 빌드·관측한 최대 메모리가 기록된다. 이는 모든 가능한 응답 길이의 메모리 상한을 보장하는 검사는 아니다.

## 5. Slurm 서버인 경우

Slurm 사용 여부와 partition/account/최대 wall time은 아직 전달받지 않았다. 서버 설정을 임의로 가정하지 않는다. **기존 호환 할당이 있으면 그 안에서 `scripts/run.sh`를 먼저 사용한다.** 새 할당이 필요할 때만 서버 관리자가 제공한 자원 옵션으로 `scripts/slurm_entry.sh`를 제출한다.

```bash
# 아래 값은 실제 서버의 허용 값으로 바꾼다. 코드를 전달한 환경에서는 제출하지 않았다.
sbatch --partition=<실제_partition> --gres=gpu:a100:1 \
  --cpus-per-task=<허용_CPU수> --mem=<허용_RAM> --time=<허용_최대시간> \
  scripts/slurm_entry.sh configs/A_loo.json orchestration/allq_k10/A_loo
```

필요하면 config의 `execution.require_slurm`을 `true`로 지정한다. Slurm batch 작업은 supervisor가 소유하며 smoke/학습/평가는 자식 단계다. 단계 실패 때문에 즉시 `scancel`하고 다시 제출하지 않는다.

## 6. 진행 확인·중단·재개

```bash
.venv/bin/python scripts/queue.py status --root orchestration/allq_k10/A_loo
tail -f orchestration/allq_k10/A_loo/logs/A_loo_train.log

# 해당 단계만 중단: 실제 stage ID는 status 또는 queue.json에서 확인
.venv/bin/python scripts/queue.py cancel --root orchestration/allq_k10/A_loo --id A_loo_train

# 원인을 해결한 뒤 같은 단계 재개
.venv/bin/python scripts/queue.py resume --root orchestration/allq_k10/A_loo --id A_loo_train
```

supervisor는 큐를 계속 읽고, 검증된 해당 단계의 PID·시작 시각·소유 표식으로만 중단한다. 다른 실험이나 supervisor에 프로세스 그룹 신호를 보내지 않는다. 실패/중단 시 기본 1,800초의 수리 대기 시간을 유지하며 자동 무한 재시도하지 않는다. 대기 시간은 큐의 `repair_grace_seconds`로 관리한다. 대기 종료 시 실패 상태로 종료하고, 완료된 모든 단계가 끝나면 정상 종료한다. 장기 유휴 예약은 만들지 않는다.

checkpoint에는 매니저·선택기 어댑터, 각 optimizer, RNG, 대화별 네 뱅크, 세션 cursor, 배치 cursor, 업데이트 수를 함께 저장한다. 새 디렉토리를 완전히 쓴 다음 `COMMITTED.json`과 원자적 rename으로 확정한다. **중간 배치에서 끊기면 마지막 확정 배치로 돌아가 해당 배치를 다시 수집**한다. 턴 단위 중간 복구를 구현한 것은 아니다. 최신 두 checkpoint와 모든 배치의 trace/metrics를 남긴다.

프로세스 자체가 종료됐다면 같은 `scripts/run.sh` 명령으로 supervisor를 다시 시작한다. 실패 상태 단계는 위 `resume` 요청이 필요하다. 과학적 조건·소스·패키지 버전이 달라지면 이어붙이지 않고 새 출력 디렉토리를 사용한다. GPU 실행 중 소스를 수정한다고 이미 실행 중인 Python이 바뀌지는 않는다.

실행 큐에 다른 승인된 실험을 이어 붙이려면:

```bash
.venv/bin/python scripts/queue.py append --root orchestration/allq_k10/A_loo --config configs/B_joint.json
```

이미 시작한 단계의 명령은 수정할 수 없다. 아직 시작하지 않은 단계는 `remove --id <ID>`로 제거할 수 있지만, 그 단계를 의존하는 후속 단계부터 제거해야 한다. 현재 할당의 GPU 종류·메모리·남은 wall time과 새 실험의 호환성은 운영자가 확인한다.

## 7. 결과 읽기와 전달 범위

- 학습 완료: `runs/allq_k10/<실험>/TRAIN_COMPLETE.json`.
- 최종 어댑터: 완료 marker가 가리키는 checkpoint의 `adapters/manager/`, `adapters/selector/`.
- trace와 각 국소 group의 점수/advantage: `runs/allq_k10/<실험>/batches/batch_XXXXXX/traces.json`.
- 전체 질문에 대한 획득/손실 문항 수와 점수상 순이익도 같은 trace의 각 후보에 기록한다.
- 매니저/선택기 업데이트 수와 loss: 같은 폴더의 `metrics.json`.
- 5대화·729문항 근거 커버리지/재현율/적중률: `runs/allq_k10/<실험>/evaluation/EVAL_COMPLETE.json`.
- A의 시작 매니저 비교값: `evaluation_initial_manager/EVAL_COMPLETE.json`.

평가는 학습 없이 전체 대화를 다시 진행한다. REPLACE마다 선택기를 **한 번 greedy로 호출**하며, K개를 CPU 정답으로 비교하지 않는다. 공식 근거는 최종 뱅크의 지표 계산에만 사용한다. QA 답변 모델을 호출하지 않으므로 EM/F1이나 답변 정확도 결과는 생성하지 않는다.

새 실행 엔진은 과거 vLLM과 수치적으로 동일한 경로를 보장하지 않는다. 특히 과거 평가의 JSON 제약 decoding을 새 HF 평가에서는 사용하지 않는다. 따라서 과거 547/555 등의 수치에 새 결과만 붙여 우위를 판단하지 말고, 제공한 동일 엔진 control과 비교한다. 각 실험은 독립 seed 반복 전 단일 실행 결과다.
