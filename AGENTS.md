# CoverMem 실험 실행 에이전트 지침

이 저장소는 REPLACE 선택기 GRPO A/B 구현이다. A100 40GB 서버에서 설치·실행·평가 작업을 시작하기 전에 다음을 읽는다.

1. `docs/AI_AGENT_HANDOFF_KO.md`: A/B 및 별도 세 번째 실험의 통합 인계서, 실행 행렬, 불변 조건, 완료 기준.
2. `docs/DOWNLOAD_AND_RESTORE_KO.md`: Google Drive 분할 ZIP 복원, 입력 가져오기.
3. `docs/SERVER_RUNBOOK_KO.md`, `METHOD_AND_CONFIG_KO.md`, 실제 실행 config, `validation/VALIDATION_KO.md`.

- 사용자의 현재 요청 범위를 따른다. 문서 검토만 요청받으면 GPU를 시작하지 않고, 실행 요청 시 제공된 smoke→train/eval 큐를 사용한다. 이미 승인된 작업마다 반복 확인하지 않는다.
- 실험당 GPU 1장이다. 16GPU DDP/FSDP나 K개 prefix cache는 구현하지 않았다.
- A/B는 K=10, 해당 훈련 대화의 전체 생성 질문 순이익 보상이다. 별도 저장소의 K=4,p=0.25/95% 전환 규칙을 이 코드에 적용하지 않는다.
- A는 완료 매니저 고정/선택기 학습, B는 독립 매니저·선택기 LoRA/optimizer다. B의 REPLACE action-value loss 마스크와 실제 샘플0 실행을 유지한다.
- 모델·prompt·입력·reward·K·dtype·loss·학습량·평가를 실행 편의 때문에 임의 변경하지 않는다. 코드/조건/runtime 변경은 새 run/fingerprint로 구분하고 실행 중인 checkout을 갱신하지 않는다.
- Slurm은 기존 호환 allocation을 우선 재사용한다. outer supervisor와 child stage를 분리하고 reloadable queue를 사용한다. routine scancel/재제출, 광범위한 pkill/process-group signal을 하지 않는다.
- 중단은 해당 stage의 검증된 PID/시작 시각/소유 표식에 한정한다. 기존 worker 종료를 확인한 뒤 재개한다. 마지막 COMMITTED checkpoint의 optimizer/RNG/bank/cursor를 함께 복원한다.
- CPU/config 검증 후 해당 allocation 안에서 GPU smoke를 거친다. 완료 checkpoint에 평가를 연결하고 승인된 작업이 끝나면 자원을 반환한다. 실패 수리 대기는 유한하며 wall time/운영 정책을 준수한다.
- 기본 조건과 선택 사항을 구분한다. `legacy/`의 옛 런처를 실행하거나 세 번째 repo의 config를 이 repo 큐에 넣지 않는다.
- 준비/실행/수리/중단/학습완료/평가완료를 구분하여 실제 상태와 할당 유지 여부를 보고한다. 근거 커버리지는 QA 정답률이 아니다.
- GPU 메모리·속도·성능은 전달 시 미검증이다. 실제 서버의 검사/완료 marker/로그를 근거로 보고한다. 데이터·가중치·비밀값·긴 trace를 Git에 올리지 않는다.
