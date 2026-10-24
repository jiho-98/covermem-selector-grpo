# 박사님께 전달할 메시지

박사님, CoverMem의 REPLACE 대상 선택기를 학습하는 GRPO A/B 실험 코드와 실행 자료 전달드립니다.

- GitHub: https://github.com/jiho-98/covermem-selector-grpo
- 첨부: `CoverMem_selector_GRPO_AB_full.zip`, `SHA256SUMS.txt`
- 실행 안내: GitHub의 `docs/SERVER_RUNBOOK_KO.md`

GitHub에는 코드·환경 설정·config·사용설명서가 있고, ZIP에는 동일 코드와 훈련/평가 데이터, SFT·LOO·H1·배치 LOO 어댑터가 포함되어 있습니다. 베이스 모델은 안내된 고정 revision으로 별도 다운로드합니다. 저장소는 비공개이므로 GitHub 계정에 접근 권한을 부여한 뒤 clone할 수 있으며, GitHub 접근 전에도 ZIP만 풀어 실행 가능합니다.

실험 A는 기존 GRPO 학습이 완료된 LOO 매니저를 고정하고 훈련 대화를 처음부터 재생하면서, REPLACE에서만 별도 선택기 LoRA를 학습합니다. 실험 B는 SFT부터 매니저와 선택기를 각각 학습합니다. 매니저는 기존 LOO 보상을, 선택기는 각 REPLACE의 후보 비교 보상을 받으며 두 optimizer는 분리되어 있습니다.

선택기 보상은 매 REPLACE마다 해당 훈련 대화의 전체 생성 질문으로 계산한 **교체 후 점수 − 교체 전 점수**입니다. 현재/미래 질문의 7:3 가중치는 사용하지 않으며 K=10개 출력을 동일한 REPLACE 상태 안에서 상대 비교합니다. 기존 점수 정의인 근거 커버리지 0.7 + 근거 재현율 0.3은 유지합니다. 질문/정답/근거 라벨은 선택기 입력에 제공하지 않습니다.

대상 서버는 A100 40GB × 16장이지만, 현재 구현은 **실험당 GPU 1장**입니다. A와 B를 GPU 2장에 독립 배치할 수 있고, B 비교용 vanilla 기준선은 다른 GPU 또는 같은 GPU의 후속 작업으로 실행할 수 있습니다. 16장 분산 학습 코드는 아닙니다.

안내서 순서대로 입력 파일 검증 → Python 3.10 환경 설치 → 모델 다운로드 → CPU 검사/full preflight → A/B 실행을 부탁드립니다. 실행 큐가 GPU smoke를 먼저 검사한 다음 학습과 평가를 자동으로 이어갑니다. A에서는 시작 매니저의 selector-off 평가도 먼저 수행합니다.

현재 전체 질문/K=10 학습 관련 CPU 테스트 21개와 GitHub 입력 가져오기 테스트 4개, 총 25개를 통과했습니다. 실제 4B 모델을 A100 40GB에서 실행한 GPU smoke·학습·평가는 아직 하지 않았으므로 메모리 사용량·속도·성능은 서버 첫 실행에서 확인해야 합니다. OOM이나 오류가 나면 K/뱅크 예산/정밀도를 임의로 줄이지 말고, 실패 단계 로그와 환경 정보를 공유 부탁드립니다.

최종 전달받고 싶은 결과는 각 실험의 `TRAIN_COMPLETE.json`, `evaluation/EVAL_COMPLETE.json`, A의 `evaluation_initial_manager/EVAL_COMPLETE.json`, `smoke/SMOKE_COMPLETE.json`, 큐 상태/로그와 실행 commit·환경 정보입니다. 전체 완료 후에는 최종 checkpoint 및 배치별 trace/metrics도 함께 보존 부탁드립니다. 평가는 근거 커버리지·재현율·적중률이며 QA 정답률은 아닙니다.
