# 행동 직전 bank + 다른 경로 잔여 이득 baseline 실험

2026-10-10 사용자 승인으로 새 실험을 시작한다. 기존 누적 GRPO를 재개하는 실험이 아니라,
동일한 RAW SFT checkpoint에서 초기화하여 변경된 credit assignment로 한 epoch 학습한다.

## 유지한 조건

- Qwen3-4B-Instruct-2507와 완료된 RAW SFT adapter.
- 원문 내용 43,500자 예산; frozen prompt/observation/action/executor/thought 출력.
- 동일 학습 대화 5개, 142세션, 세션별 중간/끝 평가. 각 cutoff까지의 누적 생성 질문 풀.
- 네 경로의 bank 계승, 세션 내 정책 고정, 세션당 optimizer step 최대 1회.
- 0.7 Complete + 0.3 Recall; 중간/끝 가중치 각각 0.5.
- 기존 actor, optimizer, lr 5e-6, KL 0.02, invalid min(A,-0.5), response-token loss 가중치.
- 원문 턴별 생성 및 실제 생성 token/old logprob 학습. 질문은 manager 입력에 포함하지 않는다.

## 변경한 조건

각 출력 직전 bank를 평가 시점의 누적 질문으로 CPU 채점한다.
u = 해당 출력 이후 평가 시점까지의 coverage 증가량이며, 앞절반은 0.5 u_mid + 0.5 u_end,
뒤절반은 0.5 u_end를 사용한다. 같은 원문 턴에서 다른 세 경로 최초 시도의 u 평균을 빼고,
고정 s=0.008103060640792133으로 나눈다. 그룹 표준편차 정규화나 ADD 보조 패널티는 없다.

배율은 이전 **학습** 기록의 유효 출력 계수를 기준으로, 세션 내 전체 response-token 분모를
사용한 RMS로 사전에 고정했다. 새 학습 중 재추정하지 않는다. 무효 출력에는 나눈 뒤 기존
min(A,-0.5)를 적용한다. 원문 턴 기준 정렬이며 retry의 출력 순번을 peer 매칭에 사용하지 않는다.

`action_credit.py`는 실제 실행 액션을 초기 bank부터 재생하고 trace의 before/after hash 및
중간/끝 bank를 확인한다. 별도 QG/QA 호출이나 가상 액션 학습 샘플은 없다.

## 검증

- OR/AND 충분근거 계산, 미래 cutoff 거부, peer 누락 거부, retry/self-exclusion, 동점 검증.
- 기존 142세션·13,778출력의 bank 재생 및 독립 CPU 계산과 대조: 최대 오차 약 1.3e-13.
- CPU supervisor fixture에서 기존 child 중단 → 새 학습 → 새 평가 → 기존 작업 재개 순서와
  동일 supervisor 유지 검증.
- 실제 H200 2장 내부 GPU smoke에서 모델 로드, 장문, actor update, adapter 재로드/logprob 검증 후
  본학습 시작. smoke의 가중치는 본학습에 사용하지 않는다.

## 실행 순서

기존 job 62130의 supervisor를 유지하며 다음 child stage를 우선한다.

1. 중단한 EM 학습의 committed checkpoint 검증·별도 보존.
2. 새 조건 GPU smoke.
3. 새 조건 SFT 초기화 GRPO.
4. 동일 held-out 5대화·729문항의 Active-only RAW Complete/Recall/Hit 및 실제 저장량 평가.
5. 기존 EM 학습 checkpoint resume → 기존 EM 평가 → 기존 대기 작업 2개.

새 실험에는 QA 평가를 추가하지 않았다. 현재 요청은 근거지표까지이다.
`runs/loo_v1`의 결과와 fingerprint를 기존 결과와 분리한다.
단계 오류 시 기존 supervisor가 할당을 보존하여 수리할 수 있으며, 자동 무한 재시도는 하지 않는다.
wall-time·노드 장애·관리자 취소 등 스케줄러 경계는 그대로 적용된다.

매 세션 `credit_assignments.json`에 원문 턴, retry, 직전 점수, u, peer 평균, residual,
최종 계수와 사용량별 RMS를 기록한다. 실제 actor의 KL, gradient norm, clipping 통계도 기존대로 남긴다.
