# 검증 상태 — 2026-10-24 전체 질문 보상·K=10 수정본

**GPU 학습·추론·smoke·평가를 실행하지 않았다.** A100 40GB에서의 실제 메모리 여유, 처리 속도, 최종 성능은 아직 미측정이다.

GitHub에는 이 검증 설명을 포함하며 원본 검증 로그는 전체 전달 ZIP에 포함한다. GitHub 전달 준비에서 입력 가져오기 테스트 4개를 추가한 뒤 **전체 CPU 테스트 25개를 통과**했다 (`cpu_tests_github_handoff.log`, 동일 로컬 torch2.12 환경). 기존 학습 코드의 21개에, 해시가 맞는 입력만 가져오기/코드 덮어쓰기 방지/변조 입력·기존 변경 파일·경로 이탈 거절 검사가 추가되었다. README와 실행 안내의 shell 블록 문법 및 상대 링크도 검사했다.

Git 추적 파일만 별도 임시 폴더로 복사한 뒤 실제 전달 ZIP의 19개 입력을 가져오고, A/B 각각의 입력 preflight에 통과했다 (`github_checkout_import.log`). README가 ZIP 내용으로 덮어써지지 않았음도 확인했다. CPU/파일 검증이며 전체 모델 로드나 GPU 실행은 아니다.

## 현재 수정본에서 통과

- 로컬 CPU: PyTorch 2.12.0, Transformers 5.8.0, PEFT 0.19.1에서 **21 tests OK** (`cpu_tests_allq_k10.log`). CUDA_VISIBLE_DEVICES를 비웠으며 실제 GPU를 초기화하지 않았다는 assertion 포함.
- 전체 질문을 동일 가중치로 평가하고 K=10을 사용하는 A/B 및 기준선 config 검증. 이전 schema·시간 비율 필드·다른 K를 거절하는 검사 포함.
- 실제 입력: 훈련 5대화·142세션·284 cutoffs·3,338 질문. 대화별 675/658/703/590/712. 새 A_loo config로 입력 preflight 통과 (`assets_A_allq_k10.json`). 보상은 매번 해당 대화의 전체 질문을 사용하며 다른 대화 질문을 합치지 않는다.
- SFT/원래 LOO/H1/배치 LOO의 실제 어댑터 가중치 SHA256과 호환 config 확인.
- Python compile 통과. 기존 manager reward·실행 supervisor·환경 설치 shell은 이번 수정에서 변경하지 않았다.

## 최초 전달본에서 통과한 기록

아래는 전체 질문 보상으로 변경하기 전 기록이다. 현재 수정본을 해당 환경에서 다시 실행했다고 해석하지 않는다. 기존 로그는 `history/temporal_mix/`에 보존했다.

- 로컬 PyTorch 2.12.0 및 기존 server3의 PyTorch **2.11.0**에서 각각 **18 tests OK** (`history/temporal_mix/cpu_tests.log`, `history/temporal_mix/cpu_tests_torch211.log`). checkpoint archive 복구 보완 후 9개 protocol 테스트도 통과했으며 이번 21개 테스트에 포함되어 다시 통과했다.
- 실행 supervisor: **2 tests OK** (`supervisor_tests.log`). 실패 후 owner 유지/재개/큐 추가, 대상 단계만 중단, 관련 없는 프로세스 생존, 완료 후 종료를 확인했다. 해당 코드 변경 없음.
- A_loo·A_h1·B의 입력 검증 (`history/temporal_mix/assets_*.json`). 입력 데이터/어댑터 변경 없음.
- 공식 평가 전처리: 원래 loader로 5대화·2,769턴·729 eligible questions 확인. CPU 전처리만 수행.
- 경로 이식: 패키지 전체를 원래 경로와 다른 임시 폴더로 복사한 뒤 입력 preflight 성공 (`history/temporal_mix/relocation_test.log`). 새 실행 코드/config에 `/home/kimjh` 절대 경로 없음.
- Python compile 및 모든 shell의 `bash -n` 통과.

## 핵심 테스트의 범위

1. REPLACE 상태별 K=10 local advantage 계산. 4×K를 한 group으로 섞지 않음.
2. 전체 질문의 교체 전후 차이 및 충분 근거의 OR/AND·부분 재현율. 앞쪽 질문 1개를 잃고 뒤쪽 질문 9개를 얻는 경우 전체 10개에 대한 +0.8을 확인. 질문 생성 시점을 바꿔도 동일 보상. 근거를 완전히 확보한 질문의 얻음/잃음/순증감과 부분 보존까지 포함한 utility gain/loss를 기록.
3. 점수가 더 낮더라도 미리 정한 샘플 0을 실행하여 oracle 실행 누출 방지.
4. 중복 후보 유지, 잘못된 ID 처리, 기존 대상 fallback, 빈 pool 처리.
5. ADD/SKIP 실행과 REPLACE에서만 선택기 호출. 매 REPLACE에 선택기 출력 정확히 10개 생성.
6. 실제 Qwen tokenizer 경계로 매니저 action loss 마스킹. 원래 응답과 실행 대상 분리.
7. 작은 실제 Qwen+PEFT 모델에서 A 선택기만 변경, B 각 loss별 해당 어댑터만 변경. 베이스와 reference 고정.
8. 실제 무작위 생성에서 수집한 log-probability와 teacher-forced 계산 일치.
9. stale/wrong probability guard, 동점 group 업데이트 생략.
10. 두 optimizer·두 어댑터 저장/재로드. B 전체 coordinator에서 중단/재개 후 파라미터와 최종 뱅크가 연속 실행과 정확히 일치.
11. 평가에서 REPLACE당 한 번만 선택기를 호출하고, 재개 시 이미 끝난 턴을 다시 생성하지 않음.

작은 모델 통합 테스트 일부는 유효한 메모리 액션을 강제한 fixture다. 그 테스트는 학습 plumbing 검증이며, 정책의 선택 성능이나 실제 on-policy 실험 결과가 아니다. 실제 샘플링 확률 검사는 별도 테스트로 분리했다.

## 미실행 또는 제한

- 실제 4B 모델을 GPU에 올리는 smoke와 학습/평가: 미실행. 박사님 서버의 첫 실행 큐에서 수행한다.
- 이번 전체 질문 보상 수정본의 PyTorch 2.11 재검증: 미실행. 설치 스크립트의 production pin은 2.11.0을 유지하며, 현재 수정본의 실제 CPU 검증은 로컬 2.12.0이다.
- 베이스 모델 shard 전체와 새 설치 환경을 한 번에 확인하는 full preflight: 이 전달본에서 미실행. 기존 모델/입력 파일을 합친 추가 원격 검증을 준비하던 중 평가 전처리 자료 복사가 자동 승인 검토에서 데이터/전송 대상 허용 범위가 불명확하다는 사유로 거절되어 중단했다. 우회 전송하지 않았다. 기존 서버에서 이미 완료한 CPU 코드 테스트와 로컬 입력 검증은 위와 같이 유효하다.
- 새 패키지의 `setup_env.sh`를 깨끗한 박사님 서버에서 설치 완료한 상태: 미검증. 설치 스크립트와 버전 고정 파일을 제공했으며 설치 후 CPU/full preflight/GPU smoke 순서로 검증한다.
- 다중 GPU 분산 학습: 미구현. 실험당 GPU 한 장. A100 40GB × 16장 중 여러 장을 각 독립 실험에 배치할 수 있다.
- 세 번째 연구 실험: 사용자 명세 대기. 임의의 제3안을 구현하거나 완료로 표시하지 않았다.

원래 vLLM과 새 HF 실행을 bitwise 재현한다고 주장하지 않는다. 비교용 vanilla와 A의 selector-off 평가를 같은 새 엔진으로 제공한다.
