# Google Drive 자료 다운로드와 분할 ZIP 복원

2026-10-24 전달본. A/B와 세 번째 실험의 어댑터·데이터는 Drive, 최신 코드·사용설명서는 GitHub에서 받는다. **권장 경로는 최신 Git checkout + 복원한 ZIP에서 입력만 가져오기**다. Drive ZIP은 최초 배포 코드 스냅샷이므로 이후 추가된 에이전트 안내 문서는 GitHub에서 읽는다.

## 1. 다운로드 위치

- [01_selector_GRPO_AB](https://drive.google.com/drive/folders/1U-sRqGiPvWDK5m0ozJzIaQNwMWuIyWFA): A/B 자료
- [02_branch_GRPO](https://drive.google.com/drive/folders/1esQN9iQvEdGWxDVirHcLZPcfAQYXIWqq): 세 번째 실험 자료

2026-10-24 확인: GitHub 두 저장소는 Public이고 아래 A/B·세 번째 실험 하위 폴더는 모두 링크가 있는 모든 사용자에게 뷰어로 공개되어 있다. 별도 초대 없이 각 하위 폴더 링크로 접근한다. 상위 전달 폴더는 비공개 상태이므로 다운로드 진입점으로 사용하지 않는다. 새 권한 요청이 보이면 올바른 하위 폴더 링크인지 먼저 확인한다.

각 하위 폴더를 **파일명 그대로 전부** 내려받는다. Drive가 다운로드용 ZIP으로 묶어주면 먼저 그 ZIP을 푼다. `.part001` 파일 하나를 압축 해제하지 않는다. 파일명에 `(1)` 등이 붙은 중복 다운로드본과 원래 조각을 섞지 않는다.

| 하위 폴더 | ZIP 조각 | 기타 파일 | 전체 파일 수 |
|---|---:|---|---:|
| `01_selector_GRPO_AB` | `.part001`부터 `.part006`까지 6개 | 아래 공통 파일 5개 | 11 |
| `02_branch_GRPO` | `.part001`, `.part002` 2개 | 아래 공통 파일 5개 | 7 |

공통 파일은 `REASSEMBLE.py`, `PARTS_MANIFEST.json`, `SHA256SUMS.parts.txt`, `SHA256SUMS.txt`, `README_DOWNLOAD_KO.txt`다. 두 폴더의 동명 파일을 한 폴더에 합치지 않는다. 연결 도구의 파일당 100MiB 제한 때문에 최대 90MiB로 나눈 것이며 ZIP 내용은 바꾸지 않았다.

## 2. 원본 ZIP 복원

Python 3 표준 라이브러리만 필요하며 GPU나 학습 환경 설치가 필요 없다. 두 하위 폴더가 들어 있는 다운로드 디렉토리에서 실행한다.

```bash
cd 01_selector_GRPO_AB
python3 REASSEMBLE.py

cd ../02_branch_GRPO
python3 REASSEMBLE.py
cd ..
```

스크립트는 각 조각의 크기/SHA256을 검사한 후 순서대로 연결하고, 완성된 ZIP의 크기/SHA256도 검사한다. 성공하면 `Verified ZIP:` 또는 이미 정상 복원된 경우 `Already verified:`가 출력된다. 기존 ZIP이 다르면 덮어쓰지 않는다. `.assembling` 파일이 남아 있으면 이전 복원 프로세스가 끝났는지 확인하고 그 임시 파일만 따로 보관한 뒤 재시도한다. 원본 조각은 보존한다.

| 복원 파일 | 크기(bytes) | SHA256 |
|---|---:|---|
| `CoverMem_selector_GRPO_AB_full.zip` | 496567084 | `2c76f7ed1eb34cbd2c3c84fb9f351c54ac855315d77672201037aa2a7d12c3a6` |
| `CoverMem_branch_GRPO_full.zip` | 129009799 | `c61235c2e92a1fb5b9ce03b0253061d1472bf181a95b392eb3e09722a565759e` |

누락/불일치가 나오면 해당 조각을 다시 받는다. 검증을 우회하거나 서로 다른 전달본의 조각을 섞지 않는다. 기존 `SHA256SUMS.txt`에는 이번 Drive에 없는 code-only ZIP 항목도 있으므로 무조건 `sha256sum -c SHA256SUMS.txt`를 실행하면 누락이 표시될 수 있다. `REASSEMBLE.py`가 실제 전달된 full ZIP과 조각을 검증한다.

## 3. 최신 Git 코드에 입력만 가져오기 — 권장

아래 경로 두 개를 실제 절대 경로로 바꾼다. `CM_INPUT_ROOT`는 두 하위 폴더가 있는 위치, `CM_WORK`는 새 checkout을 둘 위치다. 새 터미널에서는 변수를 다시 설정한다. 명령은 Bash 기준이다. Python 3.10이 없으면 각 저장소의 `environment.yml`과 [서버 안내서](SERVER_RUNBOOK_KO.md)에 따라 먼저 준비한다.

```bash
CM_INPUT_ROOT=/absolute/path/to/downloaded_delivery
CM_WORK=/absolute/path/to/covermem_work
mkdir -p "$CM_WORK"
cd "$CM_WORK"

git clone https://github.com/jiho-98/covermem-selector-grpo.git
git clone https://github.com/jiho-98/covermem-branch-grpo.git

cd "$CM_WORK/covermem-selector-grpo"
python3.10 scripts/import_assets.py --archive "$CM_INPUT_ROOT/01_selector_GRPO_AB/CoverMem_selector_GRPO_AB_full.zip"
python3.10 scripts/import_assets.py --check-only

cd "$CM_WORK/covermem-branch-grpo"
python3.10 scripts/import_assets.py --archive "$CM_INPUT_ROOT/02_branch_GRPO/CoverMem_branch_GRPO_full.zip"
python3.10 scripts/import_assets.py --check-only
```

A/B는 `inputs_verified`, `files: 19`, 세 번째는 `files: 13`을 확인한다. 가져오기는 `INPUTS.sha256.json`에 지정된 입력만 처리하며 ZIP의 코드/문서가 checkout을 덮어쓰지 않는다. 기존 입력이 다르면 실패하므로 새 디렉토리에서 준비한다. 이미 실행 중인 checkout에서 `git pull`하거나 입력을 교체하지 않는다.

A/B에는 SFT·LOO·H1·배치 LOO 어댑터가, 세 번째에는 SFT 어댑터가 포함된다. 베이스 모델은 ZIP에 없다. 고정 revision 다운로드와 환경 설치는 [서버 안내서](SERVER_RUNBOOK_KO.md), 전체 실행 계획은 [AI 에이전트 인계서](AI_AGENT_HANDOFF_KO.md)를 따른다.

## 4. GitHub 없이 ZIP으로 실행하는 경우

접근 가능한 최신 문서를 먼저 읽고, 복원한 각 ZIP을 새로운 디렉토리에 푼다. 기존 실행 폴더 위에 덮어쓰지 않는다.

```bash
# 두 하위 폴더가 있는 다운로드 디렉토리에서
cd 01_selector_GRPO_AB
unzip CoverMem_selector_GRPO_AB_full.zip
# 코드 루트: 이 폴더 아래 covermem_selector/

cd ../02_branch_GRPO
unzip CoverMem_branch_GRPO_full.zip
# 코드 루트: 이 폴더 아래 covermem_branch/
```

ZIP에는 초기 코드와 설치 안내가 들어 있지만 이 인계서와 루트 `AGENTS.md`는 나중에 GitHub에 추가되었다. ZIP 실행은 해당 archive SHA256을 실행 버전으로 기록하고, Git 실행은 실제 commit과 working-tree 변경 상태를 기록한다. 코드가 같다는 이유로 실행 버전 기록을 생략하지 않는다.

## 5. 전달 당시 확인한 범위

로컬에서 두 ZIP의 실제 복원과 원본 SHA256 일치를 검증했다. Drive의 18개 파일 이름·크기·부모 폴더와 두 `PARTS_MANIFEST.json`의 내용을 readback했다. 원격 조각 전체를 재다운로드하여 byte 검증한 것은 아니므로 수신 서버에서도 `REASSEMBLE.py`를 실행한다. 다운로드/복원은 GPU 실행과 무관하다.
