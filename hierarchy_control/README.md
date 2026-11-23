# CTBA 계층 제어: DreamerV3 상위 정책 + Betaflight Angle 모드

DreamerV3 월드모델로 **집단 추력 + roll/pitch 목표각 + yaw rate(CTBA)** 를 출력하는 정책을 학습합니다.

하위 제어는 FC(SpeedyBee F405, Betaflight 4.4.3)의 **Angle 모드**가 맡습니다. FC 안에서 다음 세 단계가 돕니다.
- 자세 추정기: Mahony, 100 Hz
- level 루프: 목표각을 rate setpoint로 바꿈
- rate 루프: 4 kHz

GateNet 마스크, StochGAN, 정보 디코더(informed DreamerV3)는 skydreamer 것을 그대로 씁니다.

```
카메라 → GateNet 마스크 ──────────┐
FC 자이로·가속도·자세 추정 (MSP) ─┼→ 인코더/RSSM → actor → (throttle, roll, pitch, yaw rate)
비행 계획 ────────────────────────┘                    100 Hz          │ MSP RC override
                                                                       ▼
                 Betaflight Angle 모드: 자세 추정(100 Hz) → level 루프 → rate PID(4 kHz) → 모터
```

수동 비행 로그(LOG3–7)가 모두 Angle 모드였습니다. 그래서 **식별한 데이터와 배포 모드가 같습니다.**

## 구성

| 경로 | 내용 |
|---|---|
| `sysid/` | 수동 비행 로그 식별. 결과는 `drone_geometry/flight_sysid.md`에 정리 |
| `sysid/fc_attitude.py` | Betaflight 자세 추정기 재현과 level 루프 식별 |
| `sysid/fit_angle_chain.py` | 로그의 스틱·throttle로 시뮬 전체 체인을 재생해 검증 |
| `ctba/params.py` | 플랜트 명목값(출처 표기)과 에피소드별 랜덤화 범위 |
| `ctba/plant.py` | CTBA 플랜트. 아래 표 참고 |
| `ctba/env.py` | 경주 환경. 트랙, 보상, 종료, 바닥 이륙, 마스크 렌더링은 skydreamer 것 |
| `ctba/camera.py` | Pi Camera v2(IMX219) 모드별 화각, 틸트, 프레임레이트, 정책 영상의 핀홀 K, 마스크 렌더링 |
| `ctba/airframe.py` | CAD 외곽점(프롭, 암, 셸, 받침대, 카메라), 무게중심, 카메라 위치 |
| `ctba/tasks.py` | 과제 이름(`c1_team44up` 등) → 환경 설정, 커리큘럼 단계 정의 |
| `ctba/expert.py` | CTBA 데모 제어기 (경로 추종 → throttle + 목표각 + yaw rate) |
| `ctba/embodied.py` | DreamerV3 어댑터 (suite `ctba`) |
| `scripts/` | DreamerV3 등록, 데모 수집, 커리큘럼 학습, 평가 |
| `scripts/camera_visibility.py`, `camera_views.py` | 카메라 모드별 게이트 가시성 비교와 화면 렌더 (`drone_geometry/camera_views.png`) |
| `tests/` | 플랜트를 실제 로그로 재현하는 검증, 환경 검증 |

`ctba/plant.py`가 모델링하는 것:
- FC 자세 추정기: Mahony kp 0.25(FC CLI에서 확인), 가속도는 0.9–1.1 g일 때만 사용, IMU 장착 기울기와 자이로 바이어스 포함
- level 루프: 게인 4.9 /s, 한계 55°
- rate 루프: roll/pitch 2차, yaw 1차, 각가속도 제한
- 추력 맵: 지연, PT1, 배터리 강하, 상승 손실
- 적분: 1 kHz

## 관측과 행동

- **행동:** `[-1, 1]^4`를 다음 명령으로 바꿉니다.
  - throttle: `(a0+1)/2`
  - roll, pitch 목표각: `a·55°`. Betaflight Angle 모드의 스틱 편향 그대로입니다. 시뮬은 FRD 기준이라 a2 > 0이 기수 올림입니다.
  - yaw rate: ±4 rad/s
- **관측:**

| 키 | 내용 |
|---|---|
| `mask` | 64×64 게이트 마스크. 카메라는 Pi Camera v2(IMX219) 1640×1232 모드, 64×64가 화각 63.2°×49.6°(캘리브레이션으로 확인한, 검은 테두리 없이 쓸 수 있는 최대 화각)를 덮음(`ctba/camera.py`). 틸트 30°(학습 25–35°). 25–30 fps(새 프레임 전에는 직전 마스크 유지). 40–70 ms 지연(실측 46–65 ms) |
| `rates` | 자이로 rad/s (IMU 좌표). 1 °/s 양자화, 노이즈 1–4 °/s |
| `acc` | 비추력. g 단위, 호버 시 (0, 0, −1). 25 Hz 필터, 노이즈 0.02–0.1 g, 바이어스 ±0.02 g |
| `att` | **FC의 자세 추정** roll, pitch rad (MSP_ATTITUDE, 0.1° 단위). FC가 "지금 몇 도라고 믿는지" |
| `flight_plan` | 비행 계획 |

- **디코더 대상(학습 때만):**
  - 위치, 속도, 실제 자세, 실제 rate, 실제 추력
  - **FC 추정 오차(roll, pitch)**
  - 플랜트 파라미터 30개, 지연·카메라 프레임레이트 3개
  - 월드 모델은 이 대상으로 "FC가 얼마나 틀려 있는지"를 추론하도록 학습합니다.
- **명령 지연:** 1–2 스텝(10–20 ms). **이미지 지연:** 4–7 스텝(40–70 ms, 실측 46–65 ms). 둘 다 에피소드마다 새로 뽑습니다.

### FC 자세 추정 오차를 다루는 방식

FC는 정책이 준 목표각을 **자기 추정** 기준으로 맞춥니다. 그래서 추정이 2° 틀리면 기체도 2° 틀어집니다.

시뮬은 실제 자세와 FC 추정을 둘 다 계산합니다. 추정기는 로그로 재현한 Betaflight 필터 그대로라서, 지속 가속 중 수평 기준이 끌려가는 오차가 물리적으로 생깁니다.

그 위에 남는 불확실성만 랜덤화합니다.
- 추정기 kp: 0.22–0.28 (FC 설정값 0.25 ±10%)
- 자이로 잔여 바이어스: ±0.5 °/s
- IMU 장착 기울기: ±1.5°
- 공중 시작 시 초기 오차: ±2°

이렇게 시뮬 비행에서 생기는 추정 오차는 p90 5°입니다. 로그에서 잰 끌림 크기(p90 3–6°)와 같은 규모입니다.

## 커리큘럼 (scripts/train.py)

| 단계 | 스텝 (누적) | 과제 | 바뀌는 것 | DreamerV3 |
|---|---|---|---|---|
| 1 | 0 → 6M | c1 완주 | 데모를 버퍼에 넣고 시작. 속도 상한 5 m/s, 무게중심 판정 창 0.60 (평가 0.45) | phase 1 |
| 2 | 6 → 8M | c2 속도 | 방의 속도 상한 4 m/s | phase 2 (batch_length 256) |
| 3 | 8 → 14M | c4 게이트 | + 바닥 하강 감점, 고도 여유 감점, 실제 게이트 면에서 CAD 기체 외곽 전체 판정, 학습 판정 창 0.45 | phase 2 |
| 4 | 14 → 20M | c4 | 마무리 미세조정 | phase 3 (actent 1e-5, lr 2e-6) |

c3(바닥 감점만 추가하는 단계)는 2026-11-19에 뺐습니다. 6.24M 평가(ce, 200회)에서 1랩 92%, 3랩 66%였고,
3랩 실패 68건 중 바닥 2건, 기체 외곽 이탈 1건, 게이트 충돌 62건이었습니다. 게이트 충돌은 c1·c2의 학습 판정 창 0.60과
평가 창 0.45의 차이 때문으로 보고, 바닥 감점이 이미 들어 있는 c4를 3M에서 6M으로 늘렸습니다.

단계 4는 2026-11-22에 16M에서 20M으로 늘렸습니다. 16M 학습의 평가 결과(ce, 3랩, 200회, seed 20000)는 다음과 같습니다.
- 단계 3: 0.5M 간격 스냅샷마다 36.5%에서 74% 사이를 오르내렸습니다.
- 단계 4: 스냅샷마다 올랐습니다(65 → 68.5 → 73 → 75.5 → 79.5%).
- 16M에서도 아직 오르는 중이었습니다.
16M에서 끝난 학습은 `./run.sh --resume`으로 단계 4를 이어서 20M까지 학습합니다.

**단계 점검 기준:** 단계가 끝날 때마다 `./run.sh --eval <스냅샷>`으로 평가합니다.
- 평가 조건은 `ce`(최종 조건, 바닥 이륙)이고 1랩·3랩 각 200회입니다.
- 같은 seed로 직전 단계와 비교합니다.

통과 기준:
- 단계 1 끝(6M): 시뮬 1랩 완주 30% 이상
- 단계 3 끝(14M): 1랩 60% 이상
- 최종 목표: 실제 판정 기준 90% 이상

기준에 못 미치거나 직전 단계보다 나빠지면, 그 단계의 직전 스냅샷(`snapshots/`, 0.5M마다 저장)으로 돌아가서 원인을 확인한 다음 진행합니다.

## StochGAN 파인튜닝 (scripts/finetune_stochgan.py)

커리큘럼은 깨끗한 렌더링 마스크로 학습합니다. 이 단계는 커리큘럼에서 가장 좋은 가중치를 가져와, 모든 렌더링 마스크를
StochGAN 생성기에 통과시킨 뒤(그다음 롤링셔터·침식, 논문 II-F 순서) 계속 학습합니다. 실제 GateNet 마스크의 결손·잡음에
월드 모델을 맞추는 것이 목적입니다.

| 단계 | 스텝 | 과제 | DreamerV3 |
|---|---|---|---|
| g | 0 → 2M (새 run) | c4g = c4 + StochGAN 마스크 | phase 3 (actent 1e-5, lr 2e-6) |

- **생성기:** 논문 부록 B 구조(ngf 32, residual 6개, InstanceNorm, 입력 = 마스크 + 잡음 채널)를 JAX로 구현했습니다
  (`skydreamer/gan.py`). 팀원이 PyTorch로 학습한 렌더링→실제 방향 생성기를 npz로 바꿔 씁니다.
- **시작 가중치:** `--run.from_checkpoint`로 가중치·옵티마이저 상태를 불러오고, 스텝은 새 run에서 0부터 셉니다.
  replay는 기본적으로 비워서 시작합니다(이전 파인튜닝 run 6~7k와 같은 방식). `--replay-steps N`이면 원래 run의
  최근 N 스텝(깨끗한 마스크)을 미리 넣습니다.
- **2M 스텝인 이유:** 단계 4와 같은 길이입니다. 단계 4는 처음 2M 동안 올랐고(16M 3랩 79.5%), 20M에서는
  61.5%로 떨어졌습니다. 그래서 0.5M 스냅샷을 `ceg`로 평가해 가장 좋은 것을 고릅니다.
- **평가:** `--stage ce`(깨끗한 마스크)와 `--stage ceg`(StochGAN 마스크) 둘 다 봅니다. 실기체에 더 가까운 쪽은 `ceg`입니다.

```
python scripts/convert_stochgan.py G_rendered2real.pth --out stochgan_G.npz [--key netG_A] [--check]
STOCHGAN_WEIGHTS=$PWD/stochgan_G.npz ./run.sh --finetune-stochgan logdir/<run>/snapshots/step_<N>
./run.sh --finetune-stochgan <같은 소스> --ft-run logdir/ctba-stochgan-<날짜>      # 중단된 run 이어 하기
STOCHGAN_WEIGHTS=... python scripts/evaluate.py --logdir <스냅샷> --stage ceg --laps 3 --episodes 200
```

## 실행

skydreamer 저장소 안의 `hierarchy_control/`에서 실행합니다. skydreamer 경로는 상위 폴더가 기본값입니다.

```
./run.sh --setup
./run.sh --smoke
./run.sh --collect-demos     # 약 800 에피소드, CPU 몇 분
./run.sh                     # 20M 스텝
./run.sh --resume
./run.sh --eval logdir/ctba-team44up-<날짜>
```

- **학습 시간 추정:**
  - A5000 실측 속도(65 steps/s)로 20M 스텝에 약 85시간입니다(16M 이후 4M은 약 17시간).
  - L40은 1.3–1.6배 빠를 것으로 보고 약 45–55시간으로 추정합니다.
  - 다만 CTBA 환경은 아직 서버에서 속도를 재지 않았습니다.
- **RAM:** replay 10M 스텝이면 약 46 GB가 필요합니다. skydreamer와 같습니다.

## 실기체 연동 시 맞출 것 (학습과 같아야 함)

1. **FC 설정은 수동 비행 로그와 같게 유지합니다.** 예외는 yaw rate 하나입니다.
   - Angle 모드, levelPID 50/50/75를 유지합니다.
   - **`angle_limit`은 반드시 55**여야 합니다. 정책의 a1, a2 = ±1이 ±55°로 학습되기 때문입니다.
   - `imu_dcm_kp` 2500을 유지합니다(CLI로 확인).
   - PID와 feedforward도 그대로 둡니다.
   - 시뮬은 이 설정으로 식별되었습니다.
2. **yaw rate만 직선으로 바꿉니다.** yaw 중앙 감도 = 최대 229 °/s, expo 0으로 설정합니다. Angle 모드에서도 yaw는 rate 명령이기 때문입니다.
3. **MSP RC override로 명령을 넣습니다.**
   - roll = 1500 + 500·a1
   - **pitch = 1500 − 500·a2.** Betaflight는 스틱을 앞으로 밀면 기수가 내려가고, 시뮬은 FRD라 a2 > 0이 기수 올림입니다. 로그로 확인한 부호입니다.
   - throttle = 1000 + 1000·thr
   - yaw = 1500 + 500·a3입니다. a3 > 0이면 기수가 오른쪽으로 돕니다. Receiver 탭에서 오른쪽 스틱이 1500보다 큰 값으로 확인했습니다(2026-11-16).
   - 넘겨받기는 조종기 스위치로 override만 끄면 됩니다. 모드가 바뀌지 않아서 부드럽습니다.
   - Betaflight 빌드에 `MSP_OVERRIDE`를 포함해야 합니다. 연동 전에 펌웨어를 다시 올릴 예정입니다.
4. **센서 변환:**
   - MSP_RAW_IMU 자이로(정수 °/s)와 가속도(512 LSB/g)를 FRD로 바꿉니다. 보드 좌표 (x, y, z) → (x, −y, −z)입니다.
   - MSP_ATTITUDE roll은 그대로 쓰고, pitch는 부호를 뒤집습니다.
   - 확인 방법: 평평하게 두면 `acc` = (0, 0, −1), `att` = (0, 0) 근처여야 합니다.
5. **카메라와 GateNet:** Pi Camera v2를 **1640×1232 @ 30 fps** 모드로 받습니다. skydreamer의 공칭 K(104°)는 쓰지 않습니다.
   - GateNet은 **f=4, 입력 240×320**(4:3)으로 확정했습니다(2026-11-17). Orin Nano FP16 기준 2.1 ms(p99 4.6 ms)입니다.
   - 순서: ISP(nvvidconv)에서 320×240으로 축소 → `ctba.camera.gatenet_K()`로 왜곡 보정 → GateNet → 마스크를 64×64로 축소. 두 핀홀이 같은 화각을 덮어서, 마지막 축소 결과가 정확히 `policy_K()`가 됩니다.
   - 축소를 CPU(OpenCV)로 하면 12–21 ms 걸립니다. 반드시 하드웨어로 축소합니다.
   - 틸트는 **30°**입니다. CAD는 40°라서 마운트를 바꿔야 합니다.
   - 카메라 캘리브레이션(체커보드)으로 실제 화각을 확인해야 합니다.
6. **첫 정책 비행은 테더 비행으로 합니다.** 그 전에 프롭을 빼고 벤치에서 명령 부호, override 스위치, failsafe를 확인합니다.

## 검증 결과 (로컬, 2026-11-16)

- **FC 추정기 재현:** 재현한 자세로 Angle 모드 setpoint가 샘플 단위로 맞습니다. R²는 0.984–0.998입니다(LOG3–7, 축별).
  - 게인 4.9, 한계 55°, kp 0.25로, Betaflight 기본값과 일치합니다.
  - 시뮬의 JAX 추정기는 numpy 재현과 0.3° 이내로 같습니다.
- **Angle 전체 체인:** 로그의 스틱과 throttle을 그대로 넣었을 때 결과입니다.
  - 시뮬 자세 R²는 0.995–0.999, rms 0.2–0.6°입니다. "자세 = 목표각" 기준선은 0.67–0.93입니다.
  - 각속도 R²는 0.84–0.99입니다. 예외로 LOG4의 9 s 구간 pitch는 0.67인데, 입력이 작은 구간입니다.
- **rate 루프:** 실제 setpoint를 넣었을 때 R²는 roll 0.98, pitch 0.98입니다.
- **추력:** 펀치와 throttle 끊기 9회에서 시뮬 가속도와 FC 가속도의 오차가 0.15–0.44 g입니다.
- **호버 throttle:** 로그 3개 전압 모두에서 오차 0.015 이내입니다.
- **추정 오차 메커니즘:** 24° 기울기를 5 s 유지하면 FC 추정이 수평 쪽으로 5.7° 끌려갑니다. 실제 기체는 그만큼 더 기웁니다.
- **데모 제어기:**
  - `c1_team44up`에서 2.5 m/s로 날렸을 때 8게이트 이상을 충돌 없이 통과한 비율이 약 69%입니다.
  - 이 제어기는 level 루프를 역산해 목표각을 냅니다(목표각 = FC 추정 + 원하는 rate ÷ 게인). 목표각을 직접 주면 level 루프의 시정수 0.2 s 때문에 25%로 떨어집니다.
- **스모크:** 테스트 16개, 데모 수집, 5단계 커리큘럼 체인, 평가가 모두 돌아갑니다(CPU, 소형 모델).
