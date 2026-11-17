# Jetson 벤치마크 (추론 + FC 통신)

Jetson Orin Nano에서 실기체 연동에 필요한 시간을 한 번에 잽니다. 결과는 파일 하나(`.tar.gz`)로 묶여 나옵니다.

| 측정 | 내용 | 통과 기준 |
|---|---|---|
| 정책 | DreamerV3 정책 한 스텝, TensorRT FP32 / FP16 | p99 5 ms 이하면 여유 (100 Hz 예산 10 ms) |
| GateNet | 입력 10가지 (196² ~ 640×480), FP16 | 카메라 프레임 간격보다 짧을 것 (30 fps: 33 ms, 60 fps: 17 ms) |
| 동시 실행 | 정책 100 Hz + GateNet 30 / 60 fps를 GPU에서 같이 | 정책 p99 10 ms 이하 |
| 프레임 축소 + 왜곡 보정 | 카메라 해상도별 CPU 처리 시간 (OpenCV 있을 때) | 프레임 간격 안 |
| UART / MSP | 10 ms마다 RAW_IMU + ATTITUDE 읽기 + SET_RAW_RC 쓰기 | 사이클 p99가 10 ms보다 충분히 짧고, 타임아웃과 오류가 거의 0 |
| 시스템 | L4T, 전원 모드, 클럭, 라이브러리 버전, tegrastats | — |

가중치는 무작위입니다. 속도는 가중치 값이 아니라 네트워크 구조와 크기로 정해집니다. 카메라가 없어도 됩니다.

---

## 1. 준비 (Jetson)

```bash
tar -xzf jetson_bench.tar.gz
cd jetson_bench
sudo nvpmodel -q          # 현재 전원 모드 확인
sudo nvpmodel -m 0        # (선택) 최대 성능 모드. 실제 비행 때 쓸 모드로 맞추는 게 원칙
sudo jetson_clocks        # 클럭 고정
```

필요한 것은 JetPack에 기본으로 있는 `python3`와 `trtexec`(TensorRT)뿐입니다. PyTorch나 OpenCV가 있으면 해당 측정이 자동으로 추가됩니다.

## 2. FC 연결 (UART 측정을 할 때만)

**⚠️ 프롭을 반드시 떼고 하세요.** 스크립트는 중립 값만 보냅니다(스틱 가운데, 스로틀 최저, AUX 전부 off). 그래서 arm되거나 모터가 돌 일은 없지만, 안전을 위해서입니다.

**배선 (Jetson 40핀 ↔ FC UART, 3.3 V끼리라 레벨 변환 불필요)**

| Jetson 40핀 | FC |
|---|---|
| 8번 (UART TX) | 사용할 UART의 **RX** |
| 10번 (UART RX) | 사용할 UART의 **TX** |
| 6번 (GND) | GND |

TX와 RX를 서로 엇갈려 연결합니다.

**FC 설정 (Betaflight Configurator)**
1. Ports 탭에서 선을 연결한 UART의 **Configuration/MSP**를 켭니다.
2. 옆의 baud를 **115200**으로 둡니다. 2번째 측정에서 921600으로 바꿉니다.
3. Save and Reboot을 누르고, Configurator 연결을 끊습니다.

**Jetson 쪽**
```bash
ls /dev/ttyTHS*                       # 40핀 UART는 보통 /dev/ttyTHS1
sudo usermod -aG dialout $USER        # 처음 한 번, 그다음 로그아웃 후 다시 로그인 (또는 아래 명령을 sudo로 실행)
```

## 3. 실행

```bash
# (최종) 확정 설정(GateNet f4 240×320, 예비 480×640)만 측정 + FC 통신 (약 10분)
#   먼저: sudo nvpmodel -m 0 && sudo jetson_clocks
./run.sh --quick --duration 15 --uart /dev/ttyTHS1 --baud 500000

# (A) 전부 한 번에: GPU/CPU 측정 + FC 통신 (30–60분)
./run.sh --uart /dev/ttyTHS1 --baud 500000

# (B) 빠른 확인 (10–15분)
./run.sh --quick --uart /dev/ttyTHS1 --baud 500000

# (C) FC 통신만 (약 1분 30초)
./run.sh --only-uart --uart /dev/ttyTHS1 --baud 500000

# FC 없이 GPU/CPU만
./run.sh
```

**FC 설정 (먼저 한 번):** Betaflight CLI에서 `set serial_update_rate_hz = 2000` → `save`.
기본값 100이면 FC가 MSP 명령을 10 ms에 하나만 처리해서 사이클당 3개(IMU, 자세, RC)가 30 ms 걸립니다.
GPU 측정 전에는 `sudo nvpmodel -m 0` (MAXN SUPER)과 `sudo jetson_clocks`를 켜 주세요.

**FC 통신 결과 읽는 법:** 실제 비행에 쓸 방식은 `sequential_100hz`(요청 하나 보내고 응답 받고 다음)입니다.
- `missed 0, timeouts 0`, cycle p99가 몇 ms면 정상
- 각 테스트 뒤의 `FC answers MSP after this test`가 `NO`면, 그 테스트에서 FC가 MSP 응답을 멈춘 것입니다.
  그때는 FC LED와 Configurator 연결 상태(재부팅/부트로더 진입 여부)를 확인하고, 전원을 다시 넣어야 살아나는지 적어 주세요.

## 4. 결과 보내기

실행할 때마다 다음 파일이 하나씩 생깁니다. **이 파일들만 보내 주세요.**
```
results/<호스트>-<시각>.tar.gz
```
화면 마지막에 요약이 나오고, 같은 내용이 `summary.txt`에 저장됩니다. 중간에 Ctrl+C로 멈춰도, 한 항목이 실패해도 나머지 결과는 저장됩니다.

## 5. 카메라 지연 (C4): `camera_latency.py`

Jetson이 직접 켠 LED가 카메라 화면에 처음 보이기까지의 시간을 100번 잽니다. 배포와 같은 경로(1640×1232@30 → 하드웨어 축소 320×240)로 받습니다.

배선: 40핀 **15번 핀 → 330 Ω 저항 → LED 긴 다리(+)**, **LED 짧은 다리(−) → 6번 핀(GND)**. LED를 렌즈 앞 3–10 cm, 화면 가운데에 둡니다. 실내는 조금 어둡게 합니다.

```bash
sudo python3 camera_latency.py --led 15 --blink        # 배선 확인: LED가 1초 간격으로 10번 깜박이면 OK
sudo python3 camera_latency.py --led 15                # 자동 노출 (약 1분)
sudo python3 camera_latency.py --led 15 --exposure-us 4000   # 노출 4 ms 고정
```
결과: `results/camera-<호스트>-<시각>/summary.txt`. 마지막 줄 `camera delay ... = X ms`가 필요한 값입니다.
LED가 없으면 모니터를 Jetson에 연결하고 `python3 camera_latency.py --screen`을 쓰되, 모니터 자체 지연이 더해진 **상한값**입니다.

## 문제 해결

| 증상 | 확인할 것 |
|---|---|
| `cannot open /dev/ttyTHS1: PermissionError` | dialout 그룹 추가 후 다시 로그인하거나, `sudo ./run.sh ...` |
| `no MSP answer` | TX와 RX가 엇갈렸는지, GND, FC Ports 탭 MSP와 baud가 맞는지, Configurator가 같은 UART를 쓰고 있지 않은지 |
| 장치가 사용 중 | `sudo systemctl stop nvgetty` (시리얼 콘솔이 같은 포트를 쓰는 경우) |
| `trtexec not found` | `/usr/src/tensorrt/bin/trtexec`가 있는지 확인. 없으면 `sudo apt install tensorrt` |

## 파일
- `run.sh`: 실행과 결과 압축
- `bench.py`: 측정 본체
- `uart_msp.py`: MSP 프로토콜과 UART 측정 (의존성 없음)
- `models.py`: 정책과 GateNet의 PyTorch 정의
- `configs.py`: GateNet 입력 크기 목록
- `export_onnx.py`: ONNX를 다시 만들 때 (torch 필요, 개발 PC에서)
