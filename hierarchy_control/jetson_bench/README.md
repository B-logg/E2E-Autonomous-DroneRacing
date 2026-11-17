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
# (A) 전부 한 번에: GPU/CPU 측정 + FC 통신 (30–60분)
./run.sh --uart /dev/ttyTHS1 --baud 115200

# (B) 빠른 확인 (10–15분)
./run.sh --quick --uart /dev/ttyTHS1 --baud 115200

# (C) FC 통신만 다른 baud로 한 번 더 (FC Ports 탭 baud를 921600으로 바꾸고 재부팅한 뒤, 약 1분)
./run.sh --only-uart --uart /dev/ttyTHS1 --baud 921600

# FC 없이 GPU/CPU만
./run.sh
```

**권장 순서:** (A)를 115200으로 한 번 → FC baud를 921600으로 바꾸고 (C)를 한 번. 이렇게 두 번 하면 됩니다.

## 4. 결과 보내기

실행할 때마다 다음 파일이 하나씩 생깁니다. **이 파일들만 보내 주세요.**
```
results/<호스트>-<시각>.tar.gz
```
화면 마지막에 요약이 나오고, 같은 내용이 `summary.txt`에 저장됩니다. 중간에 Ctrl+C로 멈춰도, 한 항목이 실패해도 나머지 결과는 저장됩니다.

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
