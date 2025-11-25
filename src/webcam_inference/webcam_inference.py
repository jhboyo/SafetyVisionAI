"""
Real-time PPE Detection using Webcam

Detect helmet, head, and vest in real-time using laptop camera or external webcam.

Usage:
    # Basic usage (laptop camera)
    uv run python src/webcam_inference/webcam_inference.py

    # External webcam
    uv run python src/webcam_inference/webcam_inference.py --camera 1

    # Adjust confidence
    uv run python src/webcam_inference/webcam_inference.py --conf 0.3

    # Custom resolution
    uv run python src/webcam_inference/webcam_inference.py --width 1280 --height 720

Keyboard Controls:
    Q - Quit
    S - Save Screenshot
    P - Pause/Resume
    + - Increase Confidence
    - - Decrease Confidence
    H - Toggle Help
"""

import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import sys
import time
import threading
import tempfile
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(str(Path(__file__).parent))

from utils import (
    FPSCounter,
    calculate_statistics_from_results,
    draw_statistics_overlay,
    draw_help_overlay,
    save_screenshot,
    get_available_cameras,
    initialize_camera
)

# 음성 경고를 위한 라이브러리 import (선택적)
try:
    from gtts import gTTS  # Google Text-to-Speech: 텍스트를 음성으로 변환
    import pygame  # 오디오 재생용
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️  음성 경고 기능을 사용하려면 gTTS와 pygame을 설치하세요: pip install gtts pygame")


# ============================================================================
# 음성 경고 시스템
# Voice Alert System
# ============================================================================

class VoiceAlertManager:
    """
    AI 음성 경고 시스템 매니저

    PPE 미착용 감지 시 한국어 음성 경고를 재생합니다.
    중복 재생 방지를 위한 쿨다운 타이머를 포함합니다.
    """

    def __init__(self, cooldown_seconds: int = 10):
        """
        음성 경고 매니저 초기화

        Args:
            cooldown_seconds: 같은 경고의 재생 간격 (초, 기본값: 10초)
        """
        self.cooldown_seconds = cooldown_seconds  # 쿨다운 시간
        self.last_alert_time = {}  # 마지막 경고 시간 기록
        self.lock = threading.Lock()  # 스레드 안전성을 위한 락
        self.audio_cache = {}  # 생성된 음성 파일 캐시 (재사용)

        # gTTS와 pygame이 설치되어 있지 않으면 비활성화
        if not AUDIO_AVAILABLE:
            self.enabled = False
            return

        # pygame mixer 초기화 시도
        try:
            pygame.mixer.init()
            self.enabled = True
            print("✅ 음성 경고 시스템 활성화")
        except Exception as e:
            print(f"⚠️  음성 경고 시스템 초기화 실패: {e}")
            self.enabled = False

    def _generate_audio(self, text: str, lang: str = 'ko') -> str:
        """
        텍스트를 음성 파일로 변환 (gTTS 사용)

        Args:
            text: 변환할 텍스트 (예: "안전모를 착용하세요")
            lang: 언어 코드 (기본값: 'ko' 한국어)

        Returns:
            str: 생성된 음성 파일 경로 (mp3), 실패 시 None
        """
        # 캐시 확인 (동일한 텍스트는 재생성하지 않고 재사용)
        cache_key = f"{text}_{lang}"
        if cache_key in self.audio_cache:
            return self.audio_cache[cache_key]

        try:
            # Google TTS로 음성 생성
            tts = gTTS(text=text, lang=lang, slow=False)

            # 시스템 임시 디렉토리에 mp3 파일로 저장
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
                temp_path = fp.name
                tts.save(temp_path)

            # 캐시에 저장 (다음번 재사용)
            self.audio_cache[cache_key] = temp_path
            return temp_path

        except Exception as e:
            print(f"⚠️  음성 생성 실패: {e}")
            return None

    def play_alert(self, alert_type: str, force: bool = False):
        """
        음성 경고 재생 (쿨다운 타이머 적용)

        Args:
            alert_type: 경고 유형
                - 'helmet': "안전모를 착용하세요"
                - 'vest': "안전 조끼를 착용하세요"
                - 'danger': "위험! 안전 장비를 착용하세요"
            force: True일 경우 쿨다운 무시하고 강제 재생 (기본값: False)
        """
        if not self.enabled:
            return

        # 쿨다운 체크 (스레드 안전)
        with self.lock:
            current_time = time.time()
            last_time = self.last_alert_time.get(alert_type, 0)

            # 쿨다운 시간이 지나지 않았으면 재생하지 않음
            if not force and (current_time - last_time) < self.cooldown_seconds:
                return

            # 마지막 재생 시간 업데이트
            self.last_alert_time[alert_type] = current_time

        # 경고 유형에 따른 메시지 선택
        messages = {
            'helmet': '안전모를 착용하세요',
            'vest': '안전 조끼를 착용하세요',
            'danger': '위험! 안전 장비를 착용하세요'
        }
        message = messages.get(alert_type, '안전 수칙을 준수하세요')

        # 별도 스레드에서 재생 (메인 스레드가 차단되지 않도록)
        thread = threading.Thread(
            target=self._play_audio_thread,
            args=(message,),
            daemon=True  # 메인 프로그램 종료 시 자동 종료
        )
        thread.start()

    def _play_audio_thread(self, text: str):
        """
        음성 재생 스레드 (별도 스레드에서 실행되는 내부 메서드)

        macOS에서는 afplay를 사용하고, 다른 OS에서는 pygame을 사용합니다.
        afplay가 더 안정적으로 작동하기 때문에 macOS에서는 이를 우선 사용합니다.

        Args:
            text: 재생할 텍스트 (예: "안전모를 착용하세요")
        """
        try:
            # 음성 파일 생성 또는 캐시에서 가져오기
            audio_path = self._generate_audio(text)
            if audio_path and os.path.exists(audio_path):
                import platform
                import subprocess

                # macOS인 경우 afplay 명령어 사용 (시스템 기본 오디오 플레이어)
                if platform.system() == 'Darwin':
                    print(f"🔊 음성 재생: {text}")
                    subprocess.run(['afplay', audio_path], check=False)
                else:
                    # Windows/Linux에서는 pygame 사용
                    pygame.mixer.music.load(audio_path)
                    pygame.mixer.music.play()

                    # 재생이 끝날 때까지 대기
                    while pygame.mixer.music.get_busy():
                        time.sleep(0.1)

        except Exception as e:
            print(f"⚠️  음성 재생 실패: {e}")

    def cleanup(self):
        """
        리소스 정리 및 임시 파일 삭제

        프로그램 종료 시 호출하여 생성된 모든 임시 음성 파일을 삭제합니다.
        """
        # pygame mixer 종료
        if self.enabled:
            pygame.mixer.quit()

        # 캐시된 모든 임시 음성 파일 삭제
        for path in self.audio_cache.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except:
                pass  # 삭제 실패해도 무시

        self.audio_cache.clear()


# ============================================================================
# 메인 실시간 추론 함수
# Main Real-time Inference Function
# ============================================================================

def run_realtime_inference(
    camera_id: int = 0,
    model_path: Path = None,
    conf_threshold: float = 0.25,
    width: int = None,
    height: int = None,
    output_dir: Path = None,
    enable_voice_alert: bool = True
):
    """
    실시간 PPE 탐지 수행

    Args:
        camera_id: 카메라 인덱스 (0: 노트북 내장, 1: 외부 웹캠)
        model_path: YOLO 모델 경로
        conf_threshold: 신뢰도 임계값
        width: 해상도 너비
        height: 해상도 높이
        output_dir: 스크린샷 저장 디렉토리
        enable_voice_alert: 음성 경고 활성화 여부
    """
    print("="*80)
    print("PPE Detection - Real-time Webcam Inference")
    print("="*80)
    print(f"Camera: {camera_id}")
    print(f"Model: {model_path}")
    print(f"Confidence Threshold: {conf_threshold}")
    if width and height:
        print(f"Resolution: {width}x{height}")
    print("="*80)
    print()

    # 사용 가능한 카메라 확인
    available_cameras = get_available_cameras()
    print(f"Available cameras: {available_cameras}")

    if camera_id not in available_cameras:
        print(f"Error: Camera {camera_id} is not available.")
        print(f"Please use one of: {available_cameras}")
        return

    # 모델 로드
    print("\nLoading YOLO model...")
    try:
        model = YOLO(str(model_path))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 카메라 초기화
    print(f"\nInitializing camera {camera_id}...")
    try:
        cap = initialize_camera(camera_id, width, height)
        print("Camera initialized successfully!")
    except Exception as e:
        print(f"Error initializing camera: {e}")
        return

    # 실제 해상도 확인
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")

    # FPS 카운터 초기화
    fps_counter = FPSCounter(window_size=30)

    # 음성 경고 매니저 초기화
    voice_manager = None
    if enable_voice_alert and AUDIO_AVAILABLE:
        voice_manager = VoiceAlertManager(cooldown_seconds=10)
        if voice_manager.enabled:
            print("🔊 음성 경고 시스템 활성화됨")
        else:
            voice_manager = None
            print("⚠️  음성 경고 시스템 비활성화됨")
    elif enable_voice_alert and not AUDIO_AVAILABLE:
        print("⚠️  음성 경고 기능을 사용하려면 gTTS와 pygame을 설치하세요")

    # 상태 변수
    paused = False
    show_help = False
    current_conf = conf_threshold

    print("\n" + "="*80)
    print("Starting real-time inference...")
    print("Press 'H' for keyboard controls")
    print("="*80 + "\n")

    # 메인 루프
    frame_count = 0
    try:
        while True:
            # 일시정지 상태가 아닐 때만 프레임 읽기
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Failed to read frame from camera")
                    break

                frame_count += 1

                # YOLO 추론 수행
                results = model(frame, conf=current_conf, verbose=False)

                # 결과 시각화 (바운딩 박스 그리기)
                annotated_frame = results[0].plot()

                # 통계 계산 (helmet, head, vest 개수 및 착용률)
                stats = calculate_statistics_from_results(results)

                # 안전 수준 평가 및 음성 경고
                if voice_manager and stats['total_workers'] > 0:
                    helmet_rate = stats['helmet_rate']
                    head_count = stats['head_count']

                    # 위험 수준 (착용률 70% 미만)
                    if helmet_rate < 70:
                        if head_count >= 2:
                            # 2명 이상 미착용 시 위험 경고
                            voice_manager.play_alert('danger')
                        elif head_count > 0:
                            # 1명 미착용 시 헬멧 경고
                            voice_manager.play_alert('helmet')
                    # 주의 수준 (착용률 70-90%)
                    elif helmet_rate < 90:
                        if head_count > 0:
                            voice_manager.play_alert('helmet')

                # FPS 업데이트
                fps = fps_counter.update()

                # 통계 오버레이 추가
                display_frame = draw_statistics_overlay(
                    annotated_frame, stats, fps, current_conf
                )
            else:
                # 일시정지 상태에서는 기존 프레임 사용
                display_frame = annotated_frame.copy()

                # 일시정지 메시지 표시
                height, width = display_frame.shape[:2]
                cv2.putText(
                    display_frame, "PAUSED (Press P to resume)",
                    (width // 2 - 200, height // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 255, 255), 2
                )

            # 도움말 오버레이 (H 키 누름)
            if show_help:
                display_frame = draw_help_overlay(display_frame)

            # 화면 표시
            cv2.imshow('PPE Detection - Real-time', display_frame)

            # 키보드 입력 처리
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                # 종료
                print("\nQuitting...")
                break

            elif key == ord('s') or key == ord('S'):
                # 스크린샷 저장
                filepath = save_screenshot(display_frame, str(output_dir))
                print(f"Screenshot saved: {filepath}")

            elif key == ord('p') or key == ord('P'):
                # 일시정지/재개
                paused = not paused
                status = "paused" if paused else "resumed"
                print(f"Video {status}")

            elif key == ord('h') or key == ord('H'):
                # 도움말 토글
                show_help = not show_help

            elif key == ord('+') or key == ord('='):
                # 신뢰도 증가
                current_conf = min(current_conf + 0.05, 0.95)
                print(f"Confidence threshold: {current_conf:.2f}")

            elif key == ord('-') or key == ord('_'):
                # 신뢰도 감소
                current_conf = max(current_conf - 0.05, 0.05)
                print(f"Confidence threshold: {current_conf:.2f}")

            elif key == ord('v') or key == ord('V'):
                # 강제 음성 테스트 (디버깅용)
                if voice_manager:
                    print("🔊 강제 음성 테스트: 'helmet' 경고 재생")
                    voice_manager.play_alert('helmet', force=True)
                else:
                    print("⚠️  음성 매니저가 비활성화되어 있습니다")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")

    finally:
        # 리소스 해제
        print("\nReleasing resources...")

        # 음성 매니저 정리
        if voice_manager:
            voice_manager.cleanup()

        cap.release()
        cv2.destroyAllWindows()

        # 최종 통계 출력
        print("\n" + "="*80)
        print("Session Summary")
        print("="*80)
        print(f"Total frames processed: {frame_count}")
        print(f"Average FPS: {fps_counter.get_fps():.1f}")
        print("="*80)
        print("\nThank you for using PPE Detection System!")


# ============================================================================
# 커맨드라인 인터페이스
# Command-Line Interface
# ============================================================================

def parse_args():
    """커맨드라인 인자 파싱"""
    parser = argparse.ArgumentParser(
        description='Real-time PPE Detection using Webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Laptop camera
  python webcam_inference.py

  # External webcam
  python webcam_inference.py --camera 1

  # Adjust confidence
  python webcam_inference.py --conf 0.3

  # Custom resolution
  python webcam_inference.py --width 1280 --height 720

Keyboard Controls:
  Q - Quit
  S - Save Screenshot
  P - Pause/Resume
  + - Increase Confidence
  - - Decrease Confidence
  H - Toggle Help
        """
    )

    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera index (0: laptop, 1: external webcam)'
    )

    parser.add_argument(
        '--model', '-m',
        type=str,
        default=None,
        help='Model file path (default: models/ppe_detection/weights/best.pt)'
    )

    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )

    parser.add_argument(
        '--width', '-w',
        type=int,
        default=None,
        help='Camera width resolution'
    )

    parser.add_argument(
        '--height', '-ht',
        type=int,
        default=None,
        help='Camera height resolution'
    )

    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Screenshot output directory (default: output/webcam_screenshots)'
    )

    parser.add_argument(
        '--voice-alert',
        action='store_true',
        default=True,
        help='Enable voice alert for safety warnings (default: True)'
    )

    parser.add_argument(
        '--no-voice-alert',
        dest='voice_alert',
        action='store_false',
        help='Disable voice alert'
    )

    return parser.parse_args()


# ============================================================================
# 메인 함수
# Main Function
# ============================================================================

def main():
    """메인 실행 함수"""
    args = parse_args()

    # 프로젝트 기본 디렉토리 (src/webcam_inference/webcam_inference.py)
    base_dir = Path(__file__).parent.parent.parent

    # 모델 경로 설정
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        model_path = base_dir / 'models' / 'ppe_detection' / 'weights' / 'best.pt'

    # 출력 디렉토리 설정
    if args.output:
        output_dir = Path(args.output)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        output_dir = base_dir / 'output' / 'webcam_screenshots'

    # 모델 파일 존재 확인
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nPlease ensure the model file exists or specify the correct path using --model")
        return

    # 실시간 추론 실행
    try:
        run_realtime_inference(
            camera_id=args.camera,
            model_path=model_path,
            conf_threshold=args.conf,
            width=args.width,
            height=args.height,
            output_dir=output_dir,
            enable_voice_alert=args.voice_alert
        )
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
