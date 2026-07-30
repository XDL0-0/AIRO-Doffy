"""Test script for camera transmission."""

import time
import cv2

import utils
from config import Config
from camera_udp import CameraUDPManager
from WebRTC_udp import WebRTCUDPManager


def main() -> None:
    cfg = Config()
    utils.logger.info("Starting Camera Test...")
    utils.logger.info(f"VIDEO_TRANSPORT: {cfg.VIDEO_TRANSPORT}")

    # Follow the same logic as main.py to select manager based on Config
    if cfg.VIDEO_TRANSPORT.lower() == "webrtc":
        cu_manager = WebRTCUDPManager()
    else:
        cu_manager = CameraUDPManager()

    # Start the background threads for camera reading and transmission (UDP/WebRTC)
    cu_manager.start_comms_threads()

    utils.logger.info("Camera transmission threads started.")
    # utils.logger.info("Press Ctrl+C in terminal or 'q' on OpenCV window to exit.")

    try:
        while True:
            # Display local images to confirm cameras are being successfully read
            with cu_manager._lock:
                images = dict(cu_manager.camera_images)

            if not images:
                time.sleep(0.1)
                continue

            for cam_name, img in images.items():
                if img is not None:
                    # Convert RGB to BGR for OpenCV display since manager stores it as RGB
                    # camera_images is stored as RGB inside manager thread
                    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    # cv2.imshow(f"Camera Test Monitor - {cam_name}", img_bgr)

            # Break if 'q' is pressed
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     utils.logger.info("Exiting...")
            #     break

            # Sleep to control the display refresh rate
            time.sleep(0.03)

    except KeyboardInterrupt:
        utils.logger.info("Stopping test upon user interrupt...")

    finally:
        utils.logger.info("Cleaning up...")
        try:
            cu_manager.close()
        except Exception as e:
            utils.logger.error(f"Error closing cu_manager: {e}")

        # cv2.destroyAllWindows()
        utils.logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()
