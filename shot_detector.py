from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import csv
from utils import score, detect_down, detect_up, in_hoop_region, clean_hoop_pos, clean_ball_pos

class ShotDetector:
    def __init__(self):
        # Load the YOLO model created from main.py - change text to your relative path
        self.model = YOLO("Yolo-Weights/best6.pt")
        self.class_names = ['Ball', 'Hoop']

        # Use video - replace text with your video path
        self.cap = cv2.VideoCapture("DNvsTW.mp4")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)  # Frames per second

        self.ball_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)
        self.hoop_pos = []  # array of tuples ((x_pos, y_pos), frame count, width, height, conf)

        self.frame_count = 0
        self.frame = None

        self.makes = 0
        self.attempts = 0

        # Used to detect shots (upper and lower region)
        self.up = False
        self.down = False
        self.up_frame = 0
        self.down_frame = 0

        # Used for green and red colors after make/miss
        self.fade_frames = 20
        self.fade_counter = 0
        self.overlay_color = (0, 0, 0)

        # Open CSV file in write mode and write header
        self.csv_file = open('shot_results.csv', mode='w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["Shot Taken", "Result", "Ball Coordinates", "Hoop Coordinates", "Current Score", "Video Timing (seconds)"])

        # Create window for displaying video and slider
        cv2.namedWindow('Frame')
        cv2.createTrackbar('Frame', 'Frame', 0, self.total_frames - 1, self.on_trackbar_change)

        self.run()

    def on_trackbar_change(self, pos):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        self.frame_count = pos

    def run(self):
        while True:
            ret, self.frame = self.cap.read()

            if not ret:
                # End of the video or an error occurred
                break
            self.frame = cv2.resize(self.frame, (1280, 720))
            results = self.model(self.frame, stream=True)

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    # Bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # Confidence
                    conf = math.ceil((box.conf[0] * 100)) / 100

                    # Only proceed if confidence is greater than 0.75
                    if conf > 0.75:
                        # Class Name
                        cls = int(box.cls[0])
                        current_class = self.class_names[cls]

                        center = (int(x1 + w / 2), int(y1 + h / 2))

                        # Define colors for different classes
                        if current_class == "Ball":
                            color = (0, 0, 255)  # Red for Ball
                        else:
                            color = (255, 0, 0)  # Blue for hoop

                        # Draw bounding box and label
                        cv2.rectangle(self.frame, (x1, y1), (x2, y2), color, 2)
                        label = f"{current_class} {conf:.2f}"
                        cv2.putText(self.frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                        # Only create ball points if high confidence or near hoop
                        if (current_class == "Ball" and conf > 0.3) or \
                                (in_hoop_region(center, self.hoop_pos) and conf > 0.15):
                            self.ball_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

                        # Create hoop points if high confidence
                        if current_class == "Hoop" and conf > 0.3:
                            self.hoop_pos.append((center, self.frame_count, w, h, conf))
                            cvzone.cornerRect(self.frame, (x1, y1, w, h))

            self.clean_motion()
            self.shot_detection()
            self.frame_count += 1

            # Update video slider position
            cv2.setTrackbarPos('Frame', 'Frame', self.frame_count)

            cv2.imshow('Frame', self.frame)

            # Close if 'q' is clicked
            if cv2.waitKey(1) & 0xFF == ord('q'):  # higher waitKey slows video down, use 1 for webcam
                break

        self.cap.release()
        cv2.destroyAllWindows()
        self.csv_file.close()

    def clean_motion(self):
        # Clean the ball position data but do not draw circles
        self.ball_pos = clean_ball_pos(self.ball_pos, self.frame_count)

        # Clean the hoop position data and display the current hoop center
        if len(self.hoop_pos) > 1:
            self.hoop_pos = clean_hoop_pos(self.hoop_pos)
            cv2.circle(self.frame, self.hoop_pos[-1][0], 2, (128, 128, 0), 2)

    def shot_detection(self):
        if len(self.hoop_pos) > 0 and len(self.ball_pos) > 0:
            # Detecting when ball is in 'up' and 'down' area - ball can only be in 'down' area after it is in 'up'
            if not self.up:
                self.up = detect_up(self.ball_pos, self.hoop_pos)
                if self.up:
                    self.up_frame = self.ball_pos[-1][1]

            if self.up and not self.down:
                self.down = detect_down(self.ball_pos, self.hoop_pos)
                if self.down:
                    self.down_frame = self.ball_pos[-1][1]

            # If ball goes from 'up' area to 'down' area in that order, increase attempt and reset
            if self.frame_count % 10 == 0:
                if self.up and self.down and self.up_frame < self.down_frame:
                    self.attempts += 1
                    self.up = False
                    self.down = False

                    # If it is a make, put a green overlay
                    if score(self.ball_pos, self.hoop_pos):
                        self.makes += 1
                        self.overlay_color = (0, 255, 0)
                        self.fade_counter = self.fade_frames
                        result = "Successful"
                    # If it is a miss, put a red overlay
                    else:
                        self.overlay_color = (0, 0, 255)
                        self.fade_counter = self.fade_frames
                        result = "Failed"

                    # Record the result and video timing (in seconds)
                    ball_center = self.ball_pos[-1][0]
                    hoop_center = self.hoop_pos[-1][0]
                    current_score = f"{self.makes} / {self.attempts}"
                    video_timing_seconds = self.frame_count / self.fps
                    self.csv_writer.writerow([self.attempts, result, ball_center, hoop_center, current_score, video_timing_seconds])

    def display_score(self):
        # Add text
        text = str(self.makes) + " / " + str(self.attempts)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 6)
        cv2.putText(self.frame, text, (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

        # Gradually fade out color after shot
        if self.fade_counter > 0:
            alpha = 0.2 * (self.fade_counter / self.fade_frames)
            self.frame = cv2.addWeighted(self.frame, 1 - alpha, np.full_like(self.frame, self.overlay_color), alpha, 0)
            self.fade_counter -= 1


if __name__ == "__main__":
    ShotDetector()

