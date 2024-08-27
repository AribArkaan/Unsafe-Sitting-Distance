import cv2
import mediapipe as mp
import numpy as np
import simpleaudio as sa


mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


threshold_distance = 130



def calculate_eye_distance(face_landmarks, image_width):
    left_eye = face_landmarks[0].location_data.relative_keypoints[0]
    right_eye = face_landmarks[0].location_data.relative_keypoints[1]
    # Menghitung jarak antar mata
    distance = np.sqrt((left_eye.x - right_eye.x) ** 2 + (left_eye.y - right_eye.y) ** 2)
    return distance * image_width



def play_beep():
    frequency = 1000  # Frekuensi bip dalam Hz
    duration = 500  # Durasi bip dalam milidetik
    sample_rate = 44100  # Frekuensi sampling
    t = np.linspace(0, duration / 1000, int(sample_rate * duration / 1000), False)
    wave = 0.5 * np.sin(2 * np.pi * frequency * t)
    wave = np.int16(wave * 32767)
    play_obj = sa.play_buffer(wave, 1, 2, sample_rate)
    play_obj.wait_done()



cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Tidak dapat membaca frame dari kamera.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        results = face_detection.process(rgb_frame)

        if results.detections:
            
            for detection in results.detections:
                
                mp_drawing.draw_detection(frame, detection)
                distance = calculate_eye_distance([detection], frame.shape[1])

                if distance > threshold_distance:
               
                    cv2.putText(frame, "Anda terlalu dekat dengan monitor!",
                                (30,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                   
                    play_beep()

        cv2.imshow('Duduk Posisi Monitoring', frame)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
