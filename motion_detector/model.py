import cv2
import RPi.GPIO as GPIO
import os
import time
import numpy as np


# Установка пинов GPIO для светодиода и пьезоизлучателя
led_pin = 18
buzzer_pin = 17
motion_detected = False
output_folder = 'motion_images'



# Создание папки для сохранения кадров с движением
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
frame_count = 0
# *************************************************


# Инициализация GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(led_pin, GPIO.OUT)
GPIO.setup(buzzer_pin, GPIO.OUT)

# Функция для воспроизведения звукового сигнала
def buzz(pitch, duration):
    period = 1.0 / pitch
    delay = period / 2
    cycles = int(duration * pitch)
    for _ in range(cycles):
        GPIO.output(buzzer_pin, GPIO.HIGH)
        time.sleep(delay)
        GPIO.output(buzzer_pin, GPIO.LOW)
        time.sleep(delay)

# Загрузка модели YOLO и конфигурации
yolo_weights = 'yolo.weights' # Замените 'yolov3.weights' на путь к вашему файлу с весами модели YOLO
yolo_cfg = 'yolo.cfg' # Замените 'yolov3.cfg' на путь к конфигурационному файлу модели YOLO
net = cv2.dnn.readNet(yolo_weights, yolo_cfg)

# Загрузка названий классов для модели YOLO
yolo_classes = 'coc.names' # Замените 'coco.names' на путь к файлу с названиями классов

with open(yolo_classes, 'r') as f:classes = f.read().strip().split('\n')

# Инициализация видеопотока (загрузка видеофайла)
video_path = 'your_video.mp4' # Замените 'your_video.mp4' на путь к вашему видеофайлу
cap = cv2.VideoCapture(video_path)

motion_detected = False

while True:
    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getUnconnectedOutLayersNames()
    detections = net.forward(layer_names)

    detected_people = 0

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == 'person':
                detected_people += 1
    # cv2.imshow('Videp with Detection', frame)
    # Проверка наличия обнаруженных людей
    if detected_people > 0:
        if not motion_detected:
            print("Движение обнаружено!")
            GPIO.output(led_pin, GPIO.HIGH)
            buzz(1000, 1) # Воспроизвести звук с пьезоизлучателя
            motion_detected = True
            
        # Сохранение кадра с движением
        frame_count += 1
        frame_filename = os.path.join(output_folder, f'motion_frame_{frame_count}.jpg')
        cv2.imwrite(frame_filename, frame)

# ***************************************************************

    else:
        if motion_detected:
            print("Движение прекратилось.")
            GPIO.output(led_pin, GPIO.LOW)
            motion_detected = False

# Освобождение ресурсов
cap.release()
GPIO.cleanup()