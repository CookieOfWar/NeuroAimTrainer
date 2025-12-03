import os
import time
import threading

import pygetwindow as gw
from PIL import ImageGrab
import numpy as np
import pyautogui
import keyboard

from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Настройки окна (Обязательно такие же, как в Aim_trainer_game.py)
WIDTH, HEIGHT = 800, 600
IMG_SIZE = (224, 224)

TITLE_BAR     = 30
BOTTOM_BORDER   = 8
LEFT_RIGHT_BORDER = 8

MAX_SHOTS = 500      # Ограничение на количество выстрелов
SHOT_DELAY = 0.05    # Пауза между выстрелами (сек)

# Директории
CURRENT_PATH = os.getcwd()
DATASET_DIR = os.path.join(CURRENT_PATH, "Dataset") #SAVEPATH
MODEL_PATH  = os.path.join(DATASET_DIR, "aim_cnn_keras.h5") #SAVEPATH

# Состояния бота
running = True     # жив ли поток бота
shoot_enabled = False  # включена ли сейчас стрельба
shot_count = 0     # сколько выстрелов сделано

# Загрузка модели
print("Загрузка модели...")
model = load_model(MODEL_PATH)
print("Модель загружена.")

# Захват окна
def get_game_bbox():
  window = gw.getWindowsWithTitle("Aim Trainer")[0]

  wx, wy = window.left, window.top
  ww, wh = window.width, window.height

  client_left   = wx + LEFT_RIGHT_BORDER
  client_top  = wy + TITLE_BAR
  client_right  = wx + ww - LEFT_RIGHT_BORDER
  client_bottom = wy + wh - BOTTOM_BORDER

  return (client_left, client_top, client_right, client_bottom)


bbox = get_game_bbox()
client_left, client_top, client_right, client_bottom = bbox
print("Клиентская область игры:", bbox)

# Захват кадра
def grab_frame():
  return ImageGrab.grab(bbox=bbox)

# Предсказание координат цели
def predict_target_coords(pil_img):
  img = pil_img.resize(IMG_SIZE)
  arr = img_to_array(img) / 255.0
  arr = np.expand_dims(arr, axis=0)
  x_norm, y_norm = model.predict(arr, verbose=0)[0]
  x = int(x_norm * WIDTH)
  y = int(y_norm * HEIGHT)
  return x, y

# Выстрел
def do_one_shot():
  global shot_count

  frame = grab_frame()
  tx, ty = predict_target_coords(frame)

  screen_x = client_left + tx
  screen_y = client_top  + ty

  pyautogui.moveTo(screen_x, screen_y)
  pyautogui.click()
  shot_count += 1
  print(f"Выстрел #{shot_count} в точку окна: ({tx}, {ty})")


# Основной поток
def bot_loop():
  global running, shoot_enabled, shot_count

  while running and shot_count < MAX_SHOTS:
    if shoot_enabled:
      do_one_shot()
      time.sleep(SHOT_DELAY)
    else:
      time.sleep(0.05)  # Idle для разгрузки процессора

  print("Бот завершил работу (лимит выстрелов или остановка).")
  shoot_enabled = False


# --- ГОРЯЧИЕ КЛАВИШИ ---
# включить/выключить огонь
def toggle_shoot():
  global shoot_enabled
  shoot_enabled = not shoot_enabled
  print(f"Стрельба {'включена' if shoot_enabled else 'выключена'}.")

# Остановить бота
def stop_bot():
  global running
  running = False
  print("Получен сигнал остановки (Esc).")


# F8 — включить/выключить огонь
keyboard.add_hotkey('f8', toggle_shoot)
# Esc — полностью остановить бота и выйти
keyboard.add_hotkey('esc', stop_bot)

print("Управление:")
print("  F8  — включить/выключить огонь")
print("  Esc — остановить бота и выйти")
print(f"Максимум выстрелов за сессию: {MAX_SHOTS}")
print("Сфокусируй окно игры и управляй ботом горячими клавишами.\n")

# Запуск бота
bot_thread = threading.Thread(target=bot_loop, daemon=True)
bot_thread.start()

# Ожидание завершения потока (по Esc или лимиту)
while running and bot_thread.is_alive():
  time.sleep(0.1)

print("Выход из скрипта.")
