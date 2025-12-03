import pygame
import random
import sys
import os
import csv

import pygetwindow as gw
from PIL import ImageGrab

pygame.init()

# Настройки датасета
CREATE_DATASET = False
CURRENT_PATH = os.getcwd()

# Настройки окна
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Aim Trainer")

WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)

font = pygame.font.SysFont(None, 36)
target_radius = 30

margin_x, margin_y = 150, 100  # Область счётчиков в левом верхнем углу (убраны, чтобы не отвлекать нейросеть)

# Минимальная дистанция между центром прицела и центром цели
MIN_CURSOR_TARGET_DIST = 2 * target_radius

def random_position():
  x = random.randint(margin_x + target_radius, WIDTH - target_radius)
  y = random.randint(margin_y + target_radius, HEIGHT - target_radius)
  return x, y

# Состояние игры
target_pos = random_position()
target_visible = True

hits = 0
misses = 0

# Состояния для датасета
game_started = False      # Игнорирование первой цели для правильной начальной позиции прицела
need_spawn_new_target = False # В начале следующего кадра требуется сменить цель
need_start_shot = False     # В конце кадра требуется сделать start
start_shot = None       # start-скриншот для текущей цели

# Курсор
pygame.mouse.set_visible(False)
clock = pygame.time.Clock()

# Окно и bbox клиентской области (без рамки и заголовка)
try:
  window = gw.getWindowsWithTitle("Aim Trainer")[0]

  wx, wy = window.left, window.top
  ww, wh = window.width, window.height

  TITLE_BAR     = 30
  BOTTOM_BORDER   = 8
  LEFT_RIGHT_BORDER = 8

  client_left   = wx + LEFT_RIGHT_BORDER
  client_top  = wy + TITLE_BAR
  client_right  = wx + ww - LEFT_RIGHT_BORDER
  client_bottom = wy + wh - BOTTOM_BORDER

  bbox = (client_left, client_top, client_right, client_bottom)
except IndexError:
  print("Окно не найдено. Проверьте заголовок.")
  sys.exit()

# Создание директорий и CSV
os.makedirs(os.path.join(CURRENT_PATH, "Dataset", "HITS"), exist_ok=True)
os.makedirs(os.path.join(CURRENT_PATH, "Dataset", "MISSES"), exist_ok=True)

csv_path = os.path.join(CURRENT_PATH, "Dataset", "dataset_labels.csv") #SAVEPATH
if CREATE_DATASET:
  csv_file = open(csv_path, "w", newline="", encoding="utf-8")
  csv_writer = csv.writer(csv_file)
  csv_writer.writerow(["type", "index", "x_norm", "y_norm"])

def grab():
  return ImageGrab.grab(bbox=bbox)

def spawn_new_target():
  global target_pos, target_visible
  # Текущая позиция курсора в координатах окна
  mx, my = pygame.mouse.get_pos()

  # Подбор позиции, не впритык к курсору
  while True:
    x, y = random_position()
    dx = x - mx
    dy = y - my
    dist = (dx*dx + dy*dy) ** 0.5
    if dist >= MIN_CURSOR_TARGET_DIST:
      target_pos = (x, y)
      break

  target_visible = True

running = True
while running:
  # НАЧАЛО КАДРА: если флаг стоит — меняем цель
  if need_spawn_new_target:
    spawn_new_target()
    need_spawn_new_target = False
    # Помечаем, что нужно сделать start-скриншот в конце текущего кадра
    need_start_shot = True
    start_shot = None

  screen.fill(WHITE)

  # Отрисовка цели
  if target_visible:
    pygame.draw.circle(screen, RED, target_pos, target_radius)

  # Отрисовка курсора
  mx, my = pygame.mouse.get_pos()
  pygame.draw.circle(screen, BLACK, (mx, my), 10)

  # Обработка событий
  for event in pygame.event.get():
    # Вызывается при закрытии окна
    if event.type == pygame.QUIT:
      running = False

    elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
      click_x, click_y = event.pos
      if target_visible:
        dist = ((click_x - target_pos[0]) ** 2 + (click_y - target_pos[1]) ** 2) ** 0.5

        # Запуск игры, игнорируя первую цель
        if not game_started:
          game_started = True
          need_spawn_new_target = True   # В следующем кадре появится первая цель, которая идет в счет
          start_shot = None
          continue

        # end-скриншот для текущей цели
        end_shot = None
        if CREATE_DATASET:
          try:
            end_shot = grab()
          except Exception as e:
            print("Ошибка end:", e)

        # Нормализованные координаты центра цели в окне (от 0 до 1)
        tx, ty = target_pos
        x_norm = tx / WIDTH
        y_norm = ty / HEIGHT

        # Проверка на попадание
        if dist <= target_radius:
          idx = hits
          hits += 1
          if CREATE_DATASET:
            try:
              # Сохранение изображения
              if start_shot is not None:
                start_shot.save(os.path.join(
                  CURRENT_PATH, "Dataset", "HITS", f"hit_start_{idx}.png" #SAVEPATH
                ))
              if end_shot is not None:
                end_shot.save(os.path.join(
                  CURRENT_PATH, "Dataset", "HITS", f"hit_end_{idx}.png" #SAVEPATH
                ))
              # Запись строки в csv файл для обучения
              csv_writer.writerow(["hit", idx, x_norm, y_norm])
            except Exception as e:
              print("Ошибка сохранения hit-пары/CSV:", e)
        else:
          idx = misses
          misses += 1
          if CREATE_DATASET:
            try:
              # Сохранение изображения
              if start_shot is not None:
                start_shot.save(os.path.join(
                  CURRENT_PATH, "Dataset", "MISSES", f"miss_start_{idx}.png" #SAVEPATH
                ))
              if end_shot is not None:
                end_shot.save(os.path.join(
                  CURRENT_PATH, "Dataset", "MISSES", f"miss_end_{idx}.png" #SAVEPATH
                ))
              # Запись строки в csv файл для обучения
              csv_writer.writerow(["miss", idx, x_norm, y_norm])
            except Exception as e:
              print("Ошибка сохранения miss-пары/CSV:", e)

        # Указываем, что нужно создать новую цель в следующем кадре
        need_spawn_new_target = True
        start_shot = None

  # Вывод счетчиков попаданий и промахов (в консоль)
  if need_spawn_new_target:
    print(f"hits: {hits}, misses: {misses}")

  # Рендер экрана
  pygame.display.flip()

  # Делаем start-скриншот после рендера экрана и новой цели
  if CREATE_DATASET and game_started and need_start_shot:
    try:
      start_shot = grab()
    except Exception as e:
      print("Ошибка start:", e)
      start_shot = None
    need_start_shot = False

  clock.tick(60)

# Корректное закрытие csv файла
if CREATE_DATASET:
  csv_file.close()

pygame.quit()
sys.exit()
