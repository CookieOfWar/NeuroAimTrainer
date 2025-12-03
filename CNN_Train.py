import os
import csv
import numpy as np

from keras.preprocessing.image import load_img, img_to_array
from keras import layers, models, optimizers, losses, metrics

# Директории
BASE_DIR  = os.getcwd()
DATASET_DIR = os.path.join(BASE_DIR, "Dataset") #SAVEPATH
CSV_PATH  = os.path.join(DATASET_DIR, "dataset_labels.csv") #SAVEPATH
HITS_DIR  = os.path.join(DATASET_DIR, "HITS") #SAVEPATH
MISSES_DIR  = os.path.join(DATASET_DIR, "MISSES") #SAVEPATH

# Размер обрабатываемого скриншота
IMG_SIZE = (224, 224)

# Коэффициент разбивки данных для обучения и тестирования
VAL_SPLIT = 0.2

# Загрузка данных из csv файла
def load_data(only_hits=True):
  X = []
  y = []

  with open(CSV_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
      sample_type = row["type"]  # hit/miss
      idx     = int(row["index"])
      x_norm    = float(row["x_norm"])
      y_norm    = float(row["y_norm"])

      if only_hits and sample_type != "hit":
        continue
      # Хотел сделать обучение еще и на неправильных данных (misses), но не целесообразно

      if sample_type == "hit":
        img_name = f"hit_start_{idx}.png"
        img_dir  = HITS_DIR
      else:
        img_name = f"miss_start_{idx}.png"
        img_dir  = MISSES_DIR

      img_path = os.path.join(img_dir, img_name)
      if not os.path.exists(img_path):
        continue

      img = load_img(img_path, target_size=IMG_SIZE)
      arr = img_to_array(img) / 255.0
      X.append(arr)
      y.append([x_norm, y_norm])

  X = np.asarray(X, dtype="float32")
  y = np.asarray(y, dtype="float32")
  return X, y


print("Загрузка данных...")
X, y = load_data(only_hits=True)
print("Размер X:", X.shape, "Размер y:", y.shape)

# Разбитие на обучающую и валидационную выборки
num_samples = X.shape[0]
indices = np.arange(num_samples)
np.random.shuffle(indices)

X = X[indices]
y = y[indices]

val_size = int(num_samples * VAL_SPLIT)
X_val, y_val   = X[:val_size], y[:val_size]
X_train, y_train = X[val_size:], y[val_size:]

print("Train:", X_train.shape, y_train.shape)
print("Val:  ", X_val.shape, y_val.shape)

# Модель
input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)

model = models.Sequential([
  layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                input_shape=input_shape),
  layers.MaxPooling2D(2, 2),

  layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
  layers.MaxPooling2D(2, 2),

  layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
  layers.MaxPooling2D(2, 2),

  layers.Flatten(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.3),
  layers.Dense(128, activation='relu'),
  layers.Dense(2, activation='sigmoid')
])

model.compile(
  optimizer=optimizers.Adam(1e-3),
  loss=losses.MeanSquaredError(),
  metrics=[metrics.MeanAbsoluteError()],
)

model.summary()

# Обучение
EPOCHS = 50
BATCH_SIZE = 32

history = model.fit(
  X_train, y_train,
  validation_data=(X_val, y_val),
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  shuffle=True,
)

# Сохранение модели
model_path = os.path.join(DATASET_DIR, "aim_cnn_keras.h5") #SAVEPATH
model.save(model_path)
print("Модель сохранена в", model_path)
