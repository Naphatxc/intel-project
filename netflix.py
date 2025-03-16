import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score

# โหลดข้อมูล Netflix
df = pd.read_csv('C:/Users/Extended-AMD/Desktop/netflix_titles.csv')

# กำจัดข้อมูลที่หายไป
df.dropna(inplace=True)

# แปลงคอลัมน์ 'type' (Movie / TV Show) เป็นตัวเลข (0: Movie, 1: TV Show)
df['type'] = df['type'].map({'Movie': 0, 'TV Show': 1})

# แปลงคอลัมน์ 'duration' จาก 'X min' หรือ 'X Seasons' เป็นตัวเลข
df['duration'] = df['duration'].apply(lambda x: int(x.split(' ')[0]) if isinstance(x, str) else 0)

# เลือกฟีเจอร์ที่ใช้ในการทำนาย (เลือก 'release_year' และ 'duration')
X = df[['release_year', 'duration']]

# การปรับขนาดข้อมูล (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ตัวแปรเป้าหมาย (ประเภทของภาพยนตร์)
y = df['type']

# แบ่งข้อมูลเป็น Train และ Test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# สร้างโมเดล Neural Network ด้วย Keras
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))  # ชั้นแรก 64 neurons
model.add(Dense(32, activation='relu'))  # ชั้นที่สอง 32 neurons
model.add(Dense(1, activation='sigmoid'))  # ชั้นสุดท้าย Sigmoid สำหรับการทำนายแบบ Binary Classification

# คอมไพล์โมเดล
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# ทำนายผล
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)  # ปรับผลทำนายให้อยู่ในช่วง 0 หรือ 1

# ประเมินผลด้วย Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# แสดงกราฟความแม่นยำในการฝึกฝน
history = model.history.history
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# ทำนายผลจากข้อมูลใน Dataset
df['predicted_type'] = model.predict(scaler.transform(df[['release_year', 'duration']]))
df['predicted_type'] = (df['predicted_type'] > 0.5).astype(int)

# แสดงผลลัพธ์
df[['title', 'type', 'predicted_type']].head(20)
