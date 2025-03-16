import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neural_network import MLPClassifier
import re
import time

def home_page():
    st.title("Machine Learning with Dataset")
    
    # อธิบายเกี่ยวกับ dataset
    st.write("""
    📊ในหน้านี้ เราจะใช้ dataset ที่เกี่ยวข้องกับข้อมูลข้อมูลการตลาดของธนาคาร มีตัวแปรต่างๆ เช่น อายุ, การศึกษา, และสถานะการสมัครผลิตภัณฑ์ 
    ซึ่งได้มาจากการสำรวจของ UCI โดยใช้ข้อมูลจากไฟล์ **bank.csv**.
    ข้อมูลนี้จะช่วยในการวิเคราะห์ความสัมพันธ์ระหว่างอายุและเงิน แหล่งที่มา https://archive.ics.uci.edu/dataset/222/bank+marketing
    """)
    
    # เพิ่มรายละเอียดของ dataset
    st.write("""ผมได้ทำการใช้ Label Encoding เพื่อแปลงค่าข้อมูลประเภท ข้อความ (String) ให้เป็น ตัวเลข (Integer) เพื่อใช้ในโมเดล""")
    st.image('picture/b1.jpg')
    st.write("""แยกข้อมูลออกเป็น 2 ส่วน:Features (X) → ตัวแปรอิสระ (ใช้ทำนาย)Target (y) → ตัวแปรเป้าหมาย (ค่าที่ต้องการทำนาย)จากนั้นแบ่งข้อมูลเป็น ชุดฝึก (train) และ ชุดทดสอบ (test)""")
    st.image('picture/b2.jpg')
    st.write("""ปรับมาตรฐานข้อมูล → ทำให้ข้อมูลอยู่ในสเกลเดียวกันสร้างและเทรนโมเดล Random Forestทดสอบโมเดลและคำนวณ Accuracy""")
    st.image('picture/b3.jpg')
    st.write("""ใช้ Confusion Matrix เพื่อดูว่าระบบทำนายผลถูก-ผิดมากน้อยแค่ไหน""")
    st.image('picture/b4.jpg')
def about_page():
    st.title("Demo Machine Learning")
    # โหลดข้อมูล
    file_path = "bank.csv"
    df = pd.read_csv(file_path, delimiter=";")

    # แปลงค่า categorical เป็นตัวเลข
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype('category').cat.codes

    # 🎯 สร้างหน้า Streamlit
    st.title("📊 Interactive Data Visualization")

    # ✅ 1. ให้ผู้ใช้เลือกช่วงอายุ
    age_range = st.slider("Select Age Range", int(df["age"].min()), int(df["age"].max()), (20, 50))

    # ✅ 2. ให้ผู้ใช้เลือกประเภทกราฟ
    plot_type = st.radio("Choose Plot Type", ["Histogram", "Box Plot", "Scatter Plot"])

    # ✅ 3. กรองข้อมูลตามช่วงอายุที่เลือก
    filtered_df = df[(df["age"] >= age_range[0]) & (df["age"] <= age_range[1])]

    # ✅ 4. แสดงกราฟตามประเภทที่เลือก
    st.subheader(f"Showing {plot_type} for Age {age_range[0]} - {age_range[1]}")

    fig, ax = plt.subplots()

    if plot_type == "Histogram":
        sns.histplot(filtered_df["age"], bins=20, kde=True, ax=ax)
        ax.set_title("Age Distribution")

    elif plot_type == "Box Plot":
        sns.boxplot(x=filtered_df["age"], ax=ax)
        ax.set_title("Age Box Plot")

    elif plot_type == "Scatter Plot":
        sns.scatterplot(x=filtered_df["age"], y=filtered_df["balance"], hue=filtered_df["y"], palette="coolwarm", ax=ax)
        ax.set_title("Age vs Balance (Colored by Subscription)")
        ax.set_xlabel("Age")
        ax.set_ylabel("Balance")

    st.pyplot(fig)

    

# หน้าบริการ (Neural Network)
def services_page():
    st.title("Neural Network")
    st.write("""
    ข้อมูลที่คุณใช้คือ Netflix Movies and TV Shows dataset ซึ่งมีข้อมูลเกี่ยวกับภาพยนตร์และรายการทีวีที่มีอยู่บน Netflix โดยคอลัมน์ (features) ใน dataset นี้อาจมีรายละเอียดต่าง ๆ ที่ให้ข้อมูลเกี่ยวกับภาพยนตร์หรือรายการทีวีที่มีอยู่ในแพลตฟอร์ม Netflix ซึ่งประกอบด้วย:
    ### features หลักใน dataset นี้:
    1. **show_id**: หมายเลขหรือรหัสเฉพาะที่ใช้ระบุแต่ละรายการ
    2. **type**: ประเภทของรายการ — เป็นการบ่งบอกว่าเป็น Movie หรือ TV Show (ภาพยนตร์หรือรายการทีวี)
    3. **title**: ชื่อของภาพยนตร์หรือรายการทีวี
    4. **director**: ชื่อของผู้กำกับ (ถ้ามี)
    5. **cast**: นักแสดงหรือบุคคลที่มีบทบาทในรายการ
    6. **country**: ประเทศที่ผลิตภาพยนตร์หรือรายการ
    7. **date_added**: วันที่รายการถูกเพิ่มเข้ามาใน Netflix
    8. **release_year**: ปีที่ภาพยนตร์หรือรายการถูกปล่อยออกมา
    9. **rating**: การจัดอันดับหรือเรตติ้งของภาพยนตร์หรือรายการ (เช่น PG, R, TV-MA)
    10. **duration**: ความยาวของภาพยนตร์หรือรายการในรูปแบบของ "X min" สำหรับภาพยนตร์ หรือ "X Seasons" สำหรับรายการทีวี
    11. **listed_in**: หมวดหมู่ที่ Netflix ใช้ในการจัดระเบียบภาพยนตร์หรือรายการ (เช่น ดราม่า, คอมเมดี้, สารคดี เป็นต้น)
    12. **description**: คำอธิบายเกี่ยวกับเนื้อหาของภาพยนตร์หรือรายการทีวี
    """)
    
    st.write(""" เริ่มจาก โหลดข้อมูล Netflix และ กำจัดข้อมูลที่หายไป""")
    st.image('picture/in1.png')
    
    st.write(""" แปลง colum เป็นตัวเลข """)
    st.image('picture/in2.png')
    
    st.write(""" เลือกว่าจะใช้ข้อมูลไหนในการแยกระหว่าง movie และ TV show ปรับและแบ่งข้อมูล """)
    st.image('picture/in3.png')
    
    st.write(""" สร้าง model neural network """)
    st.image('picture/in4.png')
    
    st.write(""" ทำการเทรนโมเดลและแสดงกราฟ """)
    st.image('picture/in5.png')
    
    st.write(""" output แสดงความแม่นยำในการเทรน """)
    st.image('picture/in6.png')

    st.write(""" โดยผมได้นำ data set มาจาก Kaggle """)
    st.markdown("Netflix Movies and TV Shows(https://www.kaggle.com/datasets/shivamb/netflix-shows)")

# หน้าติดต่อเรา (Demo Neural Network)
@st.cache_data
def load_data():
    df = pd.read_csv('netflix_titles.csv')
    df.dropna(subset=['duration', 'release_year'], inplace=True)
    df['type'] = df['type'].map({'Movie': 0, 'TV Show': 1})

    def convert_duration(duration):
        match = re.match(r'(\d+)\s*(min|Season)', duration)
        if match:
            value, unit = match.groups()
            return int(value), 'Movie' if unit == 'min' else 'TV Show'
        return 0, 'Unknown'

    df['duration_value'], df['category'] = zip(*df['duration'].apply(convert_duration))
    return df

def contact_page():
    st.title("Demo Neural Network Without TensorFlow")
    st.image('picture/ne.png')

    df = load_data()  # โหลดข้อมูลจาก cache

    # แยกข้อมูล Movie และ TV Show
    movies_df = df[df['category'] == 'Movie']
    tv_shows_df = df[df['category'] == 'TV Show']

    # แสดงข้อมูล Movies และ TV Shows
    st.write("Movies")
    st.write(movies_df[['title', 'duration', 'release_year']].head(10))

    st.write("TV Shows")
    st.write(tv_shows_df[['title', 'duration', 'release_year']].head(10))

    # สร้าง StandardScaler
    scaler = StandardScaler()
    X = df[['release_year', 'duration_value']]
    y = df['type']
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # ตรวจสอบว่าโมเดลถูกเก็บใน session_state หรือไม่
    if 'model' not in st.session_state:
        st.session_state.model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1000)

    model = st.session_state.model

    with st.form("train_form"):  # ใช้ st.form เพื่อฝึกโมเดล
        if st.form_submit_button("Train Model"):
            with st.spinner("Training model..."):
                model.fit(X_train, y_train)
                st.success("Model trained successfully!")
                time.sleep(3)
                
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                st.write(f"Accuracy: {accuracy}")
                st.session_state.model_fitted = True  # เก็บสถานะว่าฝึกแล้ว
                st.rerun()  # รีโหลดหน้าเพื่อให้โมเดลใหม่ถูกโหลด

    # ฟอร์มการทำนายประเภท (Movie หรือ TV Show)
    with st.form("predict_form"):
        duration_type = st.selectbox("Select type of duration:", ["Minutes (Movie)", "Seasons (TV Show)"])
        if duration_type == "Minutes (Movie)":
            duration = st.number_input("Enter the duration in minutes for Movie:", min_value=0)
            input_category = 'Movie'
        else:
            duration = st.number_input("Enter the duration in seasons for TV Show:", min_value=0)
            input_category = 'TV Show'

        if st.form_submit_button("Predict Category"):
            if hasattr(model, "coef_") or st.session_state.get("model_fitted", False):
                # ทำการทำนายประเภท
                input_data = pd.DataFrame([[2025, duration]], columns=['release_year', 'duration_value'])
                input_scaled = scaler.transform(input_data)

                prediction = model.predict(input_scaled)

                # แสดงผลลัพธ์การทำนาย
                if prediction == 1:
                    st.write(f"It is likely a {input_category}.")
                else:
                    st.write(f"It is likely a {input_category}.")
            else:
                st.error("Please train the model first by clicking 'Train Model'.")

# สร้างแถบเลือกหน้า
pages = {
    "Machine Learning": home_page,
    "Demo Machine Learning": about_page,
    "Neural Network": services_page,
    "Demo Neural Network": contact_page,
}

# ใช้ radio button เพื่อเลือกหน้า
page = st.sidebar.radio("Select a page", options=list(pages.keys()))

# แสดงเนื้อหาของหน้าที่เลือก
pages[page]()
