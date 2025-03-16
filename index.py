import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

    

def services_page():
    st.title("Neural Network")
    st.write("Neural Network.")
    
def contact_page():
    st.title("Demo Neural Network")
    st.write("Demo Neural Network.")

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
