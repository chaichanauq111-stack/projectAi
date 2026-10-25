from sklearn.datasets import make_regression , make_classification , make_blobs
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
import warnings
import lightgbm as lgb
warnings.filterwarnings("ignore")

#ขั้นที่ 2 การสำรวจและวิเคราะห์ข้อมูล (Exploratory Data Analysis - EDA)
# นำเอาข้อมูลมาใส่
df = pd.read_csv('test_predictions_h2.csv')
#ข้อมูลอัตราป่วย-ตายต่อแสนประชากร แยกรายจังหวัด
#ยังไม่เปลี่ยนไฟล์เป็น csv เพราะมีปัญหาการอ่านไฟล์ rtf 
df_65 = pd.read_csv('trate_DHF_Total_65.rtf')
df_66 = pd.read_csv('trate_DHF_Total_66.rtf')
df_67 = pd.read_csv('trate_DHF_Total_67.rtf')

# 2. นำข้อมูลทั้ง 3 ปีมารวมกันต่อกันทางแนตั้ง (Concat)
df_cases = pd.concat([df_65, df_66, df_67], ignore_index=True)

# ตรวจสอบผลลัพธ์
display(df_cases.head())
display(df)

# ตรวจสอบข้อมูลเบื้องต้น
df.info()
display(df.describe())
display(df.isnull().sum())

# การกระจายตัวของข้อมูลเชิงหมวดหมู่
# 1. กราฟข้อมูลเชิงหมวดหมู่ (Categorical Features)
cat_features = ['sex', 'region', 'province', 'year']
plt.figure(figsize=(16, 8))
for i, col in enumerate(cat_features):
    plt.subplot(2, 2, i+1)
    if df[col].nunique() <= 10:  # ถ้าหมวดหมู่ไม่เยอะเกินไปค่อยใช้ Pie Chart หรือ Countplot
        x = df[col].value_counts()
        plt.pie(x.values, labels=x.index, autopct='%1.1f%%')
        plt.title(f'Distribution of {col}')
    else:  # ถ้าจังหวัดเยอะเกินไป ให้แสดง Top 10 แทน
        df[col].value_counts().head(10).plot(kind='bar')
        plt.title(f'Top 10 {col}')
plt.tight_layout()
plt.show()

# 2. เพิ่มเติม: กราฟกระจายตัวของสภาพอากาศ (Numerical Features)
num_features = ['ปริมาณน้ำฝน', 'อุณหภูมิ', 'ความชื้นสัมพัทธ์']
plt.figure(figsize=(15, 4))
for i, col in enumerate(num_features):
    plt.subplot(1, 3, i+1)
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# เรียงลำดับตามพื้นที่และเวลา
if 'week' in df.columns:
    df = df.sort_values(by=['province', 'year', 'week']).reset_index(drop=True)

# แปลงข้อมูล Categorical เป็น Category type สำหรับ LightGBM / XGBoost
categorical_cols = ['sex', 'region', 'province']
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype('category')

# สร้าง Lag Features สำหรับสภาพอากาศ (หน่วงเวลา 1-4 สัปดาห์)
weather_cols = ['ปริมาณน้ำฝน', 'อุณหภูมิ', 'ความชื้นสัมพัทธ์']
if 'week' in df.columns and 'province' in df.columns:
    for col in weather_cols:
        if col in df.columns:
            for lag in [1, 2, 4]:
                df[f'{col}_lag_{lag}'] = df.groupby('province')[col].shift(lag)

# ลบค่า NaN ที่เกิดจากการทำ Lag
df_model = df.dropna().copy()

#ขั้นที่ 4 การพัฒนาและเปรียบเทียบแบบจำลอง (Model Development and Comparison)
#ข้อมูลจะถูกแบ่งออกเป็นชุดฝึก (Training Set) 80% และชุดทดสอบ (Testing Set) 20%
# ทดสอบระบบ
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
