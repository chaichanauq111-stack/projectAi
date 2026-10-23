from sklearn.datasets import make_regression , make_classification , make_blobs
import pandas as pd

# นำเอาข้อมูลมาใส่
df = pd.read_csv('test_predictions_h2.csv')

# ทดสอบระบบ
X, y, _ = make_regression(n_samples=100, n_features=1, noise=0.1, coef=True)


