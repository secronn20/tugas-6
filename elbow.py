import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Student_Performance.csv")

features = [
'Hours Studied',
'Previous Scores',
'Sleep Hours',
'Sample Question Papers Practiced',
'Performance Index'
]

X=df[features]

scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

inertia=[]

for k in range(1,11):
    km=KMeans(n_clusters=k,random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.plot(range(1,11),inertia,marker='o')
plt.xlabel("Jumlah Cluster")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()