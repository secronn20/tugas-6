import os
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

os.makedirs("uploads",exist_ok=True)
os.makedirs("static",exist_ok=True)


@app.route("/")
def home():
    return render_template(
        "index.html",
        processed=False
    )


@app.route("/cluster",methods=["POST"])
def cluster():

    file=request.files["dataset"]

    if not file:
        return "Upload file CSV terlebih dahulu"

    k=int(request.form["cluster"])

    path=os.path.join(
        "uploads",
        file.filename
    )

    file.save(path)

    df=pd.read_csv(path)


    # ======================
    # AUTO DETEKSI NUMERIK
    # ======================
    X=df.select_dtypes(include=['number'])

    if X.shape[1] <2:
        return "Dataset harus punya minimal dua kolom numerik"


    # scaling
    scaler=StandardScaler()
    X_scaled=scaler.fit_transform(X)


    # ======================
    # ELBOW
    # ======================
    inertia=[]

    for i in range(1,11):

        km=KMeans(
            n_clusters=i,
            random_state=42,
            n_init=10
        )

        km.fit(X_scaled)

        inertia.append(km.inertia_)

    plt.figure()
    plt.plot(range(1,11),inertia,marker='o')
    plt.title("Elbow Method")
    plt.savefig("static/elbow.png")
    plt.close()


    # ======================
    # KMEANS
    # ======================
    model=KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    df["Cluster"]=model.fit_predict(X_scaled)


    # ======================
    # VISUALISASI
    # pakai 2 fitur numerik pertama
    # ======================
    col1=X.columns[0]
    col2=X.columns[1]

    plt.figure(figsize=(8,6))

    sns.scatterplot(
        data=df,
        x=col1,
        y=col2,
        hue='Cluster',
        palette='viridis'
    )

    plt.title("Visualisasi Clustering")
    plt.tight_layout()
    plt.savefig("static/cluster.png")
    plt.close()


    # ======================
    # SUMMARY
    # ======================
    summary=(
        df.groupby("Cluster")
        [X.columns[:2]]
        .mean()
        .round(2)
        .to_html(
            classes="table table-bordered"
        )
    )


    tables=(
        df.head(50)
        .to_html(
            classes="table table-striped",
            index=False
        )
    )


    return render_template(
        "index.html",
        processed=True,
        tables=tables,
        summary=summary
    )


if __name__=="__main__":
    app.run(debug=True)