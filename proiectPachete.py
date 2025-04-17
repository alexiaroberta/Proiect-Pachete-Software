import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split
import math
import matplotlib.pyplot as plt
import seaborn as sb
import geopandas as gpd

# Incarcarea datelor
@st.cache_data
def load_data():
    df = pd.read_csv("healthcare-dataset-stroke-data.csv")
    return df

df = load_data()

menu = st.sidebar.selectbox("Selectează opțiunea", ["Analiza Datelor despre AVC", "Geopandas - Slovacia"])

if menu == "Analiza Datelor despre AVC":
    st.title('Analiza Datelor despre Accidente Vasculare Cerebrale')

    # Vizualizare date
    st.subheader('Datele initiale procesate')
    st.dataframe(df)

    st.write(f"Dimensiunea DataFrame-ului: {df.shape}")

    # Tratarea valorilor lipsă
    st.subheader('Tratarea valorilor lipsă')
    imputer_choice = st.selectbox("Alege metoda pentru imputarea valorilor lipsă", ('Mean', 'Median', 'Most Frequent'))

    # Definirea coloanelor numerice si categorice
    numerical_cols = ['age', 'avg_glucose_level', 'bmi']
    categorical_cols = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

    # Crearea si aplicare imputerului
    numerical_imputer = SimpleImputer(strategy='mean' if imputer_choice == 'Mean' else 'median')
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df.loc[:, numerical_cols] = numerical_imputer.fit_transform(df[numerical_cols])
    df.loc[:, categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])

    # Afisare date dupa imputare
    st.write(f"* Date după imputarea valorilor lipsă folosind {imputer_choice}:")
    st.dataframe(df)

    # Tratarea valorilor extreme
    st.subheader('Tratarea valorilor extreme')

    extreme_value_treatment = st.selectbox('Alege metoda pentru tratamentul valorilor extreme', ('Înlocuire cu medie', 'Eliminare'))

    if extreme_value_treatment == 'Înlocuire cu medie':
        for col in numerical_cols:
            if col == 'age':
                mean_value = int(df['age'].mean())
                df['age'] = np.where((df['age'] < 10) | (df['age'] > 100), mean_value, df['age'])
            else:
                mean_value = df[col].mean()
                df[col] = np.where(df[col] > df[col].max() * 1.5, mean_value, df[col])
    elif extreme_value_treatment == 'Eliminare':
        for col in numerical_cols:
            if col == 'age':
                df = df[(df['age'] >= 10) & (df['age'] <= 100)]
            else:
                df = df[df[col] <= df[col].max() * 1.5]

    # Afisare date dupa tratarea valorilor extreme
    st.write(f"* Date după tratarea valorilor extreme (metoda: {extreme_value_treatment}):")
    st.dataframe(df)

    # Generarea histogramelor pentru variabilele numerice
    # Calculăm numărul de rânduri și coloane necesare pentru subgrafice
    st.subheader("Distributia variabilelor numerice")
    n_cols = 3
    n_rows = math.ceil(len(numerical_cols) / n_cols)

    # Setăm dimensiunea figurii
    plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    # Iterăm prin fiecare coloană numerică și generăm histogramă
    for i, col in enumerate(numerical_cols):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.hist(df[col].dropna(), bins=30, edgecolor='black', color='skyblue')
        plt.title(f'Distribuția: {col}')
        plt.xlabel(col)
        plt.ylabel('Frecvență')

    # Ajustăm aspectul subgraficele pentru a evita suprapunerea
    plt.tight_layout()

    # Afișăm graficul în aplicația Streamlit
    st.pyplot(plt)

    st.subheader("Distributia variabilelor categorice")
    for col in categorical_cols:
        plt.figure(figsize=(8, 4))
        unique_count = df[col].nunique()  # Calculăm numărul de valori unice din coloana curentă

        if unique_count > 10:  # Dacă numărul de categorii este mai mare de 10
            # Selectăm top 10 categorii după frecvență
            top_categories = df[col].value_counts().nlargest(10)
            # Construim un barplot folosind cele mai frecvente 10 categorii
            sb.barplot(x=top_categories.index, y=top_categories.values, palette='viridis')
            plt.title(f"Top 10 valori pentru {col}")  # Setăm titlul graficului
            plt.xlabel(col)  # Etichetă pentru axa x
            plt.ylabel("Frecvență")  # Etichetă pentru axa y
            plt.xticks(rotation=45)  # Rotim etichetele de pe axa x pentru o vizibilitate mai bună
        else:
            # Dacă numărul de categorii este mic, construim direct un countplot
            sb.countplot(x=col, data=df, palette='viridis')
            plt.title(f'Distribuția: {col}')  # Setăm titlul graficului
            plt.xlabel(col)  # Etichetă pentru axa x
            plt.ylabel('Frecvență')  # Etichetă pentru axa y
            plt.xticks(rotation=45)  # Rotim etichetele de pe axa x

        plt.tight_layout()  # Ajustăm automat spațiile între subgrafice pentru a evita suprapunerea
        st.pyplot(plt)

    st.subheader("Pairplot pentru variabilele numerice")
    def plot_pairplot_numeric(df, numeric_cols):
        sb.pairplot(df[numeric_cols], diag_kind='kde')
        plt.suptitle("Pairplot pentru variabilele numerice", y=1.02)
        st.pyplot(plt)
    plot_pairplot_numeric(df, numerical_cols)

    st.subheader("Boxplot pentru variatia nivelul mediu de glucoză în funcție de istoricul de hipertensiune")
    def plot_boxplot_cat_numeric(df, cat_col, num_col):
        """
        Creează un boxplot care compară distribuția unei variabile numerice (num_col)
        în funcție de o variabilă categorică (cat_col).
        """
        plt.figure(figsize=(8, 4))
        sb.boxplot(data=df, x=cat_col, y=num_col, palette='viridis')
        plt.title(f"Boxplot pentru {num_col} în funcție de {cat_col}")
        plt.xlabel(cat_col)
        plt.ylabel(num_col)
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(plt)

    plot_boxplot_cat_numeric(df, cat_col='hypertension', num_col='avg_glucose_level')

    #matrice de corelatie
    numerical_cols_with_stroke = numerical_cols + ['stroke']
    corr_matrix = df[numerical_cols_with_stroke].corr()

    st.subheader("Matricea de Corelație pentru Variabilele Numerice si Stroke")
    plt.figure(figsize=(6, 4))
    sb.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Corelația dintre Stroke și variabilele numerice")
    plt.tight_layout()
    st.pyplot(plt)

    # Codificarea variabilelor categorice
    st.subheader("Codificarea variabilelor categorice")

    categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    encoder = LabelEncoder()

    dict_codif = {}
    for col in categorical_columns:
        df[col] = encoder.fit_transform(df[col])
        dict_codif[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

    # Afișare mapare valori originale -> valori codificate
    st.write("* Maparea variabilelor categorice după codificare:")
    for col, valoare in dict_codif.items():
        st.write(f"{col}: {valoare}")

    # Afisare date dupa codificare
    st.write("* Date după codificarea variabilelor categorice:")
    st.dataframe(df)

    # Filtrare și sortare
    st.subheader('Filtrare și sortare')
    st.write('* Filtrare pe varsta')

    # Filtrare pe varsta
    age_filter = st.slider('Selectează vârsta minimă', int(df['age'].min()), int(df['age'].max()), 20)
    filtered_data = df[df['age'] >= age_filter]

    # Afisare date filtrate
    st.write(f"* Datele filtrate pentru varsta minima {age_filter}:")
    st.dataframe(filtered_data)

    # Filtrare folosind loc: selectăm persoanele care nu au fumat niciodată
    non_smokers = df.loc[df['smoking_status'] == dict_codif['smoking_status']['never smoked']]

    # Afișare rezultate
    st.write("* Datele persoanelor care nu au fumat niciodată:")
    st.dataframe(non_smokers)

    # Sortare pe coloane
    st.write('* Sortare pe coloane')

    # Alege coloana pentru sortare
    sort_column = st.selectbox('Alege coloana pentru sortare', df.columns)

    # Alege ordinea de sortare (crescător sau descrescător)
    sort_order = st.selectbox('Alege ordinea de sortare', ('Crescatoare', 'Descrescatoare'))

    # Aplicarea sortării în funcție de alegerea utilizatorului
    sorted_data = filtered_data.sort_values(by=sort_column, ascending=True if sort_order == 'Crescatoare' else False)

    # Afisare date sortate
    st.write(f"* Datele sortate după {sort_column} în ordine {sort_order.lower()}:")
    st.dataframe(sorted_data)

    #Prelucrări statistice, gruparea și agregarea datalor în pachetul pandas
    st.subheader("Prelucrari statistice, gruparea si agregarea datelor")

    #Numarul de valori unice din fiecare coloana
    st.write("* Numarul de valori unice din fiecare coloana")
    st.dataframe(df.nunique())

    #Statistici descriptive
    statistics = df[numerical_cols].describe()
    st.write("* Statistici descriptive pentru coloanele numerice")
    st.dataframe(statistics)

    #Gruparea după tipul locului de muncă și calcularea mediei vârstei și nivelului de glucoză
    date_grupate1 = df.groupby('work_type')[['age', 'avg_glucose_level']].mean()
    st.write("* Media varstei si a nivelului de glucoza a persoanelor grupate dupa tipul locului de munca")
    st.dataframe(date_grupate1)

    #Media,maximul si minimul bmi pentru fiecare varsta
    st.write("* Media,maximul si minimul bmi pentru fiecare varsta")
    st.dataframe(df.groupby("age")["bmi"].agg(['mean', 'max', 'min']))

    #Gruparea datelor dupa hipertensiune
    date_grupate2 =  df.groupby('hypertension').agg(
        num_persoane=('age', 'count'),
        media_bmi=('bmi', 'mean')
    )

    st.write("* Datele agregate după prezenta hipertensiunii , calculand nr de persoane si media pentru bmi")
    st.dataframe(date_grupate2)

    date_grupate3 = df.groupby('Residence_type').agg(
        num_persoane=('age', 'count'),
        varsta_medie=('age', 'mean')
    )
    st.write("* Numarul total de persoane si varsta medie in fiecare tip de rezidenta:")
    st.dataframe(date_grupate3)

    # Scalarea variabilelor numerice
    st.subheader("Scalarea")
    scalar_choice = st.selectbox("Alege metoda de scalare", ('StandardScaler', 'MinMaxScaler'))
    if scalar_choice == 'StandardScaler':
        scaler = StandardScaler()
    elif scalar_choice == 'MinMaxScaler':
        scaler = MinMaxScaler()

    # Aplicarea scalării doar pe coloanele numerice
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Afisare date dupa scalare
    st.write("* Date după scalare")
    st.dataframe(df)

    # K-MEANS
    st.subheader("K-Means Clustering")

    # Alegerea numarului de clustere
    num_clusters = st.slider("Selectează numărul de clustere (K)", min_value=2, max_value=10, value=3)

    # Aplicarea KMeans
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[numerical_cols])

    st.write("Etichetele clusterelor au fost adăugate în DataFrame:")
    st.dataframe(df[['age', 'avg_glucose_level', 'bmi', 'cluster']])

    # Vizualizare cu PCA
    st.write("Proiecția cu PCA pentru vizualizarea clusterelor: ")
    pca = PCA(n_components=2)
    cluster_data = pca.fit_transform(df[numerical_cols])
    cluster_df = pd.DataFrame(cluster_data, columns=['PC1', 'PC2'])
    cluster_df['cluster'] = df['cluster']

    plt.figure(figsize=(8, 5))
    sb.scatterplot(data=cluster_df, x='PC1', y='PC2', hue='cluster', palette='Set1')
    plt.title(f"Vizualizarea Clusterelor (K={num_clusters}) cu PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    st.pyplot(plt)

    # Afisare medii pe clustere
    st.write("**Medii ale variabilelor numerice în fiecare cluster:**")
    cluster_summary = df.groupby('cluster')[numerical_cols].mean()
    st.dataframe(cluster_summary)

    # REGRESIA LINIARA
    st.subheader('Regresie Liniară')

    # Permite utilizatorului sa aleaga variabila dependenta dintre coloanele numerice
    dependent_var = st.selectbox("Selectează variabila dependentă", numerical_cols)

    # Permite utilizatorului sa aleaga variabilele independente dintre coloanele numerice
    independent_vars = st.multiselect("Selectează variabilele independente", numerical_cols)

    # Verifica daca utilizatorul a ales cel putin o variabila independenta
    if independent_vars:
        # Creaza un subset din datele pentru regresie
        X = df[independent_vars]
        y = df[dependent_var]

        # Impartirea setului de date in training si testare (80% training, 20% test)
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Crearea si antrenarea modelului de regresie liniara
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Prezicerea valorilor pe setul de test
        y_pred = model.predict(X_test)

        # Calcularea erorii si a coeficientilor
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.write(f"Coeficientii regresiei: {model.coef_}")
        st.write(f"Interceptul: {model.intercept_}")
        st.write(f"Erorile medii pătratice (MSE): {mse}")
        st.write(f"Scorul R^2: {r2}")

        # Graficul de comparare intre valorile reale si cele prezise
        plt.figure(figsize=(8, 4))
        plt.scatter(y_test, y_pred, color='blue')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
        plt.title("Comparare valori reale vs prezise")
        plt.xlabel("Valori reale")
        plt.ylabel("Valori prezise")
        plt.tight_layout()
        st.pyplot(plt)

    # REGRESIE LOGISTICA + ROC
    st.subheader("Regresie Logistică - Predicția AVC (Stroke)")

    # Selectam caracteristicile (X) si tinta (y)
    feature_cols = numerical_cols + categorical_columns
    X = df[feature_cols]
    y = df['stroke']

    # Impartirea datelor in train si test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Antrenarea modelului
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # Prezicerea probabilitatilor pentru clasa pozitiva
    y_prob = log_model.predict_proba(X_test)[:, 1]

    # Calcularea ROC și AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)

    # Afisarea scorului AUC
    st.write(f"Scorul AUC: {auc_score:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('Rata fals pozitivă (FPR)')
    plt.ylabel('Rata adevărat pozitivă (TPR)')
    plt.title('Curba ROC - Regresie Logistică')
    plt.legend(loc='lower right')
    plt.tight_layout()
    st.pyplot(plt)

elif menu == "Geopandas - Slovacia":
    st.title("Analiza datelor geospațiale cu GeoPandas - Slovacia")

    st.subheader("Vecinii Slovaciei")
    neighbours = gpd.read_file("slovacia_neighbours.json", encoding='latin-1')
    st.write(neighbours)

    st.subheader("Suprafața țărilor vecine Slovaciei (km²)")
    neighbours.to_crs(epsg=32635, inplace=True)
    neighbours["area_km2"] = neighbours.geometry.area / 1_000_000
    fig, ax = plt.subplots(figsize=(6,3))
    neighbours.plot(column="area_km2", ax=ax, legend=True, legend_kwds={'label': "Aria (km²)"})
    for x, y, label in zip(neighbours.geometry.centroid.x, neighbours.geometry.centroid.y, neighbours['name']):
        ax.text(x, y, label, fontsize=6, ha='center', color='black')
    st.pyplot(fig)
    fig, ax = plt.subplots(figsize=(6, 3))

    st.subheader("Incidența AVC în Slovacia si țările vecine (per 100k locuitori)")
    neighbours.plot(column="stroke_incidence_per_100k", ax=ax, legend=True, cmap="Reds",
                    legend_kwds={'label': "Incidența AVC (per 100k locuitori)", 'orientation': "vertical"})
    for x, y, label in zip(neighbours.geometry.centroid.x, neighbours.geometry.centroid.y, neighbours['name']):
        ax.text(x, y, label, fontsize=6, ha='center', color='black')
    st.pyplot(fig)

