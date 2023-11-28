import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
import utils as helper
warnings.simplefilter(action="ignore")
# ss
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

data = pd.read_csv('../OkulProje/database/diabetes.csv')

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()


# BASE MODEL KURULUMU


y = data["OUTCOME"]
X = data.drop("OUTCOME", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=28)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)


print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

# Accuracy: 0.77 for random_state=17
# Recall: 0.706
# Precision: 0.59
# F1: 0.64
# Auc: 0.75

# Accuracy: 0.74 for random_state=32
# Recall: 0.667
# Precision: 0.57
# F1: 0.62
# Auc: 0.72

# Accuracy: 0.8 for random_state=12
# Recall: 0.768
# Precision: 0.63
# F1: 0.69
# Auc: 0.79

# Accuracy: 0.81 for random_state=25
# Recall: 0.68
# Precision: 0.72
# F1: 0.7
# Auc: 0.78

# Accuracy: 0.73 for random_statement=28
# Recall: 0.632
# Precision: 0.54
# F1: 0.58
# Auc: 0.7

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    #plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)

data = pd.read_csv('../OkulProje/database/diabetes.csv')

helper.check_df(data)

# Kolon isimlerini büyültüyorum ki sorgulama vs yaparken yazmak, okumak kolay olsun.
data.columns = [col.upper() for col in data.columns]
data.head()

#Outlier değerleri grafik üzerinde görebilmek için
f, ax = plt.subplots(figsize=(20,20)) #f->figure and ax->axis
fig = sns.boxplot(data=data, orient="h") #horizontally (grafiği yatayda alabilmek için)
#plt.show()


sns.clustermap(data.corr(), annot = True, fmt = ".2f")
#plt.show()

cat_cols, num_cols, cat_but_car = helper.grab_col_names(data)

#numerik kolonlarin birbiriyle iliskisi
for col in num_cols:
    helper.target_summary_with_num(data, "OUTCOME", col)

for col in num_cols:
    helper.num_summary(data, col, plot=True)

#kategorik degisken analizi yani sadece outcome

helper.cat_summary(data, "OUTCOME")

#0 değerlerini NaN olarak değiştirelim.
nan_col=['GLUCOSE','BLOODPRESSURE','SKINTHICKNESS', 'INSULIN', 'BMI']
data[nan_col]=data[nan_col].replace(0, np.NaN)
#kaç boş değer olduğunu gördük ki deri kalınlığı ve insülin değerinde fazlasıyla boş değer var.
data.isnull().sum()


# "Outcome" değeri 1 olan gözlemler için medyan değer hesaplama
median_value_outcome_1 = data.loc[data['OUTCOME'] == 1].median()
# "Outcome" değeri 0 olan gözlemler için medyan değer hesaplama
median_value_outcome_0 = data.loc[data['OUTCOME'] == 0].median()
# Boş değerleri doldurma
data.loc[data['OUTCOME'] == 1] = data.loc[data['OUTCOME'] == 1].fillna(median_value_outcome_1)
data.loc[data['OUTCOME'] == 0] = data.loc[data['OUTCOME'] == 0].fillna(median_value_outcome_0)

data.isnull().sum()



for col in num_cols:
  print(col,"için LOW LIMIT, UP LIMIT değerleri = ", helper.outlier_thresholds(data,col),"\n")

for col in num_cols:
  print(col,"için outlier kontrolü", helper.check_outlier(data,col),"\n")


#Bu çıktıdan alınan değerlere göre hamilelik (PREGNANCIES) 14-17 aralığındakiler outlier,
#Glikoz değerleri için 0 görünen kısımlar null (glikoz 0 olamaz),
#Kan basıncı değeri (BLOODPRESSURE) 0 olamaz null değerler var.
#BMI değerleri 0 olamaz null değerler var.
#Deri kalınlığı min 0.6 mm, max 2.4mm dir. 0 olan değerler null değerdir.
for col in num_cols:
  print(col,"için var olan outlierlar\n", helper.grab_outliers(data,col))

#replace_with_thresholds fonksiyonunu uygulamadan önce yukarıda kararını verdiğimiz 0 değerleri null yapmalıyız ki onları da outlier olarak görmesin.

#şimdi kalan aykırı değerlerin değişimini sağlayabiliriz.
for col in num_cols:
  print(helper.replace_with_thresholds(data, col))


plt.hist(data, bins=10, edgecolor='black')  # 'bins' parametresiyle aralık sayısını belirleyebilirsiniz
plt.xlabel('Değer Aralığı')
plt.ylabel('Frekans')
plt.title('Veri Seti Histogramı')
#plt.show()


#FEATURE ENGINEERING

# Yaş değişkenini kategorilere ayırıp yeni yaş değişkeni oluşturulması
data.loc[(data["AGE"] >= 18) & (data["AGE"] <= 32), "NEW_AGE_CAT"] = "Young"
data.loc[(data["AGE"] >  32) & (data["AGE"] <  50), "NEW_AGE_CAT"] = "Adult"
data.loc[(data["AGE"] >= 50), "NEW_AGE_CAT"] = "Mature"

# BMI 18,5 aşağısı underweight, 18.5 ile 24.9 arası normal, 24.9 ile 29.9 arası Overweight ve 30 üstü obez
data['NEW_BMI'] = pd.cut(x=data['BMI'], bins=[0, 18.5, 24.9, 29.9, 100],labels=["Underweight", "Healthy", "Overweight", "Obese"])

# Glukoz degerini kategorik değişkene çevirme
data["NEW_GLUCOSE"] = pd.cut(x=data["GLUCOSE"], bins=[0, 70, 140, 200, 300], labels=["Low", "Healthy", "Prediabetes", "Diabetes"])

# Deri Kalinligi degerini kategorik degiskene cevirme mm turunden
data["NEW_SKIN_THIC"] = pd.cut(x=data["SKINTHICKNESS"], bins=[1.25, 2, 2.5, 3.25], labels=["Thin", "Healthy", "Thick"])

# # Yaş ve beden kitle indeksini bir arada düşünerek kategorik değişken oluşturma 3 kırılım yakalandı
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "UnderweightYoung"
data.loc[(data["BMI"] < 18.5) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "UnderweightAdult"
data.loc[(data["BMI"] < 18.5) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "UnderweightMature"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "HealthyYoung"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "HealthyAdult"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "HealthyMature"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "OverweightYoung"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "OverweightAdult"
data.loc[((data["BMI"] >= 25) & (data["BMI"] < 30)) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "OverweightMature"
data.loc[(data["BMI"] > 30) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_BMI_NOM"] = "ObeseYoung"
data.loc[(data["BMI"] > 30) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_BMI_NOM"] = "ObeseAdult"
data.loc[(data["BMI"] > 30) & (data["AGE"] >= 50), "NEW_AGE_BMI_NOM"] = "ObeseMature"

# Yaş ve Glikoz değerlerini bir arada düşünerek kategorik değişken oluşturma
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "LowYoung"
data.loc[(data["GLUCOSE"] < 70) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "LowAdult"
data.loc[(data["GLUCOSE"] < 70) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "LowMature"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyYoung"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "HealthyAdult"
data.loc[((data["GLUCOSE"] >= 70) & (data["GLUCOSE"] < 140)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "HealthyMature"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesYoung"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesAdult"
data.loc[((data["GLUCOSE"] >= 140) & (data["GLUCOSE"] <= 200)) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "PrediabetesMature"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] >= 18) & (data["AGE"] <= 32)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesYoung"
data.loc[(data["GLUCOSE"] > 200) & ((data["AGE"] > 32) & (data["AGE"] < 50)), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesAdult"
data.loc[(data["GLUCOSE"] > 200) & (data["AGE"] >= 50), "NEW_AGE_GLUCOSE_NOM"] = "DiabetesMature"

# Deri Kalinligi ve Beden Kitle Indeksi degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThin"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightHealthy"
data.loc[(data["BMI"] < 18.5) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "UnderweightThick"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThin"
data.loc[((data["BMI"] >= 18.5) & (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
#data.loc[((data["BMI"] >= 18.5) % (data["BMI"] < 25)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "Healthy_Healthy"
data.loc[((data["BMI"] >= 18.5)) & (data["BMI"] < 25) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyThick"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThin"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightHealthy"
data.loc[((data["BMI"] >= 25) % (data["BMI"] < 30)) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "OverweightThick"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThin"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseHealthy"
data.loc[(data["BMI"] > 30) & ((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3.25)), "NEW_BMI_SKIN_THIC_NOM"] = "ObeseThick"

# Deri Kalinligi ve Insulin degerlerini bir arada dusunerek kategorik degisken olusturma
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThinNormal"
data.loc[((data["SKINTHICKNESS"] > 1.25) & (data["SKINTHICKNESS"] <= 2)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThinAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyNormal"
data.loc[((data["SKINTHICKNESS"] > 2) & (data["SKINTHICKNESS"] <= 2.5)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "HealthyAbNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & ((data["INSULIN"] >= 16) & (data["INSULIN"] <=160)), "NEW_BMI_SKIN_THIC_NOM"] = "ThickNormal"
data.loc[((data["SKINTHICKNESS"] > 2.5) & (data["SKINTHICKNESS"] <= 3)) & (data["INSULIN"] >=160), "NEW_BMI_SKIN_THIC_NOM"] = "ThickAbNormal"



# İnsulin Değeri ile Kategorik değişken türetmek
def set_insulin(dataframe, col_name="INSULIN"):
    if 16 <= dataframe[col_name] <= 166:
        return "Normal"
    else:
        return "Abnormal"


# ENCODING

# Değişkenlerin tiplerine göre ayrılması işlemi
cat_cols, num_cols, cat_but_car = helper.grab_col_names(data)

# LABEL ENCODING
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in data.columns if data[col].dtypes == "O" and data[col].nunique() == 2 ]
binary_cols
data.head(20)
for col in binary_cols:
    data = label_encoder(data, col)

# One-Hot Encoding İşlemi
# cat_cols listesinin güncelleme işlemi
cat_cols = [col for col in cat_cols if col not in binary_cols and col not in ["OUTCOME"]]
cat_cols

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

data = one_hot_encoder(data, cat_cols, drop_first=True)

data.head()


