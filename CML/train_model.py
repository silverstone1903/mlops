import pandas as pd 
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 100

def scorer(y_true, y_preds, is_return = False):
    
    print("-"*20)
    print("F1: {:.4f}".format(f1_score(np.round(y_true), np.round(y_preds))))
    print("Accuracy: {:.4f}".format(accuracy_score(np.round(y_true), np.round(y_preds))))
    print("Recall: {:.4f}".format(recall_score(np.round(y_true), np.round(y_preds))))
    print("Precision: {:.4f}".format(precision_score(np.round(y_true), np.round(y_preds))))
    print((confusion_matrix(np.round(y_true), np.round(y_preds))))
    print("-"*20)
    if is_return:
        return [f1_score(np.round(y_true), np.round(y_preds)), accuracy_score(np.round(y_true), np.round(y_preds)), recall_score(np.round(y_true), np.round(y_preds)), precision_score(np.round(y_true), np.round(y_preds))]
   
feature_ext = False
top_n = 15
seed = 2020

print("[INFO] Data Preprocessing")
data = pd.read_csv("HR-Employee-Attrition.csv")
data["Attrition"] = data["Attrition"].map({'Yes':1, 'No':0})
constant_cols = data.nunique()[data.nunique() == 1].keys().tolist()
data.drop(constant_cols, axis=1, inplace=True)

if feature_ext:
    print("[INFO] Feature Extraction")
    data["RoleChangeYear"] = data["YearsAtCompany"] - data["YearsInCurrentRole"]
    data["PromChangeYear"] = data["YearsAtCompany"] - data["YearsSinceLastPromotion"]
    data["ManagerChangeYear"] = data["YearsAtCompany"] - data["YearsWithCurrManager"]
    data["JobChangeYear"] = data["TotalWorkingYears"] - data["YearsAtCompany"]
    data["AvgCompYear"] = data["TotalWorkingYears"] / data["NumCompaniesWorked"]
    data["DayMonthRate"] = data["DailyRate"] / data["MonthlyRate"]
    data["StartAge"] = data["Age"] - data["TotalWorkingYears"]
    data["WorkLifePercent"] = data["StartAge"] / data["Age"]
    data["is_firstJob"] = np.where(data["NumCompaniesWorked"] == 0, 1, 0).astype(str)
    data["is_TrainedLY"] = np.where(data["TrainingTimesLastYear"] != 0, 1, 0).astype(str)
    data["AnnualSalary"] = data["MonthlyIncome"] * 12 
    data["is_promoted"] = np.where(((data["YearsInCurrentRole"] - data["YearsSinceLastPromotion"]) != 0), 1, 0)
    data["MonthlyIncomeDistMean"] = data["MonthlyIncome"] / data.groupby("DistanceFromHome")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeJobLvlMean"] = data["MonthlyIncome"] / data.groupby("JobLevel")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeEduMean"] = data["MonthlyIncome"] / data.groupby("Education")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeRoleMean"] = data["MonthlyIncome"] / data.groupby("JobRole")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeStockOptMean"] = data["MonthlyIncome"] / data.groupby("StockOptionLevel")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeOTMean"] = data["MonthlyIncome"] / data.groupby("OverTime")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeDepMean"] = data["MonthlyIncome"] / data.groupby("Department")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeEduFieldMean"] = data["MonthlyIncome"] / data.groupby("EducationField")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeAgeMean"] = data["MonthlyIncome"] / data.groupby("Age")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeGenMean"] = data["MonthlyIncome"] / data.groupby("Gender")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeisFJMean"] = data["MonthlyIncome"] / data.groupby("is_firstJob")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeTrvlMean"] = data["MonthlyIncome"] / data.groupby("BusinessTravel")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeMaritMean"] = data["MonthlyIncome"] / data.groupby("MaritalStatus")["MonthlyIncome"].transform("mean")
    data["MonthlyIncomeWLB"] = data["MonthlyIncome"] / data.groupby("WorkLifeBalance")["MonthlyIncome"].transform("mean")
    data.loc[data.AvgCompYear.isna(), "AvgCompYear"] = 0

print("[INFO] Data Preprocessing")
target = ["Attrition"]
num_cols = [
    num for num in data.select_dtypes(exclude=["O"]).columns.tolist()
    if num not in target + ["EmployeeNumber"] + constant_cols
]
cat_cols = [
    cat for cat in data.select_dtypes("O").columns.tolist()
    if cat not in constant_cols
]

scale_cols = [
    c for c in data.nunique()[data.nunique() > 10].keys().tolist()
    if c not in ["EmployeeNumber"] + target]

df = data.copy()
df = df.replace([np.inf, -np.inf], np.nan)

for scl in scale_cols:
    df[scl] = pd.DataFrame(StandardScaler().fit_transform(
        df[scl].fillna(-1).values.reshape(-1, 1)),
                           columns=[scl])

le = LabelEncoder()
for cat in cat_cols:
    df[cat] = le.fit_transform(df[cat])
print("[INFO] Segmentation")
ncls_cat = 2
model_cat = KMeans(n_clusters=ncls_cat, random_state=2020)
model_cat.fit(df[cat_cols])

ncls_num = 3
model_num = KMeans(n_clusters=ncls_num, random_state=2020)
model_num.fit(df[num_cols])

df["clsNum"] = model_num.predict(df[num_cols])
df["clsCat"] = model_cat.predict(df[cat_cols])
("[INFO] Modelling")
target = "Attrition"
train_cols = [c for c in df.columns
              if c not in [target]]

x_train, x_test, y_train, y_test = train_test_split(df[train_cols],
                                                    df[target],
                                                    test_size=0.2,
                                                    random_state=2020,
                                                    stratify=df[target])

(x_train.shape, y_train.shape), (x_test.shape, y_test.shape)

model = RandomForestClassifier(n_estimators=150,
                               class_weight="balanced",
                               min_samples_leaf=10,
                               min_samples_split=5,
                               random_state=seed)

model.fit(x_train, y_train)
model_preds = model.predict_proba(x_test)[:, 1]

print("[INFO] Model Metrics")
f1_train, acc_train, rec_train, prec_train = scorer(y_train,
                                                    model.predict(x_train),
                                                    is_return=True)
f1_pred, acc_pred, rec_pred, prec_pred = scorer(y_test,
                                                model_preds,
                                                is_return=True)
print("[INFO] Feature Importance Calculation")
feature_df = pd.DataFrame(list(zip(train_cols, model.feature_importances_)),
                          columns=["feature", "importance"])
feature_df = feature_df.sort_values(
    by='importance',
    ascending=False,
)

axis_fs = 14
title_fs = 18
sns.set(style="whitegrid")
ax = sns.barplot(x="importance", y="feature", data=feature_df.head(top_n))
ax.set_xlabel('Importance', fontsize=axis_fs)
ax.set_ylabel('Feature', fontsize=axis_fs)
ax.set_title('Random Forest Feature Importance',
             fontsize=title_fs)
plt.tight_layout()
plt.savefig("rf_feature_importance.png", dpi=250)
plt.close()

print("[INFO] Metrics")
with open("rf_metrics.txt", 'w') as outfile:
    outfile.write(
        "Train Metrics\nF1: %0.5f \nAccuracy: %0.5f \nRecall: %0.5f \nPrecision: %0.5f \n"
        % (f1_train, acc_train, rec_train, prec_train))
    outfile.write(
        "Test Metrics\nF1: %0.5f \nAccuracy: %0.5f \nRecall: %0.5f \nPrecision: %0.5f \n"
        % (f1_pred, acc_pred, rec_pred, prec_pred))


print("[INFO] Done!\n")



