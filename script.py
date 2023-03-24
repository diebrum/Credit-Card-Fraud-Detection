#Importando bibliotecas

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve, auc
import seaborn as sns

#--------------------------------------------//-------------------------------------

#Importando e examinando os dados
#https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud?resource=download

df = pd.read_csv('card_transdata.csv')
df.head()
#--------------------------------------------//-------------------------------------
#Importando os dados via Kaggle API
"""
try: 
  import opendatasets as od
except:
  !pip install opendatasets
  import opendatasets as od

od.download(
	"https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/download?datasetVersionNumber=1")#Será necessário inserir suas credenciais, elas devem ser geradas no seu perfil no Kaggle.

df = pd.read_csv('/content/credit-card-fraud/card_transdata.csv')
df.head()
"""

#--------------------------------------------//-------------------------------------

#Realizando algumas análises exploratórias (EDA)

plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);

plt.figure(figsize=(16, 6))
heatmap = sns.heatmap(df.corr(), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':18}, pad=12);

plt.savefig('heatmap.png', dpi=300, bbox_inches='tight')
plt.figure(figsize=(16, 6))
mask = np.triu(np.ones_like(df.corr(), dtype=np.bool))
heatmap = sns.heatmap(df.corr(), mask=mask, vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Triangle Correlation Heatmap', fontdict={'fontsize':18}, pad=16);
plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(df.corr()[['fraud']].sort_values(by='fraud', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Fraud Values', fontdict={'fontsize':18}, pad=16);

#--------------------------------------------//-------------------------------------

#Analisando o comportamento da proporção do valor da compra com relação ao ticket médio nas operações fraudulentas


fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(df['ratio_to_median_purchase_price'].where(df['fraud']==1),bins=[0,2,4,6,8,10,12,14,16,18,20,22,24,26])

#--------------------------------------------//-------------------------------------

#Geração de Dashboard interativo com o Sweetviz

try:
  import sweetviz as sv
except:
  !pip install sweetviz
  import sweetviz as sv



dashboard=sv.analyze(df)
dashboard.show_html()

#--------------------------------------------//-------------------------------------

#Organizando os dados para treinamento dos modelos


X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#--------------------------------------------//-------------------------------------

#KNeighbors Classifier

clf=KNeighborsClassifier(3)
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=clf.score(X_test,y_test)
print(score)

cm = confusion_matrix(y_test, prediction, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()
fp_lr, tp_lr, _ = roc_curve(y_test, prediction)
roc_auc_lr = auc(fp_lr, tp_lr)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fp_lr, tp_lr, lw=3, label='ROC-CURVE (area = {:.2f})'.format(roc_auc_lr))
plt.xlabel('True Positive Rate', fontsize = 14)
plt.ylabel('False Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc = 'Lower right', fontsize=13)
plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle = '--')
plt.show()

#--------------------------------------------//-------------------------------------

#Random Forest

clf=RandomForestClassifier(max_depth=5, n_estimators=100, max_features=5)
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=clf.score(X_test,y_test)
print(score)

cm = confusion_matrix(y_test, prediction, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()
fp_lr2, tp_lr2, _ = roc_curve(y_test, prediction)
roc_auc_lr2 = auc(fp_lr2, tp_lr2)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fp_lr2, tp_lr2, lw=3, label='ROC-CURVE (area = {:.2f})'.format(roc_auc_lr2))
plt.xlabel('True Positive Rate', fontsize = 14)
plt.ylabel('False Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc = 'Lower right', fontsize=13)
plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle = '--')
plt.show()

#--------------------------------------------//-------------------------------------

#AdaBoost Classifier

clf=AdaBoostClassifier()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=clf.score(X_test,y_test)
print(score)

cm = confusion_matrix(y_test, prediction, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()
fp_lr3, tp_lr3, _ = roc_curve(y_test, prediction)
roc_auc_lr3 = auc(fp_lr3, tp_lr3)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fp_lr3, tp_lr3, lw=3, label='ROC-CURVE (area = {:.2f})'.format(roc_auc_lr3))
plt.xlabel('True Positive Rate', fontsize = 14)
plt.ylabel('False Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc = 'Lower right', fontsize=13)
plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle = '--')
plt.show()

#--------------------------------------------//-------------------------------------

#Logistic Regression

clf=LogisticRegression()
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
score=clf.score(X_test,y_test)
print(score)

cm = confusion_matrix(y_test, prediction, labels=clf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=clf.classes_)
disp.plot()
plt.show()
fp_lr4, tp_lr4, _ = roc_curve(y_test, prediction)
roc_auc_lr4 = auc(fp_lr4, tp_lr4)

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fp_lr4, tp_lr4, lw=3, label='ROC-CURVE (area = {:.2f})'.format(roc_auc_lr4))
plt.xlabel('True Positive Rate', fontsize = 14)
plt.ylabel('False Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc = 'Lower right', fontsize=13)
plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle = '--')
plt.show()

#--------------------------------------------//-------------------------------------

#Plotando todas as curvas ROC juntas

plt.figure()
plt.xlim([-0.01, 1.00])
plt.ylim([-0.01, 1.01])
plt.plot(fp_lr, tp_lr, lw=3, label='K-neighbors (area = {:.2f})'.format(roc_auc_lr))
plt.plot(fp_lr2, tp_lr2, lw=3, label='Random Forest (area = {:.2f})'.format(roc_auc_lr2))
plt.plot(fp_lr3, tp_lr3, lw=3, label='AdaBoost (area = {:.2f})'.format(roc_auc_lr3))
plt.plot(fp_lr4, tp_lr4, lw=3, label='Logistic Regression (area = {:.2f})'.format(roc_auc_lr4))
plt.xlabel('True Positive Rate', fontsize = 14)
plt.ylabel('False Positive Rate', fontsize = 14)
plt.title('ROC Curve', fontsize=14)
plt.legend(loc = 'Lower right', fontsize=10)
plt.plot([0, 1], [0, 1], lw=2, color='navy', linestyle = '--')
plt.show()

#--------------------------------------------//-------------------------------------


#Matthews Correlation Coefficient (MCC)

truep=228271
falsep=2
falsen=31
truen=21696

def matthews_corr_coef(TP,FP,FN,TN):  
  
  return (TP*TN-FP*FN)/(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5)


mcc=matthews_corr_coef(truep,falsep,falsen,truen)
mcc

#--------------------------------------------//-------------------------------------

#Aplicação no Negócio

#https://www.clientesa.com.br/estatisticas/69061/a-forca-do-cartao-de-credito-no-brasil


ticket_medio=73
ticket_ratio=X_test['ratio_to_median_purchase_price']*ticket_medio
valor_estimado=ticket_ratio.sum()
valor_estimado
dif=y_test-prediction
tam=dif.size
ind=[]

for i in range(tam):
  if dif.iloc[i]!=0:
    ind.append(X_test.iloc[i,2])



valores_n_rec=[ind*73 for ind in ind]
valores_n_rec[0]
soma_n_rec=np.sum(valores_n_rec)
soma_n_rec

#--------------------------------------------//-------------------------------------
