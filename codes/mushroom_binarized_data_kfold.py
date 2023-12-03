import fcalc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

#load the dataset
df = pd.read_csv('data_sets/mushrooms.csv')
df = df.sample(n=1000, random_state=None, axis=0)

#Handle the missing data values
df.replace('?', np.nan).fillna(method='pad')

# one hot encoding
df['class'] = [x == 'e' for x in df['class']]

column_names = [ "cap-shape", "cap-surface", "cap-color", "bruises","odor", "gill-attachment",
                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root","stalk-surface-above-ring",
	            "stalk-surface-below-ring","stalk-color-above-ring","stalk-color-below-ring","veil-type",
	            "veil-color","ring-number","ring-type","spore-print-color","population","habitat"]

X = pd.get_dummies(df[column_names[:-1]], prefix=column_names[:-1]).astype(bool)
y = df['class']

#kFold
kf = KFold(n_splits=5, random_state=None)
accuracy=[]
f1=[]
alpha_num=[]

for c in range(10):
  a = 0
  f = 0
  for i, (train_index, test_index) in enumerate(kf.split(X,y)):
       print(f"Fold {i}:")
       X_train=X.iloc[train_index]
       y_train=y.iloc[train_index]
       X_test=X.iloc[test_index]
       y_test = y.iloc[test_index]

       bin_cls = fcalc.classifier.BinarizedBinaryClassifier(X_train.values, y_train.to_numpy(), method="standard-support",alpha=c/10000)
       bin_cls.predict(X_test.values)
       a+=accuracy_score(y_test, bin_cls.predictions)
       #f+=f1_score(y_test, bin_cls.predictions)

       print(accuracy_score(y_test, bin_cls.predictions))
       #print(f1_score(y_test, bin_cls.predictions))

  accuracy.append(a/5)
  #f1.append(f/5)
  alpha_num.append(c/10000)

  print("average accuracy:", a/5)
  #print("average f1 score:", f/5)


max_index, max_number = max(enumerate(accuracy), key=operator.itemgetter(1))

print("\n")
print("the biggest accuracy:",max_number)
print("alpha:",alpha_num[max_index])

plt.title('mushroom_binarized')
plt.ylabel('accuracy')
plt.xlabel('alpha')
plt.plot(alpha_num,accuracy)
plt.show()





