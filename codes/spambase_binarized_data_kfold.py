import fcalc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score

#load the dataset
df = pd.read_csv('data_sets/spambase.csv')
df = df.sample(n=1000, random_state=None, axis=0)

X = df.iloc[:,:-1]
y = df['spam']

column_names=['word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d',
       'word_freq_our', 'word_freq_over', 'word_freq_remove',
       'word_freq_internet', 'word_freq_order', 'word_freq_mail',
       'word_freq_receive', 'word_freq_will', 'word_freq_people',
       'word_freq_report', 'word_freq_addresses', 'word_freq_free',
       'word_freq_business', 'word_freq_email', 'word_freq_you',
       'word_freq_credit', 'word_freq_your', 'word_freq_font', 'word_freq_000',
       'word_freq_money', 'word_freq_hp', 'word_freq_hpl', 'word_freq_george',
       'word_freq_650', 'word_freq_lab', 'word_freq_labs', 'word_freq_telnet',
       'word_freq_857', 'word_freq_data', 'word_freq_415', 'word_freq_85',
       'word_freq_technology', 'word_freq_1999', 'word_freq_parts',
       'word_freq_pm', 'word_freq_direct', 'word_freq_cs', 'word_freq_meeting',
       'word_freq_original', 'word_freq_project', 'word_freq_re',
       'word_freq_edu', 'word_freq_table', 'word_freq_conference',
       'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!',
       'char_freq_$', 'char_freq_#', 'capital_run_length_average',
       'capital_run_length_longest', 'capital_run_length_total']

def mapfun(x):
    if x < a:
        return 'A'
    elif x >= a and x < b:
        return 'B'
    elif x >= b and x < c:
        return 'C'
    else:
        return 'D'

for i in column_names:
    a=np.percentile(X[i], 25)
    b=np.percentile(X[i], 50)
    c=np.percentile(X[i], 75)
    X[i] = X[i].map(mapfun)

#print(X.head())
df = pd.DataFrame(X)
X = pd.get_dummies(df[column_names[:-1]], prefix=column_names[:-1]).astype(bool)

#kFold
kf = KFold(n_splits=5, random_state=None)
accuracy=[]
f1=[]
alpha_num=[]


for c in range(100):
  a = 0
  f = 0
  for i, (train_index, test_index) in enumerate(kf.split(X,y)):
       print(f"Fold {i}:")
       X_train=X.iloc[train_index]
       y_train=y.iloc[train_index]
       X_test=X.iloc[test_index]
       y_test = y.iloc[test_index]

       bin_cls = fcalc.classifier.BinarizedBinaryClassifier(X_train.values, y_train.to_numpy(), method="standard-support",alpha=c/1000)
       bin_cls.predict(X_test.values)
       a+=accuracy_score(y_test, bin_cls.predictions)
       #f+=f1_score(y_test, bin_cls.predictions)

       print(accuracy_score(y_test, bin_cls.predictions))
       #print(f1_score(y_test, bin_cls.predictions))

  accuracy.append(a/5)
  #f1.append(f/5)
  alpha_num.append(c/1000)

  print("average accuracy:", a/5)
  #print("average f1 score:", f/5)


max_index, max_number = max(enumerate(accuracy), key=operator.itemgetter(1))

print("\n")
print("the biggest accuracy:",max_number)
print("alpha:",alpha_num[max_index])

plt.title('spambase_binarized')
plt.ylabel('accuracy')
plt.xlabel('alpha')
plt.plot(alpha_num,accuracy)
plt.show()



