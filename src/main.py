import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model.logistic import LogisticRegression

TEST_RATIO = 0.1
NOPAY_THREASH_HOLD = 0.3

def split_train_test(data, test_ratio):
    np.random.seed(143)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices= shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def modifiy_data():
    pd.set_option('display.max_columns', None)

    df_new = pd.DataFrame()

    df = pd.DataFrame.from_csv('foo.tsv', sep='\t')
    f = df.describe()

    for ix, item in f.iteritems():
        if item['count'] != 0 and item['std'] != 0:
            df_new.insert(len(df_new.columns), item.name, df[item.name])

    drop_item = ['order_id', 'user_id', 'order_status', 'created_at', 'updated_at', 'sub_order_type',
                 'plan_repayment_time',
                 'true_repayment_time', 'is_active', 'loan_time']

    for item in drop_item:
        df_new.drop(item, axis=1, inplace=True)

    df_new.to_csv('foo_modified.csv')

    df = pd.DataFrame.from_csv('foo_modified.csv')
    # print df.describe()
    # print '-------------------------------------------------------------------------------------------'
    imputer = Imputer(strategy='median')
    imputer.fit(df)
    df_fit = imputer.transform(df)
    df_fit = pd.DataFrame(df_fit, columns=df.columns)
    # print df_fit.describe()

    label_col = []
    repay_col = df['repayment_status']
    overdue_col = df['overdue_day']
    for repay, overdue in zip(repay_col, overdue_col):
        if repay == 4:  # and overdue == 0:
            label_col.append(1)
        else:
            label_col.append(0)

    df_fit.drop('overdue_day', axis=1, inplace=True)
    df_fit.drop('repayment_status', axis=1, inplace=True)
    df_fit.insert(len(df_fit.columns), 'label', label_col)

    df_fit.to_csv('foo_final.csv')

modifiy_data()
df = pd.DataFrame.from_csv('foo_final.csv')

train_split, test_split = split_train_test(df, TEST_RATIO)

train_data = train_split.drop('label', axis = 1)
train_label = train_split['label']
test_data = test_split.drop('label', axis = 1)
test_label = test_split['label']

standard_scaler = StandardScaler()
standard_scaler.fit(train_data)
train_data  = standard_scaler.transform(train_data)
test_data = standard_scaler.transform(test_data)

lr = LogisticRegression(max_iter=5000)
lr.fit(train_data, train_label)
#test_predict =  lr.predict(test_data)
test_predict_prob = lr.predict_proba(test_data)
test_predict = []

for nopay_prob, pay_prob in test_predict_prob:
    if nopay_prob > NOPAY_THREASH_HOLD:
        test_predict.append(0)
    else:
        test_predict.append(1)



# gt_predict
pay_nopay = 0
pay_pay = 0
nopay_pay = 0
nopay_nopay = 0
# gt
pay = 0
nopay = 0

for gt, predict in zip(test_label, test_predict):
    if gt == 1 and predict == 1:
        pay_pay += 1
        pay += 1
    elif gt == 1 and predict == 0:
        pay_nopay += 1
        pay += 1
    elif gt == 0 and predict == 1:
        nopay_pay += 1
        nopay += 1
    elif gt == 0 and predict == 0:
        nopay_nopay += 1
        nopay += 1

pay_recall = float(pay_pay) / (pay)
pay_precision = float(pay_pay) / (pay_pay + nopay_pay)

nopay_recall = float(nopay_nopay) / (nopay)
nopay_precision = float(nopay_nopay) / (nopay_nopay + pay_nopay)

print ('pay_recall :{} ({}/{}) pay_precision :{} ({}/{})').format(pay_recall, pay_pay, pay, pay_precision, pay_pay, pay_pay + nopay_pay)
print ('nopay_recall :{} ({}/{}) nopay_precision: {} ({}/{})').format(nopay_recall, nopay_nopay, nopay, nopay_precision, nopay_nopay, nopay_nopay + pay_nopay)













