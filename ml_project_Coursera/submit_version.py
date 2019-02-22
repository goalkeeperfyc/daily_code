
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

train_file = 'train.csv'
test_file = 'readonly/test.csv'

train_data = pd.read_csv(train_file, encoding='latin1')
test_data = pd.read_csv(test_file, encoding='latin1')

train_data = train_data.dropna(subset=['compliance'])

#delete useless column
delete_list = ["payment_amount", "payment_date", "payment_status", 
               "balance_due", "collection_status", "compliance_detail"]

for line in delete_list:
    del train_data[line]

#these are equal in value
real_drop = ['admin_fee', 'grafitti_status', 'clean_up_cost', 'violation_zip_code',
             'state_fee']

#these are unrelated with paid
real_drop2 = ['inspector_name','violation_description','violator_name']

#extend
real_drop.extend(real_drop2)

for line in real_drop:
    del train_data[line]
  
del train_data['non_us_str_code']
del train_data['country']

new_train_data = train_data[['ticket_id', 'agency_name', 'disposition', 
                             'fine_amount', 'late_fee', 'compliance', 
                             'discount_amount', 'judgment_amount']]

new_train_data = new_train_data.set_index(new_train_data['ticket_id'])

y = new_train_data['compliance']
y = y.astype('int')

del new_train_data['compliance']
X = new_train_data

X['bool1'] = X['agency_name'] == 'Buildings, Safety Engineering & Env Department'
X['BSEED'] = X['bool1'].astype('int')
del X['bool1']

X['bool1'] = X['agency_name'] == 'Department of Public Works'
X['Public_works'] = X['bool1'].astype('int')
del X['bool1']

X['bool1'] = X['agency_name'] == 'Detroit Police Department'
X['Police Department'] = X['bool1'].astype('int')
del X['bool1']

X['bool1'] = X['agency_name'] == 'Health Department'
X['Health_Department'] = X['bool1'].astype('int')
del X['bool1']

X['bool1'] = X['agency_name'] == 'Neighborhood City Halls'
X['Neighborhood'] = X['bool1'].astype('int')
del X['bool1']
del X['agency_name']

disposition_dict = {"Default": "Responsible by Default", 
                    "Admission": "Responsible by Admission",
                    "Determination": "Responsible by Determination",
                    "Deter": "Responsible (Fine Waived) by Deter"}

for key, value in disposition_dict.items():
    X['bool'] = X['disposition'] == value
    X[value] = X['bool'].astype('int')
    del X['bool']

del X['disposition']

for key, value in disposition_dict.items():
    X[key] = X[value]
    del X[value]

X_train, X_test, y_train, y_test = train_test_split(X, y)

del X_train['ticket_id']
del X_test['ticket_id']

RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(X_train, y_train)
RFC.score(X_test, y_test)

y_train_score = RFC.predict_proba(X_train)
y_test_score = RFC.predict_proba(X_test)

y_train_df = pd.DataFrame(y_train_score)
y_train_score2 = y_train_df[1]

y_test_df = pd.DataFrame(y_test_score)
y_test_score2 = y_test_df[1]

train_roc_score = roc_auc_score(y_train, y_train_score2)
test_roc_score = roc_auc_score(y_test, y_test_score2)

#make test data ready
test_data = test_data.set_index(test_data['ticket_id'])

new_test_data = test_data[['ticket_id', 'fine_amount', 'late_fee', 'discount_amount', 
                           'judgment_amount','disposition', 'agency_name']]

new_test_data['bool1'] = new_test_data['agency_name'] == 'Buildings, Safety Engineering & Env Department'
new_test_data['BSEED'] = new_test_data['bool1'].astype('int')
del new_test_data['bool1']

new_test_data['bool1'] = new_test_data['agency_name'] == 'Department of Public Works'
new_test_data['Public_works'] = new_test_data['bool1'].astype('int')
del new_test_data['bool1']

new_test_data['bool1'] = new_test_data['agency_name'] == 'Detroit Police Department'
new_test_data['Police Department'] = new_test_data['bool1'].astype('int')
del new_test_data['bool1']

new_test_data['Neighborhood'] = 0
new_test_data['Health_Department'] = 0
del new_test_data['ticket_id']

for key, value in disposition_dict.items():
    new_test_data['bool'] = new_test_data['disposition'] == value
    new_test_data[key] = new_test_data['bool'].astype('int')
    del new_test_data['bool']

del new_test_data['agency_name']
del new_test_data['disposition']

test_score = RFC.predict_proba(new_test_data)

test_score_df = pd.DataFrame(test_score)
test_score_df = test_score_df.set_index(test_data['ticket_id'])
result = test_score_df[1]
result = result.rename('compliance')
result = result.astype('float32')