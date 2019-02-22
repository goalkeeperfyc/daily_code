#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 21:37:27 2019

@author: fangyucheng
Email: 664947387@qq.com
"""

"""
ticket_id - unique identifier for tickets
agency_name - Agency that issued the ticket
inspector_name - Name of inspector that issued the ticket
violator_name - Name of the person/organization that the ticket was issued to
violation_street_number, violation_street_name, violation_zip_code - Address where the violation occurred
mailing_address_str_number, mailing_address_str_name, city, state, zip_code, non_us_str_code, country - Mailing address of the violator
ticket_issued_date - Date and time the ticket was issued
hearing_date - Date and time the violator's hearing was scheduled
violation_code, violation_description - Type of violation
disposition - Judgment and judgement type
fine_amount - Violation fine amount, excluding fees
admin_fee - $20 fee assigned to responsible judgments
state_fee - $10 fee assigned to responsible judgments
late_fee - 10% fee assigned to responsible judgments
discount_amount - discount applied, if any
clean_up_cost - DPW clean-up or graffiti removal cost
judgment_amount - Sum of all fines and fees
grafitti_status - Flag for graffiti violations

compliance [target variable for prediction] 
 Null = Not responsible
 0 = Responsible, non-compliant
 1 = Responsible, compliant

"""


"""
Your predictions will be given as the probability that the corresponding blight ticket will be paid on time.
The evaluation metric for this assignment is the Area Under the ROC Curve (AUC).
Your grade will be based on the AUC score computed for your classifier. 
A model which with an AUROC of 0.7 passes this assignment, over 0.75 will recieve full points.
For this assignment, create a function that trains a model to predict blight ticket compliance in Detroit using train.csv. 
Using this model, return a series of length 61001 with the data being the probability that each corresponding ticket from test.csv will be paid, 
and the index being the ticket_id.
"""


import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


file_path = '/Users/fangyucheng/documents/code/python_code/first_ml_project/dataset'
train_file = file_path + '/train.csv'
address_file = file_path + '/addresses.csv'
test_file = file_path + '/test.csv'

train_data = pd.read_csv(train_file, encoding='latin1')
address_data = pd.read_csv(address_file, encoding='latin1')
test_data = pd.read_csv(test_file, encoding='latin1')
address_data['address'].value_counts()
#the compose of every column
train_data.columns.values
train_data['country'].value_counts()
train_data['compliance'].value_counts()
train_data['judgment_amount'].value_counts()
train_data['late_fee'].value_counts()
train_data = train_data.dropna(subset=['compliance'])

#delete useless column
delete_list = ["payment_amount", "payment_date", "payment_status", 
               "balance_due", "collection_status", "compliance_detail"]

for line in delete_list:
    del train_data[line]

#merge dataframe
train_data = pd.merge(train_data, address_data, how='left')
train_data['address'].value_counts()
#train_data['country'].count()
#df = train_data.group_by['country'].nunique()

headers = train_data.columns.values
headers_list = list(headers)

for line in headers_list:
    print(line, len(train_data[line].value_counts()))

keep_list = []
drop_list = ['agency_name', 'mailing_address_str_number',
             'violation_street_number', 'violation_street_name',
             'mailing_address_str_name', 'address',  'violation_zip_code',
             'state_fee', 'late_fee', 'ticket_issued_date', 'hearing_date', 
             'fine_amount', 'clean_up_cost', 'disposition', 'grafitti_status',
             'violation_code', 'city']

remain_list = list(set(headers_list) - set(drop_list))

#violation 
new_df = train_data[['violation_description', 'violation_code']]
violation = new_df.drop_duplicates()

#these are equal in value
real_drop = ['admin_fee', 'grafitti_status', 'clean_up_cost', 'violation_zip_code',
             'state_fee']

#these are unrelated with paid
real_drop2 = ['inspector_name','violation_description','violator_name']

#extend
real_drop.extend(real_drop2)

for line in real_drop:
    del train_data[line]

headers = train_data.columns.values
headers_list = list(headers)

for line in headers_list:
    print(line, len(train_data[line].value_counts()))
    
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

#def dummify(column, value):
#    if X[column] == value:
#        return 1
#    else:
#        return 0
    
#change categorical variables to binary 
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
    print(value, len(X[value].value_counts()))

del X['disposition']

for key, value in disposition_dict.items():
    X[key] = X[value]
    del X[value]

X_train, X_test, y_train, y_test = train_test_split(X, y)

del X_train['ticket_id']
del X_test['ticket_id']
del X['ticket_id']

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