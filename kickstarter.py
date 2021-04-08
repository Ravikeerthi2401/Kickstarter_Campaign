# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 10:30:44 2019

@author: Ravi Keerthi
"""


import pandas as pd

datasets = [pd.read_csv("D:/dataset//Kickstarter.csv")]

for i in range(1,10):
    datasets.append(pd.read_csv("D:/dataset//Kickstarter00" + str(i) + ".csv"))

for i in range(10,51):
    datasets.append(pd.read_csv("D:/dataset//Kickstarter0" + str(i) + ".csv"))
    
from functools import reduce

data = reduce(lambda x,y: pd.concat([x,y]),datasets)
data['category'] = data['category'].astype(str).str.rsplit(':').str[2]
data['category'] = data['category'].astype(str).str.rsplit('"').str[1]
data.to_csv("newKickstarter.csv")

data =  pd.read_csv("C:/Users//Ravi Keerthi//Desktop//Disserattion//newKickstarter.csv")


Rdata= data[['backers_count', 'blurb', 'converted_pledged_amount',
       'country',  'currency','created_at','category',
       'currency_trailing_code', 'current_currency', 'deadline',
       'disable_communication', 'fx_rate', 'goal', 'is_starrable', 
       'spotlight', 'staff_pick', 'state',  
       'static_usd_rate',  'pledged', 'usd_type','name']]



ttype=list(Rdata['currency'].unique())
for prop in ttype:
    Rdata.loc[Rdata['currency']==prop,'currency']=ttype.index(prop)+1
    
qtype=list(Rdata['country'].unique())
for prop in qtype:
    Rdata.loc[Rdata['country']==prop,'country']=qtype.index(prop)+1

rtype=list(Rdata['current_currency'].unique())
for prop in rtype:
    Rdata.loc[Rdata['current_currency']==prop,'current_currency']=rtype.index(prop)+1
    
stype=list(Rdata['category'].unique())
for prop in stype:
    Rdata.loc[Rdata['category']==prop,'category']=stype.index(prop)+1  
    
mtype=list(Rdata['usd_type'].unique())
for prop in mtype:
    Rdata.loc[Rdata['usd_type']==prop,'usd_type']=mtype.index(prop)+1      



Rdata['currency_trailing_code'] = Rdata['currency_trailing_code'].apply(lambda x: 0 if x=='no' else 1)
Rdata['disable_communication'] = Rdata['disable_communication'].apply(lambda x: 0 if x=='no' else 1)
Rdata['is_starrable'] = Rdata['is_starrable'].apply(lambda x: 0 if x=='no' else 1)
Rdata['spotlight'] = Rdata['spotlight'].apply(lambda x: 0 if x=='no' else 1)
Rdata['staff_pick'] = Rdata['staff_pick'].apply(lambda x: 0 if x=='no' else 1)


Rdata['deadline']=pd.to_datetime(Rdata['deadline'], unit='s')
Rdata['created_at']=pd.to_datetime(Rdata['created_at'], unit='s')


Rdata['deadline']=(Rdata.deadline-Rdata.created_at) # deadline-created 
Rdata['category'] = Rdata['category'].astype(str).str.rsplit(':').str[2]
Rdata['category'] = Rdata['category'].astype(str).str.rsplit('"').str[1]


Rdata['deadline'] = Rdata['deadline'].dt.days

Rdata['goal'] = Rdata['goal'].apply(lambda x: '{:.2f}'.format(x))

Rdata['usd_pledged'] = Rdata['pledged'].apply(lambda x: '{:.2f}'.format(x))


##

#creating features from the project name

#length of name
Rdata['name_len'] = Rdata.name.str.len()

# presence of !
Rdata['name_exclaim'] = (Rdata.name.str[-1] == '!').astype(int)

# presence of !
Rdata['name_question'] = (Rdata.name.str[-1] == '?').astype(int)

# number of words in the name
Rdata['name_words'] = Rdata.name.apply(lambda x: len(str(x).split(' ')))

# if name is uppercase
Rdata['name_is_upper'] = Rdata.name.str.isupper().astype(float)

Rdata.to_csv("TestData.csv", index = False)

Rdata['state'] = Rdata['state'].apply(lambda x: 1 if x=='successful' else 0)

del Rdata['created_at']
del Rdata['blurb']
del Rdata['name']

Rdata.to_csv("C:/Users//Ravi Keerthi//Desktop//Disserattion//Cleaned_KIVA_Data.csv", index=False)

