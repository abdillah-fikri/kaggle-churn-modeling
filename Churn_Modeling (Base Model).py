# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% execution={"iopub.execute_input": "2020-10-14T04:15:35.821848Z", "iopub.status.busy": "2020-10-14T04:15:35.820851Z", "iopub.status.idle": "2020-10-14T04:15:35.836809Z", "shell.execute_reply": "2020-10-14T04:15:35.835813Z", "shell.execute_reply.started": "2020-10-14T04:15:35.821848Z"}
# # !pip install -r 'requirements.txt'

# %% execution={"iopub.execute_input": "2020-10-14T04:15:35.839800Z", "iopub.status.busy": "2020-10-14T04:15:35.839800Z", "iopub.status.idle": "2020-10-14T04:15:41.484135Z", "shell.execute_reply": "2020-10-14T04:15:41.482140Z", "shell.execute_reply.started": "2020-10-14T04:15:35.839800Z"} executionInfo={"elapsed": 65785, "status": "ok", "timestamp": 1602616018458, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="dygeoqoCIlSB" outputId="b1754089-e015-4b25-ad49-8606b5602aa0"
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
import missingno as msno
import category_encoders as ce
import optuna
import warnings
warnings.filterwarnings('ignore')

from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.pipeline import Pipeline

# %% [markdown] id="wBNQbpc9LGht"
# # Data Exploration

# %% execution={"iopub.execute_input": "2020-10-14T04:15:41.487128Z", "iopub.status.busy": "2020-10-14T04:15:41.486130Z", "iopub.status.idle": "2020-10-14T04:15:41.576887Z", "shell.execute_reply": "2020-10-14T04:15:41.575890Z", "shell.execute_reply.started": "2020-10-14T04:15:41.487128Z"} executionInfo={"elapsed": 66328, "status": "ok", "timestamp": 1602616019020, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="WtB98MCwItEc" outputId="c6f2f4c9-752d-4e78-a3a6-591c5cee4031"
df = pd.read_csv('Churn_Modelling.csv')
df.head()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:41.587857Z", "iopub.status.busy": "2020-10-14T04:15:41.587857Z", "iopub.status.idle": "2020-10-14T04:15:41.608801Z", "shell.execute_reply": "2020-10-14T04:15:41.607804Z", "shell.execute_reply.started": "2020-10-14T04:15:41.587857Z"} executionInfo={"elapsed": 66311, "status": "ok", "timestamp": 1602616019021, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="BZYqz64JJ75s" outputId="61a47fba-ee3c-4bd4-e45b-3462fb659fef"
df.shape

# %% execution={"iopub.execute_input": "2020-10-14T04:15:41.612794Z", "iopub.status.busy": "2020-10-14T04:15:41.611793Z", "iopub.status.idle": "2020-10-14T04:15:41.656673Z", "shell.execute_reply": "2020-10-14T04:15:41.655675Z", "shell.execute_reply.started": "2020-10-14T04:15:41.612794Z"} executionInfo={"elapsed": 66295, "status": "ok", "timestamp": 1602616019022, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="t_9qL3zRJCz4" outputId="e0a37187-1412-46ce-d3d0-fc8365fea87c"
df.info()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:41.659666Z", "iopub.status.busy": "2020-10-14T04:15:41.658669Z", "iopub.status.idle": "2020-10-14T04:15:42.597156Z", "shell.execute_reply": "2020-10-14T04:15:42.596159Z", "shell.execute_reply.started": "2020-10-14T04:15:41.659666Z"} executionInfo={"elapsed": 67057, "status": "ok", "timestamp": 1602616019802, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Vgfp0PMMqJT9" outputId="7a170d58-6db1-4751-b0b2-5b1345c3b0e5"
msno.matrix(df)

# %% [markdown] id="c1JsmU87JdRo"
# Our data is clean and there is no missing value.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:42.600149Z", "iopub.status.busy": "2020-10-14T04:15:42.599151Z", "iopub.status.idle": "2020-10-14T04:15:42.612117Z", "shell.execute_reply": "2020-10-14T04:15:42.611119Z", "shell.execute_reply.started": "2020-10-14T04:15:42.600149Z"} executionInfo={"elapsed": 67052, "status": "ok", "timestamp": 1602616019803, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="EN8zFhOpJJwH"
df = df.drop(columns=['RowNumber', 'CustomerId'])

# %% [markdown] id="kSz1N8zWKNzo"
# We are removing the first 2 columns because we don't need them for prediction.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:42.618101Z", "iopub.status.busy": "2020-10-14T04:15:42.617103Z", "iopub.status.idle": "2020-10-14T04:15:42.643033Z", "shell.execute_reply": "2020-10-14T04:15:42.642038Z", "shell.execute_reply.started": "2020-10-14T04:15:42.618101Z"} executionInfo={"elapsed": 67033, "status": "ok", "timestamp": 1602616019804, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="MgRjYBSbKVit" outputId="edc9f44d-6845-4fba-8b59-db69e3441f60"
df['Surname'].nunique()

# %% [markdown] id="AZapDRiaKflR"
# For the Surname column, we'll save it first because it might work. <br>
# Assuming the same family has a similar Churn probability.

# %% [markdown] id="0WZ3UvgoLKVM"
# ## Statistic Summary

# %% execution={"iopub.execute_input": "2020-10-14T04:15:42.647023Z", "iopub.status.busy": "2020-10-14T04:15:42.646026Z", "iopub.status.idle": "2020-10-14T04:15:42.719828Z", "shell.execute_reply": "2020-10-14T04:15:42.718831Z", "shell.execute_reply.started": "2020-10-14T04:15:42.647023Z"} executionInfo={"elapsed": 67456, "status": "ok", "timestamp": 1602616020250, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Zv3J9zRrKY8a" outputId="c31b2155-afaf-45c0-b6da-fd188d82bdf5"
df.describe()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:42.721824Z", "iopub.status.busy": "2020-10-14T04:15:42.720825Z", "iopub.status.idle": "2020-10-14T04:15:42.766703Z", "shell.execute_reply": "2020-10-14T04:15:42.765706Z", "shell.execute_reply.started": "2020-10-14T04:15:42.721824Z"} executionInfo={"elapsed": 67439, "status": "ok", "timestamp": 1602616020251, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="99K-Oga4LUGp" outputId="27488ac9-b14b-4ac9-9b6c-3d1530410eea"
df.describe(include='object')

# %% [markdown] id="1vbzYY_oLclP"
# ## Visualization

# %% [markdown] id="clhnB47kb4Wb"
# ### Target Feature

# %% execution={"iopub.execute_input": "2020-10-14T04:15:42.768699Z", "iopub.status.busy": "2020-10-14T04:15:42.768699Z", "iopub.status.idle": "2020-10-14T04:15:44.132049Z", "shell.execute_reply": "2020-10-14T04:15:44.131051Z", "shell.execute_reply.started": "2020-10-14T04:15:42.768699Z"} executionInfo={"elapsed": 69325, "status": "ok", "timestamp": 1602616022154, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="RActDUyNLXtf" outputId="aff8e597-a2fd-45ec-e907-409e76a34a93"
fig = px.histogram(df, x='Exited',
                   height=400, width=500,
                   title='Target Feature Distribution')
fig.update_xaxes(type='category')
fig.show()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:44.134045Z", "iopub.status.busy": "2020-10-14T04:15:44.134045Z", "iopub.status.idle": "2020-10-14T04:15:44.148007Z", "shell.execute_reply": "2020-10-14T04:15:44.147009Z", "shell.execute_reply.started": "2020-10-14T04:15:44.134045Z"} executionInfo={"elapsed": 69313, "status": "ok", "timestamp": 1602616022155, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="dZyqiHhyMPnV" outputId="b4096f0f-f387-4d1c-d686-51789df54fea"
df['Exited'].value_counts(normalize=True)*100

# %% [markdown] id="fEHe65GuMwkc"
# The target distribution on the dataset is unbalanced. But this is normal because this is a customer churn dataset.
#
# In terms of the dataset, it can be said to be good because there are enough positive classes so that the model will be easier to detect positive class. <br>
# However, from a business perspective, it is not good because the Churn rate is quite high.
#
# Because of this we will use the AUC ROC metric at the modeling stage with focus in higher recall on positive class.

# %% [markdown] id="dWftoXg3cePU"
# ### Numerical Features

# %% execution={"iopub.execute_input": "2020-10-14T04:15:44.150001Z", "iopub.status.busy": "2020-10-14T04:15:44.150001Z", "iopub.status.idle": "2020-10-14T04:15:44.490092Z", "shell.execute_reply": "2020-10-14T04:15:44.489094Z", "shell.execute_reply.started": "2020-10-14T04:15:44.150001Z"} executionInfo={"elapsed": 69301, "status": "ok", "timestamp": 1602616022157, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="NDx6xO41O9h_" outputId="b05e7150-ebcc-458f-f092-a2e451e7befe"
fig = make_subplots(rows=2, cols=3)

fig.append_trace(go.Histogram(
    x=df['CreditScore'], name='Credit Score', nbinsx=50
), row=1, col=1)
fig.update_xaxes(title_text='Credit Score', row=1, col=1)

fig.append_trace(go.Histogram(
    x=df['Age'], name='Age', nbinsx=30
), row=1, col=2)
fig.update_xaxes(title_text='Age', row=1, col=2)

fig.append_trace(go.Histogram(
    x=df['Balance'], name='Balance', nbinsx=20
), row=1, col=3)
fig.update_xaxes(title_text='Balance', row=1, col=3)

for col, feature in enumerate(['Tenure', 'EstimatedSalary']):
    fig.append_trace(go.Histogram(
        x=df[feature], name=feature,
        nbinsx=20
    ), row=2, col=col+1)
    fig.update_xaxes(title_text=feature, row=2, col=col+1)

fig.update_layout(
    height=600, width=900, 
    title_text='Features Distribution with Histogram'
)
fig.show()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:44.493084Z", "iopub.status.busy": "2020-10-14T04:15:44.492086Z", "iopub.status.idle": "2020-10-14T04:15:44.505052Z", "shell.execute_reply": "2020-10-14T04:15:44.504054Z", "shell.execute_reply.started": "2020-10-14T04:15:44.493084Z"} executionInfo={"elapsed": 69288, "status": "ok", "timestamp": 1602616022158, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="LpKeypn-zEIO" outputId="9f8bc4db-0fa2-44bf-b1ab-11be686a497a"
print('Customer dengan Balance 0:',len(df[df['Balance']==0]))

# %% [markdown] id="g9YyUWBOWYaa"
# * Credit Score has a fairly normal distribution but there is an anomaly in customers with a credit score of 840 to 859, which is quite high compared to the previous range.
# * Age has the right-skewed distribution with the largest number of customers in the 35 to 29 year age segment (2308 people)
# * Balance has a normal distribution but there is an anomaly in the Balance with a value of 0 with a total of 3617 people.
# * Tenure and EstimatedSalary have a uniform distribution

# %% execution={"iopub.execute_input": "2020-10-14T04:15:44.507046Z", "iopub.status.busy": "2020-10-14T04:15:44.507046Z", "iopub.status.idle": "2020-10-14T04:15:44.849131Z", "shell.execute_reply": "2020-10-14T04:15:44.847136Z", "shell.execute_reply.started": "2020-10-14T04:15:44.507046Z"} executionInfo={"elapsed": 70788, "status": "ok", "timestamp": 1602616023670, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="auJIKDpDUlya" outputId="9f518a22-10b1-4c36-da7a-94cc7c01c849"
fig = px.histogram(
    df, x='CreditScore', color='Exited',
    marginal='box', nbins=50,
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barmode='overlay'
)

fig.update_layout(
    height=500, width=700, 
    title_text='Credit Score Feature in Detail'
)
fig.show()

# %% [markdown] id="MqVtwS9Fc6K1"
# Customers with class 0 and 1 on the Credit Score feature both have a normal distribution with the anomaly on the right. <br>
# For class 1 there are several outliers on the left.
#
# There is a median difference between the two classes but not significant. The median for class 1 is slightly lower. In other words, customers with a low credit score have a higher (but not significant) churn rate.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:44.851125Z", "iopub.status.busy": "2020-10-14T04:15:44.850128Z", "iopub.status.idle": "2020-10-14T04:15:45.191215Z", "shell.execute_reply": "2020-10-14T04:15:45.190219Z", "shell.execute_reply.started": "2020-10-14T04:15:44.851125Z"} executionInfo={"elapsed": 70779, "status": "ok", "timestamp": 1602616023674, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="8x3zshEXZo4U" outputId="bc957f4a-f616-43fd-c024-0bc80cccda15"
fig = px.histogram(
    df, x='Age', color='Exited',
    marginal='box', nbins=30,
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barmode='overlay'
)

fig.update_layout(height=500, width=700, 
                  title_text='Age Feature in Detail')
fig.show()

# %% [markdown] id="SPniAUdRejis"
# Customers with class 0 in the Age feature have a right-skew distribution while those for class 1 have a normal distribution. There are several outliers in class 1 and quite a number of outliers in class 0.
#
# When viewed from the median, customers with old age have a higher tendency to churn.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:45.193210Z", "iopub.status.busy": "2020-10-14T04:15:45.193210Z", "iopub.status.idle": "2020-10-14T04:15:45.506373Z", "shell.execute_reply": "2020-10-14T04:15:45.505377Z", "shell.execute_reply.started": "2020-10-14T04:15:45.193210Z"} executionInfo={"elapsed": 70766, "status": "ok", "timestamp": 1602616023677, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="UMt8nwgkbEhn" outputId="b3809264-9b25-40d1-fdc4-34515a44675c"
fig = px.histogram(
    df, x='Balance', color='Exited',
    marginal='box', nbins=20,
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barmode='overlay'
)

fig.update_layout(height=500, width=700, 
                  title_text='Balance Feature in Detail')
fig.show()

# %% [markdown] id="kjdzRVnxfu4B"
# Both classes have the same distribution with the anomaly at Balance 0. There are no outliers.
#
# Regardless of the anomaly, the two distributions appear to have the same median. However, due to the anomaly at value 0, the median for class 0 is lower because at value 0 there are more class 0 compared to class 1.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:45.508366Z", "iopub.status.busy": "2020-10-14T04:15:45.507372Z", "iopub.status.idle": "2020-10-14T04:15:45.774656Z", "shell.execute_reply": "2020-10-14T04:15:45.773657Z", "shell.execute_reply.started": "2020-10-14T04:15:45.508366Z"} executionInfo={"elapsed": 71547, "status": "ok", "timestamp": 1602616024474, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="Zoywqrlhdr_f" outputId="ab84552a-24d6-43a5-f77d-9afa3d15e283"
fig = px.histogram(
    df, x='Tenure', color='Exited', marginal='box',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barmode='overlay'
)
fig.update_layout(height=500, width=700, 
                  title_text='Tenure Feature in Detail')
fig.show()

# %% [markdown] id="3NzP4j09i69K"
# Both classes in the Tenure feature have the same distribution and both have no outliers.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:45.777656Z", "iopub.status.busy": "2020-10-14T04:15:45.776649Z", "iopub.status.idle": "2020-10-14T04:15:46.009029Z", "shell.execute_reply": "2020-10-14T04:15:46.008031Z", "shell.execute_reply.started": "2020-10-14T04:15:45.777656Z"} executionInfo={"elapsed": 71533, "status": "ok", "timestamp": 1602616024476, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="tMg-WTEPryni" outputId="73dc58e3-bb2b-4996-f137-e8b91cf3ce43"
fig = px.histogram(
    df, x='Tenure', color='Exited',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    category_orders={'Tenure': [0,1,2,3,4,5,6,7,8,9,10]},
    barnorm='percent'
)

fig.update_layout(
    height=500, width=700, 
    title_text='Tenure Feature in Detail',
    yaxis_title='Percentage of Churn',
    yaxis={'ticksuffix':'%'}
)
fig.update_xaxes(type='category')
fig.show()

# %% [markdown] id="m8VJTs6MjJq1"
# There is no significant difference between the churn and the average level is 20%. <br>
# Customers with a 7 year Tenure had the lowest churn rate (17.2%).

# %% execution={"iopub.execute_input": "2020-10-14T04:15:46.012020Z", "iopub.status.busy": "2020-10-14T04:15:46.012020Z", "iopub.status.idle": "2020-10-14T04:15:46.410952Z", "shell.execute_reply": "2020-10-14T04:15:46.408957Z", "shell.execute_reply.started": "2020-10-14T04:15:46.012020Z"} executionInfo={"elapsed": 72779, "status": "ok", "timestamp": 1602616025738, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="O6nDWB48egmT" outputId="7b2f79b2-f751-48ef-9700-ddd65ebc1464"
fig = px.histogram(
    df, x='EstimatedSalary', color='Exited', marginal='box',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barmode='overlay', nbins=20
)

fig.update_layout(height=500, width=700, 
                  title_text='EstimatedSalary Feature in Detail')
fig.show()

# %% [markdown] id="PX8yNvyAjvrU"
# The two classes in the EstimatedSalary feature have a uniform distribution, with a slightly higher median value in class 1.

# %% [markdown] id="xxOiuWEKgvql"
# ### Categorical Features

# %% execution={"iopub.execute_input": "2020-10-14T04:15:46.415939Z", "iopub.status.busy": "2020-10-14T04:15:46.414942Z", "iopub.status.idle": "2020-10-14T04:15:46.828834Z", "shell.execute_reply": "2020-10-14T04:15:46.826840Z", "shell.execute_reply.started": "2020-10-14T04:15:46.414942Z"} executionInfo={"elapsed": 73501, "status": "ok", "timestamp": 1602616026476, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="FJsOhqvtdVK6" outputId="29051f30-9ac8-4ca5-840b-a2860e3a6fe7"
fig = make_subplots(rows=2, cols=3)

# For loop for the first row
for col, feature in enumerate(['NumOfProducts', 'HasCrCard', 'IsActiveMember']):
    fig.append_trace(go.Histogram(
        x=df[feature], name=feature,
    ), row=1, col=col+1)
    fig.update_xaxes(title_text=feature, row=1, col=col+1)

# For loop for the second row
for col, feature in enumerate(['Geography', 'Gender']):
    fig.append_trace(go.Histogram(
        x=df[feature], name=feature,
    ), row=2, col=col+1)
    fig.update_xaxes(title_text=feature, row=2, col=col+1)

fig.update_xaxes(type='category', 
                 categoryorder='category ascending')
fig.update_layout(height=600, width=900, 
                  title_text='Categorical Features Distribution')
fig.show()

# %% [markdown] id="sNPU0u9vg6a7"
# * Our dataset is dominated by customers who have 1 and 2 products. The intensity of customers who have 3 and 4 products is only a few.
# * Customers who have more credit cards (more than 2 times than those who do not)
# * There are quite a lot of inactive members, almost equal to active members.
# * There are far more customers from France than customers from Germany and Spain.
# * There are more male customers

# %% execution={"iopub.execute_input": "2020-10-14T04:15:46.831825Z", "iopub.status.busy": "2020-10-14T04:15:46.830828Z", "iopub.status.idle": "2020-10-14T04:15:47.064215Z", "shell.execute_reply": "2020-10-14T04:15:47.062210Z", "shell.execute_reply.started": "2020-10-14T04:15:46.831825Z"} executionInfo={"elapsed": 73488, "status": "ok", "timestamp": 1602616026477, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="4cniLiDFeV-I" outputId="edce9fcd-3158-48d1-e811-6249cb3d98fe"
fig = px.histogram(
    df, x='NumOfProducts', color='Exited',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barnorm='percent'
)
fig.update_layout(
    height=500, width=700, 
    title_text='NumOfProducts Feature in Detail',
    yaxis_title='Percentage of Churn',
    yaxis={'ticksuffix':'%'}
)
fig.update_xaxes(
    type='category',
    categoryorder='category ascending'
)
fig.show()

# %% [markdown] id="z0IUYOxikBiI"
# From categories 1 and 2 with the highest number of customers, we can see that customers who only have 1 product have a higher churn rate (27.7%).
#
# Meanwhile, customers with 3 products had a churn rate of 82.7% and the most were customers with 4 products that had a churn rate of 100%.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:47.067198Z", "iopub.status.busy": "2020-10-14T04:15:47.066200Z", "iopub.status.idle": "2020-10-14T04:15:47.295585Z", "shell.execute_reply": "2020-10-14T04:15:47.293591Z", "shell.execute_reply.started": "2020-10-14T04:15:47.067198Z"} executionInfo={"elapsed": 73953, "status": "ok", "timestamp": 1602616026956, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="w88W_ZYPz3mZ" outputId="8a123eb7-e046-41e7-d660-a0b42121dcc3"
fig = px.histogram(
    df, x='HasCrCard', color='Exited',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barnorm='percent'
)
fig.update_layout(height=500, width=700, 
                  title_text='HasCrCard Feature in Detail',
                  yaxis_title='Percentage of Churn',
                  yaxis={'ticksuffix':'%'})
fig.update_xaxes(
    type='category',
    categoryorder='category ascending'
)
fig.show()

# %% [markdown] id="KrPaU3Zok0J2"
# There is no significant difference in this feature. Both categories have the same churn rate.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:47.298577Z", "iopub.status.busy": "2020-10-14T04:15:47.297580Z", "iopub.status.idle": "2020-10-14T04:15:47.502036Z", "shell.execute_reply": "2020-10-14T04:15:47.501035Z", "shell.execute_reply.started": "2020-10-14T04:15:47.298577Z"} executionInfo={"elapsed": 74440, "status": "ok", "timestamp": 1602616027456, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="41VWmy1pz2_M" outputId="13638857-c830-40b8-823f-b14b5e7e7137"
fig = px.histogram(
    df, x='IsActiveMember', color='Exited',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'},
    barnorm='percent'
)
fig.update_layout(height=500, width=700, 
                  title_text='IsActiveMember Feature in Detail',
                  yaxis_title='Percentage of Churn',
                  yaxis={'ticksuffix':'%'})
fig.update_xaxes(
    type='category',
    categoryorder='category ascending'
)
fig.show()

# %% [markdown] id="r_d74cS5lJDE"
# Inactive customers have a higher churn rate with a portion of 26.8% compared to active customers (14.2%).

# %% execution={"iopub.execute_input": "2020-10-14T04:15:47.504027Z", "iopub.status.busy": "2020-10-14T04:15:47.504027Z", "iopub.status.idle": "2020-10-14T04:15:47.765328Z", "shell.execute_reply": "2020-10-14T04:15:47.764330Z", "shell.execute_reply.started": "2020-10-14T04:15:47.504027Z"} executionInfo={"elapsed": 75123, "status": "ok", "timestamp": 1602616028188, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="YKPw8mcn39oj" outputId="c9a38211-a2af-4dac-d661-9d94fafbe0ee"
fig = px.histogram(
    df, x='Geography', color='Exited',
    barnorm='percent',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'}
)
fig.update_yaxes(title_text='Percentage of Churn')
fig.update_layout(height=500, width=700, 
                  title_text='Exited Percentage by Geography',
                  yaxis={'ticksuffix':'%'})
fig.show()

# %% [markdown] id="Rdk3kyIJl2ZQ"
# Customers from Germany have a churn rate of 32.4%, while customers from France are 16.2% and Spain 16.7%.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:47.770314Z", "iopub.status.busy": "2020-10-14T04:15:47.770314Z", "iopub.status.idle": "2020-10-14T04:15:47.999702Z", "shell.execute_reply": "2020-10-14T04:15:47.998704Z", "shell.execute_reply.started": "2020-10-14T04:15:47.770314Z"} executionInfo={"elapsed": 76422, "status": "ok", "timestamp": 1602616029508, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="44vBP2Y_7yQ9" outputId="d5b4fabb-426c-4ed3-a6fa-9fab99ebb083"
fig = px.histogram(
    df, x='Gender', color='Exited',
    barnorm='percent',
    color_discrete_map={0: '#636EFA', 1: '#EF553B'}
)
fig.update_yaxes(title_text='Percent')

fig.update_layout(height=500, width=700, 
                  title_text='Exited Percentage by Gender',
                  yaxis={'ticksuffix':'%'})
fig.show()

# %% [markdown] id="rI5u37Z0mPYj"
# Female customers have a higher churn rate (25%) than male customers (16.5%).

# %% [markdown] id="7S_v5Qi1nD32"
# ### Heatmap Correlation

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.003690Z", "iopub.status.busy": "2020-10-14T04:15:48.002693Z", "iopub.status.idle": "2020-10-14T04:15:48.358742Z", "shell.execute_reply": "2020-10-14T04:15:48.357743Z", "shell.execute_reply.started": "2020-10-14T04:15:48.003690Z"} executionInfo={"elapsed": 76410, "status": "ok", "timestamp": 1602616029509, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="iyvBwv5Goy2Y" outputId="a7ad3ffe-d22a-4b8d-bf87-8fd60b10519e"
encoder = ce.TargetEncoder()
df_temp = encoder.fit_transform(df.drop(columns='Exited'), df['Exited'])
df_corr = df_temp.join(df['Exited']).corr()

fig = ff.create_annotated_heatmap(
    z=df_corr.values,
    x=list(df_corr.columns),
    y=list(df_corr.index),
    annotation_text=df_corr.round(2).values,
    showscale=True, colorscale='Viridis'
)
fig.update_layout(height=600, width=800, 
                  title_text='Feature Correlation')
fig.update_xaxes(side='bottom')
fig.show()

# %% [markdown] id="rOzxBfS7nLJp"
# The highest correlation to Target is the Surname feature (0.36), and the second is the Age feature with a value of 0.29.
#
# The insights that can be obtained from this data are:
# * Customers with higher Age have a higher churn rate
# * There are family names (Surname) whose churn level is higher

# %% [markdown] id="JTz-7rwEmrY2"
# ## Data Preprocessing

# %% [markdown] id="BESo5h6Omwr1"
# ### Feature Enginering

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.361732Z", "iopub.status.busy": "2020-10-14T04:15:48.360735Z", "iopub.status.idle": "2020-10-14T04:15:48.373701Z", "shell.execute_reply": "2020-10-14T04:15:48.372702Z", "shell.execute_reply.started": "2020-10-14T04:15:48.361732Z"} executionInfo={"elapsed": 76855, "status": "ok", "timestamp": 1602616029958, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="KWoYPc3rkPUI"
df['BalanceToSalaryRatio'] = df['Balance'] / df['EstimatedSalary']

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.375695Z", "iopub.status.busy": "2020-10-14T04:15:48.375695Z", "iopub.status.idle": "2020-10-14T04:15:48.434537Z", "shell.execute_reply": "2020-10-14T04:15:48.433540Z", "shell.execute_reply.started": "2020-10-14T04:15:48.375695Z"} executionInfo={"elapsed": 76844, "status": "ok", "timestamp": 1602616029959, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="qwZtLl8d2Bek" outputId="fd47c785-2523-4a69-a4f9-d12222a7ffd7"
from itertools import combinations
cat_cols = df.select_dtypes('object').columns

for col in combinations(cat_cols, 2):
    df[col[0]+'_'+col[1]] = df[col[0]] + "_" + df[col[1]]
    
df.head()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.437529Z", "iopub.status.busy": "2020-10-14T04:15:48.436532Z", "iopub.status.idle": "2020-10-14T04:15:48.510334Z", "shell.execute_reply": "2020-10-14T04:15:48.509337Z", "shell.execute_reply.started": "2020-10-14T04:15:48.437529Z"} executionInfo={"elapsed": 76836, "status": "ok", "timestamp": 1602616029960, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="mXYj1beUrRWR" outputId="dac98bad-8082-49ee-bd6d-dbecf159059c"
df.describe(include='object')

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.513327Z", "iopub.status.busy": "2020-10-14T04:15:48.512329Z", "iopub.status.idle": "2020-10-14T04:15:48.992045Z", "shell.execute_reply": "2020-10-14T04:15:48.990051Z", "shell.execute_reply.started": "2020-10-14T04:15:48.512329Z"} executionInfo={"elapsed": 78703, "status": "ok", "timestamp": 1602616031837, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="gtr6UmRFrVer" outputId="c2e9519e-f586-4100-8f5b-4b5fc3700e03"
encoder = ce.TargetEncoder()
df_temp = encoder.fit_transform(df.drop(columns='Exited'), df['Exited'])
df_corr = df_temp.join(df['Exited']).corr()

fig = ff.create_annotated_heatmap(
    z=df_corr.values,
    x=list(df_corr.columns),
    y=list(df_corr.index),
    annotation_text=df_corr.round(2).values,
    showscale=True, colorscale='Viridis'
)
fig.update_layout(height=600, width=800, 
                  title_text='Feature Correlation')
fig.update_xaxes(side='bottom')
fig.show()

# %% [markdown] id="nLB6Pmtjryez"
# After performing feature engineering we get a new feature with a higher correlation with the target.

# %% execution={"iopub.execute_input": "2020-10-14T04:15:48.994041Z", "iopub.status.busy": "2020-10-14T04:15:48.994041Z", "iopub.status.idle": "2020-10-14T04:15:49.021966Z", "shell.execute_reply": "2020-10-14T04:15:49.020968Z", "shell.execute_reply.started": "2020-10-14T04:15:48.994041Z"} executionInfo={"elapsed": 78695, "status": "ok", "timestamp": 1602616031839, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="I7T4ZI1UuK-J" outputId="de8796e9-f335-4130-f934-981d5c9ffd50"
df.head()

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.024958Z", "iopub.status.busy": "2020-10-14T04:15:49.023961Z", "iopub.status.idle": "2020-10-14T04:15:49.083800Z", "shell.execute_reply": "2020-10-14T04:15:49.082803Z", "shell.execute_reply.started": "2020-10-14T04:15:49.024958Z"} executionInfo={"elapsed": 78684, "status": "ok", "timestamp": 1602616031840, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="nM6fSudT3b5k" outputId="d8808e4b-527b-4460-adc2-13a67768a0ac"
df.describe(include='object')

# %% [markdown] id="CB-jY2Y_5A6h"
# ### Train Test Split

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.086792Z", "iopub.status.busy": "2020-10-14T04:15:49.085794Z", "iopub.status.idle": "2020-10-14T04:15:49.115715Z", "shell.execute_reply": "2020-10-14T04:15:49.114717Z", "shell.execute_reply.started": "2020-10-14T04:15:49.086792Z"} executionInfo={"elapsed": 78681, "status": "ok", "timestamp": 1602616031841, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="sPFGLWsq5Nq6"
X_train, X_test, y_train, y_test = train_test_split(
    df.drop(columns='Exited'), df['Exited'],
    test_size=0.2, random_state=0,
)

# %% [markdown] id="N6HL0ZWesigm"
# ### Building Pipeline

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.118706Z", "iopub.status.busy": "2020-10-14T04:15:49.117709Z", "iopub.status.idle": "2020-10-14T04:15:49.131671Z", "shell.execute_reply": "2020-10-14T04:15:49.130675Z", "shell.execute_reply.started": "2020-10-14T04:15:49.118706Z"} executionInfo={"elapsed": 78680, "status": "ok", "timestamp": 1602616031843, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="AXc-WcWfRree"
# Ratio using for scale_pos_weight to get better recall on imbalance class
ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.133667Z", "iopub.status.busy": "2020-10-14T04:15:49.132669Z", "iopub.status.idle": "2020-10-14T04:15:49.147630Z", "shell.execute_reply": "2020-10-14T04:15:49.146632Z", "shell.execute_reply.started": "2020-10-14T04:15:49.133667Z"} executionInfo={"elapsed": 78678, "status": "ok", "timestamp": 1602616031844, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="gw-IoBmernS_"
xgb_pipeline = Pipeline([
    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),
    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),
    ('xgb', XGBClassifier(scale_pos_weight=ratio))
])

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.150621Z", "iopub.status.busy": "2020-10-14T04:15:49.149625Z", "iopub.status.idle": "2020-10-14T04:15:49.163587Z", "shell.execute_reply": "2020-10-14T04:15:49.162590Z", "shell.execute_reply.started": "2020-10-14T04:15:49.150621Z"} executionInfo={"elapsed": 78679, "status": "ok", "timestamp": 1602616031848, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ue3NFSzn6oYP"
lgb_pipeline = Pipeline([
    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),
    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),
    ('lgb', LGBMClassifier(scale_pos_weight=ratio))
])

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.165582Z", "iopub.status.busy": "2020-10-14T04:15:49.164585Z", "iopub.status.idle": "2020-10-14T04:15:49.178546Z", "shell.execute_reply": "2020-10-14T04:15:49.177549Z", "shell.execute_reply.started": "2020-10-14T04:15:49.165582Z"} executionInfo={"elapsed": 78677, "status": "ok", "timestamp": 1602616031850, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="b4Zdp_7h6o-u"
cat_pipeline = Pipeline([
    ('one_hot', ce.OneHotEncoder(cols=['Geography', 'Gender', 'Geography_Gender'])),
    ('catboost', ce.CatBoostEncoder(cols=['Surname', 'Surname_Geography', 'Surname_Gender'])),
    ('cat', CatBoostClassifier(scale_pos_weight=ratio, verbose=0))
])

# %% [markdown] id="Bjhz3VbE4_GV"
# ## Modeling

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.180542Z", "iopub.status.busy": "2020-10-14T04:15:49.180542Z", "iopub.status.idle": "2020-10-14T04:15:49.225421Z", "shell.execute_reply": "2020-10-14T04:15:49.224425Z", "shell.execute_reply.started": "2020-10-14T04:15:49.180542Z"} executionInfo={"elapsed": 78672, "status": "ok", "timestamp": 1602616031852, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="ziQmBB_I_t_G"
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn import metrics

def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nPrecision={:0.3f} | Recall={:0.3f}\nAccuracy={:0.3f} | F1 Score={:0.3f}".format(
                precision, recall, accuracy, f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    plt.figure(figsize=figsize)
    plt.subplot(1,2,1)
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)
    
    if title:
        plt.title(title)


def model_eval(model, X_train, y_train, 
               scoring_='roc_auc', cv_=5):
  
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    train_predprob = model.predict_proba(X_train)[:,1]
           
    cv_score = cross_val_score(model, X_train, y_train, cv=cv_, scoring=scoring_)
    print('Model Report on Train and CV Set:')
    print('--------')
    print('Train Accuracy: {:0.6f}'.format(metrics.accuracy_score(y_train, train_pred)))
    print('Train AUC Score: {:0.6f}'.format(metrics.roc_auc_score(y_train, train_predprob)))
    print('CV AUC Score: Mean - {:0.6f} | Std - {:0.6f} | Min - {:0.6f} | Max - {:0.6f} \n'.format(
        np.mean(cv_score),np.std(cv_score),np.min(cv_score),np.max(cv_score)))



def test_eval(model, X_train, X_test, y_train, y_test):
    
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    predprob = model.predict_proba(X_test)[:,1]
    
    print('Model Report on Test Set:')
    print('--------')
    print('Classification Report \n', metrics.classification_report(y_test, pred))

    conf = metrics.confusion_matrix(y_test, pred)
    group_names = ['True Negative', 'False Positive', 'False Negtive', 'True Positive']
    make_confusion_matrix(conf, percent=False, group_names=group_names,
                          figsize=(14,5), title='Confusion Matrix')

    plt.subplot(1,2,2)
    fpr, tpr, _ = metrics.roc_curve(y_test, predprob)
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve\nAUC Score: {:0.3f}'.format(metrics.roc_auc_score(y_test, predprob)))
    plt.legend()


# %% [markdown]
# Confusion Matrix function credit to [DTrimarchi10](https://github.com/DTrimarchi10/confusion_matrix)

# %% [markdown]
# #### XGBoost

# %% execution={"iopub.execute_input": "2020-10-14T04:15:49.228414Z", "iopub.status.busy": "2020-10-14T04:15:49.227416Z", "iopub.status.idle": "2020-10-14T04:15:52.175528Z", "shell.execute_reply": "2020-10-14T04:15:52.174530Z", "shell.execute_reply.started": "2020-10-14T04:15:49.228414Z"} executionInfo={"elapsed": 81283, "status": "ok", "timestamp": 1602616034475, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="vMmp16PW4Zok" outputId="41fe188d-f73a-483c-82bd-85ba0de53550"
test_eval(xgb_pipeline, X_train, X_test, y_train, y_test)

# %% [markdown]
# #### LightGBM

# %% execution={"iopub.execute_input": "2020-10-14T04:15:52.178519Z", "iopub.status.busy": "2020-10-14T04:15:52.177523Z", "iopub.status.idle": "2020-10-14T04:15:54.150243Z", "shell.execute_reply": "2020-10-14T04:15:54.149247Z", "shell.execute_reply.started": "2020-10-14T04:15:52.177523Z"} executionInfo={"elapsed": 82081, "status": "ok", "timestamp": 1602616035291, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="q2BdNY0o551Q" outputId="70afede2-8361-4921-f909-b4a7262c708d"
test_eval(lgb_pipeline, X_train, X_test, y_train, y_test)

# %% [markdown]
# #### CatBoost

# %% execution={"iopub.execute_input": "2020-10-14T04:15:54.153235Z", "iopub.status.busy": "2020-10-14T04:15:54.152239Z", "iopub.status.idle": "2020-10-14T04:16:05.475077Z", "shell.execute_reply": "2020-10-14T04:16:05.474114Z", "shell.execute_reply.started": "2020-10-14T04:15:54.152239Z"} executionInfo={"elapsed": 94067, "status": "ok", "timestamp": 1602616047299, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="rOkRSKmE6l_t" outputId="f9904c3c-65b5-4cf9-9041-02c0ca7a7905"
test_eval(cat_pipeline, X_train, X_test, y_train, y_test)

# %% [markdown]
# #### CatBoost (built in categorical encoder)

# %% execution={"iopub.execute_input": "2020-10-14T04:16:05.477070Z", "iopub.status.busy": "2020-10-14T04:16:05.476073Z"} executionInfo={"elapsed": 112896, "status": "ok", "timestamp": 1602616066146, "user": {"displayName": "Abdillah Fikri", "photoUrl": "", "userId": "04470220666512949031"}, "user_tz": -420} id="cSaC6Hvz7xgH" outputId="0ac5c335-e543-4209-c673-71fa1b9e9d0b"
cat_features = df.select_dtypes('object').columns

cat = CatBoostClassifier(scale_pos_weight=ratio,
                         verbose=0, cat_features=cat_features)
test_eval(cat, X_train, X_test, y_train, y_test)

# %% [markdown]
# The best model is CatBoost (built in categorical encoder) with ROC AUC Score 0.874, and Recall rate 0.76 on positive class.
#
# I will update this notebook with a model after hyperparamete tuning.
