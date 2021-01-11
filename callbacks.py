

import plotly.graph_objs as go
from plotly import tools
import numpy as np
import pandas as pd
from datetime import datetime as dt
from datetime import date, timedelta
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
from dash.dependencies import Input ,Output
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import classification_report,confusion_matrix,precision_score,recall_score,accuracy_score ,f1_score,r2_score,roc_curve,roc_auc_score,balanced_accuracy_score
import pickle
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from app import app
from dash.dependencies import Input, Output
import io
pd.options.mode.chained_assignment = None
import flask


#from components import formatter_currency, formatter_currency_with_cents, formatter_percent, formatter_percent_2_digits, formatter_number
#from components import update_first_datatable, update_first_download, update_second_datatable, update_graph



data = pd.read_csv('data/processedData.csv')


def convertYN(x):
    if x=='yes':
        return 1
    return 0

y=data['y']

X=data.drop(columns=['Unnamed: 0','y'])
y=y.apply(convertYN)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=16)

sm = SMOTE(random_state=16)
X_SMOTE, y_SMOTE = sm.fit_resample(X_train, y_train)

def trainRF(rf_criterion,rf_max_depth,rf_max_features,rf_min_samples_leaf,rf_n_estimators):
    classifier= RandomForestClassifier(criterion=rf_criterion, n_estimators=int(rf_n_estimators), max_depth=int(rf_max_depth), max_features=int(rf_max_features), min_samples_leaf=int(rf_min_samples_leaf),
                           random_state=16)
    model = classifier.fit(X_SMOTE, y_SMOTE)
    return model

def trainGBM(gbm_learning_rate,gbm_max_depth,gbm_max_features,gbm_min_samples_leaf,gbm_n_estimators):
    classifier= GradientBoostingClassifier(learning_rate=float(gbm_learning_rate), n_estimators=int(gbm_n_estimators), max_depth=int(gbm_max_depth), max_features=int(gbm_max_features), min_samples_leaf=int(gbm_min_samples_leaf),
                           random_state=16)
    model = classifier.fit(X_SMOTE, y_SMOTE)
    return model

def trainKNN(KNN_k_neighbors):
    classifier= KNeighborsClassifier(n_neighbors=int(KNN_k_neighbors))
    model = classifier.fit(X_SMOTE, y_SMOTE)
    return model

def trainLGR(lgr_solver,lgr_C):
    classifier= LogisticRegression(C= float(lgr_C), random_state= 12, solver=lgr_solver)
    model = classifier.fit(X_SMOTE, y_SMOTE)
    return model

def getFeatureImportance(model):

    feature_imp = pd.Series(model.feature_importances_,index=X_SMOTE.columns).sort_values(ascending=False)
    return feature_imp


def cR_to_df(model, y_test, y_pred):
    cr = classification_report(y_test, y_pred, output_dict=True)
    row = {}
    row['Precision'] = [round(float(cr['1']['precision'])*100,2)]
    row['Recall'] = [round(float(cr['1']['recall'])*100,2)]
    row['Accuracy'] = [round(metrics.accuracy_score(y_test, y_pred) * 100,2)]
    row['Balanced Accuracy'] = [round(metrics.balanced_accuracy_score(y_test, y_pred) * 100,2)]
    row['Parameters'] = [str(model.get_params(deep=False))]

    df = pd.DataFrame.from_dict(row)
    # df=pd.DataFrame(row,index=['precision','recall','Accuracy','Balanced Accuracy'])
    return df

def cm2df(y_test,y_pred):
    cm=confusion_matrix(y_test,y_pred)
    labels=['0','1']
    df = pd.DataFrame()
    # rows
    for i, row_label in enumerate(labels):
        rowdata={}
        # columns
        for j, col_label in enumerate(labels):
            rowdata[col_label]=cm[i,j]
        df = df.append(pd.DataFrame.from_dict({row_label:rowdata}, orient='index'))
    return df[labels]


@app.callback(Output('modelPerformanceRFGBM','children'),
              [Input('rf_criterion', 'value'),
               Input('rf_max_depth', 'value'),
                Input('rf_max_features', 'value'),
                Input('rf_min_samples_leaf', 'value'),
                Input('rf_n_estimators', 'value')
               ]
              )
def update_RFdatatable(rf_criterion,rf_max_depth, rf_max_features, rf_min_samples_leaf,rf_n_estimators):

    if (rf_criterion=='entropy'and rf_max_depth==8 and rf_max_features==6 and rf_min_samples_leaf==2 and rf_n_estimators==100):
        filename="models/" + "RF_model_Best" + ".pkl"
        with open(filename, 'rb') as file1:
            pk_model = pickle.load(file1)

    else:
        pk_model = trainRF(rf_criterion, rf_max_depth, rf_max_features, rf_min_samples_leaf, rf_n_estimators)



    y_pred = pk_model.predict(X_test)
    performaceDF = cR_to_df(pk_model, y_test, y_pred)
    confusionDF = cm2df(y_test, y_pred)
    FeatureImportance= getFeatureImportance(pk_model)

    tab1 = html.Div(children=[
        html.Label(children='Performance Matrix', style={'width': '50%', 'display': 'inline-block',
                                                         'margin': 0, 'padding': '8px'}),
        dash_table.DataTable(
            id='table_no1',
            columns=[{"name": i, "id": i} for i in performaceDF.columns],
            data=performaceDF.to_dict("rows"),
            style_table={'width': '80%',
                         },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Parameters'},
                 'width': '50%'}
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={'width': '180px',
                        'height': '60px',
                        'textAlign': 'left',
                        'minWidth': '0px',
                        'maxWidth': '180px'
                        })])

    tab2 = html.Div(
        children=[html.Label(children='Confusion Matrix', style={'width': '50%', 'display': 'inline-block',
                                                                 'margin': 0, 'padding': '8px'}),
                  dash_table.DataTable(
                      id='confusionMatrix',
                      columns=[{"name": i, "id": i} for i in confusionDF.columns],
                      data=confusionDF.to_dict("rows"),
                      style_table={'width': '60%',
                                   },
                      style_data_conditional=[
                          {
                              'if': {'row_index': 'odd'},
                              'backgroundColor': 'rgb(248, 248, 248)'
                          }
                      ],
                      style_header={
                          'backgroundColor': 'rgb(230, 230, 230)',
                          'fontWeight': 'bold'
                      },
                      style_cell={'width': '180px',
                                  'height': '42px',
                                  'textAlign': 'left',
                                  'minWidth': '0px',
                                  'maxWidth': '180px'
                                  })])

    FI=html.Div(
        children=[dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': FeatureImportance.index[:5], 'y': FeatureImportance[:5], 'type': 'bar', 'name': 'SF'}
            ],
            'layout': {
                'title': 'Feature Importance of the Model'
            }
        }
    )])
    final =html.Div([
        html.Br([]),
        tab1, tab2], style=dict(display='flex')),FI
    #testdiv=([html.H4('Random Forest', style=dict(color='white', background='red'))])
    return final


@app.callback(Output('modelPerformanceKNN','children'),
              [Input('KNN_k_neighbors', 'value')
               ]
              )
def update_KNNdatatable(KNN_k_neighbors):

    #filename = "models/" + "KNN_model_Best" + ".pkl"
    if (KNN_k_neighbors==5):
        filename="models/" + "KNN_model_Best" + ".pkl"
        with open(filename, 'rb') as file1:
            pk_model = pickle.load(file1)

    else:
        pk_model = trainKNN(KNN_k_neighbors)
    #  filename = "models/" + "RF_model2" + ".pkl"
    #y=  rf_criterion,rf_max_depth, rf_max_features, rf_min_samples_leaf,rf_n_estimators
    #


    y_pred = pk_model.predict(X_test)
    performaceDF = cR_to_df(pk_model, y_test, y_pred)
    confusionDF = cm2df(y_test, y_pred)
    #FeatureImportance = getFeatureImportance(pk_model)

    tab1 = html.Div(children=[
        html.Label(children='Performance Matrix', style={'width': '50%', 'display': 'inline-block',
                                                         'margin': 0, 'padding': '8px'}),
        dash_table.DataTable(
            id='table_no1',
            columns=[{"name": i, "id": i} for i in performaceDF.columns],
            data=performaceDF.to_dict("rows"),
            style_table={'width': '80%',
                         },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Parameters'},
                 'width': '50%'}
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={'width': '180px',
                        'height': '60px',
                        'textAlign': 'left',
                        'minWidth': '0px',
                        'maxWidth': '180px'
                        })])

    tab2 = html.Div(
        children=[html.Label(children='Confusion Matrix', style={'width': '50%', 'display': 'inline-block',
                                                                 'margin': 0, 'padding': '8px'}),
                  dash_table.DataTable(
                      id='confusionMatrix',
                      columns=[{"name": i, "id": i} for i in confusionDF.columns],
                      data=confusionDF.to_dict("rows"),
                      style_table={'width': '60%',
                                   },
                      style_data_conditional=[
                          {
                              'if': {'row_index': 'odd'},
                              'backgroundColor': 'rgb(248, 248, 248)'
                          }
                      ],
                      style_header={
                          'backgroundColor': 'rgb(230, 230, 230)',
                          'fontWeight': 'bold'
                      },
                      style_cell={'width': '180px',
                                  'height': '42px',
                                  'textAlign': 'left',
                                  'minWidth': '0px',
                                  'maxWidth': '180px'
                                  })])
    final =html.Div([
        html.Br([]),
        tab1, tab2], style=dict(display='flex'))
    #testdiv=([html.H4('Random Forest', style=dict(color='white', background='red'))])
    return final


@app.callback(Output('modelPerformanceGBM','children'),
              [Input('gbm_learning_rate', 'value'),
               Input('gbm_max_depth', 'value'),
                Input('gbm_max_features', 'value'),
                Input('gbm_min_samples_leaf', 'value'),
                Input('gbm_n_estimators', 'value')
               ]
              )
def update_gbmdatatable(gbm_learning_rate,gbm_max_depth,gbm_max_features,gbm_min_samples_leaf,gbm_n_estimators):

    if (gbm_learning_rate== 0.1 and gbm_max_depth==8 and gbm_max_features==7 and gbm_min_samples_leaf==4 and gbm_n_estimators==100):
        filename="models/" + "GBM_model_Best" + ".pkl"
        with open(filename, 'rb') as file1:
            pk_model = pickle.load(file1)

    else:
        pk_model = trainGBM(gbm_learning_rate,gbm_max_depth,gbm_max_features,gbm_min_samples_leaf,gbm_n_estimators)
        #filename = "models/" + "RF_model2" + ".pkl"



    y_pred = pk_model.predict(X_test)
    performaceDF = cR_to_df(pk_model, y_test, y_pred)
    confusionDF = cm2df(y_test, y_pred)
    FeatureImportance= getFeatureImportance(pk_model)

    tab1 = html.Div(children=[
        html.Label(children='Performance Matrix', style={'width': '50%', 'display': 'inline-block',
                                                         'margin': 0, 'padding': '8px'}),
        dash_table.DataTable(
            id='table_no1',
            columns=[{"name": i, "id": i} for i in performaceDF.columns],
            data=performaceDF.to_dict("rows"),
            style_table={'width': '80%',
                         },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Parameters'},
                 'width': '50%'}
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={'width': '180px',
                        'height': '60px',
                        'textAlign': 'left',
                        'minWidth': '0px',
                        'maxWidth': '180px'
                        })])

    tab2 = html.Div(
        children=[html.Label(children='Confusion Matrix', style={'width': '50%', 'display': 'inline-block',
                                                                 'margin': 0, 'padding': '8px'}),
                  dash_table.DataTable(
                      id='confusionMatrix',
                      columns=[{"name": i, "id": i} for i in confusionDF.columns],
                      data=confusionDF.to_dict("rows"),
                      style_table={'width': '60%',
                                   },
                      style_data_conditional=[
                          {
                              'if': {'row_index': 'odd'},
                              'backgroundColor': 'rgb(248, 248, 248)'
                          }
                      ],
                      style_header={
                          'backgroundColor': 'rgb(230, 230, 230)',
                          'fontWeight': 'bold'
                      },
                      style_cell={'width': '180px',
                                  'height': '42px',
                                  'textAlign': 'left',
                                  'minWidth': '0px',
                                  'maxWidth': '180px'
                                  })])

    FI=html.Div(
        children=[dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': FeatureImportance.index[:5], 'y': FeatureImportance[:5], 'type': 'bar', 'name': 'SF'}
            ],
            'layout': {
                'title': 'Feature Importance of the Model'
            }
        }
    )])
    final =html.Div([
        html.Br([]),
        tab1, tab2], style=dict(display='flex')),FI
    #testdiv=([html.H4('Random Forest', style=dict(color='white', background='red'))])
    return final


@app.callback(Output('modelPerformanceLGR','children'),
              [Input('lgr_solver', 'value'),
               Input('lgr_C', 'value')
               ]
              )
def update_lgrdatatable(lgr_solver,lgr_C):


    if (lgr_C== 0.01 and lgr_solver=='liblinear'):
        filename = "models/" + "LGR_model_Best" + ".pkl"
        with open(filename, 'rb') as file1:
            pk_model = pickle.load(file1)

    else:
        pk_model = trainLGR(lgr_solver,lgr_C)
        #filename = "models/" + "RF_model2" + ".pkl"



    y_pred = pk_model.predict(X_test)
    performaceDF = cR_to_df(pk_model, y_test, y_pred)
    confusionDF = cm2df(y_test, y_pred)
    FeatureCofficient= pk_model.coef_[0]


    tab1 = html.Div(children=[
        html.Label(children='Performance Matrix', style={'width': '50%', 'display': 'inline-block',
                                                         'margin': 0, 'padding': '8px'}),
        dash_table.DataTable(
            id='table_no1',
            columns=[{"name": i, "id": i} for i in performaceDF.columns],
            data=performaceDF.to_dict("rows"),
            style_table={'width': '80%',
                         },
            style_data={
                'whiteSpace': 'normal',
                'height': 'auto'
            },
            style_cell_conditional=[
                {'if': {'column_id': 'Parameters'},
                 'width': '50%'}
            ],
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={'width': '180px',
                        'height': '60px',
                        'textAlign': 'left',
                        'minWidth': '0px',
                        'maxWidth': '180px'
                        })])

    tab2 = html.Div(
        children=[html.Label(children='Confusion Matrix', style={'width': '50%', 'display': 'inline-block',
                                                                 'margin': 0, 'padding': '8px'}),
                  dash_table.DataTable(
                      id='confusionMatrix',
                      columns=[{"name": i, "id": i} for i in confusionDF.columns],
                      data=confusionDF.to_dict("rows"),
                      style_table={'width': '60%',
                                   },
                      style_data_conditional=[
                          {
                              'if': {'row_index': 'odd'},
                              'backgroundColor': 'rgb(248, 248, 248)'
                          }
                      ],
                      style_header={
                          'backgroundColor': 'rgb(230, 230, 230)',
                          'fontWeight': 'bold'
                      },
                      style_cell={'width': '180px',
                                  'height': '42px',
                                  'textAlign': 'left',
                                  'minWidth': '0px',
                                  'maxWidth': '180px'
                                  })])

    FI=html.Div(
        children=[dcc.Graph(
        id='example-graph',
        figure={
            'data': [
                {'x': X_SMOTE.columns, 'y': FeatureCofficient, 'type': 'bar', 'name': 'SF'}
            ],
            'layout': {
                'title': 'Coef of variables of the Model'
            }
        }
    )])
    final =html.Div([
        html.Br([]),
        tab1, tab2], style=dict(display='flex')),FI
    #testdiv=([html.H4('Random Forest', style=dict(color='white', background='red'))])
    return final
