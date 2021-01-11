import dash_core_components as dcc
import dash_html_components as html
import dash_table
from components import Header

#from components import print_button
from datetime import datetime as dt
from datetime import date, timedelta
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

from components import printButton



######################## START Birst Category Layout ########################
Overview = html.Div([


    html.Div([
        # CC Header
        Header.Header(),
        # Header Bar
        html.Br([]),
        html.Br([]),

        html.Div([
            html.Img(src='assets/overview.jpeg', height='800',
                     width='1050')
        ]),

    ], className="subpage")
], className="page")




######################## END Overview Layout ########################

######################## START RandomForest Layout ########################

RandomForest = html.Div([

    #    print_button(),

    html.Div([
        # CC Header
        Header.Header(),
        # Header Bar
        html.Br([]),
        html.Div([
        html.H4('Random Forest', style=dict(color='white', background='red',width='30%')),

        ]),
        html.Br([]),
        html.Div(
    [   html.H6("Hyper Parameters"),
        html.Label(children='Criterion :', style={'width': '12%',
                                                 'margin': 0, 'padding': '8px'}),
        dcc.Dropdown(id='rf_criterion',
                     options=[{'label': i, 'value': i} for i in ['entropy','gini']],
                     value='gini',style={'width': '53%','display': 'inline-block'}),

            html.Label(children='max_depth:', style={'width': '12%',
                                                            'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="rf_max_depth", type="number", value =8,
            min=2, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='max_features:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="rf_max_features", type="number", value=6,
            min=4, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='min_samples_leaf:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="rf_min_samples_leaf", type="number", value=2,
            min=1, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='n_estimators:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="rf_n_estimators", type="number", value=100,
            min=1, max=500, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Div(
            id='modelPerformanceRFGBM'
        ),
        # dcc.Input(
        #     id="modelPerformanceRFGBM", type="number", value=200,
        #     min=1, max=500, step=1, style={'width': '28%', 'display': 'inline-block'}
        # ),
        html.Hr(),

    ]
),

    ], className="subpage")
], className="page")

######################## START Random Forest Layout ########################

######################## START GBM Layout ########################

GBM = html.Div([

    #    print_button(),

    html.Div([
        # CC Header
        Header.Header(),
        # Header Bar
        html.Br([]),
        html.Div([
        html.H4('GBM', style=dict(color='white', background='red',width='20%')),

        ]),
        html.Br([]),
        html.Div(
    [   html.H6("Hyper Parameters"),
        html.Label(children='learning_rate :', style={'width': '12%',
                                                 'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="gbm_learning_rate", type="number", value =0.1,
            min=0.01, max=1, step=0.01,style={'width': '28%','display': 'inline-block'}
        ),

            html.Label(children='max_depth:', style={'width': '12%',
                                                            'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="gbm_max_depth", type="number", value =8,
            min=2, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='max_features:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="gbm_max_features", type="number", value=7,
            min=4, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='min_samples_leaf:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="gbm_min_samples_leaf", type="number", value=2,
            min=1, max=20, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Label(children='n_estimators:', style={'width': '12%',
                                                    'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="gbm_n_estimators", type="number", value=100,
            min=1, max=500, step=1,style={'width': '28%','display': 'inline-block'}
        ),
        html.Div(
            id='modelPerformanceGBM'
        ),
        html.Hr(),

    ]
),

    ], className="subpage")
], className="page")

######################## END GBM Layout ########################

######################## START KNN Layout ########################
KNN = html.Div([

    #    print_button(),

    html.Div([
        # CC Header
        Header.Header(),
        # Header Bar
        html.Br([]),
        html.Div([
        html.H4('KNN', style=dict(color='white', background='red',width='20%')),

        ]),
html.Br([]),
        html.Div(
    [   html.H6("Hyper Parameters"),

            html.Label(children='k_neighbors:', style={'width': '12%',
                                                            'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="KNN_k_neighbors", type="number", value =5,
            min=2, max=100, step=1,style={'width': '28%','display': 'inline-block'}
        )]),
        html.Div(
            id='modelPerformanceKNN'
        ),
    ], className="subpage")
], className="page")

######################## Logistic Regrression ########################

LGR = html.Div([

    #    print_button(),

    html.Div([
        # CC Header
        Header.Header(),
        # Header Bar
        html.Br([]),
        html.Div([
        html.H4('Logistic Regression', style=dict(color='white', background='red',width='30%')),

        ]),
html.Br([]),
        html.Div(
    [   html.H6("Hyper Parameters"),


            html.Label(children='Solver :', style={'width': '12%',
                                                 'margin': 0, 'padding': '8px'}),
            dcc.Dropdown(id='lgr_solver',
                     options=[{'label': i, 'value': i} for i in ['liblinear', 'lbfgs',  'saga']],
                     value='liblinear',style={'width': '53%','display': 'inline-block'}),

            html.Label(children='C:', style={'width': '12%',
                                                            'margin': 0, 'padding': '8px'}),
        dcc.Input(
            id="lgr_C", type="number", value =0.01,
            #
            style={'width': '28%','display': 'inline-block'}
        )]),
        html.Div(
            id='modelPerformanceLGR'
        ),
    ], className="subpage")
], className="page")


######################## 404 Page ########################
noPage = html.Div([
    # CC Header
    Header.Header(),
    html.P(["404 Page not found"])
    ], className="no-page")


