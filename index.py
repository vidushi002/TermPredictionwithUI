
from app import app

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
from app import server
import callbacks
import io
pd.options.mode.chained_assignment = None
import flask

# see https://community.plot.ly/t/nolayoutexception-on-deployment-of-multi-page-dash-app-example-code/12463/2?u=dcomfort

#from app import server
#from app import app, server

import layouts
from layouts import Overview, RandomForest, noPage


#from callbacks import *
#import callbacks
# see https://dash.plot.ly/external-resources to alter header, footer and favicon
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Prediction of term deposit subscription Report</title>
        {%favicon%}
        {%css%}
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
        <div>Report</div>
    </body>
</html>
'''
# data = pd.read_csv('data/processedData.csv')
# print(data.shape)
# print(data.columns)




###########################################################################################################

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# Update page
# # # # # # # # #
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/Overview/':
        return Overview
    elif pathname == '/RandomForest/':
        return RandomForest
    elif pathname == '/KNN/':
        return layouts.KNN
    elif pathname == '/GBM/':
         return layouts.GBM
    elif pathname == '/LGR/':
         return layouts.LGR
    # elif pathname == '/cc-travel-report/display/':
    #     return layout_display
    # elif pathname == '/cc-travel-report/publishing/':
    #     return layout_publishing
    # elif pathname == '/cc-travel-report/metasearch-and-travel-ads/':
    #     return layout_metasearch
    else:
         return noPage

# # # # # # # # #
# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
#                 "https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
#                 "//fonts.googleapis.com/css?family=Raleway:400,300,600",
#                 "https://codepen.io/bcd/pen/KQrXdb.css",
#                 "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
#                 "https://codepen.io/dmcomfort/pen/JzdzEZ.css"]

# for css in external_css:
#     app.css.append_css({"external_url": css})

# external_js = ["https://code.jquery.com/jquery-3.2.1.min.js",
#                "https://codepen.io/bcd/pen/YaXojL.js"]
#
# for js in external_js:
#     app.scripts.append_script({"external_url": js})






#from components import formatter_currency, formatter_currency_with_cents, formatter_percent, formatter_percent_2_digits, formatter_number
#from components import update_first_datatable, update_first_download, update_second_datatable, update_graph


if __name__ == '__main__':
 app.run_server(debug=False)