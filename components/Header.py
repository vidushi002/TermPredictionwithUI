import dash_html_components as html
import dash_core_components as dcc

def Header():
    return html.Div([
        get_logo(),
        get_header(),
        html.Br([]),
        get_menu()
    ])

def get_logo():
    logo = html.Div([

        html.Div([
            html.Img(src='https://cdn0.iconfinder.com/data/icons/business-1-31/129/93-512.png', height='101', width='141')
        ], className="ten columns padded"),

        # html.Div([
        #     dcc.Link('Full View   ', href='/cc-travel-report/full-view')
        # ], className="two columns page-view no-print")

    ], className="row gs-header")
    return logo


def get_header():
    header = html.Div([

        html.Div([
            html.H3(
                'Prediction of term deposit subscription')
        ], className="twelve columns padded")

    ], className="row gs-header gs-text-header")
    return header


def get_menu():
    menu = html.Div([

        dcc.Link('Overview', href='/Overview/',style={'width': '18%','display': 'inline-block',
			'border': '1px solid','textAlign':'center','fontWeight': 'bold','background':'blue','color':'white'
            ,'font-size': '22px'
    }),
        dcc.Link('Logistic Regression   ', href='/LGR/', className="tab", style={'width': '22%', 'display': 'inline-block',
                                                                 'border': '1px solid', 'textAlign': 'center',
                                                                 'fontWeight': 'bold', 'background': 'blue',
                                                                 'color': 'white', 'font-size': '22px'
                                                                 }),

        dcc.Link('Random Forest         ', href='/RandomForest/',style={'width': '18%','display': 'inline-block',
			'border': '1px solid','textAlign':'center','fontWeight': 'bold','background':'blue','color':'white','font-size': '22px'
			}),

        dcc.Link('KNN   ', href='/KNN/', className="tab",style={'width': '18%','display': 'inline-block',
			'border': '1px solid','textAlign':'center','fontWeight': 'bold','background':'blue','color':'white','font-size': '22px'
			}),
        dcc.Link('GBM   ', href='/GBM/', className="tab", style={'width': '18%', 'display': 'inline-block',
                                                                 'border': '1px solid', 'textAlign': 'center',
                                                                 'fontWeight': 'bold', 'background': 'blue',
                                                                 'color': 'white', 'font-size': '22px'
                                                                 }),

        #
        # dcc.Link('Display   ', href='/cc-travel-report/display/', className="tab"),
        #
        # dcc.Link('Publishing   ', href='/cc-travel-report/publishing/', className="tab"),
        #
        # dcc.Link('Metasearch and Travel Ads   ', href='/cc-travel-report/metasearch-and-travel-ads/', className="tab"),

    ], className="row ")
    return menu
