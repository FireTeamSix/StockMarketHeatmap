import bs4 as bs
import datetime as dt
import os
import pandas as pd
from pandas_datareader import data as pdr
import pickle
import requests
import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
from collections import Counter
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

style.use('ggplot')
yf.pdr_override()


def save_sp500_tickers():
    """Uses bs4 to pull list of S&P 500 companies from Wikipedia"""

    resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable sortable'})

    tickers1 = []

    for row in table.findAll('tr')[1:201]:

        ticker = row.findAll('td')[0].text.replace('.', '-')
        ticker = ticker[:-1]
        tickers1.append(ticker)

    with open("sp500tickers1.pickle", "wb") as f:
        pickle.dump(tickers1, f)

    tickers2 = []

    for row in table.findAll('tr')[201:401]:

        ticker2 = row.findAll('td')[0].text.replace('.', '-')
        ticker2 = ticker2[:-1]
        tickers2.append(ticker2)

    with open("sp500tickers2.pickle", "wb") as f:
        pickle.dump(tickers2, f)

    tickers3 = []

    for row in table.findAll('tr')[401:506]:

        ticker3 = row.findAll('td')[0].text.replace('.', '-')
        ticker3 = ticker3[:-1]
        tickers3.append(ticker3)

    with open("sp500tickers3.pickle", "wb") as f:
        pickle.dump(tickers3, f)

    return tickers1, tickers2, tickers3


def get_data_from_yahoo(reload_sp500=False):
    """Downloads all data from 1,1,2000 using Yahoo Stocks API 
    (NOTE: this was shut down, but a fix has been implemented using then "fix_yahoo_finance" library)"""

    if reload_sp500:
        tickers1, tickers2, tickers3 = save_sp500_tickers()

    else:
        with open("sp500tickers1.pickle", "rb") as f:
            tickers1 = pickle.load(f)

        with open("sp500tickers2.pickle", "rb") as f:
            tickers2 = pickle.load(f)

        with open("sp500tickers3.pickle", "rb") as f:
            tickers3 = pickle.load(f)

    if not os.path.exists('stock_dfs1'):
        os.makedirs('stock_dfs1')

    if not os.path.exists('stock_dfs2'):
        os.makedirs('stock_dfs2')
    
    if not os.path.exists('stock_dfs3'):
        os.makedirs('stock_dfs3')

    start = dt.datetime(2000, 1, 1)
    end = dt.datetime.now()

    for ticker in tickers1:
        print(ticker)
        if not os.path.exists('stock_dfs1/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs1/{}.csv'.format(ticker))

        else:
            print('Already have {}'.format(ticker))

    for ticker in tickers2:
        print(ticker)
        if not os.path.exists('stock_dfs2/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs2/{}.csv'.format(ticker))

        else:
            print('Already have {}'.format(ticker))

    for ticker in tickers3:
        print(ticker)
        if not os.path.exists('stock_dfs3/{}.csv'.format(ticker)):
            df = pdr.get_data_yahoo(ticker, start, end)
            df.reset_index(inplace=True)
            df.set_index("Date", inplace=True)
            df.to_csv('stock_dfs3/{}.csv'.format(ticker))

        else:
            print('Already have {}'.format(ticker))



def compile_data():
    """Compiles data into 3 separate .pickle files"""

    with open('sp500tickers1.pickle', 'rb') as f:
        tickers1 = pickle.load(f)

    with open('sp500tickers2.pickle', 'rb') as f:
        tickers2 = pickle.load(f)

    with open('sp500tickers3.pickle', 'rb') as f:
        tickers3 = pickle.load(f)

    # #############################################################################################

    main_df1 = pd.DataFrame()
    print("1st Round (stocks 1-200)")
    for count, ticker in enumerate(tickers1):
        df = pd.read_csv('stock_dfs1/{}.csv'.format(ticker.replace('.', '-')))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close'], 1, inplace=True)

        if main_df1.empty:
            main_df1 = df

        else:
            main_df1 = main_df1.merge(df, how='outer')

        

        if count % 10 == 0: 
            print(count)
        
    print("200")

    main_df1.to_csv('sp500_joined_closes1.csv')

    # #############################################################################################

    main_df2 = pd.DataFrame()
    print("2nd Round (stocks 201-400)")
    for count, ticker in enumerate(tickers2):
        df = pd.read_csv('stock_dfs2/{}.csv'.format(ticker.replace('.', '-')))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close'], 1, inplace=True)

        if main_df2.empty:
            main_df2 = df

        else:
            main_df2 = main_df2.merge(df, how='outer')

        

        if count % 10 == 0: 
            print(count)

    main_df2.to_csv('sp500_joined_closes2.csv')

    # #############################################################################################

    main_df3 = pd.DataFrame()
    print("3rd Round (stocks 400-500)")
    for count, ticker in enumerate(tickers3):
        df = pd.read_csv('stock_dfs3/{}.csv'.format(ticker.replace('.', '-')))
        df.set_index('Date', inplace=True)

        df.rename(columns={'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close'], 1, inplace=True)

        if main_df3.empty:
            main_df3 = df

        else:
            main_df3 = main_df3.merge(df, how='outer')

        

        if count % 10 == 0: 
            print(count)

    main_df3.to_csv('sp500_joined_closes3.csv')

    print("All 500 S&P stocks have been compiled")


def visualize_data():
    """Uses pyplot from matplotlib to graph heatmap of data"""
    ########################################################################################
    df1 = pd.read_csv('sp500_joined_closes1.csv')

    df_corr1 = df1.corr()
    print(df_corr1.head())
    
    data = df_corr1.values
    fig = plt.figure(1)
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr1.columns
    row_labels = df_corr1.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)
    
    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    ########################################################################################
    df2 = pd.read_csv('sp500_joined_closes2.csv')

    df_corr2 = df2.corr()
    print(df_corr2.head())
    
    data = df_corr2.values
    fig = plt.figure(2)
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr2.columns
    row_labels = df_corr2.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    ########################################################################################
    df3 = pd.read_csv('sp500_joined_closes3.csv')

    df_corr3 = df3.corr()
    print(df_corr3.head())
    
    data = df_corr3.values
    fig = plt.figure(3)
    ax = fig.add_subplot(1,1,1)

    heatmap = ax.pcolor(data, cmap=plt.cm.RdYlGn)
    fig.colorbar(heatmap)

    ax.set_xticks(np.arange(data.shape[0]) + 0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1]) + 0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()
    
    column_labels = df_corr3.columns
    row_labels = df_corr3.index

    ax.set_xticklabels(column_labels)
    ax.set_yticklabels(row_labels)

    plt.xticks(rotation=90)
    heatmap.set_clim(-1,1)
    plt.tight_layout()

    plt.show()


# Preprocessing

def process_data_for_labels(ticker):
    """Processes .pickle files to return lists of labels (tickers)"""
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes1.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    """Determines to buy, sell, or hold data"""
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0


def extract_featuresets(ticker):
    """Extracts values of all stocks from .pickle files"""
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df


def do_ml(ticker):

    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    clf = neighbors.KNeighborsClassifier()

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    
    predictions = clf.predict(X_test)
    print('Predicted Spread:', Counter(predictions))

    return confidence


# save_sp500_tickers()
# get_data_from_yahoo()
# compile_data()
# visualize_data()

# process_data_for_labels()
# buy_sell_hold()
# extract_featuresets())

do_ml('BAC')
