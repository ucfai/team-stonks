import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def readStockData(symbol):
    return pd.read_csv("data/stock-data/" + symbol + ".csv", delimiter=",")


def chart(df, averages=[]):
    # plot data
    fig, ax = plt.subplots(figsize=(20,10))
    ax.plot(df['Adj Close'], color='green')

    for avg in averages:
        ax.plot(df['MA' + str(avg)])

    ax.set_ylabel('Price')
    ax.set_xlabel('Days')
    ax.legend()

    plt.show()



def movingAverage(df, averages=[5,15,50,200]):
    # return a dataframe with a new moving average column

    print(df.tail())

    # expand our data to several different moving averages
    for avg in averages:
        # easier way to calculate moving averages than using for loop method
        df['MA' + str(avg)] = df['Adj Close'].rolling(window=avg).mean()

    print(df.tail())

    return df


if __name__=="__main__":
    df = readStockData("AAPL")
    df = movingAverage(df)
    chart(df,  averages=[5, 15, 50, 200])
