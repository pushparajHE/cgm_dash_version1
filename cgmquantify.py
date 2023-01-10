import pandas as pd
import datetime as datetime
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess


def importdexcom(filename):
    """
        Imports data from Dexcom continuous glucose monitor devices
        Args:
            filename (String): path to file
        Returns:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
    """
    data = pd.read_csv(filename)
    df = pd.DataFrame()
    df['Actual time'] = data['Timestamp (YYYY-MM-DDThh:mm:ss)']
    df['Glucose reading'] = pd.to_numeric(data['Glucose Value (mg/dL)'])
    df.drop(df.index[:12], inplace=True)
    df['Actual time'] = pd.to_datetime(df['Actual time'], format='%Y-%m-%dT%H:%M:%S')
    df['Day'] = df['Actual time'].dt.date
    df = df.reset_index()
    return df


def interdaycv(df):
    """
        Computes and returns the interday coefficient of variation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            cvx (float): interday coefficient of variation averaged over all days

    """
    cvx = (np.std(df['Glucose reading']) / (np.mean(df['Glucose reading']))) * 100
    return cvx


def interdaysd(df):
    """
        Computes and returns the interday standard deviation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            interdaysd (float): interday standard deviation averaged over all days

    """
    interdaysd = np.std(df['Glucose reading'])
    return interdaysd


def intradaycv(df):
    """
        Computes and returns the intraday coefficient of variation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            intradaycv_mean (float): intraday coefficient of variation averaged over all days
            intradaycv_medan (float): intraday coefficient of variation median over all days
            intradaycv_sd (float): intraday coefficient of variation standard deviation over all days

    """
    intradaycv = []
    for i in pd.unique(df['Day']):
        intradaycv.append(interdaycv(df[df['Day'] == i]))

    intradaycv_mean = np.mean(intradaycv)
    intradaycv_median = np.median(intradaycv)
    intradaycv_sd = np.std(intradaycv)

    return intradaycv_mean, intradaycv_median, intradaycv_sd


def intradaysd(df):
    """
        Computes and returns the intraday standard deviation of glucose
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            intradaysd_mean (float): intraday standard deviation averaged over all days
            intradaysd_medan (float): intraday standard deviation median over all days
            intradaysd_sd (float): intraday standard deviation standard deviation over all days

    """
    intradaysd = []

    for i in pd.unique(df['Day']):
        intradaysd.append(np.std(df[df['Day'] == i]))

    intradaysd_mean = np.mean(intradaysd)
    intradaysd_median = np.median(intradaysd)
    intradaysd_sd = np.std(intradaysd)
    return intradaysd_mean, intradaysd_median, intradaysd_sd


def TIR(df, sd=1, sr=5):
    """
        Computes and returns the time in range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TIR (float): time in range, units=minutes

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TIR = len(df[(df['Glucose reading'] <= up) & (df['Glucose reading'] >= dw)]) * sr
    return TIR


def TOR(df, sd=1, sr=5):
    """
        Computes and returns the time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing  range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            TOR (float): time outside range, units=minutes

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TOR = len(df[(df['Glucose reading'] >= up) | (df['Glucose reading'] <= dw)]) * sr
    return TOR


def POR(df, sd=1, sr=5):
    """
        Computes and returns the percent time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            POR (float): percent time outside range, units=%

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TOR = len(df[(df['Glucose reading'] >= up) | (df['Glucose reading'] <= dw)]) * sr
    POR = (TOR / (len(df) * sr)) * 100
    return POR


def POut_side_R(df, dw=70,up=180):
    """
        Computes and returns the percent time outside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            POR (float): percent time outside range, units=%

    """
    #up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    #dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TOR = df[(df['Glucose reading'] > up) | (df['Glucose reading'] < dw)].shape[0]
    POR = (TOR / df.shape[0]) * 100
    return POR


def PInsideRange(df, dw=70, up=150):
    """
        Computes and returns the percent time inside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            PIR (float): percent time inside range, units=%

    """
    #up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    #dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TIR = len(df[(df['Glucose reading'] <= up) & (df['Glucose reading'] >= dw)])
    PIR = (TIR / (len(df))) * 100
    return PIR


def PIR(df, sd=1, sr=5):
    """
        Computes and returns the percent time inside range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
            sr (integer): sampling rate (default=5[minutes, once every 5 minutes glucose is recorded])
        Returns:
            PIR (float): percent time inside range, units=%

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    TIR = len(df[(df['Glucose reading'] <= up) | (df['Glucose reading'] >= dw)]) * sr
    PIR = (TIR / (len(df) * sr)) * 100
    return PIR


def MGE(df, sd=1):
    """
        Computes and returns the mean of glucose outside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGE (float): the mean of glucose excursions (outside specified range)

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    MGE = np.mean(df[(df['Glucose reading'] >= up) | (df['Glucose reading'] <= dw)])
    return MGE


def MGN(df, sd=1):
    """
        Computes and returns the mean of glucose inside specified range
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MGN (float): the mean of glucose excursions (inside specified range)

    """
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])
    MGN = np.mean(df[(df['Glucose reading'] <= up) & (df['Glucose reading'] >= dw)])
    return MGN


def MAGE(df, std=1):
    """
        Computes and returns the mean amplitude of glucose excursions
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation for computing range (default=1)
        Returns:
            MAGE (float): the mean amplitude of glucose excursions
        Refs:
            Sneh Gajiwala: https://github.com/snehG0205/NCSA_genomics/tree/2bfbb87c9c872b1458ef3597d9fb2e56ac13ad64

    """

    # extracting glucose values and incdices
    glucose = df['Glucose reading'].tolist()
    ix = [1 * i for i in range(len(glucose))]
    stdev = std

    # local minima & maxima
    a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1
    # local min
    valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1
    # local max
    peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1
    # +1 -- diff reduces original index number

    # store local minima and maxima -> identify + remove turning points
    excursion_points = pd.DataFrame(columns=['Index', 'Actual time', 'Glucose reading', 'Type'])
    k = 0
    for i in range(len(peaks)):
        excursion_points.loc[k] = [peaks[i]] + [df['Actual time'][k]] + [df['Glucose reading'][k]] + ["P"]
        k += 1

    for i in range(len(valleys)):
        excursion_points.loc[k] = [valleys[i]] + [df['Actual time'][k]] + [df['Glucose reading'][k]] + ["V"]
        k += 1

    excursion_points = excursion_points.sort_values(by=['Index'])
    excursion_points = excursion_points.reset_index(drop=True)

    # selecting turning points
    turning_points = pd.DataFrame(columns=['Index', 'Actual time', 'Glucose reading', 'Type'])
    k = 0
    for i in range(stdev, len(excursion_points.Index) - stdev):
        positions = [i - stdev, i, i + stdev]
        for j in range(0, len(positions) - 1):
            if (excursion_points.Type[positions[j]] == excursion_points.Type[positions[j + 1]]):
                if (excursion_points.Type[positions[j]] == 'P'):
                    if excursion_points['Glucose reading'][positions[j]] >= excursion_points['Glucose reading'][positions[j + 1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j + 1]]
                        k += 1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j + 1]]
                        k += 1
                else:
                    if excursion_points['Glucose reading'][positions[j]] <= excursion_points['Glucose reading'][positions[j + 1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j]]
                        k += 1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j + 1]]
                        k += 1

    if len(turning_points.index) < 10:
        turning_points = excursion_points.copy()
        excursion_count = len(excursion_points.index)
    else:
        excursion_count = len(excursion_points.index) / 2

    turning_points = turning_points.drop_duplicates(subset="Index", keep="first")
    turning_points = turning_points.reset_index(drop=True)
    excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
    excursion_points = excursion_points.reset_index(drop=True)

    # calculating MAGE
    mage = turning_points.Glucose.sum() / excursion_count

    return round(mage, 3)


def J_index(df):
    """
        Computes and returns the J-index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            J (float): J-index of glucose

    """
    J = 0.001 * ((np.mean(df['Glucose reading']) + np.std(df['Glucose reading'])) ** 2)
    return J


def LBGI_HBGI(df):
    """
        Connecter function to calculate rh and rl, used for ADRR function
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index
            HBGI (float): High blood glucose index
            rl (float): See calculation of LBGI
            rh (float): See calculation of HBGI

    """
    f = ((np.log(df['Glucose reading']) ** 1.084) - 5.381)
    rl = []
    for i in f:
        if (i <= 0):
            rl.append(22.77 * (i ** 2))
        else:
            rl.append(0)

    LBGI = np.mean(rl)

    rh = []
    for i in f:
        if (i > 0):
            rh.append(22.77 * (i ** 2))
        else:
            rh.append(0)

    HBGI = np.mean(rh)

    return LBGI, HBGI, rh, rl


def LBGI(df):
    """
        Computes and returns the low blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            LBGI (float): Low blood glucose index

    """
    f = ((np.log(df['Glucose reading']) ** 1.084) - 5.381)
    rl = []
    for i in f:
        if (i <= 0):
            rl.append(22.77 * (i ** 2))
        else:
            rl.append(0)

    LBGI = np.mean(rl)
    return LBGI


def HBGI(df):
    """
        Computes and returns the high blood glucose index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            HBGI (float): High blood glucose index

    """
    f = ((np.log(df['Glucose reading']) ** 1.084) - 5.381)
    rh = []
    for i in f:
        if (i > 0):
            rh.append(22.77 * (i ** 2))
        else:
            rh.append(0)

    HBGI = np.mean(rh)
    return HBGI


def ADRR(df):
    """
        Computes and returns the average daily risk range, an assessment of total daily glucose variations within risk space
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            ADRRx (float): average daily risk range

    """
    ADRRl = []
    for i in pd.unique(df['Day']):
        LBGI, HBGI, rh, rl = LBGI_HBGI(df[df['Day'] == i])
        LR = np.max(rl)
        HR = np.max(rh)
        ADRRl.append(LR + HR)

    ADRRx = np.mean(ADRRl)
    return ADRRx


def uniquevalfilter(df, value):
    """
        Supporting function for MODD and CONGA24 functions
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            value (datetime): time to match up with previous 24 hours
        Returns:
            MODD_n (float): Best matched with unique value, value

    """
    xdf = df[df['Minfrommid'] == value]
    n = len(xdf)
    diff = abs(xdf['Glucose reading'].diff())
    MODD_n = np.nanmean(diff)
    return MODD_n


def MODD(df):
    """
        Computes and returns the mean of daily differences. Examines mean of value + value 24 hours before
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Requires:
            uniquevalfilter (function)
        Returns:
            MODD (float): Mean of daily differences

    """
    df['Timefrommidnight'] = df['Actual time'].dt.time
    lists = []
    for i in range(0, len(df['Timefrommidnight'])):
        lists.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2]) * 60 + int(
            df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(
            int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9]) / 60))
    df['Minfrommid'] = lists
    df = df.drop(columns=['Timefrommidnight'])

    # Calculation of MODD and CONGA:
    MODD_n = []
    uniquetimes = df['Minfrommid'].unique()

    for i in uniquetimes:
        MODD_n.append(uniquevalfilter(df, i))

    # Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
    MODD_n[MODD_n == 0] = np.nan

    MODD = np.nanmean(MODD_n)
    return MODD


def CONGA24(df):
    """
        Computes and returns the continuous overall net glycemic action over 24 hours
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Requires:
            uniquevalfilter (function)
        Returns:
            CONGA24 (float): continuous overall net glycemic action over 24 hours

    """
    df['Timefrommidnight'] = df['Actual time'].dt.time
    lists = []
    for i in range(0, len(df['Timefrommidnight'])):
        lists.append(int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[0:2]) * 60 + int(
            df['Timefrommidnight'][i].strftime('%H:%M:%S')[3:5]) + round(
            int(df['Timefrommidnight'][i].strftime('%H:%M:%S')[6:9]) / 60))
    df['Minfrommid'] = lists
    df = df.drop(columns=['Timefrommidnight'])

    # Calculation of MODD and CONGA:
    MODD_n = []
    uniquetimes = df['Minfrommid'].unique()

    for i in uniquetimes:
        MODD_n.append(uniquevalfilter(df, i))

    # Remove zeros from dataframe for calculation (in case there are random unique values that result in a mean of 0)
    MODD_n[MODD_n == 0] = np.nan

    CONGA24 = np.nanstd(MODD_n)
    return CONGA24


def GMI(df):
    """
        Computes and returns the glucose management index
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            GMI (float): glucose management index (an estimate of HbA1c)

    """
    GMI = 3.31 + (0.02392 * np.mean(df['Glucose reading']))
    return GMI


def eA1c(df):
    """
        Computes and returns the American Diabetes Association estimated HbA1c
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            eA1c (float): an estimate of HbA1c from the American Diabetes Association

    """
    eA1c = (46.7 + np.mean(df['Glucose reading'])) / 28.7
    return eA1c


def summary(df):
    """
        Computes and returns glucose summary metrics
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
        Returns:
            meanG (float): interday mean of glucose
            medianG (float): interday median of glucose
            minG (float): interday minimum of glucose
            maxG (float): interday maximum of glucose
            Q1G (float): interday first quartile of glucose
            Q3G (float): interday third quartile of glucose

    """
    meanG = np.nanmean(df['Glucose reading'])
    medianG = np.nanmedian(df['Glucose reading'])
    minG = np.nanmin(df['Glucose reading'])
    maxG = np.nanmax(df['Glucose reading'])
    Q1G = np.nanpercentile(df['Glucose reading'], 25)
    Q3G = np.nanpercentile(df['Glucose reading'], 75)

    return meanG, medianG, minG, maxG, Q1G, Q3G


def plotglucosesd(df, sd=1, size=15):
    """
        Plots glucose with specified standard deviation lines
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            sd (integer): standard deviation lines to plot (default=1)
            size (integer): font size for plot (default=15)
        Returns:
            plot of glucose with standard deviation lines

    """
    glucose_mean = np.mean(df['Glucose reading'])
    up = np.mean(df['Glucose reading']) + sd * np.std(df['Glucose reading'])
    dw = np.mean(df['Glucose reading']) - sd * np.std(df['Glucose reading'])

    plt.figure(figsize=(20, 5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Actual time'], df['Glucose reading'], '.', color='#1f77b4')
    plt.axhline(y=glucose_mean, color='red', linestyle='-')
    plt.axhline(y=up, color='pink', linestyle='-')
    plt.axhline(y=dw, color='pink', linestyle='-')
    plt.ylabel('Glucose reading')
    plt.show()


def plotglucosebounds(df, upperbound=180, lowerbound=70, size=15):
    """
        Plots glucose with user-defined boundaries
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            upperbound (integer): user defined upper bound for glucose line to plot (default=180)
            lowerbound (integer): user defined lower bound for glucose line to plot (default=70)
            size (integer): font size for plot (default=15)
        Returns:
            plot of glucose with user defined boundary lines

    """
    plt.figure(figsize=(20, 5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Actual time'], df['Glucose reading'], '.', color='#1f77b4')
    plt.axhline(y=upperbound, color='red', linestyle='-')
    plt.axhline(y=lowerbound, color='orange', linestyle='-')
    plt.ylabel('Glucose reading')
    plt.show()


def plotglucosesmooth(df,st, size=15):
    """
        Plots smoothed glucose plot (with LOWESS smoothing)
        Args:
            (pd.DataFrame): dataframe of data with DateTime, Time and Glucose columns
            size (integer): font size for plot (default=15)
        Returns:
            LOWESS-smoothed plot of glucose

    """
    filteres = lowess(df['Glucose reading'], df['Actual time'], is_sorted=True, frac=0.025, it=0)
    filtered = pd.to_datetime(filteres[:, 0], format='%Y-%m-%dT%H:%M:%S')

    fig = plt.figure(figsize=(20, 5))
    plt.rcParams.update({'font.size': size})
    plt.plot(df['Actual time'], df['Glucose reading'], '.')
    plt.plot(filtered, filteres[:, 1], 'r')
    plt.ylabel('Glucose reading')
    st.plotly_chart(fig)
    #plt.show()