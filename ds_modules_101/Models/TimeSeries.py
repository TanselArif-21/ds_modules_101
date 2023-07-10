import pandas as pd
import numpy as np
from numpy.random import seed
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels as sm
import statsmodels.api as smapi
from time import sleep

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from itertools import product


def get_time_series(num_obs, arparams, maparams, this_seed=None, suptitle='', title=None,
                    title_prepend='', title_append='', ax=None, seasonal_magnitude=0,
                    seasonal_period=0, trend_gradient=0, trend_gradient_list=(),
                    struct_change=False, get_figure=False):
    """
    This function generates a time series with the specified parameters.

    Args:
        num_obs (int): Number of observations in the time series.
        arparams (list or np.array): Parameters for the AR process.
        maparams (list or np.array): Parameters for the MA process.
        this_seed (int, optional): Random seed for reproducibility. Defaults to None.
        suptitle (str, optional): Supertitle for the plot. Defaults to ''.
        title (str, optional): Title for the plot. Defaults to None.
        title_prepend (str, optional): String to prepend to the title. Defaults to ''.
        title_append (str, optional): String to append to the title. Defaults to ''.
        ax (matplotlib axis object, optional): An existing axis to plot on. Defaults to None.
        seasonal_magnitude (int, optional): Amplitude of the seasonal component. Defaults to 0.
        seasonal_period (int, optional): Period of the seasonal component. Defaults to 0.
        trend_gradient (float, optional): Gradient of the trend component. Defaults to 0.
        trend_gradient_list (list, optional): List of gradients for multiple trend components. Defaults to [].
        struct_change (bool, optional): Whether to allow for structural changes in the trend component.
                                        Defaults to False.
        get_figure (bool, optional): Whether to plot the generated time series. Defaults to False.

    Returns:
        pd.DataFrame: Time series data as a DataFrame with 'Day' and 'Value' columns.
        matplotlib.figure.Figure: Figure of the generated time series, or None if get_figure is False.
    """

    # Setting random seed if provided
    if this_seed is not None:
        np.random.seed(this_seed)

    # Converting AR and MA parameters to numpy arrays
    arparams = np.array(arparams)
    maparams = np.array(maparams)

    # Generating the seasonal component if the period is greater than or equal to 2
    if seasonal_period >= 2:
        y_seasonal = seasonal_magnitude * np.sin(np.linspace(0, (50000 * 2 / seasonal_period) * np.pi, 50000))[:num_obs]
    else:
        print('Seasonal period can be minimum 2')
        y_seasonal = np.zeros(num_obs)

    # Setting up the title for the plot
    if title is None:
        title = 'A sample time series generated \nwith ar={}, ma={}, s={}, t={}, ts={} and struct={}'
    title = title_prepend + title + title_append

    # Set up the AR and MA components with zero-lag
    ar = np.r_[1, -arparams]  # add zero-lag and negate
    ma = np.r_[1, maparams]  # add zero-lag

    # Generating the time series with the AR and MA components
    y = sm.tsa.arima_process.arma_generate_sample(ar, ma, num_obs)

    # Add the seasonal component
    y = y + y_seasonal

    # Counting the number of structural changes
    num_struct_change = len(trend_gradient_list)

    # Setting up the trend component
    y_trend = None
    if struct_change:
        for i_struct_change in range(num_struct_change):
            num_obs_temp = num_obs // num_struct_change
            if i_struct_change == num_struct_change - 1:
                num_obs_temp = num_obs - len(y_trend)
            if y_trend is None:
                y_trend = np.linspace(0, num_obs_temp, num_obs_temp) * trend_gradient_list[i_struct_change]
            else:
                y_trend_temp = np.linspace(0, num_obs_temp, num_obs_temp) * trend_gradient_list[i_struct_change]
                y_trend = np.concatenate((y_trend, y_trend_temp), axis=0)
    else:
        y_trend = np.linspace(0, num_obs, num_obs) * trend_gradient

    # Add the trend component to the series
    y = y + y_trend

    # Create a DataFrame from the generated time series data
    time_series = pd.DataFrame({'Day': pd.date_range('2018-01-01', periods=num_obs, freq='D'),
                                'Value': y
                                })

    # Handling the case of no seasonal magnitude
    if seasonal_magnitude == 0:
        seasonal_period = 0

    fig = None
    # Plot the time series if required
    if get_figure:
        if ax is None:
            fig = plt.figure(figsize=(30, 15))
            ax = fig.add_subplot(1, 1, 1)

        sns.lineplot(x=time_series.Day, y=time_series.Value, linewidth=3, ax=ax)
        plt.suptitle(suptitle, ha='center', va='top', x=0.5, y=1.1, fontsize=20, color='grey')
        ax.set_title(title.format(ar, ma, seasonal_period, trend_gradient, trend_gradient_list, struct_change),
                     fontsize=15, color='grey', loc='left')
        ax.set_xlabel('Day', fontsize=15, color='grey')
        ax.set_ylabel('Value', fontsize=15, color='grey')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.tick_params(axis='x', colors='grey')
        ax.tick_params(axis='y', colors='grey')

    # Return the time series data and the figure if it was generated
    return time_series, fig


def stationarity_check(t_series):
    """
    This function performs a stationarity check on a given time series.

    Args:
        t_series (pd.Series): A pandas Series representing the time series data.

    Returns:
        str: A string indicating whether the series is trend stationary or not
             and the associated p-value from the Augmented Dickey-Fuller test.

    Raises:
        Exception: An error occurred if the input t_series is not a pandas Series.
    """

    # Check if the input time series is of type pandas Series
    if not isinstance(t_series, pd.Series):
        raise Exception('t_series should be of type pandas Series')

    # Compute the p-value using the Augmented Dickey-Fuller test
    pvalue = adfuller(t_series)[1]

    # Check the p-value against the significance level (0.05) to determine stationarity
    if pvalue <= 0.05:
        return 'Trend stationary with p-value: {}'.format(pvalue)
    else:
        return 'Not trend stationary with p-value: {}'.format(pvalue)


def plot_time_series_line(t_series, title):
    """
    This function plots a given time series.

    Args:
        t_series (pd.Series): A pandas Series representing the time series data.
        title (str): The title of the plot.

    Returns:
        None
    """
    # Creating a new figure with specified dimensions
    fig = plt.figure(figsize=(20, 8))

    # Adding a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # Plotting the time series data with seaborn's lineplot function
    sns.lineplot(x=t_series.index, y=t_series, linewidth=3, ax=ax)

    # Setting the title, xlabel and ylabel with the provided title and standard names
    ax.set_title(title, fontsize=15, color='grey', loc='left')
    ax.set_xlabel('Time', fontsize=15, color='grey')
    ax.set_ylabel('Value', fontsize=15, color='grey')

    # Removing the top, right, bottom and left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Setting the color of the x and y tick params to grey
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')


def plot_time_series_line_traintest(t_series_train, t_series_test, title):
    """
    This function plots two given time series, intended to represent train and test sets.

    Args:
        t_series_train (pd.Series): A pandas Series representing the training time series data.
        t_series_test (pd.Series): A pandas Series representing the testing time series data.
        title (str): The title of the plot.

    Returns:
        None
    """

    # Creating a new figure with specified dimensions
    fig = plt.figure(figsize=(20, 8))

    # Adding a subplot to the figure
    ax = fig.add_subplot(1, 1, 1)

    # Plotting the train and test time series data with seaborn's lineplot function
    sns.lineplot(x=t_series_train.index, y=t_series_train, linewidth=3, ax=ax, color='#2596be', label='Train')
    sns.lineplot(x=t_series_test.index, y=t_series_test, linewidth=3, ax=ax, color='#e28743', label='Test')

    # Setting the title, xlabel and ylabel with the provided title and standard names
    ax.set_title(title, fontsize=15, color='grey', loc='left')
    ax.set_xlabel('Time', fontsize=15, color='grey')
    ax.set_ylabel('Value', fontsize=15, color='grey')

    # Removing the top, right, bottom and left spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    # Setting the color of the x and y tick params to grey
    ax.tick_params(axis='x', colors='grey')
    ax.tick_params(axis='y', colors='grey')


def plot_acf_pacf(t_series, lags):
    """ If seasonality is difficult to observe, look at the acf,pacf """
    # Plot the ACF and PACF
    plot_acf(t_series, lags=lags)
    plot_pacf(t_series, method='yw', lags=lags)


def check_stationarity(t_series):
    """ Check stationarity """
    s = stationarity_check(t_series)
    print('The time series is {}'.format(s))
    return s


def check_ljungbox(t_series):
    """ Does ljung box test show evidence of autocorrelation? """
    ljung_box_pvalue = \
        smapi.stats.acorr_ljungbox(t_series, lags=[np.log(len(t_series))], return_df=True)['lb_pvalue'].values[0]
    if ljung_box_pvalue <= 0.05:
        print(
            'By the Ljung-Box test, there is evidence of autocorrelation. So a SARIMA model is reasonable. Pvalue:{}'.format(
                ljung_box_pvalue))
    else:
        print(
            'By the Ljung-Box test, there no is evidence of autocorrelation. So a SARIMA model might not help. Pvalue:{}'.format(
                ljung_box_pvalue))


def apply_all_models(t_series_train, p_ls, d_ls, q_ls, P_ls, D_ls, Q_ls, s_ls, trend_ls, max_order=8,
                     t_series_test=None, min_order=0, t_series_full=None, exog_train=None, exog_test=None,
                     exog_full=None):
    """ Iterate through all candidate models """

    if exog_train is None:
        exog_test = None

    if exog_test is None:
        exog_train = None

    a = list(product(p_ls, d_ls, q_ls, P_ls, D_ls, Q_ls, s_ls, trend_ls))
    a = sorted(a, key=lambda x: np.sum(x[:-2]))
    idx_list = []
    aic_list = []
    lb_list = []
    mse_list = []
    degree_list = []
    p_list = []
    d_list = []
    q_list = []
    P_list = []
    D_list = []
    Q_list = []
    s_list = []
    t_list = []

    for idx, (p, d, q, P, D, Q, s, t) in enumerate(a):
        order = p + d + q + P + D + Q
        if (order <= max_order) and (order >= min_order):
            try:
                mod = SARIMAX(t_series_train, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=t, exog=exog_train)
                model = mod.fit()

                if t_series_full is not None:
                    mod2 = SARIMAX(t_series_full, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=t, exog=exog_full)
                    model2 = mod2.fit()
                    aic = model2.aic
                    lb = \
                        smapi.stats.acorr_ljungbox(model2.resid, lags=[np.log(len(model2.resid))], return_df=True)[
                            'lb_pvalue'].values[0]
                else:
                    aic = model.aic
                    lb = \
                        smapi.stats.acorr_ljungbox(model.resid, lags=[np.log(len(model.resid))], return_df=True)[
                            'lb_pvalue'].values[0]

                if t_series_test is not None:
                    preds = model.predict(len(t_series_train), len(t_series_train) + len(t_series_test) - 1,
                                          exog=exog_test)
                    mse_test = ((preds - t_series_test) ** 2).sum() / len(t_series_test)
                else:
                    mse_test = 0

                idx_list.append(idx)
                aic_list.append(aic)
                lb_list.append(lb)
                mse_list.append(mse_test)
                degree_list.append(order)
                p_list.append(p)
                d_list.append(d)
                q_list.append(q)
                P_list.append(P)
                D_list.append(D)
                Q_list.append(Q)
                s_list.append(s)
                t_list.append(t)

                if lb <= 0.05:
                    print(
                        '{idx} - SARIMA({p},{d},{q},{P},{D},{Q},{s}),trend={t} - aic:{aic}, test_mse = {mse_test}, Ljung-Box:{lb} - Degree={order}'.format(
                            idx=idx, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, t=t, aic=aic, mse_test=mse_test, lb=lb,
                            order=order))
                else:
                    print(
                        '{idx} - SARIMA({p},{d},{q},{P},{D},{Q},{s}),trend={t} - aic:{aic}, test_mse = {mse_test}, Ljung-Box:{lb} - Degree={order} - LB Not Significant'.format(
                            idx=idx, p=p, d=d, q=q, P=P, D=D, Q=Q, s=s, t=t, aic=aic, mse_test=mse_test, lb=lb,
                            order=order))
            except:
                pass

    df_results = pd.concat(
        [pd.Series(idx_list), pd.Series(aic_list), pd.Series(lb_list), pd.Series(mse_list), pd.Series(degree_list),
         pd.Series(p_list), pd.Series(d_list), pd.Series(q_list), pd.Series(P_list), pd.Series(D_list),
         pd.Series(Q_list), pd.Series(s_list), pd.Series(t_list)], axis=1)
    df_results.columns = ['idx', 'AIC', 'LBPvalue', 'Test MSE', 'Degree', 'p', 'd', 'q', 'P', 'D', 'Q', 's', 't']

    return df_results


def final_model(t_series, p, d, q, P, D, Q, s, trend='n', exog=None):
    mod = SARIMAX(t_series, order=(p, d, q), seasonal_order=(P, D, Q, s), trend=trend, exog=exog)
    model = mod.fit()
    t = pd.concat([model.params, model.pvalues], axis=1)
    t.columns = ['coefficients', 'pvalue']
    t = t[(t['pvalue'] <= 0.05) & (t.index.str.contains('.L'))].copy()
    print('***These are the lag coefficients in a regression***')
    print(t)
    print('***These are the full results of the regression***')
    print(model.summary())
    return model


def predict_test(model, df_series, train_size, dynamic=True, val2=None, target_col_name='Value'):
    df_series_fit = df_series.iloc[:train_size]
    df_series_test = df_series.iloc[train_size:]
    t_series_fit = df_series_fit[target_col_name]

    if val2 is not None:
        t_series2_test = df_series_test[val2]

    start = len(t_series_fit)
    end = len(df_series) - 1

    # Predictions
    if val2 is not None:
        t_series_predicted = model.predict(start, end, dynamic=dynamic, exog=t_series2_test).rename("Predictions")
    else:
        t_series_predicted = model.predict(start, end, dynamic=dynamic).rename("Predictions")

    t_series_fit = t_series_fit.append(t_series_predicted)
    df_series_final = pd.concat([df_series, t_series_fit], axis=1)
    df_series_final = df_series_final.rename(columns={0: 'Predicted'})

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    sns.lineplot(x=df_series_final.index, y=df_series_final['Value'], label='Original', ax=ax)
    sns.lineplot(x=df_series_final.index[train_size:], y=df_series_final['Predicted'][train_size:], label='Predicted',
                 ax=ax)


def forecast_time_series(model, df_series, forecast_size, dynamic=True, exog=None, target_col_name='Value',
                         frequency=None):
    df_series_fit = df_series
    t_series_fit = df_series_fit[target_col_name]
    start = len(df_series)
    end = len(df_series) + forecast_size + 1

    if exog is not None:
        if end - start + 1 > len(exog):
            print('exog has {} rows but you are trying to forcast {}'.format(len(exog), forecast_size))
            print('Adjusting to compensate')
            forecast_size = len(exog) - 2
            end = len(df_series) + forecast_size + 1

    # Predictions
    if exog is not None:
        t_series_predicted = model.predict(start + 1, end, dynamic=dynamic, exog=exog.iloc[:end - start + 1, :]).rename(
            "Predictions")
    else:
        t_series_predicted = model.predict(start + 1, end, dynamic=dynamic, exog=None).rename("Predictions")

    t_series_fit = t_series_fit.append(t_series_predicted)
    df_series_final = pd.concat([df_series, t_series_fit], axis=1)
    df_series_final = df_series_final.rename(columns={0: 'Predicted'})

    print(df_series_final)
    df_new_predictions = pd.DataFrame({'Time': pd.date_range(df_series_final.index.max(), periods=end, freq=frequency),
                                       'Predicted': df_series_final['Predicted'].shift(-1)
                                       })

    df_new_predictions = pd.concat([df_new_predictions, df_series_final[target_col_name]], axis=1)

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    sns.lineplot(x=df_new_predictions['Time'], y=df_new_predictions[target_col_name], label='Original', ax=ax)
    sns.lineplot(x=df_new_predictions['Time'][start:], y=df_new_predictions['Predicted'][start:], label='Predicted',
                 ax=ax)


def test_train_split_timeseries(df, prop_train):
    # Do a train test split
    n_total = len(df)
    n_train = int(n_total * prop_train)
    df_train = df.iloc[:n_train, :].copy()
    df_test = df.iloc[n_train:, :].copy()
    return df_train, df_test


def plot_lags(df, user_params, target_col_name):
    # If user parameters already specifies lags use it otherwise loop through with the user
    if user_params.get('lags') is not None:
        lags = str(user_params.get('lags'))

        plot_acf_pacf(df[target_col_name], int(lags))

        plt.show()
        sleep(1)
    else:
        lags = '20'

    while lags.lower() != 'c':
        if lags == 'q':
            return False

        plot_acf_pacf(df[target_col_name], int(lags))

        plt.show()
        sleep(1)

        lags = input('Enter ACF/PACF lags you want to see (c:continue,q:quit):')

    return True


def analyse_time_series(df, target_col_name, covariates=(), time_interval='D', main_title='', user_params=dict(),
                        exog=None):
    """
    Analyse a given time series using a SARIMAX approach

    Parameters
    ----------
    df : DataFrame
      Should have at least 1 column and an index that is a datetime. Other columns can also be included as covariates.
      It is recommended to set the freq attribute (i.e. 'D' for Day) of the index prior to feeding into this function
    target_col_name : str
      The name of the target/value column
    covariates : list
      A list of column names that are to be the covariates. Defaults to empty list
    time_interval : str
      The time interval to set the index to if the index of df doesn't already have a freq attribute. If the index
      already has a freq attribute, this is ignored. Defaults to 'D'.
      Choices for time_interval:
        Alias    Description
        B        business day frequency
        C        custom business day frequency
        D        calendar day frequency
        W        weekly frequency
        M        month end frequency
        SM       semi-month end frequency (15th and end of month)
        BM       business month end frequency
        CBM      custom business month end frequency
        MS       month start frequency
        SMS      semi-month start frequency (1st and 15th)
        BMS      business month start frequency
        CBMS     custom business month start frequency
        Q        quarter end frequency
        BQ       business quarter end frequency
        QS       quarter start frequency
        BQS      business quarter start frequency
        A, Y     year end frequency
        BA, BY   business year end frequency
        AS, YS   year start frequency
        BAS, BYS business year start frequency
        BH       business hour frequency
        H        hourly frequency
        T, min   minutely frequency
        S        secondly frequency
        L, ms    milliseconds
        U, us    microseconds
        N        nanoseconds
    main_title : str
      The title of the initial plot. Defaults to an empty string
    user_params : dict
      A dictionary containing the configuration. If provided, the user will not be prompted for the configuration.
      If not provided, the user will be prompted
      Example dictionary:
        user_params = {
        'p':1,
        'd':0,
        'q':1,
        'P':3,
        'D':1,
        'Q':1,
        'max_order':3,
        'min_order':2,
        'seasonal_period':20,
        'prop_train':0.8,
        'lags':30
        }
    exog : DataFrame
      If covariates are used (i.e. if the Covariates list is not empty), any forecasts will depend on those covariates.
      This is the covariates DataFrame containing future covariate data

    Returns
    -------
    model
      Returns a fitted SARIMAX model

    Examples
    --------
    # Get time series dataframe (this is an ARMA 1,1 model with seasonality with period 20)
    df = get_time_series(num_obs=600,arparams=[0.5],maparams=[0.2],this_seed=42,suptitle='Testing get time series
                        function',title='Test Time Series',title_prepend='',title_append='',ax=None,
                        seasonal_magnitude=2, seasonal_period=20,
                        trend_gradient = 0,trend_gradient_list=[], struct_change = False,
                        get_figure=False).set_index('Day')

    # Create a random covariate and add it to the target variable
    covariate = np.random.randint(0,10,size=len(df))/10
    df['Value'] = df['Value'] + covariate
    df['Covariate1'] = covariate

    # Create another random covariate and add 5 times it to the target variable
    covariate = np.random.randint(0,10,size=len(df))/10
    df['Value'] = df['Value'] + covariate*5
    df['Covariate2'] = covariate

    # Set the target column and covariate columns
    target_col_name = 'Value'
    covariate_names = ['Covariate1','Covariate2']

    n_train = 500
    n_forecast = len(df) - n_train

    # Get a dataset to train on (df) and get the covariates for the portion we want to forecast (exog). This is only because we're fabricating data. In reality, you will have a df and you will have an exog for the future
    exog = df.iloc[500:,:][covariate_names].copy()
    df = df.iloc[:500,:].copy()

    # Run a model without user_params
    model = analyse_time_series(df,target_col_name=target_col_name,covariates=covariate_names,time_interval='D',main_title = 'Main Title',exog=exog,user_params=user_params)

    # Set user params
    user_params = {
      'p':1,
      'd':0,
      'q':1,
      'P':3,
      'D':1,
      'Q':1,
      'max_order':3,
      'min_order':2,
      'seasonal_period':20,
      'prop_train':0.8,
      'lags':30
      }

    # Run a model using user_params
    model = analyse_time_series(df,target_col_name=target_col_name,covariates=covariate_names,time_interval='D',main_title = 'Main Title',exog=exog,user_params=user_params)
    """

    covariate_names = covariates

    # Check if df is of type dataframe. If it is a Series, convert it to dataframe
    if not isinstance(df, pd.DataFrame):
        if isinstance(df, pd.Series):
            df = pd.DataFrame(df)
        else:
            raise Exception('df should be a pandas DataFrame or a pandas Series')

    if df.index.freq is None:
        df.index.freq = time_interval

    if user_params.get('prop_train') is not None:
        prop_train = user_params.get('prop_train')
    else:
        user_prop_train = input(
            'Please specify the proportion of the data to use as train set in the analysis (c:continue,q:quit):')
        if user_prop_train.lower() == 'q':
            return None

        if user_prop_train.lower() not in ['', 'c']:
            prop_train = float(user_prop_train)
        else:
            prop_train = None

    # Do a train test split
    if prop_train is not None:
        df_train, df_test = test_train_split_timeseries(df, prop_train)
    else:
        print(
            'Note that prop_train is not set meaning metrics will be calculated on the full dataset and not a test set')
        df_train = df.copy()
        df_test = df.copy()

    # Look at the time series
    if prop_train is not None:
        plot_time_series_line_traintest(df_train[target_col_name], df_test[target_col_name], title=main_title)
    else:
        plot_time_series_line(df_train[target_col_name], title=main_title)

    plt.show()
    sleep(1)

    if not plot_lags(df, user_params, target_col_name):
        return None

    # Create a copy of the data to investigate with
    df_train_investigation = df_train.copy()

    # Seasonality checks
    if user_params.get('seasonal_period') is not None:
        seasonal_period = str(user_params.get('seasonal_period'))

        df_train_investigation[target_col_name] = df_train_investigation[target_col_name].diff(int(seasonal_period))
        df_train_investigation = df_train_investigation.iloc[int(seasonal_period):, :].copy()

        plot_time_series_line(df_train_investigation[target_col_name], title='After Removing Seasonality')
        plt.show()
        sleep(1)
    else:
        seasonal_period = input('Enter seasonal period (0 or 1:no seasonality,q:quit):')

        if seasonal_period == 'q':
            return None
        elif int(seasonal_period) > 1:
            df_train_investigation[target_col_name] = df_train_investigation[target_col_name].diff(int(seasonal_period))
            df_train_investigation = df_train_investigation.iloc[int(seasonal_period):, :].copy()

        plot_time_series_line(df_train_investigation[target_col_name], title='After Removing Seasonality')
        plt.show()
        sleep(1)

    if not plot_lags(df_train_investigation, user_params, target_col_name):
        return None

    # Check stationarity again
    check_stationarity(df_train_investigation[target_col_name])

    if user_params.get('d') is not None:
        if user_params.get('d') > 0:
            d = str(user_params.get('d'))

            df_train_investigation[target_col_name] = df_train_investigation[target_col_name].diff(int(d))
            df_train_investigation = df_train_investigation.iloc[int(d):, :]

            plot_time_series_line(df_train_investigation[target_col_name],
                                  title='After Differencing {} times'.format(int(d)))
            plt.show()
            sleep(1)
    else:
        d = input('Enter required differencing (0:no differencing,q:quit):')

        df_temp = df_train_investigation.copy()
        while d.lower() not in ['c', '0', '']:
            df_temp = df_train_investigation.copy()
            if d == 'q':
                return None
            elif int(d) > 0:
                df_temp[target_col_name] = df_temp[target_col_name].diff(int(d))
                df_temp = df_temp.iloc[int(d):, :]

            plot_time_series_line(df_temp[target_col_name], title='After Differencing {} times'.format(int(d)))
            plt.show()
            sleep(1)

            # Check stationarity again
            check_stationarity(df_temp[target_col_name])

            d = input('Enter required differencing (0:no differencing,q:quit):')

        df_train_investigation = df_temp.copy()

    # Check Ljung Box
    check_ljungbox(df_train_investigation[target_col_name])
    if len(user_params) == 0:
        user_continue = input('Do you want to continue (c/enter:continue,q:quit)?')
        if user_continue.lower() == 'q':
            return None

    if user_params.get('p') is not None:
        p = int(user_params.get('p'))
    else:
        p = int(input('Please specify the max PACF lag you want to try (p):'))

    if user_params.get('d') is not None:
        d = int(user_params.get('d'))
    else:
        d = int(input('Please specify the max differencing you want to try (d):'))

    if user_params.get('q') is not None:
        q = int(user_params.get('q'))
    else:
        q = int(input('Please specify the max ACF lag you want to try (q):'))

    if user_params.get('P') is not None:
        P = int(user_params.get('P'))
    else:
        P = int(input('Please specify the max seasonal PACF lag you want to try (P):'))

    if user_params.get('D') is not None:
        D = int(user_params.get('D'))
    else:
        D = int(input('Please specify the max seasonal differencing you want to try (D):'))

    if user_params.get('Q') is not None:
        Q = int(user_params.get('Q'))
    else:
        Q = int(input('Please specify the max seasonal ACF lag you want to try (Q):'))

    if user_params.get('seasonal_period') is not None:
        seasonal_period = int(user_params.get('seasonal_period'))
    else:
        seasonal_period_response = input(
            'Please specify the seasonal period (Press enter to use as specified before. Note that if this is 0 or 1, P,D and Q is ignored):')
        if seasonal_period_response != '':
            seasonal_period = int(seasonal_period_response)

    if user_params.get('max_order') is not None:
        max_order = int(user_params.get('max_order'))
    else:
        max_order = int(input('Please specify the max order to go up to:'))

    if user_params.get('min_order') is not None:
        min_order = int(user_params.get('min_order'))
    else:
        min_order = int(input('Please specify the min to start from:'))

    p_ls = list(range(int(p) + 1))
    d_ls = list(range(int(d) + 1))
    q_ls = list(range(int(q) + 1))

    if int(seasonal_period) > 1:
        P_ls = list(range(int(P) + 1))
        D_ls = list(range(int(D) + 1))
        Q_ls = list(range(int(Q) + 1))
        s_ls = [int(seasonal_period)]
    else:
        P_ls = []
        D_ls = []
        Q_ls = []
        s_ls = []

    trend_ls = ['n']
    import time
    tic = time.perf_counter()

    if len(covariate_names) > 0:
        df_results = apply_all_models(df_train[target_col_name], p_ls=p_ls, d_ls=d_ls, q_ls=q_ls, P_ls=P_ls, D_ls=D_ls,
                                      Q_ls=Q_ls, s_ls=s_ls, trend_ls=trend_ls, max_order=max_order,
                                      t_series_test=df_test[target_col_name], min_order=min_order,
                                      t_series_full=df[target_col_name], exog_train=df_train[covariate_names],
                                      exog_test=df_test[covariate_names], exog_full=df[covariate_names])
    else:
        df_results = apply_all_models(df_train[target_col_name], p_ls=p_ls, d_ls=d_ls, q_ls=q_ls, P_ls=P_ls, D_ls=D_ls,
                                      Q_ls=Q_ls, s_ls=s_ls, trend_ls=trend_ls, max_order=max_order,
                                      t_series_test=df_test[target_col_name], min_order=min_order,
                                      t_series_full=df[target_col_name], exog_train=None, exog_test=None,
                                      exog_full=None)

    toc = time.perf_counter()
    print(f"Finished in {toc - tic:0.4f} seconds")

    print(df_results.sort_values(by='AIC', ascending=True))
    if len(user_params) == 0:
        user_lb_non_sig = input('Show only Ljung-Box residuals non-signifianct (y:yes,n:no)?')
        if user_lb_non_sig.lower() in ['y', 'yes']:
            print(df_results[df_results['LBPvalue'] > 0.05].sort_values(by='AIC', ascending=True))

    happy = False

    while not happy:
        model_idx = input('Please specify the idx of the model you would like to use (q:quit):')
        if model_idx.lower() == 'q':
            return None
        else:
            p = df_results.loc[df_results['idx'] == int(model_idx), 'p'].values[0]
            d = df_results.loc[df_results['idx'] == int(model_idx), 'd'].values[0]
            q = df_results.loc[df_results['idx'] == int(model_idx), 'q'].values[0]
            P = df_results.loc[df_results['idx'] == int(model_idx), 'P'].values[0]
            D = df_results.loc[df_results['idx'] == int(model_idx), 'D'].values[0]
            Q = df_results.loc[df_results['idx'] == int(model_idx), 'Q'].values[0]
            seasonal_period = df_results.loc[df_results['idx'] == int(model_idx), 's'].values[0]

        if len(covariate_names) > 0:
            model = final_model(df[target_col_name], p=p, d=d, q=q, P=P, D=D, Q=Q, s=int(seasonal_period), trend='n',
                                exog=df[covariate_names])
            predict_test(model, df, len(df_train), dynamic=True, val2=covariate_names, target_col_name=target_col_name)
            plt.show()
            sleep(1)
            if exog is None:
                raise Exception(
                    'The model is trained using exogeneous variable (covariates) but exog was not supplied for the forecast')
            else:
                n_train = len(df)
                n_forecast = len(exog)

                forecast_time_series(model, df.iloc[:n_train, :], forecast_size=n_forecast, dynamic=True, exog=exog)
                plt.show()
            sleep(1)
        else:
            model = final_model(df[target_col_name], p=p, d=d, q=q, P=P, D=D, Q=Q, s=int(seasonal_period), trend='n',
                                exog=None)
            predict_test(model, df, len(df_train), dynamic=True, val2=None, target_col_name=target_col_name)
            plt.show()
            sleep(1)
            n_train = len(df)
            n_forecast = int(input("How many steps would you like to forecast?"))

            forecast_time_series(model, df.iloc[:n_train, :], forecast_size=n_forecast, dynamic=True, exog=None)
            plt.show()
            sleep(1)
        user_happy = input('Are you happy with this model (y:yes,n:no)?')

        if user_happy.lower() in ['y', 'yes']:
            happy = True
        else:
            happy = False

    return model
