# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---

import numpy as np
import pandas as pd
from pandas import DataFrame
from datetime import datetime
from typing import Optional, Union

# additional imports required
from datetime import datetime
from freqtrade.persistence import Trade


from functools import reduce
from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter, 
                                IStrategy, IntParameter)
import freqtrade.vendor.qtpylib.indicators as qtpylib


# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import pandas_ta as pta
from technical import qtpylib


import logging
logger = logging.getLogger(__name__)




class tripleExp02(IStrategy):
    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Optimal timeframe for the strategy.
    timeframe = '1d'

    # Establecer el stop loss absoluto
    stoploss = -1
    trailing_stop = False
    use_custom_stoploss = True
    trailing_stop_positive: None
    trailing_stop_positive_offset: 0.0
    trailing_only_offset_is_reached: False

    buy_ema_fast = IntParameter(2, 50, default=3)#4
    buy_ema_slow = IntParameter(2, 80, default=8)#9

    # Optional time in force for orders
    order_time_in_force = {
        'buy': 'gtc',
        'sell': 'gtc',
    }
    @property
    def plot_config(self):
        return {
            # Main plot indicators (Moving averages, ...)
            'main_plot': {
                'emaFast': {'color':'red'},
                'emaSlow': {'color':'blue'},
            },
            # Additional plot indicators for trade entry
            'subplots': {
                'entry': {
                    'trades': {
                        '_method': 'plot_trades',
                        'symbol': 'buy',
                        'marker': 'green',
                        'markersize': 10,
                    }
                }
            }
        }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Adds several different TA indicators to the given DataFrame

        Performance Note: For the best performance be frugal on the number of indicators
        you are using. Let uncomment only the indicator you are using in your strategies
        or your hyperopt configuration, otherwise you will waste your memory and CPU usage.
        :param dataframe: Raw data from the exchange and parsed by parse_ticker_dataframe()
        :param metadata: Additional information, like the currently traded pair
        :return: a Dataframe with all mandatory indicators for the strategies
        """
        # EMA - Exponential Moving Average

        dataframe['emaFast'] = ta.EMA(dataframe, timeperiod=8) 
        dataframe['emaSlow'] = ta.EMA(dataframe, timeperiod=33) 

        #Para las 10criptos < 500m:
        #-3 37 timperiodatr14atr2
        #Para las 10criptos +marketcap:
        #3 8 timeperiod9atr2
        # Calculate all ema_short values
        # 8 33 esta bien
        for val in self.buy_ema_fast.range:
            dataframe[f'ema_short_{val}'] = ta.EMA(dataframe, timeperiod=8)#12 #4 #19 # 2 #(3 <500) #3  
        # Calculate all ema_long values

        for val in self.buy_ema_slow.range:
            dataframe[f'ema_long_{val}'] = ta.EMA(dataframe, timeperiod=33)#6 #8 #54 # 26 #(37 < 500)  #38

        # Calcular el indicador RSI
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)

        # Average true range
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        
        return dataframe

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the buy signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """
        """dataframe.loc[
            (
                #(dataframe["rsi"] < 40) &
                (dataframe['emaFast']>dataframe['emaSlow'])
            ),
            ['enter_long', 'enter_tag']] = (1, 'buyEmaFastCrossSlow')"""
        
        conditions = []
        conditions.append(
                (dataframe[f'ema_short_{self.buy_ema_fast.value}']>dataframe[f'ema_long_{self.buy_ema_slow.value}']) |
                (dataframe["emaFast"] > dataframe["emaSlow"])
            )

        # Check that volume is not 0
        conditions.append(dataframe['volume'] > 0)

        if conditions:
            dataframe.loc[
                reduce(lambda x, y: x & y, conditions),
                'buy'] = 1

        return dataframe
    

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Based on TA indicators, populates the sell signal for the given dataframe
        :param dataframe: DataFrame
        :param metadata: Additional information, like the currently traded pair
        :return: DataFrame with buy column
        """

        """dataframe.loc[
            (
                #(qtpylib.crossed_below(dataframe['emaFast'],dataframe['emaSlow']))
                #(dataframe['rsi'] > 70)
                #(dataframe['close'] < dataframe['trailing_stop'])  #(dataframe['close'] < stop_loss)&
            ),
            ['sell', 'exit_tag']] = (1, 'Salida por estrategia')"""
        
        return dataframe
    

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> Union[float, None]:
        """
        Calculates stop loss based on ATR.
        """
        # Obtener el DataFrame analizado para el par y el marco temporal actual
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        # Calcular el ATR y obtener el valor del último cierre
        #atr = ta.atr(dataframe, timeperiod=14)
        tr1 = dataframe['high'].iloc[:-1] - dataframe['low'].iloc[:-1] # con el iloc -2 mas retorno, pero no es correcto no?
        tr2 = abs(dataframe['high'].iloc[:-1] - dataframe['close'].iloc[:-1].shift())
        tr3 = abs(dataframe['low'].iloc[:-1] - dataframe['close'].iloc[:-1].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean() #16 #14 prime #9 bastante prime

        #atr = dataframe["atr"]

        stop_loss = dataframe["close"].iloc[-1] - (atr.iloc[-1] * 2) #si es 1 es mas seguro actual menos ganancias, si es 2 es menos seguro actual mas ganancias ->el 2 elejimos porque ighual fue mas seguro en años anteriores

        # Calcular el porcentaje de pérdida basado en el valor del stoploss y el precio actual
        loss_percent = (current_rate - stop_loss) / current_rate

        return loss_percent
"""        if current_rate < stop_loss:
            return loss_percent
        else:
            return 1
        
        #The absolute value of the return value is used (the sign is ignored),
        #, so returning 0.05 or -0.05 have the same result, a stoploss 5% below the current price.
"""