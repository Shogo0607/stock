#ライブラリのインポート
from pandas_datareader.stooq import StooqDailyReader
from datetime import datetime,date,timedelta
import streamlit as st
from pyecharts import options as opts
from pyecharts.charts import Kline,Line, Bar, Grid, Scatter
import pandas as pd
from streamlit_echarts import st_pyecharts
import numpy as np
from PIL import Image

st.set_page_config(layout="wide")

@st.cache(allow_output_mutation=True)
def Reading(stock,start,end):
    df = StooqDailyReader(stock, start=start, end=end)
    return df

def bollinger(close,span):
    mean = close.rolling(window=span).mean()
    std  = close.rolling(window=span).std()
    upper = mean + (std * 2)
    lower = mean - (std * 2)
    return upper, lower

def rousoku(day,data):
    c = (
        Kline(
            # init_opts=opts.InitOpts(
            # width="1000px",
            # height="800px",
            # animation_opts=opts.AnimationOpts(animation=False),)
            )
        .add_xaxis(day)
        .add_yaxis("日足", data)
        .set_global_opts(
            xaxis_opts=opts.AxisOpts(is_scale=True,axislabel_opts=opts.LabelOpts(rotate=15)),
            yaxis_opts=opts.AxisOpts(
                is_scale=True,
                splitarea_opts=opts.SplitAreaOpts(
                    is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=1)
                ),
            ),
            title_opts=opts.TitleOpts(title=""),
        )
    )
    return c

def line_chart(x,y,y_name):
    b = (
            Line()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(
                    title="",
                ),
                
            )
        )
    return b

def line2_chart(x,y,y_name):
    b = (
            Line()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                legend_opts=opts.LegendOpts(is_show=False),
                title_opts=opts.TitleOpts(
                    title="",
                ),
                
            )
        )
    return b

def scatter_chart(x,y,y_name):
    b = (
            Scatter()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(axislabel_opts=opts.LabelOpts(rotate=15,)),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(
                    title="",
                ),
                
            )
        )
    return b

def bar_chart(x,y,y_name):
    b = (
            Bar()
            .add_xaxis(x)
            .add_yaxis(y_name,
                 y,label_opts=opts.LabelOpts(is_show=False)
            )
            .set_global_opts(
                xaxis_opts=opts.AxisOpts(
                type_="category",
                is_scale=True,
                grid_index=1,
                boundary_gap=False,
                axisline_opts=opts.AxisLineOpts(is_on_zero=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                split_number=20,
                min_="dataMin",
                max_="dataMax",
                axislabel_opts=opts.LabelOpts(rotate=15)),
                yaxis_opts=opts.AxisOpts(
                grid_index=1,
                is_scale=True,
                split_number=2,
                axislabel_opts=opts.LabelOpts(is_show=False),
                axisline_opts=opts.AxisLineOpts(is_show=False),
                axistick_opts=opts.AxisTickOpts(is_show=False),
                splitline_opts=opts.SplitLineOpts(is_show=False),
                ),
                legend_opts=opts.LegendOpts(is_show=False),
                datazoom_opts=[opts.DataZoomOpts(), opts.DataZoomOpts(type_="inside")],
                title_opts=opts.TitleOpts(
                    title="",
                ),
                
            )
        )
    return b
# 単純移動平均線を作成
def make_sma(close, span):
    return close.rolling(window = span).mean()

# 指数平滑移動平均線を作成
def make_ema(close, span):
    sma = make_sma(close, span)[:span]
    return pd.concat([sma, close[span:]]).ewm(span = span, adjust = False).mean()

# 移動平均線の傾きを作成
def make_ma_slope(ma, span):
    ma_slope = []
    for i in range(len(ma)):
        ma_slope.append((ma[i] - ma[i - span]) / (i - (i - span)))
    return ma_slope

def rsi(close,span):
    # 前日との差分を計算
    df_diff = close.diff(1)
 
    # 計算用のDataFrameを定義
    df_up, df_down = df_diff.copy(), df_diff.copy()
    
    # df_upはマイナス値を0に変換
    # df_downはプラス値を0に変換して正負反転
    df_up[df_up < 0] = 0
    df_down[df_down > 0] = 0
    df_down = df_down * -1
    
    # 期間14でそれぞれの平均を算出
    df_up_sma14 = df_up.rolling(window=span, center=False).mean()
    df_down_sma14 = df_down.rolling(window=span, center=False).mean()
 
    # RSIを算出
    rsi = 100.0 * (df_up_sma14 / (df_up_sma14 + df_down_sma14))
 
    return rsi

st.sidebar.header("Stock Analyzer")

#今日の日付を取得
d_today = date.today()
#1年前を取得
td = timedelta(weeks=28)
d_start = d_today-td
d_end   = d_today
st.sidebar.subheader("Settings")
start = st.sidebar.date_input("開始日時",value=d_start)
end = st.sidebar.date_input("終了日時",value=d_end)

#銘柄コード
stock = st.sidebar.text_input("国内銘柄コード",value="6501")
if stock == "":
    st.warning("銘柄コードを入力してください")
    st.stop()

meigara = pd.read_csv("meigara.csv")
meigara_data = meigara[meigara["コード"] == int(stock)]
if meigara_data.empty:
    st.warning("銘柄が存在しません")
    st.stop()

name = meigara_data["コード"].values[0]

top1, top2, top3 = st.columns([1,2,2])
top1.metric("銘柄コード",str(name))
top2.metric("銘柄名", str(meigara_data["銘柄名"].values[0]))
top3.metric("業種区分", str(meigara_data["33業種区分"].values[0]))

stock = stock + ".JP"

#株価取得
df = Reading(stock,start,end)
df_stock = df.read()

open = df_stock["Open"].iloc[::-1]
close = df_stock["Close"].iloc[::-1]
high = df_stock["High"].iloc[::-1]
low = df_stock["Low"].iloc[::-1]
day = pd.Series(df_stock.index.values).apply(str).str[:10]
day = list(reversed(list(day)))
col1,col2 = st.columns([1, 4])

with col1:
    span1 = int(st.number_input("移動平均線1",value=12))
    span2 = int(st.number_input("移動平均線2",value=26))
    span3 = int(st.number_input("Bollinger Band",value=20))
    span4 = int(st.number_input("RSI",value=14))
    if span1 == 0 or span2 == 0:
        st.warning("移動平均には0より大きい整数を設定してください。")
        st.stop()
    elif span1 >= span2:
        st.warning("移動平均線1には2より小さい値を入力してください。") 
        st.stop()

with col2:
    ma1 = make_ema(close,span1)
    ma2 = make_ema(close,span2)
    
    cross = ma1 > ma2
    cross_shift = cross.shift(1)

    temp_gc = (cross != cross_shift) & (cross == True)
    temp_dc  = (cross != cross_shift) & (cross == False)
    gc = [m if g == True else np.nan for g, m in zip(temp_gc, ma1)]
    dc = [m if d == True else np.nan for d, m in zip(temp_dc, ma2)]

    ma3 = make_ema(close,50)
    ma_slope = make_ma_slope(ma3, 1)

    span = int(20)
    upper,lower = bollinger(close,span3)
    boli = make_ema(close,span3)
    rsi_data = rsi(close,span4)

    up = line_chart(day,upper,"Bollinger upper ( +2σ )")
    lower = line_chart(day,lower,"Bollinger lower ( -2σ )")
    center = line_chart(day,boli,"Bollinger center")
    rsi_chart = line2_chart(day,rsi_data,"RSI")

    data = pd.concat([open,close,low,high],axis=1)

    mean1 = list(ma1)
    mean2 = list(ma2)

    data = data.values.tolist()

    c = rousoku(day,data)
    b = line_chart(day,mean1,str(span1)+"日移動平均")
    a = line_chart(day,mean2,str(span2)+"日移動平均")
    sc = scatter_chart(day,gc,"Golden Cross")
    dc = scatter_chart(day,dc,"Dead Cross")
    d = c.overlap(b).overlap(a).overlap(sc).overlap(dc).overlap(up).overlap(lower).overlap(center)
    bar = bar_chart(day,ma_slope,"50日統計")

    # Grid Overlap + Bar
    grid_chart = Grid(
        init_opts=opts.InitOpts(
            width="1000px",
            height="1200px",
            animation_opts=opts.AnimationOpts(animation=False),
        )
    )
    grid_chart.add(
        d,
        grid_opts=opts.GridOpts(pos_left="10%", pos_right="8%", height="40%"),
    )
    grid_chart.add(
        bar,
        grid_opts=opts.GridOpts(
            pos_left="10%", pos_right="8%", pos_top="53%", height="15%"
        ),
    )
    grid_chart.add(
        rsi_chart,
        grid_opts=opts.GridOpts(
            pos_left="10%", pos_right="8%", pos_top="75%", height="15%"
        ),
    )
    st_pyecharts(grid_chart, height="700px",)
    # st_pyecharts(grid_chart, height="700px",)
