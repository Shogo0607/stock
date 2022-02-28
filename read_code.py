import requests
url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls"
r = requests.get(url)
with open('data_j.xls', 'wb') as output:
    output.write(r.content)

import pandas as pd
stocklist = pd.read_excel("./data_j.xls")
stock = stocklist.loc[stocklist["市場・商品区分"]=="市場第一部（内国株）",
              ["コード","銘柄名","33業種コード","33業種区分","規模コード","規模区分"]
             ]
stock.to_csv("meigara.csv",index=False,encoding='utf_8_sig')