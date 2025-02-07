import streamlit as st
import joblib as jb
import yfinance as yf 
from datetime import datetime, timedelta
from cal_indicator import psar, cal_tech_indicators

stock_options = {
    "FPT Corporation (FPT)": "FPT.VN",
    "Vinamilk (VNM)": "VNM.VN",
    "Hoa Phat Group (HPG)": "HPG.VN",
    "Viettel Global Investment Corp (VGI)": "VGI.VN",
    "Vietcombank (VCB)": "VCB.VN",
}

try:
    model = jb.load('Linear model trained.pkl')
except Exception as e:
    st.error(f"Error when loading model: {e}")
    st.stop()
    
# try:
#     loaded_charts = jb.load("chart.pkl")
#     loaded_charts_rolling = jb.load("chart_rolling.pkl")
# except Exception as e:
#     st.error(f"Error when loading chart: {e}")
#     st.stop()

tickers = ["FPT.VN", 'HPG.VN','MWG.VN', 'VNM.VN', 'VCB.VN']
stocks = {}

start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
end_date = datetime.now().strftime("%Y-%m-%d")

for ticker in tickers:
    stock = yf.Ticker(ticker)
    key = ticker[:3]
    stocks[f"{key}"] = stock.history(start=start_date, end=end_date)

tab1, tab2= st.tabs(["📊 Chart", "💰 Prediction Close Price"])

# Tab 1 - Charts
with tab1:
    st.header("📊 Biểu đồ dữ liệu")

    # companies = list(loaded_charts.keys())
    companies = ["FPT", 'HPG','MWG', 'VNM', 'VCB']
    selected_company = st.selectbox("Chọn công ty:", companies)
    if selected_company == "FPT":
        st.image(r'Photos/FPT_price.png')
        st.image(r'Photos/FPT_rolling.png')
    elif selected_company == "HPG":
        st.image(r'Photos/HPG_price.png')
        st.image(r'Photos/HPG_rolling.png')
    elif selected_company == "MWG":
        st.image(r'Photos/MWG_price.png')
        st.image(r'Photos/MWG_rolling.png')
    elif selected_company == "VNM":
        st.image(r'Photos/VNM_price.png')
        st.image(r'Photos/VNM_rolling.png')
    else:
        st.image(r'Photos/VCB_price.png')
        st.image(r'Photos/VCB_rolling.png')

    # st.pyplot(loaded_charts[selected_company])
    # st.pyplot(loaded_charts_rolling[selected_company])

# Tab 2 - Prediction
with tab2:
    st.header("💰 Dự đoán giá")
    # calculate indicators
    for company, data in stocks.items():
        psar(data)
    cal_tech_indicators(stocks)
    
    results = {}
    features = ['Change %', 'MACD', 'RSI', 'OBV',
                'PSAR', 'SMA_10', 'SMA_50', 'EMA_50']
    for company, data in stocks.items():
        X_last = data[features].iloc[-1].values.reshape(1, -1)
        results[company] = model[company].predict(X_last)
    
    stock_name = st.selectbox("Select a stock", options=list(stock_options.keys()))
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    if st.button("Close Price Tomorrow"):
        ticker = stock_options[stock_name].split(".")[0]
        
        if ticker in results:
            st.success(f"📈 Close Price ({tomorrow}): {round(float(results[ticker]), 3)} VND")
        else:
            st.error("⚠ Error!")
