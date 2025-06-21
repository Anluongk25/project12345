import pandas as pd
import re
import streamlit as st
from datetime import datetime, timedelta
import requests
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Cài đặt hiển thị số dễ đọc ---
pd.set_option('display.float_format', '{:,.0f}'.format)

# ======================
# Đọc & xử lý dữ liệu Mỹ
# ======================
df_usa_car = pd.read_csv('./USA_car_price.csv')

usa_columns = ['country', 'Car Make', 'Car Model', 'Car Year', 'Sale Price', 'Quantity']
df_usa_clean = df_usa_car[usa_columns].copy()
df_usa_clean = df_usa_clean[df_usa_clean['Quantity'] == 1].dropna()

df_usa_clean.rename(columns={
    'Car Make': 'brand',
    'Car Model': 'grade',
    'Car Year': 'year_of_manufacture',
    'Sale Price': 'price'
}, inplace=True)

df_usa_ready = df_usa_clean.drop(columns=['Quantity'])

# ======================
# Đọc & xử lý dữ liệu Việt Nam
# ======================
df_vn_car = pd.read_csv('./Vietnamese_car_price.csv')
vn_columns = ['country', 'grade', 'year_of_manufacture', 'brand', 'price. price']
df_vn_clean = df_vn_car[vn_columns].copy()

df_vn_clean.rename(columns={'price. price': 'price'}, inplace=True)

# Hàm chuyển giá dạng text thành số
def convert_price_text_to_number(text):
    try:
        text = str(text).lower().strip()
        pattern = r'(\d+(\.\d+)?)\s*(billion|million|k)'
        matches = re.findall(pattern, text)

        total = 0
        for match in matches:
            number = float(match[0])
            unit = match[2]
            if unit == 'billion':
                total += number * 1_000_000_000
            elif unit == 'million':
                total += number * 1_000_000
            elif unit == 'k':
                total += number * 1_000

        if total == 0:
            return float(text)

        return total
    except:
        return None

df_vn_clean['price'] = df_vn_clean['price'].apply(convert_price_text_to_number)

# ======================
# Chuẩn hóa brand trước khi gộp
# ======================
brand_map = {
    'mercedes': 'mercedes benz',
    'chevy': 'chevrolet',
    'vw': 'volkswagen',
}

def clean_brand(brand_series):
    return brand_series.str.lower().replace(brand_map).str.title()

df_usa_ready['brand'] = clean_brand(df_usa_ready['brand'])
df_vn_clean['brand'] = clean_brand(df_vn_clean['brand'])

# ======================
# Gộp 2 bộ dữ liệu
# ======================
df_all = pd.concat([df_usa_ready, df_vn_clean], ignore_index=True)

# ======================
# Lấy tỷ giá USD → VND hôm qua
# ======================
@st.cache_data
def fetch_yesterday_usd_to_vnd_rate():
    try:
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        url = f"https://api.exchangerate.host/{yesterday}?base=USD&symbols=VND"
        response = requests.get(url)
        data = response.json()
        return data['rates']['VND']
    except:
        st.warning("Không lấy được tỷ giá, sử dụng mặc định 25,000 VND/USD.")
        return 25000

rate_usd_vnd = fetch_yesterday_usd_to_vnd_rate()

# Hàm chuyển đổi giá xe Mỹ sang VND sau thuế
def convert_us_price_to_vnd(usd_price):
    shipping = 2000
    import_tax = 0.70
    special_tax = 0.35
    vat = 0.10

    base = usd_price + shipping
    taxed = base * (1 + import_tax)
    with_special = taxed * (1 + special_tax)
    total_usd = with_special * (1 + vat)

    return total_usd * rate_usd_vnd

# ======================
# Chuẩn bị dữ liệu đầu ra
# ======================
df_all_processed = df_all.copy()

df_all_processed['price_vnd'] = df_all_processed.apply(
    lambda row: convert_us_price_to_vnd(row['price']) if row['country'].lower() == 'usa' else row['price'],
    axis=1
)

st.title("🚗 So sánh giá xe trung bình - Mỹ vs Việt Nam")

# ===== Lọc theo brand
tab1_brand = st.selectbox("Chọn hãng xe", sorted(df_all_processed['brand'].dropna().unique()), key="brand_1")

# ===== Lọc theo grade (dòng xe)
tab1_grades = sorted(df_all_processed[df_all_processed['brand'] == tab1_brand]['grade'].dropna().unique())
tab1_grade = st.selectbox("Chọn dòng xe (grade)", tab1_grades, key="grade_1")

# ===== Lọc theo năm sản xuất
tab1_years = sorted(df_all_processed[
    (df_all_processed['brand'] == tab1_brand) &
    (df_all_processed['grade'] == tab1_grade)
]['year_of_manufacture'].dropna().unique())
tab1_year = st.selectbox("Chọn năm sản xuất", tab1_years, key="year_1")

# ======================
# Lọc dữ liệu theo lựa chọn
# ======================
filtered_df = df_all_processed[
    (df_all_processed['brand'] == tab1_brand) &
    (df_all_processed['grade'] == tab1_grade) &
    (df_all_processed['year_of_manufacture'] == tab1_year)
]

if filtered_df.empty:
    st.warning("❗Không có dữ liệu cho lựa chọn này.")
else:
    group_by_country = filtered_df.groupby('country').agg(
        avg_usd=('price', 'mean'),
        avg_vnd=('price_vnd', 'mean'),
        count=('price', 'count')
    ).reset_index()

    for _, row in group_by_country.iterrows():
        if row['country'].lower() == 'usa':
            st.subheader("🇺🇸 USA")
            st.metric("💵 Giá gốc tại Mỹ (USD)", f"${row['avg_usd']:,.0f}")
            st.metric("📦 Giá sau thuế + vận chuyển (VND)", f"{row['avg_vnd']:,.0f}₫")
            st.caption(f"Số mẫu: {int(row['count'])}")
        elif row['country'].lower() == 'vietnam':
            st.subheader("🇻🇳 Việt Nam")
            st.metric("💰 Giá mua tại Việt Nam (VND)", f"{row['avg_vnd']:,.0f}₫")
            st.caption(f"Số mẫu: {int(row['count'])}")

st.title("🚗 So sánh giá xe trung bình theo năm (USA vs Vietnam)")

# ===== Lọc brand và grade
brand2 = st.selectbox("Chọn hãng xe", sorted(df_all['brand'].dropna().unique()), key="brand_2")
grade2 = st.selectbox("Chọn dòng xe (grade)", sorted(df_all[df_all['brand'] == brand2]['grade'].dropna().unique()), key="grade_2")
future_years = st.multiselect("📅 Chọn năm muốn dự đoán", [2026, 2027, 2028], default=[2026])

filtered_model = df_all_processed[
    (df_all_processed['brand'] == brand2) &
    (df_all_processed['grade'] == grade2)
]

if filtered_model.empty:
    st.warning("❗Không có dữ liệu cho lựa chọn này.")
else:
    avg_price_by_year = filtered_model.groupby(['country', 'year_of_manufacture']).agg(
        avg_price_vnd=('price_vnd', 'mean')
    ).reset_index()

    full_data = []
    for country in ['USA', 'Vietnam']:
        data = avg_price_by_year[avg_price_by_year['country'].str.lower() == country.lower()]
        if data.empty:
            continue
        min_year, max_year = int(data['year_of_manufacture'].min()), int(data['year_of_manufacture'].max())
        last_price = None
        for year in range(min_year, max_year + 1):
            row = data[data['year_of_manufacture'] == year]
            last_price = row['avg_price_vnd'].values[0] if not row.empty else last_price
            if last_price is not None:
                full_data.append((country, year, last_price))

    df_filled = pd.DataFrame(full_data, columns=['country', 'year', 'price'])

    predict_results = []
    for country in ['USA', 'Vietnam']:
        data = df_filled[df_filled['country'] == country]
        if len(data) < 2:
            continue
        X = data['year'].values.reshape(-1, 1)
        y = data['price'].values
        model = LinearRegression()
        model.fit(X, y)
        for next_year in future_years:
            pred_price = model.predict([[next_year]])[0]
            predict_results.append((country, next_year, pred_price))

    df_predict = pd.DataFrame(predict_results, columns=['country', 'year', 'price'])

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {'USA': 'blue', 'Vietnam': 'green'}
    pred_colors = {'USA': 'skyblue', 'Vietnam': 'lightgreen'}

    for country in ['USA', 'Vietnam']:
        actual = df_filled[df_filled['country'] == country]
        pred = df_predict[df_predict['country'] == country]
        ax.plot(actual['year'], actual['price'] / 1e6, label=f"{country} - Thực tế", color=colors[country], marker='o')
        if not pred.empty:
            ax.plot(pred['year'], pred['price'] / 1e6, label=f"{country} - Dự đoán", color=pred_colors[country], linestyle='--', marker='x')

    ax.set_title(f"📈 Giá trung bình {brand2} {grade2} qua các năm")
    ax.set_xlabel("Năm")
    ax.set_ylabel("Giá trung bình (triệu VND)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
