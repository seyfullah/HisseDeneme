import pandas as pd
from sklearn.linear_model import LinearRegression

# Örnek veri seti
data = pd.DataFrame({
    'Tarih': ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'Fiyat': [100, 110, 120, 130, 140]
})

# Özelliklerin ve hedefin ayrıştırılması
X = pd.to_numeric(data['Tarih'].str.replace('-', ''), errors='coerce').values.reshape(-1, 1)
y = data['Fiyat'].values

# Lineer regresyon modelinin oluşturulması ve eğitilmesi
model = LinearRegression()
model.fit(X, y)

# Tahmin yapılacak tarihler
tahmin_tarihleri = pd.to_numeric(['2020-06-01', '2020-07-01', '2020-08-01'], errors='coerce').reshape(-1, 1)

# Tahminlerin yapılması
tahminler = model.predict(tahmin_tarihleri)

print(tahminler)
