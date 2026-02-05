"""
main2.py - POPRAWIONA wersja - używa cena_pln
"""

print("="*60)
print("PROJEKT NLP - POPRAWIONA WERSJA")
print("="*60)

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ========== 1. WCZYTANIE ==========
print("\n1. 📥 WCZYTUJĘ DANE...")
sciezka = "data/samochody.csv"

if not os.path.exists(sciezka):
    print("❌ Brak danych!")
    exit()

dane = pd.read_csv(sciezka)
print(f"✅ Wczytano: {len(dane)} samochodów")
print(f"Kolumny: {list(dane.columns)}")

# SPRAWDŹ CZY JEST cena_pln
if 'cena_pln' not in dane.columns:
    print("❌ UWAGA: Brak 'cena_pln' w danych!")
    print("Dostępne kolumny:", list(dane.columns))
    # Spróbuj znaleźć kolumnę z ceną
    for col in dane.columns:
        if 'cena' in col.lower() or 'price' in col.lower():
            print(f"Może chodzi o kolumnę: '{col}'?")
    exit()

# ========== 2. ANALIZA ==========
print("\n2. 📊 ANALIZA...")
print(f"Średnia cena: {dane['cena_pln'].mean():,.0f} zł")

# ========== 3. PRZETWARZANIE ==========
print("\n3. 🔧 PRZETWARZANIE...")

# Uzupełnij braki
dane = dane.fillna(dane.mean(numeric_only=True))

# Normalizacja
num_cols = dane.select_dtypes(include=[np.number]).columns.tolist()
if 'cena_pln' in num_cols:
    num_cols.remove('cena_pln')

scaler = StandardScaler()
dane[num_cols] = scaler.fit_transform(dane[num_cols])
print(f"Znormalizowano: {num_cols}")

# Kodowanie kategoryczne
for col in dane.select_dtypes(include=['object']).columns:
    dummies = pd.get_dummies(dane[col], prefix=col, drop_first=True)
    dane = pd.concat([dane, dummies], axis=1)
    dane.drop(columns=[col], inplace=True)
    print(f"Zakodowano: {col}")

# ========== 4. MODELOWANIE ==========
print("\n4. 🤖 MODELOWANIE...")

X = dane.drop(columns=['cena_pln'])
y = dane['cena_pln']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model 1
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)

# Model 2
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print(f"\nRegresja liniowa: {mae_lr:,.0f} zł")
print(f"Las losowy: {mae_rf:,.0f} zł")

# ========== 5. WYNIKI ==========
print("\n5. 📈 WYNIKI...")
if mae_rf < mae_lr:
    print("🏆 NAJLEPSZY: Las losowy")
else:
    print("🏆 NAJLEPSZY: Regresja liniowa")

print("\n" + "="*60)
print("✅ PROGRAM ZAKOŃCZONY!")
print("="*60)
