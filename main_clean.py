"""
main_clean.py - Projekt NLP: Przewidywanie cen samochodów
Autor: [Twoje Imię]
Data: [Data]
Opis: System ML do przewidywania cen samochodów na podstawie ich cech
"""

# ========== IMPORTY ==========
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def main():
    """Główna funkcja programu"""

    print("="*70)
    print("SYSTEM PRZEWIDYWANIA CEN SAMOCHODÓW")
    print("="*70)

    # ========== 1. WCZYTANIE DANYCH ==========
    print("
1. WCZYTANIE DANYCH")
    print("-" * 40)

    sciezka = 'data/cars93_real.csv'

    if not os.path.exists(sciezka):
        print(f"Błąd: Brak pliku {sciezka}")
        print("Pobierz z: https://raw.githubusercontent.com/selva86/datasets/master/Cars93.csv")
        return

    # Wczytaj dane
    data = pd.read_csv(sciezka)
    print(f"Wczytano: {len(data)} rekordów")
    print(f"Liczba cech: {len(data.columns)}")

    # ========== 2. ANALIZA DANYCH ==========
    print("
2. ANALIZA DANYCH")
    print("-" * 40)

    # Podstawowe statystyki
    if 'Price' in data.columns:
        print(f"Analiza zmiennej docelowej (Price):")
        print(f"  Średnia: ${data['Price'].mean():.2f}")
        print(f"  Mediana: ${data['Price'].median():.2f}")
        print(f"  Min: ${data['Price'].min():.2f}")
        print(f"  Max: ${data['Price'].max():.2f}")
        print(f"  Odchylenie standardowe: ${data['Price'].std():.2f}")

    # Sprawdź brakujące dane
    missing = data.isnull().sum().sum()
    if missing > 0:
        print(f"
Uwaga: Znaleziono {missing} brakujących wartości")

    # ========== 3. PRZYGOTOWANIE DANYCH ==========
    print("
3. PRZYGOTOWANIE DANYCH")
    print("-" * 40)

    # Wybierz tylko kolumny numeryczne
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

    if 'Price' not in numeric_cols:
        print("Błąd: Brak kolumny 'Price' w danych numerycznych")
        return

    # Utwórz DataFrame z tylko numerycznymi kolumnami
    df_numeric = data[numeric_cols].copy()

    # Uzupełnij brakujące wartości
    df_numeric = df_numeric.fillna(df_numeric.mean())

    # Przygotuj dane do modelowania
    X = df_numeric.drop(columns=['Price'])
    y = df_numeric['Price']

    print(f"Liczba cech: {X.shape[1]}")
    print(f"Liczba próbek: {X.shape[0]}")

    # Normalizacja danych
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("Dane znormalizowane (StandardScaler)")

    # ========== 4. PODZIAŁ DANYCH ==========
    print("
4. PODZIAŁ DANYCH")
    print("-" * 40)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42
    )

    print(f"Zbiór treningowy: {X_train.shape[0]} próbek")
    print(f"Zbiór testowy: {X_test.shape[0]} próbek")

    # ========== 5. TRENOWANIE MODELU ==========
    print("
5. TRENOWANIE MODELU")
    print("-" * 40)

    # Model Random Forest
    print("Model: Random Forest Regressor")

    # Domyślny model
    model_default = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    model_default.fit(X_train, y_train)
    y_pred_default = model_default.predict(X_test)

    mae_default = mean_absolute_error(y_test, y_pred_default)
    r2_default = r2_score(y_test, y_pred_default)

    print(f"
Wyniki modelu domyślnego:")
    print(f"  MAE: ${mae_default:.2f}")
    print(f"  R²: {r2_default:.3f}")

    # ========== 6. FINE-TUNING ==========
    print("
6. DOSTRAJANIE PARAMETRÓW (FINE-TUNING)")
    print("-" * 40)

    try:
        # Definiuj przestrzeń parametrów
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10]
        }

        print("Przeszukiwanie siatki parametrów...")

        grid_search = GridSearchCV(
            RandomForestRegressor(random_state=42),
            param_grid,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=0
        )

        grid_search.fit(X_train, y_train)

        print(f"
Najlepsze parametry znalezione:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        # Użyj najlepszego modelu
        best_model = grid_search.best_estimator_
        y_pred_best = best_model.predict(X_test)

        mae_best = mean_absolute_error(y_test, y_pred_best)
        r2_best = r2_score(y_test, y_pred_best)

        print(f"
Wyniki po dostrojeniu:")
        print(f"  MAE: ${mae_best:.2f}")
        print(f"  R²: {r2_best:.3f}")

        if mae_best < mae_default:
            improvement = (mae_default - mae_best) / mae_default * 100
            print(f"  Poprawa: {improvement:.1f}%")

        model = best_model
        y_pred = y_pred_best
        mae_final = mae_best
        r2_final = r2_best

    except Exception as e:
        print(f"Uwaga: Fine-tuning nie powiódł się: {e}")
        print("Używam modelu domyślnego")
        model = model_default
        y_pred = y_pred_default
        mae_final = mae_default
        r2_final = r2_default

    # ========== 7. ANALIZA WAŻNOŚCI CECH ==========
    print("
7. ANALIZA WAŻNOŚCI CECH")
    print("-" * 40)

    if hasattr(model, 'feature_importances_'):
        # Pobierz ważność cech
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("Najważniejsze cechy wpływające na cenę:")
        for idx, row in feature_importance.head(5).iterrows():
            print(f"  {row['feature']}: {row['importance']:.3f}")
    else:
        print("Model nie dostarcza informacji o ważności cech")

    # ========== 8. WIZUALIZACJA ==========
    print("
8. WIZUALIZACJA WYNIKÓW")
    print("-" * 40)

    # Utwórz folder na wykresy
    os.makedirs('plots', exist_ok=True)

    # Wykres 1: Porównanie rzeczywistych i przewidywanych wartości
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)

    # Linia idealnego dopasowania
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

    plt.xlabel('Rzeczywista cena ($)')
    plt.ylabel('Przewidziana cena ($)')
    plt.title('Rzeczywiste vs Przewidziane ceny')
    plt.grid(True, alpha=0.3)

    # Wykres 2: Rozkład błędów
    plt.subplot(1, 2, 2)
    errors = y_test - y_pred
    plt.hist(errors, bins=15, edgecolor='black', alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Błąd przewidywania ($)')
    plt.ylabel('Liczba próbek')
    plt.title('Rozkład błędów przewidywania')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('plots/results_visualization.png', dpi=120, bbox_inches='tight')
    plt.show()

    print("Zapisano wykres: plots/results_visualization.png")

    # ========== 9. PRZYKŁAD PRZEWIDYWANIA ==========
    print("
9. PRZYKŁAD PRZEWIDYWANIA")
    print("-" * 40)

    if len(X_test) > 0:
        # Wybierz losową próbkę testową
        sample_idx = 0
        sample_features = X_test[sample_idx].reshape(1, -1)
        actual_price = y_test.iloc[sample_idx]
        predicted_price = model.predict(sample_features)[0]

        # Znajdź oryginalny rekord
        original_record = data.iloc[data.index[y_test.index[sample_idx]]]

        print("Przykładowy samochód:")
        if 'Manufacturer' in original_record and 'Model' in original_record:
            print(f"  {original_record['Manufacturer']} {original_record['Model']}")
        if 'Type' in original_record:
            print(f"  Typ: {original_record['Type']}")

        print(f"
Cena rzeczywista: ${actual_price:.2f}")
        print(f"Cena przewidziana: ${predicted_price:.2f}")
        print(f"Różnica: ${abs(actual_price - predicted_price):.2f}")

        error_percent = abs(actual_price - predicted_price) / actual_price * 100
        print(f"Błąd względny: {error_percent:.1f}%")

    # ========== 10. PODSUMOWANIE ==========
    print("
" + "="*70)
    print("PODSUMOWANIE")
    print("="*70)

    print(f"
Najlepszy model: Random Forest Regressor")
    print(f"Średni błąd bezwzględny (MAE): ${mae_final:.2f}")
    print(f"Współczynnik determinacji (R²): {r2_final:.3f}")

    if r2_final > 0:
        print(f"Model wyjaśnia {r2_final*100:.1f}% wariancji danych")

    print(f"
Błąd względny: {(mae_final / y.mean()) * 100:.1f}% średniej ceny")

    print("
" + "="*70)
    print("PROGRAM ZAKOŃCZONY")
    print("="*70)

if __name__ == "__main__":
    main()
