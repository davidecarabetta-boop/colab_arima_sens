import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import traceback
import sys
import holidays  

# Ignora i warning
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SERVICE_ACCOUNT_FILE = 'credentials.json'
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1rSeZ1BtU3ipbFfnTeeXFKMRsH5r2yjprSTsFUmN7aVs/edit?gid=1633708881#gid=1633708881'
INPUT_SHEET_NAME = 'sensation_dati_storici_arima'
OUTPUT_SHEET_NAME = 'Previsione_Output_SARIMAX'

FORECAST_STEPS = 30
RETRAIN_START_DATE = '2025-09-01'
OUTLIER_DATE = pd.to_datetime('2026-02-28')
ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 0, 1, 7)

def authenticate_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    client = gspread.authorize(creds)
    print("Autenticazione Google Sheets OK.")
    return client

def load_and_clean_data(client):
    print("Fase 1: Caricamento e Pulizia Dati...")
    sheet = client.open_by_url(SHEET_URL).worksheet(INPUT_SHEET_NAME)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    if df.empty:
        print("ERRORE: Il dataframe è vuoto!")
        sys.exit(1)

    # 1. Conversione Data e rimozione righe nulle
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Data'])

    # 2. Pulizia Numerica
    df['Entrate totali'] = df['Entrate totali'].astype(str).str.replace('€', '').str.strip()
    df['Entrate totali'] = df['Entrate totali'].str.replace('.', '', regex=False)
    df['Entrate totali'] = pd.to_numeric(df['Entrate totali'].str.replace(',', '.', regex=False), errors='coerce')
    df = df.dropna(subset=['Entrate totali'])

    # --- FIX CRITICO: AGGREGAZIONE DUPLICATI ---
    # Se ci sono date doppie, sommiamo le entrate. Risolve il ValueError.
    df = df.groupby('Data')['Entrate totali'].sum().reset_index()
    # -------------------------------------------

    # 3. Imposta Indice e ordina temporalmente
    df = df.set_index('Data').sort_index()

    # 4. Filtro e Trattamento Outlier
    df_filtered = df[df.index >= RETRAIN_START_DATE].copy()
    df_cleaned = df_filtered[df_filtered.index != OUTLIER_DATE].copy()
    
    endog_original = df_cleaned['Entrate totali'].copy()
    
    # Ora il reindex funzionerà perché le date sono univoche
    full_index = pd.date_range(start=endog_original.index.min(), end=endog_original.index.max(), freq='D')
    endog_continuous = endog_original.reindex(full_index)

    # Riempimento buchi (es. giorni mancanti)
    endog_final_fixed = endog_continuous.fillna(method='ffill')
    endog_final_fixed.index.freq = 'D'
    exo_data = prepare_exogenous_variables(endog_final_fixed.index)
    
    return endog_final_fixed, exo_data
    
def prepare_exogenous_variables(index):
    it_holidays = holidays.Italy()
    exo = pd.DataFrame(index=index)
    exo['is_holiday'] = exo.index.map(lambda x: 1 if x in it_holidays else 0)
    return exo

def run_sarimax_forecast(endog_series, exo_train, steps):
    
    # Definiamo le variabili esogene per il futuro (i prossimi 30 giorni)
    future_index = pd.date_range(start=endog_series.index[-1] + pd.Timedelta(days=1), periods=steps, freq='D')
    exo_forecast = prepare_exogenous_variables(future_index)

    model = SARIMAX(endog_series, 
                    exog=exo_train, 
                    order=ORDER, 
                    seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, 
                    enforce_invertibility=False)
    
    results = model.fit(disp=False)
    
    # Previsione passando le esogene future
    forecast_object = results.get_forecast(steps=steps, exog=exo_forecast) # <--- E qui
    forecast_result = forecast_object.predicted_mean / 100
    forecast_ci = forecast_object.conf_int() / 100

    forecast_df = pd.DataFrame({
        'Data': forecast_result.index.strftime('%Y-%m-%d'),
        'Previsione_media': forecast_result.round(2),
        'Limite_superiore_CI': forecast_ci.iloc[:, 1].round(2),
        'Limite_inferiore_CI': forecast_ci.iloc[:, 0].round(2),
        'Is_Holiday': exo_forecast['is_holiday'].values # Opzionale: vedi se è festivo nel log
    })
    return forecast_df

# # --- Modifica nel Main ---
# if __name__ == '__main__':
#     try:
#         client = authenticate_google_sheets()
#         # Ora la funzione restituisce due oggetti
#         endog_data, exo_data = load_and_clean_data(client) 
#         # Passiamo exo_data al forecast
#         forecast_output_df = run_sarimax_forecast(endog_data, exo_data, FORECAST_STEPS)
#         push_to_google_sheets(client, forecast_output_df)
#         print("\n*** Pipeline con Esogene completata! ***")
#     except Exception as e:
#         # ... (gestione errore)
        
# def run_sarimax_forecast(endog_series, steps):
#     print("Fase 2: Addestramento Modello...")
#     model = SARIMAX(endog_series, order=ORDER, seasonal_order=SEASONAL_ORDER,
#                     enforce_stationarity=False, enforce_invertibility=False)
#     results = model.fit(disp=False)
    
#     forecast_object = results.get_forecast(steps=steps)
#     forecast_result = forecast_object.predicted_mean / 100
#     forecast_ci = forecast_object.conf_int() / 100

#     forecast_df = pd.DataFrame({
#         'Data': forecast_result.index.strftime('%Y-%m-%d'),
#         'Previsione_media': forecast_result.round(2),
#         'Limite_superiore_CI': forecast_ci.iloc[:, 1].round(2),
#         'Limite_inferiore_CI': forecast_ci.iloc[:, 0].round(2)
#     })
#     return forecast_df

def push_to_google_sheets(client, df_forecast):
    print(f"Fase 3: Salvataggio su {OUTPUT_SHEET_NAME}...")
    sheet = client.open_by_url(SHEET_URL)
    try:
        worksheet = sheet.worksheet(OUTPUT_SHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=OUTPUT_SHEET_NAME, rows=100, cols=10)

    data_to_write = [df_forecast.columns.values.tolist()] + df_forecast.values.tolist()
    worksheet.clear()
    worksheet.update('A1', data_to_write)
    print("Update completato.")

if __name__ == '__main__':
    try:
        client = authenticate_google_sheets()
        endog_data, exo_data = load_and_clean_data(client)
        forecast_output_df = run_sarimax_forecast(endog_data, exo_data, FORECAST_STEPS)
        push_to_google_sheets(client, forecast_output_df)
        print("\n*** Pipeline completata con successo! ***")
    except Exception as e:
        print(f"\nERRORE CRITICO: {e}")
        traceback.print_exc()
        sys.exit(1)
