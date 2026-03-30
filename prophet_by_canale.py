import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
import traceback
import numpy as np
import os
import json
import sys
import holidays  

# Ignora i warning
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE ---
SERVICE_ACCOUNT_FILE = 'credentials.json'
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1rSeZ1BtU3ipbFfnTeeXFKMRsH5r2yjprSTsFUmN7aVs/edit?gid=1633708881#gid=1633708881'
INPUT_SHEET_NAME = 'sensation_dati_storici_reali_by_canale'
OUTPUT_SHEET_NAME = 'Previsione_Output_Prophet_canali'

FORECAST_STEPS = 365
RETRAIN_START_DATE = '2025-09-01'
OUTLIER_DATE = pd.to_datetime('2026-03-20')
ORDER = (1, 0, 1)
SEASONAL_ORDER = (1, 0, 0, 7)

def authenticate_google_sheets():
    scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    client = gspread.authorize(creds)
    print("Autenticazione Google Sheets OK.")
    return client

def load_and_clean_data_multi_channel(client):
    sheet = client.open_by_url(SHEET_URL).worksheet(INPUT_SHEET_NAME)
    df_raw = pd.DataFrame(sheet.get_all_records()).copy()

    # 1. Pulizia Date
    df_raw['ds'] = pd.to_datetime(df_raw['ds'], errors='coerce') 
    df_raw = df_raw.dropna(subset=['ds']) 
    
    # 2. Identificazione canali (escludiamo ds e totali se presenti)
    # Assumiamo che le colonne siano i nomi dei canali come visto nel tuo dataset precedente
    cols_to_exclude = ['ds', 'Entrate totali', 'Data']
    canali = [col for col in df_raw.columns if col not in cols_to_exclude]

    # 3. Pulizia Valori per ogni canale
    for col in canali:
        # Rimuoviamo simboli valuta e sistemiamo i separatori
        df_raw[col] = df_raw[col].astype(str).str.replace('€', '').str.replace('.', '', regex=False)
        df_raw[col] = pd.to_numeric(df_raw[col].str.replace(',', '.', regex=False), errors='coerce')
        # Riempiamo i vuoti con 0.01 (per evitare problemi con mode='multiplicative')
        df_raw[col] = df_raw[col].interpolate(method='linear').fillna(0.01)
        df_raw.loc[df_raw[col] <= 0, col] = 0.01

    # Aggregazione e filtro temporale
    df_raw = df_raw.groupby('ds')[canali].sum().reset_index()
    df_raw = df_raw[df_raw['ds'].dt.date < (pd.Timestamp.now().date() - pd.Timedelta(days=2))]

    return df_raw, canali

def get_complete_gift_holidays():
    years = [2023, 2024, 2025, 2026, 2027, 2028]
    holidays_list = []
    for year in years:
        holidays_list.append({'holiday': 'regali_natale', 'ds': f'{year}-12-18', 'lower_window': -25, 'upper_window': 0, 'prior_scale': 15})
        holidays_list.append({'holiday': 'san_lorenzo', 'ds': f'{year}-08-11', 'lower_window': -5, 'upper_window': 0, 'prior_scale': 15})
        holidays_list.append({'holiday': 'san_valentino', 'ds': f'{year}-02-14', 'lower_window': -10, 'upper_window': 0})
        
        may_days = pd.date_range(start=f'{year}-05-01', end=f'{year}-05-14')
        mamma_date = may_days[may_days.weekday == 6][1]
        holidays_list.append({'holiday': 'festa_mamma', 'ds': mamma_date, 'lower_window': -10, 'upper_window': 0})
        
        nov_days = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30')
        black_friday = nov_days[nov_days.weekday == 4][3]
        holidays_list.append({'holiday': 'black_friday_week', 'ds': black_friday, 'lower_window': -4, 'upper_window': 3})
    return pd.DataFrame(holidays_list)

def run_prophet_forecast_channel(df_single_channel, channel_name, steps, gift_holidays):
    print(f"Elaborazione Canale: {channel_name}...")

    model = Prophet(
        holidays=gift_holidays,
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False, 
        changepoint_prior_scale=0.05,
        holidays_prior_scale=10,
        interval_width=0.8,
        seasonality_mode='multiplicative'
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=6)
    model.add_country_holidays(country_name='IT')
    
    # Prepariamo il df per Prophet
    df_prophet = df_single_channel[['ds', channel_name]].rename(columns={channel_name: 'y'})
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    divisor = 100
    
    # Calcolo componenti
    forecast['trend_val'] = forecast['trend'] / divisor
    forecast['impact_weekly'] = (forecast['weekly'] * forecast['trend']) / divisor
    forecast['impact_monthly'] = (forecast['monthly'] * forecast['trend']) / divisor
    forecast['impact_holidays'] = (forecast['holidays'] * forecast['trend']) / divisor

    # Merge con dati reali
    df_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_val', 
                          'impact_weekly', 'impact_monthly', 'impact_holidays']].copy()
    df_output = df_output.merge(df_prophet, on='ds', how='left')

    # Creazione DataFrame Finale Giornaliero
    final_df = pd.DataFrame({
        'Canale': channel_name,
        'Data': df_output['ds'].dt.strftime('%Y-%m-%d'),
        'Tipo': pd.notnull(df_output['y']).map({True: 'REALE', False: 'PREVISIONE'}),
        'Previsione': (df_output['yhat'] / divisor).round(2),
        'Dato_reale': (df_output['y'] / divisor).round(2),
        'CI_Superiore': (df_output['yhat_upper'] / divisor).round(2),
        'CI_Inferiore': (df_output['yhat_lower'] / divisor).round(2),
        'Trend_Base': df_output['trend_val'].round(2),
        'Effetto_Festivita': df_output['impact_holidays'].round(2)
    })

    return final_df

# --- ESECUZIONE MAIN ---

# 1. Caricamento
df_all, lista_canali = load_and_clean_data_multi_channel(gc)
gift_holidays = get_complete_gift_holidays()

# 2. Ciclo su tutti i canali
risultati_finali = []

for canale in lista_canali:
    df_res = run_prophet_forecast_channel(df_all, canale, steps=365, gift_holidays=gift_holidays)
    risultati_finali.append(df_res)

# 3. Unione di tutti i canali in un unico grande DataFrame
df_finale_canali = pd.concat(risultati_finali, ignore_index=True)

# 4. (Opzionale) Calcolo del totale riconciliato sommando i canali
df_totale_previsto = df_finale_canali.groupby('Data').agg({
    'Previsione': 'sum',
    'Dato_reale': 'sum'
}).reset_index()


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

# --- FUNZIONE DI PUSH MODIFICATA PER MULTI-CANALE ---
def push_to_google_sheets(client, df_forecast, df_forecast_week):
    print(f"Fase 3: Salvataggio su {OUTPUT_SHEET_NAME}...")
    print(f"Fase 3: Salvataggio settimanale su {OUTPUT_SHEET_NAME_WEEK}...")
    
    sheet = client.open_by_url(SHEET_URL)
    
    # Gestione Foglio Giornaliero
    try:
        worksheet = sheet.worksheet(OUTPUT_SHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=OUTPUT_SHEET_NAME, rows=10000, cols=15)

    # Gestione Foglio Settimanale
    try:
        worksheet_week = sheet.worksheet(OUTPUT_SHEET_NAME_WEEK)
    except gspread.WorksheetNotFound:
        worksheet_week = sheet.add_worksheet(title=OUTPUT_SHEET_NAME_WEEK, rows=5000, cols=15)

    # Preparazione dati (Header + Dati)
    # Convertiamo tutto in stringa o formati compatibili con JSON per evitare errori di gspread
    data_to_write = [df_forecast.columns.values.tolist()] + df_forecast.astype(str).values.tolist()
    data_to_write_week = [df_forecast_week.columns.values.tolist()] + df_forecast_week.astype(str).values.tolist()

    # Pulizia e Upload
    worksheet.clear()
    worksheet.update('A1', data_to_write)
    
    worksheet_week.clear()
    worksheet_week.update('A1', data_to_write_week)
    
    print(f"✅ Update completato con successo su {SHEET_URL}")

# --- BLOCCO DI ESECUZIONE (MAIN) ---
if __name__ == '__main__':
    try:
        print("🚀 Avvio Pipeline Gerarchica...")
        client = authenticate_google_sheets()
        
        # 1. Caricamento dati (restituisce il DF "wide" e la lista dei canali)
        clean_df_wide, lista_canali = load_and_clean_data_multi_channel(client)
        
        all_daily_results = []
        all_weekly_results = []
        
        # 2. Loop di addestramento per ogni canale
        gift_holidays = get_complete_gift_holidays()
        
        for canale in lista_canali:
            # Esegue Prophet sul singolo canale
            # Nota: run_prophet_forecast deve essere adattata per restituire i due DF (daily, weekly)
            df_daily, df_weekly = run_prophet_forecast_channel(clean_df_wide, canale, FORECAST_STEPS, gift_holidays)
            
            all_daily_results.append(df_daily)
            all_weekly_results.append(df_weekly)
        
        # 3. Consolidamento dei risultati (uniamo tutti i canali uno sotto l'altro)
        final_forecast_df = pd.concat(all_daily_results, ignore_index=True)
        final_forecast_df_week = pd.concat(all_weekly_results, ignore_index=True)
        
        # 4. Push finale su Google Sheets
        push_to_google_sheets(client, final_forecast_df, final_forecast_df_week)
        
        print("\n*** 🎯 Pipeline completata con successo per tutti i canali! ***")

    except Exception as e:
        print(f"\n❌ ERRORE CRITICO NELLA PIPELINE: {e}")
        traceback.print_exc()
        sys.exit(1)
