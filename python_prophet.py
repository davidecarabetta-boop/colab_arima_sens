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
INPUT_SHEET_NAME = 'Sensation_dati_storici_reali'
OUTPUT_SHEET_NAME = 'Previsione_Output_Prophet_w'
OUTPUT_SHEET_NAME_WEEK = 'Previsione_Output_Prophet_week'


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

# def load_and_clean_data(client):
#     print("Fase 1: Caricamento e Pulizia Dati...")
#     sheet = client.open_by_url(SHEET_URL).worksheet(INPUT_SHEET_NAME)
#     data = sheet.get_all_records()
#     df = pd.DataFrame(data)

#     if df.empty:
#         print("ERRORE: Il dataframe è vuoto!")
#         sys.exit(1)

#     # 1. Conversione Data e rimozione righe nulle
#     df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
#     df = df.dropna(subset=['Data'])

#     # 2. Pulizia Numerica
#     df['Entrate totali'] = df['Entrate totali'].astype(str).str.replace('€', '').str.strip()
#     df['Entrate totali'] = df['Entrate totali'].str.replace('.', '', regex=False)
#     df['Entrate totali'] = pd.to_numeric(df['Entrate totali'].str.replace(',', '.', regex=False), errors='coerce')
#     df = df.dropna(subset=['Entrate totali'])

#     # --- FIX CRITICO: AGGREGAZIONE DUPLICATI ---
#     # Se ci sono date doppie, sommiamo le entrate. Risolve il ValueError.
#     df = df.groupby('Data')['Entrate totali'].sum().reset_index()
#     # -------------------------------------------

#     # 3. Imposta Indice e ordina temporalmente
#     df = df.set_index('Data').sort_index()

#     # 4. Filtro e Trattamento Outlier
#     df_filtered = df[df.index >= RETRAIN_START_DATE].copy()
#     df_cleaned = df_filtered[df_filtered.index != OUTLIER_DATE].copy()
    
#     endog_original = df_cleaned['Entrate totali'].copy()
    
#     # Ora il reindex funzionerà perché le date sono univoche
#     full_index = pd.date_range(start=endog_original.index.min(), end=endog_original.index.max(), freq='D')
#     endog_continuous = endog_original.reindex(full_index)

#     # Riempimento buchi (es. giorni mancanti)
#     endog_final_fixed = endog_continuous.fillna(method='ffill')
#     endog_final_fixed.index.freq = 'D'
    
#     return endog_final_fixed

def load_and_clean_data(client):
    sheet = client.open_by_url(SHEET_URL).worksheet(INPUT_SHEET_NAME)
    df = pd.DataFrame(sheet.get_all_records()).copy()

    # Pulizia Date
    df['ds'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce') 
    df = df.dropna(subset=['ds']) 
    soglia = pd.Timestamp('2025-11-20')

    # 3. Aggiungi 1 giorno SOLO se la data è maggiore del 20/11/2025
    # Se la condizione è vera aggiunge 1 giorno, altrimenti lascia la data originale
    df['ds'] = np.where(
        df['ds'] > soglia, 
        df['ds'] + pd.Timedelta(days=1), 
        df['ds']
    )

    # Pulizia Valuta
    df['y'] = df['Entrate reali'].astype(str).str.replace('€', '').str.replace('.', '', regex=False)
    df['y'] = pd.to_numeric(df['y'].str.replace(',', '.', regex=False), errors='coerce')    
    df = df.dropna(subset=['y'])

    # Aggregazione duplicati (fondamentale per Prophet)
    df = df.groupby('ds')['y'].sum().reset_index()
    df.loc[df['ds'].dt.date >= (pd.Timestamp.now().date() - pd.Timedelta(days=2)), 'y'] = np.nan
    
    # Filtro Outlier (es. valori negativi o errori macroscopici nel database)
    df = df[df['y'] >= 0]

    return df

def get_complete_gift_holidays():
    years = [2024, 2025, 2026]
    holidays_list = []

    for year in years:
        # 1. NATALE 
        holidays_list.append({
            'holiday': 'regali_natale',
            'ds': f'{year}-12-18',
            'lower_window': -25, 'upper_window': 0,
            'prior_scale': 25
        })
        
        # 11 AGOSTO San Lorenzo
        holidays_list.append({
            'holiday': 'san_lorenzo',
            'ds': f'{year}-08-11',
            'lower_window': -5, 'upper_window': 0,
            'prior_scale': 18
        })
        
        # 2. SAN VALENTINO 
        holidays_list.append({
            'holiday': 'san_valentino',
            'ds': f'{year}-02-14',
            'lower_window': -10, 'upper_window': 0,
            'prior_scale': 12
        })

        # 3. FESTA DELLA MAMMA 
        may_days = pd.date_range(start=f'{year}-05-01', end=f'{year}-05-14')
        mamma_date = may_days[may_days.weekday == 6][1] 
        holidays_list.append({
            'holiday': 'festa_mamma',
            'ds': mamma_date,
            'lower_window': -10, 'upper_window': 0,
            'prior_scale': 4
        })

        # 4. BLACK FRIDAY (Quarto venerdì di Novembre)
        nov_days = pd.date_range(start=f'{year}-11-01', end=f'{year}-11-30')
        black_friday = nov_days[nov_days.weekday == 4][3]
        holidays_list.append({
            'holiday': 'black_friday_week',
            'ds': black_friday,
            'lower_window': -4, 
            'upper_window': 3,
            'prior_scale': 15
        })

    return pd.DataFrame(holidays_list)
    
def run_prophet_forecast(df, steps):
    print("Inizio Addestramento Prophet...")

    gift_holidays = get_complete_gift_holidays()

    model = Prophet(
        holidays=gift_holidays,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False, 
        changepoint_prior_scale=0.04,
        interval_width=0.8,
        seasonality_mode='multiplicative'
    )
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_country_holidays(country_name='IT')
    model.fit(df)

    future = model.make_future_dataframe(periods=steps)
    forecast = model.predict(future)

    divisor = 100
    
    # --- CALCOLO COMPONENTI (Valore Assoluto) ---
    # Moltiplichiamo il coefficiente stagionale per il trend per ottenere il valore in valuta
    forecast['trend_val'] = forecast['trend'] / divisor
    forecast['impact_yearly'] = (forecast['yearly'] * forecast['trend']) / divisor
    forecast['impact_monthly'] = (forecast['monthly'] * forecast['trend']) / divisor
    forecast['impact_holidays'] = (forecast['holidays'] * forecast['trend']) / divisor

    # --- LOGICA DI OUTPUT ---
    df_output = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper', 'trend_val', 
                          'impact_yearly', 'impact_monthly', 'impact_holidays']].copy()
    df_output = df_output.merge(df[['ds', 'y']], on='ds', how='left')
    df_output['week'] = df_output['ds'].dt.to_period('W').apply(lambda r: r.start_time)

    # Aggregazione settimanale (Sommiamo gli impatti giornalieri per avere il totale settimanale)
    df_weekly = df_output.groupby('week').agg({
        'yhat': 'sum',
        'yhat_lower': 'sum',
        'yhat_upper': 'sum',
        'y': 'sum',
        'trend_val': 'sum',
        'impact_yearly': 'sum',
        'impact_monthly': 'sum',
        'impact_holidays': 'sum',
    }).reset_index()

    df_output['is_real'] = pd.notnull(df_output['y'])
    df_weekly['is_real'] = (pd.notnull(df_weekly['y'])) & (df_weekly['y'] > 0)
    df_weekly.loc[~df_weekly['is_real'], 'y'] = np.nan

    # --- CREAZIONE FINAL_DF (Giornaliero) ---
    final_df = pd.DataFrame({
        'Data': df_output['ds'].dt.strftime('%Y-%m-%d'),
        'Tipo': df_output['is_real'].map({True: 'REALE', False: 'PREVISIONE'}),
        'Previsione': pd.to_numeric(df_output['yhat'] / divisor).round(2),
        'Dato_reale': pd.to_numeric(df_output['y'] / divisor, errors='coerce').round(2),
        'CI_Superiore': pd.to_numeric(df_output['yhat_upper'] / divisor).round(2),
        'CI_Inferiore': pd.to_numeric(df_output['yhat_lower'] / divisor).round(2),
        'Trend_Base': df_output['trend_val'].round(2),
        'Effetto_Annuale': df_output['impact_yearly'].round(2),
        'Effetto_Mensile': df_output['impact_monthly'].round(2),
        'Effetto_Festivita': df_output['impact_holidays'].round(2)
    })

    # --- CREAZIONE FINAL_DF_WEEKLY (Settimanale) ---
    final_df_weekly = pd.DataFrame({
        'Data': df_weekly['week'].dt.strftime('%Y-%m-%d'),
        'Tipo': df_weekly['is_real'].map({True: 'REALE', False: 'PREVISIONE'}),
        'Previsione': pd.to_numeric(df_weekly['yhat'] / divisor).round(2),
        'Dato_reale': pd.to_numeric(df_weekly['y'] / divisor, errors='coerce').round(2),
        'CI_Superiore': pd.to_numeric(df_weekly['yhat_upper'] / divisor).round(2),
        'CI_Inferiore': pd.to_numeric(df_weekly['yhat_lower'] / divisor).round(2),
        'Trend_Base': df_weekly['trend_val'].round(2),
        'Effetto_Annuale': df_weekly['impact_yearly'].round(2),
        'Effetto_Mensile': df_weekly['impact_monthly'].round(2),
        'Effetto_Festivita': df_weekly['impact_holidays'].round(2)
    })
    
    # Pulizia finale (NaN -> "")
    final_df = final_df.replace([np.inf, -np.inf], np.nan).fillna("")
    final_df_weekly = final_df_weekly.replace([np.inf, -np.inf], np.nan).fillna("")

    return final_df, final_df_weekly

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

def push_to_google_sheets(client, df_forecast, df_forecast_week):
    print(f"Fase 3: Salvataggio su {OUTPUT_SHEET_NAME}...")
    print(f"Fase 3: Salvataggio settimanale su {OUTPUT_SHEET_NAME_WEEK}...")
    
    sheet = client.open_by_url(SHEET_URL)
    
    try:
        worksheet = sheet.worksheet(OUTPUT_SHEET_NAME)
    except gspread.WorksheetNotFound:
        worksheet = sheet.add_worksheet(title=OUTPUT_SHEET_NAME, rows=1000, cols=15)

    try:
        worksheet_week = sheet.worksheet(OUTPUT_SHEET_NAME_WEEK)
    except gspread.WorksheetNotFound:
        worksheet_week = sheet.add_worksheet(title=OUTPUT_SHEET_NAME_WEEK, rows=1000, cols=15)

    data_to_write = [df_forecast.columns.values.tolist()] + df_forecast.values.tolist()
    data_to_write_week = [df_forecast_week.columns.values.tolist()] + df_forecast_week.values.tolist()

    worksheet.clear()
    worksheet.update('A1', data_to_write)
    worksheet_week.clear()
    worksheet_week.update('A1', data_to_write_week)
    print("Update completato.")

# if __name__ == '__main__':
#     try:
#         client = authenticate_google_sheets()
#         endog_data = load_and_clean_data(client)
#         forecast_output_df = run_sarimax_forecast(endog_data, FORECAST_STEPS)
#         push_to_google_sheets(client, forecast_output_df)
#         print("\n*** Pipeline completata con successo! ***")
#     except Exception as e:
#         print(f"\nERRORE CRITICO: {e}")
#         traceback.print_exc()
#         sys.exit(1)

if __name__ == '__main__':
    try:
        client = authenticate_google_sheets()
        clean_df = load_and_clean_data(client)
        forecast_df, forecast_df_week = run_prophet_forecast(clean_df, FORECAST_STEPS)
        push_to_google_sheets(client, forecast_df, forecast_df_week)
    except Exception as e:
        print(f"ERRORE: {e}")
        sys.exit(1)
