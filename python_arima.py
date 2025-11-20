import gspread
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2.service_account import Credentials
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
import traceback # Importa il modulo traceback

# Ignora i warning generati da statsmodels durante l'addestramento
warnings.filterwarnings("ignore")

# --- CONFIGURAZIONE (MODIFICA QUESTI TRE VALORI ESSENZIALI) ---
SERVICE_ACCOUNT_FILE = 'credentials.json'
SHEET_URL = 'https://docs.google.com/spreadsheets/d/1rSeZ1BtU3ipbFfnTeeXFKMRsH5r2yjprSTsFUmN7aVs/edit?gid=1633708881#gid=1633708881' # <-- URL DEL FOGLIO 'sensation_arima'
INPUT_SHEET_NAME = 'sensation_dati_storici_arima' # <-- Foglio interno con i dati GA4
OUTPUT_SHEET_NAME = 'Previsione_Output_SARIMAX'

FORECAST_STEPS = 30
RETRAIN_START_DATE = '2025-06-01'
OUTLIER_DATE = pd.to_datetime('2025-10-31')
# --- PARAMETRI FINALI OTTIMIZZATI ---
ORDER = (0, 1, 1)
SEASONAL_ORDER = (0, 0, 1, 7)
# ------------------------------------

def authenticate_google_sheets():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    client = gspread.authorize(creds)

    print("Autenticazione Google Sheets OK con:", creds.service_account_email)
    return client

def load_and_clean_data(client):
    """Carica, pulisce, filtra (bias) e corregge l'indice dei dati."""
    print("Fase 1: Caricamento e Pulizia Dati...")

    # 1. Lettura dati dal foglio interno
    sheet = client.open_by_url(SHEET_URL).worksheet(INPUT_SHEET_NAME)
    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    # 2. Pulizia Iniziale e Indice
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df = df.set_index('Data').sort_index().dropna(subset=['Entrate totali'])

    # 3. Pulizia Numerica (Formato italiano/europeo)
    df['Entrate totali'] = df['Entrate totali'].astype(str).str.replace('€', '').str.strip()
    df['Entrate totali'] = df['Entrate totali'].str.replace('.', '', regex=False)
    df['Entrate totali'] = pd.to_numeric(df['Entrate totali'].str.replace(',', '.', regex=False), errors='coerce')

    # 4. Filtro di Calibrazione (Retraining Recente)
    df_filtered = df[df.index >= RETRAIN_START_DATE].copy()

    # 5. Trattamento Outlier (31/10)
    df_cleaned = df_filtered[df_filtered.index != OUTLIER_DATE].copy()
    endog_original = df_cleaned['Entrate totali'].copy()

    # 6. Correzione Indice (Riempie il buco del 31/10 per la frequenza 'D')
    full_index = pd.date_range(start=endog_original.index.min(), end=endog_original.index.max(), freq='D')
    endog_continuous = endog_original.reindex(full_index)

    endog_final_fixed = endog_continuous.fillna(method='ffill')
    endog_final_fixed.index.freq = 'D' # Imposta la frequenza necessaria per SARIMAX

    return endog_final_fixed

def run_sarimax_forecast(endog_series, steps):
    """Addestra il modello SARIMAX finale, genera la previsione e CORREGGE LA SCALA."""
    print("Fase 2: Addestramento e Previsione Modello...")

    model = SARIMAX(endog_series, order=ORDER, seasonal_order=SEASONAL_ORDER,
                    enforce_stationarity=False, enforce_invertibility=False)

    results = model.fit(disp=False)

    print("Modello SARIMAX addestrato con successo.")

    # Generazione della Previsione
    forecast_object = results.get_forecast(steps=steps)
    forecast_result = forecast_object.predicted_mean
    forecast_ci = forecast_object.conf_int()

    # --- CORREZIONE DELLA SCALA (NUOVA LOGICA AGGIUNTA QUI) ---
    # Dividiamo tutti i risultati della previsione per 100 per correggere lo scaling
    # (assumendo che l'errore sia dovuto ai centesimi)
    forecast_result_corrected = forecast_result / 100
    forecast_ci_corrected = forecast_ci / 100
    # --------------------------------------------------------

    # Struttura dell'output usa i risultati corretti
    forecast_df = pd.DataFrame({
        'Data': forecast_result_corrected.index.strftime('%Y-%m-%d'),
        # Uso i risultati corretti nella previsione media
        'Previsione_media': forecast_result_corrected.round(2),
        # Uso i risultati corretti per i limiti di confidenza
        'Limite_superiore_CI': forecast_ci_corrected.iloc[:, 1].round(2),
        'Limite_inferiore_CI': forecast_ci_corrected.iloc[:, 0].round(2)
    })

    return forecast_df

def push_to_google_sheets(client, df_forecast):
    """Aggiorna la scheda di output su Fogli Google."""
    print(f"Fase 3: Salvataggio Risultati su '{OUTPUT_SHEET_NAME}'...")

    sheet = client.open_by_url(SHEET_URL)

    try:
        # Tenta di aprire una scheda esistente
        worksheet = sheet.worksheet(OUTPUT_SHEET_NAME)
    except gspread.WorksheetNotFound:
        # Se non esiste, creane una nuova
        worksheet = sheet.add_worksheet(title=OUTPUT_SHEET_NAME, rows=len(df_forecast) + 1, cols=len(df_forecast.columns))

    # Prepara i dati per l'API
    data_to_write = [df_forecast.columns.values.tolist()] + df_forecast.values.tolist()

    worksheet.clear()
    worksheet.update('A1', data_to_write)
    print(f"Previsione aggiornata con successo nella scheda '{OUTPUT_SHEET_NAME}' del foglio 'sensation_arima'.")


# --- ESECUZIONE DELLA PIPELINE ---
if __name__ == '__main__':
    try:
        client = authenticate_google_sheets()

        # 1. Caricamento e Calibrazione
        endog_data = load_and_clean_data(client)

        # 2. Esecuzione del Modello
        forecast_output_df = run_sarimax_forecast(endog_data, FORECAST_STEPS)

        # 3. Salvataggio su Fogli Google
        push_to_google_sheets(client, forecast_output_df)

        print("\n*** Pipeline SARIMAX completata con successo! Ora connetti Looker Studio! ***")

    except Exception as e:
        print(f"\nERRORE CRITICO NELLA PIPELINE: {e}")
        print("Traceback completo dell'errore:")
        print(traceback.format_exc())
        print("Verifica i percorsi, gli URL, e i permessi dell'Account di Servizio.")
