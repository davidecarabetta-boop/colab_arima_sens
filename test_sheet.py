import datetime
import gspread
from google.oauth2.service_account import Credentials

SERVICE_ACCOUNT_FILE = "credentials.json"
SHEET_URL = "https://docs.google.com/spreadsheets/d/1rSeZ1BtU3ipbFfnTeeXFKMRsH5r2yjprSTsFUmN7aVs/edit?gid=1633708881#gid=1633708881"
OUTPUT_SHEET_NAME = "Previsione_Output_SARIMAX"

def authenticate_google_sheets():
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
    print("🔑 Autenticato come:", creds.service_account_email)
    client = gspread.authorize(creds)
    return client

def main():
    client = authenticate_google_sheets()

    sh = client.open_by_url(SHEET_URL)
    ws = sh.worksheet(OUTPUT_SHEET_NAME)

    value = "Test GitHub Actions " + datetime.datetime.utcnow().isoformat() + " UTC"
    ws.update("A1", value)
    print("✅ Scritto in A1:", value)

if __name__ == "__main__":
    main()
