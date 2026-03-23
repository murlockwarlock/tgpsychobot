import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
if not BOT_TOKEN:
    raise ValueError("Не задана переменная окружения BOT_TOKEN")

DATABASE_URL = os.environ.get('DATABASE_URL')
if not DATABASE_URL:
    raise ValueError("Не задана переменная окружения DATABASE_URL")

try:
    OWNER_IDS = [int(x.strip()) for x in os.environ.get('OWNER_IDS', '').split(',') if x.strip()]
except Exception as e:
    print(f"Ошибка при чтении OWNER_IDS: {e}")
    OWNER_IDS = []