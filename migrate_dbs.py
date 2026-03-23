import asyncio
import asyncpg
import re
import os

CONFIG_FILE = 'ecosystem.config.js'

async def update_database(db_url):
    clean_url = db_url.replace('postgresql+asyncpg://', 'postgresql://')
    db_name = clean_url.split('/')[-1]

    print(f"🔄 Подключение к {db_name}...")

    conn = None
    try:
        conn = await asyncpg.connect(clean_url)

        # 1. Сначала убедимся, что таблица вообще есть
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS random_messages (
                id SERIAL PRIMARY KEY,
                content TEXT NOT NULL
            );
        """)

        # 2. Добавляем недостающие колонки (если их нет)
        try:
            await conn.execute("""
                ALTER TABLE random_messages 
                ADD COLUMN IF NOT EXISTS topic_id INTEGER,
                ADD COLUMN IF NOT EXISTS category VARCHAR DEFAULT 'default';
            """)
            print(f"✅ {db_name}: Колонки topic_id и category добавлены (или уже были).")
        except Exception as e:
            print(f"⚠️ {db_name}: Ошибка при добавлении колонок: {e}")

    except Exception as e:
        print(f"❌ {db_name}: Ошибка подключения: {e}")
    finally:
        if conn:
            await conn.close()

async def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"Файл {CONFIG_FILE} не найден!")
        return

    with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
        content = f.read()

    urls = re.findall(r'"DATABASE_URL":\s*"([^"]+)"', content)

    if not urls:
        print("Не найдено ни одной DATABASE_URL в конфиге.")
        return

    print(f"Найдено баз данных: {len(urls)}")
    print("-" * 30)

    for url in urls:
        await update_database(url)

    print("-" * 30)
    print("Готово. Теперь попробуй загрузить файл снова.")

if __name__ == "__main__":
    asyncio.run(main())