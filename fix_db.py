import asyncio
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy import text

# Список баз данных
DATABASES = [
    "postgresql+asyncpg://bot_user:asdasd2@localhost/psy5d2_db",
    "postgresql+asyncpg://bot_user:asdasd2@localhost/veraveda2_db",
    "postgresql+asyncpg://bot_user:asdasd2@localhost/someonewithyou01_db",
    "postgresql+asyncpg://bot_user:asdasd2@localhost/someonewithyou02_db",
    "postgresql+asyncpg://bot_user:asdasd2@localhost/veraveda_db",
    "postgresql+asyncpg://my_bot_user:asdasd2@localhost/someone02",
    "postgresql+asyncpg://my_bot_user:asdasd2@localhost/someone01",
    "postgresql+asyncpg://my_bot_user:asdasd2@localhost/psy5d_db",
    "postgresql+asyncpg://my_bot_user:asdasd2@localhost/test01_db",
    "postgresql+asyncpg://my_bot_user:asdasd2@localhost/test02_db",
    "postgresql+asyncpg://bot_user:asdasd2@localhost/someonewithyou03_db"
]

DEFAULT_TEST_PROMPT = "Ты — психолог Алёна Верловицкая. Действуй строго по разделу 'ЗАДАЧА 1: СЦЕНАРИСТ'. Твоя цель: написать историю персонажа-двойника. Не показывай цифры. Только история."


async def fix_all():
    print("🚀 Начинаю обновление всех баз данных...")

    for db_url in DATABASES:
        db_name = db_url.split('/')[-1]
        print(f"🛠  Обработка: {db_name}...")

        try:
            engine = create_async_engine(db_url)
            async with engine.begin() as conn:

                # 1. Создаем таблицу настроек теста (TestConfig), если её нет
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS test_config (
                        id SERIAL PRIMARY KEY,
                        is_enabled BOOLEAN DEFAULT TRUE,
                        admin_username VARCHAR DEFAULT 'AlenaVV2004',
                        marathon_url VARCHAR DEFAULT 'https://t.me/psihogipno'
                    );
                """))

                # 2. ВАЖНО: Используем $$ для экранирования текста с кавычками
                await conn.execute(text(f"""
                    ALTER TABLE test_config 
                    ADD COLUMN IF NOT EXISTS test_system_prompt TEXT DEFAULT $${DEFAULT_TEST_PROMPT}$$;
                """))

                # 3. Заполняем дефолтные настройки, если запись с id=1 не существует
                # Здесь используем параметр :prompt, это безопасно само по себе
                await conn.execute(text("""
                    INSERT INTO test_config (id, is_enabled, admin_username, marathon_url, test_system_prompt)
                    SELECT 1, TRUE, 'AlenaVV2004', 'https://t.me/psihogipno', :prompt
                    WHERE NOT EXISTS (SELECT 1 FROM test_config WHERE id = 1);
                """), {"prompt": DEFAULT_TEST_PROMPT})

                # 4. Обновляем существующие записи, где поле NULL
                await conn.execute(text("""
                    UPDATE test_config SET test_system_prompt = :prompt WHERE test_system_prompt IS NULL;
                """), {"prompt": DEFAULT_TEST_PROMPT})

                # --- Остальные таблицы и поля ---

                # Таблица секретных вопросов
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS secret_test_questions (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        sort_order INTEGER DEFAULT 0
                    );
                """))

                # Таблица кейсов
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS case_studies (
                        id SERIAL PRIMARY KEY,
                        text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                """))

                # Поле age для пользователей
                await conn.execute(text("ALTER TABLE users ADD COLUMN IF NOT EXISTS age VARCHAR;"))

                # Конфиг подписок
                await conn.execute(text("""
                    CREATE TABLE IF NOT EXISTS subscription_config (
                        id SERIAL PRIMARY KEY,
                        subscriptions_enabled BOOLEAN DEFAULT TRUE,
                        topics_enabled BOOLEAN DEFAULT TRUE,
                        notifications_enabled BOOLEAN DEFAULT TRUE,
                        test_button_enabled BOOLEAN DEFAULT TRUE
                    );
                """))

                # Поле кнопки теста
                await conn.execute(text(
                    "ALTER TABLE subscription_config ADD COLUMN IF NOT EXISTS test_button_enabled BOOLEAN DEFAULT TRUE;"
                ))

            await engine.dispose()
            print(f"✅ {db_name}: Обновлена успешно.")

        except Exception as e:
            print(f"❌ {db_name}: Ошибка - {e}")

    print("🏁 Все базы данных обработаны.")


if __name__ == "__main__":
    asyncio.run(fix_all())