# MAX Bot

Изолированная реализация бота под `MAX`, не меняющая основной Telegram-код.

## Что уже вынесено

- отдельная точка входа: `python -m max_messenger_bot.app`
- конфиг `MAX` через env или `config.ini [max]`
- webhook/polling клиент для `platform-api.max.ru`
- собственное хранилище состояний: `max_bot_states`
- отдельные таблицы под MAX-медиа:
  - `max_content_media`
  - `max_topic_media`
- пользовательские блоки:
  - `/start`, `/help`
  - главное меню
  - темы диалога
  - настройки
  - подписки, тарифы, промокоды
  - реферальный экран
  - базовый сценарий теста
- базовая админ-панель MAX

## Запуск

Пример через polling:

```bash
export MAX_BOT_TOKEN="..."
export MAX_USE_POLLING=1
python -m max_messenger_bot.app
```

Пример через webhook:

```bash
export MAX_BOT_TOKEN="..."
export MAX_WEBHOOK_BASE_URL="https://your-domain.com"
export MAX_WEBHOOK_SECRET="..."
python -m max_messenger_bot.app
```

## Важное ограничение по медиа

Старые `file_id` из Telegram нельзя использовать в MAX.

Поэтому:

- тексты, кнопки, темы, планы, промокоды и остальная логика переиспользуются из общей БД
- медиа для контента и AI-выдачи нужно заново загрузить в MAX и хранить уже в новых таблицах `max_content_media` и `max_topic_media`

Это техническое ограничение платформ, а не структуры проекта.
