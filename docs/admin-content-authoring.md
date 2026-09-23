# Упрощённый multilingual authoring

Telegram Admin всегда остаётся русским. У каждого bot/database есть независимый флаг `BotGeneralConfig.multilingual_authoring_enabled`.

- `ВЫКЛ` — обычные формы работают как в русскоязычном админ-панеле до multilingual authoring. Locale tabs, статусы переводов и глобальный выбор языка не показываются.
- `ВКЛ` — языки выбираются только внутри карточки поддерживаемого объекта. Выбор не сохраняется как глобальная настройка администратора и не влияет на язык пользователя, язык по умолчанию или список доступных языков.

## Матрица поддерживаемых объектов

| Объект | Локализуемые поля | Общие поля | Карточка при ВКЛ |
|---|---|---|---|
| Topic | `name`, `description`, `start_message`, `start_button_text` | ID, порядок, видимость, payload, routing, prompts, связи | Да |
| Content | `button_title`, `text_content`, `action_btn_text`, локальный список media | key, visibility, order, action payload | Да |
| SubscriptionPlan | `name`, `description` | цена, срок, provider, permissions, flags | Да |
| SubscriptionConfig | пользовательские labels меню | платежи, бонусы, flags, URLs | Да |
| ReferralTemplate | пользовательский `text` | ID, порядок, enabled, начисления | Да |
| MediaLibrary | пользовательский `description`/caption | file ID, имя для AI, категория, media type, collections | Да |

TestQuestion, answer options, automations, followups, prompts, knowledge base, campaigns, admin messages и внутренние настройки не получают новые locale tabs. Их существующие русские формы не меняются.

## Хранение

Canonical RU остаётся в существующих полях бизнес-объектов. EN/PT variants используют уже существующий `BotTranslation` с ключом `resource_type.stable_id.field`; новые колонки `name_en`/`name_pt` не создаются. Системные packs по-прежнему работают только с system registry, а dynamic keys не экспортируются и не перезаписываются импортом.

Для Content локальный media override хранится тем же ключом в JSON-варианте `content.<key>.media`. Если варианта нет, runtime использует общий/RU media. В текущей модели MediaLibrary file identity остаётся общей, а caption/description локализуется; отдельный файл для языка не создаётся без соответствующей продуктовой семантики.

Новый объект создаётся обычным RU-first способом. Затем в той же карточке можно сохранить EN/PT. Переводы не создают копии и не меняют stable ID.

## Runtime fallback

Для user locale используется вариант этого locale, затем реальный RU variant. Если обязательного display value нет ни там, ни там, материал не показывается в списке. Отсутствующий вариант в карточке явно обозначается как `⚠️ Перевод не задан` с русским reference; reference никогда автоматически не сохраняется как перевод.

Недостаток dynamic content не меняет system readiness EN/PT и не отключает язык целиком. `🌐 Языки` управляет user-language availability, selector, system readiness и system pack import/export. В пояснении экрана указано, что темы, контент, тарифы, реферальные сообщения и media переводятся в своих карточках.

## Совместимость

`admin_content_preferences` и старые per-admin locale helpers оставлены для безопасного rollback, но больше не используются обычным runtime flow. Existing EN/PT `BotTranslation` rows читаются без копирования и удаления. Existing user preferences, default locale, enabled locales, system pack data и provider settings не мигрируются.

Нормальный путь админа:

1. `🌐 Языки` → `🌐 Мультиязычность: ВКЛ`.
2. `Темы` → открыть Topic ID.
3. В карточке выбрать `🇬🇧 English` или `🇵🇹 Português`.
4. Изменить только нужные display fields и сохранить.
5. Переключение на другую карточку снова начинается с RU, без глобального locale state.
6. `🌐 Языки` использовать только для пользовательских языков и системных packs.
