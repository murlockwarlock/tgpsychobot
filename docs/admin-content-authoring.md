# Единый язык редактирования контента

Telegram Admin остаётся русским. Язык контента — отдельная настройка рабочего пространства `(bot_id, admin_id)`, по умолчанию `ru`. Она не изменяет язык пользователя, default/enabled языки бота или предпочтения другого администратора. Список языков берётся из общего каталога `SUPPORTED_TELEGRAM_LOCALES`; редакторы не содержат отдельных EN/PT-форм. Новые языки этим изменением не включаются.

## Матрица ресурсов

Все перечисленные поля уже доступны администраторам в бизнес-разделах. Колонка «Контент» означает единый selector и BotTranslation для не-RU; обычный system pack их исключает. Общие поля не локализуются.

| Ресурс | Поля | Видимость пользователю | Управление | Общие поля / исключения | Проверки |
|---|---|---|---|---|---|
| Topic | name, description, start_message, start_button_text | Да | Контент | ID, порядок, доступность, prompts, payload, привязки | Field matrix, PT-first, IDs, runtime, dispatcher |
| Content | button_title, text_content, action_btn_text | Да | Контент | key, media, порядок, visibility, action payload | Field matrix, PT-first, HTML/targets, static rendering |
| Приветствие, меню, дисклеймер | Те же поля Content | Да | Контент | Ключи start_message/menu/disclaimer | Content editor, system/dynamic split |
| Вступление/результат теста, финал секретного теста | Content.text_content и кнопки | Да | Контент | Ключи test_intro/test_results/secret_test_outro | Content matrix, test regression |
| SubscriptionPlan | name, description | Да | Контент | Цена, длительность, trial, provider, upgrade, ID | Field matrix, PT-first, plan lists, MAX |
| TestQuestion | text, comment | Да | Контент | ID, category, variable, scoring, sort, branching | Field matrix, insertion/deletion/reorder, active snapshots |
| Вариант ответа | text, button_text | Да | Контент | identity, translation_slot, callback_id, value | Insert/delete/reorder, old EN/PT aliases, scores/callbacks |
| SecretTestQuestion | text | Да | Контент | ID, sort | Field matrix, PT-first, missing availability |
| Mailing | text | Да | Контент | Аудитория, расписание, media, delivery state | Field matrix, legacy save, mailing runtime |
| AutomationAction | message_template для selected_user | Да | Контент | Получатель/ID, условия, metadata, порядок | Field matrix, legacy save, automation regression |
| Сообщения всем администраторам | message_template | Только админам | RU, не pack | Остаются русскими | Registry boundary |
| FollowupStep | message_text для static | Да | Контент | Delay, campaign, scope, AI instruction | Field matrix, legacy save, followup regression |
| ReferralTemplate | text | Да | Контент | ID, порядок, enabled, правила начисления | Field matrix, PT-first, referral regression |
| SubscriptionConfig | topics_btn_name, referral_btn_name, referral_sub_btn_name | Да | Контент | Provider config, бонусы, flags, URLs | Field matrix, legacy save, workspace isolation |
| BotGeneralConfig | ai_processing_message_text | Да | Контент | Enabled flag, customer locale settings | Field matrix, entities, legacy save |
| MediaLibrary | description | Да, caption | Контент | File IDs/names, category, media type, collections | Field matrix, media regression |
| CaseStudy | text | Да, в MAX; также материал для AI | Контент | ID, RU search index | Field matrix, MAX empty-RU filtering |
| TestConfig | System/result prompts, formulas, selected variables | Инструкции, не готовый пользовательский текст | Общие | Не переводятся | Test/scoring regression |
| RandomMessage | content | AI-контекст/метафора, не прямое сообщение | Общие | Не переводится | Audit: get_random_message_by_topic |
| KnowledgeBase, prompts, metadata, provider config | Документы/инструкции/технические значения | Не прямой display copy | Общие | Не переводятся | Existing regression |
| Фиксированные меню, ошибки, уведомления, validation copy | STATIC_TRANSLATION_SOURCES | Да | System pack | Нет обычной content-формы | Pack/registry/readiness tests |

У тарифов нет отдельного редактируемого поля benefits: пользовательские преимущества находятся в description или Content. Единственного отдельного объекта определения теста с переводимым названием нет: TestConfig хранит общие настройки, а пользовательские тексты — Content и TestQuestion.

## Хранение и создание

RU хранится в существующих canonical-полях. Не-RU upsert использует существующий `BotTranslation(locale, translation_key)` и текущий source_hash; импортированные строки читаются напрямую. Изменения увеличивают translations_revision; runtime обновляет cache без перезапуска.

Для non-RU-first создаётся один объект с обычным стабильным PK. Обязательное canonical-поле содержит пустую строку (отсутствие RU), не чужой язык и не видимый placeholder. Content получает непрозрачный машинный key. Тариф создаётся неактивным до настройки цены/длительности. PostgreSQL использует существующие sequences; SQLite harness дополнительно исключает повторное использование ID сохранённых переводов.

Additive schema: `admin_content_preferences`, `content_identity_counters`, `user_menu_bindings`; nullable `test_sessions.question_snapshot`. Языковых колонок на бизнес-таблицах нет. Bulk backfill и перезапись BotTranslation отсутствуют.

## Идентичность и совместимость

Legacy question keys уже содержат TestQuestion.id. Legacy answer slot был индексом, поэтому при первом структурном редактировании варианты получают UUID identity, неизменный translation_slot и callback_id. Старые numeric slots/значения callback сохраняются. Новый вариант получает новый slot и монотонный callback ID; удалённые IDs не перераспределяются. Сами строки переводов не копируются и не переименовываются.

Активное прохождение получает snapshot вопросов, scoring, вариантов и отображаемых переводов. Вставка/удаление/перестановка live-definition не меняет уже выданные callbacks и баллы. Старый импорт файла больше не удаляет всю таблицу. Обновление существующих вопросов требует явного ID; варианты без stable IDs изменяются через карточку, а не неоднозначным позиционным импортом. Переводной импорт не сбрасывает формулы.

ReplyKeyboard labels сохраняют привязку к stable resource ID в user_menu_bindings. Старое меню без такой привязки обновляется при нажатии; название не используется для выбора бизнес-объекта. Повторное назначение выданной подписи другому объекту запрещено.

## Отсутствие перевода и review

Админ видит именно выбранный язык. При отсутствии: «Перевод не задан», с RU-справкой при наличии. Runtime использует выбранный язык, затем только настоящий RU; без обязательного display value ресурс скрывается/недоступен. MAX не использует EN/PT и пропускает RU-пустые ресурсы.

После RU-изменения существующий перевод остаётся сохранённым и доступным, помечается «Требует проверки после изменения русского текста». Сохранение/подтверждение фиксирует актуальный hash. Если RU изменился во время открытой формы, требуется открыть её заново. Некорректная machine syntax не выдаётся пользователю. При non-RU-first остальные языки проверяются против уже существующего перевода на сохранность placeholders и embedded targets.

System readiness проверяет только system keys. Content completeness и review показываются отдельно по ресурсам; недостающая тема не выключает EN/PT.

## Packs

Обычный export содержит system scope. Legacy full pack распознаётся: admin-managed prefixes пропускаются, preview/result сообщают количество пропущенных ключей. Даже устаревшая dynamic-строка не перезаписывает свежую правку из Admin. System keys сохраняют строгую проверку locale, target, source hash, формата и completeness.

## PostgreSQL restore gate

До исправления production predicate содержал cast всего массива:

```sql
((status)::text = ANY ((ARRAY['claimed'::character varying, 'pending'::character varying, 'unknown'::character varying])::text[]))
```

После pg_dump/restore PostgreSQL вывел эквивалентные casts элементов:

```sql
((status)::text = ANY (ARRAY[('claimed'::character varying)::text, ('pending'::character varying)::text, ('unknown'::character varying)::text]))
```

В обоих случаях сравниваются те же три строковых значения без усечения или изменения регистра. Проверка PostgreSQL теперь читает pg_index/pg_class/pg_attribute/pg_opclass и pg_get_expr: unique/valid/ready/immediate, ровно subscription_id, btree, обычная колонка и default operator class. Predicate разбирается ограниченной грамматикой; разрешены только text/varchar casts без typmod, pg_catalog qualification и скобки. Иное поле, набор значений, оператор, выражение/функция или дополнительное условие отвергаются. SQLite-проверка не ослаблялась. Production index не изменяется и не пересоздаётся.

Изолированный gate использует восстановленный свежий backup: два init_db, проверка неизменного catalog, подключение application routers, сборка commands/cache, read-only payment sanity. Внешние workers/webhooks и отправки пользователям не запускаются. Восстановление production-backup допускается только как отдельная recovery-операция с остановкой writers: оно не является автоматическим откатом живых новых пользовательских транзакций. Старый binary без stable-answer/snapshot поддержки нельзя возвращать поверх уже изменённых определений; требуется совместимый runtime либо согласованное восстановление данных.

## Walkthrough

1. `/admin` → «Сменить язык контента» → «Português».
2. «Темы» → карточка нужного ID → «Изменить: Название» → португальское значение → сообщение «Сохранено».
3. «Добавить» в «Темы» → `Autoestima`. Создан один ID, RU отсутствует.
4. `/admin` → «Сменить язык контента» → «Русский» → тот же ID → «Изменить: Название» → `Самооценка`.
5. Аналогично «Контент», «Подписки → Тарифные планы», «Тест → Вопросы теста», рассылки, автоматизации и догонялки. Общие настройки действуют для всех языков.
6. Для проверки устаревшего перевода открыть EN/PT-поле и сохранить либо нажать «Подтвердить перевод».
7. «Языки» использовать для доступности языков пользователям и системного pack, не для названий тем/контента/вопросов.

Telegram native UI проверяется dispatcher/API-тестами; browser viewport screenshots не являются выполненной проверкой этой интерфейсной поверхности.
