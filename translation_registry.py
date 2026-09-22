from __future__ import annotations

import base64
import json
import zlib
from html.parser import HTMLParser
from dataclasses import dataclass

from sqlalchemy import select

from response_buttons import extract_response_buttons
from translation_service import format_signature, source_hash, validate_translation_format
from content_locales import is_admin_content_key


TELEGRAM_MESSAGE_TEXT_LIMIT = 4096
TELEGRAM_CAPTION_TEXT_LIMIT = 1024
TELEGRAM_BUTTON_TEXT_LIMIT = 64


@dataclass(frozen=True)
class TranslationSource:
    translation_key: str
    source: str
    required: bool = True
    kind: str = "text"

    @property
    def domain(self) -> str:
        return "admin_content" if is_admin_content_key(self.translation_key) else "system"

    @property
    def source_hash(self) -> str:
        return source_hash(self.source)


STATIC_TRANSLATION_SOURCES = {
    "ui.language_selection_prompt": ("Пожалуйста, выберите язык:", "text", True),
    "ui.language.invalid": ("Недопустимый язык.", "text", True),
    "ui.language.unavailable": ("Этот язык сейчас недоступен.", "text", True),
    "ui.language.already_processed": ("Выбор языка уже обработан.", "text", True),
    "ui.language.resume_error": ("Не удалось продолжить регистрацию. Повторите /start.", "text", True),
    "ui.help.user": ("👋 Здравствуйте! Я ваш персональный ИИ-помощник.\n\nПросто напишите ваш вопрос в этот чат, и я постараюсь на него ответить. Вы можете использовать кнопки внизу для навигации по основным разделам или для управления диалогом.", "text", True),
    "ui.start.welcome_bonus": ("🎁 <b>Вам начислен приветственный бонус!</b>\nБесплатный доступ ко всем функциям на {days} дн.", "html", True),
    "ui.referral.new_registration": ("🎉 По вашей реферальной ссылке зарегистрировался новый пользователь!\nВам начислено <b>{days} бонусных дн.</b> к доступу. Спасибо, что рекомендуете нас!", "html", True),
    "ui.referral.registration_bonus": ("🎁 <b>Вам начислено {days} бонусных дн.</b> за регистрацию по пригласительной ссылке!", "html", True),
    "ui.settings.title": ("⚙️ <b>Настройки</b>", "html", True),
    "ui.settings.screen": ("⚙️ <b>Настройки</b>\n\n<b>Имя:</b> {name}\n<b>Пол:</b> {gender}\n<b>Возраст:</b> {age}\n<b>Длина ответов:</b> {response_length}\n", "html", True),
    "ui.settings.language": ("🌐 Язык", "reply_button", True),
    "ui.settings.language_prompt": ("Выберите язык:", "text", True),
    "ui.settings.language_changed": ("✅ Язык изменён.", "text", True),
    "ui.settings.language_invalid": ("Недопустимый язык.", "text", True),
    "ui.settings.language_unavailable": ("Этот язык сейчас недоступен.", "text", True),
    "ui.settings.user_not_found": ("Пользователь не найден.", "text", True),
    "ui.settings.name_changed": ("✅ Имя изменено на <b>{name}</b>", "html", True),
    "ui.settings.name_prompt": ("Пожалуйста, введите новое имя, как мне к вам обращаться?", "text", True),
    "ui.settings.gender_prompt": ("Выберите ваш пол:", "text", True),
    "ui.settings.age_prompt": ("Введите ваш возраст числом (например, 25):", "text", True),
    "ui.settings.length_changed": ("✅ Длина ответов изменена: {length}", "text", True),
    "ui.settings.gender_changed": ("✅ Пол изменён: {gender}", "text", True),
    "ui.settings.age_changed": ("✅ Возраст установлен: {age}", "text", True),
    "ui.settings.name": ("Имя", "text", True),
    "ui.settings.gender": ("Пол", "text", True),
    "ui.settings.age": ("Возраст", "text", True),
    "ui.settings.response_length": ("Длина ответов", "text", True),
    "ui.settings.not_specified": ("Не указано", "text", True),
    "ui.settings.not_specified_age": ("Не указан", "text", True),
    "ui.settings.change_name": ("✏️ Изменить имя", "inline_button", True),
    "ui.settings.close": ("❌ Закрыть", "inline_button", True),
    "ui.settings.male": ("👨 Мужской", "text", True),
    "ui.settings.female": ("👩 Женский", "text", True),
    "ui.settings.gender_unknown": ("❓ Не указан", "text", True),
    "ui.settings.length_normal": ("📏 Обычный", "text", True),
    "ui.settings.length_short": ("📏 Короткий", "text", True),
    "ui.settings.gender_button": ("👤 Пол: {gender}", "inline_button", True),
    "ui.settings.age_button": ("🎂 Возраст: {age}", "inline_button", True),
    "ui.settings.length_button": ("Длина ответов: {length}", "inline_button", True),
    "ui.button.test": ("📝 Пройти тест", "reply_button", True),
    "ui.button.subscription": ("⭐️ Подписка", "reply_button", True),
    "ui.button.settings": ("⚙️ Настройки", "reply_button", True),
    "ui.button.new_dialogue": ("🗑️ Новый диалог", "reply_button", True),
    "ui.button.topics": ("📚 Темы диалога", "reply_button", True),
    "ui.button.referral": ("👥 Пригласить друзей", "reply_button", True),
    "ui.button.back": ("⬅️ Назад", "text", True),
    "ui.subscription.choose_plan": ("💳 Оформить/Сменить тариф", "inline_button", True),
    "ui.subscription.cancel_renewal": ("❌ Отменить автопродление", "inline_button", True),
    "ui.subscription.enable_renewal": ("✅ Включить автопродление", "inline_button", True),
    "ui.subscription.enter_promo": ("🎁 Ввести промокод", "inline_button", True),
    "ui.subscription.retry_now": ("💳 Списать сейчас", "inline_button", True),
    "ui.subscription.cancel_retry": ("🔄 Отменить и оформить заново", "inline_button", True),
    "ui.subscription.check_status": ("🔄 Проверить статус", "inline_button", True),
    "ui.subscription.back_to_plans": ("⬅️ Назад к выбору тарифа", "inline_button", True),
    "ui.subscription.subscribe": ("💳 Оформить подписку", "inline_button", True),
    "ui.subscription.days": ("дн.", "text", True),
    "ui.subscription.months": ("мес.", "text", True),
    "ui.subscription.rubles": ("руб.", "text", True),
    "ui.subscription.unknown_plan": ("Неизвестный тариф", "text", True),
    "ui.subscription.active_screen": ("<b>⭐️ Ваша подписка активна</b>\n\n<b>Тариф:</b> {plan}\n<b>Действует до:</b> {end_date} МСК{renewal_line}", "html", True),
    "ui.subscription.renewal_line": ("\n<b>Автопродление:</b> {status}", "html", True),
    "ui.subscription.renewal_enabled": ("✅ Включено", "text", True),
    "ui.subscription.renewal_disabled": ("❌ Выключено", "text", True),
    "ui.subscription.discount_line": ("\n<b>Ваша скидка:</b> {percent}%", "html", True),
    "ui.subscription.discount_suffix": (" (со скидкой)", "text", True),
    "ui.subscription.discount_short": (" (со ск. {percent}%)", "text", True),
    "ui.subscription.upgrade_note": (" (далее {price:.2f} {rubles}/{duration} {unit})", "text", True),
    "ui.subscription.main_price": ("\n<b>Стоимость основного тарифа{discount_suffix}</b>: {price:.2f} руб за {duration} {unit}{renewal_note}", "html", True),
    "ui.subscription.price": ("\n<b>Стоимость{discount_suffix}</b>: {price:.2f} руб.", "html", True),
    "ui.subscription.renewal_charge_note": (" (спишется при включенном автопродлении)", "text", True),
    "ui.subscription.manual_trial_note": (" (оформляется вручную после окончания пробного периода)", "text", True),
    "ui.subscription.amount_line": ("\n<b>Сумма к списанию:</b> {amount:.2f} руб.", "html", True),
    "ui.subscription.period_line": ("\n<b>Период:</b> {period}", "html", True),
    "ui.subscription.attempts_line": ("\n<b>Попыток списания:</b> {count} из 3", "html", True),
    "ui.subscription.retry_pending": ("<b>⚠️ Подписка истекла, ожидаем результат автопродления</b>\n\n<b>Тариф:</b> {plan}{period_line}{amount_line}{attempts_line}\n\nЗапрос на списание уже отправлен в Robokassa. Можете проверить статус или отменить автопродление и оформить подписку заново.", "html", True),
    "ui.subscription.retry_waiting": ("<b>⚠️ Подписка истекла, ожидает оплаты по автопродлению</b>\n\n<b>Тариф:</b> {plan}{period_line}{amount_line}{attempts_line}\n\nК вашей карте привязан метод оплаты. Можете попробовать списание прямо сейчас или отменить автопродление и оформить новую подписку.", "html", True),
    "ui.subscription.no_active": ("У вас нет активной подписки.\n", "text", True),
    "ui.subscription.bonus_days": ("\nДоступные бонусные дни: {days}\n", "text", True),
    "ui.subscription.bonus_hours": ("\nДоступные бонусные часы: ~{hours}\n", "text", True),
    "ui.subscription.bonus_ending": ("\nБонусный доступ скоро закончится.\n", "text", True),
    "ui.subscription.discount_expiring": ("🔥 У вас есть скидка <b>{percent}%</b>, которая <b>сгорит</b>, если не оформить подписку до окончания бонусных дней!\n", "html", True),
    "ui.subscription.purchase_prompt": ("\nОформите подписку, чтобы получить доступ ко всем возможностям бота!", "text", True),
    "ui.subscription.available_discount": ("\nДоступная скидка: {percent} % (применяется к подходящим тарифам)\n", "text", True),
    "ui.subscription.purchase_prompt_short": ("\nОформите ее, чтобы получить доступ ко всем возможностям бота!", "text", True),
    "ui.subscription.choose_plan_prompt": ("Выберите подходящий тариф:", "text", True),
    "ui.subscription.switch_notice": ("\n\n<b>При смене тарифа срок оплаты нового тарифа добавится к текущему (прибавятся неиспользуемые дни).</b>", "html", True),
    "ui.subscription.discount_notice": ("\n\n<i>У вас есть активные скидки! Они будут применены к подходящим тарифам.</i>", "html", True),
    "ui.subscription.no_plans": ("К сожалению, сейчас нет доступных тарифных планов.", "text", True),
    "ui.subscription.plan_label": ("<b>Тариф:</b>", "html", True),
    "ui.subscription.cost_label": ("<b>Стоимость:</b>", "html", True),
    "ui.subscription.next_label": ("<b>Далее:</b>", "html", True),
    "ui.subscription.after_trial_label": ("<b>После пробного периода:</b>", "html", True),
    "ui.subscription.auto_switch_note": ("(автопереход на «{plan_name}»)\n\n", "text", True),
    "ui.subscription.manual_plan_note": ("(тариф «{plan_name}», оформление вручную)\n\n", "text", True),
    "ui.subscription.payment_method_prompt": ("Выберите способ оплаты:", "text", True),
    "ui.subscription.not_found": ("Не удалось найти вашу подписку.", "text", True),
    "ui.subscription.plan_not_found": ("Тариф не найден.", "text", True),
    "ui.subscription.renewal_enabled_alert": ("Автопродление подписки включено.", "text", True),
    "ui.subscription.renewal_disabled_alert": ("Автопродление подписки отменено.", "text", True),
    "ui.subscription.renewal_cancelled_alert": ("Автопродление отменено.", "text", True),
    "ui.subscription.retry_unavailable": ("Невозможно выполнить списание.", "text", True),
    "ui.subscription.renewed_until": ("✅ Подписка продлена до {end_date} МСК.", "text", True),
    "ui.subscription.retry_pending_alert": ("Запрос уже в обработке, ожидайте подтверждения.", "text", True),
    "ui.subscription.bank_declined": ("Банк отклонил списание. Подписка пока активна до {end_date}.", "text", True),
    "ui.subscription.robokassa_wait": ("Предыдущий запрос ещё не отражён в Robokassa. Подождите до 3 часов.", "text", True),
    "ui.subscription.robokassa_status_unknown": ("Статус платежа в Robokassa не подтвердился. Новый запрос сейчас не отправлялся.", "text", True),
    "ui.subscription.configuration_error": ("Ошибка конфигурации.", "text", True),
    "ui.subscription.retry_limit": ("Повторное списание недоступно: исчерпан лимит из 3 попыток.", "text", True),
    "ui.subscription.charge_request": ("Отправляем запрос на списание...", "text", True),
    "ui.subscription.retry_after": ("Повторное списание пока недоступно. Следующая попытка после {next_retry}.", "text", True),
    "ui.subscription.payment_processed_active": ("Платёж уже обработан. Подписка активна.", "text", True),
    "ui.subscription.payment_processed_method_disabled": ("Платёж уже обработан. Способ оплаты был отключён.", "text", True),
    "ui.subscription.payment_processed": ("Платёж уже обработан.", "text", True),
    "ui.subscription.previous_attempt_done": ("Предыдущая попытка списания завершена. Текущие настройки вашей подписки сохранены.", "text", True),
    "ui.subscription.unknown_bank_response": ("Не удалось провести оплату (нестандартный ответ банка). Автопродление приостановлено во избежание повторных списаний. Пожалуйста, оформите подписку вручную.", "text", True),
    "ui.subscription.request_error": ("Произошла ошибка при обработке запроса. Пожалуйста, попробуйте позже или выберите тариф в меню.", "text", True),
    "ui.subscription.payment_processing": ("Предыдущий платёж ещё обрабатывается / проверяется банком. Пожалуйста, подождите завершения операции.", "text", True),
    "ui.subscription.retry_unavailable_short": ("Повторное списание недоступно.", "text", True),
    "ui.subscription.robokassa_request_sent": ("⏳ Запрос на списание отправлен. Ожидайте подтверждения оплаты.", "text", True),
    "ui.subscription.robokassa_deactivated": ("Не удалось списать средства (Robokassa). Автопродление отключено.\n\nОформите подписку вручную.", "text", True),
    "ui.subscription.robokassa_provider_error": ("Robokassa временно недоступна. Эта ошибка не засчитана как попытка списания.\n\nПопробуйте снова позже.", "text", True),
    "ui.subscription.robokassa_failed": ("Не удалось списать средства (Robokassa). Проверьте состояние карты и попробуйте позже.", "text", True),
    "ui.subscription.renewal_expired": ("Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\nПродлите подписку вручную в меню.", "text", True),
    "ui.subscription.provider_unsupported": ("Провайдер не поддерживает ручное списание.", "text", True),
    "ui.subscription.manual_subscription_unresolved": ("⚠️ Мы получили оплату ({amount:.2f} руб), но не удалось найти вашу подписку для автоматического продления. Платёж отправлен на проверку администратору.", "text", True),
    "ui.subscription.manual_cross_plan": ("⚠️ Мы получили оплату ({amount:.2f} руб) по вашему предыдущему тарифу «{paid_name}». Поскольку сейчас у вас активен тариф «{current_name}», платёж отправлен на проверку администратору. Срок действия текущей подписки не был изменён автоматически.", "text", True),
    "ui.subscription.expired_payment_method_manual": ("Ваша подписка истекла. Сохранённый способ оплаты больше недоступен в ЮKassa. Оформите подписку вручную.", "text", True),
    "ui.subscription.expired_payment_method": ("Ваша подписка истекла. Сохранённый способ оплаты больше недоступен в ЮKassa (автопродление отключено).\n\nОформите подписку вручную.", "text", True),
    "ui.subscription.manual_review": ("Автопродление приостановлено для ручной проверки. Пожалуйста, оформите подписку заново в меню.", "text", True),
    "ui.subscription.pending_yookassa": ("Запрос в ЮKassa принят и ожидает подтверждения оплаты. Мы проверяем статус операции.", "text", True),
    "ui.subscription.unknown_yookassa": ("Платёжный шлюз ЮKassa обрабатывает запрос. Мы проверяем статус операции. Попробуйте снова позже.", "text", True),
    "ui.subscription.provider_yookassa": ("ЮKassa временно недоступна. Эта ошибка не засчитана как попытка списания.\n\nПопробуйте снова позже.", "text", True),
    "ui.subscription.limit_exceeded": ("Не удалось провести списание (превышен лимит по карте). Следующая попытка будет завтра. Вы также можете привязать другую карту в меню.", "text", True),
    "ui.subscription.retry_failed_1": ("Не удалось списать средства (ЮKassa). Повторим попытку {next_retry}.", "text", True),
    "ui.subscription.retry_failed_2": ("Не удалось списать средства (ЮKassa). Последняя попытка — {next_retry}.", "text", True),
    "ui.subscription.charge_declined": ("Не удалось списать средства ({reason}).", "text", True),
    "ui.subscription.payment_created_link_failed": ("Платёж создан, но не удалось показать ссылку. Обратитесь в поддержку, чтобы не создавать повторный платёж.", "text", True),
    "ui.subscription.payment_link_ready": ("Ваша ссылка на оплату готова.\n\nНажимая «Оплатить», я даю согласие на <a href='{privacy_url}'>обработку персональных данных</a> и принимаю <a href='{offer_url}'>договор оферты</a>.\n\n<b>Сумма:</b> {price:.2f} руб.\n{payment_type_line}", "html", True),
    "ui.subscription.payment_recurring": ("Регулярная оплата, можно отключить в любой момент", "text", True),
    "ui.subscription.payment_one_time": ("Разовая оплата", "text", True),
    "ui.subscription.yookassa_flow_error": ("❌ Не удалось подготовить платёж. Попробуйте ещё раз через несколько минут.", "text", True),
    "ui.subscription.yookassa_unconfigured": ("❌ Платёжная система временно недоступна. Администратор не настроил ключи API.", "text", True),
    "ui.subscription.yookassa_auth_error": ("❌ Не удалось создать платеж. Похоже, возникла проблема с настройками платежной системы. Мы уже работаем над этим.", "text", True),
    "ui.subscription.payment_service_error": ("❌ Не удалось связаться с платёжным сервисом. Попробуйте ещё раз через несколько минут.", "text", True),
    "ui.subscription.telegram_pay_unconfigured": ("❌ Оплата через Telegram Pay временно недоступна. Администратор не настроил токен.", "text", True),
    "ui.subscription.telegram_pay_error": ("❌ Не удалось связаться с платёжным сервисом. Попробуйте ещё раз через несколько минут.", "text", True),
    "ui.subscription.robokassa_flow_error": ("❌ Не удалось подготовить ссылку на оплату. Попробуйте ещё раз через несколько минут.", "text", True),
    "ui.subscription.robokassa_unconfigured": ("❌ Платёжная система Robokassa временно недоступна. Администратор не настроил ключи API.", "text", True),
    "ui.subscription.robokassa_minimum_amount": ("❌ Сумма к оплате меньше 1 руб. — Robokassa не принимает такие платежи. Обратитесь к администратору.", "text", True),
    "ui.subscription.robokassa_payment_link_ready": ("Ваша ссылка на оплату готова.\n\n{consent_line}\n\n<b>Сумма:</b> {price:.2f} руб.\n{payment_type_line}\n<b>Счёт действует до:</b> {expires_at}\n\nЕсли срок действия истечёт, по кнопке ниже автоматически откроется новый счёт.", "html", True),
    "ui.subscription.payment_provider_yookassa": ("💳 Оплатить через ЮKassa", "inline_button", True),
    "ui.subscription.payment_provider_robokassa": ("💳 Оплатить через Robokassa", "inline_button", True),
    "ui.subscription.payment_provider_telegram": ("💳 Оплатить через Telegram", "inline_button", True),
    "ui.promo.prompt": ("Пожалуйста, введите ваш промокод:", "text", True),
    "ui.promo.cancelled": ("Ввод промокода отменен. Вы можете продолжить общение.", "text", True),
    "ui.promo.profile_error": ("Произошла ошибка, не удалось найти ваш профиль.", "text", True),
    "ui.promo.no_offers": ("К сожалению, в данный момент нет доступных промо-предложений.", "text", True),
    "ui.promo.heading": ("<b>⭐️ Специальные предложения!</b>", "html", True),
    "ui.promo.terms": ("«Оформляя пробную подписку, вы соглашаетесь с условиями выбранного тарифа. Если для него доступно автопродление, по окончании пробного периода подписка автоматически перейдет на обычный тариф (отмена в любое время)»." , "text", True),
    "ui.promo.active_notice": ("<b>У вас уже есть активная подписка.</b> Новый пробный тариф добавится к текущему сроку.", "html", True),
    "ui.promo.choose_trial": ("Выберите пробный тариф:", "text", True),
    "ui.promo.invalid": ("❌ Промокод не найден, истёк или недействителен. Попробуйте ещё раз или нажмите «Назад».", "text", True),
    "ui.promo.already_used": ("❌ Вы уже активировали этот промокод.", "text", True),
    "ui.promo.inactive": ("❌ Этот промокод неактивен (0% скидки и 0 дней). Обратитесь к администратору.", "text", True),
    "ui.promo.trial_wait": ("ℹ️ Этот промокод даёт пробный период ({days} дн.). Активируйте его после окончания текущей подписки.", "text", True),
    "ui.promo.discount_saved_trial_blocked": ("✅ Скидка {percent}% сохранена, но пробный период ({days} дн.) нельзя активировать, пока у вас есть другая активная платная подписка.", "text", True),
    "ui.promo.activated_trial_discount": ("✅ Вам начислен пробный период: <b>{days} дн.</b> (до {end_date} МСК).\n\n🔥 Также вам назначена скидка <b>{percent}%</b>! Она сохранится для всех автоплатежей, <b>если вы оформите подписку до окончания пробного периода</b>.", "html", True),
    "ui.promo.activated_trial": ("✅ Пробный период успешно активирован!\nВам начислено: <b>{days} бесплатных дней</b>.\nДоступ активен до: {end_date} МСК.", "html", True),
    "ui.promo.saved_discount": ("✅ Скидка <b>{percent}%</b> сохранена!\nОна будет автоматически применена при выборе тарифа и всех последующих автоплатежах.", "html", True),
    "ui.promo.saved_discount_active": ("✅ Скидка <b>{percent}%</b> сохранена! Она будет применена при <b>следующем</b> продлении или смене тарифа.", "html", True),
    "ui.promo.activation_error": ("Произошла системная ошибка при активации промокода. Обратитесь в поддержку.", "text", True),
    "ui.voice.disabled": ("Извините, но распознавание голосовых сообщений в данный момент отключено администратором.", "text", True),
    "ui.voice.too_long": ("К сожалению, слишком длинное голосовое сообщение ({duration} сек.).\nПопробуйте ещё раз, максимум до {minutes} минут(ы).", "text", True),
    "ui.voice.processing": ("🤖 Распознаю ваше голосовое сообщение...", "text", True),
    "ui.voice.balance_error": ("К сожалению, сервис транскрибации временно недоступен из-за технической проблемы. Мы уже работаем над ее решением.", "text", True),
    "ui.voice.overloaded": ("Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос. Я буду ждать!", "text", True),
    "ui.voice.unexpected_error": ("Произошла непредвиденная ошибка при обработке аудио. Попробуйте ещё раз через несколько минут.", "text", True),
    "ui.test.invalid_option": ("Такого варианта ответа нет.", "text", True),
    "ui.test.invalid_callback": ("Некорректный вариант ответа.", "text", True),
    "ui.test.text_required": ("Пожалуйста, напишите ответ текстом.", "text", True),
    "ui.test.choose_option": ("Пожалуйста, выберите один из вариантов ниже.", "text", True),
    "ui.test.closed": ("Этот вопрос уже закрыт. Ответьте на текущий вопрос.", "text", True),
    "ui.payment.subscription_title": ("Оплата подписки на тариф «{plan_name}»", "text", True),
    "ui.payment.plan_label": ("Тариф «{plan_name}»", "text", True),
    "ui.profile.name_prompt": ("Прежде чем мы начнем, подскажите, как я могу к вам обращаться?", "text", True),
    "ui.profile.name_saved": ("Приятно познакомиться, {name}! Укажи свой пол:", "text", True),
    "ui.profile.name_change_saved": ("Отлично! Теперь я буду называть вас {name}. Укажите ваш пол:", "text", True),
    "ui.profile.gender_prompt": ("Укажите ваш пол:", "text", True),
    "ui.profile.gender_selected": ("Пол: {gender} ✅", "text", True),
    "ui.profile.age_prompt": ("Укажите ваш возраст:", "text", True),
    "ui.profile.age_prompt_before_start": ("А перед началом скажи: сколько тебе лет?", "text", True),
    "ui.profile.invalid_name": ("Пожалуйста, напишите имя обычным текстом, без команд и кнопок меню.", "text", True),
    "ui.profile.invalid_age": ("Пожалуйста, введите корректный возраст числом.", "text", True),
    "ui.profile.disclaimer_accept": ("✅ Я понимаю и принимаю", "inline_button", True),
    "ui.test.unavailable": ("⚠️ Тест временно недоступен: вопросы еще не загружены.", "text", True),
    "ui.test.disabled": ("⚠️ Тестирование в данный момент отключено.", "text", True),
    "ui.test.questions_missing": ("Ошибка: Вопросы теста не загружены. Обратитесь к администратору.", "text", True),
    "ui.test.session_missing": ("Ошибка: сессия теста не найдена. Попробуйте начать заново: /test", "text", True),
    "ui.test.session_finished": ("Эта сессия уже завершена или не существует.", "text", True),
    "ui.test.questions_finished": ("Вопросы теста уже закончились.", "text", True),
    "ui.test.invalid_text_answer": ("Пожалуйста, напишите ответ текстом.", "text", True),
    "ui.test.answer_loading": ("🤖 Спасибо! Анализирую ответы...", "text", True),
    "ui.test.cancelled_alert": ("❌ Тест прерван", "text", True),
    "ui.test.cancelled": ("Тестирование прервано. Возвращаемся в главное меню.", "text", True),
    "ui.test.interpretation_unavailable": ("Интерпретация результата сейчас недоступна. Попробуйте открыть результат позже.", "text", True),
    "ui.test.question_heading": ("<b>Вопрос {current} из {total}</b>", "html", True),
    "ui.test.free_text_hint": ("Напишите свой ответ или выберите из предложенных ниже.", "text", True),
    "ui.test.text_answer_hint": ("Напишите ответ сообщением.", "text", True),
    "ui.test.exit": ("❌ Выйти из теста", "inline_button", True),
    "ui.test.case_continue": ("Поехали дальше", "inline_button", True),
    "ui.secret.heading": ("<b>🔐 Секретный блок вопросов</b>", "html", True),
    "ui.secret.prompt": ("👇 <b>Напиши ответы одним сообщением ниже.</b>", "html", True),
    "ui.secret.instructions": ("Ответь на них максимально честно.\n\n", "text", True),
    "ui.secret.not_added": ("Вопросы еще не добавлены администратором.", "text", True),
    "ui.secret.empty_answer": ("Пожалуйста, напишите ответы.", "text", True),
    "ui.secret.deep_prompt": ("Готовы копнуть глубже и получить личный разбор от меня?", "text", True),
    "ui.secret.start_button": ("🔐 Пройти секретный тест", "inline_button", True),
    "ui.secret.marathon_button": ("Сразу на марафон 🚀", "inline_button", True),
    "ui.secret.continue_message": ("Я здесь! Мы можем обсудить твои результаты или поговорить на любую другую тему. Слушаю тебя.", "text", True),
    "ui.secret.generating": ("⏳ Генерирую подробную расшифровку и план действий...", "text", True),
    "ui.secret.thanks": ("Спасибо за ответы!", "text", True),
    "ui.secret.program": ("🔥 Программа марафона", "inline_button", True),
    "ui.secret.continue": ("🗣 Продолжить общение", "inline_button", True),
    "ui.ai.thinking": ("🤖 Думаю...", "text", True),
    "ui.ai.processing_default": ("Думаю...", "text", True),
    "ui.ai.button_accepted": ("Ответ принят: {button_text}", "text", True),
    "ui.ai.choose_action": ("Выберите действие:", "text", True),
    "ui.ai.image_ready": ("✨ Готово!", "text", True),
    "ui.ai.image_failed": ("😔 Не удалось сгенерировать изображение.", "text", True),
    "ui.ai.service_unavailable": ("К сожалению, сервис временно недоступен из-за технической проблемы.", "text", True),
    "ui.ai.overloaded": ("Ой. Нейросеть сейчас перегружена и не отвечает. Загляни через несколько минут и повтори запрос. Я буду ждать!", "text", True),
    "ui.ai.processing_failed": ("Произошла ошибка при обработке сообщения.", "text", True),
    "ui.ai.unexpected_failed": ("Произошла непредвиденная ошибка. Пожалуйста, попробуйте позже.", "text", True),
    "ui.ai.busy": ("Пожалуйста, не так быстро — я ещё разбираю твоё предыдущее сообщение. Дай мне немного времени, чтобы всё хорошенько обдумать.", "text", True),
    "ui.ai.image_inspecting": ("👀 Тщательно изучаю изображение...", "text", True),
    "ui.ai.editing_image": ("🎨 Редактирую ваше фото...", "text", True),
    "ui.ai.edited_image": ("✨ Результат редактирования:", "text", True),
    "ui.ai.edit_image_failed": ("😔 К сожалению, не удалось отредактировать изображение. Возможно, сервис дал сбой или запрос был отклонен фильтрами безопасности.", "text", True),
    "ui.ai.generating_image": ("🖼 Генерирую новое изображение...", "text", True),
    "ui.ai.generated_image": ("✨ Новая генерация:", "text", True),
    "ui.ai.photo_failed": ("Произошла ошибка при обработке фото.", "text", True),
    "ui.card.selection_prompt": ("Выбери карту, которая тебе откликается:", "text", True),
    "ui.card.next_selection_prompt": ("Выбери следующую карту:", "text", True),
    "ui.card.stale_choice": ("Эта карта уже не участвует в текущем выборе.", "text", True),
    "ui.card.selected": ("Твой выбор подтвержден.", "text", True),
    "ui.card.invalid_back": ("Эта техническая рубашка не должна выбираться.", "text", True),
    "ui.card.interpretation_failed": ("Не удалось получить интерпретацию карты. Попробуйте выбрать карту ещё раз через несколько минут.", "text", True),
    "ui.card.not_found": ("Ошибка: карта не найдена. Попробуйте начать выбор заново.", "text", True),
    "ui.card.spread_heading": ("Твой расклад целиком:", "text", True),
    "ui.card.next_failed": ("Не удалось показать следующие карты. Попробуй выбрать карту ещё раз или начни расклад заново.", "text", True),
    "ui.card.round_prompt": ("Сейчас раунд {current_round} из {total_rounds}. Выбери одну из карт кнопкой ниже:", "text", True),
    "ui.navigation.menu_hint": ("Нажмите на кнопку или воспользуйтесь меню для навигации", "text", True),
    "ui.navigation.continue_prompt": ("Введите ваше сообщение для начала/продолжения диалога:", "text", True),
    "ui.navigation.main_resume": ("✅ Мы вернулись в основной диалог.", "text", True),
    "ui.navigation.continue_ready": ("Спасибо! Теперь вы можете задать свой вопрос.", "text", True),
    "ui.action.unassigned": ("Действие не назначено.", "text", True),
    "ui.navigation.disclaimer_missing": ("Текст дисклеймера не задан.", "text", True),
    "ui.navigation.user_error": ("Ошибка пользователя.", "text", True),
    "ui.content.not_found": ("Контент не найден.", "text", True),
    "ui.access.subscription_required": ("Чтобы продолжить диалог, активируйте подписку / бонусные дни.", "text", True),
    "ui.access.photo_subscription_required": ("Чтобы отправлять фото и получать разборы, активируйте подписку.", "text", True),
    "ui.access.subscription_button": ("Начать пользоваться ботом", "inline_button", True),
    "ui.dialogue.reset_topic_prompt": ("Вы находитесь в диалоге: <b>{topic_name}</b>.\nПри начале нового диалога или переходе в основной память ИИ будет очищена.\nВыберите подходящее действие.", "html", True),
    "ui.dialogue.reset_main_prompt": ("При начале нового диалога память ИИ будет полностью очищена. Вы уверены?", "text", True),
    "ui.dialogue.confirm_delete": ("🗑️ Да, удалить", "inline_button", True),
    "ui.dialogue.confirmation_stale": ("Подтверждение устарело или уже использовано.", "text", True),
    "ui.dialogue.state_changed": ("Состояние диалога изменилось. Действие отменено.", "text", True),
    "ui.dialogue.memory_cleared": ("✅ Память очищена.", "text", True),
    "ui.dialogue.continue_current": ("Ок. Продолжаем текущий диалог.", "text", True),
    "ui.dialogue.new_topic": ("Начать новый диалог в данной теме", "inline_button", True),
    "ui.dialogue.main_topic": ("Перейти в основной диалог", "inline_button", True),
    "ui.dialogue.cancel": ("Отмена", "inline_button", True),
    "ui.dialogue.cancel_with_icon": ("❌ Отмена", "inline_button", True),
    "ui.content.not_configured": ("Приветствие не задано.", "text", True),
    "ui.topics.empty": ("К сожалению, сейчас нет доступных тем для диалога.", "text", True),
    "ui.topics.current_main": ("в <b>Основном диалоге</b>", "html", True),
    "ui.topics.current_prefix": ("Вы находитесь {status}.", "html", True),
    "ui.topics.menu_prompt": ("Выберите подходящую тему для общения.", "text", True),
    "ui.topics.menu_description": ("Бот будет использовать специальные знания и инструкции для ответов по выбранной теме.", "text", True),
    "ui.topics.main_button": ("🏠 Перейти в основной диалог", "inline_button", True),
    "ui.topics.cancel": ("❌ Отмена", "inline_button", True),
    "ui.topics.unavailable": ("Тема больше недоступна. Выберите другую тему в меню.", "text", True),
    "ui.topics.already_current": ("Вы уже находитесь в теме «{topic_name}».\n\nПродолжайте диалог — просто напишите ваш вопрос или сообщение.\n\nЕсли хотите начать эту тему заново, нажмите «{new_dialogue}».", "text", True),
    "ui.topics.already_current_generic": ("Вы уже находитесь в этой теме.\n\nПродолжайте диалог — просто напишите ваш вопрос или сообщение.\n\nЕсли хотите начать эту тему заново, нажмите «{new_dialogue}».", "text", True),
    "ui.topics.resume": ("✅ Продолжаем тему: «{topic_name}».", "text", True),
    "ui.topics.switch_restored": ("✅ Продолжаем тему: **{topic_name}**.", "text", True),
    "ui.topics.switch_global": ("✅ Отлично! Мы переключились на тему: **{topic_name}**.\n\nКонтекст диалога сохранен. Дальше бот будет использовать промпт текущей темы.", "text", True),
    "ui.topics.switch_reset": ("✅ Отлично! Мы переключились на тему: **{topic_name}**.\n\nПамять диалога была очищена. Можете задавать свой вопрос.", "text", True),
    "ui.referral.unavailable": ("Реферальная программа недоступна.", "text", True),
    "ui.referral.screen": (
        "🔗 <b>Реферальная программа</b>\n\n"
        "Пригласите друга по ссылке и получайте бонусные дни!\n\n"
        "🎁 <b>Ваши бонусы:</b>\n{referrer_bonus}\n\n"
        "🎁 <b>Бонусы вашего друга:</b>\n{friend_bonus}\n\n"
        "👥 <b>Приглашено:</b> {referral_count} чел.\n\n"
        "<b>Ваша ссылка:</b>\n{ref_link}",
        "html",
        True,
    ),
    "ui.referral.referrer.registration": ("• +{days} дн. при регистрации приглашённого", "text", True),
    "ui.referral.referrer.first_payment": ("• +{days} дн. при первой оплате приглашённого", "text", True),
    "ui.referral.referrer.each_payment": ("• +{days} дн. при оплате приглашённого (за каждую оплату)", "text", True),
    "ui.referral.referrer.none": ("• Сейчас бонусы для приглашающего не начисляются", "text", True),
    "ui.referral.friend.registration": ("• +{days} дн. при регистрации по вашей ссылке", "text", True),
    "ui.referral.friend.none": ("• Сейчас бонусы для друга не начисляются", "text", True),
    "ui.referral.templates.heading": ("📩 <b>Шаблоны приглашений</b>", "html", True),
    "ui.referral.templates.intro": (
        "Я отправлю несколько готовых сообщений ниже отдельными сообщениями. Выбери любой и отправь своим друзьям.",
        "html",
        True,
    ),
    "ui.referral.share": ("📤 Поделиться", "inline_button", True),
    "command.start.description": ("Запустить / Перезапустить бота", "text", True),
    "command.help.description": ("Помощь", "text", True),
    "command.topics.description": ("Выбрать тему", "text", True),
    "command.new_dialogue.description": ("Новый диалог", "text", True),
    "command.settings.description": ("Настройки", "text", True),
    "command.subscription.description": ("Подписка", "text", True),
    "command.ref.description": ("🤝 Пригласить друзей", "text", True),
    "notification.purchase_success": ("✅ Ваша подписка на тариф «{plan_name}» успешно оформлена!", "html", True),
    "notification.purchase_success.robokassa": ("Мы получили оплату {amount:.2f} руб по вашему тарифу «{plan_name}».\nДействие тарифа продлено до {end_date_msk} МСК.\n\nБлагодарим, что продолжаете пользоваться ботом!\nВы всегда можете направить нам свои пожелания, предложения по его работе.", "html", True),
    "notification.renewal_success": ("✅ Подписка продлена до {end_date_msk} МСК.", "html", True),
    "notification.renewal_success.robokassa": ("Мы получили оплату {amount:.2f} руб по вашему тарифу «{plan_name}».\nДействие тарифа продлено до {end_date_msk} МСК.\n\nБлагодарим, что продолжаете пользоваться ботом!\nВы всегда можете направить нам свои пожелания, предложения по его работе.", "html", True),
    "notification.referral_bonus": ("💰 Ваш реферал оформил подписку! Вам начислено <b>{bonus_days} бонусных дн.</b>", "html", True),
    "notification.deactivate": ("Ваша подписка истекла. Сохранённый способ оплаты больше недоступен в ЮKassa (автопродление отключено).\n\nПродлите подписку вручную в меню.", "text", True),
    "notification.deactivate.robokassa": ("Ваша подписка истекла. Ошибка при автоплатеже (Robokassa) — автопродление отключено.\n\nПродлите подписку вручную в меню.", "text", True),
    "notification.unknown_cancellation": ("Не удалось выполнить автоматическое списание (нестандартный ответ банка). Автопродление приостановлено во избежание повторных списаний.\n\nПожалуйста, оформите или продлите подписку вручную в меню бота.", "text", True),
    "notification.unknown_expired": ("Не удалось подтвердить результат списания за 24 часа. Чтобы избежать двойных списаний, автопродление приостановлено.\n\nПроверьте статус в банке или оформите подписку в меню.", "text", True),
    "notification.final_decline": ("Ваша подписка истекла. Не удалось списать средства после 3 попыток — автопродление отключено.\n\nПродлите подписку вручную в меню.", "text", True),
    "notification.retry_limit": ("Не удалось провести списание (превышен лимит по карте). Следующая попытка будет завтра. Вы также можете привязать другую карту в меню.", "text", True),
    "notification.retry_failed_1": ("Не удалось списать средства ({provider}). Повторим попытку {next_retry}.", "text", True),
    "notification.retry_failed_2": ("Не удалось списать средства ({provider}). Последняя попытка — {next_retry}.", "text", True),
    "notification.provider_error": ("Платёжный шлюз ЮKassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\nПовторим запрос позже.", "text", True),
    "notification.provider_error.robokassa": ("Платёжный шлюз Robokassa временно недоступен. Эта ошибка не засчитана как попытка списания.\n\nПовторим запрос после {next_retry}.", "text", True),
    "notification.cross_plan": ("⚠️ Мы получили оплату ({amount:.2f} руб) по вашему предыдущему тарифу «{paid_name}». Поскольку сейчас у вас активен тариф «{curr_name}», платёж отправлен на проверку администратору. Срок действия текущей подписки не был изменён автоматически.", "text", True),
    "notification.subscription_unresolved": ("⚠️ Мы получили оплату ({amount:.2f} руб), но не удалось найти вашу подписку для автоматического продления. Платёж отправлен на проверку администратору.", "text", True),
    "notification.paid_plan_unresolved": ("⚠️ Мы получили оплату ({amount:.2f} руб), но оплаченный тариф не найден в системе. Платёж отправлен на проверку администратору. Срок действия текущей подписки не был изменён автоматически.", "text", True),
    "notification.manual_review": ("Платёж отправлен на проверку администратору.", "text", True),
    "notification.trial_expiring": ("Ваш пробный период истекает {date_str} (через {time_display}).{discount_line}", "text", True),
    "notification.trial_discount_keep": ("\n\nОформите подписку, чтобы сохранить скидку {discount}%!", "text", True),
    "notification.trial_continue": ("\n\nОформите подписку для продолжения работы.", "text", True),
    "notification.renewal_reminder": ("Напоминаем: {date_str} продление тарифа «{plan_name}» на сумму {price:.2f} руб.", "text", True),
    "notification.trial_finished": ("Пробный период завершен. Выберите тариф для продолжения.", "text", True),
    "notification.subscription_expired": ("Подписка истекла. Продлите её в меню.", "text", True),
    "notification.yookassa_pending": ("⏳ Запрос автопродления ЮKassa принят и ожидает подтверждения банка.", "text", True),
    "notification.yookassa_unknown": ("Платёжный шлюз ЮKassa обрабатывает запрос. Мы проверяем статус операции.", "text", True),
    "notification.robokassa_pending": ("⏳ Попытка автопродления подписки «{plan_name}» на сумму {price:.2f} руб.\n\nЕсли деньги не спишутся в течение нескольких часов — проверьте, что карта активна и разрешены интернет-платежи.", "text", True),
}


def embedded_target_signature(text: str) -> tuple[tuple[tuple[str, str], ...], ...]:
    _, rows = extract_response_buttons(text)
    return tuple(
        tuple((button.kind, button.value) for button in row)
        for row in rows
    )


class TranslationRegistry:
    def __init__(self, sources: list[TranslationSource] | tuple[TranslationSource, ...]):
        by_key: dict[str, TranslationSource] = {}
        for source in sources:
            if not source.translation_key or source.translation_key in by_key:
                raise ValueError(f"Duplicate or empty translation key: {source.translation_key!r}")
            by_key[source.translation_key] = source
        self._sources = by_key

    def keys(self) -> list[str]:
        return sorted(self._sources)

    def get(self, translation_key: str) -> TranslationSource | None:
        return self._sources.get(translation_key)

    def snapshot(self) -> dict[str, TranslationSource]:
        return {key: self._sources[key] for key in self.keys()}

    def required_keys(self) -> tuple[str, ...]:
        return tuple(
            key for key in self.keys() if self._sources[key].required
        )

    def system(self) -> TranslationRegistry:
        return TranslationRegistry([source for source in self._sources.values() if source.domain == "system"])


def validate_translation_value(source: TranslationSource, translated: str) -> None:
    validate_translation_format(source.source, translated)
    if embedded_target_signature(source.source) != embedded_target_signature(translated):
        raise ValueError(
            f"Embedded button targets differ for {source.translation_key}"
        )
    if source.kind in {"reply_button", "inline_button"}:
        if not 1 <= len(translated) <= TELEGRAM_BUTTON_TEXT_LIMIT:
            raise ValueError(
                f"Telegram button text must contain 1..{TELEGRAM_BUTTON_TEXT_LIMIT} characters"
            )
    elif source.kind == "caption":
        if len(translated) > TELEGRAM_CAPTION_TEXT_LIMIT:
            raise ValueError(
                f"Telegram caption text must contain at most {TELEGRAM_CAPTION_TEXT_LIMIT} characters"
            )
    elif len(translated) > TELEGRAM_MESSAGE_TEXT_LIMIT:
        raise ValueError(
            f"Telegram message text must contain at most {TELEGRAM_MESSAGE_TEXT_LIMIT} characters"
        )
    validate_telegram_html(source.source, translated)


class _TelegramHTMLParser(HTMLParser):
    allowed_tags = {
        "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
        "span", "tg-spoiler", "tg-emoji", "a", "code", "pre", "blockquote", "br",
    }
    void_tags = {"br"}

    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.stack: list[str] = []
        self.link_targets: list[str] = []

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag not in self.allowed_tags:
            raise ValueError(f"unsupported Telegram HTML tag: {tag}")
        if tag == "a":
            self.link_targets.append(dict(attrs).get("href", ""))
        if tag not in self.void_tags:
            self.stack.append(tag)

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag.lower() not in self.void_tags and self.stack:
            self.stack.pop()

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag not in self.allowed_tags or tag in self.void_tags:
            raise ValueError(f"invalid Telegram HTML closing tag: {tag}")
        if not self.stack or self.stack[-1] != tag:
            raise ValueError(f"unbalanced Telegram HTML tag: {tag}")
        self.stack.pop()

    def finish(self):
        if self.stack:
            raise ValueError(f"unclosed Telegram HTML tag: {self.stack[-1]}")


def _validate_telegram_html_value(value: str) -> tuple[str, ...]:
    parser = _TelegramHTMLParser()
    parser.feed(value)
    parser.close()
    parser.finish()
    return tuple(parser.link_targets)


def validate_telegram_html(source: str, translated: str) -> None:
    try:
        source_links = _validate_telegram_html_value(source)
    except ValueError:
        source_links = None
    translated_links = _validate_telegram_html_value(translated)
    if source_links is not None and source_links != translated_links:
        raise ValueError("Telegram HTML link targets differ")


def _add_source(sources: list[TranslationSource], key: str, value, *, kind="text", required=True):
    sources.append(TranslationSource(key, value if isinstance(value, str) else "", required=required, kind=kind))


def _decode_ai_processing_text(value: str | None) -> str:
    if not isinstance(value, str):
        return ""
    prefix = "tg_entities_v1:"
    if not value.startswith(prefix):
        return value
    try:
        payload = json.loads(
            zlib.decompress(
                base64.urlsafe_b64decode(value[len(prefix):])
            ).decode("utf-8")
        )
        text = payload.get("text")
        return text if isinstance(text, str) else ""
    except Exception:
        return ""


def build_static_translation_registry() -> TranslationRegistry:
    return TranslationRegistry(
        [
            TranslationSource(key, value, required=required, kind=kind)
            for key, (value, kind, required) in STATIC_TRANSLATION_SOURCES.items()
        ]
    )


async def build_translation_registry(session) -> TranslationRegistry:
    from database import (
        AutomationAction,
        AutomationHandler,
        BotGeneralConfig,
        Content,
        CaseStudy,
        FollowupCampaign,
        FollowupStep,
        Mailing,
        MediaLibrary,
        ReferralTemplate,
        SecretTestQuestion,
        SubscriptionConfig,
        SubscriptionPlan,
        TestConfig,
        TestQuestion,
        Topic,
    )
    from universal_tests import get_answer_options

    sources = [
        TranslationSource(key, value, required=required, kind=kind)
        for key, (value, kind, required) in STATIC_TRANSLATION_SOURCES.items()
    ]

    general_config = await session.get(BotGeneralConfig, 1)
    if general_config:
        _add_source(
            sources,
            f"bot_general_config.{general_config.id}.ai_processing_message_text",
            _decode_ai_processing_text(getattr(general_config, "ai_processing_message_text", None)),
            required=bool(getattr(general_config, "ai_processing_message_enabled", False)),
        )

    test_config = await session.get(TestConfig, 1)
    test_enabled = bool(getattr(test_config, "is_enabled", False))
    secret_test_enabled = bool(test_enabled and getattr(test_config, "secret_test_enabled", False))
    content_rows = (await session.execute(select(Content))).scalars().all()
    for item in content_rows:
        if item.key in {"test_intro", "test_results", "test_button"}:
            required = test_enabled
        elif item.key == "secret_test_outro":
            required = secret_test_enabled
        else:
            required = bool(item.is_visible or item.key == "start_message")
        _add_source(sources, f"content.{item.key}.button_title", item.button_title, kind="reply_button", required=required)
        _add_source(sources, f"content.{item.key}.text_content", item.text_content, kind="html", required=required)
        _add_source(sources, f"content.{item.key}.action_btn_text", item.action_btn_text, kind="inline_button", required=required)

    topic_rows = (await session.execute(select(Topic))).scalars().all()
    for item in topic_rows:
        required = bool(item.is_active)
        _add_source(sources, f"topic.{item.id}.name", item.name, kind="reply_button", required=required)
        _add_source(sources, f"topic.{item.id}.description", item.description, required=required)
        _add_source(sources, f"topic.{item.id}.start_message", item.start_message, kind="html", required=required)
        _add_source(sources, f"topic.{item.id}.start_button_text", item.start_button_text, kind="inline_button", required=required)

    plan_rows = (await session.execute(select(SubscriptionPlan))).scalars().all()
    for item in plan_rows:
        required = bool(item.is_active)
        _add_source(sources, f"plan.{item.id}.name", item.name, required=required)
        _add_source(sources, f"plan.{item.id}.description", item.description, required=required)

    config = await session.get(SubscriptionConfig, 1)
    if config:
        required_fields = {
            "topics_btn_name": bool(getattr(config, "topics_enabled", True)),
            "referral_btn_name": bool(getattr(config, "referral_enabled", False)),
            "referral_sub_btn_name": bool(getattr(config, "referral_enabled", False)),
        }
        for field in ("topics_btn_name", "referral_btn_name", "referral_sub_btn_name"):
            _add_source(
                sources,
                f"subscription_config.{config.id}.{field}",
                getattr(config, field, None),
                kind="reply_button",
                required=required_fields[field],
            )

    referral_rows = (await session.execute(select(ReferralTemplate))).scalars().all()
    for item in referral_rows:
        _add_source(sources, f"referral_template.{item.id}.text", item.text, kind="html", required=bool(item.is_enabled))

    followup_rows = (
        await session.execute(select(FollowupStep, FollowupCampaign).join(FollowupCampaign))
    ).all()
    for item, campaign in followup_rows:
        if item.message_type == "static":
            _add_source(
                sources,
                f"followup_step.{item.id}.message_text",
                item.message_text,
                kind="html",
                required=bool(campaign.is_active),
            )

    mailing_rows = (await session.execute(select(Mailing))).scalars().all()
    for item in mailing_rows:
        _add_source(
            sources,
            f"mailing.{item.id}.text",
            item.text,
            kind="html",
            required=bool(item.is_enabled and item.recurring_type),
        )

    action_rows = (
        await session.execute(
            select(AutomationAction, AutomationHandler).join(
                AutomationHandler,
                AutomationHandler.id == AutomationAction.handler_id,
            )
        )
    ).all()
    for item, handler in action_rows:
        recipient_type = str(item.recipient_type or "").lower()
        if item.action_type != "send_message" or "admin" in recipient_type:
            continue
        required = bool(
            handler.is_active
            and item.action_type == "send_message"
            and "admin" not in recipient_type
        )
        _add_source(sources, f"automation_action.{item.id}.message_template", item.message_template, kind="html", required=required)

    question_rows = (await session.execute(select(TestQuestion))).scalars().all()
    for item in question_rows:
        _add_source(sources, f"test_question.{item.id}.text", item.text, kind="html", required=test_enabled)
        _add_source(sources, f"test_question.{item.id}.comment", item.comment, kind="html", required=test_enabled)
        for index, option in enumerate(get_answer_options(item)):
            slot = option.translation_slot if option.translation_slot is not None else str(index)
            _add_source(sources, f"test_question.{item.id}.option.{slot}.text", option.text, required=test_enabled)
            _add_source(sources, f"test_question.{item.id}.option.{slot}.button_text", option.button_text, kind="inline_button", required=False)

    secret_rows = (await session.execute(select(SecretTestQuestion))).scalars().all()
    for item in secret_rows:
        _add_source(sources, f"secret_test_question.{item.id}.text", item.text, kind="html", required=secret_test_enabled)

    for item in (await session.execute(select(MediaLibrary))).scalars():
        _add_source(sources, f"media_library.{item.id}.description", item.description, kind="caption", required=False)
    for item in (await session.execute(select(CaseStudy))).scalars():
        _add_source(sources, f"case_study.{item.id}.text", item.text, required=False)

    return TranslationRegistry(sources)
