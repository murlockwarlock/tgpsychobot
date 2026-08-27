# Production bot instances

The authoritative mapping is [config/bot_instances.json](../config/bot_instances.json).
The table below is a human-readable projection of that registry; do not edit it
as an independent configuration source.

| Platform | Public username/name | PM2 name | Database | Secret env key | Mode |
| --- | --- | --- | --- | --- | --- |
| telegram | autobusbusbot | tg_autobusbusbot_new | psy5d2_db | TELEGRAM_AUTOBUSBUSBOT_TOKEN | new |
| telegram | alena322bot | tg_alena322bot_new | veraveda2_db | TELEGRAM_ALENA322BOT_TOKEN | new |
| telegram | someonewithyou01_bot | tg_someonewithyou01_bot_new | someonewithyou01_db | TELEGRAM_SOMEONEWITHYOU01_BOT_TOKEN | new |
| telegram | someonewithyou02_bot | tg_someonewithyou02_bot_new | someonewithyou02_db | TELEGRAM_SOMEONEWITHYOU02_BOT_TOKEN | new |
| telegram | veraveda777_bot | tg_veraveda777_bot_legacy | veraveda_db | TELEGRAM_VERAVEDA777_BOT_TOKEN | legacy |
| telegram | someonelikeyouai03_bot | tg_someonelikeyouai03_bot_legacy | someone02 | TELEGRAM_SOMEONELIKEYOUAI03_BOT_TOKEN | legacy |
| telegram | someonelikeyouai04_bot | tg_someonelikeyouai04_bot_legacy | someone01 | TELEGRAM_SOMEONELIKEYOUAI04_BOT_TOKEN | legacy |
| telegram | psy5d_bot | tg_psy5d_bot_legacy | psy5d_db | TELEGRAM_PSY5D_BOT_TOKEN | legacy |
| telegram | someonelikeyouai_bot | tg_someonelikeyouai_bot_legacy | test01_db | TELEGRAM_SOMEONELIKEYOUAI_BOT_TOKEN | legacy |
| telegram | someonelikeyouai02_bot | tg_someonelikeyouai02_bot_legacy | test02_db | TELEGRAM_SOMEONELIKEYOUAI02_BOT_TOKEN | legacy |
| telegram | someonewithyou03_bot | tg_someonewithyou03_bot_new | someonewithyou03_db | TELEGRAM_SOMEONEWITHYOU03_BOT_TOKEN | new |
| telegram | someonelikeyouai05_bot | tg_someonelikeyouai05_bot_new | someonewithyou05_db | TELEGRAM_SOMEONELIKEYOUAI05_BOT_TOKEN | new |
| telegram | someonewithyou05_bot | tg_someonewithyou05_bot_new | someonewithyou06_db | TELEGRAM_SOMEONEWITHYOU05_BOT_TOKEN | new |
| telegram | someonewithyou06_bot | tg_someonewithyou06_bot_new | someonewithyou07_db | TELEGRAM_SOMEONEWITHYOU06_BOT_TOKEN | new |
| telegram | someonewithyou04_bot | tg_someonewithyou04_bot_new | someonewithyou08_db | TELEGRAM_SOMEONEWITHYOU04_BOT_TOKEN | new |
| telegram | yourself_way_bot | tg_yourself_way_bot_new | yourself_way_db | TELEGRAM_YOURSELF_WAY_BOT_TOKEN | new |
| max | se13639182_bot | max_se13639182_bot_legacy | veraveda_db | MAX_SE13639182_BOT_TOKEN | legacy |
| max | id519010411655_bot | max_id519010411655_bot_new | yourself_way_db | MAX_ID519010411655_BOT_TOKEN | new |
| telegram | kontentzavod322bot | tg_kontentzavod322bot_fulltest | psy5d_fulltest | PSY5D_FULLTEST_BOT_TOKEN | fulltest |

The last entry is the stopped, opt-in full-test process and is not part of the
18-process production deploy. `vitaliy_bot` and `darimiru_bot` are PM2 entries
outside this repository; their credentials and identities were not exposed by
this application's runtime and they are not renamed here.

The old PM2 names remain only in the registry's `legacy_pm2_name` fields and in
stable PM2 log filenames so the first rename can preserve the existing log
baseline; they are not active canonical process names.

Never identify a bot by someone03/someone04 numbering again; use the canonical
username and registry.

Before a production rollout, place each canonical token and database URL in the
untracked runtime environment under the `token_env` and `database_env` names,
then run:

```bash
python3 scripts/verify_bot_instances.py --validate
python3 scripts/verify_bot_instances.py --validate-runtime-env --runtime-env /path/to/runtime.env
python3 scripts/verify_bot_instances.py --runtime --migration-aware
```

The verifier compares the process token/database with those canonical values,
reports only short token fingerprints, and can perform read-only identity checks
with `--runtime --identity-check`.

Rolling migration accepts one active process per registered instance: a migrated
instance uses its canonical PM2 name and an unmigrated instance uses its
`legacy_pm2_name`. An old process is allowed beside a canonical process only
during the bounded cutover check while the old process is stopped; the normal
migration-aware verifier rejects any remaining legacy entry after cutover.

Use `scripts/migrate_bot_secrets.py --plan` before `--apply` to copy canonical
secret keys from a verified running legacy environment. The utility prints only
token fingerprints and database names.

Normal deploys may omit `PROD_PM2_NAMES` to verify and reload all 18 canonical
instances, or specify a canonical subset. Set `PROD_ALLOW_PM2_RENAME=1` only for
a one-instance cutover; that mode requires exactly one canonical PM2 name and
uses the rollback-safe cutover helper.
