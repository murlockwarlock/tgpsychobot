# Content authoring: historical safety checkpoint

Update: the user authorized a representation-only validator fix. A fresh backup was restored into `authoring_isolated_51de7cc9049844c2`; two init_db runs, catalog equality, minimum application startup and payment-table sanity passed. All 16 production catalogs passed the replacement read-only validator. The account below records the earlier pause, not the current release status. See `admin-content-authoring.md` for the implemented model.

Implementation is incomplete and has not been committed, merged or deployed.

- Branch: `feat/admin-content-authoring`.
- Production revision remains `3229b948f3bd3127439b73ed93e0b52ad8e08889`.
- Working changes implement a per-admin/bot locale preference, dynamic/system translation separation, generic authoring screens, neutral creation, stable answer aliases and active-test snapshots.
- Existing user-owned `.tmp*` files were not modified.

## Restore blocker

All 16 production database backups were created and their archive catalogs verified in `/root/translation_work/admin_authoring__8xl8w4z`. The baseline is `baseline.json` in that directory.

Restoring `psy5d2_db.dump` into disposable database `authoring_isolated_f911790e14b44d37` succeeded, but application initialization failed in `verify_yookassa_recurring_safety_schema`:

```
Critical index idx_unresolved_yookassa_attempt contains invalid state item: '('claimed'::character varying)::text'
```

Read-only inspection confirmed that PostgreSQL represents the existing production predicate with an array-level text cast, while the restored database has equivalent per-element text casts. The existing strict payment-index validator does not recognize the restored representation. No payment verifier or production index was modified. Safe application startup after backup restoration remains unverified; release work stopped under the requested unexpected migration / rollback safety guardrail.

## Verification so far

- Last completed focused authoring run: 19 passed. Later working changes still require reruns.
- Earlier focused relevant run: 127 passed on its then-current snapshot.
- Independent broad file runner completed 21 files: 16 passed, four timed out, one failed on an attempted request to a mocked provider host. These results are not a green regression gate; baseline comparison and investigation remain outstanding.
- All heavy tests ran on the server, with fake bot credentials and isolated databases.
- Production read-only baseline: 18 managed processes online, 16 databases; RU default, RU/EN/PT enabled and selector on. Full translation-row digests capture text, keys, hashes and timestamps: 6,610 EN and 6,610 PT entries.
- Production migration-permission checks passed for all 16 databases.
- Task-owned broad regression runner and its test process were stopped. Unrelated server processes were not stopped.

## Remaining work

Resolve and test strict payment-index validation of restored PostgreSQL syntax before treating backup restoration as a safe recovery path. Then finish the resource audit, missing-value/runtime and legacy-form checks, compatible feature rollback, focused and broad regression, review, PR, normal merge, deployment and production verification. No readiness claim is justified yet.
