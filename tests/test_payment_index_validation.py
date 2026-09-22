import pytest

from payment_index_validation import validate_postgres_unresolved_index, validate_unresolved_predicate


PRODUCTION_PREDICATE = "((status)::text = ANY ((ARRAY['claimed'::character varying, 'pending'::character varying, 'unknown'::character varying])::text[]))"
RESTORED_PREDICATE = "((status)::text = ANY (ARRAY[('claimed'::character varying)::text, ('pending'::character varying)::text, ('unknown'::character varying)::text]))"


def valid_catalog(predicate=PRODUCTION_PREDICATE):
    return dict(indisunique=True, indisvalid=True, indisready=True, indimmediate=True, indnkeyatts=1, indnatts=1, plain_columns=True, amname="btree", column_name="subscription_id", opcdefault=True, collation_id=0, options=0, predicate=predicate)


@pytest.mark.parametrize("predicate", [
    PRODUCTION_PREDICATE,
    RESTORED_PREDICATE,
    "status IN ('claimed', 'pending', 'unknown')",
    "(((status))) IN ((('unknown')), (('pending')), (('claimed')))",
    '''(("status")::pg_catalog.text = ANY ((ARRAY[(('claimed')::pg_catalog.varchar)::pg_catalog.text, ('pending')::pg_catalog.text, 'unknown'::pg_catalog.text])::pg_catalog.text[]))''',
    '''"public"."yookassa_recurring_attempts"."status" IN ('claimed', 'pending', 'unknown')''',
])
def test_equivalent_postgres_predicates(predicate):
    validate_unresolved_predicate(predicate)
    validate_postgres_unresolved_index(valid_catalog(predicate))


@pytest.mark.parametrize("predicate", [
    "other_field IN ('claimed', 'pending', 'unknown')",
    "status IN ('claimed', 'pending', 'succeeded')",
    "status IN ('claimed', 'pending')",
    "status IN ('claimed', 'pending', 'unknown', 'cancelled')",
    "status IN ('CLAIMED', 'pending', 'unknown')",
    "status NOT IN ('claimed', 'pending', 'unknown')",
    "status IN ('claimed', 'pending', 'unknown') OR true",
    "status IN ('claimed', 'pending', 'unknown') AND subscription_id > 0",
    "status != ANY (ARRAY['claimed', 'pending', 'unknown'])",
    "status = ALL (ARRAY['claimed', 'pending', 'unknown'])",
    "status = ANY (ARRAY['claimed', 'pending', other_column])",
    "status::evil.text = ANY (ARRAY['claimed', 'pending', 'unknown'])",
    "status::varchar(3) IN ('claimed', 'pending', 'unknown')",
    "status::char IN ('claimed', 'pending', 'unknown')",
    "status::integer IN ('claimed', 'pending', 'unknown')",
    "status IN ('claimed', 'pending', 'unknown'); SELECT true",
    "status IN ('claimed', 'pending', 'unknown') -- bypass",
    "status = ANY (ARRAY['claimed', 'pending', 'unknown'])::text",
    "(status IN ('claimed', 'pending', 'unknown')",
    None,
    "",
])
def test_non_equivalent_predicates_fail_closed(predicate):
    with pytest.raises(RuntimeError, match="semantically equivalent"):
        validate_postgres_unresolved_index(valid_catalog(predicate))


@pytest.mark.parametrize("field,value", [
    ("indisunique", False), ("indisvalid", False), ("indisready", False),
    ("indimmediate", False), ("indnkeyatts", 2), ("indnatts", 2),
    ("plain_columns", False), ("column_name", "user_id"), ("amname", "hash"),
    ("opcdefault", False), ("collation_id", 100), ("options", 1),
])
def test_correct_name_does_not_override_wrong_catalog_structure(field, value):
    catalog = valid_catalog()
    catalog[field] = value
    with pytest.raises(RuntimeError, match="catalog structure"):
        validate_postgres_unresolved_index(catalog)


def test_missing_index_fails():
    with pytest.raises(RuntimeError, match="missing"):
        validate_postgres_unresolved_index(None)
