import re

from sqlalchemy import text


INDEX_NAME = "idx_unresolved_yookassa_attempt"
EXPECTED_STATES = {"claimed", "pending", "unknown"}
TOKEN = re.compile(r'''\s*(::|'(?:[^']|'')*'|"(?:[^"]|"")*"|[A-Za-z_][A-Za-z_0-9]*|[(),.\[\]=])''')


class PredicateParser:
    def __init__(self, source):
        self.tokens = []
        position = 0
        while position < len(source.rstrip()):
            match = TOKEN.match(source, position)
            if not match:
                raise ValueError("Unknown predicate syntax")
            value = match.group(1)
            self.tokens.append(value if value.startswith(("'", '"')) else value.lower())
            position = match.end()
        self.position = 0

    def accept(self, token):
        if self.position < len(self.tokens) and self.tokens[self.position] == token:
            self.position += 1
            return True
        return False

    def require(self, token):
        if not self.accept(token):
            raise ValueError("Unexpected predicate token")

    def identifier(self):
        if self.position >= len(self.tokens):
            raise ValueError("Missing identifier")
        value = self.tokens[self.position]
        if not re.fullmatch(r'[a-z_][a-z_0-9]*|"(?:[^"]|"")*"', value):
            raise ValueError("Invalid identifier")
        self.position += 1
        return value[1:-1].replace('""', '"') if value.startswith('"') else value

    def qualified_identifier(self):
        parts = [self.identifier()]
        while self.accept("."):
            parts.append(self.identifier())
        return parts

    def cast(self, node):
        parts = self.qualified_identifier()
        if len(parts) == 2 and parts[0] == "pg_catalog":
            parts = parts[1:]
        if parts == ["character"]:
            self.require("varying")
            parts = ["varchar"]
        if parts not in (["text"], ["varchar"]):
            raise ValueError("Non-identity cast")
        array = self.accept("[")
        if array:
            self.require("]")
        if array != (node[0] == "array") or node[0] not in {"array", "column", "literal"}:
            raise ValueError("Cast shape mismatch")
        return node

    def atom(self):
        if self.accept("("):
            node = self.expression()
            self.require(")")
        elif self.accept("array"):
            self.require("[")
            values = [self.atom()]
            while self.accept(","):
                values.append(self.atom())
            self.require("]")
            if any(value[0] != "literal" for value in values):
                raise ValueError("Non-literal array")
            node = ("array", tuple(value[1] for value in values))
        elif self.position < len(self.tokens) and self.tokens[self.position].startswith("'"):
            value = self.tokens[self.position]
            self.position += 1
            node = ("literal", value[1:-1].replace("''", "'"))
        else:
            parts = self.qualified_identifier()
            if parts not in (["status"], ["yookassa_recurring_attempts", "status"], ["public", "yookassa_recurring_attempts", "status"]):
                raise ValueError("Wrong predicate column")
            node = ("column", "status")
        while self.accept("::"):
            node = self.cast(node)
        return node

    def expression(self):
        node = self.atom()
        if self.accept("in"):
            self.require("(")
            values = [self.atom()]
            while self.accept(","):
                values.append(self.atom())
            self.require(")")
            if any(value[0] != "literal" for value in values):
                raise ValueError("Non-literal membership")
            return ("membership", node, tuple(value[1] for value in values))
        if self.accept("="):
            self.require("any")
            self.require("(")
            values = self.atom()
            self.require(")")
            if values[0] != "array":
                raise ValueError("Non-array membership")
            return ("membership", node, values[1])
        return node


def validate_unresolved_predicate(predicate):
    try:
        parser = PredicateParser(predicate or "")
        node = parser.expression()
        if parser.position != len(parser.tokens) or node[0] != "membership" or node[1] != ("column", "status"):
            raise ValueError("Wrong predicate structure")
        if set(node[2]) != EXPECTED_STATES:
            raise ValueError("Wrong predicate states")
    except (ValueError, IndexError, RecursionError) as exc:
        raise RuntimeError(f"Critical index {INDEX_NAME} predicate is not semantically equivalent") from exc


def read_postgres_unresolved_index(connection):
    return connection.execute(text("""
        SELECT i.indisunique, i.indisvalid, i.indisready, i.indimmediate,
               i.indnkeyatts, i.indnatts, i.indexprs IS NULL AS plain_columns,
               am.amname, a.attname AS column_name, opc.opcdefault,
               i.indcollation[0] AS collation_id, i.indoption[0] AS options,
               pg_catalog.pg_get_expr(i.indpred, i.indrelid, false) AS predicate
        FROM pg_catalog.pg_index i
        JOIN pg_catalog.pg_class idx ON idx.oid = i.indexrelid
        JOIN pg_catalog.pg_class tbl ON tbl.oid = i.indrelid
        JOIN pg_catalog.pg_am am ON am.oid = idx.relam
        LEFT JOIN pg_catalog.pg_attribute a ON a.attrelid = tbl.oid AND a.attnum = i.indkey[0]
        LEFT JOIN pg_catalog.pg_opclass opc ON opc.oid = i.indclass[0]
        WHERE i.indrelid = pg_catalog.to_regclass('yookassa_recurring_attempts')
          AND idx.relname = 'idx_unresolved_yookassa_attempt'
    """)).mappings().first()


def validate_postgres_unresolved_index(index):
    if index is None:
        raise RuntimeError(f"Critical index {INDEX_NAME} is missing in PostgreSQL")
    flags = ("indisunique", "indisvalid", "indisready", "indimmediate", "plain_columns", "opcdefault")
    if not all(index[key] is True for key in flags) or index["indnkeyatts"] != 1 or index["indnatts"] != 1 or index["column_name"] != "subscription_id" or index["amname"] != "btree" or index["collation_id"] != 0 or index["options"] != 0:
        raise RuntimeError(f"Critical index {INDEX_NAME} has incorrect catalog structure")
    validate_unresolved_predicate(index["predicate"])
