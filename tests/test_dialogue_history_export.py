from dialogue_history_export import serialize_human_dialogue_history


def test_serializer_marks_first_and_each_dialogue_boundary_without_reordering():
    records = [
        (1, "first user"),
        (1, "first bot"),
        (3, "third user"),
        (3, "third bot"),
        (8, "eighth user"),
    ]

    result = serialize_human_dialogue_history(records)

    assert result.splitlines() == [
        "--- **Начало нового диалога №1** ---",
        "first user",
        "first bot",
        "--- **Начало нового диалога №3** ---",
        "third user",
        "third bot",
        "--- **Начало нового диалога №8** ---",
        "eighth user",
    ]


def test_serializer_handles_single_dialogue():
    assert serialize_human_dialogue_history([(4, "only message")]) == (
        "--- **Начало нового диалога №4** ---\nonly message"
    )


def test_serializer_preserves_multiline_rendered_entries():
    assert serialize_human_dialogue_history([(2, "line one\nline two"), (5, "next")]) == (
        "--- **Начало нового диалога №2** ---\n"
        "line one\nline two\n"
        "--- **Начало нового диалога №5** ---\n"
        "next"
    )
