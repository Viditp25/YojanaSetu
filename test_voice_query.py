"""Quick verification test for YojanaSetu Voice NL-to-SQLite pipeline."""

from voice_assistant import inspect_database_schema, query_schemes


def main() -> None:
    schema = inspect_database_schema("schemes.db")
    assert schema["primary_table"] == "schemes", f"Unexpected primary table: {schema['primary_table']}"
    assert len(schema["tables"]["schemes"]["columns"]) > 0, "No columns inspected"
    print(
        f"Schema inspected OK -> table='{schema['primary_table']}', "
        f"rows={schema['tables']['schemes']['row_count']}, "
        f"columns={len(schema['tables']['schemes']['columns'])}"
    )

    sample_queries = [
        (
            "Hindi",
            "मैं उत्तर प्रदेश का 45 साल का किसान हूँ, मुझे खेती और आर्थिक मदद के लिए कौन सी योजना मिल सकती है?",
        ),
        (
            "Marathi",
            "मी महाराष्ट्रातील 28 वर्षांची महिला आहे, मला स्वतःचा व्यवसाय सुरू करण्यासाठी किंवा आर्थिक मदतीसाठी कोणती योजना आहे?",
        ),
        (
            "English",
            "I am a 21 year old student looking for skill training and education scholarship schemes.",
        ),
    ]

    for expected_lang, query in sample_queries:
        res = query_schemes(query, db_path="schemes.db", top_k=3)
        assert res["language"] == expected_lang, f"Expected {expected_lang}, got {res['language']}"
        assert 1 <= len(res["schemes"]) <= 3, f"Expected 1-3 schemes, got {len(res['schemes'])}"
        for sc in res["schemes"]:
            assert set(sc.keys()) == {
                "scheme_name",
                "eligibility",
                "documents_needed",
                "official_website",
            }, f"Unexpected keys: {sc.keys()}"
            assert sc["scheme_name"], "Empty scheme_name"
            assert sc["eligibility"], "Empty eligibility"
            assert sc["documents_needed"], "Empty documents_needed"
            assert sc["official_website"].startswith("http"), f"Invalid URL: {sc['official_website']}"

        print(f"\n=== [{res['language']}] Provider: {res['provider']} ===")
        print(f"Entities: {res['entities']}")
        print(res["formatted_reply"])

    print("\nAll Hindi, Marathi, and English voice query checks passed!")


if __name__ == "__main__":
    main()
