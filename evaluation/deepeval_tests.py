if __name__ == "__main__":
    print("=" * 50)
    print("MEDASSIST DEEPEVAL TESTS")
    print("=" * 50)

    passed = 0
    failed = 0

    test_results = {}

    # Schema tests
    schema_tests = [test_schema_validation, test_ragas_scores]

    for test in schema_tests:
        try:
            test()
            passed += 1
            test_results[test.__name__] = {"status": "PASS"}

        except AssertionError as e:
            print(f"{e}")
            failed += 1
            test_results[test.__name__] = {
                "status": "FAIL",
                "error": str(e)
            }

        except Exception as e:
            print(f"ERROR: {e}")
            failed += 1
            test_results[test.__name__] = {
                "status": "ERROR",
                "error": str(e)
            }

    # Agent tests
    agent_tests = [
        (
            "chest_pain_emergency",
            "I have severe chest pain and cannot breathe since 30 minutes",
            ["HIGH", "EMERGENCY"]
        ),
        (
            "mild_cold_low",
            "I have mild runny nose and slight cough for 1 day. No fever.",
            ["LOW", "MEDIUM"]
        ),
        (
            "dengue_high",
            "fever for 3 days, severe joint pain, red spots on skin, feeling weak",
            ["HIGH", "EMERGENCY"]
        ),
    ]

    for name, symptoms, expected in agent_tests:
        try:
            run_test(name, symptoms, expected)

            passed += 1

            test_results[name] = {
                "status": "PASS",
                "expected": expected
            }

        except AssertionError as e:
            print(f"{e}")

            failed += 1

            test_results[name] = {
                "status": "FAIL",
                "expected": expected,
                "error": str(e)
            }

        except Exception as e:
            print(f"ERROR in {name}: {e}")

            failed += 1

            test_results[name] = {
                "status": "ERROR",
                "expected": expected,
                "error": str(e)
            }

    print(f"\n{'=' * 50}")
    print(f"Results: {passed} passed, {failed} failed")

    # Read RAGAS scores
    faithfulness_score = None

    ragas_path = Path("evaluation/ragas_scores.json")

    if ragas_path.exists():
        try:
            with open(ragas_path) as f:
                ragas_scores = json.load(f)

            faithfulness_score = ragas_scores.get("faithfulness")

            if "test_ragas_scores" in test_results:
                test_results["test_ragas_scores"][
                    "actual_faithfulness"
                ] = faithfulness_score

                test_results["test_ragas_scores"][
                    "faithfulness_threshold"
                ] = 0.60

        except Exception:
            pass

    # Save DeepEval results
    results = {
        "total_tests": passed + failed,
        "passed": passed,
        "failed": failed,
        "pass_rate": round(
            passed / (passed + failed),
            2
        ) if (passed + failed) > 0 else 0,
        "tests": test_results
    }

    output_path = Path("evaluation/deepeval_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"DeepEval results saved to {output_path}")

    if failed == 0:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
        sys.exit(1)