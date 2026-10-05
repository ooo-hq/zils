"""Contract tests: known distributions provide independent answer oracles."""

import importlib
import importlib.util
import json
import unittest


def request():
    return {
        "state": {"query": "USB-C cable"},
        "model": "zils-shared",
        "questions": {
            "c": {
                "type": "choice",
                "criteria": {"10": None, "2": {"meaning": "second"}, "other": "none"},
            },
            "n": {"type": "noul", "instructions": None},
            "s": {"type": "score", "criteria": ["none", {"match": "partial"}, ["exact"]]},
        },
    }


class DecisionTest(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(
            importlib.util.find_spec("zils.decisions"), "decision contract is missing"
        )
        return importlib.import_module("zils.decisions")

    def test_mixed_answers_and_structured_legend(self):
        d = self.module()
        body = d.validate_request(request())
        response = d.make_response(
            "release-1",
            body,
            {
                "c": {"probabilities": {"10": 0.6, "2": 0.3, "other": 0.1}, "input_tokens": 10},
                "n": {"probabilities": {"true": 0.8, "false": 0.2}, "input_tokens": 20},
                "s": {"probabilities": {"0": 0.0, "1": 0.5, "2": 0.5}, "input_tokens": 30},
            },
        )
        self.assertEqual(response["model"], "release-1")
        self.assertEqual(response["answers"]["c"]["choice"], "10")
        self.assertAlmostEqual(response["answers"]["c"]["confidence"], 0.4)
        self.assertEqual(response["answers"]["n"], {"type": "noul", "noul": 0.8})
        self.assertEqual(response["answers"]["s"]["score"], 1.5)
        self.assertAlmostEqual(response["answers"]["s"]["confidence"], 0.25)
        self.assertEqual(
            response["answers"]["s"]["legend"],
            {"0": "none", "1": {"match": "partial"}, "2": ["exact"]},
        )
        self.assertEqual(response["usage"], {"input_tokens": 60, "output_tokens": 0})

    def test_large_choice_and_many_questions(self):
        d = self.module()
        body = request()
        body["questions"] = {
            str(i): {"type": "choice", "criteria": {str(j): None for j in range(255)}}
            for i in range(13)
        }
        self.assertEqual(len(d.validate_request(body)["questions"]), 13)
        options = d.option_descriptions(body["questions"]["0"])
        self.assertEqual(list(options)[:3], ["0", "1", "2"])
        body["questions"]["0"]["criteria"]["256"] = "extra"
        with self.assertRaises(d.DecisionError) as err:
            d.validate_request(body)
        self.assertEqual(err.exception.status, 422)

    def test_json_and_schema_rejections(self):
        d = self.module()
        for raw in (b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}', b"\xff", b"{"):
            with self.subTest(raw=raw), self.assertRaises(d.DecisionError) as err:
                d.decode_body(raw)
            self.assertEqual(err.exception.status, 400)
        with self.assertRaises(d.DecisionError) as err:
            d.decode_body(b" " * (d.MAX_BODY + 1))
        self.assertEqual(err.exception.status, 413)
        for value in (True, 1, None, "text", []):
            with self.subTest(value=value), self.assertRaises(d.DecisionError):
                d.validate_request(value)
        for value in (float("nan"), float("inf"), "\ud800"):
            body = request()
            body["state"] = {"bad": value}
            with self.subTest(value=repr(value)), self.assertRaises(d.DecisionError):
                d.validate_request(body)
        body = request()
        nested = []
        for _ in range(33):
            nested = [nested]
        body["state"] = nested
        with self.assertRaises(d.DecisionError):
            d.validate_request(body)

    def test_bad_backend_output_never_becomes_an_answer(self):
        d = self.module()
        body = {
            "state": "private",
            "model": "m",
            "questions": {"q": {"type": "choice", "criteria": {"yes": None, "no": None}}},
        }
        for probs in (
            {"yes": True, "no": 0.0},
            {"yes": 0.7, "no": 0.4},
            {"yes": float("nan"), "no": 0.5},
            {"yes": 1.0},
            {"yes": 1.0, "no": 0.0, "extra": 0.0},
        ):
            with self.subTest(probs=probs), self.assertRaises(d.DecisionError) as err:
                d.make_response("m", body, {"q": {"probabilities": probs, "input_tokens": 3}})
            self.assertEqual(err.exception.status, 502)
            self.assertNotIn("private", str(err.exception))
        with self.assertRaises(d.DecisionError):
            d.make_response("m", body, {})

    def test_ties_order_rounding_and_noul_defaults(self):
        d = self.module()
        body = {
            "state": [],
            "model": "m",
            "questions": {"q": {"type": "choice", "criteria": {"z": "first", "a": "second"}}},
        }
        answer = d.make_response(
            "m", body, {"q": {"probabilities": {"a": 0.5, "z": 0.5}, "input_tokens": 0}}
        )["answers"]["q"]
        self.assertEqual(answer["choice"], "z")
        self.assertEqual(answer["confidence"], 0)
        self.assertEqual(list(d.option_descriptions({"type": "noul"})), ["true", "false"])
        self.assertEqual(d.render({"key": "value"}), '{"key":"value"}')
        self.assertEqual(d.decode_body(json.dumps(request()).encode()), request())


if __name__ == "__main__":
    unittest.main()
