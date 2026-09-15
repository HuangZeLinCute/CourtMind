import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.services import llm_service


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class _StreamResponse(_Response):
    def __iter__(self):
        for chunk in self.payload:
            yield ("data: " + (chunk if isinstance(chunk, str) else json.dumps(chunk)) + "\n").encode("utf-8")


class LlmServiceTests(unittest.TestCase):
    def _result(self, output_dir):
        return {
            "status": "completed",
            "output_dir": str(output_dir),
            "metadata": {
                "analysis": {"name": "Final"},
                "video": {"duration_sec": 20},
                "models": {"shuttlecock": "ensemble"},
                "private_extra": "must-not-be-sent",
            },
            "report": {
                "summary": {"rally_count": 2, "total_hits": 7},
                "rallies": [{"rally_id": 1, "hit_count": 4, "unknown": "drop"}],
                "limitations": "estimated",
                "private_extra": "must-not-be-sent",
            },
        }

    def test_context_is_grounded_and_whitelisted(self):
        with tempfile.TemporaryDirectory() as root, patch.object(
            llm_service.job_service, "job_result", return_value=self._result(root)
        ):
            context = llm_service.build_match_context("job1")
        self.assertEqual(context["report"]["summary"]["total_hits"], 7)
        self.assertNotIn("private_extra", context)
        self.assertNotIn("private_extra", context["report"])
        self.assertNotIn("unknown", context["report"]["rallies"][0])

    def test_chat_sends_match_data_and_persists_only_visible_messages(self):
        captured = {}

        def fake_open(request, timeout):
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return _Response({"choices": [{"message": {"content": "共 7 次 [summary.total_hits]"}}]})

        with tempfile.TemporaryDirectory() as root, patch.object(
            llm_service.job_service, "job_result", return_value=self._result(root)
        ), patch.dict(os.environ, {"AGNES_API_KEY": "test-only"}), patch.object(
            llm_service, "urlopen", side_effect=fake_open
        ):
            response = llm_service.chat("job1", "agnes", "多少次击球？")
            saved = json.loads((Path(root) / "chat_history.json").read_text(encoding="utf-8"))

        self.assertEqual(response["model"], "agnes-2.5-flash")
        serialized = json.dumps(captured["payload"], ensure_ascii=False)
        self.assertIn("MATCH_DATA_BEGIN", serialized)
        self.assertIn("total_hits", serialized)
        self.assertNotIn("test-only", serialized)
        self.assertEqual([item["role"] for item in saved["agnes"]], ["user", "assistant"])

    def test_missing_key_fails_before_network_call(self):
        with patch.dict(os.environ, {"AGNES_API_KEY": ""}, clear=False):
            with self.assertRaisesRegex(RuntimeError, "not configured"):
                llm_service.chat("unused", "agnes", "hello")

    def test_stream_yields_before_completion_and_retains_full_history(self):
        packets = [
            {"choices": [{"delta": {"role": "assistant"}}]},
            {"choices": [{"delta": {"content": "检测到"}}]},
            {"choices": [{"delta": {"content": "7 次击球。"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            "[DONE]",
        ]
        with tempfile.TemporaryDirectory() as root, patch.object(
            llm_service.job_service, "job_result", return_value=self._result(root)
        ), patch.dict(os.environ, {"AGNES_API_KEY": "test-only"}), patch.object(
            llm_service, "urlopen", return_value=_StreamResponse(packets)
        ) as mock_open:
            previous = [{"role": "user" if i % 2 == 0 else "assistant", "content": str(i), "created_at": i} for i in range(20)]
            llm_service._save_history(Path(root), "agnes", previous)
            events = llm_service.stream_chat("job1", "agnes", "多少次？")
            self.assertEqual(next(events), {"type": "delta", "content": "检测到"})
            self.assertEqual(len(llm_service.get_history("job1", "agnes")), 20)
            rest = list(events)
            self.assertEqual(rest[-1]["type"], "done")
            self.assertEqual(rest[-1]["message"]["content"], "检测到7 次击球。")
            self.assertEqual(len(llm_service.get_history("job1", "agnes")), 22)
            payload = json.loads(mock_open.call_args.args[0].data)
            self.assertTrue(payload["stream"])
            self.assertEqual(len(payload["messages"]), 15)

    def test_interrupted_stream_does_not_save_partial_answer(self):
        packets = [{"choices": [{"delta": {"content": "partial"}}]}]
        with tempfile.TemporaryDirectory() as root, patch.object(
            llm_service.job_service, "job_result", return_value=self._result(root)
        ), patch.dict(os.environ, {"DEEPSEEK_API_KEY": "test-only"}), patch.object(
            llm_service, "urlopen", return_value=_StreamResponse(packets)
        ):
            events = list(llm_service.stream_chat("job1", "deepseek", "question"))
            self.assertEqual(events[0]["type"], "delta")
            self.assertEqual(events[-1]["type"], "error")
            self.assertFalse((Path(root) / "chat_history.json").exists())

    def test_cancel_closes_upstream_without_saving(self):
        packets = [{"choices": [{"delta": {"content": "first"}}]}, "[DONE]"]
        with tempfile.TemporaryDirectory() as root, patch.object(
            llm_service.job_service, "job_result", return_value=self._result(root)
        ), patch.dict(os.environ, {"AGNES_API_KEY": "test-only"}), patch.object(
            llm_service, "urlopen", return_value=_StreamResponse(packets)
        ):
            events = llm_service.stream_chat("job1", "agnes", "question")
            self.assertEqual(next(events)["type"], "delta")
            events.close()
            self.assertFalse((Path(root) / "chat_history.json").exists())


if __name__ == "__main__":
    unittest.main()
