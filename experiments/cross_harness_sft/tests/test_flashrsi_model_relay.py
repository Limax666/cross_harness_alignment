import threading

import pytest

from flashrsi_model_relay import create_model_relay, relay_request


def test_local_relay_authenticates_and_preserves_one_nonstreaming_response(tmp_path):
    token = "test-relay-token-long-enough-32-characters"
    seen = []
    response = {"choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}]}
    server = create_model_relay(str(tmp_path / "model.sock"), token, "qwen4b",
                                lambda request: seen.append(request) or response)
    thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
    try:
        request = {"model": "qwen4b", "messages": [], "stream": False}
        assert relay_request(str(tmp_path / "model.sock"), token, request) == response
        assert seen == [request]
        with pytest.raises(ValueError):
            relay_request(str(tmp_path / "model.sock"), "wrong-token-long-enough-32-chars", request)
        with pytest.raises(ValueError):
            relay_request(str(tmp_path / "model.sock"), token, {**request, "stream": True})
        assert seen == [request]
    finally:
        server.shutdown(); thread.join(timeout=5); server.server_close()


def test_relay_refuses_shared_directory_and_existing_socket_path(tmp_path):
    token = "test-relay-token-long-enough-32-characters"
    shared = tmp_path / "shared"; shared.mkdir(mode=0o777); shared.chmod(0o777)
    with pytest.raises(ValueError):
        create_model_relay(str(shared / "socket"), token, "qwen4b", lambda _: {})
    existing = tmp_path / "existing"; existing.write_text("keep")
    with pytest.raises(ValueError):
        create_model_relay(str(existing), token, "qwen4b", lambda _: {})
    assert existing.read_text() == "keep"
