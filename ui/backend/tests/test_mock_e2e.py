"""End-to-end smoke test using the mock env. Verifies:
  - /api/healthz
  - /api/envs lists mock
  - /api/mock/partners lists partners
  - /api/mock/play/new starts a session
  - /api/mock/play/step advances the game (human turn + partner turn)
  - /api/mock/play/save persists a trajectory
  - save file ends up on disk under the tmp data root
"""
import json
import tempfile
from pathlib import Path

from ui.backend.app import create_app
from ui.backend.storage.file import build_backends as build_file_backends


def _app_with_tmpdata(tmp: Path):
    return create_app(backends=build_file_backends(str(tmp)))


def test_healthz():
    with tempfile.TemporaryDirectory() as tmp:
        client = _app_with_tmpdata(Path(tmp)).test_client()
        resp = client.get("/api/healthz")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert "mock" in data["envs"]


def test_envs_and_partners():
    with tempfile.TemporaryDirectory() as tmp:
        client = _app_with_tmpdata(Path(tmp)).test_client()

        # Mock is hidden by default; surfaces when include_hidden=true.
        resp = client.get("/api/envs")
        assert resp.status_code == 200
        envs = {e["env_name"] for e in resp.get_json()["envs"]}
        assert "mock" not in envs
        assert "hanabi" in envs

        resp = client.get("/api/envs?include_hidden=true")
        assert resp.status_code == 200
        envs_all = {e["env_name"] for e in resp.get_json()["envs"]}
        assert "mock" in envs_all

        # Routes under a hidden env still resolve (dev + test paths).
        resp = client.get("/api/mock/partners")
        assert resp.status_code == 200
        partners = resp.get_json()["partners"]
        keys = {p["key"] for p in partners}
        assert {"random", "echo"} == keys


def test_play_new_step_save():
    with tempfile.TemporaryDirectory() as tmp:
        data_root = Path(tmp)
        client = _app_with_tmpdata(data_root).test_client()

        # new game
        resp = client.post("/api/mock/play/new", json={
            "partner_key": "echo",
            "seed": 42,
        })
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        session_id = body["session_id"]
        assert body["state"]["turn"] == 0
        assert body["score"]["score"] == 0

        # step 1 - human plays 3
        resp = client.post("/api/mock/play/step", json={
            "session_id": session_id,
            "action": {"action": 3},
        })
        assert resp.status_code == 200, resp.get_json()
        body = resp.get_json()
        # Echo partner should replay our 3 on its turn
        assert body["state"]["last_partner_action"] == 3
        assert body["partner_acted"] is True

        # step 2 - human plays 3 again, should score because last_partner_action == 3
        resp = client.post("/api/mock/play/step", json={
            "session_id": session_id,
            "action": {"action": 3},
        })
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["score"]["score"] >= 1

        # save trajectory
        resp = client.post("/api/mock/play/save", json={
            "session_id": session_id,
            "agent_name": "Test",
        })
        assert resp.status_code == 200
        traj_id = resp.get_json()["trajectory_id"]
        saved = data_root / "trajectories" / "mock" / f"{traj_id}.json"
        assert saved.exists()
        loaded = json.loads(saved.read_text())
        assert loaded["agent_name"] == "Test"
        assert loaded["partner_key"] == "echo"
        assert len(loaded["history"]) >= 4  # 2 human + 2 partner actions


def test_bad_action_returns_400():
    with tempfile.TemporaryDirectory() as tmp:
        client = _app_with_tmpdata(Path(tmp)).test_client()
        new = client.post("/api/mock/play/new", json={
            "partner_key": "echo",
        }).get_json()

        # action outside 0-4 should be rejected
        resp = client.post("/api/mock/play/step", json={
            "session_id": new["session_id"],
            "action": {"action": 99},
        })
        assert resp.status_code == 400
        body = resp.get_json()
        assert body["error"] == "bad_request"


def test_unknown_env_returns_404():
    with tempfile.TemporaryDirectory() as tmp:
        client = _app_with_tmpdata(Path(tmp)).test_client()
        resp = client.get("/api/notarealenv/partners")
        assert resp.status_code == 404
        assert resp.get_json()["error"] == "not_found"


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as e:
            print(f"FAIL  {fn.__name__}: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"ERROR {fn.__name__}: {e!r}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    print(f"\nall {len(fns)} tests passed")
