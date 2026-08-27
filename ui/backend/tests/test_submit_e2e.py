"""End-to-end test of the submit + leaderboard flow on the mock env.
Verifies:
  - /api/<env>/egos lists builtin egos
  - /api/<env>/heldout lists held-out partners
  - /api/<env>/submit kicks off a job, returns job_id
  - /api/<env>/submit/status polls until done with progress updates
  - /api/<env>/submit/result returns per-partner breakdown + aggregate
  - /api/<env>/leaderboard shows the new entry
  - Two submissions sort correctly by aggregate score
"""
import os
import tempfile
import time
from pathlib import Path

os.environ["DISABLE_PURGE_THREAD"] = "1"

from ui.backend.app import create_app
from ui.backend.storage.file import build_backends


def _wait_for_job(client, env, job_id, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        resp = client.get(f"/api/{env}/submit/status/{job_id}")
        assert resp.status_code == 200
        body = resp.get_json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} did not finish in {timeout}s")


def _shutdown(app):
    """Tear down the JobManager so its executor stops touching the
    tempdir before TemporaryDirectory tries to rm it. Avoids spurious
    'Directory not empty' cleanup races."""
    jobs = app.extensions.get("jobs")
    if jobs:
        jobs.shutdown()


def test_egos_and_heldout_endpoints():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()
        egos = client.get("/api/mock/egos").get_json()["egos"]
        keys = {e["key"] for e in egos}
        assert keys == {"random", "constant_2", "mimic"}

        heldout = client.get("/api/mock/heldout?version=v1").get_json()
        assert heldout["version"] == "v1"
        partner_keys = {p["key"] for p in heldout["partners"]}
        assert partner_keys == {"random", "echo"}


def test_submit_flow_produces_leaderboard_entry():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()

        # Submit the constant-2 ego.
        resp = client.post("/api/mock/submit", json={
            "agent_name": "Always2",
            "ego_kind": "builtin",
            "builtin_key": "constant_2",
            "num_episodes": 4,
        })
        assert resp.status_code == 200, resp.get_json()
        job_id = resp.get_json()["job_id"]

        final = _wait_for_job(client, "mock", job_id)
        assert final["status"] == "done", final.get("error")

        # Progress should have ticked through both partners.
        assert final["progress"]["completed"] == 2
        assert final["progress"]["total"] == 2

        # Result endpoint returns the full evaluation payload.
        result = client.get(f"/api/mock/submit/result/{job_id}").get_json()
        assert "aggregate" in result
        assert "per_partner" in result
        assert len(result["per_partner"]) == 2
        keys = {p["key"] for p in result["per_partner"]}
        assert keys == {"random", "echo"}

        # Leaderboard lists the new entry.
        lb = client.get("/api/mock/leaderboard").get_json()
        assert lb["env"] == "mock"
        assert lb["version"] == "v1"
        assert len(lb["entries"]) == 1
        entry = lb["entries"][0]
        assert entry["agent_name"] == "Always2"
        assert entry["ego_kind"] == "builtin"
        assert entry["builtin_key"] == "constant_2"
        assert "aggregate" in entry
        assert len(entry["per_partner"]) == 2


def test_leaderboard_sorts_by_aggregate_descending():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()

        for ego_key in ["random", "mimic", "constant_2"]:
            resp = client.post("/api/mock/submit", json={
                "agent_name": ego_key,
                "ego_kind": "builtin",
                "builtin_key": ego_key,
                "num_episodes": 4,
            })
            job_id = resp.get_json()["job_id"]
            _wait_for_job(client, "mock", job_id)

        lb = client.get("/api/mock/leaderboard").get_json()
        scores = [e["aggregate_score"] for e in lb["entries"]]
        assert scores == sorted(scores, reverse=True), (
            f"leaderboard not sorted descending: {scores}"
        )
        # Mimic should dominate given the echo partner exists.
        assert lb["entries"][0]["agent_name"] == "mimic", [e["agent_name"] for e in lb["entries"]]


def test_bad_builtin_key_is_400():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()
        resp = client.post("/api/mock/submit", json={
            "agent_name": "X",
            "ego_kind": "builtin",
            "builtin_key": "no-such-ego",
            "num_episodes": 4,
        })
        assert resp.status_code == 400
        assert "no-such-ego" in (resp.get_json().get("detail") or "")


def test_upload_kind_returns_400_until_implemented():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()
        resp = client.post("/api/mock/submit", json={
            "agent_name": "X",
            "ego_kind": "upload",
            "num_episodes": 4,
        })
        assert resp.status_code == 400


def test_status_unknown_job_404():
    with tempfile.TemporaryDirectory() as tmp:
        client = create_app(backends=build_backends(tmp)).test_client()
        resp = client.get("/api/mock/submit/status/nonexistent")
        assert resp.status_code == 404


def test_status_wrong_env_404():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(backends=build_backends(tmp))
        client = app.test_client()
        try:
            resp = client.post("/api/mock/submit", json={
                "agent_name": "Y",
                "ego_kind": "builtin",
                "builtin_key": "random",
                "num_episodes": 2,
            })
            job_id = resp.get_json()["job_id"]
            # Wrong env in the URL should 404 even with a valid job_id.
            resp2 = client.get(f"/api/hanabi/submit/status/{job_id}")
            assert resp2.status_code == 404
            _wait_for_job(client, "mock", job_id)   # drain before tempdir cleanup
        finally:
            _shutdown(app)


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
