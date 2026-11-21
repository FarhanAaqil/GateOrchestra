"""Limits shared by several worker processes through the database."""

from __future__ import annotations

import multiprocessing as mp
from pathlib import Path

import pytest

from assistant.sandbox import SharedSandboxLimits
from assistant.security import RateLimiter
from assistant.shared_limits import HostedAwareLimiter, Leases, SharedRateLimiter
from assistant.store import Store


def two_workers(tmp_path: Path) -> tuple[Store, Store]:
    """Two independent Store objects on one file, as two worker processes would have."""
    path = tmp_path / "shared.db"
    return Store(path), Store(path)


def test_hits_from_every_worker_count_towards_one_limit(tmp_path: Path) -> None:
    a, b = two_workers(tmp_path)
    first = SharedRateLimiter("chat", 3, window=60, store=lambda: a)
    second = SharedRateLimiter("chat", 3, window=60, store=lambda: b)
    assert first.check("ada", now=100) == 0
    assert second.check("ada", now=101) == 0
    assert first.check("ada", now=102) == 0
    assert second.check("ada", now=103) == pytest.approx(57)  # the 4th hit, from either worker
    assert first.check("bob", now=103) == 0  # per key
    assert second.check("ada", now=161) == 0  # the window slid
    assert SharedRateLimiter("off", 0, store=lambda: a).check("x") == 0


def test_leases_cap_concurrency_and_expire(tmp_path: Path) -> None:
    a, b = two_workers(tmp_path)
    one, two = Leases(lambda: a), Leases(lambda: b)
    first = one.acquire("chat:ada", limit=2, ttl=60, now=0)
    second = two.acquire("chat:ada", limit=2, ttl=60, now=1)
    assert first and second
    assert one.acquire("chat:ada", limit=2, ttl=60, now=2) is None
    assert one.held("chat:ada", now=2) == 2
    two.release(first)
    assert one.acquire("chat:ada", limit=2, ttl=60, now=3) is not None
    # A worker that died never releases; the lease frees itself when it expires.
    assert one.acquire("chat:ada", limit=2, ttl=60, now=70) is not None


def test_acquire_all_is_all_or_nothing(tmp_path: Path) -> None:
    leases = Leases(lambda: Store(tmp_path / "s.db"))
    held = leases.acquire("global", limit=1, ttl=60)
    assert held
    assert leases.acquire_all([("user:ada", 1), ("global", 1)], ttl=60) is None
    assert leases.held("user:ada") == 0  # the first lease was handed back
    leases.release(held)
    assert leases.acquire_all([("user:ada", 1), ("global", 1)], ttl=60)


def _grab(path: str, out: mp.Queue) -> None:
    from assistant.shared_limits import Leases as WorkerLeases
    from assistant.store import Store as WorkerStore

    store = WorkerStore(Path(path))
    out.put(WorkerLeases(lambda: store).acquire("only-one", limit=1, ttl=60) is not None)


def test_separate_processes_cannot_both_take_the_last_slot(tmp_path: Path) -> None:
    path = tmp_path / "race.db"
    Store(path)
    ctx = mp.get_context("spawn")
    out: mp.Queue = ctx.Queue()
    procs = [ctx.Process(target=_grab, args=(str(path), out)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
    results = [out.get(timeout=5) for _ in procs]
    assert results.count(True) == 1 and results.count(False) == 3


def test_single_tenant_mode_keeps_the_in_memory_limiter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = Store(tmp_path / "s.db")
    limiter = HostedAwareLimiter(RateLimiter(1), SharedRateLimiter("addr", 1, store=lambda: store))
    monkeypatch.delenv("GATE_HOSTED", raising=False)
    assert limiter.check("1.1.1.1", now=0) == 0 and limiter.check("1.1.1.1", now=1) > 0
    monkeypatch.setenv("GATE_HOSTED", "true")
    assert limiter.check("2.2.2.2") == 0 and limiter.check("2.2.2.2") > 0
    assert Leases(lambda: store).held("anything") == 0


def test_sandbox_rationing_is_shared(tmp_path: Path) -> None:
    a, b = two_workers(tmp_path)
    worker_one = SharedSandboxLimits(per_user=1, total=1, leases=Leases(lambda: a))
    worker_two = SharedSandboxLimits(per_user=1, total=1, leases=Leases(lambda: b))
    assert worker_one.acquire("ada") is None
    assert "already have code running" in (worker_two.acquire("ada") or "")
    assert "busy" in (worker_two.acquire("bob", wait=0.3) or "")
    worker_one.release("ada")
    assert worker_two.acquire("bob") is None
    worker_two.release("bob")
