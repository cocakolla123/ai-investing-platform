"""Token-bucket unit tests (dynamo/data/ratelimit.py)."""

from __future__ import annotations

import pytest

from dynamo.data.ratelimit import TokenBucket


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_bucket(rate: float, capacity: float) -> tuple[TokenBucket, FakeClock]:
    clock = FakeClock()
    return TokenBucket(rate, capacity, clock=clock, sleep=clock.sleep), clock


def test_try_acquire_drains_and_refills() -> None:
    bucket, clock = make_bucket(rate=1.0, capacity=2.0)
    assert bucket.try_acquire()
    assert bucket.try_acquire()
    assert not bucket.try_acquire()  # empty
    clock.now += 1.0  # refill 1 token
    assert bucket.try_acquire()
    assert not bucket.try_acquire()


def test_acquire_blocks_until_refill() -> None:
    bucket, clock = make_bucket(rate=2.0, capacity=1.0)
    bucket.acquire()
    bucket.acquire()  # must sleep ~0.5s for 1 token at 2/s
    assert clock.slept and abs(clock.slept[0] - 0.5) < 1e-9


def test_acquire_more_than_capacity_rejected() -> None:
    bucket, _ = make_bucket(rate=1.0, capacity=1.0)
    with pytest.raises(ValueError):
        bucket.acquire(2.0)


def test_invalid_construction() -> None:
    with pytest.raises(ValueError):
        TokenBucket(0.0, 1.0)
    with pytest.raises(ValueError):
        TokenBucket(1.0, 0.0)
