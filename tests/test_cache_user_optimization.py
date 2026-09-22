# -*- coding: utf-8 -*-
from src.cache.cache_service import CacheService
from src.utils.config import get_config


def test_cache_ttl_is_one_day():
    assert get_config().cache.ttl_seconds == 86400


def test_cache_key_is_user_scoped_but_session_independent():
    first = CacheService._make_key("第三季度营收是多少？", user_id="user_a")
    second = CacheService._make_key("第三季度营收是多少？", user_id="user_a")
    other_user = CacheService._make_key("第三季度营收是多少？", user_id="user_b")

    assert first == second
    assert first != other_user


def test_memory_cache_matches_same_user_across_sessions():
    cache = CacheService()
    cache.set(
        "年度动销目标是多少？",
        {"answer": "answer-a"},
        user_id="user_a",
    )

    assert cache.get(
        "年度动销目标是多少？",
        user_id="user_a",
    )["answer"] == "answer-a"
    assert cache.get(
        "年度动销目标是多少？",
        user_id="user_b",
    ) is None


def test_regenerated_answer_overwrites_user_cache():
    cache = CacheService()
    cache.set(
        "2026年订单目标是多少？",
        {"answer": "old-answer"},
        user_id="user_a",
    )
    cache.set(
        "2026年订单目标是多少？",
        {"answer": "new-answer"},
        user_id="user_a",
    )

    assert cache.get(
        "2026年订单目标是多少？",
        user_id="user_a",
    )["answer"] == "new-answer"
