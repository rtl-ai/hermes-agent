"""Regression: GitLab retries reuse the provider's immutable event UUID."""
from gateway.platforms.webhook import _header_delivery_id


def test_gitlab_event_id_precedes_non_native_transport_headers():
    headers = {"X-Gitlab-Event-UUID": "same-gitlab-event",
               "X-Request-ID": "first-transport-id",
               "svix-id": "svix-transport"}
    assert _header_delivery_id(headers) == "same-gitlab-event"
    headers["X-Request-ID"] = "new-transport-id"
    assert _header_delivery_id(headers) == "same-gitlab-event"


def test_empty_headers_get_fresh_distinct_id():
    first = _header_delivery_id({})
    second = _header_delivery_id({})
    assert first and second and first != second


def test_github_native_id_precedes_gitlab_id():
    assert _header_delivery_id({"X-GitHub-Delivery": "github-id",
                                "X-Gitlab-Event-UUID": "gitlab-id"}) == "github-id"
