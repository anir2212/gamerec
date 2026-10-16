from fastapi.testclient import TestClient

from recstack.serve import app, trained_models


def test_api_excludes_owned_and_cold_user_uses_genre_fallback():
    client = TestClient(app)
    assert client.get("/health").json()["data"] == "seeded synthetic"
    known = client.get("/recommend", params={"user_id": "u015", "k": 10})
    assert known.status_code == 200
    items = [item["item_id"] for item in known.json()["items"]]
    assert not set(items).intersection(trained_models().history["u015"])
    assert len(items) == 10
    new = client.get("/recommend", params={"user_id": "new-user", "k": 5,
                                                "genres": "racing"})
    assert new.status_code == 200
    assert not new.json()["known_user"]
    assert new.json()["items"][0]["genre"] == "racing"
    assert len({item["item_id"] for item in new.json()["items"]}) == 5
    assert client.get("/recommend", params={"user_id": "u015", "method": "unknown"}).status_code == 400
