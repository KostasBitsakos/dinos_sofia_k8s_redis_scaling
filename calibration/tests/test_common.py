import common

SAMPLE = """\
aaa 10.0.0.1:6379@16379 myself,master - 0 0 1 connected 0-5460
bbb 10.0.0.2:6379@16379 master - 0 1 2 connected 5461-10922 [5500->-ccc]
ccc 10.0.0.3:6379@16379 master,fail - 0 1 3 connected 10923-16383
ddd 10.0.0.4:6379@16379 master - 0 1 0 connected
"""


def test_parse_and_coverage():
    nodes = common.parse_cluster_nodes(SAMPLE)
    assert len(nodes) == 4 and nodes[1]["slots"] == [(5461, 10922)]  # migrating marker skipped
    assert nodes[2]["flags"] == ["master", "fail"] and nodes[3]["slots"] == []
    assert common.slots_covered(nodes) == 16384


def test_event_and_metric_files(tmp_path, monkeypatch):
    monkeypatch.setenv("QRL_RUN_DIR", str(tmp_path))
    common.event("x", a=1)
    common.metric(0, v=1.5)
    common.write_progress(1, 2)
    assert '"event": "x"' in (tmp_path / "events.jsonl").read_text()
    assert (tmp_path / "metrics.jsonl").read_text().strip() == '{"step": 0, "v": 1.5}'
