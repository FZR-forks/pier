from pathlib import Path


def test_first_file():
    assert Path("/app/first.txt").read_text().strip() == "FIRST"


def test_second_file():
    assert Path("/app/second.txt").read_text().strip() == "SECOND"


def test_report_file():
    assert Path("/app/report.txt").read_text().strip() == "DELEGATED"
