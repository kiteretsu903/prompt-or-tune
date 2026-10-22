import csv
import hashlib

from fireworks_lab.banking77_prepare import read_source_csv, sha256_file


def test_read_source_csv_preserves_embedded_newlines_and_hash(tmp_path) -> None:
    source = tmp_path / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "category"))
        writer.writeheader()
        writer.writerow(
            {
                "text": "a quoted query\nwith a newline",
                "category": "activate_my_card",
            }
        )

    assert read_source_csv(source) == [
        {
            "text": "a quoted query\nwith a newline",
            "category": "activate_my_card",
        }
    ]
    assert sha256_file(source) == hashlib.sha256(source.read_bytes()).hexdigest()
