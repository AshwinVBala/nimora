import json

import numpy as np

from nimora.data import ShardWriter, load_shards


def test_shard_writer_round_trip(tmp_path):
    writer = ShardWriter(tmp_path, shard_tokens=4)
    writer.add([1, 2, 3, 4, 5], [0, 1, 1, 0, 1])
    writer.flush()
    metadata = {"shards": writer.shards}
    (tmp_path / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    shards = load_shards(tmp_path)
    assert len(shards) == 2
    first_tokens = np.fromfile(shards[0].tokens_path, dtype=np.uint16)
    first_mask = np.fromfile(shards[0].mask_path, dtype=np.uint8)
    assert first_tokens.tolist() == [1, 2, 3, 4]
    assert first_mask.tolist() == [0, 1, 1, 0]

