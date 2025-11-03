"""
Blockchain Logger
-----------------
A minimal local blockchain to store IDS alerts as tamper-evident records.
Blocks are appended to a JSON file with fields: {index, timestamp, data, prev_hash, hash}.
Includes verify_chain() to validate integrity.

Usage:
    from src.blockchain_logger import BlockchainLogger
    logger = BlockchainLogger(chain_path="results/alerts_chain.json")
    logger.append_alert({"alert": "possible_scan", "score": 0.97})
    assert logger.verify_chain()
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from typing import Any, Dict, List


class BlockchainLogger:
    def __init__(self, chain_path: str):
        self.chain_path = chain_path
        os.makedirs(os.path.dirname(chain_path), exist_ok=True)
        if not os.path.exists(chain_path):
            # Create genesis block
            genesis = self._create_block(index=0, data={"genesis": True}, prev_hash="0")
            self._write_chain([genesis])

    def _read_chain(self) -> List[Dict[str, Any]]:
        with open(self.chain_path, "r") as f:
            return json.load(f)

    def _write_chain(self, chain: List[Dict[str, Any]]):
        with open(self.chain_path, "w") as f:
            json.dump(chain, f, indent=2)

    @staticmethod
    def _hash_block(block: Dict[str, Any]) -> str:
        block_copy = {k: block[k] for k in sorted(block.keys()) if k != "hash"}
        payload = json.dumps(block_copy, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _create_block(self, index: int, data: Dict[str, Any], prev_hash: str) -> Dict[str, Any]:
        block = {
            "index": index,
            "timestamp": int(time.time()),
            "data": data,
            "prev_hash": prev_hash,
        }
        block["hash"] = self._hash_block(block)
        return block

    def append_alert(self, data: Dict[str, Any]) -> Dict[str, Any]:
        chain = self._read_chain()
        last = chain[-1]
        block = self._create_block(index=last["index"] + 1, data=data, prev_hash=last["hash"])
        chain.append(block)
        self._write_chain(chain)
        return block

    def verify_chain(self) -> bool:
        chain = self._read_chain()
        for i, block in enumerate(chain):
            if i == 0:
                # genesis
                if block.get("prev_hash") != "0":
                    return False
                if block.get("hash") != self._hash_block(block):
                    return False
            else:
                prev = chain[i - 1]
                if block.get("prev_hash") != prev.get("hash"):
                    return False
                if block.get("hash") != self._hash_block(block):
                    return False
        return True


__all__ = ["BlockchainLogger"]
