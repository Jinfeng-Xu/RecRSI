from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset


WINDOWS = ("select", "audit", "confirm", "test")


@dataclass(frozen=True)
class UserSequence:
    user: int
    fit: Tuple[int, ...]
    select: Tuple[int, ...] = ()
    audit: Tuple[int, ...] = ()
    confirm: Tuple[int, ...] = ()
    test: Tuple[int, ...] = ()

    def history_before(self, window: str) -> Tuple[int, ...]:
        if window not in WINDOWS:
            raise KeyError(window)
        history = list(self.fit)
        for name in WINDOWS:
            if name == window:
                return tuple(history)
            history.extend(getattr(self, name))
        raise KeyError(window)


class SequenceCorpus:
    def __init__(self, users: Sequence[UserSequence], num_items: int, metadata: Optional[Mapping] = None):
        self.users = list(users)
        self.num_items = int(num_items)
        self.metadata = dict(metadata or {})
        if self.num_items < 2:
            raise ValueError("num_items includes padding and must be at least 2")
        for row in self.users:
            values = row.fit + row.select + row.audit + row.confirm + row.test
            if any(item <= 0 or item >= self.num_items for item in values):
                raise ValueError("Item ids must be in [1, num_items)")

    @classmethod
    def load(cls, path: str) -> "SequenceCorpus":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        users = [
            UserSequence(
                user=int(row["user"]),
                fit=tuple(map(int, row["fit"])),
                select=tuple(map(int, row.get("select", []))),
                audit=tuple(map(int, row.get("audit", []))),
                confirm=tuple(map(int, row.get("confirm", []))),
                test=tuple(map(int, row.get("test", []))),
            )
            for row in payload["users"]
        ]
        return cls(users, payload["num_items"], payload.get("metadata"))

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "rec-rsi-sequences-v1",
            "num_items": self.num_items,
            "metadata": self.metadata,
            "users": [
                {
                    "user": row.user,
                    "fit": list(row.fit),
                    "select": list(row.select),
                    "audit": list(row.audit),
                    "confirm": list(row.confirm),
                    "test": list(row.test),
                }
                for row in self.users
            ],
        }
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def training_sequences(self, augmented: Optional[Sequence[Sequence[int]]] = None) -> List[Tuple[int, ...]]:
        rows = [row.fit for row in self.users if len(row.fit) >= 2]
        if augmented:
            rows.extend(tuple(map(int, seq)) for seq in augmented if len(seq) >= 2)
        return rows

    def evaluation_examples(self, window: str) -> List[Tuple[int, Tuple[int, ...], int]]:
        examples: List[Tuple[int, Tuple[int, ...], int]] = []
        for row in self.users:
            history = list(row.history_before(window))
            for target in getattr(row, window):
                if history:
                    examples.append((row.user, tuple(history), target))
                history.append(target)
        return examples


def prepare_interactions(
    source: str,
    destination: str,
    user_column: str = "user_id",
    item_column: str = "item_id",
    time_column: str = "timestamp",
    min_events: int = 8,
) -> SequenceCorpus:
    """Map a chronological interaction CSV to the RecRSI split format.

    The final four interactions become select, audit, confirm, and test. Earlier
    interactions form the fit prefix. Item id 0 is reserved for padding.
    """
    grouped: Dict[str, List[Tuple[float, str]]] = {}
    with Path(source).open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {user_column, item_column, time_column}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV columns must include: " + ", ".join(sorted(required)))
        for order, row in enumerate(reader):
            raw_time = row[time_column]
            try:
                timestamp = float(raw_time)
            except ValueError:
                timestamp = float(order)
            grouped.setdefault(row[user_column], []).append((timestamp, row[item_column]))

    kept = {user: sorted(events, key=lambda pair: pair[0]) for user, events in grouped.items() if len(events) >= min_events}
    item_values = sorted({item for events in kept.values() for _, item in events})
    item_map = {item: index + 1 for index, item in enumerate(item_values)}
    users: List[UserSequence] = []
    for user_index, user in enumerate(sorted(kept)):
        sequence = [item_map[item] for _, item in kept[user]]
        users.append(
            UserSequence(
                user=user_index + 1,
                fit=tuple(sequence[:-4]),
                select=(sequence[-4],),
                audit=(sequence[-3],),
                confirm=(sequence[-2],),
                test=(sequence[-1],),
            )
        )
    corpus = SequenceCorpus(
        users,
        num_items=len(item_map) + 1,
        metadata={
            "source": str(Path(source).name),
            "user_column": user_column,
            "item_column": item_column,
            "time_column": time_column,
            "min_events": min_events,
            "item_map": item_map,
        },
    )
    corpus.save(destination)
    return corpus


class NextItemDataset(Dataset):
    def __init__(self, sequences: Sequence[Sequence[int]], max_length: int):
        self.max_length = int(max_length)
        self.examples: List[Tuple[Tuple[int, ...], int]] = []
        for sequence in sequences:
            seq = tuple(map(int, sequence))
            for index in range(1, len(seq)):
                history = seq[max(0, index - self.max_length):index]
                self.examples.append((history, seq[index]))
        if not self.examples:
            raise ValueError("No next-item examples can be formed")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> Tuple[Tuple[int, ...], int]:
        return self.examples[index]


def collate_next_item(batch: Sequence[Tuple[Sequence[int], int]]) -> Dict[str, torch.Tensor]:
    lengths = torch.tensor([len(history) for history, _ in batch], dtype=torch.long)
    width = int(lengths.max())
    items = torch.zeros(len(batch), width, dtype=torch.long)
    for row, (history, _) in enumerate(batch):
        items[row, : len(history)] = torch.tensor(history, dtype=torch.long)
    targets = torch.tensor([target for _, target in batch], dtype=torch.long)
    return {"items": items, "lengths": lengths, "targets": targets}


def collate_evaluation(batch: Sequence[Tuple[int, Sequence[int], int]], max_length: int) -> Dict[str, torch.Tensor]:
    clipped = [(user, tuple(history[-max_length:]), target) for user, history, target in batch]
    lengths = torch.tensor([len(history) for _, history, _ in clipped], dtype=torch.long)
    width = int(lengths.max())
    items = torch.zeros(len(batch), width, dtype=torch.long)
    full_histories: List[Tuple[int, ...]] = []
    for row, (_, history, _) in enumerate(clipped):
        items[row, : len(history)] = torch.tensor(history, dtype=torch.long)
        full_histories.append(history)
    return {
        "users": torch.tensor([user for user, _, _ in clipped], dtype=torch.long),
        "items": items,
        "lengths": lengths,
        "targets": torch.tensor([target for _, _, target in clipped], dtype=torch.long),
        "histories": full_histories,
    }


def batched(values: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(values), batch_size):
        yield values[start:start + batch_size]

