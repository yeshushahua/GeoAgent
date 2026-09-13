from collections import deque
from threading import Lock

from backend.app.agent.schemas import AgentRunTrace


class AgentTraceStore:
    def __init__(self, capacity: int = 50):
        self._items: deque[AgentRunTrace] = deque(maxlen=capacity)
        self._lock = Lock()

    def add(self, trace: AgentRunTrace) -> None:
        with self._lock:
            self._items.appendleft(trace)

    def list(self, limit: int = 20) -> list[AgentRunTrace]:
        with self._lock:
            return list(self._items)[:limit]
