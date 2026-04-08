from __future__ import annotations

import json
from typing import Any, Dict, List
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class PipePulseClient:
    def __init__(self, base_url: str = "http://localhost:7860", timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(self, method: str, path: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any] | List[Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Content-Type": "application/json"} if data is not None else {}
        request = Request(url=url, data=data, method=method, headers=headers)

        try:
            with urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
                if not body:
                    return {}
                return json.loads(body)
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {url} failed with {exc.code}: {detail}") from exc
        except URLError as exc:
            raise RuntimeError(f"{method} {url} failed: {exc.reason}") from exc

    def tasks(self) -> List[Dict[str, Any]]:
        data = self._request("GET", "/tasks")
        if not isinstance(data, list):
            raise RuntimeError("Invalid /tasks response shape")
        return data

    def reset(self, task_id: str) -> Dict[str, Any]:
        data = self._request("POST", "/reset", {"task_id": task_id})
        if not isinstance(data, dict):
            raise RuntimeError("Invalid /reset response shape")
        return data

    def step(self, action: Dict[str, Any]) -> Dict[str, Any]:
        data = self._request("POST", "/step", action)
        if not isinstance(data, dict):
            raise RuntimeError("Invalid /step response shape")
        return data

    def state(self) -> Dict[str, Any]:
        data = self._request("GET", "/state")
        if not isinstance(data, dict):
            raise RuntimeError("Invalid /state response shape")
        return data
