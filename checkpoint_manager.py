"""
断点续执行管理器：记录并持久化清单执行的进度状态，
支持任意中断后从上次位置恢复执行，同时提供超时保护和健康监控。
"""
import json
import os
import time
import threading
import signal
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field, asdict

from config import BASE_DIR

CHECKPOINT_DIR = BASE_DIR / ".checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "execution_state.json"
HEARTBEAT_FILE = CHECKPOINT_DIR / "heartbeat.json"
MAX_HEARTBEAT_AGE = 30
DEFAULT_TIMEOUT = 10.0


@dataclass
class StepState:
    step_id: str
    step_name: str
    status: str = "pending"
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    attempts: int = 0
    last_error: Optional[str] = None
    result: Optional[dict] = None


@dataclass
class ExecutionState:
    execution_id: str
    task_type: str
    task_params: dict = field(default_factory=dict)
    steps: list[StepState] = field(default_factory=list)
    current_step: int = 0
    total_steps: int = 0
    status: str = "idle"
    created_at: str = ""
    updated_at: str = ""
    stats: dict = field(default_factory=lambda: {
        "total_processed": 0,
        "total_succeeded": 0,
        "total_failed": 0,
        "total_skipped": 0,
    })


class TimeoutError(Exception):
    pass


class CheckpointManager:
    def __init__(self):
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._state: Optional[ExecutionState] = None
        self._heartbeat_interval = 5
        self._heartbeat_thread: Optional[threading.Thread] = None
        self._stop_heartbeat = threading.Event()
        self._timeout_callbacks: list[Callable] = []

    def _save_state(self):
        with self._lock:
            if self._state is None:
                return
            self._state.updated_at = datetime.now().isoformat()
            data = {
                "execution_id": self._state.execution_id,
                "task_type": self._state.task_type,
                "task_params": self._state.task_params,
                "steps": [asdict(s) for s in self._state.steps],
                "current_step": self._state.current_step,
                "total_steps": self._state.total_steps,
                "status": self._state.status,
                "created_at": self._state.created_at,
                "updated_at": self._state.updated_at,
                "stats": self._state.stats,
            }
            with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def init_task(self, task_type: str, task_params: dict,
                  step_definitions: list[dict]) -> str:
        """初始化新任务，返回 execution_id"""
        with self._lock:
            exec_id = f"{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
            steps = [
                StepState(
                    step_id=s["id"],
                    step_name=s["name"],
                )
                for s in step_definitions
            ]
            self._state = ExecutionState(
                execution_id=exec_id,
                task_type=task_type,
                task_params=task_params,
                steps=steps,
                total_steps=len(steps),
                status="running",
                created_at=datetime.now().isoformat(),
                updated_at=datetime.now().isoformat(),
            )
            self._save_state()
            self._start_heartbeat()
            return exec_id

    def load_state(self) -> Optional[dict]:
        """加载最近的持久化状态"""
        if not CHECKPOINT_FILE.exists():
            return None
        try:
            with open(CHECKPOINT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            steps = [StepState(**s) for s in data.get("steps", [])]
            self._state = ExecutionState(
                execution_id=data["execution_id"],
                task_type=data["task_type"],
                task_params=data.get("task_params", {}),
                steps=steps,
                current_step=data.get("current_step", 0),
                total_steps=data.get("total_steps", 0),
                status=data.get("status", "idle"),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
                stats=data.get("stats", {
                    "total_processed": 0,
                    "total_succeeded": 0,
                    "total_failed": 0,
                    "total_skipped": 0,
                }),
            )
            return data
        except Exception as e:
            return None

    def get_pending_steps(self) -> list[StepState]:
        """获取所有待执行的步骤（支持断点续执行）"""
        if self._state is None:
            return []
        return [
            s for i, s in enumerate(self._state.steps)
            if i >= self._state.current_step and s.status in ("pending", "failed")
        ]

    def mark_step_start(self, step_index: int):
        with self._lock:
            if self._state and step_index < len(self._state.steps):
                step = self._state.steps[step_index]
                step.status = "running"
                step.started_at = datetime.now().isoformat()
                step.attempts += 1
                step.last_error = None
                self._state.current_step = step_index
                self._save_state()

    def mark_step_success(self, step_index: int, result: dict = None):
        with self._lock:
            if self._state and step_index < len(self._state.steps):
                step = self._state.steps[step_index]
                step.status = "completed"
                step.completed_at = datetime.now().isoformat()
                step.result = result
                self._state.stats["total_succeeded"] += 1
                self._state.stats["total_processed"] += 1
                self._save_state()

    def mark_step_failed(self, step_index: int, error: str):
        with self._lock:
            if self._state and step_index < len(self._state.steps):
                step = self._state.steps[step_index]
                step.status = "failed"
                step.last_error = error
                step.completed_at = datetime.now().isoformat()
                self._state.stats["total_failed"] += 1
                self._state.stats["total_processed"] += 1
                self._save_state()

    def mark_step_skipped(self, step_index: int, reason: str = ""):
        with self._lock:
            if self._state and step_index < len(self._state.steps):
                step = self._state.steps[step_index]
                step.status = "skipped"
                step.last_error = reason
                self._state.stats["total_skipped"] += 1
                self._state.stats["total_processed"] += 1
                self._save_state()

    def complete_task(self, status: str = "completed"):
        with self._lock:
            if self._state:
                self._state.status = status
                self._state.updated_at = datetime.now().isoformat()
                self._save_state()
        self._write_heartbeat()
        self._stop_heartbeat.set()

    def get_progress(self) -> dict:
        if self._state is None:
            return {"progress": 0, "status": "idle", "details": {}}
        completed = sum(1 for s in self._state.steps if s.status == "completed")
        return {
            "execution_id": self._state.execution_id,
            "task_type": self._state.task_type,
            "progress": completed / max(self._state.total_steps, 1),
            "current_step": self._state.current_step,
            "total_steps": self._state.total_steps,
            "status": self._state.status,
            "stats": self._state.stats,
            "step_details": [
                {"id": s.step_id, "name": s.step_name, "status": s.status,
                 "attempts": s.attempts, "error": s.last_error}
                for s in self._state.steps
            ],
        }

    def can_resume(self) -> bool:
        """检查是否存在可续点的未完成任务"""
        if not CHECKPOINT_FILE.exists():
            return False
        data = self.load_state()
        if data is None:
            return False
        return data.get("status") in ("running", "paused")

    # ─── 超时保护 ───────────────────────────────

    def execute_with_timeout(self, func: Callable, timeout: float = None,
                             *args, **kwargs):
        """在超时保护下执行函数，超时抛出 TimeoutError"""
        timeout = timeout or DEFAULT_TIMEOUT
        result_holder = []
        exception_holder = []

        def target():
            try:
                result_holder.append(func(*args, **kwargs))
            except Exception as e:
                exception_holder.append(e)

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            raise TimeoutError(
                f"操作超时 ({timeout}s): {getattr(func, '__name__', str(func))}"
            )
        if exception_holder:
            raise exception_holder[0]
        return result_holder[0] if result_holder else None

    def safe_execute(self, step_index: int, func: Callable,
                     timeout: float = None, *args, **kwargs):
        """安全执行一个步骤：标记开始→超时执行→标记成功/失败"""
        self.mark_step_start(step_index)
        try:
            result = self.execute_with_timeout(func, timeout, *args, **kwargs)
            self.mark_step_success(step_index, result if isinstance(result, dict) else None)
            return result
        except TimeoutError as e:
            self.mark_step_failed(step_index, str(e))
            raise
        except Exception as e:
            self.mark_step_failed(step_index, f"{type(e).__name__}: {str(e)}")
            raise

    # ─── 心跳监控 ───────────────────────────────

    def _start_heartbeat(self):
        self._stop_heartbeat.clear()
        self._write_heartbeat()
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    def _write_heartbeat(self):
        with self._lock:
            heartbeat = {
                "timestamp": datetime.now().isoformat(),
                "execution_id": self._state.execution_id if self._state else "",
                "status": self._state.status if self._state else "unknown",
                "current_step": self._state.current_step if self._state else 0,
                "pid": os.getpid(),
            }
        try:
            with open(HEARTBEAT_FILE, "w", encoding="utf-8") as f:
                json.dump(heartbeat, f)
        except Exception:
            pass

    def _heartbeat_loop(self):
        while not self._stop_heartbeat.wait(timeout=self._heartbeat_interval):
            self._write_heartbeat()

    def is_alive(self) -> dict:
        """检查服务是否存活（基于心跳文件）"""
        if not HEARTBEAT_FILE.exists():
            return {"alive": False, "age_seconds": -1, "reason": "无心跳文件"}
        try:
            with open(HEARTBEAT_FILE, "r", encoding="utf-8") as f:
                hb = json.load(f)
            last_ts = datetime.fromisoformat(hb["timestamp"])
            age = (datetime.now() - last_ts).total_seconds()
            return {
                "alive": age < MAX_HEARTBEAT_AGE,
                "last_heartbeat": hb["timestamp"],
                "age_seconds": round(age, 1),
                "execution_id": hb.get("execution_id", ""),
                "status": hb.get("status", "unknown"),
                "current_step": hb.get("current_step", 0),
            }
        except Exception as e:
            return {"alive": False, "reason": str(e)}

    def clear_checkpoint(self):
        """清除所有断点数据"""
        with self._lock:
            self._stop_heartbeat.set()
            self._state = None
            for f in (CHECKPOINT_FILE, HEARTBEAT_FILE):
                if f.exists():
                    f.unlink(missing_ok=True)


_checkpoint_instance: Optional[CheckpointManager] = None


def get_checkpoint() -> CheckpointManager:
    global _checkpoint_instance
    if _checkpoint_instance is None:
        _checkpoint_instance = CheckpointManager()
    return _checkpoint_instance
