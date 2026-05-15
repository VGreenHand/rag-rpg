"""
公共工具模块
"""
import threading
import logging
from typing import Callable, TypeVar, Any

logger = logging.getLogger("rag-rpg.common")

T = TypeVar("T")


class SafeTimer:
    @staticmethod
    def run(func: Callable[..., T], timeout: float, *args: Any, **kwargs: Any) -> T:
        from checkpoint_manager import TimeoutError

        result_holder: list[T] = []
        exc_holder: list[Exception] = []

        def target():
            try:
                result_holder.append(func(*args, **kwargs))
            except Exception as e:
                exc_holder.append(e)

        t = threading.Thread(target=target, daemon=True)
        t.start()
        t.join(timeout=timeout)

        if t.is_alive():
            logger.warning(f"操作超时 ({timeout}s): {func.__name__}")
            raise TimeoutError(f"操作超时 ({timeout}s)")

        if exc_holder:
            raise exc_holder[0]

        return result_holder[0] if result_holder else None


class SingletonFactory:
    def __init__(self, factory_class):
        self._factory_class = factory_class
        self._instances: dict[str, Any] = {}

    def get(self, *args, **kwargs) -> Any:
        key = str(args) + str(sorted(kwargs.items()))
        if key not in self._instances:
            self._instances[key] = self._factory_class(*args, **kwargs)
        return self._instances[key]

    def __call__(self, *args, **kwargs) -> Any:
        return self.get(*args, **kwargs)

    def clear(self):
        self._instances.clear()
