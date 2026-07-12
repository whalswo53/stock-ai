"""
timeout 파라미터를 지원하지 않는 외부 호출(대표적으로 pykrx — 내부 requests
세션에 timeout이 전혀 없고, 로그인 세션이 없으면 호출마다 재로그인을 재시도함)에
스레드 기반으로 타임아웃을 강제로 씌운다.

세그폴트 인시던트 원인 분석: KRX 서비스 장애 중 pykrx 대량 호출(수천 건)이
호출당 최대 수십 초씩 블로킹되며 스레드·소켓이 누적, 리소스 고갈로 이어진
것으로 추정됨. 이 헬퍼 + 호출부의 연속 실패 서킷브레이커로 방어한다.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeoutError
from typing import Callable, TypeVar

T = TypeVar("T")


def call_with_timeout(fn: Callable[..., T], *args, timeout: float = 8.0, **kwargs) -> T:
    """fn(*args, **kwargs)을 최대 timeout초까지만 기다린다.

    시간 초과 시 TimeoutError를 던진다. 내부 스레드는 wait=False로 즉시
    포기하므로 호출부는 블로킹되지 않는다 (단, 실제 소켓은 pykrx 내부에서
    나중에 알아서 끝나거나 타임아웃될 때까지 백그라운드에 남을 수 있음 —
    타임아웃 파라미터가 아예 없는 라이브러리에 대한 마지막 방어선일 뿐,
    소켓을 강제로 끊는 진짜 취소는 아니다).
    """
    ex = ThreadPoolExecutor(max_workers=1)
    future = ex.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except _FutureTimeoutError:
        raise TimeoutError(f"{getattr(fn, '__name__', fn)} timed out after {timeout}s")
    finally:
        ex.shutdown(wait=False)
