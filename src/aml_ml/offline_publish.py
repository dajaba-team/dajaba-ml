"""닫힌 데이터셋 폴더만 발행하며 Windows의 짧은 이동 거부를 제한적으로 재시도합니다.

SHA256 검증과 체크포인트 전이는 호출자가 담당합니다. 이곳의 메타데이터 비교는
재시도 중 관찰 가능한 변경을 거부하며, 동시 수정을 허용하는 파일 잠금은 아닙니다.
"""

import logging
import os
import stat
from pathlib import Path
from time import monotonic, sleep

from .offline_progress import check_cancelled

LOG = logging.getLogger(__name__)
_WINDOWS = os.name == "nt"
_DELAYS = (0.05, 0.1, 0.2)
_MAX_SECONDS = 1.0


def _stamp(path: Path, *, directory: bool, content: bool) -> tuple[int, ...]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ValueError("Publication path must not be a link, junction or reparse point")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode):
        raise ValueError("Publication requires a directory containing only regular files")
    identity = (info.st_dev, info.st_ino, info.st_mode)
    if content:
        identity += (info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_nlink)
    return identity


def _snapshot(stage: Path, output: Path):
    # lstat은 끊어진 링크도 기존 목적지로 취급하고, 접근 거부를 부재로 바꾸지 않습니다.
    try:
        output.lstat()
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError("Publication destination already exists; preserve it")
    parents = {
        path: _stamp(path, directory=True, content=False)
        for path in set(stage.parents) | set(output.parents)
    }
    directory = _stamp(stage, directory=True, content=True)
    if directory[0] != parents[output.parent][0]:
        raise ValueError("Atomic publication requires the same filesystem")
    children = {
        child.name: _stamp(child, directory=False, content=True) for child in stage.iterdir()
    }
    if _stamp(stage, directory=True, content=True) != directory:
        raise ValueError("Publication source changed while inspecting it")
    return parents, directory, children


def publish_directory(stage: Path, output: Path) -> None:
    """최초 포함 4회·1초 이내에서만 WinError 5를 재시도하고 다른 실패는 전파합니다."""
    if ".." in stage.parts or ".." in output.parts:
        raise ValueError("Publication paths must not contain parent traversal")
    stage, output = stage.absolute(), output.absolute()
    if stage.is_relative_to(output) or output.is_relative_to(stage):
        raise ValueError("Publication source and destination must be separate trees")
    check_cancelled()
    initial = _snapshot(stage, output)
    started = monotonic()
    last_denial: PermissionError | None = None
    for attempt in range(len(_DELAYS) + 1):
        check_cancelled()
        if attempt:
            try:
                if _snapshot(stage, output) != initial:
                    raise ValueError("Publication source or parent changed; preserve all artifacts")
            except Exception as state_error:
                raise state_error from last_denial
            check_cancelled()
            if monotonic() - started >= _MAX_SECONDS:
                if last_denial is None:
                    raise RuntimeError("Missing original publication denial")
                last_denial.add_note("상태 재확인 후 폴더 발행 시간 상한에 도달했습니다")
                raise last_denial
        try:
            stage.rename(output)
        except PermissionError as error:
            if not _WINDOWS or getattr(error, "winerror", None) != 5:
                raise
            last_denial = error
            elapsed = monotonic() - started
            if attempt == len(_DELAYS) or elapsed >= _MAX_SECONDS:
                error.add_note(f"폴더 발행 중단: 시도 {attempt + 1}회, 경과 {elapsed:.3f}초")
                raise
            # 실패한 이동의 결과가 불확실하거나 대상이 바뀌면 다시 이동하지 않습니다.
            if _snapshot(stage, output) != initial:
                raise ValueError(
                    "Publication source or parent changed after denied rename"
                ) from error
            check_cancelled()
            delay = _DELAYS[attempt]
            if monotonic() - started + delay >= _MAX_SECONDS:
                error.add_note("폴더 발행 재시도 시간 상한에 도달했습니다")
                raise
            LOG.warning(
                "폴더 발행 재시도 %d/3: WinError 5, %.3f초 대기 (%s -> %s)",
                attempt + 1,
                delay,
                stage,
                output,
            )
            sleep(delay)
            check_cancelled()
            if monotonic() - started >= _MAX_SECONDS:
                error.add_note("대기 후 폴더 발행 재시도 시간 상한에 도달했습니다")
                raise
        else:
            return
