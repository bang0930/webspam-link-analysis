"""담당: 박병언 (팀 작성 코드).

실행 환경 기록, 확정안 10절, 13절. 강의 11강 p36-40 (희소행렬 실행 환경).
새 라이브러리 없이 macOS sysctl과 Linux /proc을 사용한다.
seconds는 점수 캐시 콜백 시간이다. reported_algorithm_seconds는 알고리즘이 보고한 시간이며,
TrustRank에서는 신뢰 전파 반복만 포함한다. 두 값 모두 통제된 단독 실행 벤치마크와 구분한다.
"""
from __future__ import annotations
import argparse
import csv
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src import config
from src.results_io import git_state


def command(*args):
    """플랫폼 명령을 실행하고 사용할 수 없으면 빈 문자열을 반환한다."""
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else ''
    except (OSError, subprocess.SubprocessError):
        return ''


def hardware():
    """CPU 모델, 논리 코어 수, 물리 코어 수, 메모리 바이트를 찾는다."""
    model = command('sysctl', '-n', 'machdep.cpu.brand_string') if sys.platform == 'darwin' else ''
    memory = command('sysctl', '-n', 'hw.memsize') if sys.platform == 'darwin' else ''
    physical = command('sysctl', '-n', 'hw.physicalcpu') if sys.platform == 'darwin' else ''
    cpuinfo = Path('/proc/cpuinfo')
    if cpuinfo.exists():
        text = cpuinfo.read_text()
        m = re.search(r'^(?:model name|Hardware)\s*:\s*(.+)$', text, re.M)
        model = model or (m.group(1) if m else '')
        pairs = set()
        for block in text.split('\n\n'):
            p = re.search(r'^physical id\s*:\s*(.+)$', block, re.M)
            c = re.search(r'^core id\s*:\s*(.+)$', block, re.M)
            if p and c:
                pairs.add((p.group(1), c.group(1)))
        physical = physical or (len(pairs) if pairs else '')
    meminfo = Path('/proc/meminfo')
    if not memory and meminfo.exists():
        m = re.search(r'^MemTotal:\s*(\d+)\s+kB', meminfo.read_text(), re.M)
        memory = int(m.group(1)) * 1024 if m else ''
    if not memory:
        try:
            memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        except (ValueError, OSError, AttributeError):
            memory = ''
    return model or platform.processor(), os.cpu_count(), physical, memory


def score_timing(record):
    """캐시 콜백 시간과 알고리즘 보고 시간을 분리하고 측정 범위를 기록한다. 확정안 10절."""
    info = record.get('info', {})
    reported = info.get('seconds')
    scope = ''
    if reported is not None:
        scope = ('trustrank_iteration_only' if info.get('method') == 'trustrank'
                 else 'method_reported_see_settings')
    return dict(seconds=record.get('seconds'), seconds_scope='score_cache_callback',
                reported_algorithm_seconds=reported, reported_algorithm_seconds_scope=scope,
                timing_note='Shared work may be charged to another row; not a standalone runtime benchmark.')


def environment_rows(score_dir=None):
    """환경과 모든 모드의 점수 메타데이터를 읽어 긴 형태의 표로 만든다."""
    state = git_state()
    dirty = command('git', '-C', str(ROOT), 'status', '--porcelain')
    base = dict(recorded_at=datetime.now().astimezone().isoformat(timespec='seconds'),
                commit=state['commit'], dirty=bool(dirty), src_dirty=state['dirty'])
    rows = []
    def add(category, item, value='', **extra):
        rows.append(dict(base, category=category, item=item, value=value, **extra))
    model, logical, physical, memory = hardware()
    for key, value in [('os', platform.system()), ('os_version', platform.platform()),
                       ('cpu_model', model), ('logical_cores', logical), ('physical_cores', physical),
                       ('memory_bytes', memory), ('python', platform.python_version())]:
        add('environment', key, value)
    pins = dict(re.findall(r'^([\w-]+)==([^\s#]+)', (ROOT / 'requirements.txt').read_text(), re.M))
    for name in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'networkx', 'matplotlib'):
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = ''
        add('package', name, version, required_version=pins.get(name, ''),
            matches_pin=version == pins.get(name) if version else False,
            status='installed' if version else 'missing')
    seen = set()
    for path in sorted(Path(score_dir or config.SCORES_DIR).rglob('*.json')):
        record = json.loads(path.read_text())
        info = record.get('info', {})
        name = record.get('name', path.stem)
        seen.add(re.sub(r'_s\d+$', '', name))
        add('score', name, mode=record.get('mode', ''), metadata_path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            **score_timing(record), n_iter=info.get('n_iter'),
            n_iter_pagerank=info.get('n_iter_pagerank'), score_commit=record.get('commit'),
            score_dirty=record.get('dirty'), score_saved_at=record.get('saved_at'), status='available',
            settings=json.dumps(info, ensure_ascii=False, sort_keys=True))
    from experiments.run_main import methods
    for method in methods():
        if method.key not in seen:
            add('score', method.key, seconds='', n_iter='', status='missing')
    return rows


def main(argv=None):
    """환경 표를 저장하고 실행 환경과 누락된 점수 파일을 출력한다."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out-dir', type=Path, default=config.TABLES_DIR)
    args = ap.parse_args(argv)
    rows = environment_rows()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    path = args.out_dir / 'environment.csv'
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f'{row["category"]}: {row["item"]} = {row["value"]} {row.get("status", "")}')
    print(f'저장: {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
