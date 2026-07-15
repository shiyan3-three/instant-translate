# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
import sys

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
    copy_metadata,
    get_package_paths,
)

project_root = Path(SPECPATH).resolve()

# PaddleOCR 2.x imports ``ppocr`` and ``tools`` as top-level packages after
# adding its own package directory to ``sys.path``.  Static analysis sees only
# ``paddleocr.paddleocr`` and otherwise omits those runtime modules.  Preserve
# the package tree as real files so those imports keep the same layout they
# have in site-packages.  Paddle's inference path also reads Cython utility
# sources at runtime (notably CppSupport.cpp), so collect those data files.
_, paddleocr_package_dir = get_package_paths('paddleocr')
paddleocr_datas = [(paddleocr_package_dir, 'paddleocr')]
cython_datas = collect_data_files('Cython')
imageio_metadata = copy_metadata('imageio')
paddle_binaries = collect_dynamic_libs('paddle')
sys.path.insert(0, paddleocr_package_dir)
try:
    paddleocr_hiddenimports = [
        *collect_submodules('paddleocr'),
        *collect_submodules('ppocr'),
        *collect_submodules('tools'),
    ]
finally:
    sys.path.remove(paddleocr_package_dir)


a = Analysis(
    [str(project_root / 'app' / 'main.py')],
    # PaddleOCR 2.x treats its package directory as an import root for the
    # sibling top-level ``ppocr`` and ``tools`` packages.
    pathex=[str(project_root), paddleocr_package_dir],
    binaries=paddle_binaries,
    datas=[
        (str(project_root / 'app'), 'app'),
        *paddleocr_datas,
        *cython_datas,
        *imageio_metadata,
    ],
    hiddenimports=[
        'app.logger',
        'paddle',
        'paddleocr',
        'PIL',
        'httpx',
        'pytesseract',
        *paddleocr_hiddenimports,
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='instant-translate',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='instant-translate',
)
