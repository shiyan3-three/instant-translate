# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['F:\\workSorfware\\forSelf\\instant-translate\\app\\main.py'],
    pathex=['F:\\workSorfware\\forSelf\\instant-translate'],
    binaries=[],
    datas=[('F:\\workSorfware\\forSelf\\instant-translate\\app', 'app')],
    hiddenimports=['app.logger', 'paddle', 'paddleocr', 'PIL', 'httpx', 'pytesseract'],
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
