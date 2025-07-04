# build.spec
import sys
import os
import matplotlib

# Get matplotlib data path
mpl_data_dir = matplotlib.get_data_path()

block_cipher = None

a = Analysis(
    ['final.py'],  # Your main script filename
    pathex=[],
    binaries=[],
    datas=[
        (os.path.join(mpl_data_dir, 'matplotlibrc'), 'matplotlib/mpl-data'),
        (os.path.join(mpl_data_dir, 'fonts'), 'matplotlib/mpl-data/fonts'),
    ],
    hiddenimports=[
        'matplotlib.backends.backend_qtagg',
        'matplotlib.backends.backend_qt5agg',
        'matplotlib.backends.qt_compat',
        'PyQt6.sip',
        'numpy',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RLPathfinding',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    icon=None,
    version='version.txt' if os.path.exists('version.txt') else None,
)