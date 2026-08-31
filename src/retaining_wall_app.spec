from pathlib import Path

from PyInstaller.utils.hooks import collect_all


APP_ROOT = Path(SPECPATH)

datas = [
    (str(APP_ROOT / "retaining_wall_model.py"), "."),
    (str(APP_ROOT / "company_logo.png"), "."),
    (str(APP_ROOT / "license_public_key.json"), "."),
]
binaries = []
hiddenimports = []

for package_name in (
    "gmsh",
    "openseespy",
    "matplotlib",
    "pandas",
    "numpy",
    "cryptography",
):
    package_datas, package_binaries, package_hiddenimports = collect_all(
        package_name
    )
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)

a = Analysis(
    [str(APP_ROOT / "retaining_wall_app.py")],
    pathex=[str(APP_ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name="TPC_Retaining_Wall_Analysis",
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
    icon=str(APP_ROOT / "company_logo.png"),
    version=str(APP_ROOT / "version_info.txt"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="TPC_Retaining_Wall_Analysis",
)
