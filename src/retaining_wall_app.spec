import os
import sys
from pathlib import Path

APP_ROOT = Path(SPECPATH)

datas = [
    (str(APP_ROOT / "retaining_wall_model.py"), "."),
    (str(APP_ROOT / "company_logo.png"), "."),
    (str(APP_ROOT / "license_public_key.json"), "."),
]
binaries = []

# The solver module is shipped as a data file and imported dynamically by the
# GUI worker, so PyInstaller cannot infer these imports from static analysis.
# Listing only the runtime entry points lets their standard PyInstaller hooks
# collect the necessary native libraries without pulling in package test suites.
hiddenimports = [
    "gmsh",
    "openseespy.opensees",
    "pandas",
]

# Anaconda's Python 3.10 _ctypes extension depends on libffi DLLs kept in
# <environment>/Library/bin. PyInstaller does not always discover these
# indirect dependencies, which leaves the packaged application unable to
# start. Bundle every available libffi alias so the build remains compatible
# with both current and older Anaconda package layouts.
conda_library_bin = Path(sys.prefix) / "Library" / "bin"
libffi_dlls = sorted(conda_library_bin.glob("ffi*.dll"))
if not libffi_dlls:
    raise FileNotFoundError(
        f"No libffi DLLs were found in the build environment: {conda_library_bin}"
    )
binaries.extend((str(dll_path), ".") for dll_path in libffi_dlls)

# Several standard-library extensions in Anaconda depend on DLLs stored in
# Library/bin. They are not reliably discovered when the environment was not
# activated before starting the build script, so include them explicitly.
for dll_name in (
    "sqlite3.dll",
    "libexpat.dll",
    "liblzma.dll",
    "libbz2.dll",
    "libcrypto-3-x64.dll",
    "libssl-3-x64.dll",
):
    dll_path = conda_library_bin / dll_name
    if dll_path.is_file():
        binaries.append((str(dll_path), "."))

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

# A build started from a developer tool may inherit unrelated native-library
# directories on PATH. Do not package DLLs from Codex's document/PDF runtimes;
# in particular, their ICU build is incompatible with the Qt wheel used here.
a.binaries = type(a.binaries)(
    item
    for item in a.binaries
    if "codex-runtimes" not in str(item[1]).lower()
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
    console=os.environ.get("TPC_BUILD_CONSOLE") == "1",
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
