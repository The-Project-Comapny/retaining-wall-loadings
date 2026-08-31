import base64
import datetime as dt
import getpass
import hashlib
import json
import os
import platform
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat


PRODUCT_NAME = "TPC Retaining Wall Soil Analysis"
LICENSE_DIR = (
    Path(os.getenv("LOCALAPPDATA", Path.home()))
    / "TheProjectCompany"
    / "RetainingWallSoilAnalysis"
)
LICENSE_FILE = LICENSE_DIR / "license.json"
PUBLIC_KEY_FILE = "license_public_key.json"


def _b64_encode(data):
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _b64_decode(value):
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def encode_json(data):
    return _b64_encode(json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def decode_json(value):
    return json.loads(_b64_decode(value).decode("utf-8"))


def get_machine_guid():
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as key:
            value, _ = winreg.QueryValueEx(key, "MachineGuid")
            return str(value)
    except Exception:
        return ""


def get_machine_label():
    return platform.node() or os.getenv("COMPUTERNAME", "Unknown computer")


def get_machine_id():
    raw_parts = [
        get_machine_guid(),
        get_machine_label(),
        platform.system(),
        platform.machine(),
    ]
    raw = "|".join(part.strip().lower() for part in raw_parts if part)
    if not raw:
        raw = f"{getpass.getuser()}|{Path.home()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def create_activation_request():
    payload = {
        "product": PRODUCT_NAME,
        "machine_id": get_machine_id(),
        "machine_label": get_machine_label(),
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    return encode_json(payload)


def decode_activation_request(activation_request):
    payload = decode_json(activation_request.strip())
    if payload.get("product") != PRODUCT_NAME:
        raise ValueError("Activation request is for a different product.")
    if not payload.get("machine_id"):
        raise ValueError("Activation request is missing the machine ID.")
    return payload


def load_public_key(app_dir):
    public_key_path = Path(app_dir) / PUBLIC_KEY_FILE
    if not public_key_path.exists():
        raise FileNotFoundError(f"Missing public license key: {public_key_path}")

    data = json.loads(public_key_path.read_text(encoding="utf-8"))
    public_bytes = _b64_decode(data["public_key"])
    return Ed25519PublicKey.from_public_bytes(public_bytes)


def verify_license_code(license_code, app_dir):
    envelope = decode_json(license_code.strip())
    payload = envelope.get("payload")
    signature = envelope.get("signature")

    if not isinstance(payload, dict) or not signature:
        raise ValueError("Invalid license code format.")

    public_key = load_public_key(app_dir)
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")

    try:
        public_key.verify(_b64_decode(signature), payload_bytes)
    except InvalidSignature as exc:
        raise ValueError("License signature is invalid.") from exc

    if payload.get("product") != PRODUCT_NAME:
        raise ValueError("License is for a different product.")

    if payload.get("machine_id") != get_machine_id():
        raise ValueError("License is for a different computer.")

    expires = payload.get("expires")
    if expires:
        expiry_date = dt.date.fromisoformat(expires)
        if dt.date.today() > expiry_date:
            raise ValueError(f"License expired on {expires}.")

    return payload


def save_license_code(license_code):
    LICENSE_DIR.mkdir(parents=True, exist_ok=True)
    LICENSE_FILE.write_text(
        json.dumps({"license_code": license_code.strip()}, indent=2),
        encoding="utf-8",
    )


def load_saved_license_code():
    if not LICENSE_FILE.exists():
        return None
    data = json.loads(LICENSE_FILE.read_text(encoding="utf-8"))
    return data.get("license_code")


def get_license_status(app_dir):
    license_code = load_saved_license_code()
    if not license_code:
        return False, "No saved license found.", None

    try:
        payload = verify_license_code(license_code, app_dir)
        return True, "Licensed", payload
    except Exception as exc:
        return False, str(exc), None


def public_key_to_json(public_key):
    public_bytes = public_key.public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )
    return {
        "product": PRODUCT_NAME,
        "algorithm": "Ed25519",
        "public_key": _b64_encode(public_bytes),
    }
