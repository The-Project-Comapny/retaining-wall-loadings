import argparse
import datetime as dt
import json
import os
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

import license_manager


APP_DIR = Path(__file__).resolve().parent
APP_VERSION = "1.0.0"
AUTHOR_NAME = "Mishal Mohanlal"
LOGO_PATH = APP_DIR / "company_logo.png"
ADMIN_KEY_DIR = (
    Path(os.getenv("LOCALAPPDATA", Path.home()))
    / "TheProjectCompany"
    / "RetainingWallSoilAnalysisAdmin"
)
PRIVATE_KEY_PATH = ADMIN_KEY_DIR / "license_private_key.json"
PUBLIC_KEY_PATH = APP_DIR / license_manager.PUBLIC_KEY_FILE


def _b64_encode(data):
    return license_manager._b64_encode(data)


def _b64_decode(value):
    return license_manager._b64_decode(value)


def ensure_keys():
    if PRIVATE_KEY_PATH.exists() and PUBLIC_KEY_PATH.exists():
        return

    if PUBLIC_KEY_PATH.exists() and not PRIVATE_KEY_PATH.exists():
        raise FileNotFoundError(
            "The administrator private signing key is missing. Restore it "
            f"to {PRIVATE_KEY_PATH}. A replacement key would invalidate "
            "the public key distributed with the application."
        )

    ADMIN_KEY_DIR.mkdir(parents=True, exist_ok=True)
    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=Encoding.Raw,
        format=PrivateFormat.Raw,
        encryption_algorithm=NoEncryption(),
    )
    public_bytes = private_key.public_key().public_bytes(
        encoding=Encoding.Raw,
        format=PublicFormat.Raw,
    )

    PRIVATE_KEY_PATH.write_text(
        json.dumps(
            {
                "product": license_manager.PRODUCT_NAME,
                "algorithm": "Ed25519",
                "private_key": _b64_encode(private_bytes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    PUBLIC_KEY_PATH.write_text(
        json.dumps(
            {
                "product": license_manager.PRODUCT_NAME,
                "algorithm": "Ed25519",
                "public_key": _b64_encode(public_bytes),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def load_private_key():
    ensure_keys()
    data = json.loads(PRIVATE_KEY_PATH.read_text(encoding="utf-8"))
    return Ed25519PrivateKey.from_private_bytes(_b64_decode(data["private_key"]))


def generate_license_code(activation_request, licensed_to, expires):
    request_payload = license_manager.decode_activation_request(activation_request)
    private_key = load_private_key()

    payload = {
        "product": license_manager.PRODUCT_NAME,
        "machine_id": request_payload["machine_id"],
        "machine_label": request_payload.get("machine_label", ""),
        "licensed_to": licensed_to.strip() or "Licensed user",
        "issued_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "expires": expires.strip(),
    }

    if payload["expires"]:
        dt.date.fromisoformat(payload["expires"])

    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    signature = private_key.sign(payload_bytes)
    return license_manager.encode_json(
        {
            "payload": payload,
            "signature": _b64_encode(signature),
        }
    )


class LicenseGeneratorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        ensure_keys()

        self.title("TPC Retaining Wall - Admin License Generator")
        self.geometry("820x620")
        self.minsize(720, 520)
        self.configure(background="#F4F4F4")

        self._build_ui()

    def _build_ui(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        style = ttk.Style(self)
        style.configure("TPC.TFrame", background="#F4F4F4")
        style.configure(
            "TPC.Title.TLabel",
            background="#006A4F",
            foreground="white",
            font=("Segoe UI", 15, "bold"),
        )
        style.configure(
            "TPC.Subtitle.TLabel",
            background="#006A4F",
            foreground="#DFC869",
        )
        style.configure("TPC.TButton", font=("Segoe UI", 10, "bold"))

        header = ttk.Frame(self, padding=14, style="TPC.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)

        if LOGO_PATH.is_file():
            self.logo_image = tk.PhotoImage(file=str(LOGO_PATH))
            if self.logo_image.width() > 320:
                factor = max((self.logo_image.width() + 319) // 320, 1)
                self.logo_image = self.logo_image.subsample(factor, factor)
            tk.Label(
                header,
                image=self.logo_image,
                background="white",
                padx=8,
                pady=8,
            ).grid(row=0, column=0, rowspan=3, sticky="w", padx=(0, 14))

        title_frame = tk.Frame(header, background="#006A4F", padx=12, pady=10)
        title_frame.grid(row=0, column=1, rowspan=3, sticky="nsew")

        ttk.Label(
            title_frame,
            text="TPC Retaining Wall - Admin License Generator",
            style="TPC.Title.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            title_frame,
            text=f"Author: {AUTHOR_NAME}  |  Version: {APP_VERSION}",
            style="TPC.Subtitle.TLabel",
        ).pack(anchor="w")
        ttk.Label(
            title_frame,
            text=license_manager.PRODUCT_NAME,
            style="TPC.Subtitle.TLabel",
        ).pack(anchor="w")

        body = ttk.Frame(self, padding=(14, 0, 14, 14), style="TPC.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.rowconfigure(1, weight=1)
        body.rowconfigure(5, weight=1)

        ttk.Label(body, text="Activation request from user").grid(row=0, column=0, sticky="w")
        self.activation_text = tk.Text(body, height=8, wrap="word")
        self.activation_text.grid(row=1, column=0, sticky="nsew", pady=(4, 10))

        form = ttk.Frame(body)
        form.grid(row=2, column=0, sticky="ew")
        form.columnconfigure(1, weight=1)
        form.columnconfigure(3, weight=1)

        ttk.Label(form, text="Licensed to").grid(row=0, column=0, sticky="w")
        self.licensed_to_var = tk.StringVar(value="The Project Company")
        ttk.Entry(form, textvariable=self.licensed_to_var).grid(row=0, column=1, sticky="ew", padx=(8, 18))

        ttk.Label(form, text="Expires").grid(row=0, column=2, sticky="w")
        default_expiry = dt.date.today().replace(year=dt.date.today().year + 1).isoformat()
        self.expires_var = tk.StringVar(value=default_expiry)
        ttk.Entry(form, textvariable=self.expires_var).grid(row=0, column=3, sticky="ew", padx=(8, 0))

        buttons = ttk.Frame(body)
        buttons.grid(row=3, column=0, sticky="ew", pady=10)
        ttk.Button(buttons, text="Generate License Code", command=self.generate, style="TPC.TButton").pack(side="left")
        ttk.Button(buttons, text="Copy License Code", command=self.copy_license, style="TPC.TButton").pack(side="left", padx=(8, 0))

        ttk.Label(body, text="License code to send back to user").grid(row=4, column=0, sticky="w")
        self.license_text = tk.Text(body, height=10, wrap="word")
        self.license_text.grid(row=5, column=0, sticky="nsew", pady=(4, 0))

    def generate(self):
        activation_request = self.activation_text.get("1.0", "end").strip()
        try:
            license_code = generate_license_code(
                activation_request=activation_request,
                licensed_to=self.licensed_to_var.get(),
                expires=self.expires_var.get(),
            )
        except Exception as exc:
            messagebox.showerror("Could not generate license", str(exc))
            return

        self.license_text.delete("1.0", "end")
        self.license_text.insert("1.0", license_code)

    def copy_license(self):
        license_code = self.license_text.get("1.0", "end").strip()
        if not license_code:
            return
        self.clipboard_clear()
        self.clipboard_append(license_code)
        messagebox.showinfo("Copied", "License code copied to clipboard.")


def main():
    parser = argparse.ArgumentParser(description="Generate offline license codes.")
    parser.add_argument("--init-keys", action="store_true", help="Create private/public key files if missing.")
    args = parser.parse_args()

    if args.init_keys:
        ensure_keys()
        print(f"Private key: {PRIVATE_KEY_PATH}")
        print(f"Public key: {PUBLIC_KEY_PATH}")
        return

    app = LicenseGeneratorApp()
    app.mainloop()


if __name__ == "__main__":
    main()
