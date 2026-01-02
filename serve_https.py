#!/usr/bin/env python3
"""Generate missing certificates and serve the repo folder over HTTPS."""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import subprocess
from http import server
from pathlib import Path


def ensure_certificates(cert_path: Path, key_path: Path) -> None:
    """Create a self-signed certificate if the files do not already exist."""
    if cert_path.exists() and key_path.exists():
        return
    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "openssl",
        "req",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-keyout",
        str(key_path),
        "-x509",
        "-days",
        "3650",
        "-out",
        str(cert_path),
        "-subj",
        "/CN=localhost",
    ]
    try:
        print("Generating self-signed certificate…")
        subprocess.run(cmd, check=True)
    except FileNotFoundError:
        raise SystemExit("OpenSSL not found; install it or provide cert/key manually.")


def serve(directory: Path, port: int, cert: Path, key: Path) -> None:
    """Start an HTTPS server rooted at `directory`."""
    handler = server.SimpleHTTPRequestHandler
    os.chdir(directory)
    httpd = server.ThreadingHTTPServer(("0.0.0.0", port), handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(certfile=str(cert), keyfile=str(key))
    httpd.socket = context.wrap_socket(httpd.socket, server_side=True)
    print(f"Serving '{directory}' over https://localhost:{port}")
    httpd.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the Loiacono WebGPU demo over HTTPS with generated certs."
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Directory to serve (defaults to repo root).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8443,
        help="Port to listen on.",
    )
    parser.add_argument(
        "--cert",
        type=Path,
        default=Path("certs/local-cert.pem"),
        help="Path to write/use the TLS certificate.",
    )
    parser.add_argument(
        "--key",
        type=Path,
        default=Path("certs/local-key.pem"),
        help="Path to write/use the TLS private key.",
    )
    args = parser.parse_args()

    ensure_certificates(args.cert, args.key)
    serve(args.dir.resolve(), args.port, args.cert.resolve(), args.key.resolve())


if __name__ == "__main__":
    main()
