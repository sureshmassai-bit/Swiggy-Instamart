"""One command to run the app AND an ngrok tunnel, then print the public login URL.

    python -m webapp.share

Needs the ngrok program installed and your token added once:  ngrok config add-authtoken <TOKEN>
Stop with Ctrl+C: this closes the tunnel and the server together.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request

PORT = 5000


def tunnel_url(timeout=20):
    """Ask ngrok's local API (127.0.0.1:4040) for the public https URL."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=2) as r:
                for t in json.load(r).get("tunnels", []):
                    if t.get("public_url", "").startswith("https://"):
                        return t["public_url"]
        except OSError:
            pass
        time.sleep(1)
    return None


def app_is_up():
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{PORT}/login", timeout=2).status == 200
    except OSError:
        return False


def main():
    if shutil.which("ngrok") is None:
        print("ngrok was not found on PATH. Install it from https://ngrok.com/download, then run:\n"
              "  ngrok config add-authtoken <YOUR_TOKEN>\nand try again.")
        return 1
    env = {**os.environ, "INSTAMART_PUBLIC_HTTPS": "1"}
    # Reuse an app that is already running (for example run_forever.py) instead of starting a second copy that
    # cannot bind the port, exits at once and would take the tunnel down with it ("ERR_NGROK_3200 endpoint offline").
    server = None if app_is_up() else subprocess.Popen([sys.executable, "-m", "webapp.app"], env=env)
    tunnel = subprocess.Popen(["ngrok", "http", str(PORT)], stdout=subprocess.DEVNULL)
    try:
        url = tunnel_url()
        if url is None:
            print("ngrok did not start a tunnel. Check your authtoken ('ngrok config check') and try again.")
            return 1
        print(f"\nLogin URL : {url}/login\nLocal URL : http://127.0.0.1:{PORT}/login\n"
              "Credentials are in the credentials/ folder. Keep this window open; press Ctrl+C to stop sharing.\n")
        if server is None:
            print("(Using the app that is already running on this computer. If it stops, the public link stops working too.)\n")
        while tunnel.poll() is None:                       # stay up until Ctrl+C or the tunnel itself closes
            time.sleep(1)
        print("The ngrok tunnel closed. Run this command again.")
    except KeyboardInterrupt:
        pass
    finally:
        for p in (tunnel, server):
            if p is not None:
                p.terminate()
    return 0


if __name__ == "__main__":
    sys.exit(main())
