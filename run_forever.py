"""Keep the Instamart Copilot web app running.

    python run_forever.py          (or double-click start_app.bat on Windows)

* does the first-time setup if it is missing (Part A database, web database and logins),
* starts the app and opens the login page in your browser,
* starts it again if it ever stops or crashes, so the page keeps loading,
* logins survive restarts: users are stored in data/app.db and the session key in data/.secret_key.
Press Ctrl+C in this window to stop it. If the computer is off or this window is closed, the page is not available.
"""
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = int(os.environ.get("INSTAMART_PORT", "5000"))
URL = f"http://127.0.0.1:{PORT}/login"


def run(*args):
    return subprocess.run([sys.executable, *args], cwd=ROOT).returncode


def up():
    try:
        return urllib.request.urlopen(URL, timeout=2).status == 200
    except OSError:
        return False


def setup():
    if not (ROOT / "data" / "instamart.db").exists():
        print("First run: building the Part A database ...")
        if run("-m", "src.data_pipeline") != 0:
            sys.exit("Could not build data/instamart.db. Fix the error above and run again.")
    if not (ROOT / "data" / "app.db").exists():
        print("First run: creating the web database and logins (see the credentials folder) ...")
        if run("-m", "webapp.manage", "init") != 0:
            sys.exit("Could not create the web database. Fix the error above and run again.")


def main():
    setup()
    if up():
        sys.exit(f"Something is already answering on port {PORT}. Close the other copy, or set INSTAMART_PORT to another number.")
    quick_exits, opened = 0, False
    try:
        while True:
            started = time.time()
            child = subprocess.Popen([sys.executable, "-m", "webapp.app"], cwd=ROOT)
            for _ in range(40):                                   # wait up to ~10 s for the page to answer
                if child.poll() is not None or up():
                    break
                time.sleep(0.25)
            if child.poll() is None and not opened:
                opened = True
                print(f"\nApp is running. Open {URL}   (Ctrl+C here stops it)\n")
                if not os.environ.get("INSTAMART_NO_BROWSER"):
                    webbrowser.open(URL)
            child.wait()
            quick_exits = quick_exits + 1 if time.time() - started < 5 else 0
            if quick_exits >= 5:
                sys.exit("The app keeps stopping right after it starts. Read the error above (is port "
                         f"{PORT} used by another program?).")
            print("The app stopped; starting it again in 2 seconds ...")
            time.sleep(2)
    except KeyboardInterrupt:
        print("\nStopping.")
        try:
            child.terminate()
        except Exception:
            pass


if __name__ == "__main__":
    main()
