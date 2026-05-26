import os
# Show ALL proxy-related vars
for key, val in sorted(os.environ.items()):
    if 'proxy' in key.lower():
        print(f"  {key} = {val}")

print("\nClearing ALL proxy vars (case-insensitive)...")
keys_to_delete = [k for k in os.environ if 'proxy' in k.lower()]
for key in keys_to_delete:
    del os.environ[key]
    print(f"  Deleted: {key}")

print("\nAfter clearing:")
for key, val in sorted(os.environ.items()):
    if 'proxy' in key.lower():
        print(f"  {key} = {val}")

# Try pip directly
import subprocess, sys
r = subprocess.run(
    [sys.executable, "-m", "pip", "install", "build", "twine"],
    capture_output=True, text=True, timeout=60,
    env={k: v for k, v in os.environ.items() if 'proxy' not in k.lower()}
)
print(f"\nPip exit: {r.returncode}")
if r.stdout:
    print(r.stdout[-300:])
if r.stderr:
    print(r.stderr[-300:])
