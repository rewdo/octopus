import os, subprocess, sys
# Clear ALL proxy env vars (case-insensitive)
for key in list(os.environ.keys()):
    if 'proxy' in key.lower():
        del os.environ[key]
        print(f"Cleared: {key}")

# Run pip
result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "build", "twine", "--no-cache-dir"],
    capture_output=True, text=True, timeout=30
)
print(result.stdout[-500:] if result.stdout else "")
print(result.stderr[-500:] if result.stderr else "")
print(f"Exit: {result.returncode}")
