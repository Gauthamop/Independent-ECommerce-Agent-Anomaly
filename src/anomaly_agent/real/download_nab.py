"""Download a few real time series from the Numenta Anomaly Benchmark (NAB) into data/real/nab_raw.

NAB is a public benchmark of real streaming data with labelled anomaly windows
(https://github.com/numenta/NAB). It has its own open-source licence, so the download is git-ignored:
re-run this script instead of committing the files.

    python -m src.anomaly_agent.real.download_nab
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/numenta/NAB/master"
FILES = {
    "nyc_taxi.csv": f"{BASE}/data/realKnownCause/nyc_taxi.csv",
    "cpu_utilization_asg_misconfiguration.csv": f"{BASE}/data/realKnownCause/cpu_utilization_asg_misconfiguration.csv",
    "ec2_request_latency_system_failure.csv": f"{BASE}/data/realKnownCause/ec2_request_latency_system_failure.csv",
    "machine_temperature_system_failure.csv": f"{BASE}/data/realKnownCause/machine_temperature_system_failure.csv",
    "combined_windows.json": f"{BASE}/labels/combined_windows.json",
}


def download(out_dir="data/real/nab_raw", force=False):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        target = out / name
        if target.exists() and not force:
            continue
        with urllib.request.urlopen(url, timeout=60) as response:  # these are data files, never executed
            data = response.read()
        if not data:
            raise RuntimeError(f"empty download: {url}")
        target.write_bytes(data)
        print(f"downloaded {name} ({len(data) // 1024} KB)")


if __name__ == "__main__":
    download()
