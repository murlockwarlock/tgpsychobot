import json
from pathlib import Path
import subprocess
import re
from concurrent.futures import ThreadPoolExecutor, as_completed


root = Path(__file__).resolve().parents[1]
reports = root / "regression-results"
reports.mkdir(exist_ok=True)
def run_file(path):
    with (reports / f"{path.stem}.log").open("w") as output:
        try:
            result = subprocess.run(["bash", str(root / "scripts/run_content_authoring_tests.sh"), str(path), "--timeout=30", "--timeout-method=thread", "--tb=short"], stdout=output, stderr=subprocess.STDOUT, timeout=180)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    output = (reports / f"{path.stem}.log").read_text()
    counts = {name: int(match.group(1)) if (match := re.search(r"(\d+) " + name, output)) else 0 for name in ("passed", "failed", "skipped")}
    return {"file": path.name, "exit": code, **counts}


results = []
with ThreadPoolExecutor(max_workers=2) as pool:
    futures = [pool.submit(run_file, path) for path in sorted((root / "tests").glob("test_*.py"))]
    for future in as_completed(futures):
        results.append(future.result())
        (reports / "summary.json").write_text(json.dumps(results))
        print(json.dumps(results[-1]), flush=True)
print(json.dumps({"files": len(results), "green": all(item["exit"] == 0 for item in results), **{name: sum(item[name] for item in results) for name in ("passed", "failed", "skipped")}}), flush=True)
raise SystemExit(0 if all(item["exit"] == 0 for item in results) else 1)
