import requests, json

r = requests.get("http://localhost:8000/api/v1/heatmap?n_runs=50", timeout=10)
data = r.json()
dets = data.get("detections", [])
print(f"Total detections: {len(dets)}")
print()
for d in dets:
    print(json.dumps({
        "id":   d.get("detection_id"),
        "risk": d.get("enforcement", {}).get("risk_level"),
        "flux": d.get("flux_kg_hr"),
        "conf": round(d.get("confidence", 0), 3),
        "evar": round(d.get("epistemic_variance", 0), 4),
    }, indent=2))

print("\n--- risk level distribution ---")
from collections import Counter
risks = [d.get("enforcement", {}).get("risk_level", "?") for d in dets]
for k, v in Counter(risks).items():
    print(f"  {k}: {v}")