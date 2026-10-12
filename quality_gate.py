
import json
import sys

MINIMUM_ACCURACY = 0.60

with open("metrics.json") as f:
    metrics = json.load(f)

accuracy = metrics["accuracy"]

print(f"Model accuracy: {accuracy:.3f}")
print(f"Required accuracy: {MINIMUM_ACCURACY:.3f}")

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    sys.exit(1)

print("QUALITY GATE PASSED")
