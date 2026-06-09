import random

import numpy as np
import pandas as pd

# -------------------------------------------------------------------------
# Simulation parameters
# -------------------------------------------------------------------------

AREA_SIZE = 100.0 # Side length of the square deployment area
L = 128  # Byte sequence length
N = 100 # Number of sensors
NUM_EVENTS = 10000 # Number of events
r = 13.0 # Event detection radius
T = 10000 # Total simulation time
MAX_DELAY = 10 # Maximum temporal spacing between events

LAMBDA_MIN = 30
LAMBDA_MAX = 80
DELTA_MIN = 1
DELTA_MAX = 20

events_data = []
observations_data = []

rng = random.Random(1234) # 1st random generator    
rng2 = np.random.default_rng(1235) # 2nd random generator
rng3 = np.random.default_rng(1236) # 3rd random generator

# -------------------------------------------------------------------------
# Degradation model
# -------------------------------------------------------------------------

def apply_delay(timestamp, distance):
   
    delayed_time = (distance / r) * MAX_DELAY + timestamp

    return delayed_time

def apply_poisson(payload, distance):
    
    corrupted = bytearray(payload)
    n_bytes = len(payload)

    norm_d = min(distance / r, 1.0)

    lam = LAMBDA_MIN + norm_d * (LAMBDA_MAX - LAMBDA_MIN)

    n_errors = int(rng3.poisson(lam))
    n_errors = int(np.clip(n_errors, 0, n_bytes))

    positions = rng3.choice(n_bytes, size=n_errors, replace=False)

    delta = int(DELTA_MIN + norm_d * (DELTA_MAX - DELTA_MIN)) 

    for pos in positions:
        change = rng2.integers(-delta, delta + 1) 
        while change == 0:
            change = rng2.integers(-delta, delta + 1)
        corrupted[pos] = int(np.clip(corrupted[pos] + change, 0, 255))

    return bytes(corrupted), n_errors

# -------------------------------------------------------------------------
# Dataset generation
# -------------------------------------------------------------------------

# Sensor Generation
sx = np.array([rng.uniform(0, AREA_SIZE) for _ in range(N)], dtype=np.float32)
sy = np.array([rng.uniform(0, AREA_SIZE) for _ in range(N)], dtype=np.float32)
sensor_ids = np.arange(N, dtype=np.int32)

print("\n***Sensor info:***\n")
for i in range(N):
    print(f"Sensor {sensor_ids[i]}: ({sx[i]:.2f}, {sy[i]:.2f})")

# Event Generation
t = np.arange(0, NUM_EVENTS*MAX_DELAY, MAX_DELAY)
payloads = [bytes(rng.getrandbits(8) for _ in range(L)) for _ in range(NUM_EVENTS)]
ex = np.array([rng.uniform(r, AREA_SIZE-r) for _ in range(NUM_EVENTS)], dtype=np.float32)
ey = np.array([rng.uniform(r, AREA_SIZE-r) for _ in range(NUM_EVENTS)], dtype=np.float32)

for i in range(NUM_EVENTS):

    if i == 0:
        print("\n***Event info:*** (Printing just the first 5 events)\n")
    if i < 5:
        print(f"Event {i}:")
        print(f"  Timestamp: {t[i]:.2f}")
        print(f"  Coordinates: ({ex[i]:.2f}, {ey[i]:.2f})")
        print(f"  Length: {len(payloads[i])} Bytes")       
        print(f"  Payload: {payloads[i].hex()}")
    
    event_row = {
        "event_id": i,
        "event_x": float(ex[i]),
        "event_y": float(ey[i]),
        "event_time": float(t[i]),
    }

    for b in range(L):
        event_row[f"target_b{b}"] = payloads[i][b]
    events_data.append(event_row)

    # Detection
    dx = sx - ex[i]
    dy = sy - ey[i]
    mask = (dx*dx + dy*dy) <= r*r
    detected_ids = sensor_ids[mask]
    distances = np.sqrt(dx[mask]*dx[mask] + dy[mask]*dy[mask])
    detected_sx = sx[mask]
    detected_sy = sy[mask]

    if i < 5:
        print(f"\n  Detected sensors:\n")
        for j in range(len(detected_ids)):
            print(f"    Sensor {detected_ids[j]} - Distance: {distances[j]:.2f}")
        print("\n  Detected event info:\n")

    for j in range(len(detected_ids)):
        sensor_id = detected_ids[j]
        d = float(distances[j])

        delayed_timestamp = apply_delay(t[i], d)
        corrupted_payload, k_errors = apply_poisson(payloads[i], d)
        delta_time = delayed_timestamp - t[i]

        if i < 5:
            print(f"\nSensor {sensor_id}")
            print(f"  Distance          : {d:.2f}")
            print(f"  Original time     : {t[i]:.2f}")
            print(f"  Delayed time      : {delayed_timestamp:.2f}")
            print(f"  Delta time        : {delta_time:.2f}")
            print(f"  Original payload  : {payloads[i].hex()[:256]}")
            print(f"  Corrupted payload (Poisson): {corrupted_payload.hex()[:256]}")
            print(f"  Number of errors: {k_errors}")

        observations_row = {
            "event_id": i,
            "sensor_id": sensor_id,
            "sensor_x": float(detected_sx[j]),
            "sensor_y": float(detected_sy[j]),
            "delta_arrival_time": float(delta_time),
            "k_errors": k_errors
        }

        for b in range(L):
            observations_row[f"rx_b{b}"] = corrupted_payload[b]
        observations_data.append(observations_row)
       
observations_data_random = []
event_ids = sorted(set(row["event_id"] for row in observations_data))

for event_id in event_ids:
    rows = [row for row in observations_data if row["event_id"] == event_id]
    rng.shuffle(rows)
    observations_data_random.extend(rows)

observations_data = observations_data_random

df_events = pd.DataFrame(events_data)
df_obs = pd.DataFrame(observations_data)
df_events.to_csv("events.csv", index=False)
df_obs.to_csv("observations.csv", index=False)