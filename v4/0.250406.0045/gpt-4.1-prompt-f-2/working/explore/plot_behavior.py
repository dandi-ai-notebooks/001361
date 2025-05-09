"""
Plots for behavioral variables from sub-m11_ses-03_behavior+ophys.nwb
Includes (1) line plots for position, speed, lick over time, (2) histograms for reward and reward zone entries
Plots are saved to PNG files in the explore/ directory.
"""

import pynwb
import h5py
import remfile
import numpy as np
import matplotlib.pyplot as plt

nwb_url = "https://api.dandiarchive.org/api/assets/d77ea78a-8978-461d-9d11-3c5cef860d82/download/"
remote_file = remfile.File(nwb_url)
h5_file = h5py.File(remote_file)
io = pynwb.NWBHDF5IO(file=h5_file)
nwb = io.read()

behavior = nwb.processing["behavior"].data_interfaces["BehavioralTimeSeries"].time_series

# Helper for timeseries plot
def plot_behavior_ts(name, ts, max_points=10000):
    data = ts.data[:max_points]
    t = ts.timestamps[:max_points]
    plt.figure(figsize=(10, 4))
    plt.plot(t, data, lw=0.75)
    plt.xlabel("Time (s)")
    plt.ylabel(f"{name} ({ts.unit})")
    plt.title(f"{name} timeseries (first {len(data)} points)")
    plt.tight_layout()
    plt.savefig(f"explore/behavior_{name}_timeseries.png")
    plt.close()

# Position
plot_behavior_ts("position", behavior["position"])
# Speed
plot_behavior_ts("speed", behavior["speed"])
# Lick
plot_behavior_ts("lick", behavior["lick"])

# Histograms
plt.figure(figsize=(6, 3))
plt.hist(behavior["reward_zone"].data[:], bins=np.arange(-0.5, np.max(behavior["reward_zone"].data[:])+1.5, 1), color="tab:blue", rwidth=0.85)
plt.xlabel("Reward zone entry value")
plt.ylabel("Count")
plt.title("Reward zone entries")
plt.tight_layout()
plt.savefig("explore/behavior_reward_zone_hist.png")
plt.close()

plt.figure(figsize=(6, 3))
plt.hist(behavior["Reward"].timestamps[:], bins=20, color="tab:orange", rwidth=0.85)
plt.xlabel("Reward delivery time (s)")
plt.ylabel("Count")
plt.title("Reward delivery times")
plt.tight_layout()
plt.savefig("explore/behavior_reward_time_hist.png")
plt.close()

io.close()