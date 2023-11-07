# packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
import seaborn as sns

# Plot Time Series and Corresponding VG
def plot_time_series_VG(adj_matrix: np.ndarray, data: np.ndarray, times: np.ndarray = None, horizontal: bool = False):
  if times is None:
    times = np.arange(len(data))

  fig, axs = plt.subplots(1, 2, figsize=(12,3))

  # Plot Connections and Time Series
  for i in range(len(data)):
      for j in range(i, len(data)):
          if adj_matrix[i, j] == 1:
              if horizontal:
                  axs[0].plot([times[i], times[j]], [data[i], data[i]], color='red', alpha=0.8)
                  axs[0].plot([times[i], times[j]], [data[j], data[j]], color='red', alpha=0.8)
              else:
                  axs[0].plot([times[i], times[j]], [data[i], data[j]], color='red', alpha=0.8)
  axs[0].plot(times, data, color='r', alpha=0.8)
  # axs[0].get_xaxis().set_ticks(list(times))
#   axs[0].bar(times, data, width=0.1)


# Plot Visibility Graph
  for i in range(len(data)):
      axs[1].plot(times[i], 0, marker='o', markersize=50/len(data), color='orange')

  for i in range(len(data)):
      for j in range(i, len(data)):
          if adj_matrix[i, j] == 1.0:
              Path = mpath.Path
              mid_time = (times[j] + times[i]) / 2.
              diff = abs(times[j] - times[i])
              pp1 = mpatches.PathPatch(Path([(times[i], 0), (mid_time, diff), (times[j], 0)],[Path.MOVETO, Path.CURVE3, Path.CURVE3]), fc="none", transform=axs[1].transData, alpha=0.5)
              axs[1].add_patch(pp1)
  axs[1].get_yaxis().set_ticks([])
  # axs[1].get_xaxis().set_ticks(list(times))
  sns.despine()
  plt.show()