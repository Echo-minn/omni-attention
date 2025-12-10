import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ["512/64", "512/128", "1024/64", "1024/128", "2048/64", "2048/128"]

# tflops = {
#     "prefetch": [1.26, 1.25, 5.48, 4.78, 14.85, 15.64],
#     "shared_kv": [3.40, 2.78, 9.64, 10.47, 21.84, 23.20],
#     "swizzle": [3.49, 3.84, 14.10, 13.64, 33.73, 33.57],
# }

# speedup = {
#     "prefetch": [0.64, 0.93, 0.69, 0.74, 0.53, 0.53],
#     "shared_kv": [1.74, 2.08, 1.21, 1.63, 0.78, 0.78],
#     "swizzle": [1.78, 2.87, 1.78, 2.12, 1.20, 1.13],
# }

# tflops = {
#     "prefetch": [1.29, 1.15, 4.45, 4.44, 12.03, 13.08],
#     "shared_kv": [4.27, 3.49, 10.39, 9.36, 19.86, 20.83],
#     "swizzle": [4.76, 4.41, 11.40, 10.68, 29.88, 26.77],
# }

# speedup = {
#     "prefetch": [0.65, 0.80, 0.49, 0.60, 0.47, 0.54],
#     "shared_kv": [2.15, 2.43, 1.14, 1.27, 0.78, 0.87],
#     "swizzle": [2.40, 3.08, 1.25, 1.45, 1.17, 1.11],
# }

# tflops = {
#     "prefetch": [1.49, 1.44, 4.78, 5.06, 12.64, 13.62],
#     "shared_kv": [4.21, 3.64, 9.25, 9.81, 18.95, 23.42],
#     "swizzle": [4.57, 4.35, 14.39, 12.47, 31.69, 32.27],
# }

# speedup = {
#     "prefetch": [0.79, 0.67, 0.64, 0.68, 0.44, 0.43],
#     "shared_kv": [2.23, 1.68, 1.23, 1.31, 0.66, 0.74],
#     "swizzle": [2.42, 2.01, 1.92, 1.67, 1.10, 1.02],
# }

# tflops = {
#     "prefetch": [1.38, 1.09, 4.33, 4.73, 9.69, 14.26],
#     "shared_kv": [4.11, 3.04, 9.31, 9.84, 18.04, 23.48],
#     "swizzle": [4.62, 3.38, 13.19, 12.24, 28.17, 33.60],
# }

# speedup = {
#     "prefetch": [1.02, 0.55, 0.54, 0.71, 0.39, 0.48],
#     "shared_kv": [3.03, 1.55, 1.15, 1.48, 0.73, 0.79],
#     "swizzle": [3.41, 1.72, 1.64, 1.84, 1.14, 1.14],
# }

speedup = {
    "document": [2.68, 2.48, 2.14, 2.14, 1.05, 1.04],
    "interleaved": [3.41, 1.72, 1.64, 1.84, 1.14, 1.14],
    "causal": [2.42, 2.01, 1.92, 1.67, 1.10, 1.02],
    "F_P": [1.78, 2.87, 1.78, 2.12, 1.20, 1.13],
    "F_C_P": [2.40, 3.08, 1.25, 1.45, 1.17, 1.11],
}

x = np.arange(len(labels))
width = 0.25

fig, ax1 = plt.subplots(figsize=(14, 7))


# Speedup lines (draw after bars to be on top)
ax1.plot(labels, speedup["document"], zorder=10)
ax1.plot(labels, speedup["interleaved"], zorder=10)
ax1.plot(labels, speedup["causal"], zorder=10)
ax1.plot(labels, speedup["F_P"], zorder=10)
ax1.plot(labels, speedup["F_C_P"], zorder=10)
ax1.set_ylabel("Speedup")
ax1.set_xlabel("Seq_len / block_size")
ax1.set_ylim(0, max(max(s) for s in speedup.values()) * 1.2)
ax1.axhline(1.0, color="pink", linestyle="--", linewidth=1.5, label="1.0× baseline")

plt.xticks(x, labels)
plt.title("Speedup comparison")

plt.tight_layout()
plt.show()

# Display legend at the top left corner
ax1.legend(["document", "interleaved", "causal", "F_P", "F_C_P"], loc="upper left")


# save the plot to a file
plt.savefig("images/speedup_comparison_mask.png")