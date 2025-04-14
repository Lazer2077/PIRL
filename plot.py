import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

def sum_fisrt_2000(values):
    return sum(values[:2000])

def load_scalar_from_event(log_dir, tag):
    # 获取事件文件路径
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "events.out.tfevents" in f]
    if not event_files:
        print(f"No event file found in {log_dir}")
        return [], []
    ea = event_accumulator.EventAccumulator(event_files[0],
                                            size_guidance={
                                                'scalars': 10000,
                                                'images': 0,
                                                'histograms': 0,
                                                'tensors': 0,
                                                'compressedHistograms': 0,
                                            })
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        print(f"Tag '{tag}' not found in {log_dir}")
        return [], []

    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return steps, values

# 指定你的 TensorBoard 日志目录
log_dir1 = "/mnt/d/RL/PIRL/LogTmp/04_13_19_33_pinnsac1_Hopper-v4"
log_dir2 = "/mnt/d/RL/PIRL/LogTmp/04_13_19_32_SAC_Hopper-v4"
# 读取数据
steps, values = load_scalar_from_event(log_dir1, "Episode/Reward")
steps2, values2 = load_scalar_from_event(log_dir2, "Episode/Reward")

# 获取 scalar 名称列表

# 读取并绘图
plt.figure(figsize=(12, 6))
tags = ["Episode/Reward", "Loss/Policy"]
# use poltly 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
PLOTY= True

if PLOTY:
    fig = go.Figure()   
    fig = make_subplots(rows=2, cols=1, subplot_titles=tags)

for i, tag in enumerate(tags):
    plt.subplot(1, 2, i+1)
    steps1, values1 = load_scalar_from_event(log_dir1, tag)
    steps2, values2 = load_scalar_from_event(log_dir2, tag)
    sum_fisrt_2000_values1 = sum_fisrt_2000(values1)
    sum_fisrt_2000_values2 = sum_fisrt_2000(values2)
    print(sum_fisrt_2000_values1, sum_fisrt_2000_values2)   
    if PLOTY:
        fig.add_trace(go.Scatter(x=steps1, y=values1, mode='lines', name=f"{tag} (Exp 1)"), row=i+1, col=1) 
        fig.add_trace(go.Scatter(x=steps2, y=values2, mode='lines', name=f"{tag} (Exp 2)"), row=i+1, col=1)
        fig.update_layout(title='Training Comparison', xaxis_title='Step', yaxis_title='Value')
    else:
        plt.plot(steps1, values1, label=f"{tag} (Exp 1)")
        plt.plot(steps2, values2, label=f"{tag} (Exp 2)", linestyle="--")
if PLOTY:
    fig.write_html("Training Curve.html")
    fig.show()
else:
    plt.xlabel("Step")
    plt.ylabel("Value")
    plt.title("Training Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
# plt.show()
    plt.savefig("Training Curve.png")