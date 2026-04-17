import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
import numpy as np

sns.set(style="whitegrid")
os.makedirs("figures", exist_ok=True)


def plot_fig1():
    df = pd.read_csv("results/exp1_problem_TSP/tsp_1000cities_comparison.csv")

    summary = df.groupby("solver", as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    solver_order = ["Serial SA", "Baseline A", "Baseline B", "AIPSA-GM (guided)"]
    summary["solver"] = pd.Categorical(summary["solver"], categories=solver_order, ordered=True)
    summary = summary.sort_values("solver")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=summary, x="solver", y="mean_cost", palette="Blues_d")

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    serial_mean = summary[summary["solver"] == "Serial SA"]["mean_cost"].values[0]
    baselineB_mean = summary[summary["solver"] == "Baseline B"]["mean_cost"].values[0]
    aipsa_mean = summary[summary["solver"] == "AIPSA-GM (guided)"]["mean_cost"].values[0]

    improve_vs_serial = (serial_mean - aipsa_mean) / serial_mean * 100
    improve_vs_B = (baselineB_mean - aipsa_mean) / baselineB_mean * 100

    x_pos = list(summary["solver"]).index("AIPSA-GM (guided)")

    plt.text(
        x_pos - 0.1,
        aipsa_mean * 1.05,
        f"-{improve_vs_serial:.1f}% vs Serial\n-{improve_vs_B:.1f}% vs Baseline B",
        ha="center",
        fontsize=10,
        color="darkblue",
        fontweight="bold"
    )

    plt.title("Exp1: TSP Solver Comparison (1000 Cities)")
    plt.xlabel("Solver")
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=15)
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.tight_layout()
    plt.savefig("figures/fig1_exp1_tsp.png")
    plt.close()


def plot_fig2():
    df = pd.read_csv("results/exp1_problem_RASTRIGIN/rastrigin_10dims_comparison.csv")

    summary = df.groupby("solver", as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    solver_order = ["Serial SA", "Baseline A", "Baseline B", "AIPSA-GM (guided)"]
    summary["solver"] = pd.Categorical(summary["solver"], categories=solver_order, ordered=True)
    summary = summary.sort_values("solver")

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(data=summary, x="solver", y="mean_cost", palette="Oranges_d")

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    serial = summary[summary["solver"] == "Serial SA"]["mean_cost"].values[0]
    aipsa = summary[summary["solver"] == "AIPSA-GM (guided)"]["mean_cost"].values[0]
    improve = (serial - aipsa) / serial * 100

    x_pos = list(summary["solver"]).index("AIPSA-GM (guided)")

    plt.text(
        x_pos,
        aipsa * 1.1,
        f"-{improve:.1f}% vs Serial",
        ha="center",
        fontsize=10,
        color="darkred",
        fontweight="bold"
    )

    plt.title("Exp1: Rastrigin Solver Comparison (10 Dimensions)")
    plt.xlabel("Solver")
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=15)
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.1f}")
    )
    plt.tight_layout()
    plt.savefig("figures/fig2_exp1_rastrigin.png")
    plt.close()

def plot_fig3():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp2_problem_TSP/tsp_2000cities_migration_policy.csv")

    policy_col = "policy" if "policy" in df.columns else "solver"

    summary = df.groupby(policy_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    summary["policy"] = summary[policy_col].str.replace("AIPSA-GM (", "").str.replace(")", "")

    order = ["random", "best_only", "quality_only", "guided"]
    summary["policy"] = pd.Categorical(summary["policy"], categories=order, ordered=True)
    summary = summary.sort_values("policy")

    plt.figure(figsize=(7,5))
    ax = sns.barplot(data=summary, x="policy", y="mean_cost", palette="Blues_d")

    ax.errorbar(range(len(summary)), summary["mean_cost"], yerr=summary["std_cost"],
                fmt="none", c="black", capsize=5)

    plt.title("Exp2: TSP Migration Policy (2000 Cities)")
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=15)

    plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    plt.tight_layout()
    plt.savefig("figures/fig3_exp2_tsp.png")
    plt.close()

def plot_fig4():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("results/exp2_problem_RASTRIGIN_dims_10_ring/rastrigin_10dims_migration_policy.csv")

    policy_col = "policy" if "policy" in df.columns else "solver"

    summary = df.groupby(policy_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    summary["policy"] = summary[policy_col].str.replace("AIPSA-GM (", "").str.replace(")", "")

    order = ["random", "best_only", "quality_only", "guided"]
    summary["policy"] = pd.Categorical(summary["policy"], categories=order, ordered=True)
    summary = summary.sort_values("policy")

    plt.figure(figsize=(7,5))
    ax = sns.barplot(data=summary, x="policy", y="mean_cost", palette="Oranges_d")

    ax.errorbar(range(len(summary)), summary["mean_cost"], yerr=summary["std_cost"],
                fmt="none", c="black", capsize=5)

    plt.title("Exp2: Rastrigin 10D Migration Policy")
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("figures/fig4_exp2_rastrigin10.png")
    plt.close()

def plot_fig5():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("results/exp2_problem_RASTRIGIN_dims_30_ring/rastrigin_30dims_migration_policy.csv")

    policy_col = "policy" if "policy" in df.columns else "solver"

    summary = df.groupby(policy_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    summary["policy"] = summary[policy_col].str.replace("AIPSA-GM (", "").str.replace(")", "")

    order = ["random", "best_only", "quality_only", "guided"]
    summary["policy"] = pd.Categorical(summary["policy"], categories=order, ordered=True)
    summary = summary.sort_values("policy")

    plt.figure(figsize=(7,5))
    ax = sns.barplot(data=summary, x="policy", y="mean_cost", palette="Reds_d")

    ax.errorbar(range(len(summary)), summary["mean_cost"], yerr=summary["std_cost"],
                fmt="none", c="black", capsize=5)

    # improvement 标注
    random_mean = summary[summary["policy"]=="random"]["mean_cost"].values[0]
    best_mean = summary[summary["policy"]=="best_only"]["mean_cost"].values[0]
    guided_mean = summary[summary["policy"]=="guided"]["mean_cost"].values[0]

    improve_random = (random_mean - guided_mean)/random_mean * 100
    improve_best = (best_mean - guided_mean)/best_mean * 100

    x_pos = list(summary["policy"]).index("guided")

    plt.text(
        x_pos,
        guided_mean * 1.2,
        f"-{improve_random:.1f}% vs random\n-{improve_best:.1f}% vs best",
        ha="center",
        fontsize=10,
        color="darkred",
        fontweight="bold"
    )

    plt.title("Exp2: Rastrigin 30D Migration Policy")
    plt.ylabel("Mean Cost")
    plt.xticks(rotation=15)

    plt.tight_layout()
    plt.savefig("figures/fig5_exp2_rastrigin30.png")
    plt.close()

def plot_fig6():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp3_problem_RASTRIGIN_10/rastrigin_10dims_adaptive_temp.csv")

    print(df.head())
    print(df.columns)

    solver_col = "solver"

    summary = df.groupby(solver_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    # 统一短名字
    def clean_name(x):
        x = str(x).lower()
        if "adaptive" in x:
            return "Adaptive"
        return "Fixed"

    summary["mode"] = summary["solver"].apply(clean_name)

    order = ["Fixed", "Adaptive"]
    summary["mode"] = pd.Categorical(summary["mode"], categories=order, ordered=True)
    summary = summary.sort_values("mode")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="mode",
        y="mean_cost",
        palette=["#FFB74D", "#1565C0"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    fixed_mean = summary[summary["mode"] == "Fixed"]["mean_cost"].values[0]
    adaptive_mean = summary[summary["mode"] == "Adaptive"]["mean_cost"].values[0]

    improve = (fixed_mean - adaptive_mean) / fixed_mean * 100

    x_pos = list(summary["mode"]).index("Adaptive")

    plt.text(
        x_pos,
        adaptive_mean * 1.12,
        f"-{improve:.1f}% vs Fixed",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    plt.title("Exp3: Rastrigin 10D — Adaptive vs Fixed Cooling")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig6_exp3_rastrigin_adaptive.png")
    plt.close()


def plot_fig7():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp3_problem_TSP_1000/tsp_1000cities_adaptive_temp.csv")

    print(df.head())
    print(df.columns)

    solver_col = "solver"

    summary = df.groupby(solver_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    def clean_name(x):
        x = str(x).lower()
        if "adaptive" in x:
            return "Adaptive"
        return "Fixed"

    summary["mode"] = summary["solver"].apply(clean_name)

    order = ["Fixed", "Adaptive"]
    summary["mode"] = pd.Categorical(summary["mode"], categories=order, ordered=True)
    summary = summary.sort_values("mode")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="mode",
        y="mean_cost",
        palette=["#1565C0", "#FFB74D"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    fixed_mean = summary[summary["mode"] == "Fixed"]["mean_cost"].values[0]
    adaptive_mean = summary[summary["mode"] == "Adaptive"]["mean_cost"].values[0]

    worse = (adaptive_mean - fixed_mean) / fixed_mean * 100

    x_pos = list(summary["mode"]).index("Fixed")

    plt.text(
        x_pos + 0.5,
        max(fixed_mean, adaptive_mean) * 1.03,
        f"Fixed better by {worse:.1f}%",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    plt.title("Exp3: TSP 1000 — Adaptive vs Fixed Cooling")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig7_exp3_tsp_adaptive.png")
    plt.close()

def plot_fig8a():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp4_problem_TSP_1000/tsp_1000cities_async_vs_sync.csv")

    print(df.head())
    print(df.columns)

    summary = df.groupby("solver", as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std"),
        mean_time=("time_sec", "mean")
    )

    def clean_name(x):
        x = str(x).lower()
        if "async" in x:
            return "Async"
        return "Sync"

    summary["mode"] = summary["solver"].apply(clean_name)

    order = ["Sync", "Async"]
    summary["mode"] = pd.Categorical(summary["mode"], categories=order, ordered=True)
    summary = summary.sort_values("mode")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="mode",
        y="mean_cost",
        palette=["#FFB74D", "#1565C0"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    sync_cost = summary[summary["mode"] == "Sync"]["mean_cost"].values[0]
    async_cost = summary[summary["mode"] == "Async"]["mean_cost"].values[0]
    sync_time = summary[summary["mode"] == "Sync"]["mean_time"].values[0]
    async_time = summary[summary["mode"] == "Async"]["mean_time"].values[0]

    quality_gain = (sync_cost - async_cost) / sync_cost * 100
    speedup = sync_time / async_time

    plt.text(
        1.1,
        max(sync_cost, async_cost) * 0.98,
        f"+{quality_gain:.1f}% quality\n{speedup:.2f}x faster",
        ha="left",
        va="top",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    plt.title("Exp4: TSP 1000 — Async vs Sync")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig8a_exp4_tsp_async_sync.png")
    plt.close()


def plot_fig8b():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp4_problem_RASTRIGIN_10/rastrigin_10dims_async_vs_sync.csv")

    print(df.head())
    print(df.columns)

    summary = df.groupby("solver", as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std"),
        mean_time=("time_sec", "mean")
    )

    def clean_name(x):
        x = str(x).lower()
        if "async" in x:
            return "Async"
        return "Sync"

    summary["mode"] = summary["solver"].apply(clean_name)

    order = ["Sync", "Async"]
    summary["mode"] = pd.Categorical(summary["mode"], categories=order, ordered=True)
    summary = summary.sort_values("mode")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="mode",
        y="mean_cost",
        palette=["#FFB74D", "#1565C0"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    sync_cost = summary[summary["mode"] == "Sync"]["mean_cost"].values[0]
    async_cost = summary[summary["mode"] == "Async"]["mean_cost"].values[0]
    sync_time = summary[summary["mode"] == "Sync"]["mean_time"].values[0]
    async_time = summary[summary["mode"] == "Async"]["mean_time"].values[0]

    quality_gain = (sync_cost - async_cost) / sync_cost * 100
    speed_ratio = async_time / sync_time

    plt.text(
        0.5,
        max(sync_cost, async_cost) * 1.04,
        f"+{quality_gain:.1f}% quality\n{speed_ratio:.2f}x slower",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    plt.title("Exp4: Rastrigin 10D — Async vs Sync")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig8b_exp4_rastrigin_async_sync.png")
    plt.close()


def plot_fig9():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    df = pd.read_csv("results/exp5_scale_TSP_full_part2/tsp_1000cities_scalability.csv")

    print(df.head())
    print(df.columns)

    islands = [2, 4, 8, 16]
    improvements = []

    for n in islands:
        baseb = df[(df["n_islands"] == n) & (df["solver"] == "Baseline B")]["mean_cost"].values[0]
        aipsa = df[(df["n_islands"] == n) & (df["solver"] == "AIPSA-GM (guided)")]["mean_cost"].values[0]
        improvements.append((baseb - aipsa) / baseb * 100)

    plt.figure(figsize=(7, 5))
    plt.plot(
        islands,
        improvements,
        "o-",
        color="#1565C0",
        linewidth=2.5,
        markersize=8
    )

    for x, y in zip(islands, improvements):
        plt.text(
            x,
            y + 0.5,
            f"{y:.1f}%",
            ha="center",
            fontsize=10,
            color="#1565C0",
            fontweight="bold"
        )

    plt.title("Exp4: Async Advantage Grows with Island Count (TSP)")
    plt.xlabel("Number of Islands")
    plt.ylabel("AIPSA-GM Improvement over Baseline B (%)")
    plt.xticks(islands)
    plt.ylim(0, max(improvements) + 5)

    plt.tight_layout()
    plt.savefig("figures/fig9_exp4_async_scalability.png")
    plt.close()

def plot_fig10():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp5_3TOPOLOGY_TSP/tsp_1000cities_topology.csv")

    print(df.head())
    print(df.columns)

    topo_col = "topology" if "topology" in df.columns else "solver"

    summary = df.groupby(topo_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    def clean_name(x):
        x = str(x).lower()
        if "ring" in x:
            return "ring"
        if "random" in x:
            return "random_k"
        return "full"

    summary["topology"] = summary[topo_col].apply(clean_name)

    order = ["ring", "full", "random_k"]
    summary["topology"] = pd.Categorical(summary["topology"], categories=order, ordered=True)
    summary = summary.sort_values("topology")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="topology",
        y="mean_cost",
        palette=["#43A047", "#1565C0", "#FFB74D"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    full_mean = summary[summary["topology"] == "full"]["mean_cost"].values[0]
    plt.text(
        1,
        full_mean * 1.03,
        "full best",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    plt.title("Exp5: TSP 1000 — Topology Comparison")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig10_exp5_tsp_topology.png")
    plt.close()


def plot_fig11():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp5_3TOPOLOGY_RASTRIGIN/rastrigin_10dims_topology.csv")

    print(df.head())
    print(df.columns)

    topo_col = "topology" if "topology" in df.columns else "solver"

    summary = df.groupby(topo_col, as_index=False).agg(
        mean_cost=("cost", "mean"),
        std_cost=("cost", "std")
    )

    def clean_name(x):
        x = str(x).lower()
        if "ring" in x:
            return "ring"
        if "random" in x:
            return "random_k"
        return "full"

    summary["topology"] = summary[topo_col].apply(clean_name)

    order = ["ring", "full", "random_k"]
    summary["topology"] = pd.Categorical(summary["topology"], categories=order, ordered=True)
    summary = summary.sort_values("topology")

    plt.figure(figsize=(7, 5))
    ax = sns.barplot(
        data=summary,
        x="topology",
        y="mean_cost",
        palette=["#43A047", "#1565C0", "#FFB74D"]
    )

    ax.errorbar(
        x=range(len(summary)),
        y=summary["mean_cost"],
        yerr=summary["std_cost"],
        fmt="none",
        c="black",
        capsize=5
    )

    ring_mean = summary[summary["topology"] == "ring"]["mean_cost"].values[0]
    plt.text(
        0,
        ring_mean * 1.12,
        "ring best",
        ha="center",
        fontsize=10,
        color="#43A047",
        fontweight="bold"
    )

    plt.title("Exp5: Rastrigin 10D — Topology Comparison")
    plt.xlabel("")
    plt.ylabel("Mean Cost")
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{x:.2f}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig11_exp5_rastrigin_topology.png")
    plt.close()


def plot_fig12():
    import pandas as pd
    import matplotlib.pyplot as plt

    df_ring = pd.read_csv("results/exp5_scale_ring_RASTRIGIN_part2/rastrigin_10dims_scalability.csv")
    df_full = pd.read_csv("results/exp5_scale_full_RASTRIGIN_part2/rastrigin_10dims_scalability.csv")
    df_random = pd.read_csv("results/exp5_scale_randomk_RASTRIGIN_part2/rastrigin_10dims_scalability.csv")

    print(df_ring.head())
    print(df_ring.columns)

    islands = [2, 4, 8, 16]

    def get_series(df):
        aipsa = df[df["solver"] == "AIPSA-GM (guided)"]
        costs = [aipsa[aipsa["n_islands"] == n]["mean_cost"].values[0] for n in islands]
        times = [aipsa[aipsa["n_islands"] == n]["mean_time"].values[0] for n in islands]
        return costs, times

    ring_costs, ring_times = get_series(df_ring)
    full_costs, full_times = get_series(df_full)
    random_costs, random_times = get_series(df_random)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    ax1.plot(islands, ring_costs, "o-", color="#43A047", linewidth=2.5, label="ring cost")
    ax1.plot(islands, full_costs, "o-", color="#1565C0", linewidth=2.5, label="full cost")
    ax1.plot(islands, random_costs, "o-", color="#FFB74D", linewidth=2.5, label="random_k cost")

    ax2.plot(islands, ring_times, "s--", color="#43A047", linewidth=1.5, label="ring time")
    ax2.plot(islands, full_times, "s--", color="#1565C0", linewidth=1.5, label="full time")
    ax2.plot(islands, random_times, "s--", color="#FFB74D", linewidth=1.5, label="random_k time")

    ax1.set_title("Exp5: Rastrigin Scalability — Topology Comparison")
    ax1.set_xlabel("Number of Islands")
    ax1.set_ylabel("Mean Cost")
    ax2.set_ylabel("Mean Time (s)")
    ax1.set_xticks(islands)

    ax2.annotate(
        "full time explosion",
        xy=(16, full_times[-1]),
        xytext=(10, full_times[-1] * 0.8),
        color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0")
    )

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/fig12_exp5_rastrigin_scalability.png")
    plt.close()


def plot_fig13():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/exp5_scale_TSP_full_part2/tsp_1000cities_scalability.csv")

    print(df.head())
    print(df.columns)

    islands = [2, 4, 8, 16]
    solvers = ["Serial SA", "Baseline A", "Baseline B", "AIPSA-GM (guided)"]
    colors = ["#9E9E9E", "#90CAF9", "#FFB74D", "#1565C0"]
    widths = [1.5, 1.5, 1.5, 3.0]

    plt.figure(figsize=(8, 5))

    for solver, color, width in zip(solvers, colors, widths):
        costs = [df[(df["n_islands"] == n) & (df["solver"] == solver)]["mean_cost"].values[0] for n in islands]
        plt.plot(islands, costs, "o-", color=color, linewidth=width, label=solver)

    serial_costs = [df[(df["n_islands"] == n) & (df["solver"] == "Serial SA")]["mean_cost"].values[0] for n in islands]
    aipsa_costs = [df[(df["n_islands"] == n) & (df["solver"] == "AIPSA-GM (guided)")]["mean_cost"].values[0] for n in islands]

    for n, sc, ac in zip(islands, serial_costs, aipsa_costs):
        imp = (sc - ac) / sc * 100
        plt.text(n + 0.2, ac * 1.01, f"-{imp:.0f}%", color="#1565C0", fontsize=9, fontweight="bold")

    plt.title("Exp5: TSP Scalability — All Solvers")
    plt.xlabel("Number of Islands")
    plt.ylabel("Mean Cost")
    plt.xticks(islands)
    plt.gca().yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )
    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("figures/fig13_exp5_tsp_scalability.png")
    plt.close()


def plot_fig14():
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.ticker as ticker

    df_full = pd.read_csv("results/exp2_problem_RASTRIGIN_dims_30_full/rastrigin_30dims_migration_policy.csv")
    df_ring = pd.read_csv("results/exp2_problem_RASTRIGIN_dims_30_ring/rastrigin_30dims_migration_policy.csv")

    def summarize(df):
        policy_col = "policy" if "policy" in df.columns else "solver"
        summary = df.groupby(policy_col, as_index=False).agg(
            mean_cost=("cost", "mean"),
            std_cost=("cost", "std")
        )

        def clean_name(x):
            x = str(x)
            x = x.replace("AIPSA-GM (", "").replace(")", "")
            return x

        summary["policy"] = summary[policy_col].apply(clean_name)
        order = ["random", "best_only", "quality_only", "guided"]
        summary["policy"] = pd.Categorical(summary["policy"], categories=order, ordered=True)
        summary = summary.sort_values("policy")
        return summary

    s_full = summarize(df_full)
    s_ring = summarize(df_ring)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def draw(ax, summary, title):
        sns.barplot(
            ax=ax,
            data=summary,
            x="policy",
            y="mean_cost",
            palette=["#9E9E9E", "#90CAF9", "#FFB74D", "#1565C0"]
        )
        ax.errorbar(
            x=range(len(summary)),
            y=summary["mean_cost"],
            yerr=summary["std_cost"],
            fmt="none",
            c="black",
            capsize=5
        )
        ax.set_title(title)
        ax.set_xlabel("")
        ax.set_ylabel("Mean Cost")
        ax.tick_params(axis="x", rotation=15)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))

    draw(axes[0], s_full, "Full Topology")
    draw(axes[1], s_ring, "Ring Topology")

    ring_random = s_ring[s_ring["policy"] == "random"]["mean_cost"].values[0]
    ring_guided = s_ring[s_ring["policy"] == "guided"]["mean_cost"].values[0]
    imp = (ring_random - ring_guided) / ring_random * 100

    axes[1].text(
        3,
        ring_guided * 1.15,
        f"-{imp:.1f}% vs random",
        ha="center",
        fontsize=10,
        color="#1565C0",
        fontweight="bold"
    )

    fig.suptitle("Exp2: Rastrigin 30D — Full vs Ring Topology")
    plt.tight_layout()
    plt.savefig("figures/fig14_exp2_full_vs_ring.png")
    plt.close()

def plot_fig15():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/gpu_rastrigin_60time_5runs/gpu_final_rastrigin.csv")

    print(df.head())
    print(df.columns)

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["n_dims"],
        df["quality_gain_pct"],
        "o-",
        color="#1565C0",
        linewidth=2.5,
        markersize=8
    )

    for _, row in df.iterrows():
        plt.text(
            row["n_dims"],
            row["quality_gain_pct"] + 1,
            f"{row['quality_gain_pct']:.1f}%",
            ha="center",
            fontsize=9,
            color="#1565C0",
            fontweight="bold"
        )

    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    plt.xscale("log")
    plt.xticks([10, 50, 100, 200, 500, 1000])
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.title("H5: GPU vs CPU — Rastrigin Quality Gain")
    plt.xlabel("Number of Dimensions (log scale)")
    plt.ylabel("Quality Gain (%)")

    plt.tight_layout()
    plt.savefig("figures/fig15_h5_rastrigin_gain.png")
    plt.close()


def plot_fig16():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/gpu_TSP_N_CHAINS_32_10runs/gpu_tsp_boltzmann.csv")

    print(df.head())
    print(df.columns)

    plt.figure(figsize=(9, 5))

    plt.plot(
        df["n_cities"],
        df["quality_gain_pct"],
        "o-",
        color="#1565C0",
        linewidth=2.5,
        markersize=8
    )

    for _, row in df.iterrows():
        plt.text(
            row["n_cities"],
            row["quality_gain_pct"] + 0.8,
            f"{row['quality_gain_pct']:.1f}%",
            ha="center",
            fontsize=9,
            color="#1565C0",
            fontweight="bold"
        )

    # inflection zone
    plt.axvspan(3000, 5000, alpha=0.12, color="orange")
    plt.annotate(
        "memory bandwidth\nsaturation zone",
        xy=(4000, 8),
        xytext=(5500, 5),
        fontsize=9,
        color="#E65100",
        arrowprops=dict(arrowstyle="->", color="#E65100")
    )

    plt.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.6)

    plt.title("H5: GPU vs CPU — TSP Quality Gain")
    plt.xlabel("Number of Cities")
    plt.ylabel("Quality Gain (%)")
    plt.gca().xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    plt.tight_layout()
    plt.savefig("figures/fig16_h5_tsp_gain.png")
    plt.close()


def plot_fig17():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np

    df_ras = pd.read_csv("results/gpu_rastrigin_60time_5runs/gpu_final_rastrigin.csv")
    df_tsp = pd.read_csv("results/gpu_TSP_N_CHAINS_32_10runs/gpu_tsp_boltzmann.csv")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Rastrigin ---
    ax = axes[0]
    ras_dims = [10, 1000]
    x = np.arange(len(ras_dims))
    width = 0.35

    cpu_vals = [df_ras[df_ras["n_dims"] == d]["cpu_cost"].values[0] for d in ras_dims]
    gpu_vals = [df_ras[df_ras["n_dims"] == d]["gpu_cost"].values[0] for d in ras_dims]
    gains = [df_ras[df_ras["n_dims"] == d]["quality_gain_pct"].values[0] for d in ras_dims]

    ax.bar(x - width/2, cpu_vals, width, label="CPU", color="#FFB74D")
    ax.bar(x + width/2, gpu_vals, width, label="GPU", color="#1565C0")

    for i, g in enumerate(gains):
        ax.text(x[i], max(cpu_vals[i], gpu_vals[i]) * 1.03, f"+{g:.1f}%",
                ha="center", color="#1565C0", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"dims={d}" for d in ras_dims])
    ax.set_title("Rastrigin")
    ax.set_ylabel("Mean Cost")

    # --- TSP ---
    ax = axes[1]
    tsp_sizes = [500, 10000]
    x = np.arange(len(tsp_sizes))

    cpu_vals = [df_tsp[df_tsp["n_cities"] == d]["cpu_cost"].values[0] for d in tsp_sizes]
    gpu_vals = [df_tsp[df_tsp["n_cities"] == d]["gpu_cost"].values[0] for d in tsp_sizes]
    gains = [df_tsp[df_tsp["n_cities"] == d]["quality_gain_pct"].values[0] for d in tsp_sizes]

    ax.bar(x - width/2, cpu_vals, width, label="CPU", color="#FFB74D")
    ax.bar(x + width/2, gpu_vals, width, label="GPU", color="#1565C0")

    for i, g in enumerate(gains):
        ax.text(x[i], max(cpu_vals[i], gpu_vals[i]) * 1.03, f"+{g:.1f}%",
                ha="center", color="#1565C0", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:,}" for d in tsp_sizes])
    ax.set_title("TSP")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, _: f"{int(x):,}")
    )

    fig.suptitle("H5: GPU vs CPU — Cost Comparison at Representative Scales")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig("figures/fig17_h5_cost_comparison.png")
    plt.close()


def plot_fig18():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/gpu_sweep/chain_sweep.csv")

    print(df.head())
    print(df.columns)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()

    city_colors = {
        1000: "#1565C0",
        3000: "#43A047",
        5000: "#E65100"
    }

    for n_cities, color in city_colors.items():
        sub = df[df["n_cities"] == n_cities].sort_values("n_chains")

        ax1.plot(
            sub["n_chains"],
            sub["gain_pct"],
            "o-",
            color=color,
            linewidth=2.2,
            markersize=7,
            label=f"{n_cities:,} cities (gain)"
        )

        ax2.plot(
            sub["n_chains"],
            sub["iters_per_chain"],
            "s--",
            color=color,
            linewidth=1.4,
            markersize=5,
            alpha=0.6
        )

    ax1.axvline(x=32, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
    ax1.annotate(
        "optimal\nN_CHAINS=32",
        xy=(32, ax1.get_ylim()[1] * 0.85),
        xytext=(70, ax1.get_ylim()[1] * 0.7),
        color="red",
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="red")
    )

    ax1.set_xscale("log")
    ax1.set_xticks([32, 64, 128, 256, 512, 1024, 2048])
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())

    ax1.set_title("H5: N_CHAINS Sweep — Diversity vs Convergence Depth")
    ax1.set_xlabel("N_CHAINS (log scale)")
    ax1.set_ylabel("Quality Gain (%)")
    ax2.set_ylabel("Iters per Chain")

    ax1.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig("figures/fig18_h5_chain_sweep.png")
    plt.close()


def plot_fig19():
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    df = pd.read_csv("results/gpu_rastrigin_60time_5runs/gpu_final_rastrigin.csv")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["n_dims"],
        df["iter_ratio"],
        "o-",
        color="#1565C0",
        linewidth=2.5,
        markersize=8
    )

    for _, row in df.iterrows():
        plt.text(
            row["n_dims"],
            row["iter_ratio"] * 1.08,
            f"{row['iter_ratio']:.0f}x",
            ha="center",
            fontsize=9,
            color="#1565C0",
            fontweight="bold"
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xticks([10, 50, 100, 200, 500, 1000])
    plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter())

    plt.title("H5: GPU Throughput Advantage — Iter Ratio")
    plt.xlabel("Number of Dimensions (log scale)")
    plt.ylabel("Iter Ratio (GPU total iters / CPU iters, log scale)")

    plt.tight_layout()
    plt.savefig("figures/fig19_h5_iter_ratio.png")
    plt.close()

if __name__ == "__main__":
    plot_fig1()
    plot_fig2()
    plot_fig3()
    plot_fig4()
    plot_fig5()
    plot_fig6()
    plot_fig7()
    plot_fig8a()
    plot_fig8b()
    plot_fig9()
    plot_fig10()
    plot_fig11()
    plot_fig12()
    plot_fig13()
    plot_fig14()
    plot_fig15()
    plot_fig16()
    plot_fig17()
    plot_fig18()
    plot_fig19()

    print("fig1–fig19 generated successfully.")