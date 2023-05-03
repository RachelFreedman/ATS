#%%
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from cycler import cycler


# make plots look nice
# -- Axes --
rcParams['axes.spines.bottom'] = True
rcParams['axes.spines.left'] = False
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.grid'] = True
rcParams['axes.grid.axis'] = 'y'
rcParams['grid.color'] = 'grey'
rcParams['grid.linewidth'] = 0.5
rcParams['axes.axisbelow'] = True
rcParams['axes.linewidth'] = 2
rcParams['axes.ymargin'] = 0
# -- Ticks and tick labels --
rcParams['axes.edgecolor'] = 'grey'
rcParams['xtick.color'] = 'grey'
rcParams['ytick.color'] = 'grey'
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 0
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 0
# -- Fonts --
rcParams['font.size'] = 12
rcParams['font.family'] = 'serif'
rcParams['text.color'] = 'grey'
rcParams['axes.labelcolor'] = 'grey'
# -- Figure size --
rcParams['figure.figsize'] = (10, 7)
# -- Saving Options --
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.dpi'] = 500
rcParams['savefig.transparent'] = True
# -- Plot Styles --
rcParams['lines.linewidth'] = 3
navy = (56 / 256, 74 / 256, 143 / 256)
teal = (106 / 256, 197 / 256, 179 / 256)
pink = [199 / 255, 99 / 255, 150 / 255]
rcParams['axes.prop_cycle'] = cycler(color=[teal, navy, pink])

B_ACTIONS = ["B1", "B2", "B3", "B"]
C_ACTIONS = ["C1", "C2", "C3"]
ACTIONS = B_ACTIONS + C_ACTIONS

class State:
    def __init__(self, t, u, d, b):
        self.t = t
        self.u = u
        self.d = d
        self.b = b
        
        assert len(u) == len(d[0]) == len(b)
        self.arm_rewards = np.dot(d,u)
    
    def __str__(self) -> str:
        return f"State(t={self.t}, u={self.u}, d={self.d}, b={self.b})"
    
    def __repr__(self) -> str:
        return f"State(t={self.t}, u={self.u}, d={self.d}, b={self.b})"
        
    def __eq__(self, other) -> bool:
        return np.all(self.u == other.u) and np.all(self.d == other.d) and np.all(self.b == other.b)
    
    def __hash__(self) -> int:
        return hash((tuple(self.u), tuple(self.d.flatten()), tuple(self.b)))
    
    def best_arm(self) -> str:
        return f'C{np.argmax(self.arm_rewards)+1}'
    
    def u_error(self, other) -> float:
        return np.linalg.norm(self.u - other.u)
    
    def d_error(self, other) -> float:
        return np.linalg.norm(self.d - other.d)
    
    def arm_reward_error(self, other) -> float:
        return np.linalg.norm(self.arm_rewards - other.arm_rewards)
    
    def parse(s: str):
        t = int(re.search(r"\d+", s).group())
        u = np.array(eval(re.search(r"\[.*?\]", s).group()))
        d = np.array(eval(re.search(r"Array{Float64}(\[.*?\]\])", s).group(1)))
        b = np.array(eval(re.findall(r'\[.*?\]', s)[-1]))
        return State(t, u, d, b)

def import_csv(ids: list[str], runs: int) -> pd.DataFrame:
    """Import data from csv file.

    Args:
        ids (list{str}): list of ids
        runs (int): number of runs

    Returns:
        pd.DataFrame: dataframe with data
    """
    data = {}
    for id in ids:
        data[id] = {}
        for run in range(runs):
            filename = f"../data/sims/{id}_run{str(run+1)}.txt"
            with open(filename, "r") as f:
                state = State.parse(f.readline())

                df = pd.read_csv(f, sep=",", header=None)
                df.columns = ["step", "a_type", "a", "o", "r"]
                df["id"] = id
                df["run"] = run + 1
                df["state"] = state
                data[id][run] = df

    return data

def get_action_proportions(data: dict[dict[pd.DataFrame]]) -> pd.DataFrame:
    """Get proportion of actions over ids and runs.

    Args:
        data (dict[dict[pd.DataFrame]]): data

    Returns:
        pd.DataFrame: dataframe with proportions
    """
    df = pd.DataFrame()
    for id in data.keys():
        for run in data[id].keys():
            df = df.append(data[id][run][["step", "a"]], ignore_index=True)
    df = df.groupby(["step", "a"]).size().unstack(fill_value=0)
    for action in ACTIONS:
        if action not in df.columns:
            df[action] = 0
    df = df[ACTIONS].div(df.sum(axis=1), axis=0)
    return df

def get_action_counts(data: dict[dict[pd.DataFrame]]) -> pd.DataFrame:
    """Get count of actions in list, summed over steps.

    Args:
        data (dict[dict[pd.DataFrame]]): data

    Returns:
        pd.DataFrame: dataframe with counts
    """
    df = pd.DataFrame()
    for id in data.keys():
        for run in data[id].keys():
            df = df.append(data[id][run][["id", "run", "a"]], ignore_index=True)
    df = df.groupby(["id", "run", "a"]).size().reset_index(name="count")
    return df

def r_avg_std(data: dict[dict[pd.DataFrame]]) -> pd.DataFrame:
    """Average and calculate the standard deviation for rewards over ids and runs.

    Args:
        data ([dict[dict[pd.DataFrame]]]): data

    Returns:
        avg (pd.DataFrame): dataframe with average rewards
        std (pd.DataFrame): dataframe with standard deviation of rewards
    """
    df = pd.DataFrame()
    for id in data.keys():
        id_df = pd.DataFrame()
        for run in data[id].keys():
            id_df = id_df.append(data[id][run][["step", "r"]], ignore_index=True)
        df = df.append(id_df, ignore_index=True)   
    avg = df.groupby("step").mean()
    std = df.groupby("step").std()
    return avg, std

def plot_r(data: list[dict[dict[pd.DataFrame]]], labels: list[str], discount: float = 1.0, window: int = 10):
    """Plot average cumulative discounted rewards, averaged over ids and runs.

    Args:
        data (list[dict[dict[pd.DataFrame]]]): data
        labels (list[str]): labels for legend
        discount (float, optional): discount factor. Defaults to 1.0.
        window (int, optional): window size for smoothing. Defaults to 10.

    Returns:
        None
    """
    for i, d in enumerate(data):
        avg, std = r_avg_std(d)
        avg["r_cum"] = (avg["r"] * (discount ** avg.index)).cumsum()
        avg["r_smooth"] = avg["r_cum"].rolling(window).mean()
        std["r_smooth"] = std["r"].rolling(window).mean()

        plt.plot(avg.index, avg["r_smooth"], label=labels[i])
        plt.fill_between(avg.index, avg["r_smooth"] - std["r_smooth"], avg["r_smooth"] + std["r_smooth"], alpha=0.2)
        
    plt.xlabel(f"Step (smoothed over {window} steps)")
    plt.ylabel(f"Reward (discount={discount})")
    plt.title(f"Average Cumulative Reward")
    plt.legend()    

def plot_actions(data: dict[dict[pd.DataFrame]], title: str = None, window: int = 10):
    """Plot proportion of actions in list over ids and runs.

    Args:
        data (dict[dict[pd.DataFrame]]): data
        labels (str): title
        window (int, optional): window size for smoothing. Defaults to 10.

    Returns:
        None
    """
    df = get_action_proportions(data)
    df = df.rolling(window).mean()
    for a in ACTIONS:
        plt.plot(df.index, df[a], label=a)

    plt.ylim(0,1)
    plt.xlabel(f"Step (smoothed over {window} steps)")
    plt.ylabel(f"Action Proportion")
    plt.legend()
    if title:
        plt.title(title)

def plot_actions_in_list(data: list[dict[dict[pd.DataFrame]]], actions: list[str], labels: list[str], window: int = 10):
    """Plot proportion of actions in list over ids and runs.

    Args:
        data (list[dict[dict[pd.DataFrame]]]): data
        actions (list[int]): list of actions
        labels (list[str]): labels for legend
        window (int, optional): window size for smoothing. Defaults to 10.

    Returns:
        None
    """
    for i, d in enumerate(data):
        df = get_action_proportions(d)
        df = df[actions].sum(axis=1)
        df = df.rolling(window).mean()
        plt.plot(df.index, df, label=labels[i])

    plt.ylim(0,1)
    plt.xlabel(f"Step (smoothed over {window} steps)")
    plt.ylabel(f"Action Proportion")
    plt.title(f"Actions in List {actions}")
    plt.legend()

def plot_teacher_actions(data: list[dict[dict[pd.DataFrame]]], labels: list[str], window: int = 10):
    plot_actions_in_list(data, B_ACTIONS, labels, window)

def plot_best_arm_actions(data: list[dict[dict[pd.DataFrame]]], labels: list[str], window: int = 10):
    plot_actions_in_list(data, [C_ACTIONS[0]], labels, window)

def a_boxplot(data: list[dict[dict[pd.DataFrame]]], action: str, labels: list[str]):
    """Plot boxplot of action counts for each experiment in data, combined over ids and runs
    x axis: experiment
    y axis: count
    box: interquartile range
    middle line: median

    Args:
        data (list[dict[dict[pd.DataFrame]]]): list of data for each experiment
        action (str): action to plot
        labels (list[str]): labels for legend

    Returns:
        None
    """
    df = pd.DataFrame()
    for i, d in enumerate(data):
        experiment_name = labels[i]
        df_exp = get_action_counts(d)
        df_exp = df_exp[df_exp["a"] == action]
        df_exp["experiment"] = experiment_name
        df = df.append(df_exp, ignore_index=True)
    
    sns.boxplot(x="experiment", y="count", data=df)
    plt.ylim(0, max(df["count"])*1.1)
    plt.title(f"Action {action}")
    plt.ylabel("count")
    plt.show()

def a_hist(data: dict[dict[pd.DataFrame]], title: str = None):
    """Plot histogram of actions taken over ids and runs.

    Args:
        data (dict[dict[pd.DataFrame]]): data
        title (str, optional): title for plot. Defaults to None.

    Returns:
        None
    """
    df = pd.DataFrame()
    for id in data.keys():
        for run in data[id].keys():
            df = df.append(data[id][run][["step", "a"]], ignore_index=True)
    df = df.groupby(["a"]).size()
    plt.bar(df.index, df)
    plt.xlabel("Action")
    plt.xticks(rotation=0)
    plt.ylim(0, max(df)*1.1)
    if title:
        plt.title(f"Action Histogram ({title})")



#%%
# example experiments to import
# runs = 5
# ids1 = ["active_infinite_23419_153118", "active_infinite_23419_153235"]
# ids2 = ["active_infinite_23419_153118"]
# ids3 = ["active_infinite_23419_153235"]

# data1 = import_csv(ids1, runs)
# data2 = import_csv(ids2, runs)
# data3 = import_csv(ids3, runs)
# plot_r([data1, data2, data3], ["both", "first", "second"], discount=.99, window=10)       
# plot_r([data1, data2, data3], ["both", "first", "second"], discount=1., window=10)       
# a_hist(data3, "second")      


# %%
