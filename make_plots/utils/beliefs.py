#%%
import re
import json
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
import matplotlib.pyplot as plt
from cycler import cycler
import time

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
c1 = [164 / 255, 36 / 255, 59 / 255]
c2 = [39 / 255, 62 / 255, 71 / 255]
c3 = [221 / 255, 115 / 255, 59 / 255]
c4 = [8 / 255, 126 / 255, 139 / 255]
c5 = [221 / 255, 155 / 255, 29 / 255]
c6 = [107 / 255, 139 / 255, 85 / 255]
c7 = [189 / 255, 99 / 255, 47 / 255]

rcParams['axes.prop_cycle'] = cycler(color=[c1, c2, c3, c4, c5, c6, c7])

 # %%
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

# experiments with belief distribution
# ai: active infinite
ids_ai = ["active_infinite_23422_131424", "active_infinite_23419_153324", "active_infinite_23419_154231", "active_infinite_23419_155139", "active_infinite_23419_160549", "active_infinite_23419_161648", "active_infinite_23419_162359", "active_infinite_23419_163115", "active_infinite_23419_16401", "active_infinite_23419_164925", "active_infinite_23419_17049", "active_infinite_23419_171313", "active_infinite_23419_172230", "active_infinite_23419_172929", "active_infinite_23419_17394", "active_infinite_23419_174759", "active_infinite_23419_180013", "active_infinite_23419_181126", "active_infinite_23419_182029", "active_infinite_23419_183010"]

# af: active finite
ids_af = ["active_finite_23419_214856", "active_finite_23419_22017", "active_finite_23419_22124", "active_finite_23419_222216", "active_finite_23419_223627", "active_finite_23419_22505", "active_finite_23419_230052", "active_finite_23419_230927", "active_finite_23419_232011", "active_finite_23419_233116", "active_finite_23419_23426", "active_finite_23419_235054", "active_finite_23420_000135", "active_finite_23420_001238", "active_finite_23420_002515", "active_finite_23420_004012", "active_finite_23420_00540", "active_finite_23420_010552", "active_finite_23420_011651", "active_finite_23420_012749"]

# pi: passive infinite
ids_pi = ["passive_infinite_23419_183940", "passive_infinite_23419_184948", "passive_infinite_23419_18573", "passive_infinite_23419_191144", "passive_infinite_23419_193217", "passive_infinite_23419_194023", "passive_infinite_23419_194838", "passive_infinite_23419_195741", "passive_infinite_23419_200444", "passive_infinite_23419_201343", "passive_infinite_23419_202155", "passive_infinite_23419_203025", "passive_infinite_23419_203934", "passive_infinite_23419_205227", "passive_infinite_23419_205917", "passive_infinite_23419_210814", "passive_infinite_23419_211748", "passive_infinite_23419_21258", "passive_infinite_23419_213346", "passive_infinite_23419_214029"]

# pf: passive finite
ids_pf = ["passive_finite_23422_141255", "passive_finite_23422_142713", "passive_finite_23422_143917", "passive_finite_23422_145237", "passive_finite_23422_150344", "passive_finite_23422_151540", "passive_finite_23422_152710", "passive_finite_23422_153914", "passive_finite_23422_155019", "passive_finite_23422_16003", "passive_finite_23422_161248", "passive_finite_23422_162128", "passive_finite_23422_163156", "passive_finite_23422_164250", "passive_finite_23422_165540", "passive_finite_23422_17041", "passive_finite_23422_17148", "passive_finite_23422_172518", "passive_finite_23422_173548", "passive_finite_23422_174423"]

# experiments with explicit state estimate
# n50: naive-50
ids_n50 = ["naive_50_23420_013912", "naive_50_23420_013928", "naive_50_23420_013944", "naive_50_23420_01401", "naive_50_23420_014017", "naive_50_23420_014034", "naive_50_23420_014050", "naive_50_23420_01417", "naive_50_23420_014123", "naive_50_23420_014139", "naive_50_23420_014156", "naive_50_23420_014213", "naive_50_23420_014230", "naive_50_23420_014247", "naive_50_23420_01433", "naive_50_23420_014320", "naive_50_23420_014336", "naive_50_23420_014352", "naive_50_23420_01448", "naive_50_23420_014424"]

# n100: naive-100
ids_n100 = ["naive_100_23420_014440", "naive_100_23420_014457", "naive_100_23420_014513", "naive_100_23420_014529", "naive_100_23420_014545", "naive_100_23420_01462", "naive_100_23420_014618", "naive_100_23420_014634", "naive_100_23420_014650", "naive_100_23420_01476", "naive_100_23420_014722", "naive_100_23420_014739", "naive_100_23420_014755", "naive_100_23420_014811", "naive_100_23420_014827", "naive_100_23420_014843", "naive_100_23420_01490", "naive_100_23420_014916", "naive_100_23420_014932", "naive_100_23420_014948"]

# n200: naive-200
ids_n200 = ["naive_200_23420_01504", "naive_200_23420_015021", "naive_200_23420_015037", "naive_200_23420_015053", "naive_200_23420_01519", "naive_200_23420_015125", "naive_200_23420_015142", "naive_200_23420_015158", "naive_200_23420_015215", "naive_200_23420_015231", "naive_200_23420_015247", "naive_200_23420_01534", "naive_200_23420_015320", "naive_200_23420_015336", "naive_200_23420_015352", "naive_200_23420_01549", "naive_200_23420_015425", "naive_200_23420_015441", "naive_200_23420_015457", "naive_200_23420_015514"]


def import_jsons(ids: list[list[str]], exps: list[str]) -> dict:
    data = {}
    for i in range(len(exps)):
        data[exps[i]] = {}
        for id in ids[i]:
            with open(f"../data/beliefs/{id}.json", "r") as f:
                data[exps[i]][id] = json.load(f)
    return data

def convert_keys_to_state(d: list[dict]):
    return [{State.parse(k): v for k, v in x.items()} for x in d]

def convert_keys_to_state_from_dict(d: dict[dict[list[list[dict]]]]):
    data = {}
    for exp,d2 in d.items():
        start = time.time()
        data[exp] = {}
        for id,l3 in d2.items():
            data[exp][id] = list(map(convert_keys_to_state, l3))
        end = time.time()
        print(f"converting {exp} took {round((end-start)/60,2)} minutes")
    return data
    
def converged_states(d: dict[dict[list[list[dict]]]]):
    data = {}
    for exp,d2 in d.items():
        data[exp] = {}
        for id,l3 in d2.items():
            data[exp][id] = list(map(lambda x: max(x[-1], key=x[-1].get), l3))
    return data

def converged_state_probs(d: dict[dict[list[list[dict]]]], converged_states: dict[dict[list]]):
    data = {}
    for exp,d2 in d.items():
        data[exp] = {}
        for id,l3 in d2.items():
            data[exp][id] = [[d5.get(converged_states[exp][id][i], 0) for d5 in l3[i]] for i in range(len(l3))]
    return data

def import_true_states(ids: list[list[str]], exps: list[str]) -> dict:
    states = {}
    for i in range(len(exps)):
        states[exps[i]] = {}
        for id in ids[i]:
            with open(f"../data/logs/{id}.txt", "r") as f:
                s = f.read()
                if "naive" in id:
                    states[exps[i]][id] = State.parse(re.findall(r"true state (.*)\n", s)[0])
                else:
                    states[exps[i]][id] = State.parse(re.findall(r"hardcoded state: (.*)\n", s)[0])
    return states

def import_state_estimates(ids: list[list[str]], exps: list[str]) -> dict:
    # Naive doesn't estimate these
    est_t = -1
    est_B = [-1, -1, -1]

    # Extract estimates for U and D
    states = {}
    for i in range(len(exps)):
        states[exps[i]] = {}
        for id in ids[i]:
            with open(f"../data/logs/{id}.txt", "r") as f:
                s = f.read()
                est_U = np.asarray(list((map(eval, re.findall("Estimated U: (.*)\n", s)))))
                est_D = np.asarray(list((map(eval, re.findall("Estimated D: Any(.*)\n", s)))))
                est_states = [State(est_t, x[0], x[1], est_B) for x in zip(est_U, est_D)]
                states[exps[i]][id] = est_states
    return states

def u_error_boxplot(est: dict[dict[list[State]]], true: dict[dict[State]]):
    df = pd.DataFrame()
    for exp, d in est.items():
        for id, l in d.items():
            for i, state in enumerate(l):
                df = df.append({"experiment": exp, "id": id, "run": i, "u_error": state.u_error(true[exp][id])}, ignore_index=True)
    sns.boxplot(x="experiment", y="u_error", data=df)
    plt.ylim(0, max(df["u_error"])*1.1)
    plt.title(f"U error")
    plt.ylabel("U error")
    plt.show()

def d_error_boxplot(est: dict[dict[list[State]]], true: dict[dict[State]]):
    df = pd.DataFrame()
    for exp, d in est.items():
        for id, l in d.items():
            for i, state in enumerate(l):
                df = df.append({"experiment": exp, "id": id, "run": i, "d_error": state.d_error(true[exp][id])}, ignore_index=True)
    sns.boxplot(x="experiment", y="d_error", data=df)
    plt.ylim(0, max(df["d_error"])*1.1)
    plt.title(f"D error")
    plt.ylabel("D error")
    plt.show()

def arm_reward_error_boxplot(est: dict[dict[list[State]]], true: dict[dict[State]]):
    df = pd.DataFrame()
    for exp, d in est.items():
        for id, l in d.items():
            for i, state in enumerate(l):
                df = df.append({"experiment": exp, "id": id, "run": i, "arm_reward_error": state.arm_reward_error(true[exp][id])}, ignore_index=True)
    sns.boxplot(x="experiment", y="arm_reward_error", data=df)
    plt.ylim(0, max(df["arm_reward_error"])*1.1)
    plt.title(f"arm reward error")
    plt.ylabel("arm reward error")
    plt.show()

def plot_confidence_by_experiment(confidence: dict[dict[list[list[dict]]]]):
    '''
        confidence: dict[dict[list[list[dict]]]]
            confidence[exp][id][run][step] = probability of converged state
        title: str
    '''
    fig, ax = plt.subplots()
    for exp in confidence.keys():
        mean = np.mean(np.asarray([np.mean(np.asarray(confidence[exp][id]), axis=0) for id in confidence[exp].keys()]), axis=0)
        ax.plot(mean, label=exp)
    ax.set_xlabel("step")
    ax.set_ylabel("confidence")
    ax.set_title(f"confidence over time")
    ax.legend()
    plt.show()
