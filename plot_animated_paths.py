import argparse
import os
import pickle
import numpy as np
import plot

parser = argparse.ArgumentParser(description="Plot animated paths.")
parser.add_argument("--speeds", type=float, nargs="+", help="Speed of the agents.")
parser.add_argument("--num", type=int, help="Which solution to plot.")
args = parser.parse_args()


# with open("out/maps/intro.txt/1/paths.pkl", "rb") as f:
#     for i in range(args.num):
#         paths = pickle.load(f)
paths = [np.array([-1, 7, 8, 0, 1, 4, -2]), np.array([-1, 10, 6, 2, 5, 3, 9, -2])]
print(paths)


# with open("out/maps/1.txt/1/scores.pkl", "rb") as f:
#     for i in range(args.num):
#         scores = pickle.load(f)
scores = [11, -48, -20]
print(scores)


with open("maps/1.txt", "r") as f:
        lines = f.readlines()
        num_rewards = float(lines[0].split()[1])
        num_agents = int(lines[1].split()[1])
        _, = [float(lines[2].split()[1])]
        _ = [1] * int(num_agents)

        rpositions = np.array(
            [list(map(float, line.split()[:-1])) for line in lines[3:]]
        )
        rvalues = np.array([float(line.split()[2]) for line in lines[3:]])
        rpositions = np.append(rpositions[1:], [rpositions[0]], axis=0)
        rvalues = np.append(rvalues[1:], rvalues[0])


directory = f"imgs/animations/drive/"
os.makedirs(directory, exist_ok=True)
plot.plot_animated_paths(
    rpositions,
    rvalues,
    paths,
    scores,
    args.speeds,
    directory=directory,
    fname="animated_path",
    update_rewards=True,
)
