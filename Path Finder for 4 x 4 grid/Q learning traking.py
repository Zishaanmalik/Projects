import random
import matplotlib.pyplot as plt
import seaborn as sns

# Environment settings
rows, cols = 4, 4
actions = ['up', 'down', 'left', 'right']
goal = (3, 3)

# Q-table initialization
Q = {}
for r in range(rows):
    for c in range(cols):
        Q[(r, c)] = {a: 0.0 for a in actions}

# Learning hyperparameters
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate
episodes = 50  # keep small for easy output


# Movement function (step)
def step(state, action):
    r, c = state
    if action == 'up':
        r = max(0, r - 1)
    elif action == 'down':
        r = min(rows - 1, r + 1)
    elif action == 'left':
        c = max(0, c - 1)
    elif action == 'right':
        c = min(cols - 1, c + 1)

    next_state = (r, c)
    reward = 10 if next_state == goal else -1
    done = (next_state == goal)
    return next_state, reward, done


# Plot initialization
fig, ax = plt.subplots(figsize=(13, 7))  # Bigger plot

# tracker list
path_x, path_y = [], []

def draw_env(agent_pos):
    ax.clear()
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(-0.5, rows - 0.5)
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.grid(True)

    # Draw all states with their max Q-values
    for r in range(rows):
        for c in range(cols):
            s = (r, c)
            max_q = max(Q[s].values())
            color = "pink"
            size = 180  #  larger default dot size
            if s == (0, 0):  # start
                color = "blue"
                size = 220
            if s == goal:  # goal
                color = "red"
                size = 260
            if s == agent_pos:  # agent current position
                color = "green"
                size = 300

            ax.scatter(c, rows - 1 - r, c=color, s=size)
            ax.text(c, rows - 1 - r, f"{max_q:.1f}",
                    ha="center", va="center",
                    color="black", fontsize=14, fontweight="bold")  # bigger visible font
            ax.plot(path_x, path_y, color="orange", linewidth=2, linestyle="--", alpha=0.8)


    plt.pause(0.00003)


# Training
for episode in range(1, episodes + 1):
    state = (0, 0)
    done = False
    path_x, path_y = [], []  # reset tracker per episode
    print(f"\n=== Episode {episode} ===")

    while not done:
        # Choose action (epsilon-greedy)
        if random.random() < epsilon:
            action = random.choice(actions)
        else:
            action = max(Q[state], key=Q[state].get)

        next_state, reward, done = step(state, action)
        old_q = Q[state][action]
        max_future_q = max(Q[next_state].values())

        # Q-learning update
        new_q = old_q + alpha * (reward + gamma * max_future_q - old_q)
        Q[state][action] = new_q

        # Print detailed step info
        print(f"State: {state} | Action: {action} | Reward: {reward} | Next: {next_state}")
        print(f"Old Q: {old_q:.2f} -> New Q: {new_q:.2f}")

        # Update path tracker
        path_x.append(state[1])
        path_y.append(rows - 1 - state[0])

        # Draw environment with agent position + updated Qs
        draw_env(next_state)

        state = next_state

plt.show()

# Final Q-table display
print("\n=== Final Q-table ===")
for state in Q:
    print(f"State {state}: ", end="")
    for action in actions:
        print(f"{action}: {Q[state][action]:.2f}  ", end="")
    print()
