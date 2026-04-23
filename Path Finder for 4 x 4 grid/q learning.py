import random
import time

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
episodes = 10  # keep small for easy output


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


# Training
for episode in range(1, episodes + 1):
    state = (0, 0)
    done = False
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

        state = next_state

# Final Q-table display
print("\n=== Final Q-table ===")
for state in Q:
    print(f"State {state}: ", end="")
    for action in actions:
        print(f"{action}: {Q[state][action]:.2f}  ", end="")
    print()