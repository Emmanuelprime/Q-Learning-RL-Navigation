import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import random

plt.switch_backend('TkAgg')

# Environment setup
GRID_SIZE = 10
START = (0, 0)
GOAL = (9, 9)
OBSTACLES = [(2, 2), (3, 2), (4, 2), (5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8)]

ACTIONS = {
    0: (-1, 0),   # up
    1: (1, 0),    # down
    2: (0, -1),   # left
    3: (0, 1),    # right
    4: (-1, -1),  # up-left
    5: (-1, 1),   # up-right
    6: (1, -1),   # down-left
    7: (1, 1),    # down-right
}


Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPISODES = 2000
MAX_STEPS = 200
FRAME_SKIP = 5

# Colors
ROBOT_COLOR = '#FF00FF'
PATH_COLOR = '#1f77b4'
OPTIMAL_PATH_COLOR = '#2ca02c'
OBSTACLE_COLOR = '#7f7f7f'

# Data containers
frames = []
robot_positions = []
explored_paths = []
ani = None

def is_valid_position(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_position(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    return (x + dx, y + dy)

def plot_rewards(reward_list):
    plt.figure(figsize=(10, 5))
    plt.plot(reward_list, label='Reward per Episode', color='blue', alpha=0.6)
    plt.title("Learning Curve - Total Reward per Episode")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def train_robot():
    global EPSILON
    success_count = 0
    reward_per_episode = []
    successful_episodes = []

    for episode in range(EPISODES):
        state = START
        path = [state]
        visited = set()
        done = False
        steps = 0
        total_reward = 0
        episode_frames = []

        while not done and steps < MAX_STEPS:
            action = choose_action(state, EPSILON)
            next_state = get_next_position(state, action)

            if not is_valid_position(next_state):
                reward = -100
                next_state = state
            else:
                reward = get_reward(next_state)

            update_q_table(state, action, next_state, reward)
            state = next_state
            path.append(state)
            visited.add(state)
            total_reward += reward

            if len(path) % FRAME_SKIP == 0:
                episode_frames.append((list(path), state))

            if state == GOAL:
                done = True
                success_count += 1
                print(f"✅ Episode {episode+1}: Reached Goal in {steps+1} steps.")
            elif state in OBSTACLES:
                done = True

            steps += 1

        EPSILON = max(0.01, EPSILON * EPSILON_DECAY)
        reward_per_episode.append(total_reward)

        if state == GOAL:
            successful_episodes.append(episode_frames)

    print(f"\nTraining completed. Goal reached in {success_count} out of {EPISODES} episodes.")
    plot_rewards(reward_per_episode)

    selected_episodes = random.sample(successful_episodes, min(10, len(successful_episodes)))
    for ep in selected_episodes:
        for frame in ep:
            frames.append(frame)
            robot_positions.append(frame[1])
            explored_paths.append(frame[0])

def update_q_table(state, action, next_state, reward):
    x, y = state
    nx, ny = next_state
    best_next_action = np.argmax(Q[nx, ny])
    Q[x, y, action] = Q[x, y, action] + ALPHA * (
        reward + GAMMA * Q[nx, ny, best_next_action] - Q[x, y, action]
    )

def choose_action(state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(list(ACTIONS.keys()))
    else:
        x, y = state
        return np.argmax(Q[x, y])

def get_reward(state):
    if state == GOAL:
        return 100
    elif state in OBSTACLES:
        return -100
    else:
        gx, gy = GOAL
        sx, sy = state

        # Penalize being near obstacles (up to 2 cells away)
        penalty = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (dx, dy) != (0, 0):
                    neighbor = (sx + dx, sy + dy)
                    if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                        if neighbor in OBSTACLES:
                            dist = abs(dx) + abs(dy)
                            penalty -= 5 if dist == 1 else 2  # Stronger penalty when closer

        distance_penalty = (abs(gx - sx) + abs(gy - sy)) * 0.01
        return -1 - distance_penalty + penalty


def find_optimal_path():
    path = [START]
    state = START
    visited = set()

    for _ in range(100):
        x, y = state
        action = np.argmax(Q[x, y])
        next_state = get_next_position(state, action)

        if not is_valid_position(next_state) or next_state in visited:
            break

        path.append(next_state)
        visited.add(next_state)
        if next_state == GOAL:
            break
        state = next_state

    return path

def init_visualization():
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

    for obs in OBSTACLES:
        rect = plt.Rectangle((obs[1], obs[0]), 1, 1,
                             linewidth=2, edgecolor='black',
                             facecolor=OBSTACLE_COLOR, hatch='//')
        ax.add_patch(rect)

    ax.add_patch(plt.Circle((START[1]+0.5, START[0]+0.5), 0.4, color='green'))
    ax.add_patch(plt.Circle((GOAL[1]+0.5, GOAL[0]+0.5), 0.4, color='red'))
    ax.text(START[1]+0.5, START[0]+0.5, 'S', ha='center', va='center', color='white', fontweight='bold')
    ax.text(GOAL[1]+0.5, GOAL[0]+0.5, 'G', ha='center', va='center', color='white', fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor='green', label='Start (S)'),
        mpatches.Patch(facecolor='red', label='Goal (G)'),
        mpatches.Patch(facecolor=OBSTACLE_COLOR, hatch='//', label='Obstacle'),
        Line2D([0], [0], color=PATH_COLOR, lw=2, label='Explored Path'),
        Line2D([0], [0], color=OPTIMAL_PATH_COLOR, lw=2, label='Optimal Path'),
        mpatches.Patch(facecolor=ROBOT_COLOR, label='Robot')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    return fig, ax

def animate(i):
    global ani

    for artist in ax.collections + ax.patches:
        if isinstance(artist, plt.Circle) and artist.get_facecolor() == (1.0, 0.0, 1.0, 1.0):
            artist.remove()

    if i < len(frames):
        path, current_pos = frames[i]
        if len(path) > 1:
            y_coords, x_coords = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in path])
            ax.plot(x_coords, y_coords, color=PATH_COLOR, linewidth=2, alpha=0.5)

        robot = plt.Circle((current_pos[1] + 0.5, current_pos[0] + 0.5), 0.3, color=ROBOT_COLOR)
        ax.add_patch(robot)

    if i == len(frames) - 1:
        optimal_path = find_optimal_path()
        if len(optimal_path) > 1:
            y_opt, x_opt = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
            ax.plot(x_opt, y_opt, color=OPTIMAL_PATH_COLOR, linewidth=3, linestyle='-')
        ani.event_source.stop()

    ax.set_title(f'Episode {i//10 + 1} - Step {i % 10 * FRAME_SKIP}')
    return []

# Main execution
if __name__ == "__main__":
    print("Training robot...")
    train_robot()

    print("Preparing visualization...")
    fig, ax = init_visualization()

    print("Creating animation...")
    ani = FuncAnimation(fig, animate, frames=len(frames), interval=200, blit=True)
    plt.rcParams['animation.embed_limit'] = 100

    print("Displaying animation...")
    plt.tight_layout()
    plt.show()