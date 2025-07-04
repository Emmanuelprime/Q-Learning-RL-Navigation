import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import random
from matplotlib.widgets import Button, TextBox, CheckButtons

plt.switch_backend('TkAgg')

# Default parameters
DEFAULT_GRID_SIZE = 10
DEFAULT_START = (0, 0)
DEFAULT_GOAL = (9, 9)
DEFAULT_OBSTACLES = [(2, 2), (3, 2), (4, 2), (5, 5), (6, 5), (7, 5), (8, 5), (5, 6), (5, 7), (5, 8)]
DEFAULT_EPISODES = 2000

# Global variables
GRID_SIZE = DEFAULT_GRID_SIZE
START = DEFAULT_START
GOAL = DEFAULT_GOAL
OBSTACLES = DEFAULT_OBSTACLES.copy()
EPISODES = DEFAULT_EPISODES
obstacle_checkboxes = []
manual_obstacle_mode = True  # ✅ Flag to know if we're using manual checkboxes

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

# Colors
ROBOT_COLOR = '#FF00FF'
PATH_COLOR = '#1f77b4'
OPTIMAL_PATH_COLOR = '#2ca02c'
OBSTACLE_COLOR = '#7f7f7f'

# Hyperparameters
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 1.0
EPSILON_DECAY = 0.999
MAX_STEPS = 200
FRAME_SKIP = 5

# Data containers
frames = []
robot_positions = []
explored_paths = []
ani = None
fig = None
ax = None

def is_valid_position(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_position(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    return (x + dx, y + dy)

def train_robot():
    global EPSILON, frames, robot_positions, explored_paths
    frames = []
    robot_positions = []
    explored_paths = []

    Q = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
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
            action = choose_action(state, EPSILON, Q)
            next_state = get_next_position(state, action)

            if not is_valid_position(next_state):
                reward = -100
                next_state = state
            else:
                reward = get_reward(next_state)

            update_q_table(state, action, next_state, reward, Q)
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

    return Q

def update_q_table(state, action, next_state, reward, Q):
    x, y = state
    nx, ny = next_state
    best_next_action = np.argmax(Q[nx, ny])
    Q[x, y, action] += ALPHA * (reward + GAMMA * Q[nx, ny, best_next_action] - Q[x, y, action])

def choose_action(state, epsilon, Q):
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
        penalty = 0
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if (dx, dy) != (0, 0):
                    neighbor = (sx + dx, sy + dy)
                    if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                        if neighbor in OBSTACLES:
                            dist = abs(dx) + abs(dy)
                            penalty -= 5 if dist == 1 else 2
        distance_penalty = (abs(gx - sx) + abs(gy - sy)) * 0.01
        return -1 - distance_penalty + penalty

def plot_rewards(reward_list):
    fig, ax = plt.subplots()
    ax.plot(reward_list, label='Reward')
    ax.set_title("Rewards over Episodes")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    plt.show()

def find_optimal_path(Q):
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

def update_visualization():
    global ax
    ax.clear()
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.grid(True)

    for obs in OBSTACLES:
        rect = plt.Rectangle((obs[1], obs[0]), 1, 1, facecolor=OBSTACLE_COLOR, hatch='//', edgecolor='black')
        ax.add_patch(rect)

    ax.add_patch(plt.Circle((START[1]+0.5, START[0]+0.5), 0.4, color='green'))
    ax.add_patch(plt.Circle((GOAL[1]+0.5, GOAL[0]+0.5), 0.4, color='red'))

    plt.draw()

def animate(i, Q):
    global ani, ax

    updated_artists = []

    # Remove previous robot(s)
    for artist in ax.patches[:]:
        if isinstance(artist, plt.Circle) and artist.get_facecolor()[:3] == (1.0, 0.0, 1.0):  # Magenta (ROBOT_COLOR)
            artist.remove()

    if i < len(frames):
        path, current_pos = frames[i]

        # Draw the explored path
        if len(path) > 1:
            y_coords, x_coords = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in path])
            line, = ax.plot(x_coords, y_coords, color=PATH_COLOR, linewidth=2, alpha=0.5)
            updated_artists.append(line)

        # Draw the robot
        robot = plt.Circle((current_pos[1] + 0.5, current_pos[0] + 0.5), 0.3, color=ROBOT_COLOR)
        ax.add_patch(robot)
        updated_artists.append(robot)

    # At the end of animation, draw optimal path
    if i == len(frames) - 1:
        optimal_path = find_optimal_path(Q)
        if len(optimal_path) > 1:
            y_opt, x_opt = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
            opt_line, = ax.plot(x_opt, y_opt, color=OPTIMAL_PATH_COLOR, linewidth=3, linestyle='-')
            updated_artists.append(opt_line)
        ani.event_source.stop()

    ax.set_title(f'Episode {i//10 + 1} - Step {i % 10 * FRAME_SKIP}')
    return updated_artists


def generate_random_obstacles(event):
    global OBSTACLES, obstacle_checkboxes, manual_obstacle_mode
    try:
        count = int(obstacle_count_text.text)
    except:
        count = 10

    max_attempts = 100
    attempts = 0

    while attempts < max_attempts:
        candidate = random.sample([(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)
                                   if (x, y) != START and (x, y) != GOAL], count)
        OBSTACLES = candidate
        if path_exists():
            break
        attempts += 1

    if attempts == max_attempts:
        print("❌ Could not generate valid map")
        return

    # ✅ Clear old checkboxes
    for cb in obstacle_checkboxes:
        cb.disconnect_events()
        cb.ax.clear()
    obstacle_checkboxes.clear()
    manual_obstacle_mode = False  # ✅ Now we're in random mode

    update_visualization()
    print("✅ Random obstacles generated")

def path_exists():
    visited = set()
    queue = [START]

    while queue:
        current = queue.pop(0)
        if current == GOAL:
            return True
        for dx, dy in ACTIONS.values():
            next_pos = (current[0] + dx, current[1] + dy)
            if is_valid_position(next_pos) and next_pos not in visited:
                visited.add(next_pos)
                queue.append(next_pos)
    return False

def run_simulation(event):
    global START, GOAL, OBSTACLES, EPISODES, ani

    try:
        sx, sy = map(int, start_text.text.split(','))
        gx, gy = map(int, goal_text.text.split(','))
        START = (sx, sy)
        GOAL = (gx, gy)
        EPISODES = int(episodes_text.text)
    except:
        print("❌ Invalid input")
        return

    if manual_obstacle_mode:
        OBSTACLES.clear()
        for i, (x, y) in enumerate(DEFAULT_OBSTACLES):
            if i < len(obstacle_checkboxes) and obstacle_checkboxes[i].get_status()[0]:
                OBSTACLES.append((x, y))

    update_visualization()
    print("Training robot...")
    Q = train_robot()

    ani = FuncAnimation(fig, lambda i: animate(i, Q), frames=len(frames), interval=200, blit=True)
    plt.tight_layout()
    plt.show()

def create_obstacle_checkboxes():
    global obstacle_checkboxes
    for i, (x, y) in enumerate(DEFAULT_OBSTACLES):
        ax_check = plt.axes([0.1 + (i % 10)*0.08, 0.02 + (i // 10)*0.05, 0.05, 0.05])
        checkbox = CheckButtons(ax_check, [f"({x},{y})"], [True])
        obstacle_checkboxes.append(checkbox)

def reset_simulation(event=None):
    global START, GOAL, OBSTACLES, EPISODES, manual_obstacle_mode

    START = DEFAULT_START
    GOAL = DEFAULT_GOAL
    OBSTACLES = DEFAULT_OBSTACLES.copy()
    EPISODES = DEFAULT_EPISODES
    manual_obstacle_mode = True

    start_text.set_val(f"{START[0]},{START[1]}")
    goal_text.set_val(f"{GOAL[0]},{GOAL[1]}")
    episodes_text.set_val(str(EPISODES))
    obstacle_count_text.set_val("10")

    for cb in obstacle_checkboxes:
        cb.disconnect_events()
        cb.ax.clear()
    obstacle_checkboxes.clear()
    create_obstacle_checkboxes()

    update_visualization()

# UI Setup
fig, ax = plt.subplots(figsize=(12, 8))
plt.subplots_adjust(bottom=0.35)
update_visualization()

ax_start = plt.axes([0.1, 0.25, 0.2, 0.05])
start_text = TextBox(ax_start, 'Start (x,y):', f"{START[0]},{START[1]}")

ax_goal = plt.axes([0.4, 0.25, 0.2, 0.05])
goal_text = TextBox(ax_goal, 'Goal (x,y):', f"{GOAL[0]},{GOAL[1]}")

ax_episodes = plt.axes([0.7, 0.25, 0.2, 0.05])
episodes_text = TextBox(ax_episodes, 'Episodes:', str(EPISODES))

ax_obstacle_count = plt.axes([0.1, 0.20, 0.2, 0.05])
obstacle_count_text = TextBox(ax_obstacle_count, 'Obstacle Count:', "10")

ax_generate = plt.axes([0.4, 0.20, 0.2, 0.05])
generate_button = Button(ax_generate, "Generate Random Obstacles")
generate_button.on_clicked(generate_random_obstacles)

ax_run = plt.axes([0.4, 0.30, 0.2, 0.05])
run_button = Button(ax_run, 'Run Simulation')
run_button.on_clicked(run_simulation)

ax_reset = plt.axes([0.7, 0.30, 0.2, 0.05])
reset_button = Button(ax_reset, 'Reset')
reset_button.on_clicked(reset_simulation)

create_obstacle_checkboxes()
plt.show()
