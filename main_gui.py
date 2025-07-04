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
obstacle_checkboxes = []

def is_valid_position(pos):
    x, y = pos
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE and pos not in OBSTACLES

def get_next_position(state, action):
    x, y = state
    dx, dy = ACTIONS[action]
    return (x + dx, y + dy)

def plot_rewards(reward_list):
    reward_fig, reward_ax = plt.subplots(figsize=(10, 5))
    reward_ax.plot(reward_list, label='Reward per Episode', color='blue', alpha=0.6)
    reward_ax.set_title("Learning Curve - Total Reward per Episode")
    reward_ax.set_xlabel("Episode")
    reward_ax.set_ylabel("Total Reward")
    reward_ax.grid(True)
    reward_ax.legend()
    reward_fig.tight_layout()
    plt.show()

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
    Q[x, y, action] = Q[x, y, action] + ALPHA * (
        reward + GAMMA * Q[nx, ny, best_next_action] - Q[x, y, action]
    )

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

def init_visualization():
    global fig, ax
    
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.25)
    
    ax.set_xticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_yticks(np.arange(0, GRID_SIZE + 1, 1))
    ax.set_xlim(0, GRID_SIZE)
    ax.set_ylim(0, GRID_SIZE)
    ax.set_aspect('equal')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)

    update_visualization()
    
    return fig, ax

def update_visualization():
    global ax
    
    ax.clear()
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
    
    plt.draw()

def animate(i, Q):
    global ani, ax

    # Clear previous robot position
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
        optimal_path = find_optimal_path(Q)
        if len(optimal_path) > 1:
            y_opt, x_opt = zip(*[(p[0] + 0.5, p[1] + 0.5) for p in optimal_path])
            ax.plot(x_opt, y_opt, color=OPTIMAL_PATH_COLOR, linewidth=3, linestyle='-')
        ani.event_source.stop()

    ax.set_title(f'Episode {i//10 + 1} - Step {i % 10 * FRAME_SKIP}')
    return []

def reset_simulation():
    global START, GOAL, OBSTACLES, EPISODES, obstacle_checkboxes
    
    # Reset to default values
    START = DEFAULT_START
    GOAL = DEFAULT_GOAL
    OBSTACLES = DEFAULT_OBSTACLES.copy()
    EPISODES = DEFAULT_EPISODES
    
    # Update text boxes with default values
    start_text.set_val(f"{START[0]},{START[1]}")
    goal_text.set_val(f"{GOAL[0]},{GOAL[1]}")
    episodes_text.set_val(str(EPISODES))
    
    # Clear and recreate obstacle checkboxes
    for checkbox in obstacle_checkboxes:
        checkbox.disconnect_events()
        checkbox.ax.clear()
    
    obstacle_checkboxes.clear()
    create_obstacle_checkboxes()
    
    update_visualization()

def create_obstacle_checkboxes():
    global obstacle_checkboxes
    
    # Create checkboxes for each default obstacle
    for i, (x, y) in enumerate(DEFAULT_OBSTACLES):
        ax_check = plt.axes([0.1 + (i % 10)*0.08, 0.02 + (i // 10)*0.05, 0.05, 0.05])
        checkbox = CheckButtons(ax_check, [f"({x},{y})"], [True])
        obstacle_checkboxes.append(checkbox)

def run_simulation(event):
    global ani, frames, robot_positions, explored_paths, START, GOAL, OBSTACLES, EPISODES
    
    # Get values from UI
    try:
        start_coords = list(map(int, start_text.text.split(',')))
        if len(start_coords) == 2 and is_valid_position((start_coords[0], start_coords[1])):
            START = (start_coords[0], start_coords[1])
        else:
            raise ValueError("Invalid start position")
            
        goal_coords = list(map(int, goal_text.text.split(',')))
        if len(goal_coords) == 2 and is_valid_position((goal_coords[0], goal_coords[1])):
            GOAL = (goal_coords[0], goal_coords[1])
        else:
            raise ValueError("Invalid goal position")
            
        EPISODES = int(episodes_text.text)
        if EPISODES <= 0:
            raise ValueError("Episodes must be positive")
            
    except Exception as e:
        print(f"Error in input: {e}")
        return
    
    # Update obstacles based on checkboxes
    OBSTACLES = []
    for i, (x, y) in enumerate(DEFAULT_OBSTACLES):
        if i < len(obstacle_checkboxes) and obstacle_checkboxes[i].get_status()[0]:
            OBSTACLES.append((x, y))
    
    update_visualization()
    
    print("Training robot...")
    Q = train_robot()
    
    print("Creating animation...")
    ani = FuncAnimation(fig, lambda i: animate(i, Q), frames=len(frames), interval=200, blit=True)
    plt.rcParams['animation.embed_limit'] = 100
    
    print("Displaying animation...")
    plt.tight_layout()
    plt.show()

# Create the UI
fig, ax = init_visualization()

# Create control panel
ax_start = plt.axes([0.1, 0.15, 0.2, 0.05])
start_text = TextBox(ax_start, 'Start (x,y):', f"{START[0]},{START[1]}")

ax_goal = plt.axes([0.4, 0.15, 0.2, 0.05])
goal_text = TextBox(ax_goal, 'Goal (x,y):', f"{GOAL[0]},{GOAL[1]}")

ax_episodes = plt.axes([0.7, 0.15, 0.2, 0.05])
episodes_text = TextBox(ax_episodes, 'Episodes:', str(EPISODES))

# Create obstacle checkboxes
create_obstacle_checkboxes()

ax_run = plt.axes([0.4, 0.2, 0.2, 0.05])
run_button = Button(ax_run, 'Run Simulation')
run_button.on_clicked(run_simulation)

ax_reset = plt.axes([0.7, 0.2, 0.2, 0.05])
reset_button = Button(ax_reset, 'Reset to Defaults')
reset_button.on_clicked(lambda event: reset_simulation())

plt.show()