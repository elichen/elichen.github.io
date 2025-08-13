#!/usr/bin/env python3
"""
Create GIF of best game from 10 plays using latest weights
"""

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import imageio.v3 as iio
from train_perfect_snake import EnhancedSnakeEnv, HybridSnakeDQN

def play_ten_games_and_save_best_gif():
    # Load model - use best_model.pth for best weights
    checkpoint = torch.load('perfect_snake_models/best_model.pth', weights_only=False)
    print(f"Model from episode {checkpoint['episode']} with best score {checkpoint['best_score']}")
    
    model = HybridSnakeDQN(
        state_size=checkpoint['state_size'],
        action_size=checkpoint['action_size'],
        vision_size=9
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Play 10 games and find the best one
    best_score = -1
    best_game_data = None
    
    print("Playing 10 games to find the best one...")
    for game_idx in range(10):
        # Create environment for this game
        env = EnhancedSnakeEnv(grid_size=14, vision_radius=4)
        state = env.reset()
        model.reset_hidden()
        
        # Record this game's data
        game_frames = []
        game_states = []
        
        # Create figure for this game
        fig, ax = plt.subplots(figsize=(8, 8))
        
        step_count = 0
        max_steps = 500
        done = False
        
        while not done and step_count < max_steps:
            # Get action
            state_tensor = torch.FloatTensor(state)
            with torch.no_grad():
                q_values = model(state_tensor)
            
            if random.random() < 0.01:
                action = random.randint(0, 3)
            else:
                action = np.argmax(q_values.numpy())
            
            # Take step
            state, reward, done = env.step(action)
            step_count += 1
            
            # Record EVERY frame - no skipping
            # Draw frame
            ax.clear()
            ax.set_xlim(-0.5, env.grid_size - 0.5)
            ax.set_ylim(-0.5, env.grid_size - 0.5)
            ax.set_title(f'Snake AI - Score: {env.score}', fontsize=14)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            
            # Remove axis labels and ticks
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
        
            # Draw snake
            for i, segment in enumerate(env.snake):
                if i == 0:  # Head
                    color = 'darkgreen'
                    rect = Rectangle((segment[0] - 0.4, segment[1] - 0.4), 0.8, 0.8,
                                   facecolor=color, edgecolor='black', linewidth=2)
                else:  # Body
                    color = 'green'
                    rect = Rectangle((segment[0] - 0.4, segment[1] - 0.4), 0.8, 0.8,
                                   facecolor=color, edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            # Draw food
            if env.food:
                food_rect = Rectangle((env.food[0] - 0.3, env.food[1] - 0.3), 0.6, 0.6,
                                     facecolor='red', edgecolor='darkred', linewidth=2)
                ax.add_patch(food_rect)
            
            # Convert matplotlib figure to image array
            fig.canvas.draw()
            
            # Get the RGBA buffer from the figure
            width, height = fig.canvas.get_width_height()
            buf = fig.canvas.buffer_rgba()
            
            # Convert to numpy array
            image = np.frombuffer(buf, dtype=np.uint8)
            
            # Handle retina displays (2x resolution)
            if len(image) == width * height * 4 * 4:
                # Retina display - reshape and downsample
                image = image.reshape((height * 2, width * 2, 4))
                # Take only RGB channels and downsample
                image = image[::2, ::2, :3]
            else:
                image = image.reshape((height, width, 4))
                # Take only RGB channels
                image = image[:, :, :3]
            
            # Make sure we have a copy of the array, not a view
            image = np.array(image, copy=True)
            
            # Add frame to list
            game_frames.append(image)
            
            if done:
                break
        
        plt.close()
        
        # Check if this game is the best so far
        final_score = env.score
        print(f"Game {game_idx + 1}: Score {final_score}, Steps: {step_count}, Frames: {len(game_frames)}")
        
        if final_score > best_score:
            best_score = final_score
            best_game_data = {
                'frames': game_frames,
                'score': final_score,
                'steps': step_count,
                'game_num': game_idx + 1
            }
    
    # Use the best game for the GIF
    if best_game_data:
        frames = best_game_data['frames']
        print(f"\nBest game was #{best_game_data['game_num']} with score {best_game_data['score']}")
        print(f"Total steps: {best_game_data['steps']}")
        print(f"Total frames collected: {len(frames)}")
        
        # Save as animated GIF using v3 API at 60fps
        if frames:
            print(f"Saving {len(frames)} frames to GIF at 60fps...")
            # 60fps = ~16.67ms per frame
            iio.imwrite('snake_60fps.gif', frames, duration=16.67, loop=0)
            print("✅ Saved to snake_60fps.gif")
            
            # Verify the GIF
            gif_check = iio.imread('snake_60fps.gif')
            print(f"Verification: GIF shape: {gif_check.shape}")
            print(f"Duration: {len(frames) * 16.67:.1f}ms total ({len(frames) / 60:.1f} seconds at 60fps)")
        else:
            print("❌ No frames to save!")
    else:
        print("❌ No valid games played!")

if __name__ == "__main__":
    play_ten_games_and_save_best_gif()