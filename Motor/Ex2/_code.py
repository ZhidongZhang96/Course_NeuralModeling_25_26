import math
import random
import sys

import numpy as np
import pygame

# ==== Parameters for different experiments 
test_mode = False # test_mode=False for recording of unbiased subjects 
subject_id = '3'
time_id = '2'   # exp id, espacially important for interference experiment

# Choose experimental setup 
# Baseline experiment: 'baseline'
# Generalization experiment: 'generalization'
# Interference experiment: 'interference'
exp_setup = 'interference' # 'baseline' (HW1),'generalization' (HW2,A),'interference' (HW2,B)
assert exp_setup != 'generalization', "Generalization experiment not implemented in this code."

if exp_setup == 'baseline':
    TARGET_ANGLE = -45      # degrees for baseline task
elif exp_setup == 'interference':
    if time_id == '1':
        TARGET_ANGLE = 135  # degrees for 1st interference task
    else:
        TARGET_ANGLE = 220  # degrees for 2nd interference task (non-interference)

TARGET_ANGLE_2_INTERFERENCE = 270  # degrees for 2nd interference task (interference)

# ====


# Total number of attempts in the experiment EDIT
from utils import ATTEMPTS_LIMIT


# Game parameters
SCREEN_X, SCREEN_Y = 1728, 1117  # your screen resolution
WIDTH, HEIGHT = SCREEN_X // 1.5, SCREEN_Y // 1.5  # be aware of monitor scaling on windows (150%)
CIRCLE_SIZE = 20
TARGET_SIZE = CIRCLE_SIZE
TARGET_RADIUS = 300
MASK_RADIUS = 0.66 * TARGET_RADIUS
START_POSITION = (WIDTH // 2, HEIGHT // 2)
PERTURBATION_ANGLE = 30
TIME_LIMIT = 1000  # time limit in ms

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Initialize Pygame
pygame.init()

# Set up the display
if test_mode:
    screen = pygame.display.set_mode((WIDTH-50, HEIGHT-50))
else:
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.FULLSCREEN)
pygame.display.set_caption("Reaching Game")

# Initialize game metrics
score = 0
attempts = 0
new_target = None
start_time = 0
move_faster = False
clock = pygame.time.Clock()

# Initialize game modes
mask_mode = True  # True: with mask; False: without mask
target_mode = 'fix'  # Mode for angular shift of target: random, fix, dynamic, sequence
start_target = math.radians(TARGET_ANGLE)  # initial target angle

sequence_target = TARGET_ANGLE
perturbation_mode = False
perturbation_type = 'sudden'  # Mode for angular shift of control: random, gradual or sudden
perturbation_angle = math.radians(PERTURBATION_ANGLE)  # Angle between mouse_pos and circle_pos
perturbed_mouse_angle = 0
gradual_step = 0
gradual_attempts = 0
perturbation_rand = random.uniform(-math.pi / 4, +math.pi / 4)

# Lists to store important angles per attempt
error_angles = []  
target_angles = []
circle_angles = []

failed_attempt_list = []

# Flag for showing mouse position and deltas
show_mouse_info = False


# Function to generate a new target position
def generate_target_position():
    if target_mode == 'random':
        angle = random.uniform(0, 2 * math.pi)
    elif target_mode == 'fix':
        angle = start_target
    elif target_mode == 'sequence':   
        angle=math.radians(sequence_target);  

    new_target_x = WIDTH // 2 + TARGET_RADIUS * math.sin(angle)
    new_target_y = HEIGHT // 2 + TARGET_RADIUS * -math.cos(angle)  # zero-angle at the top
    return [new_target_x, new_target_y]


# Function to check if the current target is reached
def check_target_reached():
    if new_target:
        distance = math.hypot(circle_pos[0] - new_target[0], circle_pos[1] - new_target[1])
        return distance <= CIRCLE_SIZE // 2
    return False


# Function to check if player is at starting position and generate new target
def at_start_position_and_generate_target(mouse_pos):
    distance = math.hypot(mouse_pos[0] - START_POSITION[0], mouse_pos[1] - START_POSITION[1])
    if distance <= CIRCLE_SIZE:
        return True
    return False


# Main game loop
running = True
show_end_position = False
while running:
    screen.fill(BLACK)

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:  # Press 'esc' to close the experiment
                running = False
            elif event.key == pygame.K_4:  # Press '4' to test perturbation_mode
                perturbation_mode = True
            elif event.key == pygame.K_5:  # Press '5' to end perturbation_mode
                perturbation_mode = False
            # elif event.key == pygame.K_h:  # Press 'h' to toggle mouse info display
            #     show_mouse_info = not show_mouse_info

    # Design experiment
    if exp_setup == 'baseline':    
        from utils import Interval_baseline_1, Interval_baseline_2 
        perturbation_type = 'sudden'
        if attempts >= ATTEMPTS_LIMIT['baseline']:
            running = False
        elif attempts == 1:
            perturbation_mode = False
        elif attempts == Interval_baseline_1[0]:  
            perturbation_mode = True
        elif attempts == Interval_baseline_1[1]: 
            perturbation_mode = False
        elif attempts == Interval_baseline_2[0]:  
            perturbation_mode = True
        elif attempts == Interval_baseline_2[1]:  
            perturbation_mode = False
    elif exp_setup == 'generalization':
        #TASK 1: DESIGN YOUR OWN EXPERIMENT (HW2_A OR HW2_B)        
        # Design experiment A
        if attempts == 1:
            perturbation_mode = False
            sequence_target = 45 # choose new target locations
        ...

    elif exp_setup == 'interference':   # EDIT
        # Design experiment B 
        from utils import Interval_inference_1, Interval_inference_2, Interval_inference_3
        perturbation_type = 'sudden'

        if attempts >= ATTEMPTS_LIMIT['interference']:
            running = False
        elif attempts == 1:
            perturbation_mode = False
        elif attempts == Interval_inference_1[0]:  # own perturbation
            perturbation_mode = True
        elif attempts == Interval_inference_1[1]: 
            perturbation_mode = False
        elif attempts == Interval_inference_2[0]:  # interference perturbation 
            perturbation_mode = True
            if time_id == '2':
                start_target = math.radians(TARGET_ANGLE_2_INTERFERENCE)   # perturbed target angle
            perturbation_angle = - math.radians(PERTURBATION_ANGLE)   # opposite direction
        elif attempts == Interval_inference_2[1]:  
            perturbation_mode = False
            if time_id == '2':
                start_target = math.radians(TARGET_ANGLE) # unperturbed target angle
        elif attempts == Interval_inference_3[0]:  # 3rd own perturbation
            perturbation_mode = True
            perturbation_angle = math.radians(PERTURBATION_ANGLE)   # back to original direction
        elif attempts == Interval_inference_3[1]:  
            perturbation_mode = False
        

    # Hide the mouse cursor
    pygame.mouse.set_visible(False)
    # Get mouse position
    mouse_pos = pygame.mouse.get_pos()

    # Calculate distance from START_POSITION to mouse_pos
    deltax = mouse_pos[0] - START_POSITION[0]
    deltay = mouse_pos[1] - START_POSITION[1]
    distance = math.hypot(deltax, deltay)
    mouse_angle = math.atan2(deltay, deltax)

    # TASK1: CALCULATE perturbed_mouse_pos
    # PRESS 'h' in game for a hint
    if perturbation_mode:
        if perturbation_type == 'sudden':
            # sudden clockwise perturbation of perturbation_angle
            perturbed_mouse_angle = perturbation_angle
        
        elif perturbation_type == 'gradual':
            # gradual counterclockwise perturbation of perturbation_angle in 10 steps, with perturbation_angle/10, each step lasts 3 attempts
            perturbed_mouse_angle = max(-(gradual_attempts // 3 + 1) * (perturbation_angle / 10), -perturbation_angle)

        perturbed_mouse_pos = [
            START_POSITION[0] + distance * math.cos(mouse_angle + perturbed_mouse_angle),
            START_POSITION[1] + distance * math.sin(mouse_angle + perturbed_mouse_angle)
        ]
        circle_pos = perturbed_mouse_pos
    else:
        circle_pos = pygame.mouse.get_pos()

    # Check if target is hit or missed
    # hit if circle touches target's center
    from utils import normalize_angle
    if check_target_reached():
        score += 1
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a hit
        if move_faster:
            error_angle = float("NaN")
        else:
            # Change to calculation of error_angle
            assert new_target is not None, "new_target must not be None when calculating error_angle on hit."
            cursor_angle = math.degrees(
                math.atan2((circle_pos[1] - START_POSITION[1]), circle_pos[0] - START_POSITION[0]))
            target_angle = math.degrees(
                math.atan2(new_target[1] - START_POSITION[1], (new_target[0] - START_POSITION[0])))
            error_angle = normalize_angle(cursor_angle - target_angle)
            # error_angle = 0.0
        error_angles.append(error_angle)

        new_target = None  # Set target to None to indicate hit
        start_time = 0  # Reset start_time after hitting the target
        if perturbation_type == 'gradual' and perturbation_mode:
            gradual_attempts += 1

    # miss if player leaves the target_radius + 1% tolerance
    elif new_target and math.hypot(circle_pos[0] - START_POSITION[0],
                                   circle_pos[1] - START_POSITION[1]) > TARGET_RADIUS * 1.01:
        attempts += 1

        # CALCULATE AND SAVE ERRORS between target and circle end position for a miss
        if move_faster:
            error_angle = float("NaN")
        else:
            cursor_angle = math.degrees(
                math.atan2((circle_pos[1] - START_POSITION[1]), circle_pos[0] - START_POSITION[0]))
            target_angle = math.degrees(
                math.atan2(new_target[1] - START_POSITION[1], (new_target[0] - START_POSITION[0])))
            error_angle = normalize_angle(cursor_angle - target_angle)
            failed_attempt_list.append(attempts)   # for failed attempts
            target_angles.append(target_angle)
            circle_angles.append(cursor_angle)
        error_angles.append(error_angle)

        new_target = None  # Set target to None to indicate miss
        start_time = 0  # Reset start_time after missing the target

        if perturbation_type == 'gradual' and perturbation_mode:
            gradual_attempts += 1

    # Check if player moved to the center and generate new target
    if not new_target and at_start_position_and_generate_target(mouse_pos):
        new_target = generate_target_position()
        move_faster = False
        start_time = pygame.time.get_ticks()  # Start the timer for the attempt
        perturbation_rand=random.uniform(-math.pi/4, +math.pi/4) # generate new random perturbation for type 'random'

    # Check if time limit for the attempt is reached
    current_time = pygame.time.get_ticks()
    if start_time != 0 and (current_time - start_time) > TIME_LIMIT:
        move_faster = True
        start_time = 0  # Reset start_time

    # Show 'MOVE FASTER!'
    if move_faster:
        font = pygame.font.Font(None, 36)
        text = font.render('MOVE FASTER!', True, RED)
        text_rect = text.get_rect(center=START_POSITION)
        screen.blit(text, text_rect)

    # Generate playing field
    # Draw current target
    if new_target:
        pygame.draw.circle(screen, BLUE, new_target, TARGET_SIZE // 2)

    # Draw circle cursor
    if mask_mode:
        if distance < MASK_RADIUS:
            pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)
    else:
        pygame.draw.circle(screen, WHITE, circle_pos, CIRCLE_SIZE // 2)

    # Draw start position
    pygame.draw.circle(screen, WHITE, START_POSITION, 5)

    # Show score
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    # Show attempts
    font = pygame.font.Font(None, 36)
    score_text = font.render(f"Attempts: {attempts}", True, WHITE)
    screen.blit(score_text, (10, 30))


    if test_mode:
        # Show score
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (10, 10))

        # Show Mouse_angle
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, WHITE)
        screen.blit(score_text, (10, 70))

        # Show pert_angle
        if perturbation_mode:
            font = pygame.font.Font(None, 36)
            score_text = font.render(f"Pert_Ang: {np.rint(np.degrees(perturbed_mouse_angle-mouse_angle))}", True, WHITE)
            screen.blit(score_text, (10, 100))
            if perturbation_type == 'gradual':
                font = pygame.font.Font(None, 36)
                score_text = font.render(f"Grad_step: {gradual_step}", True, WHITE)
                screen.blit(score_text, (10, 130))

    # if show_mouse_info:
    #     mouse_info_text = font.render(f"Mouse: x={mouse_pos[0]}, y={mouse_pos[1]}", True, WHITE)
    #     delta_info_text = font.render(f"Delta: Δx={deltax}, Δy={deltay}", True, WHITE)
    #     mouse_angle_text = font.render(f"Mouse_Ang: {np.rint(np.degrees(mouse_angle))}", True, WHITE)
    #     screen.blit(mouse_info_text, (10, 60))
    #     screen.blit(delta_info_text, (10, 90))
    #     screen.blit(mouse_angle_text, (10, 120))

    # Update display
    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()


## TASK 1, SAVE IMPORTANT VARIABLES IN .CSV 

# Save important variables in CSV file if test_mode==False:

    
    # TASK 2 GENERATE A BETTER PLOT
    # Load data from CSV file

    # Extract data for plotting

from utils import save_list_to_csv

# save important variables in CSV file, if test_mode==False

if test_mode==False:
    if exp_setup == 'interference':
        # For interference experiments, include TARGET_ANGLE_2_INTERFERENCE in filename
        save_list_to_csv(circle_angles, failed_attempt_list, header = ['attempt', 'circle_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_circle_angles_TargetAngle_{TARGET_ANGLE}_InterfAngle_{TARGET_ANGLE_2_INTERFERENCE}.csv',
                        subject_id=subject_id)
        save_list_to_csv(target_angles, failed_attempt_list, header = ['attempt', 'target_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_target_angles_TargetAngle_{TARGET_ANGLE}_InterfAngle_{TARGET_ANGLE_2_INTERFERENCE}.csv',
                        subject_id=subject_id)
        save_list_to_csv(error_angles, header = ['attempt', 'error_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_error_angles_TargetAngle_{TARGET_ANGLE}_InterfAngle_{TARGET_ANGLE_2_INTERFERENCE}.csv',
                        subject_id=subject_id)
    else:
        # For baseline experiments, use original naming
        save_list_to_csv(circle_angles, failed_attempt_list, header = ['attempt', 'circle_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_circle_angles_TargetAngle_{TARGET_ANGLE}.csv',
                        subject_id=subject_id)
        save_list_to_csv(target_angles, failed_attempt_list, header = ['attempt', 'target_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_target_angles_TargetAngle_{TARGET_ANGLE}.csv',
                        subject_id=subject_id)
        save_list_to_csv(error_angles, header = ['attempt', 'error_angle_degrees'],
                        file_name=f'{exp_setup}_{time_id}_error_angles_TargetAngle_{TARGET_ANGLE}.csv',
                        subject_id=subject_id)
sys.exit()
