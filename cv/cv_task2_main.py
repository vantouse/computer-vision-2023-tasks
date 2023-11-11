import cv2
import mediapipe as mp
import random
import time


class Bubble:
    def __init__(self, pos, size, hand_type, finger_type):
        self.pos = pos
        self.size = size
        self.hand_type = hand_type
        self.finger_type = finger_type

    def busted_with_finger(self, f_hand_type, f_type, f_pos):
        lower_bound_x, lower_bound_y = self.pos[0] - self.size, self.pos[1] - self.size
        upper_bound_x, upper_bound_y = self.pos[0] + self.size, self.pos[1] + self.size
        return (f_hand_type == self.hand_type and f_type == self.finger_type and
                lower_bound_x < f_pos.x * frame_w < upper_bound_x and
                lower_bound_y < f_pos.y * frame_h < upper_bound_y)


# capture camera input
capture_cam = cv2.VideoCapture(0)
frame_w, frame_h = 640, 480
default_res = [frame_w, frame_h]

# hand detection model preparation
mp_hands = mp.solutions.holistic
hands = mp_hands.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.6)
fingers_lr = [{4: (0, 0, 50), 8: (0, 0, 100), 12: (0, 0, 150), 16: (0, 0, 200), 20: (0, 0, 255)},
              {4: (0, 50, 0), 8: (0, 100, 0), 12: (0, 150, 0), 16: (0, 200, 0), 20: (0, 255, 0)}]
bubbles = list()

fps = 0
time_init = time.time_ns() // (10 ** 9)
time_prev = time.time_ns() // (10 ** 9)
time_interval = 2

time_game_res = time.time_ns() // (10 ** 9)
game_res = None

while capture_cam.isOpened():
    # timer and game state tracking mechanism
    time_curr = time.time_ns() // (10 ** 9)
    if not game_res and len(bubbles) == 0 and time_curr - time_init >= 2 > time_curr - time_prev:
        # winning conditions
        game_res = 'YOU WIN!'
        time_game_res = time_curr
    if not game_res and len(bubbles) >= 10:
        # losing conditions
        game_res = 'YOU LOSE!'
        time_game_res = time_curr
    if game_res and time_curr - time_game_res > 3:
        # wait 3 sec after win/loss
        break

    if time_curr - time_prev >= time_interval and not game_res:
        # add new bubbles to frame in random positions
        bubble = Bubble((random.randrange(default_res[0]), random.randrange(default_res[1])),
                        round(frame_h / 20),
                        random.choice([0, 1]),
                        random.choice(list(fingers_lr[0].keys())))
        if bubble.pos[0] >= round(default_res[0] / 2):
            bubble.hand_type = 0
        elif bubble.pos[0] < round(default_res[0] / 2):
            bubble.hand_type = 1
        bubbles.append(bubble)
        # estimate FPS
        time_prev = time_curr
        time_total = time_curr - time_init
        print(round(fps / 2))
        fps = 0

    ret, cam_frame = capture_cam.read()
    cam_frame = cv2.cvtColor(cv2.flip(cv2.resize(cam_frame, default_res), 1), cv2.COLOR_BGR2RGB)
    fps += 1

    # detect hands with mediapipe
    results = hands.process(cam_frame)
    cam_frame = cv2.cvtColor(cam_frame, cv2.COLOR_RGB2BGR)

    # create bubbles in random positions and bust them with fingertips
    if results.left_hand_landmarks:
        for landmark_num in range(len(results.left_hand_landmarks.landmark)):
            landmark = results.left_hand_landmarks.landmark[landmark_num]
            # circle fingertips with different colors
            if landmark_num in [4, 8, 12, 16, 20]:
                cv2.circle(cam_frame,
                           (round(landmark.x * frame_w),
                            round(landmark.y * frame_h)),
                           7, fingers_lr[0][landmark_num], 2)
        for fingertip in list(fingers_lr[0].keys()):
            # bust bubbles with fingertips
            for bubble in bubbles:
                if bubble.busted_with_finger(0, fingertip, results.left_hand_landmarks.landmark[fingertip]):
                    bubbles.remove(bubble)

    if results.right_hand_landmarks:
        for landmark_num in range(len(results.right_hand_landmarks.landmark)):
            landmark = results.right_hand_landmarks.landmark[landmark_num]
            if landmark_num in [4, 8, 12, 16, 20]:
                cv2.circle(cam_frame,
                           (round(landmark.x * frame_w),
                            round(landmark.y * frame_h)),
                           7, fingers_lr[1][landmark_num], 2)
        for fingertip in list(fingers_lr[1].keys()):
            for bubble in bubbles:
                if bubble.busted_with_finger(1, fingertip, results.right_hand_landmarks.landmark[fingertip]):
                    bubbles.remove(bubble)

    # draw bubbles
    for bubble in bubbles:
        cv2.circle(cam_frame, bubble.pos, bubble.size, fingers_lr[bubble.hand_type][bubble.finger_type], -1)

    # print game result
    if game_res:
        game_res_txt_color = (0, 255, 0) if game_res == 'YOU WIN!' else (0, 0, 255)
        cv2.putText(cam_frame, game_res, (round(frame_w / 2) - 182, round(frame_h / 2) + 25), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 16, cv2.LINE_AA)
        cv2.putText(cam_frame, game_res, (round(frame_w / 2) - 182, round(frame_h / 2) + 25), cv2.FONT_HERSHEY_COMPLEX, 2, game_res_txt_color, 8, cv2.LINE_AA)

    # camera preview
    cv2.imshow('cam_frame', cam_frame)
    cv2.moveWindow('cam_frame', 0, 0)

    # exit on 'Q' key pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture_cam.release()
cv2.destroyAllWindows()
