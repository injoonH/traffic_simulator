# 2020.09.01
# 에이전트 학습용 환경

import numpy as np
import random

ROW, COL = 2, 3
LANE = 4
LEN = 20

layer = {"loc": 0, "next": 1, "check": 2, "row": 3, "col": 4, "lane": 5, "light": 6, "goal": 7}
board = np.zeros((LEN * (ROW + 1) + ROW * LANE * 2,
                  LEN * (COL + 1) + COL * LANE * 2, 8), dtype=np.int32)
# print("shape", board.shape)
for i in range(ROW):
    r = LEN * (i + 1) + LANE * 2 * i
    board[r: r + LANE * 2, :, layer["row"]] = i
    board[r: r + LANE, 0: LEN, layer["col"]] = 0
    board[r + LANE: r + LANE * 2, board.shape[1] - LEN: board.shape[1], layer["col"]] = COL - 1
    board[r: r + LANE, :, layer["light"]] = 1
    board[r + LANE: r + LANE * 2, :, layer["light"]] = 3
for i in range(COL):
    c = LEN * (i + 1) + LANE * 2 * i
    board[:, c: c + LANE * 2, layer["col"]] = i
    board[0: LEN, c + LANE: c + LANE * 2, layer["row"]] = 0
    board[board.shape[0] - LEN: board.shape[0], c: c + LANE, layer["row"]] = ROW - 1
    board[:, c: c + LANE, layer["light"]] = 0
    board[:, c + LANE: c + LANE * 2, layer["light"]] = 2
for i in range(LANE):
    for j in range(ROW):
        board[LEN * (j + 1) + LANE * 2 * j + i, :, layer["lane"]] = LANE - i
        board[(LEN + LANE * 2) * (j + 1) - 1 - i, :, layer["lane"]] = LANE - i
    for j in range(COL):
        board[:, LEN * (j + 1) + LANE * 2 * j + i, layer["lane"]] = LANE - i
        board[:, (LEN + LANE * 2) * (j + 1) - 1 - i, layer["lane"]] = LANE - i
for i in range(ROW):
    for j in range(COL):
        r = LEN * (i + 1) + LANE * 2 * i
        c = LEN * (j + 1) + LANE * 2 * j

        # 신호 확인 좌표 (layer 2)
        board[r - 1, c: c + LANE, layer["check"]] = 1
        board[r: r + LANE, c + LANE * 2, layer["check"]] = 1
        board[r + LANE * 2, c + LANE: c + LANE * 2, layer["check"]] = 1
        board[r + LANE: r + LANE * 2, c - 1, layer["check"]] = 1

        # 행 (layer 3)
        board[r - LEN: r, c: c + LANE, layer["row"]] = i
        board[r + LANE * 2: r + LANE * 2 + LEN, c + LANE: c + LANE * 2, layer["row"]] = i

        # 열 (layer 4)
        board[r: r + LANE, c + LANE * 2: c + LANE * 2 + LEN, layer["col"]] = j
        board[r + LANE: r + LANE * 2, c - LEN: c, layer["col"]] = j

        # 신호등 (layer 6)
        board[r: r + LANE * 2, c: c + LANE * 2, layer["light"]] = -1

# start, goal list 생성
start = []
goal = []
for i in range(ROW):
    r = LEN * (i + 1) + LANE * 2 * i
    for j in range(LANE):
        start.append([r + j, board.shape[1] - 1])
        goal.append([r + j, 0])
        board[r + j, 0, layer["goal"]] = 1
    for j in range(LANE, LANE * 2):
        start.append([r + j, 0])
        goal.append([r + j, board.shape[1] - 1])
        board[r + j, board.shape[1] - 1, layer["goal"]] = 1
for i in range(COL):
    c = LEN * (i + 1) + LANE * 2 * i
    for j in range(LANE):
        start.append([0, c + j])
        goal.append([board.shape[0] - 1, c + j])
        board[board.shape[0] - 1, c + j, layer["goal"]] = 1
    for j in range(LANE, LANE * 2):
        start.append([board.shape[0] - 1, c + j])
        goal.append([0, c + j])
        board[0, c + j, layer["goal"]] = 1

direct = {0: (-1, 0),   # 북
          1: (-1, 1),   # 북동
          2: (0, 1),    # 동
          3: (1, 1),    # 남동
          4: (1, 0),    # 남
          5: (1, -1),   # 남서
          6: (0, -1),   # 서
          7: (-1, -1)}  # 북서
way = {"str": 0, "lft": 1, "rgt": 2, "U": 3}
light = {"lr": 0, "ud": 1, "tlr": 2, "tud": 3, "stp": 4}
intersection = np.zeros((ROW, COL))


class Car:
    def __init__(self, lc, gl):
        self.loc = lc       # 현 좌표
        self.goal = gl      # 목적지 좌표
        self.lane = None    # 목표 차선
        self.direct = None  # direct{} key 저장
        self.nav = []       # 이상적 way{} value list
        self.time = 0       # 정차 시간


def get_way(vector1, vector2):
    dtm = vector1[0] * vector2[1] - vector1[1] * vector2[0]
    if dtm == 1:
        return way["lft"]
    if dtm == -1:
        return way["rgt"]
    if vector1[0] == vector2[0] and vector1[1] == vector2[1]:
        return way["str"]
    return way["U"]


def set_nav(car):
    dr = board[car.goal[0], car.goal[1], layer["row"]] - board[car.loc[0], car.loc[1], layer["row"]]
    dc = board[car.goal[0], car.goal[1], layer["col"]] - board[car.loc[0], car.loc[1], layer["col"]]
    gl = board[car.goal[0], car.goal[1], layer["light"]]

    arr = []

    for i in range(int(abs(dr))):
        arr.append((dr / abs(dr), 0))
    for i in range(int(abs(dc))):
        arr.append((0, dc / abs(dc)))

    random.shuffle(arr)

    if len(arr) > 0:
        car.nav = []
        car.nav.append(get_way(direct[car.direct], arr[0]))
        for i in range(len(arr) - 1):
            car.nav.append(get_way(arr[i], arr[i + 1]))
        car.nav.append(get_way(arr[-1], direct[(gl * 2 + 4) % 8]))
    else:
        car.nav = [get_way(direct[car.direct], direct[(gl * 2 + 4) % 8])]


def refresh_nav(car, wy):
    car.nav = [wy]

    if wy == way["str"]:
        d = car.direct
    elif wy == way["lft"]:
        d = (car.direct + 6) % 8
    elif wy == way["rgt"]:
        d = (car.direct + 2) % 8
    else:
        d = (car.direct + 4) % 8

    r = board[car.loc[0], car.loc[1], layer["row"]] + direct[d][0]
    c = board[car.loc[0], car.loc[1], layer["col"]] + direct[d][1]

    if 0 <= r < ROW and 0 <= c < COL:
        dr = board[car.goal[0], car.goal[1], layer["row"]] - r
        dc = board[car.goal[0], car.goal[1], layer["col"]] - c
        gl = board[car.goal[0], car.goal[1], layer["light"]]

        arr = []

        for i in range(int(abs(dr))):
            arr.append((dr / abs(dr), 0))
        for i in range(int(abs(dc))):
            arr.append((0, dc / abs(dc)))

        random.shuffle(arr)

        if len(arr) > 0:
            car.nav.append(get_way(direct[d], arr[0]))
            for i in range(len(arr) - 1):
                car.nav.append(get_way(arr[i], arr[i + 1]))
            car.nav.append(get_way(arr[-1], direct[(gl * 2 + 4) % 8]))
        else:
            car.nav.append(get_way(direct[d], direct[(gl * 2 + 4) % 8]))


def refresh_lane(car):
    if len(car.nav) == 0:
        car.lane = board[car.goal[0], car.goal[1], layer["lane"]]
    else:
        w = car.nav[0]
        if w == way["str"]:
            ln = board[car.loc[0], car.loc[1], layer["lane"]]
            if ln == 1 and LANE > 1:
                car.lane = 2
            else:
                car.lane = ln
        elif w == way["rgt"]:
            car.lane = LANE
        else:
            car.lane = 1


def refresh_direct(car):
    lgt = board[car.loc[0], car.loc[1], layer["light"]]
    if lgt == 0:
        car.direct = 4  # 남
    elif lgt == 1:
        car.direct = 6  # 서
    elif lgt == 2:
        car.direct = 0  # 북
    else:
        car.direct = 2  # 동


def get_reward(cars):
    # {0: 정차 차량, 1: 대기 시간}
    rwd = np.zeros((ROW, COL, 4, 2))
    for car in cars:
        if board[car.loc[0], car.loc[1], layer["light"]] == -1:
            continue
        if car.time > 0:
            r = board[car.loc[0], car.loc[1], layer["row"]]
            c = board[car.loc[0], car.loc[1], layer["col"]]
            lgt = board[car.loc[0], car.loc[1], layer["light"]]
            rwd[r, c, lgt, 0] += 1
            if board[car.loc[0], car.loc[1], layer["check"]] == 1:
                rwd[r, c, lgt, 1] += car.time
    return rwd


# 새로운 환경으로 재시작
# 매개변수로 실행할 프레임 수 넣으면 됨
def restart_everything(frame):
    board[:, :, layer["loc"]] = 0
    board[:, :, layer["next"]] = 0

    enter_prob = np.random.rand(2 * (ROW + COL)) / 6
    to_where_prob = np.random.rand(2 * (ROW + COL), 2 * (ROW + COL))
    to_where_prob = to_where_prob / to_where_prob.sum(axis=1)[:, None]

    intersection[:, :] = 0

    cars = []

    for frame_num in range(frame):
        # ==========
        # 자동차 유입
        # ==========
        rand_prob = np.random.rand(2 * (ROW + COL))
        arr, = np.where(rand_prob < enter_prob)

        for s in arr:
            g = np.random.choice(list(range(2 * (ROW + COL))), p=to_where_prob[s, :])
            i = s * LANE + random.randint(0, LANE - 1)
            j = g * LANE + random.randint(0, LANE - 1)
            if board[start[i][0], start[i][1], layer["loc"]] != 1:
                c = Car(list(start[i]), list(goal[j]))
                refresh_direct(c)
                set_nav(c)
                refresh_lane(c)
                board[start[i][0], start[i][1], layer["loc"]] = 1
                cars.append(c)

        # ========
        # step 1
        # ========
        board[:, :, layer["next"]] = 0
        out_indices = []
        indices = list(range(len(cars)))
        for idx in range(len(cars) - 1, -1, -1):
            c = cars[idx]
            if board[c.loc[0], c.loc[1], layer["check"]] == 1:
                if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                    if c.nav[0] == way["rgt"]:
                        if board[c.loc[0] + direct[(c.direct + 1) % 8][0],
                                 c.loc[1] + direct[(c.direct + 1) % 8][1],
                                 layer["loc"]] == 1:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                    elif c.nav[0] == way["str"]:
                        row = board[c.loc[0], c.loc[1], layer["row"]]
                        col = board[c.loc[0], c.loc[1], layer["col"]]
                        b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light[
                            "ud"]
                        b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light[
                            "lr"]
                        if b1 or b2:
                            if board[c.loc[0] + direct[c.direct][0],
                                     c.loc[1] + direct[c.direct][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                        else:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                    else:
                        row = board[c.loc[0], c.loc[1], layer["row"]]
                        col = board[c.loc[0], c.loc[1], layer["col"]]
                        b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light[
                            "tud"]
                        b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light[
                            "tlr"]
                        if b1 or b2:
                            if c.nav[0] == way["lft"]:
                                if board[c.loc[0] + direct[(c.direct + 7) % 8][0],
                                         c.loc[1] + direct[(c.direct + 7) % 8][1],
                                         layer["loc"]] == 1:
                                    c.time += 1
                                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                                    del indices[idx]
                            else:
                                if board[c.loc[0] + direct[(c.direct + 6) % 8][0],
                                         c.loc[1] + direct[(c.direct + 6) % 8][1],
                                         layer["loc"]] == 1:
                                    c.time += 1
                                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                                    del indices[idx]
                        else:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                else:
                    if board[c.loc[0], c.loc[1], layer["lane"]] == 1:
                        row = board[c.loc[0], c.loc[1], layer["row"]]
                        col = board[c.loc[0], c.loc[1], layer["col"]]
                        b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light[
                            "tud"]
                        b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light[
                            "tlr"]
                        if b1 or b2:
                            if random.random() < 0.5:
                                refresh_nav(c, way["lft"])
                                refresh_lane(c)
                                if board[c.loc[0] + direct[(c.direct + 7) % 8][0],
                                         c.loc[1] + direct[(c.direct + 7) % 8][1],
                                         layer["loc"]] == 1:
                                    c.time += 1
                                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                                    del indices[idx]
                            else:
                                refresh_nav(c, way["U"])
                                refresh_lane(c)
                                if board[c.loc[0] + direct[(c.direct + 6) % 8][0],
                                         c.loc[1] + direct[(c.direct + 6) % 8][1],
                                         layer["loc"]] == 1:
                                    c.time += 1
                                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                                    del indices[idx]
                        else:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]
                    else:
                        if random.random() < 0.5:
                            refresh_nav(c, way["str"])
                            refresh_lane(c)
                            row = board[c.loc[0], c.loc[1], layer["row"]]
                            col = board[c.loc[0], c.loc[1], layer["col"]]
                            b1 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 0 and intersection[row, col] == light[
                                "ud"]
                            b2 = board[c.loc[0], c.loc[1], layer["light"]] % 2 == 1 and intersection[row, col] == light[
                                "lr"]
                            if b1 or b2:
                                if board[c.loc[0] + direct[c.direct][0],
                                         c.loc[1] + direct[c.direct][1],
                                         layer["loc"]] == 1:
                                    c.time += 1
                                    board[c.loc[0], c.loc[1], layer["next"]] = 1
                                    del indices[idx]
                            else:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
                        else:
                            refresh_nav(c, way["rgt"])
                            refresh_lane(c)
                            if board[c.loc[0] + direct[(c.direct + 1) % 8][0],
                                     c.loc[1] + direct[(c.direct + 1) % 8][1],
                                     layer["loc"]] == 1:
                                c.time += 1
                                board[c.loc[0], c.loc[1], layer["next"]] = 1
                                del indices[idx]
            else:
                if board[c.loc[0], c.loc[1], layer["light"]] == -1:
                    if board[c.loc[0] + direct[c.direct][0],
                             c.loc[1] + direct[c.direct][1],
                             layer["loc"]] == 1:
                        c.time += 1
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]
                else:
                    if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                        if board[c.loc[0] + direct[c.direct][0],
                                 c.loc[1] + direct[c.direct][1],
                                 layer["loc"]] == 1:
                            c.time += 1
                            board[c.loc[0], c.loc[1], layer["next"]] = 1
                            del indices[idx]

        # ========
        # step 2
        # ========
        random.shuffle(indices)
        for idx in range(len(indices) - 1, -1, -1):
            c = cars[indices[idx]]
            if board[c.loc[0], c.loc[1], layer["light"]] != -1 and board[c.loc[0], c.loc[1], layer["check"]] != 1:
                d = c.lane - board[c.loc[0], c.loc[1], layer["lane"]]
                if d != 0:
                    d = int(abs(d) / d)
                    row = c.loc[0] + direct[(c.direct + 8 + d) % 8][0]
                    col = c.loc[1] + direct[(c.direct + 8 + d) % 8][1]
                    if board[row, col, layer["next"]] != 1:
                        if board[row, col, layer["loc"]] == 1:
                            if random.random() < 0.5:
                                c.time += 1
                            else:
                                sr = c.loc[0] + direct[c.direct][0]
                                sc = c.loc[1] + direct[c.direct][1]
                                if board[sr, sc, layer["next"]] != 1 and board[sr, sc, layer["loc"]] != 1:
                                    c.loc[0] = sr
                                    c.loc[1] = sc
                                    if board[sr, sc, layer["goal"]] == 1:
                                        out_indices.append(indices[idx])
                                    else:
                                        c.time = 0
                                else:
                                    c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                        board[c.loc[0], c.loc[1], layer["next"]] = 1
                        del indices[idx]

        # ========
        # step 3
        # ========
        for idx in range(len(indices) - 1, -1, -1):
            c = cars[indices[idx]]
            if board[c.loc[0], c.loc[1], layer["check"]] == 1:
                if board[c.loc[0], c.loc[1], layer["lane"]] == c.lane:
                    if c.nav[0] == way["rgt"]:
                        row = c.loc[0] + direct[(c.direct + 1) % 8][0]
                        col = c.loc[1] + direct[(c.direct + 1) % 8][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 2) % 8
                            del c.nav[0]
                            refresh_lane(c)
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                    elif c.nav[0] == way["str"]:
                        row = c.loc[0] + direct[c.direct][0]
                        col = c.loc[1] + direct[c.direct][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                    elif c.nav[0] == way["lft"]:
                        row = c.loc[0] + direct[(c.direct + 7) % 8][0]
                        col = c.loc[1] + direct[(c.direct + 7) % 8][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 7) % 8
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                    else:
                        row = c.loc[0] + direct[(c.direct + 6) % 8][0]
                        col = c.loc[1] + direct[(c.direct + 6) % 8][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            c.direct = (c.direct + 4) % 8
                            del c.nav[0]
                            refresh_lane(c)
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
            else:
                if board[c.loc[0], c.loc[1], layer["light"]] == -1:
                    if c.direct % 2 == 1:
                        row = c.loc[0] + direct[c.direct][0]
                        col = c.loc[1] + direct[c.direct][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if board[row, col, layer["light"]] != -1:
                                c.direct = (c.direct + 7) % 8
                                del c.nav[0]
                                refresh_lane(c)
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                    else:
                        row = c.loc[0] + direct[c.direct][0]
                        col = c.loc[1] + direct[c.direct][1]
                        if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                            c.time += 1
                        else:
                            c.loc[0] = row
                            c.loc[1] = col
                            if board[row, col, layer["light"]] != -1:
                                del c.nav[0]
                                refresh_lane(c)
                            if board[row, col, layer["goal"]] == 1:
                                out_indices.append(indices[idx])
                            else:
                                c.time = 0
                else:
                    row = c.loc[0] + direct[c.direct][0]
                    col = c.loc[1] + direct[c.direct][1]
                    if board[row, col, layer["next"]] == 1 or board[row, col, layer["loc"]] == 1:
                        c.time += 1
                    else:
                        c.loc[0] = row
                        c.loc[1] = col
                        if board[row, col, layer["goal"]] == 1:
                            out_indices.append(indices[idx])
                        else:
                            c.time = 0
            board[c.loc[0], c.loc[1], layer["next"]] = 1
        board[:, :, layer["loc"]] = board[:, :, layer["next"]]

        # ========
        # step 4
        # ========
        out_indices.sort()
        for idx in range(len(out_indices) - 1, -1, -1):
            del cars[out_indices[idx]]

        # =====================
        # 교차로 신호 바꾸는 바분
        # =====================
        mod = LANE * 4
        if frame_num % mod == 0:
            intersection[:, :] = light["stp"]
        elif frame_num % mod == LANE * 2:
            rwd = get_reward(cars)  # 보상
            cur = np.copy(intersection)  # 현 교차로 신호 상황

            #############################################################
            # 에이전트 동작 부분
            # rwd(보상) 이용해서 다음 교차로 신호 new_signal
            # 우변에 있는 거 지우고 에이전트에서 계산한 값 넣으면 됨
            new_signal = np.zeros((ROW, COL))
            #############################################################

            intersection[:, :] = new_signal


restart_everything(1000)  # 1000 프레임
restart_everything(200)  # 200 프레임
