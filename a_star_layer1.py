#!/usr/bin/env python

# reference:
# https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/AStar/a_star.py
# Hongsheng Chang
# changh@kth.se

import numpy as np
import matplotlib.pyplot as plt
import math
'''
update: fix the bug of reverse in final route (rx, ry)
        add yaw calculation
'''


class Node:
    def __init__(self, x, y, cost, pind):
        self.x = x
        self.y = y
        self.cost = cost
        self.pind = pind


def calObsWall(sx, sy, ex, ey, obsMap, reso, airSpace):
    '''
    sx, sy start point of the wall
    ex, ey end point of the wall
    '''
    xMax = int(math.ceil(max(sx, ex) / reso))  # Ex. 3.2-> 4
    yMax = int(math.ceil(max(sy, ey) / reso))
    if min(sx, ex) < 0:
        xMin = int(min(sx, ex) / reso)-1  # Ex.-3.2-> -4
    else:
        xMin = int(min(sx, ex) / reso)  # Ex. 3.2-> 3
    if min(sy, ey) < 0:
        yMin = int(min(sy, ey) / reso)-1
    else:
        yMin = int(min(sy, ey) / reso)

    xWidth = xMax - xMin  # number of grid
    yWidth = yMax - yMin
    if sx == ex:
        k = 1000000  # give a Slope close to inf
    else:
        k = (sy - ey) / (sx - ex)
    if xWidth == 0:
        xWidth = 1
    if yWidth == 0:
        yWidth = 1

    for ix in range(xWidth+1):
        x = (ix + xMin) * reso
        for iy in range(yWidth+1):
            y = (iy + yMin) * reso  # (x,y): position of each node
            d = abs(k * (x - sx) - (y - sy)) / math.sqrt(1 + k * k)
            if d < 1.05 * reso:
                # if distance from node to obstacle less than 1.5, regard as occupied

                obsMap[int(ix + xMin - airSpace[0][0] / reso)][int(iy + yMin - airSpace[1][0] / reso)] = True  # move the zero points(need to change if the airspace change)
    return obsMap


def calObsGate(posx, posy, angle, obsMap, reso, airSpace):
    '''
    posx posy postion of gate
    angle: angle of gate
    '''
    angle = angle - 90  # orientation of the wall made by gate
    gateSize = 0.6  # 60cm
    if angle < 0:
        angle = angle + 180
    if angle >= 180:
        angle = angle - 180  # get angle in (0, 180)
    sx = posx - math.cos(math.radians(angle)) * gateSize * 0.5  # start point and end point
    sy = posy - math.sin(math.radians(angle)) * gateSize * 0.5  # treat it as wall
    ex = posx + math.cos(math.radians(angle)) * gateSize * 0.5  # temperately(!!! the center of gates can be free)
    ey = posy + math.sin(math.radians(angle)) * gateSize * 0.5
    obsMap = calObsWall(sx, sy, ex, ey, obsMap, reso, airSpace)

    return obsMap


def calMap(reso, airSpace, z):
    xWidth = int((airSpace[0][1] - airSpace[0][0]) / reso)
    yWidth = int((airSpace[1][1] - airSpace[1][0]) / reso)
    obsMap = np.array([[False for i in range(yWidth+1)] for i in range(xWidth+1)])  # obsMap[x][y]

    # data of wall and gate
    # wall [sx, sy, ex, ey]
    wall = np.array([[-2.0, 2.0, -2.0, 0.25]])  # [-2.0, 0.25, -1.0, 0.25] ignore the low wall
    # gate [x, y, angle]
    gate = [[ 1.25, -0.50, 135.0],
            [ 0.25,  0.50, 135.0],
            [-1.50,  1.00, 180.0],
            [-3.00,  0.50, 180.0],
            [-2.50, -0.75, -90.0],
            [-1.50, -0.75,   0.0],
            [ 0.25, -0.50,  45.0],
            [ 1.25,  0.50,  45.0]]

    if z <= 0.5:  # only care about wall if height lower than 0.5
        for i in wall:
            obsMap = calObsWall(i[0], i[1], i[2], i[3], obsMap, reso, airSpace)

    for i in gate:
        obsMap = calObsGate(i[0], i[1], i[2], obsMap, reso, airSpace)

    return obsMap


def calHeuristic(n1, n2):
    w = 1.0  # weight of heuristic
    d = w * math.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)
    return d


def calIndex(node, reso, airSpace):
    yWidth = int((airSpace[1][1] - airSpace[1][0]) / reso)
    return (node.x - airSpace[0][0]) * yWidth + (node.y - airSpace[1][0])


def verifyNode(node, obsmap, reso, airSpace):
    '''
    check outlier and obsticle
    '''
    if node.x < (airSpace[0][0] / reso):
        return False
    elif node.y < (airSpace[1][0] / reso):
        return False
    elif node.x >= (airSpace[0][1] / reso):
        return False
    elif node.y >= (airSpace[1][1] / reso):
        return False

    if obsmap[int(node.x - airSpace[0][0] / reso)][int(node.y - airSpace[1][0] / reso)]:
        return False

    return True


def calc_final_path(ngoal, closedset, reso):
    # generate final course
    rx, ry = [ngoal.x * reso], [ngoal.y * reso]
    pind = ngoal.pind
    while pind != -1:
        n = closedset[pind]
        rx.append(n.x * reso)
        ry.append(n.y * reso)
        pind = n.pind
    rx.reverse()
    ry.reverse()
    return rx, ry


def aStarPlanning(sx, sy, gx, gy, z):
    '''
    (sx,sy) (gx,gy) start point and goal point
    '''
    airSpace = [[-4, 2], [-2, 2]]
    reso = 0.05  # Resolution = 10cm
    len_x = int((airSpace[0][1] - airSpace[0][0]) / reso)
    len_y = int((airSpace[1][1] - airSpace[1][0]) / reso)
    nstart = Node(round(sx / reso), round(sy / reso), 0.0, -1)
    ngoal = Node(round(gx / reso), round(gy / reso), 0.0, -1)

    obsMap = calMap(reso, airSpace, z)
    # show map
    for x in range(len_x):
        for y in range(len_y):
            if obsMap[x][y]:
                plt.plot(x * reso + airSpace[0][0], y * reso + airSpace[1][0], ".k")
    plt.plot(sx, sy, "xr")
    plt.plot(gx, gy, "xb")
    plt.grid(True)
    plt.axis("equal")

    # dx, dy, cost
    motion = [[ 1,  0, 1],
              [ 0,  1, 1],
              [-1,  0, 1],
              [ 0, -1, 1],
              [-1, -1, math.sqrt(2)],
              [-1,  1, math.sqrt(2)],
              [ 1, -1, math.sqrt(2)],
              [ 1,  1, math.sqrt(2)]]

    openSet, closeSet = dict(), dict()
    openSet[calIndex(nstart, reso, airSpace)] = nstart

    while 1:
        cId = min(openSet, key=lambda o: openSet[o].cost + calHeuristic(ngoal, openSet[o]))
        current = openSet[cId]
        # show graph
        plt.plot(current.x * reso, current.y * reso, "xc")
        if len(closeSet.keys()) % 10 == 0:
            plt.pause(0.001)

        if current.x == ngoal.x and current.y == ngoal.y:
            print("Find goal")
            ngoal.pind = current.pind
            ngoal.cost = current.cost
            break

        # Remove the item from the open set
        del openSet[cId]
        # Add it to the closed set
        closeSet[cId] = current

        # expand search grid based on motion model
        for i, _ in enumerate(motion):
            node = Node(current.x + motion[i][0], current.y + motion[i][1], current.cost + motion[i][2], cId)
            nId = calIndex(node, reso, airSpace)

            if nId in closeSet:
                continue

            if not verifyNode(node, obsMap, reso, airSpace):
                continue

            if nId not in openSet:
                openSet[nId] = node  # Discover a new node
            else:
                if openSet[nId].cost >= node.cost:
                    # This path is the best until now. record it!
                    openSet[nId] = node

    rx, ry = calc_final_path(ngoal, closeSet, reso)
    rx, ry, ryaw = yaw_planning(rx, ry, obsMap, reso)
    return rx, ry, ryaw


def yaw_planning(x, y, obsMap, reso):
    '''
    choose the most possible marker to see
    '''
    # pose of marker [x, y, z, roll, pitch, yaw]
    marker = [[ 1.25, -0.50, 0.10,  0.0, -90.0, -135.0],
              [ 0.25,  0.50, 0.10,  0.0, -90.0, -135.0],
              [-1.50,  1.00, 0.10,  0.0, -90.0,  -90.0],
              [-3.00,  0.50, 0.10,  0.0, -90.0,  -90.0],
              [-2.50, -0.75, 0.10,  0.0, -90.0,    0.0],
              [-1.50, -0.75, 0.10,  0.0, -90.0,   90.0],
              [ 0.25, -0.50, 0.10,  0.0, -90.0,  135.0],
              [ 1.25,  0.50, 0.10,  0.0, -90.0,  135.0],
              [-2.00,  0.00, 0.00, 90.0,   0.0,    0.0],
              [-2.50,  0.50, 0.00, 90.0,   0.0,    0.0],
              [-1.50,  0.50, 0.00, 90.0,   0.0,    0.0],
              [-0.50,  0.50, 0.00, 90.0,   0.0,    0.0],
              [-0.50,  0.00, 0.00, 90.0,   0.0,    0.0],
              [ 2.00,  0.00, 0.00, 90.0,   0.0,    0.0]]

    t = len(x)  # length of route
    yaw = [None for k in range(t)]
    for i in range(t):
        id = None
        buf = [None for k in range(15)]
        # choose nearest marker
        d_buf = 1000
        for j in range(14):
            flag = 0

            dx = marker[j][0] - x[i]
            dy = marker[j][1] - y[i]
            d = math.sqrt(dx**2 + dy**2)

            # for all markers, distance should be in proper range
            if d < 0.4 or d > 1.5:
                flag = 1
                continue

            # cal degree from drone to marker range(0-360)
            if d == 0:
                d = 0.0001
            dx /= d  # normalize
            dy /= d
            if dy >= 0:
                degree = math.degrees(math.acos(dx))
            else:
                degree = 360 - math.degrees(math.acos(dx))

            # vertical markers
            if marker[j][4] == -90:
                angle = marker[j][5] + 90  # orientation of marker(+-180)
                if angle > 180:
                    angle -= 360

                if abs(degree - 180 - angle) > 60:
                    flag = 1
                    continue
                else:
                    d -= 0.3  # give more privilage to vertical sign

            if flag == 0:
                buf[j] = degree
                if d < d_buf:
                    id = j
                    d_buf = d

        # if don't find ideal marker
        if id is None:
            print("don't find")
            dx = x[i] - x[i - 1]
            dy = y[i] - y[i - 1]
            d = math.sqrt(dx**2 + dy**2)
            dx /= d  # normalize
            dy /= d
            if dy >= 0:
                yaw[i] = math.degrees(math.acos(dx))
            else:
                yaw[i] = 360 - math.degrees(math.acos(dx))

        else:
            # save the nearest marker and yaw
            yaw[i] = buf[id]

    return x, y, yaw


def a_star_layer(sx, sy, gx, gy):  # choose layer in planning
    Lx, Ly, Lyaw = aStarPlanning(sx, sy, gx, gy, 0.4)
    plt.clf()
    Hx, Hy, Hyaw = aStarPlanning(sx, sy, gx, gy, 0.6)
    if len(Hx) < len(Lx):
        return Hx, Hy, Hyaw, 0.6
    else:
        return Lx, Ly, Lyaw, 0.4


x, y, yaw, height = a_star_layer(0.0, 0.5, -2.5, 0.9)
print(height)
plt.plot(x, y, "-r")
plt.show()
