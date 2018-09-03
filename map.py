# Self Driving Car

# Importing the libraries
import numpy as np
from random import random, randint
import matplotlib.pyplot as plt
import time

# Importing the Kivy packages
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.button import Button
from kivy.graphics import Color, Ellipse, Line
from kivy.config import Config
from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.clock import Clock
import math
import thread
import skimage.measure
import datetime

# Importing the Dqn object from our AI in ai.py
from ai import Dqn

# Adding this line if we don't want the right click to put a red point
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')

# Introducing last_x and last_y, used to keep the last point in memory when we draw the sand on the map
last_x = 0
last_y = 0
n_points = 0
length = 0

# Getting our AI, which we call "brain", and that contains our neural network that represents our Q-function
action2rotation = [0,20,-20]
last_reward = 0
scores = []

# Initializing the map
first_update = True
def init():
    global sand
    global goal_x
    global goal_y
    global first_update
    global shrink_sand
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
last_distance = 0
shrink_updated = False

# Creating the car class

class Car(Widget):

    angle = NumericProperty(0)
    rotation = NumericProperty(0)
    velocity_x = NumericProperty(0)
    velocity_y = NumericProperty(0)
    velocity = ReferenceListProperty(velocity_x, velocity_y)
    sensor1_x = NumericProperty(0)
    sensor1_y = NumericProperty(0)
    sensor1 = ReferenceListProperty(sensor1_x, sensor1_y)
    sensor2_x = NumericProperty(0)
    sensor2_y = NumericProperty(0)
    sensor2 = ReferenceListProperty(sensor2_x, sensor2_y)
    sensor3_x = NumericProperty(0)
    sensor3_y = NumericProperty(0)
    sensor3 = ReferenceListProperty(sensor3_x, sensor3_y)
    signal1 = NumericProperty(0)
    signal2 = NumericProperty(0)
    signal3 = NumericProperty(0)

    def move(self, rotation):
        self.pos = Vector(*self.velocity) + self.pos
        self.rotation = rotation
        self.angle = self.angle + self.rotation
        self.sensor1 = Vector(30, 0).rotate(self.angle) + self.pos
        self.sensor2 = Vector(30, 0).rotate((self.angle+30)%360) + self.pos
        self.sensor3 = Vector(30, 0).rotate((self.angle-30)%360) + self.pos
        self.signal1 = int(np.sum(sand[int(self.sensor1_x)-10:int(self.sensor1_x)+10, int(self.sensor1_y)-10:int(self.sensor1_y)+10]))/400.
        self.signal2 = int(np.sum(sand[int(self.sensor2_x)-10:int(self.sensor2_x)+10, int(self.sensor2_y)-10:int(self.sensor2_y)+10]))/400.
        self.signal3 = int(np.sum(sand[int(self.sensor3_x)-10:int(self.sensor3_x)+10, int(self.sensor3_y)-10:int(self.sensor3_y)+10]))/400.
        if self.sensor1_x>longueur-10 or self.sensor1_x<10 or self.sensor1_y>largeur-10 or self.sensor1_y<10:
            self.signal1 = 1.
        if self.sensor2_x>longueur-10 or self.sensor2_x<10 or self.sensor2_y>largeur-10 or self.sensor2_y<10:
            self.signal2 = 1.
        if self.sensor3_x>longueur-10 or self.sensor3_x<10 or self.sensor3_y>largeur-10 or self.sensor3_y<10:
            self.signal3 = 1.

    def getPolarLidar(self, num, box_size):
         result = [-1] * num
         pt = self.pos
         rotation = self.velocity

         clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
         half = int(math.floor(box_size / 2))

         x_range = range( clamp(int(pt[0] - half), 0, longueur)  , clamp(int(pt[0] + half), 0, longueur))
         y_range = range(   clamp(int(pt[1] - half), 0, largeur) , clamp(int(pt[1] + half), 0, largeur))

         st = time.time()
         i = 0
         for px in x_range:
             for py in y_range:
                 i += 1
                 if sand[px][py] != 0 or px == 0 or px == largeur - 1 or py == 0 or py == largeur - 1:
                     #print((px, py))
                     dx = px - pt[0]
                     dy = py - pt[1]
                     angle = Vector(*rotation).angle((dx,dy))
                     if angle < 0 :
                         angle = 360.0 + angle

                     dist = Vector(px,py).distance(pt)
                     index = int(math.floor(angle / 360 * num))
                     index = clamp(index, 0, num - 1)

                     if dist < box_size and (result[index] == -1 or result[index] > dist):
                         result[index] = dist

         #print("sig:  " + str(time.time() - st))
         #print(i)
         return result

    def getShrinkPolarLidar(self, num, box_size, shrink_factor):
        global shrink_sand
        result = [-1] * num
        pt = [self.pos[0], self.pos[1]]
        pt[0] = pt[0] / shrink_factor
        pt[1] = pt[1] / shrink_factor
        rotation = self.velocity

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        half = int(math.floor(box_size / 2 / shrink_factor))

        width = len(shrink_sand)
        height = len(shrink_sand[0])

        x_range = range( clamp(int(pt[0] - half), 0, width)  , clamp(int(pt[0] + half), 0, width))
        y_range = range( clamp(int(pt[1] - half), 0, height) , clamp(int(pt[1] + half), 0, height))

        #st = time.time()
        for px in x_range:
         for py in y_range:
             if shrink_sand[px][py] != 0 or px == 0 or px == width - 1 or py == 0 or py == height - 1:
                 #print((px, py))
                 dx = px - pt[0]
                 dy = py - pt[1]
                 angle = Vector(*rotation).angle((dx,dy))
                 if angle < 0 :
                     angle = 360.0 + angle

                 dist = Vector(px,py).distance(pt)
                 index = int(math.floor(angle / 360 * num))
                 index = clamp(index, 0, num - 1)

                 if dist < box_size and (result[index] == -1 or result[index] > dist):
                     result[index] = dist

        #print("sig:  " + str(time.time() - st))
        #print(i)
        return result

    def fromLidarToDensity(self, point_num, num, box_size, lidar = None):
        if lidar == None:
            lidar = self.getShrinkPolarLidar(point_num, box_size, 4)
            #lidar = self.getPolarLidar(point_num, box_size)
        anglePerSector = int(point_num / num)
        result = []
        for i in range(num):
            index = i * anglePerSector
            sum = 0
            count = 0
            for j in range(index, index + anglePerSector):
                if lidar[j] != -1:
                    count += 1
                    sum += lidar[j]
            if count != 0:
                result.append(sum / count)
            else:
                result.append(-1)
            #result.append(count)
        return result

class Ball1(Widget):
    pass
class Ball2(Widget):
    pass
class Ball3(Widget):
    pass
class Circle(Widget):
    pass
class Destinity(Widget):
    pass

# Creating the game class

class Game(Widget):

    car = ObjectProperty(None)
    ball1 = ObjectProperty(None)
    ball2 = ObjectProperty(None)
    ball3 = ObjectProperty(None)


    def updateGoal(self):
        global goal_x, goal_y

        if len(self.goals) == 0:
            dist = 200
            if goal_x == dist:
                goal_x = longueur - dist
                goal_y = dist
            else:
                goal_x = dist
                goal_y = largeur - dist
        else:
            goal_x = self.goals[self.goal_index][0]
            goal_y = self.goals[self.goal_index][1]
            self.goal_index += 1
            self.goal_index = self.goal_index % len(self.goals)
        self.dest.center = (goal_x, goal_y)

    def serve_car(self, sec_num, box_size, brain, circle, dest, goals):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.sec_num = sec_num
        self.box_size = box_size
        self.brain = brain
        self.dest = dest
        self.circle = circle
        self.goals = goals
        self.goal_index = 0
        self.consecutive_pos = 0
        self.goal_num = 90

    def update(self):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur
        global shrink_updated, sand

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.updateGoal()

        if shrink_updated == False:
            self.shrinkSand(sand, (4,4))
            print("updated")

        orientation = (self.car.angle % 360) /180.0

        #last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        last_signal = self.car.fromLidarToDensity(3000, self.sec_num, self.box_size)
        last_signal.append(orientation)
        last_signal.append(-orientation)

        if last_reward >= 0:
            if self.consecutive_pos > self.goal_num and False:
                self.goal_num += 30
                self.brain.save(str(datetime.datetime.now()))
            self.consecutive_pos += 1
            pass
        else:
            self.consecutive_pos = 0

        action = self.brain.update(last_reward, last_signal)
        scores.append(self.brain.score())
        rotation = action2rotation[action]
        last_pos = self.car.center

        self.car.move(rotation)
        self.circle.center = self.car.pos
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
        x_min = int(clamp(self.car.pos[0] - 7, 0, longueur))
        x_max = int(clamp(self.car.pos[0] + 7, 0, longueur))
        y_min = int(clamp(self.car.pos[1] - 7, 0, largeur))
        y_max = int(clamp(self.car.pos[1] + 7, 0, largeur))
        surround_sand = 0
        for i in range(x_min, x_max):
            for j in range(y_min, y_max):
                surround_sand += sand[i][j]

        if surround_sand > 0:
            last_reward = -1
            # backward
            new_v = Vector(*self.car.velocity).normalize()
            new_v = new_v * -15
            self.car.pos = new_v + self.car.pos
            #self.car.velocity = Vector(1, 0).rotate(self.car.angle)

        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.2

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1.5
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1.5
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1.5
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1.5

        if distance < 100:
            self.updateGoal()
        last_distance = distance

    def shrinkSand(self, m, kernale_shape):
        global shrink_sand, shrink_updated
        shrink_sand = skimage.measure.block_reduce(m, kernale_shape, np.max)
        shrink_updated = True

# Adding the painting tools

class MyPaintWidget(Widget):

    def __init__(self, goals):
        super(MyPaintWidget, self).__init__()
        self.setgoals = False
        self.goals = goals


    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y, shrink_updated
        if self.setgoals == False:
            with self.canvas:
                Color(0.8,0.7,0)
                d = 10.
                touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
                last_x = int(touch.x)
                last_y = int(touch.y)
                n_points = 0
                length = 0
                sand[int(touch.x),int(touch.y)] = 1
                shrink_updated = False
        else:
            self.goals.append( (int(touch.x), int(touch.y)) )
            self.parent.dest.center = ((int(touch.x), int(touch.y)))

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y, shrink_updated
        if self.setgoals == False:
            if touch.button == 'left':
                touch.ud['line'].points += [touch.x, touch.y]
                x = int(touch.x)
                y = int(touch.y)
                length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
                n_points += 1.
                touch.ud['line'].width = int(30)
                sand[int(touch.x) - 30 : int(touch.x) + 30, int(touch.y) - 30 : int(touch.y) + 30] = 1
                last_x = x
                last_y = y
                shrink_updated = False
        else:
            pass

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        self.paused = True
        self.last_size = [0,0]
        self.goals = []

        self.parent = Game()
        parent = self.parent
        sec_num = 27
        box_size = 800
        self.brain = Dqn(sec_num + 2,3,0.9)

        circle = Circle()
        dest = Destinity()

        parent.serve_car(circle = circle, dest = dest, sec_num = sec_num, box_size = box_size, brain = self.brain, goals = self.goals)
        Clock.schedule_interval(self.pauseCheck, 1.0/60.0)
        #Clock.schedule_interval(parent.update, 0)

        self.painter = MyPaintWidget(self.goals)
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        self.pausebtn = Button(text = 'start', pos = (3 * parent.width, 0))
        plotbtn = Button(text = 'plot', pos = (4 * parent.width, 0))
        self.setGoalsbtn = Button(text = 'drawing', pos = (5 * parent.width, 0))

        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        self.pausebtn.bind(on_release = self.pauseSwitch)
        plotbtn.bind(on_release = self.plot)
        self.setGoalsbtn.bind(on_release = self.setGoals)

        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(self.pausebtn)
        parent.add_widget(plotbtn)
        parent.add_widget(self.setGoalsbtn)

        parent.add_widget(dest)
        parent.add_widget(circle)
        return parent

    def pauseCheck(self, dt):
        if self.paused == False:
            self.parent.update()
        else:
            if self.last_size != self.parent.size:
                print("resized")
                global longueur
                global largeur
                longueur = self.parent.width
                largeur = self.parent.height
                self.parent.car.pos[0] = longueur * 0.5
                self.parent.car.pos[1] = largeur * 0.125
                self.painter.canvas.clear()
                init()
                self.parent.update()
                while len(self.goals) > 0:
                    self.goals.pop()
                print('goals cleared')
                self.last_size[0] = self.parent.size[0]
                self.last_size[1] = self.parent.size[1]
                shrink_updated = False

    def pauseSwitch(self,obj):
        self.paused = 1 - self.paused
        if self.paused == True:
            self.pausebtn.text = 'start'
        else:
            self.pausebtn.text = 'pause'
            self.parent.updateGoal()

    def setGoals(self, obj):
        self.painter.setgoals = 1 - self.painter.setgoals

        if self.painter.setgoals == True:
            self.setGoalsbtn.text = 'setting goals'
        else:
            self.setGoalsbtn.text = 'drawing'
            del self.goals[:]
            print('goals cleared')

    def plot(self, obj):
        plt.plot(scores)
        plt.show()

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))
        shrink_updated = False

    def save(self, obj):
        print("saving brain...")
        self.brain.save()
        plt.plot(scores)
        plt.show()

    def load(self, obj):
        print("loading last saved brain...")
        self.brain.load()

# Running the whole thing
if __name__ == '__main__':
    CarApp().run()
