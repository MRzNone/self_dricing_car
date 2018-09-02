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
    sand = np.zeros((longueur,largeur))
    goal_x = 20
    goal_y = largeur - 20
    first_update = False

# Initializing the last distance
last_distance = 0

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
         pt = self.center
         rotation = self.velocity

         clamp = lambda n, minn, maxn: max(min(maxn, n), minn)
         half = int(math.floor(box_size / 2))

         x_range = range( clamp(int(pt[0] - half), 0, longueur)  , clamp(int(pt[0] + half), 0, longueur)  )
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

    def fromLidarToDensity(self, point_num, num, box_size):
         lidar = self.getPolarLidar(point_num, box_size)
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
                 result.append(0)
             result.append(count)
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
        dist = 150
        global goal_x, goal_y

        if goal_x == dist:
            goal_x = longueur - dist
            goal_y = dist
        else:
            goal_x = dist
            goal_y = largeur - dist
        self.dest.center = (goal_x, goal_y)

    def serve_car(self, sec_num, box_size, brain, circle, dest):
        self.car.center = self.center
        self.car.velocity = Vector(6, 0)
        self.sec_num = sec_num
        self.box_size = box_size
        self.brain = brain
        self.dest = dest
        self.circle = circle

    def update(self):

        global brain
        global last_reward
        global scores
        global last_distance
        global goal_x
        global goal_y
        global longueur
        global largeur

        longueur = self.width
        largeur = self.height
        if first_update:
            init()
            self.updateGoal()

        xx = goal_x - self.car.x
        yy = goal_y - self.car.y
        orientation = Vector(*self.car.velocity).angle((xx,yy))/180.

        #last_signal = [self.car.signal1, self.car.signal2, self.car.signal3, orientation, -orientation]
        last_signal = self.car.fromLidarToDensity(3000, self.sec_num, self.box_size)
        last_signal.append(orientation)
        last_signal.append(-orientation)

        action = self.brain.update(last_reward, last_signal)
        scores.append(self.brain.score())
        rotation = action2rotation[action]
        self.car.move(rotation)
        self.circle.center = self.car.center
        distance = np.sqrt((self.car.x - goal_x)**2 + (self.car.y - goal_y)**2)
        self.ball1.pos = self.car.sensor1
        self.ball2.pos = self.car.sensor2
        self.ball3.pos = self.car.sensor3

        if sand[int(self.car.x),int(self.car.y)] > 0:
            self.car.velocity = Vector(1, 0).rotate(self.car.angle)
            last_reward = -1
        else: # otherwise
            self.car.velocity = Vector(6, 0).rotate(self.car.angle)
            last_reward = -0.2
            if distance < last_distance:
                last_reward = 0.1

        if self.car.x < 10:
            self.car.x = 10
            last_reward = -1
        if self.car.x > self.width - 10:
            self.car.x = self.width - 10
            last_reward = -1
        if self.car.y < 10:
            self.car.y = 10
            last_reward = -1
        if self.car.y > self.height - 10:
            self.car.y = self.height - 10
            last_reward = -1

        if distance < 100:
            self.updateGoal()
        last_distance = distance

# Adding the painting tools

class MyPaintWidget(Widget):

    def on_touch_down(self, touch):
        global length, n_points, last_x, last_y
        with self.canvas:
            Color(0.8,0.7,0)
            d = 10.
            touch.ud['line'] = Line(points = (touch.x, touch.y), width = 10)
            last_x = int(touch.x)
            last_y = int(touch.y)
            n_points = 0
            length = 0
            sand[int(touch.x),int(touch.y)] = 1

    def on_touch_move(self, touch):
        global length, n_points, last_x, last_y
        if touch.button == 'left':
            touch.ud['line'].points += [touch.x, touch.y]
            x = int(touch.x)
            y = int(touch.y)
            length += np.sqrt(max((x - last_x)**2 + (y - last_y)**2, 2))
            n_points += 1.
            density = n_points/(length)
            touch.ud['line'].width = int(20 * density + 1)
            sand[int(touch.x) - 10 : int(touch.x) + 10, int(touch.y) - 10 : int(touch.y) + 10] = 1
            last_x = x
            last_y = y

# Adding the API Buttons (clear, save and load)

class CarApp(App):

    def build(self):
        self.paused = True
        self.last_size = [0,0]

        self.parent = Game()
        parent = self.parent
        sec_num = 12
        box_size = 150
        self.brain = Dqn(sec_num * 2 + 2,3,0.9)

        circle = Circle()
        dest = Destinity()

        parent.serve_car(circle = circle, dest = dest, sec_num = sec_num, box_size = box_size, brain = self.brain)
        Clock.schedule_interval(self.pauseCheck, 1.0/60.0)
        #Clock.schedule_interval(parent.update, 0)
        self.painter = MyPaintWidget()
        clearbtn = Button(text = 'clear')
        savebtn = Button(text = 'save', pos = (parent.width, 0))
        loadbtn = Button(text = 'load', pos = (2 * parent.width, 0))
        pausebtn = Button(text = 'start', pos = (2 * parent.width, 0))
        clearbtn.bind(on_release = self.clear_canvas)
        savebtn.bind(on_release = self.save)
        loadbtn.bind(on_release = self.load)
        pausebtn.bind(on_release = self.pauseSwitch)
        parent.add_widget(self.painter)
        parent.add_widget(clearbtn)
        parent.add_widget(savebtn)
        parent.add_widget(loadbtn)
        parent.add_widget(pausebtn)
        parent.add_widget(dest)
        parent.add_widget(circle)
        return parent

    def pauseCheck(self, dt):
        if self.paused == False:
            self.parent.update()
        else:
            if self.last_size != self.parent.size:
                global longueur
                global largeur
                longueur = self.parent.width
                largeur = self.parent.height
                self.parent.car.center[0] = longueur * 0.5
                self.parent.car.center[1] = largeur * 0.25
                self.painter.canvas.clear()
                init()
                self.parent.update()
                self.parent.updateGoal()
                self.last_size[0] = self.parent.size[0]
                self.last_size[1] = self.parent.size[1]

    def pauseSwitch(self,obj):
        self.paused = 1 - self.paused

    def clear_canvas(self, obj):
        global sand
        self.painter.canvas.clear()
        sand = np.zeros((longueur,largeur))

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
