class Rotation(object):
    def __init__(self,yaw=0,roll=0,pitch=0):
        self.yaw = yaw
        self.roll = roll
        self.pitch = pitch

class Location(object):
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z

class Transform(object):
    def __init__(self,rotation,location):
        self.rotation = rotation
        self.location = location

