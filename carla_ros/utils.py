import numpy as np
import cv2

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
    

class CamVehicle(object):
    def __init__(self,x,y,z,dh,dw,dl,ry,cid=-1):
        self.x = x
        self.y = y
        self.z = z
        self.height = dh
        self.width = dw
        self.length = dl
        self.rotation_y = ry
        self.id = cid
    
    @classmethod
    def by_location(self,location,dh,dw,dl,ry):
        return self(location.x,location.y,location.z,dh,dw,dl,ry)

    def compute_box_3d(self):
        # dim: 3
        # location: 3
        # rotation_y: 1
        # return: 8 x 3
        c, s = np.cos(self.rotation_y), np.sin(self.rotation_y)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
        l, w, h = self.length, self.width, self.height
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0,0,0,0,-h,-h,-h,-h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
        corners_3d = np.dot(R, corners) 
        corners_3d = corners_3d + np.array([self.x,self.y,self.z], dtype=np.float32).reshape(3, 1)
        return corners_3d.transpose(1, 0)
        

class ClientSideBoundingBoxes(object):
    """
    This is a module responsible for creating 3D bounding boxes and drawing them
    client-side on pygame surface.
    """

    @staticmethod
    def get_bounding_boxes(vehicles, camera):
        """
        Creates 3D bounding boxes based on carla vehicle list and camera.
        """

        bounding_boxes = [ClientSideBoundingBoxes.get_bounding_box(vehicle, camera) for vehicle in vehicles]
        # filter objects behind camera
        bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
        return bounding_boxes

    @staticmethod
    def draw_bounding_boxes(display, bounding_boxes):
        """
        Draws bounding boxes on pygame display.
        """

        bb_surface = pygame.Surface((VIEW_WIDTH, VIEW_HEIGHT))
        bb_surface.set_colorkey((0, 0, 0))
        for bbox in bounding_boxes:
            points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
            # pdb.set_trace()
            # draw lines
            # base
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
            # top
            pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
            pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
            # base-top
            pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
            pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
            pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
            pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
        display.blit(bb_surface, (0, 0))

    @staticmethod
    def get_bounding_box(vehicle, camera):
        """
        Returns 3D bounding box for a vehicle based on camera view.
        """

        bb_cords = ClientSideBoundingBoxes._create_bb_points(vehicle)
        cords_x_y_z = ClientSideBoundingBoxes._vehicle_to_sensor(bb_cords, vehicle, camera)[:3, :]
        cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
        bbox = np.transpose(np.dot(camera_coordinate(camera), cords_y_minus_z_x))
        camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
        return camera_bbox

    @staticmethod
    def _create_bb_points(vehicle):
        """
        Returns 3D bounding box for a vehicle.
        """

        cords = np.zeros((8, 4))
        extent = vehicle.bounding_box.extent
        cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
        cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
        cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
        cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
        cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
        cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
        cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
        cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
        return cords

    @staticmethod
    def _vehicle_to_sensor(cords, vehicle, sensor):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = ClientSideBoundingBoxes._vehicle_to_world(cords, vehicle)
        sensor_cord = ClientSideBoundingBoxes._world_to_sensor(world_cord, sensor)
        return sensor_cord

    @staticmethod
    def _vehicle_to_world(cords, vehicle):
        """
        Transforms coordinates of a vehicle bounding box to world.
        """
        bb_transform = carla.Transform(vehicle.bounding_box.location)
        bb_vehicle_matrix = ClientSideBoundingBoxes.get_matrix(bb_transform)
        vehicle_world_matrix = ClientSideBoundingBoxes.get_matrix(vehicle.get_transform())
        bb_world_matrix = np.dot(vehicle_world_matrix, bb_vehicle_matrix)
        world_cords = np.dot(bb_world_matrix, np.transpose(cords))
        return world_cords

    @staticmethod
    def _world_to_sensor(cords, sensor):
        """
        Transforms world coordinates to sensor.
        """

        sensor_world_matrix = ClientSideBoundingBoxes.get_matrix(sensor.get_transform()) #注意世界坐标其实就是相对世界原点的变换（平移旋转）矩阵，这点很重要
        world_sensor_matrix = np.linalg.inv(sensor_world_matrix)  #相机的世界坐标求逆，将世界坐标转换到了相机的局部坐标！！！
        sensor_cords = np.dot(world_sensor_matrix, cords)
        return sensor_cords

    @staticmethod
    def get_matrix(transform):
        """
        Creates matrix from carla transform.
        """
        rotation = transform.rotation
        location = transform.location
        c_y = np.cos(np.radians(rotation.yaw))
        s_y = np.sin(np.radians(rotation.yaw))
        c_r = np.cos(np.radians(rotation.roll))
        s_r = np.sin(np.radians(rotation.roll))
        c_p = np.cos(np.radians(rotation.pitch))
        s_p = np.sin(np.radians(rotation.pitch))
        matrix = np.matrix(np.identity(4))
        matrix[0, 3] = location.x
        matrix[1, 3] = location.y
        matrix[2, 3] = location.z
        matrix[0, 0] = c_p * c_y
        matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
        matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
        matrix[1, 0] = s_y * c_p
        matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
        matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
        matrix[2, 0] = s_p
        matrix[2, 1] = -c_p * s_r
        matrix[2, 2] = c_p * c_r
        return matrix

def compute_box_3d(dim, location, rotation_y):
    # dim: 3
    # location: 3
    # rotation_y: 1
    # return: 8 x 3
    c, s = np.cos(rotation_y), np.sin(rotation_y)
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[2], dim[1], dim[0]
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [0,0,0,0,-h,-h,-h,-h]
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

    corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = np.dot(R, corners) 
    corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
    return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
    # pts_3d: n x 3
    # P: 3 x 4
    # return: n x 2
    #   pts_3d_homo = np.concatenate(
    #     [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
    pts_2d = np.dot(P, pts_3d.transpose(1, 0)).transpose(1, 0)
    pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
    # import pdb; pdb.set_trace()
    return pts_2d

def draw_box_3d(image, corners, c=(0, 0, 255)):
    face_idx = [[0,1,5,4],
                [1,2,6, 5],
                [2,3,7,6],
                [3,0,4,7]]
    for ind_f in range(3, -1, -1):
        f = face_idx[ind_f]
        for j in range(4):
            cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
                    (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 1, lineType=cv2.LINE_AA)
        if ind_f == 0:
            cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
                    (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
            cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
                    (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
    return image

def draw_box_2d(image, corners, c=(255,0,0)):
    face_idx = [[0,1,2,1],
                [2,1,2,3],
                [2,3,0,3],
                [0,3,0,1]]
    for ind_f in range(4):
        f = face_idx[ind_f]
        cv2.line(image, (corners[f[0]],corners[f[1]]),
                  (corners[f[2]], corners[f[3]]), c, 1, lineType=cv2.LINE_AA)
    return image


def read_clib(calib_path):
    f = open(calib_path, 'r')
    for i, line in enumerate(f):
        if i == 0:
            calib = np.array(line[:-1].split(' ')[1:], dtype=np.float32)
            calib = calib.reshape(3, 3)
            return calib

outsize = 384
world_size = 128
def add_bird_view(rects, bird_view = None,center_thresh=0.3, img_id='bird',outsize=384,lc=(250, 152, 12),lw=2):
    if bird_view is None:
        bird_view = np.ones((outsize, outsize, 3), dtype=np.uint8) * 230
    #bird_view = np.ones((outsize, outsize, 3), dtype=np.uint8) * 230
    #lc = (250, 152, 12)
    for rect in rects:
        rect = rect[:4, [0, 2]]
        for k in range(4):
            rect[k] = project_3d_to_bird(rect[k])
            # cv2.circle(bird_view, (rect[k][0], rect[k][1]), 2, lc, -1)
        cv2.polylines(
            bird_view,[rect.reshape(-1, 1, 2).astype(np.int32)],
            True,lc,lw,lineType=cv2.LINE_AA)
            # for e in [[0, 1]]:
            #     t = 4 if e == [0, 1] else 1
            #     cv2.line(bird_view, (rect[e[0]][0], rect[e[0]][1]),
            #             (rect[e[1]][0], rect[e[1]][1]), lc, t,
            #             lineType=cv2.LINE_AA)
    return bird_view

def project_3d_to_bird(pt):
    pt[0] += world_size / 2
    pt[1] = world_size - pt[1] - world_size / 3
    pt = pt * outsize / world_size
    return pt.astype(np.int32)

def ry_filter(ry):
    if ry>np.pi:
        ry-=np.pi
    if ry<-np.pi:
        ry+=np.pi
    return ry

def ry_filter_a(ry):
    if ry>180:
        ry-=180
    if ry<-180:
        ry+=180
    return ry

# #draw two camera field of view in BEV, relatived to camera tatget
# def cam_bird_view(camtarget,camsource,bird_view=None,FOV=90,lc1=(100,100,100),lc2=(255,255,0)):
#     #camtarget(Transform)
#     #camsource(Transform)
#     if bird_view is None:
#         bird_view = np.ones((outsize, outsize, 3), dtype=np.uint8) * 230
#     # cam1_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
#     # cam2_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
#     #cam target
#     cam1_point = (int(outsize/2), int(outsize * 2 / 3))
#     cv2.line(bird_view, cam1_point,
#         (int(cam1_point[0]+outsize/2), int(cam1_point[1]-outsize/2)), lc1, 1,
#         lineType=cv2.LINE_AA)
#     cv2.line(bird_view, cam1_point,
#         (int(cam1_point[0]-outsize/2), int(cam1_point[1]-outsize/2)), lc1, 1,
#         lineType=cv2.LINE_AA)
    
#     #cam source
#     # cord_p = np.zeros((1,4))
#     # cord_p[0][3] = 1
#     pol = outsize / world_size
#     cam2_point = (int(cam1_point[0]-pol*(camtarget.location.y-camsource.location.y)),int(cam1_point[1]-pol*(camtarget.location.x-camsource.location.x)))
#     yaw = -(camtarget.rotation.yaw-camsource.rotation.yaw)
#     cv2.line(bird_view, cam2_point,
#         (int(cam2_point[0]+pol*outsize * np.cos(np.radians(yaw-FOV/2))), int(cam2_point[1]-pol*outsize*np.sin(np.radians(yaw-FOV/2)))), lc2, 1,
#         lineType=cv2.LINE_AA)
#     cv2.line(bird_view, cam2_point,
#         (int(cam2_point[0]+pol*outsize * np.cos(np.radians(yaw+FOV/2))), int(cam2_point[1]-pol*outsize*np.sin(np.radians(yaw+FOV/2)))), lc2, 1,
#         lineType=cv2.LINE_AA)
#     # cv2.line(bird_view, cam2_point,
#     #     (cam2_point[0]-world_size/2, cam2_point[1]-world_size/2), lc2, 1,
#     #     lineType=cv2.LINE_AA)
#     return bird_view

#draw two camera field of view in BEV, relatived to camera tatget
def cam_bird_view(camtarget,camsource,bird_view=None,FOV=90,lc1=(100,100,100),lc2=(255,255,0)):
    #camtarget(Transform)
    #camsource(Transform)
    cam_range = 50
    pol = outsize / world_size
    if bird_view is None:
        bird_view = np.ones((outsize, outsize, 3), dtype=np.uint8) * 230
    # cam1_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
    # cam2_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
    #cam target
    cam1_point = (int(outsize/2), int(outsize * 2 / 3))
    cv2.line(bird_view, cam1_point,
        (int(cam1_point[0]+pol*cam_range), int(cam1_point[1]-pol*cam_range)), lc1, 1,
        lineType=cv2.LINE_AA)
    cv2.line(bird_view, cam1_point,
        (int(cam1_point[0]-pol*cam_range), int(cam1_point[1]-pol*cam_range)), lc1, 1,
        lineType=cv2.LINE_AA)
    
    #cam source
    # cord_p = np.zeros((1,4))
    # cord_p[0][3] = 1
    
    
    cam2_point = (int(cam1_point[0]-pol*(camtarget.location.y-camsource.location.y)),int(cam1_point[1]-pol*(camtarget.location.x-camsource.location.x)))
    yaw = -(camtarget.rotation.yaw-camsource.rotation.yaw)
    nadd1 = cam2_point[0] + pol*cam_range * np.cos(np.radians(yaw-FOV/2))
    nred1 = cam2_point[1] - pol*cam_range*np.sin(np.radians(yaw-FOV/2))
    nadd2 = cam2_point[0]+pol*cam_range * np.cos(np.radians(yaw+FOV/2))
    nred2 = cam2_point[1]-pol*cam_range*np.sin(np.radians(yaw+FOV/2))
    cv2.line(bird_view, cam2_point,
        (int(nadd1), int(nred1)), lc2, 1,
        lineType=cv2.LINE_AA)
    cv2.line(bird_view, cam2_point,
        (int(nadd2), int(nred2)), lc2, 1,
        lineType=cv2.LINE_AA)
    # cv2.line(bird_view, cam2_point,
    #     (cam2_point[0]-world_size/2, cam2_point[1]-world_size/2), lc2, 1,
    #     lineType=cv2.LINE_AA)
    return bird_view

def cams_bird_view(centerpoint,camTransform_list,bird_view=None,FOV=90):
    #camtarget(Transform)
    #camsource(Transform)

    ncam = len(camTransform_list)
    cam_range = 50
    pol = outsize / world_size
    if bird_view is None:
        bird_view = np.ones((outsize, outsize, 3), dtype=np.uint8) * 230
    # cam1_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
    # cam2_matrix = ClientSideBoundingBoxes.get_matrix(camtarget)
    #cam target
    centerimg_point = (int(outsize/2), int(outsize * 2 / 3))
    
    for i,cam in enumerate(camTransform_list):
        cam_point = (int(centerimg_point[0]-pol*(centerpoint.y - cam.location.y)),int(centerimg_point[1]-pol*(centerpoint.x - cam.location.x)))
        yaw = -(cam.rotation.yaw - 90)
        nadd1 = cam_point[0] + pol*cam_range * np.cos(np.radians(yaw-FOV/2))
        nred1 = cam_point[1] - pol*cam_range*np.sin(np.radians(yaw-FOV/2))
        nadd2 = cam_point[0]+pol*cam_range * np.cos(np.radians(yaw+FOV/2))
        nred2 = cam_point[1]-pol*cam_range*np.sin(np.radians(yaw+FOV/2))
        lc2 = (int(255/n*i),int(255-255/n*i),255)
        cv2.line(bird_view, cam2_point,
        (int(nadd1), int(nred1)), lc2, 1,
        lineType=cv2.LINE_AA)
        cv2.line(bird_view, cam2_point,
            (int(nadd2), int(nred2)), lc2, 1,
            lineType=cv2.LINE_AA)
    #cam source
    # cord_p = np.zeros((1,4))
    # cord_p[0][3] = 1
    # cv2.line(bird_view, cam2_point,
    #     (cam2_point[0]-world_size/2, cam2_point[1]-world_size/2), lc2, 1,
    #     lineType=cv2.LINE_AA)
    return bird_view