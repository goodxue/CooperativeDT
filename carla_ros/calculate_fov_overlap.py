import glob,pdb
import os
import sys
import json
import argparse
import logging
import time
import csv
import cv2

import numpy as np
import matplotlib.pyplot as plt
from utils import Transform, Rotation, Location

class Edge:
    def __init__(self, point_a, point_b):
        self._support_vector = np.array(point_a)
        self._direction_vector = np.subtract(point_b, point_a)

    def get_intersection_point(self, other):
        t = self._get_intersection_parameter(other)
        return None if t is None else self._get_point(t)

    def _get_point(self, parameter):
        return self._support_vector + parameter * self._direction_vector

    def _get_intersection_parameter(self, other):
        A = np.array([-self._direction_vector, other._direction_vector]).T
        if np.linalg.matrix_rank(A) < 2:
        	return None
        b = np.subtract(self._support_vector, other._support_vector)
        x = np.linalg.solve(A, b)
        return x[0] if 0 <= x[0] <= 1 and 0 <= x[1] <= 1 else None

def intersect(polygon1, polygon2):
    """
    The given polygons must be convex and their vertices must be in anti-clockwise order (this is not checked!)
    Example: polygon1 = [[0,0], [0,1], [1,1]]
    """
    polygon3 = list()
    polygon3.extend(_get_vertices_lying_in_the_other_polygon(polygon1, polygon2))
    polygon3.extend(_get_edge_intersection_points(polygon1, polygon2))
    return _sort_vertices_anti_clockwise_and_remove_duplicates(polygon3)


def _get_vertices_lying_in_the_other_polygon(polygon1, polygon2):
    vertices_lying_in_the_other_polygon = list()
    for corner in polygon1:
        if _polygon_contains_point(polygon2, corner):
            vertices_lying_in_the_other_polygon.append(corner)
    for corner in polygon2:
        if _polygon_contains_point(polygon1, corner):
            vertices_lying_in_the_other_polygon.append(corner)
    return vertices_lying_in_the_other_polygon


def _get_edge_intersection_points(polygon1, polygon2):
    intersection_points = list()
    for i in range(len(polygon1)):
        edge1 = Edge(polygon1[i-1], polygon1[i])
        for j in range(len(polygon2)):
            edge2 = Edge(polygon2[j-1], polygon2[j])
            intersection_point = edge1.get_intersection_point(edge2)
            if intersection_point is not None:
                intersection_points.append(intersection_point)
    return intersection_points


def _polygon_contains_point(polygon, point):
    for i in range(len(polygon)):
        a = np.subtract(polygon[i], polygon[i-1])
        b = np.subtract(point, polygon[i-1])
        if np.cross(a,b) < 0:
            return False
    return True


def _sort_vertices_anti_clockwise_and_remove_duplicates(polygon, tolerance=1e-7):
    polygon = sorted(polygon, key=lambda p: _get_angle_in_radians(_get_inner_point(polygon), p))

    def vertex_not_similar_to_previous(polygon, i):
        diff = np.subtract(polygon[i-1], polygon[i])
        return i==0 or np.linalg.norm(diff, np.inf) > tolerance

    return np.array([np.array(p) for i, p in enumerate(polygon) if vertex_not_similar_to_previous(polygon, i)])


def _get_angle_in_radians(p1, p2):
    return np.arctan2(p2[1]-p1[1], p2[0]-p1[0])


def _get_inner_point(polygon):
    x_coords = [p[0] for p in polygon]
    y_coords = [p[1] for p in polygon]
    return [(np.max(x_coords)+np.min(x_coords)) / 2.,(np.max(y_coords)+np.min(y_coords)) / 2.]


def plot_polygon(polygon):
    polygon = list(polygon)
    polygon.append(polygon[0])
    x,y = zip(*polygon)
    #print(type(x))
    plt.plot(x,y,'o-')


class FOV(object):
    def __init__(self,cam_transform,fov=90,sight=65,block=None):
        cam = cam_transform.location
        self.points = []
        self.points.append((cam.x,cam.y))
        self.points.append((cam.x + sight * np.cos(np.radians(cam_transform.rotation.yaw-0.5*fov)),cam.y + sight * np.sin(np.radians(cam_transform.rotation.yaw-0.5*fov))))
        self.points.append((cam.x+sight*np.cos(np.radians(cam_transform.rotation.yaw)),cam.y+sight*np.sin(np.radians(cam_transform.rotation.yaw))))
        self.points.append((cam.x + sight * np.cos(np.radians(cam_transform.rotation.yaw+0.5*fov)),cam.y + sight * np.sin(np.radians(cam_transform.rotation.yaw+0.5*fov))))
        
        if block is not None:
            raise NotImplementedError('block has not been implemented!')

    def is_inside(self,point):
        #1 in
        #0 on
        #-1 out
        return cv2.pointPolygonTest(points,point,True)

    def caculate_iou(self,fov):
        polygon1 = _sort_vertices_anti_clockwise_and_remove_duplicates(self.points)
        polygon2 = _sort_vertices_anti_clockwise_and_remove_duplicates(fov.points)
        polygon3 = intersect(polygon1, polygon2)
        plot_polygon(polygon1)
        plot_polygon(polygon2)
        img = np.zeros([500,500],np.int8)
        cv2.drawContours(img,[np.array(polygon3).reshape(-1,1,2).astype(np.int32)],0,(255,255,255),2)
        if len(polygon3) > 0:
            plot_polygon(polygon3))
            print(cv2.contourArea(np.array(polygon3).reshape(-1,1,2).astype(np.int32))/cv2.contourArea(np.array(polygon1).reshape(-1,1,2).astype(np.int32)))
        plt.show()

if __name__ == '__main__':
    cam1 = Transform(location=Location(x=100, y=30, z=4),rotation=Rotation(pitch=0, yaw=90, roll=0))
    cam2 = Transform(location=Location(x=100, y=60, z=4),rotation=Rotation(pitch=0, yaw=-90, roll=0))
    fov1 = FOV(cam1)
    fov2 = FOV(cam2)
    fov1.caculate_iou(fov2)
