#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, VehicleCorners, BoolStamped
from enum import Enum, auto
from collections import namedtuple


ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
DEBUG = False
DEBUG_TEXT = True
ENGLISH = False

class ControlNode(DTROS):
    class State(Enum):
        LF=auto()
        TAILING=auto()
        INTERSECTION=auto()

    class PD:
        def __init__(self,P=-0.049,D=0.004):
            self.proportional = None
            self.P = P
            self.D = D
            self.last_error = 0
            self.last_time = rospy.get_time()
            self.disabled_value=0

        def __repr__(self):
            return "<P={} D={} E={} DIS={}>".format(self.P, self.D, self.proportional, self.disabled_value)

        def get(self):
            if self.disabled_value:
                return self.disabled_value
            # P Term
            P = self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time) if self.last_error else 0
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            # print("PD terms: {} {}".format(P,D))

            return P+D

        def reset(self):
            self.proportional=0
            self.last_error=0
            self.last_time = rospy.get_time()

        def set_disable(self,value):
            self.disabled_value = value
            self.reset()



    def __init__(self, node_name):
        super(ControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        self.v_p = rospy.get_param("/e4/v_p",0.45)
        self.v_d = rospy.get_param("/e4/v_d",-0.014)
        self.o_lf_p = rospy.get_param("/e4/o_lf_p",-0.049)
        self.o_lf_d = rospy.get_param("/e4/o_lf_d",0.004)
        self.o_t_p = rospy.get_param("/e4/o_t_p",-0.025)
        self.o_t_d = rospy.get_param("/e4/o_t_d",0.006)

        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # Properties
        self.state = self.State.LF

        self.det_distance = None
        self.det_centers = None
        self.det_retry=10
        self.det_retry_counter = 0
        self.veh_distance = rospy.get_param("/e4/veh_dist", 0.24)

        self.lf_velocity = rospy.get_param("/e4/lf_v", 0.4)
        self.det_offset = rospy.get_param("/e4/det_offset", 50)

        self.omega_cap = rospy.get_param("/e4/o_cap", 11.0)
        self.det_tolerance = rospy.get_param("/e4/det_tor", 15.0)

        self.twist = Twist2DStamped(v=self.lf_velocity, omega=0)


        # PID Variables
        self.pd_omega_tail = self.PD(self.o_t_p,self.o_t_d)
        self.pd_omega_lf = self.PD(self.o_lf_p,self.o_lf_d)

        self.pd_omega = self.pd_omega_lf

        self.pd_v = self.PD(self.v_p,self.v_d)
        self.pd_v.set_disable(self.lf_velocity)


        if ENGLISH:
            self.lf_offset = -220
        else:
            self.lf_offset = 220

        # Publishers & Subscribers
        self.pub = rospy.Publisher("/" + self.veh + "/output/image/mask/compressed",
                                   CompressedImage,
                                   queue_size=1)
        self.sub = rospy.Subscriber("/" + self.veh + "/camera_node/image/compressed",
                                    CompressedImage,
                                    self.callback,
                                    queue_size=1,
                                    buff_size="20MB")
        self.vel_pub = rospy.Publisher("/" + self.veh + "/car_cmd_switch_node/cmd",
                                       Twist2DStamped,
                                       queue_size=1)

        self.sub_centers = rospy.Subscriber("/{}/duckiebot_detection_node/centers".format(self.veh), VehicleCorners,
                                            self.cb_process_centers, queue_size=1)

        self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.veh),
                                                            Float32,
                                                            self.cb_dist_bot, queue_size=1)

        self.sub_det = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.veh),
                                                            BoolStamped,
                                                            self.cb_det, queue_size=1)

        # Shutdown hook
        rospy.on_shutdown(self.hook)

    def cb_process_centers(self, msg):
        if self.state!=self.State.TAILING:
            return
        detection = msg.detection.data
        if detection:
            self.det_centers=msg.corners
            xs=np.array([i.x for i in self.det_centers])
            self.pd_omega.proportional = (xm:=xs.mean())-400+self.det_offset
            if (abs(self.pd_omega.proportional)<self.det_tolerance) or (self.det_distance<self.veh_distance):
                self.pd_omega.reset()
            if DEBUG_TEXT:
                self.log("det centers: {}".format(xm))

    def cb_dist_bot(self, msg):
        if msg.data:
            self.det_distance = msg.data

    def cb_det(self, msg):
        if msg.data:
            if self.state != self.State.TAILING:
                self.state=self.State.TAILING
                self.pd_v.set_disable(0)
                self.pd_omega = self.pd_omega_tail
                self.pd_omega.reset()
                if DEBUG_TEXT:
                    self.log("start tailing")
            self.log("det: {}".format(self.det_distance))
            self.pd_v.proportional = self.det_distance - self.veh_distance
        else:
            if self.det_retry_counter<self.det_retry:
                self.log("lost det, retry")
                self.det_retry_counter+=1
                self.pd_v.reset()
                self.pd_omega.reset()
            else:
                self.state=self.State.LF
                self.pd_v.set_disable(self.lf_velocity)
                self.pd_omega=self.pd_omega_lf
                self.pd_omega.reset()
                self.det_retry_counter=0
                if DEBUG_TEXT:
                    self.log("end tailing")


    def callback(self, msg):
        if self.state!=self.State.LF:
            return
        img = self.jpeg.decode(msg.data)
        crop = img[300:-1, :, :]
        crop_width = crop.shape[1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, ROAD_MASK[0], ROAD_MASK[1])
        crop = cv2.bitwise_and(crop, crop, mask=mask)
        contours, hierarchy = cv2.findContours(mask,
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_NONE)

        # Search for lane in front
        max_area = 20
        max_idx = -1
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > max_area:
                max_idx = i
                max_area = area

        if max_idx != -1:
            M = cv2.moments(contours[max_idx])
            try:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                self.pd_omega.proportional = cx - int(crop_width / 2) + self.lf_offset
                if DEBUG:
                    cv2.drawContours(crop, contours, max_idx, (0, 255, 0), 3)
                    cv2.circle(crop, (cx, cy), 7, (0, 0, 255), -1)
            except:
                pass
        else:
            self.pd_omega.proportional = None

        if DEBUG:
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop))
            self.pub.publish(rect_img_msg)

    def drive(self):
        if self.pd_omega.proportional is None:
            self.twist.omega = 0
        else:
            self.twist.v = max(self.pd_v.get(),0)
            self.twist.omega = min(self.pd_omega.get(), self.omega_cap)
            if DEBUG_TEXT:
                self.loginfo([self.state, self.pd_omega, self.pd_v, self.twist.omega, self.twist.v])

        self.vel_pub.publish(self.twist)

    def hook(self):
        print("SHUTTING DOWN")
        self.twist.v = 0
        self.twist.omega = 0
        self.vel_pub.publish(self.twist)
        for i in range(8):
            self.vel_pub.publish(self.twist)


if __name__ == "__main__":
    node = ControlNode("control_node")
    rate = rospy.Rate(8)  # 8hz
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()