#!/usr/bin/env python3

import rospy

from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CameraInfo, CompressedImage
from std_msgs.msg import Float32
from turbojpeg import TurboJPEG
import cv2
import numpy as np
import math
from duckietown_msgs.msg import WheelsCmdStamped, Twist2DStamped, VehicleCorners, BoolStamped
from duckietown_msgs.srv import ChangePattern, ChangePatternResponse
from std_msgs.msg import String
from enum import Enum, auto
from collections import namedtuple


ROAD_MASK = [(20, 60, 0), (50, 255, 255)]
STOP_LINE_MASK = [(0, 130, 178), (179, 255, 255)]
DEBUG = False
DEBUG_TEXT = True
ENGLISH = False

class ControlNode(DTROS):
    class State(Enum):
        LF=auto()
        TAILING=auto()
        INTERSECTION=auto()
        STOPPING=auto()
        TURNING=auto()

    # PD class
    class PD:
        def __init__(self,P=-0.049,D=0.004):
            self.proportional = None
            self.P = P
            self.D = D
            self.last_error = 0
            self.last_time = rospy.get_time()
            self.disabled_value= None

        def __repr__(self):
            return "<P={} D={} E={} DIS={}>".format(self.P, self.D, self.proportional, self.disabled_value)

        def get(self):
            # get the output of the PD
            if self.disabled_value is not None:
                return self.disabled_value
            # P Term
            P = self.proportional * self.P

            # D Term
            d_error = (self.proportional - self.last_error) / (rospy.get_time() - self.last_time) if self.last_error else 0
            self.last_error = self.proportional
            self.last_time = rospy.get_time()
            D = d_error * self.D

            return P+D

        def reset(self):
            # reset the PD controller
            self.proportional=0
            self.last_error=0
            self.last_time = rospy.get_time()

        def set_disable(self,value):
            # set the PD controller to output a constant
            self.disabled_value = value
            self.reset()



    def __init__(self, node_name):
        super(ControlNode, self).__init__(node_name=node_name, node_type=NodeType.GENERIC)
        self.node_name = node_name
        self.veh = rospy.get_param("~veh")

        self.params={}
        def get_param(name, default):
            # getting parameters from rosparam
            if name not in self.params:
                self.params[name]=rospy.get_param(name, default)
            return self.params[name]

        self.v_p = get_param("/e4/v_p",1.1)
        self.v_d = get_param("/e4/v_d",-0.028)
        self.o_lf_p = get_param("/e4/o_lf_p",-0.049)
        self.o_lf_d = get_param("/e4/o_lf_d",0.004)
        self.o_t_p = get_param("/e4/o_t_p",-0.027)
        self.o_t_d = get_param("/e4/o_t_d",0.005)
        self.P_2 = get_param("/e4/v_stop_p",0.0055)


        self.jpeg = TurboJPEG()

        self.loginfo("Initialized")

        # Properties
        self.state = self.State.LF

        self.det_distance = None
        self.det_centers = None
        self.det_retry=10
        self.det_retry_counter = 0
        self.det_th=get_param("/e4/det_th", 0.5)
        self.veh_distance = get_param("/e4/veh_dist", 0.1)

        self.lf_velocity = get_param("/e4/lf_v", 0.42)
        self.det_offset = get_param("/e4/det_offset", 50)

        self.omega_cap = get_param("/e4/o_cap", 11.0)
        self.v_cap = get_param("/e4/v_cap", 0.82)
        self.det_tolerance = get_param("/e4/det_tor", 15.0)

        self.twist = Twist2DStamped(v=self.lf_velocity, omega=0)

        self.stop_ofs = 0.0
        self.stop_times_up = False
        self.stop_off=False
        self.stop_cb=None

        self.turn_hist=[]
        self.len_turn_hist=20
        self.turn_th=get_param("/e4/turn_th",50)
        self.turn_factor=get_param("/e4/turn_factor",6.6)
        self.turn_v = get_param("/e4/turn_v", 0.5)
        self.turn_time = get_param("/e4/turn_time", 2.0)
        self.LED=None
        global DEBUG, DEBUG_TEXT
        DEBUG = get_param("/e4/DEBUG", False)
        DEBUG_TEXT = get_param("/e4/DEBUG_TEXT", False)

        # PID Variables
        self.pd_omega_tail = self.PD(self.o_t_p,self.o_t_d)
        self.pd_omega_lf = self.PD(self.o_lf_p,self.o_lf_d)
        self.pd_omega = self.pd_omega_lf

        self.pd_v_tail = self.PD(self.v_p,self.v_d)
        self.pd_v = self.pd_v_tail
        self.pd_stopping_v = self.PD(P=self.P_2,D=0)
        self.pd_v_tail.set_disable(self.lf_velocity)


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

        # self.sub_centers = rospy.Subscriber("/{}/duckiebot_detection_node/centers".format(self.veh), VehicleCorners,
        #                                     self.cb_process_centers, queue_size=1)

        # self.sub_distance_to_robot_ahead = rospy.Subscriber("/{}/duckiebot_distance_node/distance".format(self.veh),
        #                                                     Float32,
        #                                                     self.cb_dist_bot, queue_size=1)

        # self.sub_det = rospy.Subscriber("/{}/duckiebot_detection_node/detection".format(self.veh),
        #                                                     BoolStamped,
        #                                                     self.cb_det, queue_size=1)

        # Services
        # self.srvp_led_emitter = rospy.ServiceProxy(
        #     "~set_pattern",
        #     ChangePattern)

        # Shutdown hook
        self.log("parameters:\n{}\n".format(self.params))
        rospy.on_shutdown(self.hook)

    def cb_process_centers(self, msg):
        # dot pattern centers callback
        detection = msg.detection.data
        if detection:
            self.det_centers=msg.corners
            xs=np.array([i.x for i in self.det_centers])
            if self.state==self.state.TAILING:
                self.pd_omega.proportional = (xm:=xs.mean())-400+self.det_offset  # x value of the pattern
                if (abs(self.pd_omega.proportional)<self.det_tolerance) or (self.det_distance<self.veh_distance):
                    self.pd_omega.reset()
                if DEBUG_TEXT:
                    self.log("det centers: {}".format(xm))
            elif self.state==self.State.STOPPING:
                self.push_hist((xm:=xs.mean())-400+self.det_offset)

    def cb_dist_bot(self, msg):
        if msg.data:
            # update detected distance
            self.det_distance = msg.data

    def cb_det(self, msg):
        # vehicle detection (boolean) callback
        if msg.data and self.det_distance<self.det_th:  # make sure under detection threshold
            if self.state == self.State.STOPPING:
                return
            if self.state != self.State.TAILING:
                # switch state and PD controllers
                self.state=self.State.TAILING
                self.pd_v_tail.set_disable(None)
                self.pd_omega = self.pd_omega_tail
                self.pd_omega.reset()
                if DEBUG_TEXT:
                    self.log("start tailing")
            if DEBUG_TEXT:
                self.log("det: {}".format(self.det_distance))
            self.pd_v.proportional = self.det_distance - self.veh_distance
        else:
            self.det_distance=math.inf  # assume infinite distance when no detection
            if self.state==self.State.LF or self.state == self.State.STOPPING:
                return
            if self.det_retry_counter<self.det_retry:
                if DEBUG_TEXT:
                    self.log("lost det, retry")
                self.det_retry_counter+=1
                self.pd_v.reset()
                self.pd_omega.reset()
            else:
                # switch state and PD controllers
                self.state=self.State.LF
                self.pd_v_tail.set_disable(self.lf_velocity)  # constant speed
                self.pd_omega=self.pd_omega_lf
                self.pd_omega.reset()
                self.det_retry_counter=0
                self.turn_hist.clear()
                if DEBUG_TEXT:
                    self.log("end tailing")

   # Calculates the midpoint of the contoured object 
    def midpoint (self, x, y, w, h):
        mid_x = int(x + (((x+w) - x)/2))
        mid_y = int(y + (((y+h) - y)))
        return (mid_x, mid_y)
    
    def set_led(self, color):
        if color is None:
            return
        self.log("Change LED: {}".format(color))
        msg = String()
        msg.data = color
        # try:
        #     self.srvp_led_emitter(msg)
        #     self.led_color=color
        # except Exception as e:
        #     self.log("Set LED error: {}".format(e))

    def cb_turn_timer(self,et):
        self.state = self.State.TAILING
        self.pd_v = self.pd_v_tail
        self.pd_omega=self.pd_omega_tail
        self.pd_v.set_disable(None)
        self.pd_omega.set_disable(None)
        if DEBUG_TEXT:
            self.log("turn end")

    def cb_stopping_timer(self, et):
        if self.state!=self.State.STOPPING:
            return
        self.pd_omega.set_disable(None)
        if DEBUG_TEXT:
            self.log("hist: {}".format(self.turn_hist))
        # hist=self.get_hist()
        #
        # if abs(hist)>self.turn_th:
        #     self.state=self.State.TURNING
        #     if DEBUG_TEXT:
        #         self.log("[stopping] turn")
        #     self.pd_v.set_disable(self.turn_v)
        #     if hist<0:
        #         self.log("[stopping] turn left {}".format(hist))
        #         self.pd_omega.set_disable(self.turn_factor)
        #     else:
        #         self.log("[stopping] turn right".format(hist))
        #         self.pd_omega.set_disable(-self.turn_factor)
        #     cb_turn=rospy.Timer(rospy.Duration(self.turn_time),self.cb_turn_timer,oneshot=True)
        # else:
        #     if DEBUG_TEXT:
        #         self.log("[stopping] resume to lane following")
        #     self.state=self.State.LF
        #     if self.pd_v!=self.pd_v_tail:
        #         self.pd_v=self.pd_v_tail
        #         self.pd_v.set_disable(self.lf_velocity)

        # switch state and PD controller
        self.state=self.State.TAILING
        self.pd_v=self.pd_v_tail
        self.pd_omega=self.pd_omega_tail
        self.pd_omega.reset()
        self.pd_v.reset()
        if DEBUG_TEXT:
            self.log("stopping timer up")
        self.stop_cb=None
        self.stop_off = True
        cb_clear = rospy.Timer(rospy.Duration(10), self.cb_clear, oneshot=True)

    def cb_clear(self,et):
        self.stop_off=False
        # if self.get_hist()>5.0:
        #     self.try_set_led("GREEN")
        # elif self.get_hist()<-5.0:
        #     self.try_set_led("BLUE")


    def callback(self, msg):
        img = self.jpeg.decode(msg.data)
        # Search for stop line
        if not self.stop_off:
            crop2 = img[320:480,300:640,:]

            cv2.line(crop2, (320, 240), (0,240), (255,0,0), 1)

            hsv2 = cv2.cvtColor(crop2, cv2.COLOR_BGR2HSV)
            mask2 = cv2.inRange(hsv2, STOP_LINE_MASK[0], STOP_LINE_MASK[1])
            contours, hierarchy = cv2.findContours(mask2,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:
                max_contour = max(contours, key=cv2.contourArea)
                # Generates the size and the cordinantes of the bounding box and draw
                x, y, w, h = cv2.boundingRect(max_contour)
                cv2.rectangle(crop2,(x,y), (x + w, y + h), (0, 255, 0), 1)
                cv2.circle(crop2, self.midpoint(x,y,w,h), 2, (63, 127, 0), -1)
                # Calculate the pixel distance from the middle of the frame
                pixel_distance = math.sqrt(math.pow((160 - self.midpoint(x,y,w,h)[1]),2))
                cv2.line(crop2, self.midpoint(x,y,w,h), (self.midpoint(x,y,w,h)[0], 240), (255,0,0), 1)
                pd_v_prop_stop = pixel_distance - self.stop_ofs
                if DEBUG_TEXT:
                    print("Stop line: {}".format(pixel_distance))
                if self.state!=self.State.STOPPING and pixel_distance<70:
                    self.state=self.State.STOPPING
                    self.pd_v=self.pd_stopping_v
                    self.pd_v.reset()
                    self.pd_omega.set_disable(0)
                    if DEBUG_TEXT:
                        self.log("stopping line detected")
                    # failsafe callback
                    cb_failsafe = rospy.Timer(rospy.Duration(3), self.cb_stopping_timer, oneshot=True)

                if abs(pd_v_prop_stop)<15 and self.stop_cb is not None:
                    self.cb=rospy.Timer(rospy.Duration(1), self.cb_stopping_timer, oneshot=True)
                self.pd_stopping_v.proportional=pd_v_prop_stop

        if self.state!=self.State.STOPPING:
            lower_blue = np.array([100, 150, 0])
            upper_blue = np.array([140, 255, 255])
            hsv_f = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            mask_f=cv2.inRange(hsv_f, lower_blue, upper_blue)
            contours, hierarchy = cv2.findContours(mask_f,
                                                   cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            max_contour=None
            area=0
            if len(contours) != 0:
                max_contour = max(contours, key=cv2.contourArea)
                M = cv2.moments(max_contour)
                cX = int(M["m10"] / M["m00"])
                # cY = int(M["m01"] / M["m00"])
                area = cv2.contourArea(max_contour)
                self.log("cx: {} area: {}".format(cX,area))

            if max_contour is not None and area>2500:
                if self.state != self.state.STOPPING:
                    self.pd_omega.proportional = cX - 400 + self.det_offset  # x value of the pattern
                    if self.state != self.State.TAILING:
                        # switch state and PD controllers
                        self.state = self.State.TAILING
                        self.pd_v_tail.set_disable(None)
                        self.pd_omega = self.pd_omega_tail
                        self.pd_omega.reset()
                        if DEBUG_TEXT:
                            self.log("start tailing")
                    if DEBUG_TEXT:
                        self.log("det: {}".format(self.det_distance))
                    self.pd_v.proportional = (area - 61454) // 61434

                elif self.state == self.State.STOPPING:
                    self.push_hist(cX - 400 + self.det_offset)

            else:
                self.det_distance = math.inf  # assume infinite distance when no detection
                if self.state == self.State.LF or self.state == self.State.STOPPING:
                    return
                if self.det_retry_counter < self.det_retry:
                    if DEBUG_TEXT:
                        self.log("lost det, retry")
                    self.det_retry_counter += 1
                    self.pd_v.reset()
                    self.pd_omega.reset()
                else:
                    # switch state and PD controllers
                    self.state = self.State.LF
                    self.pd_v_tail.set_disable(self.lf_velocity)  # constant speed
                    self.pd_omega = self.pd_omega_lf
                    self.pd_omega.reset()
                    self.det_retry_counter = 0
                    self.turn_hist.clear()
                    if DEBUG_TEXT:
                        self.log("end tailing")

        # Part for Lane Following Detection
        if self.state==self.State.TAILING:
            return

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
            rect_img_msg = CompressedImage(format="jpeg", data=self.jpeg.encode(crop2))
            self.pub.publish(rect_img_msg)

    def cb_led_timer(self,te):
        self.set_led("LIGHT_OFF")
        self.LED=None
        return

    def try_set_led(self,pattern):
        if self.LED is None:
            self.LED=pattern
            self.set_led(self.LED)
            cbt=rospy.Timer(rospy.Duration(2),self.cb_led_timer,oneshot=True)

    def push_hist(self,omega):
        if len(self.turn_hist)>self.len_turn_hist:
            self.turn_hist.pop(0)
        self.turn_hist.append(omega)

    def get_hist(self):
        if not self.turn_hist:
            return 0
        hist=sum(self.turn_hist)/len(self.turn_hist)
        self.turn_hist.clear()
        return hist

    def drive(self):
        if self.pd_omega.proportional is None:
            self.twist.omega = 0
        else:
            # get omega and v from current PD
            omega_n=min(self.pd_omega.get(), self.omega_cap)
            self.twist.v = min(max(self.pd_v.get(),0), self.v_cap)
            self.twist.omega = omega_n
            if self.state==self.State.STOPPING:
                # self.try_set_led("RED")
                self.log("STOP")
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
