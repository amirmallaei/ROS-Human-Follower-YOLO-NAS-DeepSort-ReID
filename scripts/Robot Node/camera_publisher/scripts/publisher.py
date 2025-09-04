
#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
#from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage

from cv_bridge import CvBridge, CvBridgeError


# Node to obtain call camera data. Separate I/O pipeline
rospy.loginfo('Init Cameras...')
cam_front = cv2.VideoCapture(0)
cam_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam_front.set(cv2.CAP_PROP_FOURCC, int(0x47504A4D))


def imagePublisher():
    #front_pub = rospy.Publisher('/center_cam', Image, queue_size=1)
    front_pub = rospy.Publisher('/center_cam', CompressedImage, queue_size=1)
    #print(front_pub)
    rospy.init_node('camera_publisherr', anonymous=True)
    rate = rospy.Rate(5)  # Adjust the publishing rate as needed
    #rate=rospy.Rate(30)#10hz
    bridge = CvBridge()

    while not rospy.is_shutdown():
        _, front_img = cam_front.read()
        # print(front_img)
        # cv2.imshow('image', front_img)        # for debugging purposes
        # front_img = bridge.cv2_to_imgmsg(front_img, "bgr8")
        front_img = bridge.cv2_to_compressed_imgmsg(front_img)

        # rospy.loginfo("images sent")
        # for debugging purposes, remove if cv2.imshow('imgae', img) is deleted
        #print(front_img)
        k = cv2.waitKey(1) & 0xFF
        if k ==27:
            break
        # print(front_pub.publish(front_img))
        front_pub.publish(front_img)
        #rate.sleep()

    cv2.destroyAllWindows()
    cam_front.release()


if __name__ == '__main__':
    try:
        imagePublisher()
        # print('sending image')
    except rospy.ROSInterruptException:
        pass
