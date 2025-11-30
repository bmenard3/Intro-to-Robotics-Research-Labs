#! /home/venv/.7785lab6/bin python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist, Point
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge
from ament_index_python.packages import get_package_share_directory
import os
import numpy as np
import math
import time
import torch
import cv2
from .models import SmallCNN

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, integral_limit=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_limit = integral_limit
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()
    
    def compute(self, error, current_time=None):
        if current_time is None:
            current_time = time.time()
        
        dt = current_time - self.prev_time
        if dt <= 0.0:
            dt = 0.01
        
        proportional = self.kp * error
        
        self.integral += error * dt
        self.integral = max(-self.integral_limit, min(self.integral_limit, self.integral))
        integral_term = self.ki * self.integral
        
        derivative = (error - self.prev_error) / dt
        derivative_term = self.kd * derivative
        
        output = proportional + integral_term + derivative_term
        
        self.prev_error = error
        self.prev_time = current_time
        
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        self.prev_time = time.time()


class MoveRobot(Node):
    def __init__(self):
        super().__init__('move_robot')
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth = 1
        )

        # ========== CNN ==========
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        package_share_directory = get_package_share_directory('team_rocket_final')
        model_path = os.path.join(package_share_directory, 'best_model_small.pth')
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            model_type = checkpoint.get('model_type', 'small')
            
            self.cnn_model = SmallCNN(num_classes=6)
            
            self.cnn_model.load_state_dict(checkpoint['model_state_dict'])
            self.cnn_model.to(self.device)
            self.cnn_model.eval()
            
            self.img_size = checkpoint.get('img_size', 64)
            self.class_names = ['empty', 'left', 'right', 'do_not_enter', 'stop', 'goal']
            
            self.get_logger().info(f'✓ Loaded {model_type.upper()} CNN (Acc: {checkpoint["val_acc"]:.2f}%)')
            self.get_logger().info(f'  Device: {self.device}')
            self.get_logger().info(f'  Image size: {self.img_size}x{self.img_size}')
        except Exception as e:
            self.get_logger().error(f'Failed to load CNN model: {e}')
            raise
        
        # CV Bridge
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_count = 0  # 用于调试
        

        self.range_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/ranges',
            self.range_callback,
            qos_profile
        )
        self.angle_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/angles',
            self.angle_callback,
            qos_profile
        )
        self.scan_subscriber = self.create_subscriber = self.create_subscription(
            Float32MultiArray,
            '/processed_scans',
            self.scan_callback,
            qos_profile
        )
        self.odom_subscriber = self.create_subscriber = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            qos_profile
        )
        self.vel_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        
        # ✅ 相机话题
        camera_topic = '/image_raw/compressed'
        
        self.camera_subscriber = self.create_subscriber = self.create_subscription(
            CompressedImage,
            camera_topic,
            self.camera_callback,
            10
        )
        
        self.get_logger().info(f'📷 Subscribing to camera topic: {camera_topic}')


        timer_period = 0.1
        self.timer = self.create_timer(timer_period, self.navigation)
        self.angular_pid = PIDController(
            kp=0.5,
            ki=0.0,
            kd=0.0,
            integral_limit=3.0
        )
        self.max_angular_speed = 1.0
        self.movement_start = False
        self.Init_pos = Point()
        self.Init_pos.x = 0.0
        self.Init_pos.y = 0.0
        self.Init_ang = 0.0
        self.globalPos = Point()
        self.globalAng = 0.0
        self.state = 1
        self.ranges = [0.5]
        self.angles = [math.pi]
        self.forward_distance = 1
        self.right_distance = 2
        self.left_distance = 2
        self.movement_goal = Point()
        self.movement_goal.x = 1.0
        self.movement_goal.y = 0.0
        
        # ========== 小角度转向设置 ==========
        self.small_turn_angle = math.pi / 4  # 45度，可以调整为 30-60度之间的值
        # 如果想要30度，使用: math.pi / 6
        # 如果想要60度，使用: math.pi / 3

    def camera_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            self.image_count += 1
            # 每10帧打印一次，避免刷屏
            if self.image_count % 10 == 0:
                self.get_logger().info(f'📷 Camera OK: received {self.image_count} images, shape={self.latest_image.shape}')
        except Exception as e:
            self.get_logger().error(f'❌ Image conversion failed: {e}')
    
    def preprocess_image(self, image):
        """预处理图像"""
        if image is None:
            self.get_logger().warn('⚠️ preprocess_image: input is None')
            return None
        
        try:
            # 调整大小
            image = cv2.resize(image, (self.img_size, self.img_size))
            
            # BGR转RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # 归一化到 [0, 1]
            image = image.astype(np.float32) / 255.0
            
            # 转换为 (C, H, W) 格式
            image = np.transpose(image, (2, 0, 1))
            
            # 转换为张量并添加batch维度
            tensor = torch.FloatTensor(image).unsqueeze(0)
            
            return tensor
        except Exception as e:
            self.get_logger().error(f'❌ preprocess_image error: {e}')
            return None
    
    def detect_sign(self):
        """
        检测标识并返回类别编号
        Returns:
            int or None: 0=empty, 1=left, 2=right, 3=do_not_enter, 4=stop, 5=goal
                         如果没有图像或检测到empty，返回None
        """
        # 1. 检查相机图像
        if self.latest_image is None:
            self.get_logger().error('❌ NO CAMERA IMAGE! Check if camera topic is publishing.')
            self.get_logger().error('   Run: ros2 topic list | grep camera')
            self.get_logger().error('   Run: ros2 topic hz /camera/image_raw')
            return None
        
        # 2. 预处理
        input_tensor = self.preprocess_image(self.latest_image)
        if input_tensor is None:
            self.get_logger().error('❌ Preprocessing failed!')
            return None
        
        # 3. 移动到设备
        input_tensor = input_tensor.to(self.device)
        
        # 4. 推理
        try:
            with torch.no_grad():
                outputs = self.cnn_model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                pred_class = predicted.item()
                conf_value = confidence.item()
        except Exception as e:
            self.get_logger().error(f'❌ CNN inference error: {e}')
            return None
        
        # 5. 打印详细结果
        class_name = self.class_names[pred_class]
        self.get_logger().info(f'🔍 CNN Detection:')
        self.get_logger().info(f'   Predicted Class: {pred_class} ({class_name})')
        self.get_logger().info(f'   Confidence: {conf_value:.3f}')
        
        # 打印所有类别的概率（用于调试）
        probs = probabilities[0].cpu().numpy()
        prob_str = ', '.join([f'{i}({self.class_names[i]}):{probs[i]:.2f}' for i in range(6)])
        self.get_logger().info(f'   All probabilities: [{prob_str}]')
        
        # 6. 返回结果
        if pred_class == 0:  # empty
            self.get_logger().info('   → Result: EMPTY (returning None)')
            return None
        
        self.get_logger().info(f'   → Result: {class_name.upper()} (returning {pred_class})')
        return pred_class  # 返回 1-5


    def odom_callback(self, msg):
        self.update_Odometry(msg)

    def range_callback(self, msg):
        self.ranges = msg.data
    
    def angle_callback(self, msg):
        angles = np.array(msg.data)
        self.angles = list((angles + np.pi) % (2 * np.pi) - np.pi)
    
    def scan_callback(self, msg):
        N = msg.layout.dim[0].size
        M = msg.layout.dim[1].size
        self.latest_scan = np.array(msg.data, dtype=np.float32).reshape(N, M)
        forward_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + (math.pi / 12)))
        forward_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - (math.pi / 12)))
        self.forward_scan = self.latest_scan[forward_scan_ind0:forward_scan_ind1, :]
        self.forward_distance = np.min(self.forward_scan[:,1])
        for i in range(self.latest_scan.shape[0]):
            if np.isnan(self.latest_scan[i,1]):
                self.latest_scan[i,1] = 5.0
        
        right_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) + (math.pi / 6))))
        right_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] + ((math.pi / 2) - (math.pi / 6))))
        self.right_scan = self.latest_scan[right_scan_ind0:right_scan_ind1, :]
        self.right_distance = np.min(self.right_scan[:,1])
        
        left_scan_ind0 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) - (math.pi / 6))))
        left_scan_ind1 = np.argmin(np.abs(self.latest_scan[:, 0] - ((math.pi / 2) + (math.pi / 6))))
        self.left_scan = self.latest_scan[left_scan_ind0:left_scan_ind1, :]
        self.left_distance = np.min(self.left_scan[:,1])


    def navigation(self):
        msg = Twist()
        closest_scan = self.ranges.index(min(self.ranges))
        closest_range = self.ranges[closest_scan]
        closest_angle = self.angles[closest_scan]


        if self.state == 0: # Detect
            self.get_logger().info('State 0: Detecting sign...')
            if (closest_range < 0.5) and (abs(closest_angle) < (math.pi/4)):
                detection = self.detect_sign()  # 返回 None 或 1-5
                
                if detection == 0 :  # 0: empty
                    self.get_logger().info('   No sign detected → State 1 (Forward)')
                    self.state = 1
                    self.movement_start = True
                elif detection == 2:  # right
                    self.get_logger().info('   RIGHT sign → State 2 (Turn Right 90°)')
                    self.state = 2
                    self.movement_start = True
                elif detection == 1:  # left
                    self.get_logger().info('   LEFT sign → State 3 (Turn Left 90°)')
                    self.state = 3
                    self.movement_start = True
                elif detection == 3 or detection == 4:  # do_not_enter or stop
                    self.get_logger().info('   STOP/DNE sign → State 4 (Turn Around)')
                    self.state = 4
                    self.movement_start = True
                elif detection == 5:  # goal
                    self.get_logger().info('   GOAL sign → State 6 (Goal Reached!)')
                    self.state = 6
            else:
                self.state = 1
                self.movement_start = True


        elif self.state == 1: # Move Forward
            if self.forward_distance < 0.55:
                self.get_logger().info('State 1: The way is blocked, stopping...')
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                
                # 尝试检测标识
                detection = self.detect_sign()  # 返回 None 或 1-5
                
                if detection == 0:  # empty (0)
                    self.get_logger().info(f'   No sign. Left={self.left_distance:.2f}m, Right={self.right_distance:.2f}m')
                    
                    # 使用小角度转向（State 7和8）而不是90度转向
                    if self.right_distance > self.left_distance:
                        self.get_logger().info(f'   → Right wider → State 7 (Small Right Turn)')
                        self.state = 7  # 小角度右转
                    else:
                        self.get_logger().info(f'   → Left wider → State 8 (Small Left Turn)')
                        self.state = 8  # 小角度左转
                        
                elif detection == 2:  # right
                    self.get_logger().info('   RIGHT sign → State 2 (Turn Right 90°)')
                    self.state = 2
                elif detection == 1:  # left
                    self.get_logger().info('   LEFT sign → State 3 (Turn Left 90°)')
                    self.state = 3
                elif detection == 3 or detection == 4:  # do_not_enter or stop
                    self.get_logger().info('   STOP/DNE sign → State 4 (Turn Around)')
                    self.state = 4
                elif detection == 5:  # goal
                    self.get_logger().info('   GOAL sign → State 6 (Goal Reached!)')
                    self.state = 6
                
                self.movement_start = True
            else:
                # 前进逻辑
                msg.linear.x = 0.1
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                angular_velocity = 0.0
                
                # 墙壁跟随
                if self.left_distance < 0.6:
                    closest_left_angle = self.left_scan[np.argmin(self.left_scan[:,1]), 0]
                    angular_velocity += 0.1*(closest_left_angle - (math.pi/2))
                if self.right_distance < 0.6:
                    closest_right_angle = self.right_scan[np.argmin(self.right_scan[:,1]), 0]
                    angular_velocity += 0.1*(-(math.pi/2) - closest_right_angle)
                
                if angular_velocity > 0.2:
                    angular_velocity = 0.2
                if angular_velocity < -0.2:
                    angular_velocity = -0.2
                msg.angular.z = angular_velocity

   
        elif self.state == 2: # Turn Right 90°
            target_angle = -1 * math.pi / 2  # -90度
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'State 2: Turning Right 90° (remaining: {math.degrees(d_theta):.1f}°)')
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn complete → State 5 (Pause)')
                self.state = 5
        
        elif self.state == 3: # Turn Left 90°
            target_angle = math.pi / 2  # 90度
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'State 3: Turning Left 90° (remaining: {math.degrees(d_theta):.1f}°)')
            
            if abs(d_theta) > 0.1:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn complete → State 5 (Pause)')
                self.state = 5
        
        elif self.state == 4: # Turn Around 180°
            target_angle = math.pi  # 180度
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'State 4: Turning Around 180° (remaining: {math.degrees(d_theta):.1f}°)')
            
            if abs(d_theta) > 0.1:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Turn around complete → State 5 (Pause)')
                self.state = 5
        
        elif self.state == 5: # Pause
            self.get_logger().info('State 5: Pausing...')
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            time.sleep(0.5)
            self.movement_start = True
            self.get_logger().info('   → State 1 (Forward)')
            self.state = 1
        
        elif self.state == 6: # Goal Reached
            msg.linear.x = 0.0
            msg.linear.y = 0.0
            msg.linear.z = 0.0
            msg.angular.x = 0.0
            msg.angular.y = 0.0
            msg.angular.z = 0.0
            self.get_logger().info('🎯 State 6: GOAL REACHED! Mission complete!')
        
        # ========== 新增：小角度转向状态 ==========
        elif self.state == 7: # Small Turn Right (30-60度)
            target_angle = -1 * self.small_turn_angle  # 负值表示右转
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'State 7: Small Right Turn {math.degrees(self.small_turn_angle):.0f}° (remaining: {math.degrees(d_theta):.1f}°)')
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Small turn complete → State 5 (Pause)')
                self.state = 5
        
        elif self.state == 8: # Small Turn Left (30-60度)
            target_angle = self.small_turn_angle  # 正值表示左转
            d_theta = target_angle - self.globalAng
            d_theta = math.atan2(math.sin(d_theta), math.cos(d_theta))
            angular_velocity = self.angular_pid.compute(d_theta, time.time())
            self.get_logger().info(f'State 8: Small Left Turn {math.degrees(self.small_turn_angle):.0f}° (remaining: {math.degrees(d_theta):.1f}°)')
            
            if abs(d_theta) > 0.05:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                if angular_velocity >= 0:
                    msg.angular.z = min(angular_velocity, self.max_angular_speed)
                else:
                    msg.angular.z = max(angular_velocity, -self.max_angular_speed)
            else:
                msg.linear.x = 0.0
                msg.linear.y = 0.0
                msg.linear.z = 0.0
                msg.angular.x = 0.0
                msg.angular.y = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('   Small turn complete → State 5 (Pause)')
                self.state = 5
        
        self.vel_publisher.publish(msg)
        

            
    def update_Odometry(self, Odom):
        position = Odom.pose.pose.position
        
        #Orientation uses the quaternion aprametrization.
        #To get the angular position along the z-axis, the following equation is required.
        q = Odom.pose.pose.orientation
        orientation = np.arctan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))

        if self.movement_start:
            #The initial data is stored to by subtracted to all the other values as we want to start at position (0,0) and orientation 0
            self.movement_start = False
            self.Init_ang = orientation
            self.globalAng = self.Init_ang
            Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        
            self.Init_pos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y
            self.Init_pos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y
            self.Init_pos.z = position.z
        Mrot = np.matrix([[np.cos(self.Init_ang), np.sin(self.Init_ang)],[-np.sin(self.Init_ang), np.cos(self.Init_ang)]])        

        #We subtract the initial values
        self.globalPos.x = Mrot.item((0,0))*position.x + Mrot.item((0,1))*position.y - self.Init_pos.x
        self.globalPos.y = Mrot.item((1,0))*position.x + Mrot.item((1,1))*position.y - self.Init_pos.y
        globalAng = orientation - self.Init_ang
        self.globalAng = math.atan2(math.sin(globalAng), math.cos(globalAng))


def main(args=None):
    rclpy.init(args=args)
    move_robot = MoveRobot()
    rclpy.spin(move_robot)

    move_robot.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
