#!/usr/bin/env python3
import time
import cv2
import os
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_srvs.srv import Trigger
from laser_msgs.msg import PoseWithHeading, UavControlDiagnostics
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        
        self.uav_name = os.getenv("UAV_NAME", "uav1")
        self.declare_parameter('takeoff_duration', 3.0)
        self.declare_parameter('land_duration', 3.0)
        self.declare_parameter('control_rate', 0.20)

        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.land_duration = self.get_parameter('land_duration').value
        self.control_step = self.get_parameter('control_rate').value

        self.current_real_z = 0.0
        self.current_real_yaw = 0.0
        self.odom_received = False
        self.landing_count = 0
        
        self.rtl_state = 0
        self.is_fly = False
        self.have_goal = False
        self.rtl_timer = 0.0

        # Services
        self.takeoff_client = self.create_client(Trigger, f'/{self.uav_name}/control_manager/takeoff')
        self.land_client = self.create_client(Trigger, f'/{self.uav_name}/control_manager/land')

        # Publishers
        self.goto_relative_publisher = self.create_publisher(PoseWithHeading, f'/{self.uav_name}/control_manager/goto_relative', 10)
        self.goto_publisher = self.create_publisher(PoseWithHeading, f'/{self.uav_name}/control_manager/goto', 10)
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.uav_name}/estimation_manager/odom_main',
            self.odom_callback,
            qos_profile_sensor_data
        )
        
        self.diag_sub = self.create_subscription(
            UavControlDiagnostics,
            f'/{self.uav_name}/control_manager/diagnostics',
            self.diag_callback,
            10
        )

        self.get_logger().info("Nó Iniciado. Gestos com GOTO_RELATIVE | RTL com GOTO global")

    def diag_callback(self, msg):
        self.is_fly = msg.is_fly
        self.have_goal = msg.have_goal

    def odom_callback(self, msg):
        self.current_real_z = msg.pose.pose.position.z
        
        o = msg.pose.pose.orientation
        t3 = +2.0 * (o.w * o.z + o.x * o.y)
        t4 = +1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        self.current_real_yaw = math.atan2(t3, t4)
        self.odom_received = True

    def call_takeoff(self):
        if self.takeoff_client.service_is_ready():
            self.takeoff_client.call_async(Trigger.Request())
            return True
        return False

    def call_land(self):
        if self.land_client.service_is_ready():
            self.land_client.call_async(Trigger.Request())
            return True
        return False

    def publish_relative(self, x=0.0, y=0.0, z=0.0, direction_msg="GOTO_RELATIVE"):
        self.get_logger().info(f"Cmd: {direction_msg} | dX={x:.2f}, dY={y:.2f}, dZ={z:.2f}")
        msg = PoseWithHeading()
        msg.position.x = float(x)
        msg.position.y = float(y)
        msg.position.z = float(z)
        msg.heading = 0.0
        self.goto_relative_publisher.publish(msg)

    def publish_global(self, direction_msg="GOTO"):
        self.get_logger().info(f"Cmd: {direction_msg} | X=0.00, Y=0.00, Z=1.50, Yaw=0.00")
        msg = PoseWithHeading()
        msg.position.x = 0.0
        msg.position.y = 0.0
        msg.position.z = 1.5
        msg.heading = 0.0
        self.goto_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()

    try:
        pkg_share = get_package_share_directory('laser_uav_missions')
        model_path = os.path.join(pkg_share, 'models', 'model.pt')
    except Exception as e:
        node.get_logger().error(f"Erro ao localizar share: {e}")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', 'model.pt')

    if not os.path.exists(model_path):
        node.get_logger().error("Modelo de gestos não encontrado!")
        return

    node.get_logger().info("Carregando modelo YOLO de gestos...")
    gesture_model = YOLO(model_path)
    node.get_logger().info("Modelo YOLO carregado com sucesso!")

    cap = cv2.VideoCapture(3)
    if not cap.isOpened():
        node.get_logger().error("Falha ao abrir a webcam!")
        return

    takeoff_start_time = None
    takeoff_triggered = False
    land_start_time = None
    land_triggered = False
    
    last_cmd_time = 0.0
    CMD_COOLDOWN = 0.5 # 2 Hz

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)

            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            frame_flipped = cv2.flip(frame, 1)
            current_time = time.time()

            # Inferência de gestos
            gesture_results = gesture_model(frame_flipped, verbose=False)

            current_gesture = None
            highest_conf = 0.0

            for result in gesture_results:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf >= 0.70 and conf > highest_conf:
                        highest_conf = conf
                        class_id = int(box.cls.item())
                        current_gesture = gesture_model.names[class_id]

            annotated_frame = gesture_results[0].plot() if len(gesture_results) > 0 else frame_flipped.copy()

            # --- LÓGICA DE RTL ---
            if node.rtl_state > 0:
                if node.rtl_state == 1:
                    cv2.putText(annotated_frame, "RTL: AGUARDANDO POUSO...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    if not node.is_fly:
                        if node.call_takeoff():
                            node.rtl_timer = current_time; node.rtl_state = 2

                elif node.rtl_state == 2:
                    cv2.putText(annotated_frame, "RTL: DECOLANDO...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    if (current_time - node.rtl_timer) > 5.0:
                        node.rtl_state = 3; last_cmd_time = 0.0

                elif node.rtl_state == 3:
                    cv2.putText(annotated_frame, "RTL: INDO PARA HOME...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.publish_global("RETORNO_HOME")
                        last_cmd_time = current_time
                    if node.have_goal:
                        node.rtl_state = 4; node.rtl_timer = current_time

                elif node.rtl_state == 4:
                    cv2.putText(annotated_frame, "RTL: NAVEGANDO...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    if not node.have_goal and (current_time - node.rtl_timer) > 4.0:
                        if node.call_land(): node.rtl_state = 5

                elif node.rtl_state == 5:
                    cv2.putText(annotated_frame, "RTL: POUSANDO...", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    if not node.is_fly: node.rtl_state = 6

                elif node.rtl_state == 6:
                    cv2.putText(annotated_frame, "MISSAO CONCLUIDA!", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

                cv2.imshow('Gesture Control ROS 2 - YOLOv8', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # Comandos de Gestos
            if current_gesture == 'one':
                land_start_time = None; land_triggered = False
                if takeoff_start_time is None: takeoff_start_time = time.time()
                elapsed = time.time() - takeoff_start_time
                if elapsed < node.takeoff_duration:
                    cv2.putText(annotated_frame, f"DECOLAR EM: {node.takeoff_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                elif not takeoff_triggered:
                    if node.call_takeoff(): takeoff_triggered = True
                else:
                    cv2.putText(annotated_frame, "DECOLANDO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'dislike':
                takeoff_start_time = None; takeoff_triggered = False
                if land_start_time is None: land_start_time = time.time()
                elapsed = time.time() - land_start_time
                if elapsed < node.land_duration:
                    cv2.putText(annotated_frame, f"POUSAR EM: {node.land_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
                elif not land_triggered:
                    if node.call_land():
                        land_triggered = True
                        node.landing_count += 1
                        if node.landing_count >= 6:
                            node.rtl_state = 1
                else:
                    cv2.putText(annotated_frame, f"POUSANDO... (Bases: {node.landing_count}/6)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            
            elif current_gesture == 'ok':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    if not node.odom_received or node.current_real_z < 15.0:
                        node.publish_relative(z=node.control_step, direction_msg="SUBIR")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "SUBINDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'rock':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    if not node.odom_received or node.current_real_z > 0.5:
                        node.publish_relative(z=-node.control_step, direction_msg="DESCER")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DESCENDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'two_up':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    x = node.control_step * math.sin(node.current_real_yaw)
                    y = -node.control_step * math.cos(node.current_real_yaw)
                    node.publish_relative(x, y, direction_msg="DIREITA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'two_up_inverted':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    x = -node.control_step * math.sin(node.current_real_yaw)
                    y = node.control_step * math.cos(node.current_real_yaw)
                    node.publish_relative(x, y, direction_msg="ESQUERDA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'xsign':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    x = node.control_step * math.cos(node.current_real_yaw)
                    y = node.control_step * math.sin(node.current_real_yaw)
                    node.publish_relative(x, y, direction_msg="FRENTE")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "FRENTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'palm':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    x = -node.control_step * math.cos(node.current_real_yaw)
                    y = -node.control_step * math.sin(node.current_real_yaw)
                    node.publish_relative(x, y, direction_msg="TRAS")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "TRAS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            else:
                takeoff_start_time = None
                land_start_time = None

            cv2.putText(annotated_frame, f"Bases: {node.landing_count}/6", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            cv2.imshow('Gesture Control ROS 2 - YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
