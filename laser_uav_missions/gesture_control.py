#!/usr/bin/env python3
import time
import cv2
import mediapipe as mp
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data 
from std_srvs.srv import Trigger
from laser_msgs.msg import PoseWithHeading 
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

# === MEDIA PIPE SHORTCUTS ===
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
RunningMode = mp.tasks.vision.RunningMode
Image = mp.Image

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        
        # --- Configurações ---
        self.uav_name = os.getenv("UAV_NAME", "uav1")
        self.declare_parameter('takeoff_duration', 3.0)
        self.declare_parameter('land_duration', 3.0)
        self.declare_parameter('control_rate', 0.10) 
        
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.land_duration = self.get_parameter('land_duration').value
        self.control_step = self.get_parameter('control_rate').value

        # Estado do ALVO 
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 1.5 
        self.target_heading = 0.0
        
        self.current_real_x = 0.0
        self.current_real_y = 0.0
        self.current_real_z = 0.0 
        
        self.odom_received = False 

        # Services
        self.takeoff_client = self.create_client(Trigger, f'/{self.uav_name}/control_manager/takeoff')
        self.land_client = self.create_client(Trigger, f'/{self.uav_name}/control_manager/land')

        # Publisher
        topic_name = f'/{self.uav_name}/control_manager/goto'
        self.goto_publisher = self.create_publisher(PoseWithHeading, topic_name, 10)
        
        # Subscriber (Odometria) 
        self.odom_sub = self.create_subscription(
            Odometry,
            f'/{self.uav_name}/estimation_manager/odom_main',
            self.odom_callback,
            qos_profile_sensor_data 
        )

        self.get_logger().info(f"Nó de Gestos Iniciado. Controle: {self.control_step}m/gesto")

    def odom_callback(self, msg):
        self.current_real_x = msg.pose.pose.position.x
        self.current_real_y = msg.pose.pose.position.y
        self.current_real_z = msg.pose.pose.position.z
        
        if not self.odom_received:
            self.target_x = self.current_real_x
            self.target_y = self.current_real_y
            if self.current_real_z > 0.5:
                self.target_z = self.current_real_z
            self.odom_received = True

    def call_takeoff(self):
        if self.takeoff_client.service_is_ready():
            req = Trigger.Request()
            future = self.takeoff_client.call_async(req)
            future.add_done_callback(self.response_callback)
            return True
        return False

    def call_land(self):
        if self.land_client.service_is_ready():
            req = Trigger.Request()
            future = self.land_client.call_async(req)
            future.add_done_callback(self.response_callback)
            return True
        return False

    def publish_position(self, direction_msg):
        msg = PoseWithHeading()
        msg.position.x = self.target_x
        msg.position.y = self.target_y
        msg.position.z = self.target_z
        msg.heading = self.target_heading
        
        self.goto_publisher.publish(msg)
        self.get_logger().info(f"CMD {direction_msg}: Target=[{self.target_x:.1f}, {self.target_y:.1f}, {self.target_z:.1f}]")

    def response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info(f"Comando executado: {response.message}")
            else:
                self.get_logger().warn(f"Falha no comando: {response.message}")
        except Exception as e:
            self.get_logger().error(f"Falha na chamada: {e}")

_latest_gesture_name = None
_latest_landmarks_norm = None

def print_result(result, output_image, timestamp_ms: int):
    global _latest_gesture_name, _latest_landmarks_norm
    if result and result.gestures:
        gesture = result.gestures[0][0]
        _latest_gesture_name = gesture.category_name 
    else:
        _latest_gesture_name = None

    if result and getattr(result, 'hand_landmarks', None):
        hand0 = result.hand_landmarks[0]
        _latest_landmarks_norm = [(lm.x, lm.y) for lm in hand0]
    else:
        _latest_landmarks_norm = None

def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()

    try:
        pkg_share = get_package_share_directory('laser_uav_missions')
        model_path = os.path.join(pkg_share, 'models', 'gesture_recognizer.task')
    except Exception as e:
        node.get_logger().error(f"Erro ao localizar share: {e}")
        # Fallback para o diretório local de desenvolvimento
        model_path = os.path.join(os.getcwd(), 'models', 'gesture_recognizer.task')

    if not os.path.exists(model_path):
        node.get_logger().error(f"Modelo não encontrado: {model_path}")
        return

    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.LIVE_STREAM, 
        result_callback=print_result, 
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return


    takeoff_start_time = None
    takeoff_triggered = False
    land_start_time = None
    land_triggered = False
    

    last_cmd_time = 0.0
    CMD_COOLDOWN = 0.5 

    HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS

    with GestureRecognizer.create_from_options(options) as recognizer:
        try:
            while rclpy.ok():
                rclpy.spin_once(node, timeout_sec=0)

                ret, frame = cap.read()
                if not ret: break
                
                frame = cv2.resize(frame, (640, 480))
                frame_flipped = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                recognizer.recognize_async(mp_image, int(time.time() * 1000))

                current_gesture = _latest_gesture_name
                current_time = time.time()

                # --- LÓGICA DE CONTROLE ---

                # 1. DECOLAR
                if current_gesture == "Open_Palm":
                    land_start_time = None; land_triggered = False
                    
                    if takeoff_start_time is None: takeoff_start_time = time.time()
                    elapsed = time.time() - takeoff_start_time
                    
                    if elapsed < node.takeoff_duration:
                        # Timer de decolagem
                        cv2.putText(frame_flipped, f"DECOLAR EM: {node.takeoff_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    elif not takeoff_triggered:
                        node.target_z = 1.5
                        node.target_x = node.current_real_x 
                        node.target_y = node.current_real_y 
                        if node.call_takeoff(): takeoff_triggered = True
                    else:
                        cv2.putText(frame_flipped, "DECOLANDO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # 2. POUSAR
                elif current_gesture == "Closed_Fist":
                    takeoff_start_time = None; takeoff_triggered = False
                    
                    if land_start_time is None: land_start_time = time.time()
                    elapsed = time.time() - land_start_time
                    
                    if elapsed < node.land_duration:
                        # Timer de pouso
                        cv2.putText(frame_flipped, f"POUSAR EM: {node.land_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    elif not land_triggered:
                        if node.call_land(): land_triggered = True
                    else:
                        cv2.putText(frame_flipped, "POUSANDO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # --- MOVIMENTAÇÃO ---
                
                # 3. SUBIR
                elif current_gesture == "Thumb_Up":
                    takeoff_start_time = None; land_start_time = None
                    
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.target_z += node.control_step
                        if node.target_z > 15.0: node.target_z = 15.0
                        node.publish_position("SUBIR")
                        last_cmd_time = current_time

                    cv2.putText(frame_flipped, "SUBINDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

                # 4. DESCER
                elif current_gesture == "Thumb_Down":
                    takeoff_start_time = None; land_start_time = None
                    
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.target_z -= node.control_step
                        if node.target_z < 0.5: node.target_z = 0.5
                        node.publish_position("DESCER")
                        last_cmd_time = current_time

                    cv2.putText(frame_flipped, "DESCENDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                # 5. DIREITA
                elif current_gesture == "Pointing_Up":
                    takeoff_start_time = None; land_start_time = None
                    
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.target_y -= node.control_step
                        if node.target_y < -6.0: node.target_y = -6.0
                        node.publish_position("DIREITA")
                        last_cmd_time = current_time

                    cv2.putText(frame_flipped, "DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

                # 6. ESQUERDA
                elif current_gesture == "Victory":
                    takeoff_start_time = None; land_start_time = None
                    
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.target_y += node.control_step
                        if node.target_y > 0.0: node.target_y = 0.0
                        node.publish_position("ESQUERDA")
                        last_cmd_time = current_time

                    cv2.putText(frame_flipped, "ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

                # 7. FRENTE
                elif current_gesture == "ILoveYou":
                    takeoff_start_time = None; land_start_time = None
                    
                    if (current_time - last_cmd_time) > CMD_COOLDOWN:
                        node.target_x += node.control_step
                        if node.target_x > 6.0: node.target_x = 6.0
                        node.publish_position("FRENTE")
                        last_cmd_time = current_time

                    cv2.putText(frame_flipped, "FRENTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

                else:
                    takeoff_start_time = None
                    land_start_time = None


                if _latest_landmarks_norm:
                    h, w = frame_flipped.shape[:2]
                    pts = [(int((1.0 - x) * w), int(y * h)) for (x, y) in _latest_landmarks_norm]
                    for a, b in HAND_CONNECTIONS:
                        if a < len(pts) and b < len(pts): cv2.line(frame_flipped, pts[a], pts[b], (0, 255, 0), 2)
                    for pt in pts: cv2.circle(frame_flipped, pt, 5, (0, 0, 255), -1)

                cv2.imshow('Gesture Control ROS2', frame_flipped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
