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
        self.declare_parameter('yaw_gain_p', 0.001) # Ganho Proporcional do Yaw (ajuste se ficar muito rápido/lento)
        
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.land_duration = self.get_parameter('land_duration').value
        self.control_step = self.get_parameter('control_rate').value
        self.yaw_gain_p = self.get_parameter('yaw_gain_p').value

        # Variáveis auxiliares de estado absoluto
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 1.5 
        self.target_heading = 0.0 
        
        self.current_real_x = 0.0
        self.current_real_y = 0.0
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

        # Publisher
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

        self.get_logger().info(f"Nó Iniciado. Auto-Yaw ativado com ganho P: {self.yaw_gain_p}")

    def diag_callback(self, msg):
        self.is_fly = msg.is_fly
        self.have_goal = msg.have_goal

    def odom_callback(self, msg):
        self.current_real_x = msg.pose.pose.position.x
        self.current_real_y = msg.pose.pose.position.y
        self.current_real_z = msg.pose.pose.position.z
        
        o = msg.pose.pose.orientation
        t3 = +2.0 * (o.w * o.z + o.x * o.y)
        t4 = +1.0 - 2.0 * (o.y * o.y + o.z * o.z)
        self.current_real_yaw = math.atan2(t3, t4)

        if not self.odom_received:
            self.odom_received = True


        if not self.is_fly:
            self.target_x = self.current_real_x
            self.target_y = self.current_real_y
            self.target_heading = self.current_real_yaw
            if self.current_real_z > 0.5:
                self.target_z = self.current_real_z

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

    def publish_position(self, direction_msg="GOTO"):
        msg = PoseWithHeading()
        msg.position.x = self.target_x
        msg.position.y = self.target_y
        msg.position.z = self.target_z
        msg.heading = self.target_heading
        self.goto_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()

    # --- 1. CARREGAR MODELO DE GESTOS ---
    try:
        pkg_share = get_package_share_directory('laser_uav_missions') 
        model_path = os.path.join(pkg_share, 'models', 'model.pt') 
    except Exception as e:
        node.get_logger().error(f"Erro ao localizar share: {e}")
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', 'model.pt')

    if not os.path.exists(model_path):
        node.get_logger().error(f"Modelo de gestos não encontrado!")
        return

    node.get_logger().info("Carregando modelos YOLO (Gestos e Pose)...")
    gesture_model = YOLO(model_path)
    pose_model = YOLO('yolov8n-pose.pt')
    node.get_logger().info("Modelos YOLO carregados com sucesso!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        node.get_logger().error("Falha ao abrir a webcam!")
        return

    # Pegar o centro da imagem para o cálculo do Yaw
    ret, frame_teste = cap.read()
    frame_teste = cv2.resize(frame_teste, (640, 480))
    center_x_image = frame_teste.shape[1] / 2

    takeoff_start_time = None
    takeoff_triggered = False
    land_start_time = None
    land_triggered = False
    
    last_cmd_time = 0.0
    last_yaw_time = 0.0
    CMD_COOLDOWN = 0.8 
    YAW_COOLDOWN = 0.1 # Atualiza o Yaw a 10Hz para ser mais suave

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)

            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            frame_flipped = cv2.flip(frame, 1)
            current_time = time.time()

            # Roda gestos
            gesture_results = gesture_model(frame_flipped, verbose=False)
            # Roda pose humana
            pose_results = pose_model(frame_flipped, verbose=False)

            current_gesture = None
            highest_conf = 0.0

            # Extrai o gesto
            for result in gesture_results:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf >= 0.70 and conf > highest_conf:
                        highest_conf = conf
                        class_id = int(box.cls.item())
                        current_gesture = gesture_model.names[class_id]

            annotated_frame = gesture_results[0].plot() if len(gesture_results) > 0 else frame_flipped.copy()


            nose_x, nose_y = None, None
            for r in pose_results:
                if r.keypoints is not None and len(r.keypoints.xy) > 0:
                    kpts = r.keypoints.xy[0] 
                    if len(kpts) > 0:
                        nose_x, nose_y = float(kpts[0][0]), float(kpts[0][1])
                        
                        # Desenha na mesma imagem anotada
                        if nose_x > 0 and nose_y > 0:
                            cv2.circle(annotated_frame, (int(nose_x), int(nose_y)), 8, (0, 255, 0), -1)
                            cv2.line(annotated_frame, (int(center_x_image), int(nose_y)), (int(nose_x), int(nose_y)), (255, 0, 0), 2)
                            cv2.putText(annotated_frame, "Rastreando Face", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            if nose_x is not None and nose_x > 0 and node.is_fly:
                if (current_time - last_yaw_time) > YAW_COOLDOWN:
                    error_x = center_x_image - nose_x 
                    
                    yaw_adjustment = error_x * node.yaw_gain_p
                    yaw_adjustment = max(-0.15, min(0.15, yaw_adjustment)) 
                    
                    node.target_heading += yaw_adjustment
                    node.publish_position("AUTO_YAW")
                    last_yaw_time = current_time

            # --- 4. LÓGICA DE GESTOS E RTL ---
            if node.rtl_state > 0:
                # Máquina de estados do RTL (igual estava antes)
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
                        node.target_x, node.target_y, node.target_z, node.target_heading = 0.0, 0.0, 1.5, 0.0
                        node.publish_position("RETORNO_HOME")
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
                    node.target_z = 1.5
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
                    node.target_z = min(15.0, node.target_z + node.control_step)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "SUBINDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'rock':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_z = max(0.5, node.target_z - node.control_step)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DESCENDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'two_up':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x += node.control_step * math.sin(node.target_heading)
                    node.target_y -= node.control_step * math.cos(node.target_heading)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'two_up_inverted':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x -= node.control_step * math.sin(node.target_heading)
                    node.target_y += node.control_step * math.cos(node.target_heading)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'xsign':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x += node.control_step * math.cos(node.target_heading)
                    node.target_y += node.control_step * math.sin(node.target_heading)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "FRENTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            elif current_gesture == 'palm':
                takeoff_start_time = None; land_start_time = None
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x -= node.control_step * math.cos(node.target_heading)
                    node.target_y -= node.control_step * math.sin(node.target_heading)
                    node.publish_position()
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "TRAS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

            else:
                takeoff_start_time = None
                land_start_time = None

            cv2.putText(annotated_frame, f"Bases: {node.landing_count}/6", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Exibe a única janela compilando tudo
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