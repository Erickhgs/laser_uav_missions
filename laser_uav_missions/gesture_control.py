#!/usr/bin/env python3
import time
import cv2
import os
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data 
from std_srvs.srv import Trigger
from laser_msgs.msg import PoseWithHeading 
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory

# === IMPORT DO YOLO ===
from ultralytics import YOLO

class GestureControlNode(Node):
    def __init__(self):
        super().__init__('gesture_control_node')
        
        # --- Configurações ---
        self.uav_name = os.getenv("UAV_NAME", "uav1")
        self.declare_parameter('takeoff_duration', 3.0)
        self.declare_parameter('land_duration', 3.0)
        self.declare_parameter('control_rate', 0.10) 
        self.declare_parameter('heading_rate', 0.20) # Novo: taxa de giro (yaw)
        
        self.takeoff_duration = self.get_parameter('takeoff_duration').value
        self.land_duration = self.get_parameter('land_duration').value
        self.control_step = self.get_parameter('control_rate').value
        self.heading_step = self.get_parameter('heading_rate').value

        # Estado do ALVO 
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_z = 1.5 
        self.target_heading = 0.0
        
        self.current_real_x = 0.0
        self.current_real_y = 0.0
        self.current_real_z = 0.0 
        
        # --- Variáveis da Missão (Home e Bases) ---
        self.odom_received = False 
        self.home_x = 0.0
        self.home_y = 0.0
        self.landing_count = 0
        self.returning_home = False

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

        self.get_logger().info(f"Nó Iniciado. Controle: {self.control_step}m | Yaw: {self.heading_step}rad")

    def odom_callback(self, msg):
        self.current_real_x = msg.pose.pose.position.x
        self.current_real_y = msg.pose.pose.position.y
        self.current_real_z = msg.pose.pose.position.z
        
        if not self.odom_received:
            self.target_x = self.current_real_x
            self.target_y = self.current_real_y
            # Grava a posição inicial como HOME
            self.home_x = self.current_real_x
            self.home_y = self.current_real_y
            
            if self.current_real_z > 0.5:
                self.target_z = self.current_real_z
            self.odom_received = True

    def call_takeoff(self):
        if self.takeoff_client.service_is_ready():
            req = Trigger.Request()
            future = self.takeoff_client.call_async(req)
            return True
        return False

    def call_land(self):
        if self.land_client.service_is_ready():
            req = Trigger.Request()
            future = self.land_client.call_async(req)
            return True
        return False

    def publish_position(self, direction_msg):
        msg = PoseWithHeading()
        msg.position.x = self.target_x
        msg.position.y = self.target_y
        msg.position.z = self.target_z
        msg.heading = self.target_heading
        
        self.goto_publisher.publish(msg)
        self.get_logger().info(f"CMD {direction_msg}: Target=[X:{self.target_x:.1f}, Y:{self.target_y:.1f}, Z:{self.target_z:.1f}, Yaw:{self.target_heading:.1f}]")


def main(args=None):
    rclpy.init(args=args)
    node = GestureControlNode()

    try:
        pkg_share = get_package_share_directory('laser_uav_missions') 
        model_path = os.path.join(pkg_share, 'models', 'model.pt') 
    except Exception as e:
        node.get_logger().error(f"Erro ao localizar share: {e}")
        # Ajuste de segurança para rodar direto sem o share se necessário
        current_dir = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(current_dir, '..', '..', 'models', 'model.pt')

    if not os.path.exists(model_path):
        node.get_logger().error(f"Modelo não encontrado: {model_path}. Verifique o setup.py!")
        return

    node.get_logger().info("Carregando modelo YOLO...")
    model = YOLO(model_path)
    node.get_logger().info("YOLO carregado!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        node.get_logger().error("Falha ao abrir a webcam!")
        return

    takeoff_start_time = None
    takeoff_triggered = False
    land_start_time = None
    land_triggered = False
    
    last_cmd_time = 0.0
    CMD_COOLDOWN = 0.5 

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0)

            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.resize(frame, (640, 480))
            frame_flipped = cv2.flip(frame, 1)

            results = model(frame_flipped, verbose=False)
            current_gesture = None
            highest_conf = 0.0

            for result in results:
                for box in result.boxes:
                    conf = box.conf.item()
                    if conf >= 0.70 and conf > highest_conf:
                        highest_conf = conf
                        class_id = int(box.cls.item())
                        current_gesture = model.names[class_id]

            current_time = time.time()
            annotated_frame = results[0].plot() if len(results) > 0 else frame_flipped

            # --- LÓGICA DE MISSÃO (RTL - Return To Launch) ---
            if node.returning_home:
                cv2.putText(annotated_frame, "MODO RTL: RETORNANDO PARA HOME", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Calcula a distância 2D até a home
                dist_to_home = ((node.current_real_x - node.home_x)**2 + (node.current_real_y - node.home_y)**2)**0.5
                
                # Publica constantemente a posição da home para o drone ir até lá
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x = node.home_x
                    node.target_y = node.home_y
                    node.target_z = 1.5
                    node.publish_position("RETORNO_HOME")
                    last_cmd_time = current_time

                # Se chegou bem perto da home, pousa automaticamente
                if dist_to_home < 0.3 and not land_triggered:
                    cv2.putText(annotated_frame, "HOME ALCANCADA! POUSANDO...", (30, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    if node.call_land(): 
                        land_triggered = True
                        node.get_logger().info("Missão concluída! Pousando na Home.")
                
                # Mostra o frame e reinicia o loop (ignora os gestos manuais durante o retorno)
                cv2.imshow('Gesture Control ROS 2 - YOLOv8', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            # --- LÓGICA DE CONTROLE (GESTOS) ---

            # 1. DECOLAR
            if current_gesture == 'one':
                land_start_time = None; land_triggered = False
                if takeoff_start_time is None: takeoff_start_time = time.time()
                elapsed = time.time() - takeoff_start_time
                
                if elapsed < node.takeoff_duration:
                    cv2.putText(annotated_frame, f"DECOLAR EM: {node.takeoff_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                elif not takeoff_triggered:
                    node.target_z = 1.5
                    node.target_x = node.current_real_x 
                    node.target_y = node.current_real_y 
                    if node.call_takeoff(): takeoff_triggered = True
                else:
                    cv2.putText(annotated_frame, "DECOLANDO...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # 2. POUSAR / CONTAR BASE
            elif current_gesture == 'dislike':
                takeoff_start_time = None; takeoff_triggered = False
                if land_start_time is None: land_start_time = time.time()
                elapsed = time.time() - land_start_time
                
                if elapsed < node.land_duration:
                    cv2.putText(annotated_frame, f"POUSAR EM: {node.land_duration - elapsed:.1f}s", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                elif not land_triggered:
                    if node.call_land(): 
                        land_triggered = True
                        node.landing_count += 1
                        node.get_logger().info(f"Pouso {node.landing_count}/6 confirmado!")
                        
                        # Verifica se completou as 6 bases
                        if node.landing_count >= 6:
                            node.get_logger().info("6 bases visitadas! Iniciando retorno...")
                            node.returning_home = True
                            land_triggered = False # Reseta para o pouso final na home
                else:
                    cv2.putText(annotated_frame, f"POUSANDO... (Bases: {node.landing_count}/6)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            
            # 3. SUBIR
            elif current_gesture == 'ok':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_z = min(15.0, node.target_z + node.control_step)
                    node.publish_position("SUBIR")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "SUBINDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # 4. DESCER
            elif current_gesture == 'rock':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_z = max(0.5, node.target_z - node.control_step)
                    node.publish_position("DESCER")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DESCENDO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # 5. DIREITA
            elif current_gesture == 'peace':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_y = max(-6.0, node.target_y - node.control_step)
                    node.publish_position("DIREITA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            # 6. ESQUERDA
            elif current_gesture == 'peace_inverted':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_y = min(0.0, node.target_y + node.control_step)
                    node.publish_position("ESQUERDA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # 7. FRENTE
            elif current_gesture == 'fist':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x = min(6.0, node.target_x + node.control_step)
                    node.publish_position("FRENTE")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "FRENTE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # 8. TRÁS
            elif current_gesture == 'palm':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_x = max(-6.0, node.target_x - node.control_step)
                    node.publish_position("TRAS")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "TRAS", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)

            # 9. YAW ESQUERDA
            elif current_gesture == 'like':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_heading += node.heading_step
                    node.publish_position("YAW_ESQUERDA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "GIRAR ESQUERDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

            # 10. YAW DIREITA
            elif current_gesture == 'four':
                reset_timers(locals())
                if (current_time - last_cmd_time) > CMD_COOLDOWN:
                    node.target_heading -= node.heading_step
                    node.publish_position("YAW_DIREITA")
                    last_cmd_time = current_time
                cv2.putText(annotated_frame, "GIRAR DIREITA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)

            else:
                takeoff_start_time = None
                land_start_time = None

            # Mostra o status das bases
            cv2.putText(annotated_frame, f"Bases: {node.landing_count}/6", (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('Gesture Control ROS 2 - YOLOv8', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

def reset_timers(local_vars):
    local_vars['takeoff_start_time'] = None
    local_vars['land_start_time'] = None

if __name__ == '__main__':
    main()