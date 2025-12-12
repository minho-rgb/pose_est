import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray
import DR_init
import threading
import time
import math
import numpy as np

# [설정]
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
LIMIT_J3 = 10.0
LIMIT_J5 = 10.0

# 로봇 초기화
DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class ControlNode(Node):
    def __init__(self):
        super().__init__('control_node')
        
        # 구독
        self.create_subscription(JointState, f'/{ROBOT_ID}/joint_states', self.joint_callback, 10)
        self.create_subscription(Float32MultiArray, f'/{ROBOT_ID}/robot_cmd', self.cmd_callback, 10)
        
        self.real_joint = None
        self.target_step = [0.0]*6
        self.lock = threading.Lock()
        
        # 로봇 모션 스레드 실행
        self.motion_thread = threading.Thread(target=self.motion_loop)
        self.motion_thread.daemon = True
        self.motion_thread.start()
        
        print(">>> Control Node Ready (Waiting for commands...)")

    def joint_callback(self, msg):
        if len(msg.position) >= 6:
            deg = [math.degrees(rad) for rad in msg.position[:6]]
            if sum(abs(v) for v in deg) > 0.1:
                self.real_joint = deg

    def cmd_callback(self, msg):
        # Vision 노드에서 온 데이터 수신
        # msg.data layout: [x, y, z, rx, ry, rz, mode]
        if len(msg.data) >= 6:
            with self.lock:
                self.target_step = list(msg.data[:6])

    def motion_loop(self):
        # DSR 라이브러리는 메인 스레드나 별도 스레드에서 호출 필요
        from DSR_ROBOT2 import amovel, DR_TOOL, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE, set_tool, set_tcp
        
        try:
            set_tool("ToolWeight")
            set_tcp("GripperDA_mh")
        except:
            pass

        while rclpy.ok():
            with self.lock:
                step = list(self.target_step) # 복사

            # 안전 로직 (Singularity Check)
            if self.real_joint:
                j3 = self.real_joint[2]
                j5 = self.real_joint[4]
                
                if abs(j3) < LIMIT_J3: # Elbow Singularity
                    print(f"[WARN] J3 Limit! {j3:.1f}")
                    if step[2] > 0: step[2] = 0 # Z축 전진 차단
                
                if abs(j5) < LIMIT_J5: # Wrist Singularity
                    print(f"[WARN] J5 Limit! {j5:.1f}")
                    step[4] = 0 # Ry 회전 차단
                    step[0] *= 0.1 # X축 감속

            # 명령 실행 (값이 있을 때만)
            if sum(abs(v) for v in step) > 0.001:
                try:
                    # 0.5초 동안 부드럽게 이동
                    amovel(step, time=0.8, mod=DR_MV_MOD_REL, ref=DR_TOOL, ra=DR_MV_RA_OVERRIDE)
                except Exception as e:
                    print(f"Motion Error: {e}")
            
            # 제어 주기 (Vision Node의 FPS보다 약간 느리거나 같게)
            time.sleep(0.05) 

def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()
    DR_init.__dsr__node = node # DSR 초기화에 필수
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()