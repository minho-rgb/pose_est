import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import JointState
import DR_init
import time
import math
import numpy as np

# ... (기존 설정 변수 유지) ...
ROBOT_ID = "dsr01"
ROBOT_MODEL = "m0609"
ROBOT_TCP = "GripperDA_mh"   
ROBOT_TOOL = "ToolWeight" 
LIMIT_J3_BUFFER = 10.0 
LIMIT_J5_BUFFER = 10.0 
LIMIT_RADIUS_MIN = 200.0 
MOVE_TIME = 0.8

# 디지털 출력 신호 상수 (필요시 수정)
ON = 1
OFF = 0

DR_init.__dsr__id = ROBOT_ID
DR_init.__dsr__model = ROBOT_MODEL

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('webcam_control_receiver', namespace=ROBOT_ID)
        self.create_subscription(Float32MultiArray, '/dsr01/robot_cmd', self.cmd_callback, 10)
        self.target_step = [0.0] * 6
        self.mode_id = 0
        self.new_cmd = False
        print(">>> Robot Control Receiver Started")

    def cmd_callback(self, msg):
        if len(msg.data) >= 7:
            self.target_step = list(msg.data[:6])
            self.mode_id = int(msg.data[6])
            self.new_cmd = True

def main(args=None):
    rclpy.init(args=args)
    node = RobotControlNode()
    DR_init.__dsr__node = node

    # [수정] set_digital_output 추가 임포트
    from DSR_ROBOT2 import set_tool, set_tcp, amovel, DR_MV_MOD_REL, DR_MV_RA_OVERRIDE, DR_TOOL, get_current_posj, get_current_posx, set_digital_output

    try:
        set_tool(ROBOT_TOOL)
        set_tcp(ROBOT_TCP)
        print(">>> Robot Initialized.")
    except Exception as e:
        print(f"[ERROR] Init Failed: {e}")
        return

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.new_cmd:
                step = list(node.target_step) 
                mode = node.mode_id
                node.new_cmd = False 

                # =========================================================
                # [수정] 그리퍼 및 정지 로직 적용
                # =========================================================
                
                # 1. 정지 모드 (손가락 3개)
                if mode == 0:
                    continue

                # 2. 그리퍼 닫기 (주먹) - Mode 3
                elif mode == 3:
                    print(">>> Gripper CLOSE (Fist Detected)")
                    try:
                        set_digital_output(1, ON)
                        set_digital_output(2, OFF)
                        # 로봇이 0.5초 멈추는 것이 의도된 동작이라면 sleep 유지
                        time.sleep(0.5) 
                    except Exception as e:
                        print(f"[Gripper Error] {e}")
                    continue # 그리퍼 동작 중에는 이동하지 않음

                # 3. 그리퍼 열기 (보자기) - Mode 4
                elif mode == 4:
                    print(">>> Gripper OPEN (Open Hand Detected)")
                    try:
                        set_digital_output(1, OFF)
                        set_digital_output(2, ON)
                        time.sleep(0.5)
                    except Exception as e:
                        print(f"[Gripper Error] {e}")
                    continue # 그리퍼 동작 중에는 이동하지 않음

                # =========================================================

                # 4. 특이점(Singularity) 및 안전 검사 (기존 유지)
                try:
                    curr_j, _ = get_current_posj()
                    curr_p, _ = get_current_posx()
                    j3 = curr_j[2]
                    j5 = curr_j[4]
                    curr_radius = math.sqrt(curr_p[0]**2 + curr_p[1]**2)

                    if abs(j3) < LIMIT_J3_BUFFER:
                        if step[2] > 0: step[2] = 0 
                    if abs(j5) < LIMIT_J5_BUFFER:
                        step[0] *= 0.1
                        step[1] = 0 
                        step[2] *= 0.1
                    if curr_radius < LIMIT_RADIUS_MIN:
                        if step[2] < 0: step[2] = 0 

                except Exception as e:
                    pass

                # 5. 이동 명령 실행
                if sum(abs(v) for v in step) > 0.001:
                    try:
                        amovel(step, time=MOVE_TIME, mod=DR_MV_MOD_REL, ref=DR_TOOL, ra=DR_MV_RA_OVERRIDE)
                    except Exception as e:
                        print(f"[Motion Error] {e}")

    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()