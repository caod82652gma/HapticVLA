#!/usr/bin/env python3
# -*-coding:utf8-*-
# 注意demo无法直接运行，需要pip安装sdk后才能运行
# 设置机械臂重置，需要在mit或者示教模式切换为位置速度控制模式时执行
import time
import argparse
from piper_sdk import *

def disable_arm(port_name):
    """失能单个机械臂"""
    try:
        piper = C_PiperInterface_V2(port_name, judge_flag=True)
        piper.ConnectPort()
        
        while piper.DisablePiper():
            time.sleep(0.01)
        print(f"✓ {port_name} 失能成功!")
        
        piper.MotionCtrl_1(0x02, 0, 0)  # 恢复
        time.sleep(0.1)
        return True
    except Exception as e:
        print(f"✗ {port_name} 失能失败: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="失能Piper机械臂")
    parser.add_argument(
        "--ports",
        nargs="+",
        default=["can_master", "can_follower", "can_master2", "can_follower2"],
        choices=["can_master", "can_follower", "can_master2", "can_follower2"],
        help="要失能的CAN端口列表（默认：全部）"
    )
    args = parser.parse_args()
    
    print(f"准备失能机械臂: {', '.join(args.ports)}")
    success_count = 0
    
    for port in args.ports:
        if disable_arm(port):
            success_count += 1
    
    print(f"\n总结: {success_count}/{len(args.ports)} 个机械臂成功失能")
    time.sleep(0.5)
    
