import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    python_interpreter = "/home/apicoo-ai/pmg/bin_picking_ws/.venv/bin/python3"
    vision_node_script = "/home/apicoo-ai/pmg/bin_picking_ws/src/indy_vision/indy_vision/vision_node.py"
    calib_broadcaster_script = "/home/apicoo-ai/pmg/bin_picking_ws/src/indy_vision/indy_vision/calibration_broadcaster.py"

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        
        # Node xử lý AI và Camera
        Node(
            package='indy_vision',
            executable=python_interpreter,
            arguments=[vision_node_script],
            name='vision_node',
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            emulate_tty=True
        ),

        # Node xử lý tọa độ Calibration
        Node(
            package='indy_vision',
            executable=python_interpreter,
            arguments=[calib_broadcaster_script],
            name='calibration_broadcaster',
            output='screen',
            emulate_tty=True
        )
    ])
