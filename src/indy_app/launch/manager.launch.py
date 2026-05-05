import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time')
    
    python_interpreter = "/home/apicoo-ai/pmg/bin_picking_ws/.venv/bin/python3"
    robot_manager_script = "/home/apicoo-ai/pmg/bin_picking_ws/src/indy_app/indy_app/robot_manager_node.py"

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        
        Node(
            package='indy_app',
            executable=python_interpreter,
            arguments=[robot_manager_script],
            parameters=[{'use_sim_time': use_sim_time}],
            output='screen',
            emulate_tty=True
        )
    ])
