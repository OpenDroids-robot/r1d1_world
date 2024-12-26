from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    share_dir = get_package_share_directory('aruco_cube')    
    world_file_path = os.path.join(share_dir, 'world', 'world_with_aruco.world')
    
    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'gazebo',  '--verbose', world_file_path 
            ],
            output='screen'  
        )
    ])
 
