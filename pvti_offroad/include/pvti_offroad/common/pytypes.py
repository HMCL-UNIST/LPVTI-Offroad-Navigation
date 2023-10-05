from dataclasses import dataclass, field
import numpy as np
import copy
import torch
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header, Float64
from tf.transformations import euler_from_quaternion
from pvti_offroad.common.utils import get_local_vel
from typing import List
from cv_bridge import CvBridge


@dataclass
class PythonMsg:
   

    def __setattr__(self, key, value):
        '''
        Overloads default attribute-setting functionality to avoid creating new fields that don't already exist
        This exists to avoid hard-to-debug errors from accidentally adding new fields instead of modifying existing ones

        To avoid this, use:
        object.__setattr__(instance, key, value)
        ONLY when absolutely necessary.
        '''
        if not hasattr(self, key):
            raise TypeError('Cannot add new field "%s" to frozen class %s' % (key, self))
        else:
            object.__setattr__(self, key, value)

    def print(self, depth=0, name=None):
        '''
        default __str__ method is not easy to read, especially for nested classes.
        This is easier to read but much longer

        Will not work with "from_str" method.
        '''
        print_str = ''
        for j in range(depth): print_str += '  '
        if name:
            print_str += name + ' (' + type(self).__name__ + '):\n'
        else:
            print_str += type(self).__name__ + ':\n'
        for key in vars(self):
            val = self.__getattribute__(key)
            if isinstance(val, PythonMsg):
                print_str += val.print(depth=depth + 1, name=key)
            else:
                for j in range(depth + 1): print_str += '  '
                print_str += str(key) + '=' + str(val)
                print_str += '\n'

        if depth == 0:
            print(print_str)
        else:
            return print_str

    def from_str(self, string_rep):
        '''
        inverts dataclass.__str__() method generated for this class so you can unpack objects sent via text (e.g. through multiprocessing.Queue)
        '''
        val_str_index = 0
        for key in vars(self):
            val_str_index = string_rep.find(key + '=', val_str_index) + len(key) + 1  # add 1 for the '=' sign
            value_substr = string_rep[val_str_index: string_rep.find(',',
                                                                     val_str_index)]  # (thomasfork) - this should work as long as there are no string entries with commas

            if '\'' in value_substr:  # strings are put in quotes
                self.__setattr__(key, value_substr[1:-1])
            if 'None' in value_substr:
                self.__setattr__(key, None)
            else:
                self.__setattr__(key, float(value_substr))

    def copy(self):
        return copy.deepcopy(self)



@dataclass
class VehicleCommand(PythonMsg):
    ax: float = field(default=0) # local desired velocity     
    steer:  float = field(default=0) # steering delta in randian    

@dataclass
class OrientationEuler(PythonMsg):
    roll: float = field(default=0)
    pitch: float = field(default=0)
    yaw: float = field(default=0)


@dataclass
class CameraIntExt(PythonMsg):
    height : int = field(default = 0)
    width : int = field(default = 0)    
    fx: float = field(default = 0)
    fy: float = field(default = 0)
    cx: float = field(default = 0)
    cy: float = field(default = 0)
    distortion: np.ndarray = field(default = None)    
    R_camera_to_base: np.ndarray = field(default=None)

    def update_cam_int_ext_info(self,info : CameraInfo, robot_to_camera_matrix: np.ndarray):    
        self.height = info.height
        self.width = info.width
        self.fx = info.K[0]
        self.fy = info.K[4]
        self.cx = info.K[2]
        self.cy = info.K[5]
        self.distortion = np.array(info.D)
        self.R_camera_to_base = robot_to_camera_matrix

        
@dataclass
class VehicleState(PythonMsg): 
    '''
    Complete vehicle state (local, global, and input)
    '''
    header: Header = field(default=None)  # time in seconds
    u: VehicleCommand = field(default=None)
    odom: Odometry = field(default=None)  # global odom        
    local_twist : Twist = field(default=None) # local twist
    euler: OrientationEuler = field(default=None)

    def __post_init__(self):
        if self.header is None: self.header = Header()
        if self.odom is None: self.odom = Odometry()                
        if self.u is None: self.u = VehicleCommand()
        if self.local_twist is None: self.local_twist =  Twist()        
        if self.euler is None: self.euler =  OrientationEuler()        
        return
    
    def update_odom(self,odom:Odometry):
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()                

    def update_from_msg(self,odom:Odometry, u : VehicleCommand):
        self.u = u        
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()        

    def update_from_auc(self,steer_cmd: Float64, ax_cmd: Float64, odom: Odometry):
        self.u.ax = ax_cmd.data
        self.u.steer = steer_cmd.data                 
        self.header = odom.header
        self.odom = odom
        self.update_euler()
        self.update_body_velocity_from_global()
        
        
    def update_global_velocity_from_body(self):
        return 
        
    def update_euler(self):
        if self.odom is not None:
            orientation = self.odom.pose.pose.orientation 
            (roll, pitch, yaw) = euler_from_quaternion([orientation.x, orientation.y, orientation.z, orientation.w])
            self.euler.roll = roll
            self.euler.pitch = pitch
            self.euler.yaw = yaw

    def update_body_velocity_from_global(self):
        if self.euler.yaw is None:
            self.update_euler()            
         
        local_vel, local_ang_vel = get_local_vel(self.odom, is_odom_local_frame = False)
        self.local_twist.linear.x = local_vel[0]
        self.local_twist.linear.y = local_vel[1]
        self.local_twist.linear.z = local_vel[2]
        self.local_twist.angular.x = local_ang_vel[0]
        self.local_twist.angular.y = local_ang_vel[1]
        self.local_twist.angular.z = local_ang_vel[2]

    def get_xhat(self):
        #   x, y, vx, vy, wz,  roll, pitch, yaw         
        if self.local_twist is None:
            self.update_body_velocity_from_global()
        if self.euler.yaw is None:
            self.update_euler()

        px = self.odom.pose.pose.position.x.cpu()
        py = self.odom.pose.pose.position.y.cpu()
        vx = self.local_twist.linear.x.cpu()
        vy = self.local_twist.linear.y.cpu()
        wz = self.local_twist.angular.z.cpu()
        yaw = self.euler.yaw.cpu()
        pitch = self.euler.pitch.cpu()
        roll = self.euler.roll.cpu()
        return torch.stack([px,py,vx,vy,wz,roll,pitch,yaw])





@dataclass
class ImageKeyFrame(PythonMsg):
    header:Header = field(default = None)
    
    

@dataclass
class AUCModelData(PythonMsg): 
    '''
    Complete AUC Input data 
    '''       
    header: Header = field(default=None)  # time in seconds 
    vehicle: VehicleState = field(default=None)  # time in seconds
    pred_vehicles: List[VehicleState] = field(default=None) 
    color: torch.tensor = field(default = torch.tensor)    
    depth: torch.tensor = field(default = torch.tensor)    
    local_deviation: torch.tensor = field(default = torch.tensor)   
    xhat:torch.tensor =  field(default = torch.tensor)   
    image: torch.tensor = field(default = torch.tensor)    
    
    def update_from_rosmsgs(self, bridge: CvBridge, steer_cmd:Float64, ax_cmd: Float64, odom:Odometry, depth_msg:Image, color_msg:Image):

        self.header = odom.header
        self.vehicle = VehicleState()
        self.vehicle.update_from_auc(steer_cmd, ax_cmd, odom)

        depth_image = np.copy(bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough"))
        # depth_image = np.frombuffer(depth_msg.data, dtype=np.float32).reshape(depth_msg.height, depth_msg.width, -1).copy().squeeze()
        # color_image = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(color_msg.height, color_msg.width, -1).copy()
        replacement_distance = 16.0  # if nan -> maximum distance as 100
        # Find NaN values in the image
        depth_nan_indices = np.isnan(depth_image)
        # Replace NaN values with the replacement distance
        if len(depth_image[depth_nan_indices]) > 0:
            depth_image[depth_nan_indices] = replacement_distance

        color_image = np.copy(bridge.imgmsg_to_cv2(color_msg, desired_encoding="passthrough"))
        replacement_color = [0, 0, 0]  # if non -> White
        # Find NaN values in the image
        nan_indices = np.isnan(color_image)
        # Replace NaN values with the replacement color
        if len(color_image[nan_indices]) > 0:
            color_image[nan_indices] = replacement_color

        self.depth = depth_image
        self.color = color_image
    
    def update_with_xhat_and_img(self, odom:Odometry, xhat: torch.tensor,img:torch.tensor):
        self.header = odom.header
        self.xhat = xhat       
        self.image = img

    def update_from_pred_u(self, odom:Odometry, pred_states: torch.tensor, batch_u: torch.tensor, depth_img:torch.tensor,color_img:torch.tensor):
                                   
        self.header = odom.header
       
        self.pred_vehicles = []             
        seq = pred_states.shape[1]
        for i in range(seq):
            tmp_vehicle = VehicleState()
            tmp_state = pred_states[0,i,:]
            #   x, y, psi, vx, vy, wz, z, roll, pitch 
            px = tmp_state[0]
            py = tmp_state[1]
            yaw = tmp_state[2]
            vx = tmp_state[3]
            vy = tmp_state[4]
            wz = tmp_state[5]
            pz = tmp_state[6]
            roll = tmp_state[7]
            pitch = tmp_state[8]
            
            tmp_vehicle.odom.pose.pose.position.x = px
            tmp_vehicle.odom.pose.pose.position.y = py
            tmp_vehicle.odom.pose.pose.position.z = pz
            tmp_vehicle.local_twist.linear.x = vx
            tmp_vehicle.local_twist.linear.y = vy
            tmp_vehicle.local_twist.angular.z = wz
            tmp_vehicle.euler.yaw = yaw 
            tmp_vehicle.euler.pitch = pitch
            tmp_vehicle.euler.roll = roll
            if i < batch_u.shape[1]:
                u = batch_u[0,i,:]
                ax = u[0]
                delta = u[1]
                tmp_vehicle.u.ax = ax
                tmp_vehicle.u.steer = delta
            self.pred_vehicles.append(tmp_vehicle.copy())

        self.vehicle = self.pred_vehicles[0]

        self.depth = depth_img
        self.color =color_img


    
    def get_pose(self):
        return np.array([self.vehicle.odom.pose.pose.position.x, self.vehicle.odom.pose.pose.position.y, self.vehicle.odom.pose.pose.position.z, self.vehicle.euler.yaw])        
        
    def get_gp_output_measurement(self):
        self.vehicle.update_body_velocity_from_global()
        px = self.vehicle.odom.pose.pose.position.x
        py = self.vehicle.odom.pose.pose.position.y                
        vx = self.vehicle.local_twist.linear.x
        vy = self.vehicle.local_twist.linear.y
        wz = self.vehicle.odom.twist.twist.angular.z        
        return np.array([px,py,vx, vy, wz])
    
    def get_local_pose(self):
    # return np.array([self.vehicle.odom.pose.pose.position.x, self.vehicle.odom.pose.pose.position.y, self.vehicle.odom.pose.pose.position.z, self.vehicle.euler.yaw, ])        
        return np.array([self.vehicle.odom.pose.pose.position.x, self.vehicle.odom.pose.pose.position.y, self.vehicle.euler.yaw,
                            self.vehicle.local_twist.linear.x,self.vehicle.local_twist.linear.y ,self.vehicle.local_twist.angular.z,
                            self.vehicle.odom.pose.pose.position.z,self.vehicle.euler.roll,self.vehicle.euler.pitch])

    def get_global_pose(self):
          return np.array([self.vehicle.odom.pose.pose.position.x, self.vehicle.odom.pose.pose.position.y, self.vehicle.euler.yaw,
                         self.vehicle.odom.twist.twist.linear.x,self.vehicle.odom.twist.twist.linear.y ,self.vehicle.odom.twist.twist.angular.z,
                         self.vehicle.odom.pose.pose.position.z,self.vehicle.euler.roll,self.vehicle.euler.pitch])
    

    def get_cur_vehicle_state_input(self):
        
        '''
        # we use local velocity and pose as vehicle current state         
        TODO: see if vz, wx, wy can be ignored and independent of estimation process
        '''
        return np.array([self.vehicle.local_twist.linear.x,
                        self.vehicle.local_twist.linear.y,
                        self.vehicle.local_twist.linear.z,
                        self.vehicle.local_twist.angular.x,
                        self.vehicle.local_twist.angular.y,
                        self.vehicle.local_twist.angular.z,
                        self.vehicle.euler.pitch,
                        self.vehicle.euler.roll])
        
    

@dataclass
class SimData():    
    N: int
    auc_data: List[AUCModelData]
    camera_info: CameraIntExt = field(default=None)  # global odom

