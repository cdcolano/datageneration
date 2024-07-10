#TODO REVISAR EL METODO DE CALCULATE SI ES VISIBLE (PARECE QUE ESTA BIEN A VECES OTRAS PARECE QUE FALLA)
#TODO REVISAR EL METODO PARA EXTRAER EL POINT CLOUD DE UNA CAMARA (SIMPLEMENTE ES EXTRAER EL CLOUD NO FILTRAR Y APLICAR LA TRANSFORMATION 
# MATRIX DROPEANDO el ultimo elemento del array [x,y,z,n]--->[x,y,z])
#TODO CREAR METODO PARA GUARDAR AJUSTES DE UNA CAMARA
#TODO create anomalies out of axis
#TODO create anomalies removing parts if len(link.get_children)==0 se añade a una lista y se coge un elemento aleatorio de la lista y se manda a la PUTA
#TODO poner ready para ejecutar, ( ya casi estaria pero pensar estrategias de multithreading para acelerar) y las fotos

#TO REVISE EN LAS JOINT NO MUTABLES PARECE QUE PRODUCE GIROS POR LO TANTO HABRIA QUE ASEGURARSE QUE EL CHILD NO FUERA EL ROOT (no deberia serlo nunca), pero bueno
# si es el caso y hay un parent hacemos la inversa fallo del programa (parece que esta resuelto)
from tqdm import tqdm
import sapien
import numpy as np
import math
from PIL import Image,ImageColor
import trimesh
import scipy
import random
from scipy.spatial.transform import Rotation as R
import quaternion
import xml.etree.ElementTree as ET
import os
import json
from torch_utils import look_at_rotation
from trimesh_utils import get_articulation_meshes
from scipy.spatial import KDTree
from pathlib import Path
import concurrent.futures

def initialize_joint_limit_array(robot):
    init_qpos=robot.get_qpos()
    #Set initial joint positions
    # Get the active joints
    active_joints = robot.get_active_joints()
    
    # Check if there are any active joints
    if not active_joints:
        return []

    # Access the first active joint
    first_joint = active_joints[0]
    print(first_joint.get_type())
    # Get the limits of the first joint
    limits = first_joint.get_limits()

    # Initialize the array with the lower limit unless it's zero
    if not math.isinf(limits[0][0]) :
        init_value = limits[0][0]
    else:
        # Define what to do if the limit is zero, e.g., use an alternative value or skip
        init_value = 0  # Or some other value as per your requirement
    init_qpos[0]=init_value
    robot.set_qpos(init_qpos)
    return

import math
import random

def initialize_joint_randomly(robot):
    init_qpos=robot.get_qpos()
    # Get the active joints
    active_joints = robot.get_active_joints()
    
    # Check if there are any active joints
    if not active_joints:
        return []
    
    # Access the first active joint
    first_joint = active_joints[0]
    
    # Get the limits of the first joint
    limits = first_joint.get_limits()
    
    # Define the range for random value generation
    lower_limit, upper_limit = limits[0][0], limits[0][1]

    # Initialize the array with a random value within the limits
    if math.isinf(lower_limit) or math.isinf(upper_limit):
        # If limits are infinite, use the range 0 to 2pi radians
        init_value = random.uniform(0, 2 * math.pi)
    else:
        # Otherwise, use the actual limits of the joint
        init_value = random.uniform(lower_limit, upper_limit)
    init_qpos[0]=init_value
    robot.set_qpos(init_qpos)
    return 

def calculate_iou_and_anomaly_box(mask1, mask2):
    """
    Calculate the Intersection over Union (IoU) and the bounding box for anomalies 
    between two segmentation masks.

    Parameters:
    - mask1: np.array, first segmentation mask
    - mask2: np.array, second segmentation mask

    Returns:
    - iou: float, Intersection over Union value
    - anomaly_box: tuple, bounding box coordinates (x_min, y_min, x_max, y_max)
    """

    # Ensure masks are binary
    mask1 = mask1.astype(np.uint8)
    mask2 = mask2.astype(np.uint8)

    # Calculate intersection and union
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Compute IoU
    iou = np.sum(intersection) / np.sum(union)

    # Calculate the difference mask to find anomalies
    difference_mask = np.abs(mask1 - mask2)

    # Find coordinates of non-zero elements
    y_coords, x_coords = np.where(difference_mask > 0)
    if y_coords.size > 0 and x_coords.size > 0:
        y_min, y_max = y_coords.min(), y_coords.max()
        x_min, x_max = x_coords.min(), x_coords.max()
        # Calculate width and height
        wi = x_max - x_min + 1
        hi = y_max - y_min + 1
        anomaly_box = (x_min, y_min, wi, hi)
        #anomaly_box = (x_min, y_min, x_max, y_max) dependiendo del formato que quiera
    else:
        anomaly_box = None
    return iou, anomaly_box


def is_mesh_visible(camera, link, proximity_threshold=0.1):
    pose=link.get_entity_pose()
    # Step 0: Take a picture with the camera and get the point cloud
    camera.take_picture()
    position = camera.get_picture("Position")
    points_opengl = position[..., :3][position[..., 3] < 1]  # Assuming correct far plane value here
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    for mesh in link.get_collision_shapes():    
        # Step 1: Get mesh vertices and apply the transformation from pose
        mesh_vertices = mesh.get_vertices()  # Assuming these are local coordinates
        mesh_faces = mesh.get_triangles()
        translation = pose.get_p()  # Shape (3,)
        quaternion = pose.get_q()   # Shape (4,)
        rotation_matrix = R.from_quat(quaternion).as_matrix()
        # Transformation matrix from pose
        RT = np.eye(4)
        RT[:3, :3] = rotation_matrix
        RT[:3, 3] = translation

        # Transform vertices into world coordinates
        mesh_vertices_homogeneous = np.hstack([mesh_vertices, np.ones((mesh_vertices.shape[0], 1))])
        transformed_vertices = (RT @ mesh_vertices_homogeneous.T).T[:, :3]

        # Rebuild mesh with transformed vertices for accurate surface sampling
        mesh_transformed = trimesh.Trimesh(vertices=transformed_vertices, faces=mesh_faces)
        points_3d = trimesh.sample.sample_surface(mesh_transformed, 2000)[0]

        # Step 2: Check proximity of each sampled point against the point cloud
        distances = scipy.spatial.distance.cdist(points_3d, points_world, 'euclidean')
        min_distances = distances.min(axis=1)
        visible = np.any(min_distances < proximity_threshold)
        if visible:
            return True
    return False

def process_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    return tree, root

def create_npy_file(camera, points_3d, camara2Params, file_name):
        # Create a PIL image from the normalized numpy array
    position = camera.get_picture("Position")
    points_opengl = position[..., :3]
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    points_flat = points_world.reshape(-1, 3)
    stride = 8
    # Build KD-Tree for camera points
    
    tree = KDTree(points_flat)
    visibility_threshold = 0.1
    # Compute the nearest distance for each point in points_3d
    visible_flags = []
    for point in points_3d:
        distance, index = tree.query(point)
        is_visible = distance < visibility_threshold
        visible_flags.append(is_visible)

    visible_array = np.array(visible_flags)
    # Downsampling
    downsampled_points = points_world[::stride, ::stride, :] #TODO SI no funciona por la dimensionalidad de points world hacer el downsampling
    x,y,z = points_3d[:,0], points_3d[:,1], points_3d[:,2]
    params=camara2Params[camera.get_name()]
    K=params['K']
    RT=params['RT']
    P = K@RT[:3]
    xy = np.stack([x,y,z, np.ones_like(z)],-1)@P.T
    xy = xy/xy[:,2:]
    xy = np.array([(int(i[1].round()), 256 - int(i[0].round())) for i in xy])
    np.save(file_name.replace('.png', '.npy'), {'pxy': xy, 'is_visible':visible_array, 'xyz':downsampled_points})
    with open(file_name.replace('.png', '.json'), 'w+') as json_file:
        json.dump({'RT':RT.tolist(), 'K':K.tolist(), 'RT_':params['RT_'].tolist()}, json_file, indent=4)

def camara_take_picture_segmentation(camera,OBJECT, t, camara2Params, points_3d):
    camera.take_picture()
    seg_labels= camera.get_picture("Segmentation")  # [H, W, 4]
    label_image = seg_labels[..., 0]  
    # colormap = sorted(set(ImageColor.colormap.values()))
    # color_palette = np.array(
    #     [ImageColor.getrgb(color) for color in colormap], dtype=np.uint8
    # )
    # label0_image = seg_labels[..., 0].astype(np.uint8)
    # label0_pil = Image.fromarray(color_palette[label0_image])
    min_val = np.min(label_image)
    max_val = np.max(label_image)
    label_image_normalized = ((label_image - min_val) / (max_val - min_val)) * 255
    label_image_normalized = label_image_normalized.astype(np.uint8)  # Convert to uint8
    label_pil_gray = Image.fromarray(label_image_normalized, 'L') 
    file_name="data/normals/{}/rendered_mesh/{}_{}_reference_model.png".format(OBJECT,camera.get_name(), t)
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    label_pil_gray.save(file_name)
    create_npy_file(camera, points_3d, camara2Params, file_name)


def save_binary_mask(camera, OBJECT, t,binary_mask0):
    word="old"
    file_name="./data/anomalies/{}/annotation/mask_{}_{}_{}_.png".format(OBJECT,word,camera.get_name(), t)
    binary_mask0_pil = Image.fromarray(binary_mask0)
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    binary_mask0_pil.save(file_name)
def create_binary_mask(camera,OBJECT, t, anomaly=False):
    camera.take_picture()
    # Extract the segmentation labels from the camera
    seg_labels = camera.get_picture("Segmentation")  # [H, W, 4]
    # Create binary masks from the segmentation labels
    # Assuming label0_image for mesh-level and label1_image for actor-level
    label0_image = seg_labels[..., 0].astype(np.uint8)  # mesh-level
    # Convert the labels to binary masks
    binary_mask0 = np.where(label0_image > 0, 255, 0).astype(np.uint8)
    if anomaly:
        binary_mask0_pil = Image.fromarray(binary_mask0)
        word = "new"
        file_name="./data/anomalies/{}/annotation/mask_{}_{}_{}_.png".format(OBJECT,word,camera.get_name(), t)
        directory = os.path.dirname(file_name)
        if not os.path.exists(directory):
            os.makedirs(directory)
        binary_mask0_pil.save(file_name)
    return binary_mask0



def camara_take_rgb_picture(camera,OBJECT, t, anomaly=False, points_3d=None, camara2Params=None ):
    camera.take_picture()
    rgba = camera.get_picture("Color")  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    word = "anomalies" if anomaly else "normals"
    file_name="./data/{}/{}/renders/{}_{}_.png".format(word, OBJECT, camera.get_name(), t)
    directory = os.path.dirname(file_name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    rgba_pil.save(file_name)
    if not anomaly:
        create_npy_file(camera, points_3d, camara2Params, file_name)
    


def create_anomaly_removing(robot, camara,scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    # Process the URDF to obtain its tree and root
    links_to_remove=[]
    tree, urdf_root = process_urdf(path)
    first_active_joint = robot.get_active_joints()[0]
    parent_link = first_active_joint.get_parent_link()
    child_link = first_active_joint.get_child_link()
    
    if len(links_to_remove)==0:
        for link in robot.get_links():
            if len(link.get_children())==0:
                joints = urdf_root.findall(".//joint")
                #print(link.get_name())
                parent_joints = [joint for joint in joints if joint.find("parent").get("link") == link.get_name()]
                #print(parent_joints)
                if link not in (parent_link, child_link):
                    if not parent_joints: 
                        links_to_remove.append(link) 
    #print(links_to_remove)
    #print("links")
    if not links_to_remove:
        return robot, False  # No leaf links to remove
    
    file_path = Path(path)
    
    # Get the parent directory
    parent_dir = file_path.parent
    
    # Define the new file path
    output_path = parent_dir / "tmp.urdf"
    output_path = str(output_path)
    # Select a random leaf link to remove
    link_to_remove = random.choice(links_to_remove)
    link_name = link_to_remove.get_name()
    # Find the link element by name and remove it
    link = urdf_root.find(f".//link[@name='{link_name}']")
    if link is not None:
        urdf_root.remove(link)

    joints = urdf_root.findall(".//joint")
    joints_to_remove = [joint for joint in joints if joint.find("child").get("link") == link_name]
    for joint in joints_to_remove:
        urdf_root.remove(joint)

    # Write the modified URDF back to file
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    # Check if the robot after modification is still valid and visible

    scene.remove_articulation(robot)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(output_path)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
    return robot, True
   



def modify_urdf_with_translation(robot, camara,scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    file_path = Path(path)
    
    # Get the parent directory
    parent_dir = file_path.parent
    
    # Define the new file path
    output_path = parent_dir / "tmp.urdf"
    output_path = str(output_path)
    MAX_DISTANCE = 0.5
    rotation_delta = np.zeros(3)
    axis = np.random.choice([0, 1, 2])
    distance = 0
    while distance == 0:
        distance = np.random.uniform(-MAX_DISTANCE, MAX_DISTANCE)
   
    rotation_delta[axis] = distance
   # This is useful if the rotation needs to be on a random plane
    # while np.all(rotation_delta == 0):
    #     rotation_delta = np.random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES, size=3)  # generate non-zero random rotation
    #rotation_delta = np.radians(rotation_delta)
    
    tree, urdf_root = process_urdf(path)
    link_sapien=random.choice(robot.get_links())
    link_name=link_sapien.get_name()
      # Find the link element by name
    #inverse_rotation = np.linalg.inv(rotation_matrix)
    link = urdf_root.find(f".//link[@name='{link_name}']")
    if link is not None:
        # Apply transformations to visual elements in the link
        for visual in link.findall('visual'):
            origin = visual.find('origin')
            if origin is not None:
                current_xyz = np.array((origin.get('xyz') or '0 0 0').split(), dtype=float)
                new_xyz = current_xyz + rotation_delta
                origin.set('xyz', ' '.join(f"{x:.6f}" for x in new_xyz))
        
        # Apply transformations to collision elements in the link
        for collision in link.findall('collision'):
            origin = collision.find('origin')
            if origin is not None:
                current_xyz = np.array((origin.get('xyz') or '0 0 0').split(), dtype=float)
                new_xyz = current_xyz + rotation_delta
                origin.set('xyz', ' '.join(f"{x:.6f}" for x in new_xyz))

    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    scene.remove_articulation(robot)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(output_path)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1,0,0,0]))
    return robot, True

def modify_urdf_with_rotation(robot,camara, scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    file_path = Path(path)
    
    # Get the parent directory
    parent_dir = file_path.parent
    
    # Define the new file path
    output_path = parent_dir / "tmp.urdf"
    output_path = str(output_path)
    MAX_ROTATION_DEGREES = 20
    rotation_delta = np.zeros(3)
    axis = np.random.choice([0, 1, 2])
    angle = 0
    while angle == 0:
        angle = np.random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES)
   
    rotation_delta[axis] = angle
   # This is useful if the rotation needs to be on a random plane
    # while np.all(rotation_delta == 0):
    #     rotation_delta = np.random.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES, size=3)  # generate non-zero random rotation
    rotation_delta = np.radians(rotation_delta)
    rotating_joints=set(['revolute', 'revolute_unwrapped', 'free'])
    tree, urdf_root = process_urdf(path)
    joints = urdf_root.findall('joint')
    child_links_from_finite_joints = {joint.find('child').get('link') for joint in joints if joint.get('type') in rotating_joints}
    #print(robot.get_active_joints())
    #print("ROBOCP")
    # Select a link that is not a child in a revolute or universal joint
    available_links = [link for link in robot.get_links() if link.get_name() not in child_links_from_finite_joints]
    if not available_links:  # If no suitable links, return with no changes
        return robot, False
    link_sapien=random.choice(robot.get_links())
    link_name=link_sapien.get_name()
      # Find the link element by name
    #inverse_rotation = np.linalg.inv(rotation_matrix)
    link = urdf_root.find(f".//link[@name='{link_name}']")
    if link is not None:
        # Apply transformations to visual elements in the link
        for visual in link.findall('visual'):
            origin = visual.find('origin')
            if origin is not None:
                current_rpy = np.array((origin.get('rpy') or '0 0 0').split(), dtype=float)
                new_rpy = current_rpy + rotation_delta
                origin.set('rpy', ' '.join(f"{x:.6f}" for x in new_rpy))
        
        # Apply transformations to collision elements in the link
        for collision in link.findall('collision'):
            origin = collision.find('origin')
            if origin is not None:
                current_rpy = np.array((origin.get('rpy') or '0 0 0').split(), dtype=float)
                new_rpy = current_rpy + rotation_delta
                origin.set('rpy', ' '.join(f"{x:.6f}" for x in new_rpy))
    
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    scene.remove_articulation(robot)
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(output_path)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1,0,0,0]))
    return robot, True

def range_anomalies(robot, camara,scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    active_joints=robot.get_active_joints()
    if not active_joints:
        return robot, False
    if len(active_joints)==0:
        return robot, False
    joint=active_joints[0]
    joint_type=joint.get_type()
    if joint_type not in FINITE_JOINT_TYPES:
        return robot, False
    
    q_pos=robot.get_qpos()
    #LOGICA PARA CAMBIAR LOS LIMITES Y LLEVAR LA JOINT HASTA EL NUEVO LIMITE UTIL PARA UNWRAPPED REVOULTE Y PARA PRISMATIC
    limits=active_joints[0].get_limits()
    positive_angle = (limits[0][0] + 2 * math.pi) % (2 * math.pi)
    lower_bound = math.pi / 8 #ESTOS LIMITES SIRVEN PARA REVOULT
    minimum=positive_angle-q_pos[0]
    upper_bound = minimum-math.pi/8
    if joint_type=="prismatic":
        difference=limits[0][1]-limits[0][0]
        min_threshold=0.4
        lower_bound=difference*0.3
        upper_bound=difference*0.7
        lower_bound = max(lower_bound, min_threshold)
        upper_bound = max(upper_bound, min_threshold)


    if lower_bound<upper_bound:    
        # Generate a random number between lower and upper
        increase = np.random.uniform(lower_bound, upper_bound)
        #limits[0][1]=limits[0][1]+upper_bound
        #active_joints[0].set_limits(limits)
        #scene.step()
        q_pos[0]=q_pos[0]+increase
        robot.set_qpos(q_pos)
        return robot, True
    return robot, False
   
   


def random_quaternion():
    axes = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    # Choose a random index for the axes
    index = np.random.choice(len(axes))
    axis = axes[index]  # Select an axis using the random index
    
    # Generate a random angle in radians, ensuring it's subtle
    angle = np.random.uniform(0.1, 2 * np.pi)  # from 0.1 radian to full circle
    angle=np.random.uniform(1/32*np.pi, 1/4 * np.pi) 
    # Compute quaternion components using the axis-angle method
    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    qx, qy, qz = s * axis
    qw = c
    
    # Create a quaternion object from the components
    quat = quaternion.quaternion(qw, qx, qy, qz)
    
    return quat

def random_quaternion_around_axis(axis):
    # Generate a random rotation around a given axis
    angle = np.random.uniform(0.3, 2*np.pi)  # Full rotation range
    s = np.sin(angle / 2)
    quat = np.array([np.cos(angle / 2), s * axis[0], s * axis[1], s * axis[2]])
    quat /= np.linalg.norm(quat)  # Normalize the quaternion
    return quat


def get_joint_axis(urdf_path, joint_name):
    """ Extracts the rotation axis of a specified joint from a URDF file using numpy for numeric processing. """
    # Load and parse the URDF file
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    # Directly find the joint and extract the axis, converting to a numpy array
    axis_element = root.find(f".//joint[@name='{joint_name}']/axis")
    if axis_element is not None and 'xyz' in axis_element.attrib:
        axis = np.fromstring(axis_element.attrib['xyz'], sep=' ')
        return axis
    else:
        raise ValueError(f"Axis information not found for joint '{joint_name}'.")
    
def create_joint_out_axis_anomaly(robot, camara, scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    rotating_joints=set(['revolute', 'revolute_unwrapped', 'free'])
    joints=robot.get_joints()
    joins_to_apply=[]
    for joint in joints:
        joint_type = joint.get_type()  # Retrieve the type of the joint
        # Check if the joint type is not 'revolute' or 'revolute_unwrapped'
        if joint_type in rotating_joints:
            joins_to_apply.append(joint)
    if not rotating_joints or len(joins_to_apply)==0:
        return robot, False

    joint = random.choice(joins_to_apply)
    normal_axis = get_joint_axis(path, joint.get_name())
    normal_axis = normal_axis / np.linalg.norm(normal_axis)
    # Generate a perpendicular axis as the abnormal axis
    abnormal_axis = np.array([-normal_axis[1], normal_axis[0], 0])
    normal_axis_norm = np.linalg.norm(normal_axis)
    if normal_axis_norm < 1e-6:  # Check if the normal axis is valid
        print("AQUI")
        return robot, False
    
    normal_axis /= normal_axis_norm
    # Generate a quaternion for the abnormal rotation
    abnormal_rotation = random_quaternion_around_axis(abnormal_axis)

    # Get the current global pose of the joint, modify it with the new rotation, and apply it
    current_pose = joint.get_pose_in_child()
    new_pose = sapien.Pose(current_pose.p, abnormal_rotation * current_pose.q)
    joint.set_pose_in_child(new_pose)
    return robot, True

def create_joint_rotation_anomaly(robot, camara, scene, object_name, path,FINITE_JOINT_TYPES, counter=0):
    joints=robot.get_joints()
    non_rotating_joints = []
    rotating_joints=set(['revolute', 'revolute_unwrapped', 'free'])
    # Check each joint and collect non-rotating ones
    for joint in joints:
        joint_type = joint.get_type()  # Retrieve the type of the joint
        # Check if the joint type is not 'revolute' or 'revolute_unwrapped'
        if joint_type not in rotating_joints:
            if not joint.get_child_link().is_root and not joint.get_parent_link().is_root:
                non_rotating_joints.append(joint)
    #print(len(non_rotating_joints))
    if len(non_rotating_joints)>0:
        random_index = random.randint(0, len(non_rotating_joints)-1)
        joint=non_rotating_joints[random_index]
        quat = random_quaternion()
        quat_array = np.array([quat.w, quat.x, quat.y, quat.z])
        #print(quat_array)
        # Get current position from the joint
        child=joint.get_child_link()
       
        pose=child.get_entity_pose()
        pose.set_q(quat_array)
        joint.set_pose_in_child(pose)
        return robot, True #TODO NO NECESITA QUE SEA VISIBLE PORQUE ES REACCION EN CADENA PODEMOS COMPROBAR COMO MUCHO QUE EL PADRE SEA VISIBLE SI EXISTE
    else:
        return robot, False
    
IOU_THRESHOLD=0.98 #TODO IMPORTANTE COMENTAR QUE EL IOU TIENE QUE SER BAJO DEBIDO A QUE LAS PIEZAS EN MUCHOS CASOS SON MUY PEQUEÑAS

def create_random_anomaly(anomalies, robot,scene,viewer,camara, OBJECT, path, FINITE_JOINT_TYPES,segmentation_original, t, camara2Parameters, current_position, rotation_velocity):
    anomalies_copy=anomalies.copy()
    counter=0
    while len(anomalies_copy)>0 and counter<10:
        anomaly_index=random.randint(0, len(anomalies_copy)-1)
        anomaly_to_apply=anomalies_copy[anomaly_index]
        #print(anomaly_to_apply)
        #print(anomaly_to_apply)
        counter+=1
        #print(robot)
        #TODO CHANGE FOR A LOOP OVER CAMARAS DE HECHO PUEDO GENERAR UNA ANOMALIA Y HACER UN LOOP SOBRE LAS CAMARAS Y TENER 20 anomalias
        #del mismo modo con las normales, podemos hacer 1 y uno y que la join se vaya moviendo
        robot, result=anomaly_to_apply(robot, camara, scene, OBJECT, path, FINITE_JOINT_TYPES)
        #print(result)
        if result:
            qf = robot.compute_passive_force    (
                gravity=True,
                coriolis_and_centrifugal=True,
            )
            robot.set_qf(qf)
            active_joints=robot.get_active_joints() 
            active_joints[0].set_drive_velocity_target(0)
            scene.step()
            scene.update_render()
            #viewer.render()
            segmentation_anomaly=create_binary_mask(camara, OBJECT, t, anomaly=True)
            save_binary_mask(camara, OBJECT, t, segmentation_original)
            iou,boxes=calculate_iou_and_anomaly_box(segmentation_original, segmentation_anomaly)
            if iou<IOU_THRESHOLD: #ENTONCES EL OBJECT PODRIA SER INCLUIDO
                file_name="data/anomalies/{}/annotation/info_{}_{}_.json".format(OBJECT, camara.get_name(), t)
                directory = os.path.dirname(file_name)
                if boxes:
                    boxes=[int(coord) for coord in boxes]
                iou=float(iou)
                params=camara2Parameters[camara.get_name()]
                camara_parameters={'RT':params['RT'].tolist(), 'K':params['K'].tolist(), 'RT_':params['RT_'].tolist()}
                info={'2d_bbox':boxes, 'iou': iou, 'exemplar_id':OBJECT, 'camara_parameters':camara_parameters, 'config':{'type':"position"}}
                # Check if the directory exists, if not create its
                if not os.path.exists(directory):
                    os.makedirs(directory)
                #TODO CREATE A JSON FILE WITH THE SPECIC FORMAT
                with open(file_name, 'w+') as json_file:
                    json.dump(info, json_file, indent=4)
                camara_take_rgb_picture(camara, OBJECT,t,True)
                scene.remove_articulation(robot)
                return True, robot
            scene.remove_articulation(robot)
            robot=reload_robot(scene,viewer, path, current_position, rotation_velocity)
        else:
            anomalies_copy.remove(anomaly_to_apply)
                    #Take PICTURE OF THE MODIFIED 
    return False, robot

def reload_robot(scene,viewer, path, current_position, rotation_velocity ):
    #print("RELOAD")
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    robot = loader.load(path)
    q_pos=robot.get_qpos()
    q_pos[0]=current_position
    robot.set_qpos(q_pos)
    robot.set_root_pose(sapien.Pose([0, 0, 0], [1,0,0,0]))
    active_joints=robot.get_active_joints() 
    joint=active_joints[0]
    joint.set_drive_property(stiffness=0.2, damping=20, force_limit=np.inf, mode='velocity')
    active_joints[0].set_drive_velocity_target(0)
    scene.step()
    scene.update_render()
    #viewer.render()
    joint.set_drive_property(stiffness=0.2, damping=20, force_limit=np.inf, mode='velocity')
    active_joints[0].set_drive_velocity_target(rotation_velocity)
    return robot

def compute_multi_view(robot, scene,viewer, camaras,camaras_query, FINITE_JOINT_TYPES, path, OBJECT,rotation_velocity, anomalies,camara2Params, query_camara_2_params):
    t=0
    generated_images=0
    #anomalies=[modify_urdf_with_rotation]
    counter_anomaly=0
    q_pos=robot.get_qpos()
    end_movement=False
    #path = './partnet-mobility-dataset/{}/mobility.urdf'.format(OBJECT)
    points_3d=trimesh.sample.sample_surface(get_articulation_meshes(robot), 2000)[0]
    joint=robot.get_active_joints()[0]
    # print(joint.get_limits()[0][1])
    # print(joint.get_limits()[0][0])
    #times=joint.get_limits()[0][1]-joint.get_limits()[0][0]
    while True:
        if not end_movement:
            q_pos=robot.get_qpos()
            current_position=q_pos[0]
            for camara in camaras:
                camara_take_picture_segmentation(camara, OBJECT, t, camara2Params, points_3d)
            if generated_images<150:
                selected_cameras = random.sample(camaras_query, random.randint(0, 2))
                for camara in selected_cameras:
                    camara_take_rgb_picture(camara,OBJECT, t, False, points_3d, query_camara_2_params)
                    generated_images+=1
                    segmentation_original=create_binary_mask(camara, OBJECT, t)
                    result, robot=create_random_anomaly(anomalies, robot,scene,viewer,camara, OBJECT, path, FINITE_JOINT_TYPES, segmentation_original, t, query_camara_2_params, current_position, rotation_velocity)
                    if result:
                        generated_images+=1
                        robot=reload_robot(scene, viewer,path, current_position, rotation_velocity)
            else:
                print(generated_images)
            qf = robot.compute_passive_force(
                gravity=True,
                coriolis_and_centrifugal=True,
            )
            t+=1
            robot.set_qf(qf)
            last_position=q_pos[0]
            scene.step()
            scene.update_render()
            #viewer.render()
            #Take picture
            #Get array of corresponding annotations in 3D world using the camara
            q_pos=robot.get_qpos() 
            #print(q_pos[0])
            active_joints=robot.get_active_joints()
            if not active_joints or len(active_joints)==0:
                end_movement=True
            joint=active_joints[0]
            new_position=q_pos[0]
            if joint.get_type() in FINITE_JOINT_TYPES:
                if  new_position-last_position<1e-8: #abs(current_position - joint.get_limits()[0][1]) < EPSILON: #has reached the limit
                    end_movement=True
            elif joint.get_type()=="revolute":
                if new_position<0: # Has performed a full rotation
                    end_movement=True
            else:
                print("AAYAYAYYAYYAYYAYAAYYAYYAYAYYAYYAYYAYAY")
       
        else:
            scene.remove_articulation(robot)
            scene.step()
            scene.update_render()
            #viewer.render()
            break
                    #LOAD THE PREVIOUS VERSION WITH A HELPER FUNCTION THAT SETS THE ACTIVE_JOINT[0] to a random position within the limits
                    
           
            ##TODO CHECKEAR QUE LA ANOMALIA SEA VISIBLE, se puede hacer con un helper en el propio metodo de la anomalia, le pasamos la camara como parametro
            ## checkeamos que cumpla las condiciones para introducir la anomalia (que tenga fixed joins en el caso de joint rotation, etc) y  ahi tenemos acceso directo
            ## a la localizacion del link a través de sus objetos fisicos por lo tanto podemos checkear que sea visible
            ## return el robot y si se ha podido o no crear la anomalia
            ## seleccionamos una creacion de anomalia aleatoria la aplicamos, comprobamos si vale, si no la volvemos a aplicar asi hasta N veces
            ## si pasan N veces y no se ha podido aplicar esa anomalia hay que seleccionar una aleatoria del resto
            ## si no quedan anomalias continue (pasamos al siguiente objeto)
            ##IF NOT REPETIR EL PROCESO (basicamente no incrementar el contador)

def look_at_rotation(source_point, target_point=np.array([0, 0, 0])):
    """
    Generate a rotation matrix to align the forward vector of an object at source_point
    to point towards target_point.
    """
    forward_vector = np.array([1, 0, 0])  # Assuming the object's forward is along the X-axis
    up_vector = np.array([0, 0, 1])  # Assuming the Z-axis is up

    direction = target_point - source_point
    direction /= np.linalg.norm(direction)  # Normalize the direction vector

    # Compute the right and up vectors using the forward direction
    right = np.cross(up_vector, direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)

    # Create a rotation matrix from the direction, right, and up vectors
    rotation_matrix = np.vstack([direction, right, up]).T

    return rotation_matrix


def include_camara(scene, azim, elev, dist, cameras, camara2Params, image_width = 256, image_height = 256):
    near, far = 0.1, 1000
    fovy=np.deg2rad(35)
    camera = scene.add_camera(
        name="camera{}".format(azim),
        width=image_width,
        height=image_height,
        fovy=fovy,
        near=near,
        far=far,
    )

    K = camera.get_intrinsic_matrix()


    T = np.array([[(dist * math.cos(math.radians(elev))) * math.cos(math.radians(azim)), \
                        (dist * math.cos(math.radians(elev))) * math.sin(math.radians(azim)), \
                        #dist * math.sin(math.radians(elev)), TODO esto solo es interesante si lo queremos rotar en el elevation
                        elev
                        ]])
    
    # Compute the camera pose by specifying forward(x), left(y) and up(z)
    # forward = -T / np.linalg.norm(T)
    # left = np.cross([0, 0, 1], forward)
    # left = left / np.linalg.norm(left)
    # up = np.cross(forward, left)
    # RT = np.eye(4)
    # RT[:3, :3] = np.stack([forward, left, up], axis=1)
    #RT[:3, 3] = T
    #TODO NO TENEMOS EL CENTROID ESTO HAY QUE CAMBIAR ENTIENDO QUE CON LOOK AT 000 valdria
    R = look_at_rotation(T)

    RT = np.eye(4,4)
    RT[:3,:3] = R
    RT[:3,3] = T
    # print(RT-mat44)
    camera.entity.set_pose(sapien.Pose(RT))

    R, T = RT[:3,:3], RT[:3,3:]
    RT_ = RT.copy()
    T1 = np.concatenate([np.eye(3,3), -T], -1)
    RT = np.eye(4,4)
    RT[:3,:4] = R.T@T1
    camara2Params[camera.get_name()]={'K':K, 'RT': RT, 'RT_':RT_}
    cameras.append(camera)

NUM_VIEWS=10
NUM_CAMARAS_SAMPLING=30
import os
import sys
import subprocess

class OutputCapture:
    def __init__(self):
        self.output = ""

    def write(self, s):
        self.output += s

def demo(dist = -3, elev = 2, image_width = 256, image_height = 256, gray_scale = True, obj_id = None, rotation_velocity=1):
    #9748
    #9912
    #179
    #3558
    #101685
    n_objects=0
    #9912
    #179
    #3558
    anomalies=[create_joint_rotation_anomaly,range_anomalies, modify_urdf_with_rotation, modify_urdf_with_translation, create_anomaly_removing, create_joint_out_axis_anomaly]
    #anomalies=[create_joint_out_axis_anomaly]
    #OBJECT=179
    #anomalies=[create_joint_out_axis_anomaly]
    
    scene = sapien.Scene()
    #scene.add_ground(0)
    FINITE_JOINT_TYPES=set(["revolute_unwrapped", "prismatic"])
    EPSILON=1e-4 #Tolerancia a la numerical imprecission
    N=1 #NUMBER OF CAMARAS 
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
    camaras=[]
    camaras_query=[]
    camara2Params={}
    camaras_query_2_params={}
    for azim in range(0,360+1,360//NUM_VIEWS):
        include_camara(scene, azim, elev, dist, camaras, camara2Params)

    for azim in range(0,360+1,360//NUM_CAMARAS_SAMPLING):
        include_camara(scene, azim, elev, dist, camaras_query, camaras_query_2_params)

    # viewer = scene.create_viewer()
    # viewer.set_camera_xyz(x=-2, y=0, z=1)
    # viewer.set_camera_rpy(r=0, p=-0.3, y=0)
    folder_path="./dataset"
    # Load URDF
    folder = Path(folder_path)
    direct_subfolders = [subfolder for subfolder in folder.iterdir() if subfolder.is_dir()]
    minimum=450 #TODO cambiar a 60
    for subfolder in tqdm(direct_subfolders, desc='Processing Subfolders'):
        if subfolder.is_dir():
            mobility_file = subfolder / 'mobility.urdf'
            if mobility_file.exists():
                if n_objects<minimum+1:
                    n_objects+=1
                    #print("ENTRA")
                    continue
                loader = scene.create_urdf_loader()
                loader.fix_root_link = True
                path=str(mobility_file)
                OBJECT= subfolder.name
                #urdf_file = sapien.asset.download_partnet_mobility(OBJECT, token)
                robot = loader.load(path)
                robot.set_root_pose(sapien.Pose([0, 0, 0], [1,0,0,0]))
                #anomalies_robot=robot.copy()

                initialize_joint_limit_array(robot)

                #Set initial joint positions
                #Get active joints and set velocity drive son las unicas que se pueden modificar
                active_joints = robot.get_active_joints()
                for i in range(len(active_joints)):
                    if i==0:
                        joint=active_joints[i]
                        if joint.get_type()=="revolute":
                            rotation_velocity=20
                        elif joint.get_type()=="revolute_unwrapped":
                            rotation_velocity=20
                        elif joint.get_type()=="prismatic":
                            rotation_velocity=0.1
                        joint.set_drive_property(stiffness=0.2, damping=20, force_limit=np.inf, mode='velocity')
                        joint.set_drive_velocity_target(rotation_velocity)
                        break
                
                scene.step()
                scene.update_render()
                #viewer.render()
                viewer=None
                    
                #print(path)
                compute_multi_view(robot, scene, viewer,camaras, camaras_query, FINITE_JOINT_TYPES, path, OBJECT, rotation_velocity, anomalies, camara2Params,camaras_query_2_params)
                n_objects+=1
                if n_objects>600:
                    return
        #PERFOM MODIFICATIONS TO PERFORM ANOMALIES
        #TAKE PICTURE
    #range_anomalies(robot, scene, viewer)

# def demo(subfolder, dist=-3, elev=2, image_width=256, image_height=256, gray_scale=True, obj_id=None, rotation_velocity=1):
#     n_objects = 0
#     anomalies = [create_joint_rotation_anomaly, range_anomalies, modify_urdf_with_rotation, modify_urdf_with_translation, create_anomaly_removing, create_joint_out_axis_anomaly]
#     FINITE_JOINT_TYPES=set(["revolute_unwrapped", "prismatic"])
#     scene = sapien.Scene()
#     scene.set_ambient_light([0.5, 0.5, 0.5])
#     scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
#     camaras = []
#     camaras_query = []
#     camara2Params = {}
#     camaras_query_2_params = {}
    
#     for azim in range(0, 360 + 1, 360 // NUM_VIEWS):
#         include_camara(scene, azim, elev, dist, camaras, camara2Params)

#     for azim in range(0, 360 + 1, 360 // NUM_CAMARAS_SAMPLING):
#         include_camara(scene, azim, elev, dist, camaras_query, camaras_query_2_params)

#     mobility_file = subfolder / 'mobility.urdf'
#     if mobility_file.exists():
#         loader = scene.create_urdf_loader()
#         loader.fix_root_link = True
#         path = str(mobility_file)
#         OBJECT = subfolder.name
#         robot = loader.load(path)
#         robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))
        
#         initialize_joint_limit_array(robot)

#         active_joints = robot.get_active_joints()
#         for i in range(len(active_joints)):
#             if i == 0:
#                 joint = active_joints[i]
#                 joint.set_drive_property(stiffness=0.2, damping=20, force_limit=np.inf, mode='velocity')
#                 joint.set_drive_velocity_target(rotation_velocity)
#                 break
        
#         scene.step()
#         scene.update_render()
#         viewer = None
                
#         compute_multi_view(robot, scene, viewer, camaras, camaras_query, FINITE_JOINT_TYPES, path, OBJECT, rotation_velocity, anomalies, camara2Params, camaras_query_2_params)
#         n_objects += 1
#         return n_objects

# def process_subfolders(max_threads=4):
#     folder_path = "./dataset"
#     folder = Path(folder_path)
#     direct_subfolders = [subfolder for subfolder in folder.iterdir() if subfolder.is_dir()]

#     with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
#         futures = {executor.submit(demo, subfolder): subfolder for subfolder in direct_subfolders}
#         for future in tqdm(concurrent.futures.as_completed(futures), total=len(direct_subfolders), desc='Processing Subfolders'):
#             subfolder = futures[future]
#             try:
#                 result = future.result()
            #     print(f"{subfolder.name} processed with result {result}")
            # except Exception as e:
            #     print(f"Error processing {subfolder.name}: {e}")

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fix-root-link", action="store_true")
    parser.add_argument("--balance-passive-force", action="store_true")
    args = parser.parse_args()

    demo()


#IF prismatic we need to define a few heuristics to simulate realistic articulation anomalies 
#IF rotation we need to define heuristics to simulate articulation anomalies: easy when unwrapped more than the allowed rotation
#



if __name__ == "__main__":
    main()