import open3d as o3d
import open3d.visualization.gui as gui
import numpy as np
import cv2
from pyquaternion import Quaternion as Quat

import time
import os
import copy
import glob

import argparse

from matplotlib import pyplot as plt

from math import sin,cos

parser = argparse.ArgumentParser()
parser.add_argument("posefile", help='pose file in .txt (x,y,z,qx,qy,qz,qw) or .npz (world_mat) formats')
parser.add_argument("--imagedir", default=None, help='directory of images')
parser.add_argument("--ply", default=None, help='.ply file to draw the poses over')
parser.add_argument("--ned", action="store_true",help='add this flag if poses are in NED format (tartanair)')
parser.add_argument("--cam", action="store_true",help='draw a camera frustum instead of a coordinate frame')
parser.add_argument("--no_floor", action="store_true",help='dont draw a floor, even if no mesh is given')
parser.add_argument("--stride","-s", type=int, default=1, help='draw every Nth pose.')
parser.add_argument("--nframes","-n", type=int, default=10000, help='draw this many poses.')
parser.add_argument("--fps", type=float, default=120, help='draw this many frames per second.')
args = parser.parse_args()

Nframes = args.nframes
stride = args.stride
delay = 1.0/args.fps

CAM_POINTS = np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5]])

CAM_LINES = np.array([
    [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

def create_camera_actor(g, scale=0.05):
    """ build open3d camera polydata """
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES))

    color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def get_intrinsic_matrix(frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    out = o3d.camera.PinholeCameraIntrinsic(640, 480, intrinsics.fx,
                                            intrinsics.fy, intrinsics.ppx,
                                            intrinsics.ppy)
    return out

def vectors_from_quat(q):
    #Revisar matem'atica de esto
    x,y,z,w = q
    forward = (2*x*z + 2*w*y,2*y*z - 2*w*x,1 - 2*(x*x+y*y))
    up      = (2*x*y -2*w*z,1-2*(x*x+z*z),2*y*z+2*w*x)
    left    = (1-2*(y*y+z*z),2*x*y+2*w*z,2*x*z-2*w*y)

    return(forward, up, left)

def transform4x4(R,T):
    res = np.eye(4)
    res[:3,:3] = R 
    res[:3,3] = T

    return(res)

def RT_from_poses(file,stride=stride,ned=False):
    
    RTposes = []
    T = []
    R = []
    Q = []
    
    with open(file,'r') as f:
        n = 0
        while(True):
            s = f.readline()
            if(s==''):
                break
            n += 1
            if(n%stride != 0):
                continue
            x,y,z,qx,qy,qz,qw = np.array([float(n) for n in s.split()])

            if(ned):
                x,y,z,qx,qy,qz=y,z,x,qy,qz,qx 
                
            sc = 1.
            T.append((sc*x,sc*y,sc*z))
            #Q.append((qw,qx,qy,qz))
            Q.append((qw,qx,qy,qz))
            rot = o3d.geometry.get_rotation_matrix_from_quaternion((Q[-1]))
            #rot = np.linalg.inv(rot)
            #rot = rot[[0,2,1]]
            R.append(rot)

            RTposes.append(transform4x4(R[-1],T[-1]))

    T = np.array(T)
    R = np.array(R)
    Q = np.array(Q)
    RTposes = np.array(RTposes,dtype=np.float64)
    #RTposes = np.linalg.inv(RTposes)

    return(RTposes,R,T,Q)

def RT_from_world_mats(file,stride=stride):
#def RT_from_world_mats(directory,stride=stride):
    #get poses from a .npz file

    RTposes = []
    T = []
    R = []
    Q = []

    worldmats = np.load(file)

    if "scale_mat_0" in list(worldmats.keys()):
        for n in range(0,int(len(worldmats) / 2)):
            ma = worldmats[f'world_mat_{n}'].copy()
            ma = ma[:3, :]
            K, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(ma)
            ma = np.eye(4)
            ma[:3, 3] = t[:3, 0] / t[3] * -1
            ma[:3, :3] = r
            scale_mat = worldmats[f'scale_mat_{n}'].copy()
            ma = np.linalg.inv(ma)
            wm = ma
            #wm = np.linalg.inv(wm)
            #wm = wm @ scale_mat
            
            #wm  = np.vstack((ma,[0,0,0,1]))
            #print(wm,'\n')

            if(n%stride != 0):
                continue

            RTposes.append(wm)
            R.append(wm[:3,:3])
            T.append(wm[:3,3])
    else:
        for n in range(0, len(worldmats)):
            ma = worldmats[f'world_mat_{n}'].copy()
            ma = ma[:3, :]
            K, r, t, _, _, _, _ = cv2.decomposeProjectionMatrix(ma)
            ma = np.eye(4)
            ma[:3, 3] = t[:3, 0] / t[3] * -1
            ma[:3, :3] = r

            if(n%stride != 0):
                continue

            wm = np.linalg.inv(ma)
            #wm = ma

            RTposes.append(wm)
            R.append(wm[:3,:3])
            T.append(wm[:3,3])

        
    #bp()
    
    T = np.array(T)
    R = np.array(R)
    Q = np.array(Q)
    RTposes = np.array(RTposes,dtype=np.float64)
    #RTposes = np.linalg.inv(RTposes)

    return(RTposes,R,T,Q)

def draw_app(pc_file='../pc.ply',pose_file='/home/felipe/Documents/DROID-SLAM/thirdparty/tartanair_tools_bak/seasonsforest_winter/Easy/seasonsforest_winter/seasonsforest_winter/Easy/P000/pose_left.txt' ):

    app = gui.Application.instance
    app.initialize()

    w = app.create_window("Turiruri", 1024, 768)
    widget3d = gui.SceneWidget()
    w.add_child(widget3d)
    widget3d.scene = o3d.visualization.rendering.Open3DScene(w.renderer)

    mat = o3d.visualization.rendering.MaterialRecord()

    img = o3d.io.read_image('fixed.png')
    widget3d.scene.set_background([0.4, 0.5, 0.6, 1], image=None)

    #widget3d.scene.add_geometry("img",img,mat)

    pcd = o3d.io.read_point_cloud(pc_file)

    CM = pcd.get_center()

    pcd.scale(10.,[0,0,0])
    pcd.translate(6*CM)


    mat.shader = "defaultUnlit"
    #widget3d.scene.add_geometry("PointCloud", pcd, mat)



    """
    cams = []
    for i in range(10):
        RTI =   [[  cos(i/100),  sin(i/100),  0,  0.5*cos(i/100)  ],
                [   -sin(i/100),  cos(i/100),  0,  0.5*sin(i/50)   ],
                [   0,  0,  1,  0.03*i          ],
                [   0,  0,  0,  1               ]]

        RTXY =  [[  cos(i/10),  sin(i/10),  0,  0.05*i],
                [   -sin(i/10),  cos(i/10),  0,  0  ],
                [   0,  0,  1,  0         ],
                [   0,  0,  0,  1               ]]


        pose = np.array(RTXY)
        cam = create_camera_actor(0.5)
        cam.transform(pose)
        cams.append(cam)
        widget3d.scene.add_geometry(f"Cam{i:02d}", cam,mat)
    #print('Camera attributes:')
    #print(dir(cams[0]))
    #print('\n\n')    

    # Add 3D labels
    for i in range(0, len(cams)):
        widget3d.add_3d_label(cams[i].points[0], str(i))
    """

    RT,R,T,Q = RT_from_poses(pose_file)
    #RT,R,T,Q = RT_from_poses('/home/felipe/Documents/DROID-SLAM/thirdparty/tartanair_tools_bak/amusement/Easy/amusement/amusement/Easy/P001/pose_left.txt',stride=stride)


    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(10.0,[0,0,0])

        
    for k,rt in enumerate(RT):
        #print(rt)

        #cam = create_camera_actor(k/len(RT))
        #cam.transform(RT[0])

        #cam.transform(rt)
        #widget3d.scene.add_geometry(f"Cam{k:03d}", cam,mat)
        #widget3d.add_3d_label(cam.points[0], str(k))

        #init = np.array((0,0,0))
        #init.transform(rt)

        m = copy.deepcopy(mesh)
        m.transform(rt)
        widget3d.scene.add_geometry(f"Cam{k:03d}", m, mat)
        widget3d.add_3d_label(m.vertices[0], str(k))



    bounds = widget3d.scene.bounding_box
    widget3d.setup_camera(60, bounds, bounds.get_center())

    app.run()

def animation_callback(vis):

    ctl = vis.get_view_control().convert_to_pinhole_camera_parameters()

    ctl = vis.get_view_control().convert_from_pinhole_camera_parameters(ctl)
         
def draw_vis_tartan(pc_file='cata.ply',pose_file='/home/felipe/Documents/DROID-SLAM/thirdparty/tartanair_tools_bak/seasonsforest_winter/Easy/seasonsforest_winter/seasonsforest_winter/Easy/P005/pose_left.txt' ):

    #vis = o3d.visualization.Visualizer()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    #vis.register_animation_callback(animation_callback)
    #vis.register_key_callback(ord("S"), increase_filter)
    #vis.register_key_callback(ord("A"), decrease_filter)
    #vis.register_key_callback(ord("E"), export_pointcloud)


    vis.create_window()
    ctl = vis.get_view_control()
    
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(1.5,[0,0,0])
    #T0 = [[0,0,-1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]
    T0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] # correct T0 for blender poses world mat
    cam =copy.deepcopy(mesh).transform(T0)

    #pointcloud = o3d.io.read_point_cloud(pc_file)#.voxel_down_sample(voxel_size=0.02)
    catamesh = o3d.io.read_triangle_mesh(pc_file)#.voxel_down_sample(voxel_size=0.02)
    catamesh.compute_vertex_normals()
    

    poses,R,T,Q = RT_from_poses(pose_file,stride=stride,ned=True)
    #poses,R,T,_ = RT_from_world_mats('/home/felipe/Documents/wizzdroid/data/Catalina/cameras.npz_FILES/',stride=stride)

    #vis.add_geometry(pointcloud)
    #vis.add_geometry(catamesh)
    vis.add_geometry(cam)
    """
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter(T[:,0],T[:,1],T[:,2])
    plt.show()
    """    

    mesh_box = o3d.geometry.TriangleMesh.create_box(width = 250.0, height = 1.0, depth = 250.0)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.2, 0.8, 0.5])
    mesh_box.translate((-20,10,5))
    vis.add_geometry(mesh_box)


    print(len(poses))

    cams = []
    reset = True

    for i,M in enumerate(poses):

        image = cv2.imread(f'{pose_file[:-14]}/image_left/{i*stride:06d}_left.png')
        cv2.imshow('image', image / 255.0)
        cv2.waitKey(1)
        
        #print(i)
        #cam =copy.deepcopy(mesh).transform(T0)
        cam.translate(T[i])
        cam.rotate(R[i])

        #print('\n',Quat(matrix=R[i]).x,'\n')


        fw,rt,up = R[i]@(0,0,1),R[i]@(1,0,0),R[i]@(0,1,0)

        ctl.set_up(-up)
        ctl.set_front(-fw)
        #ctl.set_right(rt)
        ctl.set_lookat(T[i])
        ctl.set_zoom(0.025)


        cams.append(copy.deepcopy(cam))
        vis.add_geometry(cams[-1], reset_bounding_box=reset)

        #mesh.transform(M)
        vis.update_geometry(cam)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.1)
        cam.translate(-T[i])
        cam.rotate(R[i].T)  

        #ctl.set_

        #ctl.rotate(85,5,5,5)
        # ? ? ? ? 

        if i==1:
            reset=False

        if(i==Nframes):
            pass
            #break     
        
        

    vis.run()

    vis.destroy_window()

def save_NED_traj(name,RT):
    with open(name,'w') as f:
        for rt in RT:

            print(rt[:3,:3].T - np.linalg.inv(rt[:3,:3]))
            q = Quat(matrix=rt[:3,:3],atol=1e-06,rtol=1e-06)
            qx, qy, qz, qw = q.x, q.y, q.z, q.w
            (x,y,z) = rt[:3,3]
            print(x,y,z)
            #f.write(f'{x} {y} {z} {qx} {qy} {qz} {qw}')
            f.write(f'{z} {x} {y} {qz} {qx} {qy} {qw}\n')

def draw_vis_cata(pc_file='cata.ply',pose_file='datasets/catalina/pose_left.txt' ):
    #vis = o3d.visualization.Visualizer()

    vis = o3d.visualization.VisualizerWithKeyCallback()
    #vis.register_animation_callback(animation_callback)
    #vis.register_key_callback(ord("S"), increase_filter)
    #vis.register_key_callback(ord("A"), decrease_filter)
    #vis.register_key_callback(ord("E"), export_pointcloud)

    vis.create_window()
    ctl = vis.get_view_control()

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh.scale(1.5,[0,0,0])
    #T0 = [[0,0,-1,0],[0,1,0,0],[-1,0,0,0],[0,0,0,1]]
    T0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] # correct T0 for blender poses world mat
    cam =copy.deepcopy(mesh).transform(T0)

    #pointcloud = o3d.io.read_point_cloud(pc_file)#.voxel_down_sample(voxel_size=0.02)
    catamesh = o3d.io.read_triangle_mesh(pc_file)#.voxel_down_sample(voxel_size=0.02)
    catamesh.compute_vertex_normals()
    

    #poses,R,T,Q = RT_from_poses(pose_file)
    poses,R,T,_ = RT_from_world_mats('/home/felipe/Documents/wizzdroid/data/Catalina/cameras.npz_FILES/',stride=stride)

    save_NED_traj('cataNED.txt',poses)

    #vis.add_geometry(pointcloud)
    vis.add_geometry(catamesh)
    vis.add_geometry(cam)

    print(len(poses),' poses found.\n')

    cams = []
    reset = True

    

    for i,M in enumerate(poses):

        image = cv2.imread(f'/home/felipe/Documents/wizzdroid/data/Catalina/lefts/{i*stride:04d}_L.png')
        cv2.imshow('image', image / 255.0)
        pkey = cv2.waitKey(1)

        if(pkey) == ord('q'):
            print('Quitting...\n')
            exit()

        #print(i)
        #cam =copy.deepcopy(mesh).transform(T0)
        cam.translate(T[i])
        cam.rotate(R[i])
        
        fw,rt,up = R[i]@(0,0,1),R[i]@(1,0,0),R[i]@(0,1,0)

        ctl.set_up(-up)
        ctl.set_front(-fw)
        #ctl.set_right(rt)
        ctl.set_lookat(T[i])
        ctl.set_zoom(0.25)


        cams.append(copy.deepcopy(cam))
        vis.add_geometry(cams[-1], reset_bounding_box=reset)

        #mesh.transform(M)
        vis.update_geometry(cam)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
        cam.translate(-T[i])
        cam.rotate(R[i].T)       
        
        if i==1:
            reset=False
            time.sleep(0.25)

        if(i==Nframes):
            pass
            #break   



    vis.run()

    vis.destroy_window()

def draw_trajectory(poses,R,T,ply):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    ctl = vis.get_view_control()

    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame().scale(1.5,[0,0,0])
    T0 = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]] # correct T0 for blender poses world mat
    cam =copy.deepcopy(mesh).transform(T0)
    vis.add_geometry(cam)

    if(ply):
        catamesh = o3d.io.read_triangle_mesh(ply)#.voxel_down_sample(voxel_size=0.02)
        catamesh.compute_vertex_normals()
        vis.add_geometry(catamesh)
    elif(not args.no_floor):
        mesh_box = o3d.geometry.TriangleMesh.create_box(width = 100.0, height = 1.0, depth = 100.0)
        mesh_box.compute_vertex_normals()
        mesh_box.paint_uniform_color([0.2, 0.8, 0.5])
        mesh_box.translate((-20,10,10))
        vis.add_geometry(mesh_box)
    
    cams = []
    reset = False
    zoom = 0.1

    if(args.imagedir):
        images = sorted(glob.glob(f'{args.imagedir}/*g'))

    for i,M in enumerate(poses[:Nframes]):
        time.sleep(delay)
        if(args.imagedir):
            image = cv2.imread(images[i])
        else:
            image = (i%255)*np.eye(10)
        cv2.imshow('image', image / 255.0)
        pkey = cv2.waitKey(1)

        if(pkey) == ord('q'):
            print('Quitting...\n')
            exit()
        if(pkey) == ord('r'):
            reset=not reset
        if(pkey) == ord('z'):
            zoom = (zoom+0.0025)%0.1+0.01

        cam.translate(T[i])
        cam.rotate(R[i])
        
        fw,rt,up = R[i]@(0,0,1),R[i]@(1,0,0),R[i]@(0,1,0)

        ctl.set_up(-up)
        ctl.set_front(-fw)
        #ctl.set_right(rt)
        ctl.set_lookat(T[i])
        ctl.set_zoom(zoom)


        cams.append(copy.deepcopy(cam))
        vis.add_geometry(cams[-1], reset_bounding_box=reset)

        #mesh.transform(M)
        vis.update_geometry(cam)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.01)
        cam.translate(-T[i])
        cam.rotate(R[i].T)       
        


    vis.run()

    vis.destroy_window()

   


if __name__ == '__main__':

    print(f'\n\nWelcome to Trajectory Visualizer!\n\nZ: change zoom\nR: Follow/unfollow camera\nQ: Quit\n\nPress enter to continue...')
   # _ = input()


    if(args.posefile[-4:] == '.txt'):
        print(f'Reading poses from text file {args.posefile}\n')
        poses,R,T,Q = RT_from_poses(args.posefile,stride=stride,ned=args.ned)

    elif(args.posefile[-4:] == '.npz'):
        print(f'Reading poses from numpy file {args.posefile}\n')
        poses,R,T,Q = RT_from_world_mats(args.posefile,stride=stride)
    else:
        print('Pose file extension not understood, please supply a .txt or .npz file')
        exit()

    print(len(poses),'poses found.\n')

    draw_trajectory(poses, R, T, args.ply)
