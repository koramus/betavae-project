import trimesh
import pyrender
from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
import os
import os.path

meshes_path = '/tmp/output'

scene = pyrender.Scene(ambient_light=(0.2, 0.2, 0.2))

camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
camera_node = pyrender.Node(camera=camera)
scene.add_node(camera_node)

light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=5.0)
light_node = pyrender.Node(light=light)
scene.add_node(light_node)

renderer = pyrender.OffscreenRenderer(viewport_width=128, viewport_height=128)

def init_scene(azimuth, elevation, light_azimuth):
    pose = np.eye(4)
    pose[0:3,0:3] = R.from_euler('xyz', [elevation, azimuth, 0], degrees=True).as_matrix()
    pose[0:3,0:3] *= 1e-5
    scene.set_pose(face_node, pose=pose)

    pose = np.eye(4)
    pose[2,3] = 2.8
    scene.set_pose(camera_node, pose=pose)

    pose = np.eye(4)
    pose[0:3,0:3] = R.from_euler('xyz', [0, light_azimuth, 0], degrees=True).as_matrix()
    scene.set_pose(light_node, pose=pose)


#face_trimesh = trimesh.load('/tmp/output/0_0.ply')
#face_mesh = pyrender.Mesh.from_trimesh(face_trimesh)
#face_node = pyrender.Node(mesh=face_mesh)
#scene.add_node(face_node)
#
#init_scene(0, 0, 0)
#
#pyrender.Viewer(scene)
#
#assert False


count = 0


images = []
factors = []
for root, dirs, files in os.walk(meshes_path):
    for name in files:
        path = os.path.join(root, name)

        face_trimesh = trimesh.load(path)
        face_mesh = pyrender.Mesh.from_trimesh(face_trimesh)
        face_node = pyrender.Node(mesh=face_mesh)
        scene.add_node(face_node)

        arg_strs = '.'.join(name.split('.')[:-1]).split('_')
        age = float(arg_strs[0])
        gender = float(arg_strs[1])

        for azimuth in range(-50, 50+1, 10):
            for elevation in range(-20, 20+1, 4):
                for light_azimuth in range(-90, 90+1, 18):
                    count += 1
                    print(count)
                    init_scene(azimuth, elevation, light_azimuth)

                    color, _ = renderer.render(scene)
                    im = Image.fromarray(color)
                    im = im.resize((64, 64), Image.BILINEAR)
                    im = im.convert('L')

                    array = np.asarray(im)
                    array = array / 255
                    array = np.expand_dims(array, 0)

                    images.append(array)
                    factors.append(np.array([azimuth, elevation, light_azimuth, age, gender]))

                    #im.save('/tmp/output2/{}_{}_{}_{}_{}.png'.format(age, gender, azimuth, elevation, light_azimuth))

        scene.remove_node(face_node)



images = np.stack(images)
factors = np.stack(factors)

np.savez(os.path.join('.', 'faces-labelled'), images=images, factors=factors)
