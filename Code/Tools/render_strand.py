import numpy as np
import pyrender
from skimage import transform
import cv2
def render_strand(strands,mesh=None,mask=False,intensity=3.0, strand_color = None, offscreen = True,cam_pos=[]):
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    if mesh:
        mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    # tri_geometry = mesh.geometry
    # for i,t in enumerate(tri_geometry):
    #     mesh = tri_geometry[t]
    #     mesh.visual = mesh.visual.to_color()
    #     mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
    #     mesh = pyrender.Mesh.from_trimesh(tri_geometry[t])
    #     scene.add(mesh)
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=int(640 / 1.0), point_size=1.0)
    lines = []
    root = []
    for i,strand in enumerate(strands):
        if isinstance(strand_color,np.ndarray):
            line = pyrender.Primitive(strand,color_0=strand_color[i],mode=3)
        else:
            line = pyrender.Primitive(strand,color_0=(0,0,0,255),mode=3)
        lines.append(line)
        root.append(strand[0])
        
    m_line = pyrender.Mesh(lines, name="strands", is_visible=True)
    scene.add(m_line)
    aspectRatio = 1.0
    bOrtho = False
    if bOrtho:
        xymag = 0.30
        pc = pyrender.OrthographicCamera(xymag, xymag, zfar=200)
    else:
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 15, aspectRatio=aspectRatio)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = camera_pose = transform.SimilarityTransform(translation=(0, 1.6, -5), rotation=(np.deg2rad(180),np.deg2rad(0),0), dimensionality=3)
    if not mask:
        scene.add(light, light_pose)
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        # light_pose = np.array([
        # [1.0, 0.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, -5.0],
        # [0.0, 0.0, 0.0, 1.0],
        # ])
        # scene.add(light1, light_pose)
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.600],
        [0.0, 0.0, 1.0, 5.00],
        [0.0, 0.0, 0.0, 1.0],
    ])
    camera_pose = transform.SimilarityTransform(translation=(0, 1.6, 5), rotation=(np.deg2rad(0),0,0), dimensionality=3)
    scene.add(pc, name="main_cam", pose=camera_pose.params)
    if not offscreen:
        # scene.main_camera_node.matrix = np.dot(move, camera_pose)
        pyrender.Viewer(scene)
    else:
        colors = []
        color, depth = r.render(scene, flags=flags)
        colors.append(color)
        # cv2.imshow("1",color)
        # cv2.waitKey()
        for i, move in enumerate(cam_pos):
            scene.main_camera_node.matrix = move
            color, depth = r.render(scene, flags=flags)
            colors.append(color)
            # cv2.imshow("1",color)
            # cv2.waitKey()
    r.delete()
    # cv2.imshow("1",color)
    # cv2.waitKey()
    return colors
def render_seg_strand(segments,strands,mesh=None,mask=False,intensity=3.0, strand_color = None, offscreen = True,cam_pos=[]):
    scene = pyrender.Scene(ambient_light=[0.1, 0.1, 0.1])
    if mesh:
        mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)
    # tri_geometry = mesh.geometry
    # for i,t in enumerate(tri_geometry):
    #     mesh = tri_geometry[t]
    #     mesh.visual = mesh.visual.to_color()
    #     mesh.visual.vertex_colors = np.array([255, 255, 255, 255])
    #     mesh = pyrender.Mesh.from_trimesh(tri_geometry[t])
    #     scene.add(mesh)
    
    lines = []
    root = []
    start = 0
    for seg in segments:
        if isinstance(strand_color,np.ndarray):
            line = pyrender.Primitive(strands[start:start+seg],color_0=strand_color[i],mode=3)
        else:
            line = pyrender.Primitive(strands[start:start+seg],color_0=(0,0,0,255),mode=3)
        lines.append(line)
        root.append(strands[start])
        start+=seg
        
    m_line = pyrender.Mesh(lines, name="strands", is_visible=True)
    scene.add(m_line)
    flags = pyrender.RenderFlags.RGBA | pyrender.RenderFlags.ALL_SOLID | pyrender.RenderFlags.SKIP_CULL_FACES
    r = pyrender.OffscreenRenderer(viewport_width=640, viewport_height=int(640 / 1.0), point_size=1.0)
    aspectRatio = 1.0
    bOrtho = False
    if bOrtho:
        xymag = 0.30
        pc = pyrender.OrthographicCamera(xymag, xymag, zfar=200)
    else:
        pc = pyrender.PerspectiveCamera(yfov=np.pi / 15, aspectRatio=aspectRatio)
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
    light_pose = camera_pose = transform.SimilarityTransform(translation=(0, 1.6, -5), rotation=(np.deg2rad(180),np.deg2rad(0),0), dimensionality=3)
    if not mask:
        scene.add(light, light_pose)
        light1 = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity)
        # light_pose = np.array([
        # [1.0, 0.0, 0.0, 0.0],
        # [0.0, 1.0, 0.0, 0.0],
        # [0.0, 0.0, 1.0, -5.0],
        # [0.0, 0.0, 0.0, 1.0],
        # ])
        # scene.add(light1, light_pose)
    
    camera_pose = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 1.600],
        [0.0, 0.0, 1.0, 5.00],
        [0.0, 0.0, 0.0, 1.0],
    ])
    camera_pose = transform.SimilarityTransform(translation=(0, 1.6, 5), rotation=(np.deg2rad(0),0,0), dimensionality=3)
    scene.add(pc, name="main_cam", pose=camera_pose.params)
    if not offscreen:
        # scene.main_camera_node.matrix = np.dot(move, camera_pose)
        pyrender.Viewer(scene)
    else:
        colors = []
        color, depth = r.render(scene, flags=flags)
        colors.append(color)
        # cv2.imshow("1",color)
        # cv2.waitKey()
        for i, move in enumerate(cam_pos):
            scene.main_camera_node.matrix = move
            color, depth = r.render(scene, flags=flags)
            colors.append(color)
            # cv2.imshow("1",color)
            # cv2.waitKey()
    r.delete()
    # cv2.imshow("1",color)
    # cv2.waitKey()
    return colors