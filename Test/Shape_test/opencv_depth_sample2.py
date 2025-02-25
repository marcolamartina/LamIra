import cv2
import numpy as np
import pcl


# base code
# https://qiita.com/SatoshiGachiFujimoto/items/eb3891116d4f49cd342d
# https://github.com/SatoshiRobatoFujimoto/PointCloudViz
# https://stackoverflow.com/questions/21849512/how-to-align-rgb-and-depth-image-of-kinect-in-opencv/21851642
# https://stackoverflow.com/questions/29270544/how-to-display-a-3d-image-when-we-have-depth-and-rgb-mats-in-opencv-captured-f
# https://github.com/IntelRealSense/librealsense/issues/2090
# https://github.com/daavoo/pyntcloud
def cvtDepth2Cloud(depth, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]
    # print(inv_fx)
    # print(inv_fy)
    # print(ox)
    # print(oy)

    # print(depth.size)
    rows, cols = depth.shape
    # print(cols)
    # print(rows)
    cloud = np.zeros((depth.size, 3), dtype=np.float32)
    # print(cloud)
    for y in range(rows):
        for x in range(cols):
            # print(x)
            # print(y)
            x1 = float(x)
            y1 = float(y)
            # print(x1)
            # print(y1)
            dist = depth[y][x]
            # print(dist)
            # print(cloud[y * cols + x][0])
            cloud[y * cols + x][0] = np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)

    # cloud = []
    # for v in range(height):
    #     for u in range(width):
    #         offset = (v * width + u) * 2
    #         depth = ord(array[offset]) + ord(array[offset+1]) * 256
    #         x = (u - CX) * depth * UNIT_SCALING / FX
    #         y = (v - CY) * depth * UNIT_SCALING / FY
    #         z = depth * UNIT_SCALING
    #         cloud.append((x, y, z))

    return cloud


def cvtDepthColor2Cloud(depth, color, cameraMatrix):
    inv_fx = 1.0 / cameraMatrix[0, 0]
    inv_fy = 1.0 / cameraMatrix[1, 1]
    ox = cameraMatrix[0, 2]
    oy = cameraMatrix[1, 2]
    # print(inv_fx)
    # print(inv_fy)
    # print(ox)
    # print(oy)

    # print(depth.size)
    rows, cols = depth.shape
    # print(cols)
    # print(rows)
    cloud = np.zeros((depth.size, 4), dtype=np.float32)
    # print(cloud)
    for y in range(rows):
        for x in range(cols):
            # print(x)
            # print(y)
            x1 = float(x)
            y1 = float(y)
            # print(x1)
            # print(y1)
            dist = -depth[y][x]
            # print(dist)
            # print(cloud[y * cols + x][0])
            cloud[y * cols + x][0] = -np.float32((x1 - ox) * dist * inv_fx)
            cloud[y * cols + x][1] = np.float32((y1 - oy) * dist * inv_fy)
            cloud[y * cols + x][2] = np.float32(dist)
            red = color[y][x][2]
            green = color[y][x][1]
            blue = color[y][x][0]
            rgb = np.left_shift(red, 16) + np.left_shift(green,
                                                         8) + np.left_shift(blue, 0)
            cloud[y * cols + x][3] = rgb

    return cloud


def main():
    # | fx  0   cx |
    # | 0   fy  cy |
    # | 0   0   1  |
    # vals = np.array(
    #     [525., 0.  , 319.5,
    #      0.  , 525., 239.5,
    #      0.  , 0.  , 1.])
    # cameraMatrix = vals.reshape((3, 3))
    # grabber_sequences/pclzf/*.xml
    cameraMatrix = np.array(
        [[525., 0., 320.0],
         [0., 525., 240.0],
         [0., 0., 1.]])

    # color0 = cv2.imread('rgb/0.png')
    color0 = cv2.imread('/home/davide/python-pcl/examples/external/opencv/grabber_sequences/tiff/frame_20121214T142255.814212_rgb.tiff')
    # 16 bit Image
    # https://github.com/eiichiromomma/CVMLAB/wiki/OpenCV-16bitImage
    # depth0 = cv2.imread('depth/0.png', -1)
    # depth0 = cv2.imread('depth/0.png', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    #depth1 = cv2.imread('/home/davide/python-pcl/examples/external/opencv/grabber_sequences/tiff/frame_20121214T142255.814212_depth.tiff', cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR)
    depth1 = cv2.imread('/home/davide/Desktop/pclpy-master/pclpy/tests/test_data/depth.png', 0)

    #print("Color: ", color0.dtype)
    #print("Depth: ", depth0.dtype)

    # colorImage1 = cv2.imread('rgb/1.png')
    # depth1 = cv2.imread('depth/1.png', -1)

    # if (color0.empty() || depth0.empty() || colorImage1.empty() || depth1.empty()):
    #     cout << "Data (rgb or depth images) is empty.";
    #     return -1;

    # gray0 = cv2.cvtColor(color0, cv2.COLOR_BGR2GRAY)
    # grayImage1 = cv2.cvtColor(colorImage1, cv2.COLOR_BGR2GRAY)
    # depthFlt0 = depth0.convertTo(cv2.CV_32FC1, 1. / 1000.0)
    # depthFlt1 = depth1.convertTo(cv2.CV_32FC1, 1. / 1000.0)
    #depthFlt0 = np.float32(depth0) / 1000.0
    # depthFlt0 = depth0 / 1000.0
    depthFlt1 = np.float32(depth1) / 1000.0

    import pcl
    # points0 = cvtDepth2Cloud(depthFlt0, cameraMatrix)
    # cloud0 = pcl.PointCloud()
    # points0 = cvtDepthColor2Cloud(depthFlt0, color0, cameraMatrix)
    # cloud0 = pcl.PointCloud_PointXYZRGBA()
    # cloud0.from_array(points0)
    # print(cloud0)

    points1 = cvtDepth2Cloud(depthFlt1, cameraMatrix)
    cloud1 = pcl.PointCloud()
    cloud1.from_array(points1)
    

    ne=cloud1.make_NormalEstimation()
    tree = cloud1.make_kdtree()
    ne.set_SearchMethod(tree)
    ne.set_RadiusSearch(0.01)

    print("start compute normal")
    normals = ne.compute()
    print("end compute normal")
    #print(dir(normals))
    #print(normals.size)
    #print(normals.get_point())
    normals_point = normals.to_array()
    #print(points1.shape)
    #print(normals_point.shape)
    #print(normals)

    for i in range(0, 10):#range(0, normals.size):
        print ('normal_x: ' + str(normals_point[i][0]) + ', normal_y : ' + str(normals_point[i][1]) + ', normal_z : ' + str(normals_point[i][2]))



    # cloud0._to_ply_file(b"/home/davide/Desktop/object_discovery_release/scenes/scene0101.ply", binary=True)
    cloud1._to_ply_file(b"/home/davide/Desktop/object_discovery_release/scenes/scene0102.ply", binary=True)
    
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    params= points1[:,0], points1[:,1], points1[:,2], normals_point[:, 0], normals_point[:, 1], normals_point[:, 2]
    result=[np.array([e for i,e in enumerate(p) if i%1000==0]) for p in params]

    X, Y, Z, U, V, W=result

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W,length=0.01)
    ax.set_xlim([np.min(X), np.max(X)])
    ax.set_ylim([np.min(Y), np.max(Y)])
    ax.set_zlim([np.min(Z), np.max(Z)])


    # wait
    try:
        import pcl.pcl_visualization
        visual = pcl.pcl_visualization.CloudViewing()
        # xyz only
        # visual.ShowMonochromeCloud(cloud0)
        # visual.ShowMonochromeCloud(cloud1)
        # color(rgba)
        # visual.ShowColorACloud(cloud0)
        visual.ShowMonochromeCloud(cloud1)
        plt.show()
        
        v = True
        while v:
            v = not(visual.WasStopped())
    except Exception as e:
        print(e)
        pass


if __name__ == "__main__":
    # import cProfile
    # cProfile.run('main()', sort='time')
    main()
