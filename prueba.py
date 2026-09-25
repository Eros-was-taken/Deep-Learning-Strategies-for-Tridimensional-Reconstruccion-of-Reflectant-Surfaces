import open3d as o3d

mesh = o3d.io.read_triangle_mesh("data/bunny.ply")
print(mesh)
o3d.visualization.draw_geometries([mesh])