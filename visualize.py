import visualization.custom_viewer as cv
from aitviewer.scene.light import Light

if __name__ == "__main__":
    v = cv.CustomViewer()

    v.scene.camera.position = (-0.791, 0.475, 2.086)

    v.scene.floor.tiling = False

    v.add_meshes()

    v.run()
