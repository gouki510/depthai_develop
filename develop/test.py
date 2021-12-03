from multiprocessing import Process
from depthai_demo import run_all
from road_segmentation import run_all1

if __name__ == "__main__":
    p1 = Process(run_all())
    print("h")
    p2 = Process(run_all1())
    p1.start()
    p2.start()
    p1.join()
    p2.join()
