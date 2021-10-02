from faceboard.face_board import FaceBoard

def test_board():
    fb = FaceBoard('10.128.128.88', '6789', "/data")
    assert fb.open_file("/data/cluster_rst/ijbc_align") == True