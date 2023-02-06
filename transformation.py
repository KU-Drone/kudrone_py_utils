from __future__ import annotations
import typing
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped

class Transformation(object):
    def __init__(self, rotation: Rotation, translation: np.ndarray):
        assert isinstance(rotation, Rotation), f"type of rotation must be Rotation, not {type(rotation)}"
        assert isinstance(translation, np.ndarray), f"type of translation must be np.ndarray, not {type(translation)}"
        assert translation.shape == (3,1), f"translation must have shape (3,1), not {translation.shape}"
        self.rotation: Rotation = rotation
        self.translation: np.ndarray = translation

    @staticmethod
    def from_Odometry(odom:Odometry) -> Transformation:
        q_w = odom.pose.pose.orientation.w
        q_x = odom.pose.pose.orientation.x
        q_y = odom.pose.pose.orientation.y
        q_z = odom.pose.pose.orientation.z

        p_x = odom.pose.pose.position.x
        p_y = odom.pose.pose.position.y
        p_z = odom.pose.pose.position.z

        return Transformation(Rotation((q_x, q_y, q_z, q_w)), np.array((p_x, p_y, p_z)).reshape(3,1))

    @staticmethod
    def from_TransformStamped(transform:TransformStamped) -> Transformation:
        q_w = transform.transform.rotation.w
        q_x = transform.transform.rotation.x
        q_y = transform.transform.rotation.y
        q_z = transform.transform.rotation.z

        p_x = transform.transform.translation.x
        p_y = transform.transform.translation.y
        p_z = transform.transform.translation.z

        return Transformation(Rotation((q_x, q_y, q_z, q_w)), np.array((p_x, p_y, p_z)).reshape(3,1))


    @staticmethod
    def identity() -> Transformation:
        return Transformation(Rotation.identity(), np.zeros((3,1)))

    def inverse(self) -> Transformation:
        rot_inv = self.rotation.inv()
        return Transformation(rot_inv, -(rot_inv.apply(self.translation.reshape(3)).reshape(3,1)))
    
    def as_htm(self) -> np.ndarray:
        return np.vstack((np.hstack((self.rotation.as_matrix(), self.translation)),np.array((0,0,0,1))))

    def apply(self, vector: np.ndarray) -> np.ndarray:
        assert isinstance(vector, np.ndarray), f"type of vector must be np.ndarray, not {type(vector)}"
        assert vector.shape == (3,1), f"vector must have shape (3,1), not {vector.shape}"
        return self.rotation.apply(vector.reshape(3)).reshape(3,1) + self.translation

    def transform(self, other: Transformation) -> Transformation:
        assert isinstance(other, Transformation), f"type of other must be Transformation, not {type(other)}"
        return Transformation(self.rotation * other.rotation, self.apply(other.translation))

    def __matmul__(self, other: typing.Union(np.ndarray, Transformation)) -> Transformation:
        if isinstance(other, np.ndarray):
            return self.apply(other)
        elif isinstance(other, Transformation):
            return self.transform(other)
        else:
            raise TypeError(f"type of other must be np.ndarray or Transformation, not {type(other)}")


    def __str__(self):
        return f"{self.as_htm()}"

    @staticmethod
    def lerp(trans_1: Transformation, trans_2: Transformation, t: float) -> Transformation:
        assert isinstance(trans_1, Transformation), f"type of trans_1 must be Transformation, not {type(trans_1)}"
        assert isinstance(trans_2, Transformation), f"type of trans_2 must be Transformation, not {type(trans_2)}"

        slerp = Slerp([0,1], Rotation.concatenate((trans_1.rotation, trans_2.rotation)))
        return Transformation(slerp(t), (1-t)*trans_1.translation + t*trans_2.translation)

if __name__ == "__main__":
    trans = Transformation(Rotation.random(), np.random.rand(3,1))
    trans_other = Transformation(Rotation.random(), np.random.rand(3,1))
    vector = np.random.rand(3,1)
    print(np.linalg.inv(trans.as_htm()))
    print(trans.inverse().as_htm())

    print(trans.as_htm() @ trans_other.as_htm())
    print((trans@trans_other))

