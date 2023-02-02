from __future__ import annotations
import typing
import numpy as np
from scipy.spatial.transform import Rotation 

class Transformation(object):
    def __init__(self, rotation: Rotation, translation: np.ndarray):
        assert isinstance(rotation, Rotation), f"type of rotation must be Rotation, not {type(rotation)}"
        assert isinstance(translation, np.ndarray), f"type of translation must be np.ndarray, not {type(translation)}"
        assert translation.shape == (3,1), f"translation must have shape (3,1), not {translation.shape}"
        self.rotation: Rotation = rotation
        self.translation: np.ndarray = translation

    @staticmethod
    def identity() -> Transformation:
        return Transformation(np.zeros(3,1), Rotation.identity())

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
        return Transformation(self.rotation * other.rotation, self.rotation.apply(other.translation.reshape(3)).reshape(3,1) + self.translation)

    def __matmul__(self, other: typing.Union(np.ndarray, Transformation)) -> Transformation:
        if isinstance(other, np.ndarray):
            return self.apply(other)
        elif isinstance(other, Transformation):
            return self.transform(other)
        else:
            raise TypeError(f"type of other must be np.ndarray or Transformation, not {type(other)}")


    def __str__(self):
        return f"{self.as_htm()}"

if __name__ == "__main__":
    trans = Transformation(Rotation.random(), np.random.rand(3,1))
    vector = np.random.rand(3,1)
    print(np.linalg.inv(trans.as_htm()))
    print(trans.inverse().as_htm())
