import numpy as np

from ml_tools.tools import Rectangle


class TestRectangle:
    def test_can_create_rectangle_from_width_and_height(self):
        rectangle = Rectangle(2, 3, 5, 6)
        assert_rectangle_values(rectangle)

    def test_crop(self):
        rectangle = Rectangle(0, 0, 100, 100)
        rectangle.crop(Rectangle(2, 3, 5, 6))

        assert_rectangle_values(rectangle)

    def test_subimage(self):
        image = np.arange(100).reshape((10, 10))
        rectangle = Rectangle(2, 3, 2, 3)

        subimage = rectangle.subimage(image)
        assert np.array_equal(subimage, [[32, 33], [42, 43], [52, 53]])


def assert_rectangle_values(rect):
    assert rect.left == 2
    assert rect.top == 3
    assert rect.width == 5
    assert rect.height == 6
