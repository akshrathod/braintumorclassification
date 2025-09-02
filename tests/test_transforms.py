from app.utils import get_transforms
def test_transforms():
    tr, va = get_transforms(224, True), get_transforms(224, False)
    assert tr is not None and va is not None
