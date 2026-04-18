from content.Managers.DataManager import DataManager
import torchvision
from content.Managers.modelManager import ModelManager

if __name__ == "__main__":
    dm = DataManager(dataset=torchvision.datasets.MNIST, augment_training=True)
    model = ModelManager(model_name="test", network=...)
