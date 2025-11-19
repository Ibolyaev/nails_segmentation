import torch
import coremltools as ct
from src.helper.decoders.deeplabv3 import DeepLabV3Plus  # импорт твоей архитектуры из репозитория

# Загружаем обученную PyTorch модель
model = DeepLabV3Plus()
model.load_state_dict(torch.load("model/best_model.pth", map_location="cpu"))
model.eval()

# Создаем пример входного тензора (размер такой же, как при обучении)
example_input = torch.rand(1, 3, 256, 256)  # batch=1, RGB, HxW=256x256

# Трейсинг модели для Core ML
traced_model = torch.jit.trace(model, example_input)

# Конвертация в Core ML
mlmodel = ct.convert(
    traced_model,
    inputs=[ct.ImageType(name="input_image", shape=example_input.shape, scale=1/255.0, bias=[0,0,0])],
    minimum_deployment_target=ct.target.iOS17  # можно указать iOS16/17
)

# Сохраняем Core ML модель
mlmodel.save("NailSegmentation.mlmodel")
print("Модель сохранена как NailSegmentation.mlmodel")
