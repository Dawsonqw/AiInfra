"""Export torchvision's ResNet-18 into the reusable local ONNX asset."""

from pathlib import Path

import torch
from torchvision.models import resnet18


def main() -> None:
    output = Path(__file__).resolve().parents[1] / "asserts" / "resnet18.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    model = resnet18(weights=None).eval()
    example = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            output,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
            dynamo=False,
        )
    print(f"exported {output}")


if __name__ == "__main__":
    main()
