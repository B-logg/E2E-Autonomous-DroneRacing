#!/usr/bin/env python3
"""Export the benchmark networks to ONNX (run once, anywhere torch is installed;
the .onnx files ship with the benchmark so the Jetson needs only TensorRT).

    python export_onnx.py            -> onnx/policy_step.onnx, onnx/gatenet_f*_HxW.onnx
"""
import pathlib

import torch

import models as M

OUT = pathlib.Path(__file__).resolve().parent / "onnx"


def main():
    OUT.mkdir(exist_ok=True)
    torch.manual_seed(0)
    pol = M.PolicyStep().eval()
    with torch.no_grad():
        a, d, s = pol(*M.PolicyStep.example_inputs())
    assert a.shape == (1, 4) and d.shape == (1, M.DETER) and s.shape == (1, M.STOCH * M.CLASSES)
    n = sum(p.numel() for p in pol.parameters())
    torch.onnx.export(pol, M.PolicyStep.example_inputs(), OUT / "policy_step.onnx", opset_version=17,
                      input_names=["mask", "vec", "deter", "stoch", "prev_action"],
                      output_names=["action", "deter_out", "stoch_out"], dynamo=False)
    print(f"policy_step.onnx  ({n / 1e6:.2f} M parameters)")
    for f, h, w in M.GATENET_CONFIGS:
        net = M.GateNet(f).eval()
        x = torch.rand(1, 3, h, w)
        with torch.no_grad():
            assert net(x).shape == (1, 1, h, w)
        name = M.gatenet_name(f, h, w)
        torch.onnx.export(net, (x,), OUT / f"{name}.onnx", opset_version=17,
                          input_names=["image"], output_names=["mask"], dynamo=False)
        print(f"{name}.onnx  ({sum(p.numel() for p in net.parameters()) / 1e6:.2f} M parameters)")


if __name__ == "__main__":
    main()
