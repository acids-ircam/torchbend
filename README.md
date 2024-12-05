# torchbend

`torchbend` is a library grounded on `torch.fx` focused on generative neural networks analysis and creative bending. This library allows you to:

- [✔︎] extend the tracing abilities of `torch.fx` with augmented parsers and proxies
    - dynamic parsing (wrapping un-traceable functions, shape propagation)
    - tracing torch distributions (currently implemented : `Bernoulli`, `Normal`, `Categorical`)
- [✔︎] easily parse and analyze model's graphs 
- [✕︎] bend model's weights and activations
- [✕︎] adapt the library to specific generative models, and provide handy interfaces for python notebooks
    - [✕︎] handful classes for image, text, and sound
    - [✕︎] panel implementation for real-time bending
    - [✕︎] model analysis UI
- [✕︎] script generative models with JIT additional bending inputs (for use in [nn~] for example)

`torchbend` provides end-to-end examples for the following libraries:
- **Audio**
    - vschaos2
    - RAVE
    - audiocraft
- **Image**
    - StyleGAN3
    - StableDiffusion
- **Text**
    - Llama

## Parse and analyse model's graphs

```python
import torch, torchbend

# make dumb module to test
module = torchbend.TestModule()
module_in = torch.randn(1, 1, 512)

# init BendedModule with the module, and trace target functions with given inputs
bended_module = torchbend.BendedModule(module)
bended_module.trace("forward", x=module_in)

# print weights and activations
print("weights : ")
bended_module.print_weights()
print("activations : ")
bended_module.print_activations()

outs = bended_module.get_activations('pre_conv', x=module_in)
print("pre_conv activation min and max : ", outs['pre_conv'].min(), outs['pre_conv'].max())
```


## Bending weights and activations

```python
import torch, torchbend

# make dumb module to test
module = torchbend.TestModule()
module_in = torch.randn(1, 1, 512)

# init BendedModule with the module, and trace target functions with given inputs
bended_module = torchbend.BendedModule(module)
bended_module.trace("forward", x=module_in)

# bend target weights and make forward pass
bended_module.bend(torchbend.bending.Mask(0.), "pre_conv.weight")
outs = bended_module(x=module_in)
print("pre_conv bended weight std : ", bended_module.bended_state_dict()['pre_conv.weight'].std())
print("pre_conv original weight std : ", bended_module.module.state_dict()['pre_conv.weight'].std())

# reset bending
bended_module.reset()

# bend target activation 
bended_module.bend(torchbend.bending.Mask(0.), "pre_conv")
outs = bended_module.get_activations('pre_conv', x=module_in, bended=True)
print("pre_conv activation min and max : ", outs['pre_conv'].min(), outs['pre_conv'].max())
```
