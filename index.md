Hello Everyone 
welcome to my Github website

# Understanding Text To Image Prompt Generator

## Introduction
We have often used text to image generator where we have already seen how prompt improves the precison of the geneated image.
Therefore we are already aware of how our words and description are affecting the generated image. Nevertheless these are still treated like black boxes where the logic and explanation behind it remains hidden, ignored and even forgotten. With the task given to me I have tried to explore and understand the text to image generator with the means of transformer lens.

The goal of this task was to find out the role of image to text generator and why is there even a need to look behind the hood.
I have tried to understand how the GPT_2 backbone inside the prompt generator represents and processes visual modifiers suc as colours, objects and styles.

## Background


## Experiment Setup
I started with importing transformer-lens, torch and matplotlib. Also I imported HookedTransformer. this is essential so that we may be able to attach Hooks. they are the points which help to inspect and look into the internal computation which further contributes in unravelling the black box.

```python
model = HookedTransformer.from_pretrained(
    "gpt2"
)
```
this snippet of code helps to convert the gpt-2 model into a transformer lens model.

```python
model.cfg
```
is used to check the configuration. It helped me gain and extract information like 
- How big is the model?
- how many layers it has?
- How many heads?
- how many neurons?

the gpt-2 model used for experiment contained 12 transformer block with each transformer block containing 12 attention head which each token represented by 768 numbers. After that I performed a controlled experiment with the experimental condition being `A cinematic shot of a red car` and the controlled statement being `A cinematic shot of a blue car`. I only changed the colour and kept everything else the same so that I compare and then infer that if the change occurs it is due to the token of colour. Since the computer cannot read text so use use `tokens_red = model.to_tokens(prompt_red)` to convert it to tokens.

## Methodolody
## Result
## Visualization
## Key Finding
## Conclusion
