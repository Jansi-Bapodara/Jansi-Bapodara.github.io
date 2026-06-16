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

the gpt-2 model used for experiment contained 12 transformer block with each transformer block containing 12 attention head which each token represented by 768 numbers. After that I performed a controlled experiment with the experimental condition being `A cinematic shot of a red car` and the controlled statement being `A cinematic shot of a blue car`.
The model activations were collected using `logits, cache = model.run_with_cache(tokens)` This allowed me to get to 
1. residual stream activations
2. attention patterns
3. intermediate laye outputs
 


## Methodolody
I created two almost idential promts with the change made only in the colour of the car. The prompts were  `A cinematic shot of a red car`  and  `A cinematic shot of a *blue* car`. I only changed the colour and kept everything else the same so that I compare and then infer that if the change occurs it is due to the token of colour. Since the computer cannot read text so use use `tokens_red = model.to_tokens(prompt_red)` to convert it to tokens.Thus the experiment was conducted to see the difference between the prompt of red car vs blue car. Then I applied activation caching this allowed me get the prediction of the next token along with all the internal activations. In simpler terms it showed the model's memory while thinking. I then moved on to inspect the attention.
```python
layer = 10
attention = cache_red[
    f"blocks.{layer}.attn.hook_pattern"
]
```
I first set the layer to 10 to inspect the attention module(`.attn`) of transformer layer 10. `.hook_pattern`  gave me the attention matrix and thus I was able to take a look at how the tokens were interdependent on each other. 
`attention.shape` gives <mark>(batch,heads,query,key)</mark>.Thus, I tested every head and printed the scores corresponding to each head using 
```python
for head in range(model.cfg.n_heads):

    score = attention[
        0,
        head,
        6,
        5
    ]

    print(
        "Head",
        head,
        "score:",
        score.item()
    )
```
I checked the residual stream for layer 11 for the prompts of red and blue car respectively.(This was the final internal  representation).I took the difference between red and blue to find what changes take place in the representation of car when there is a change in colour from red to blue. I only took the absolute of the difference because sign does not matter we are only concerned with the magnitude as in how big was the change.

```python
torch.topk(values,k=20)
print(top.indices)
```
this helped me to locate which indices changed by a huge value and these dimensiond encoded colour. 
I then proceeded to activation patching.Using cache i saved the internal thought of the red car in layer 10. I attached a hook at layer 10 using patch residual so when the model reaches layer 10 it pauses and gives the activation

Using the command 
```python
activation[:] = red_activation
```
I am forcing the model to take *A cinematic shot of a blue car* as input where as the layer 10 internally has the information stored of *A cinematic shot of a red car*. Thus from this point onwards the model has to move forward with the modified activation.

After that I ran the code normally without any intervention. This gave me the result for the thinking of the model of blue car running normally and the thinking of the patched layer containg blue car input and layer 10 of red car's thinking. I then compared the predictions to give the probabality of the the last token for the first sentence.

## Result

## Visualization
## Key Finding
## Conclusion
