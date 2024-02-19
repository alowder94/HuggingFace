# Transformers and HuggingFace Platform Notes

## Transformers
### General Intro
* Transformer models basically take information and turn it into something else in some type of way. 
* GPT, BART, BERT, and T5 are all examples of transformer models. I would assume that LLAMA is also likely a transformer model
* Grouped into 3 categories
    - GPT-Like -- also known as auto-regressive transformer models
    - BERT-Like -- also known as auto-encoding transformer models
    - BART/T5 Like -- also known as sequence-to-sequence transformer models
* These models are generic and not very useful for any specific task
    - This is becuase they are _self-supervised_ - meaning that they derive the objective of the data used to train them based off of that dataset itsself without any human intervention. 
    - From here, the generic models are fine tuned by humans for whatever purpose is required of the usecase -- this is done in a supervised way through a process called _transfer-learning_ \

### Parts of a Transformer 
* Encoder
    - Encoder recieves input and builds a representation of that inputs features. This means that the model is optimized to understand what is important in the input
    
* Decoder 
    - Decoder recieves encoded input (and potentially other inputs) and generates what is called a "target sequence" -- think answer. This means this model is optimized to generate outputs

* Models can be either a Encoder, Decoder, or both
    - Encoder is good at things like classification etc (things that require pulling relevant information out of given text)
    - Decoder is good at things like text generation
    - Encoder-Decoder (or sequence-to-sequence models) are good for anything that requires encoding some kind of input and generating something based off of that input (think something like Chat GPT)

### Encoders
* At each stage in an encoder model, the attention layer can access the entire input sentence. 
* These models are characterized as having "bi-directional attention", and are often referenced as _auto-encoding models_
* These types of models are best suited for tasks that require an understanding of the entire sentence
    - Sentence Classification
    - Named Entity Recognition
    - Extractive Question Answering
* Examples: BERT, ALBERT, DistilBERT, ELECTRA, RoBERTa

### Decoders
* At each stage of a decoder model, the model is only able to access the words it has already decoded
* These models are often called _auto-regressive models_
* Pretraining of these models typically involves trying to predict the next word of a provided sentence
* Examples: CTRL, GPT/GPT2, Transformer XL

### Sequence-to-Sequence
* Sequence-to-Sequence models are essentially a combination of encoders feeding into a decoder. Encoder encodes the model into something the decoder can effectively (and accurately) generate some kind of target sequence from
* Each respective component of the works exactly as they would on their own, but in a cohesive manor
* Training is usually similar to what an encoder or decoder would see, but more complex. For exmaple, T5 was trained by replacing a random span of text in a string with a special "mask" word with the objective of predicting the initail text the mask word replaced
* Examples: BART, mBART, Marian, T5

## Using HuggingFace's Transformer API
* The pipeline function is designed to tokenize, process through the model, then pass to the post-processer to turn the output back into something human readable. 
* You don't _have_ to do it this way, you can manually define the steps yourself. You will first need to tokenize - likely using the AutoTokenizer module from Transformers library. 
    - NOTE: The tokenizer __must__ encode using the exact same algorithm that was used when training the model that the tokens will be passed to (otherwise the model won't properly function). This means if you are planning to do this in some custom way, you have to have access to the "checkpoints" from the model you are using
* Similarly, you will grab the model from HuggingFace using the AutoModel module from Transformers
    - The output you are going to get from a model created like this is going to be a _hidden-state_, or a high dimensional vector representing the contextual understanding of the input by the transformer model
    - In practice this will probably actually be AutoModelFor${something you are trying to do}...this provides the head that is able to understand what is coming from the tokenizer
        - Some examples of AutoModels/AutoTokenizers with a specified purpose [here](https://huggingface.co/learn/nlp-course/chapter2/2?fw=pt#:~:text=a%20specific%20task.-,Here,-is%20a%20non)
    
## Tokenizers
* Basically used to encode text into numeric values to be interpreted by the model
* Two step process
1. "Tokenize" or break the sequence into
    * Done either by whole words, single characters, or more commonly __subwords__ (think something like breaking up "commonly" into "common" and "ly" to allow the machine to determine a relationship between common and commonly, as well as develop an understanding of the "ly" word ending)
    * Python code for this - tokenizer.tokenize(sequence)
2. Encode the tokens into "IDs"
    * This envolvs the actual conversion of the string values into some kind of numeric value for the model to interpret. 
    * Python code for this - tokenizer.convert_tokens_to_ids(tokens)

* Decoding is the exact opposite - but just as important. This is the act of decoding the ids returned from the model, and converting those decoded ids into a coherent, readable string.
    * tokenizer.decode(value from model)

### Batching Inputs
* A model is expecting a batch of inputs - not a single input.
* If you manually tokenize and convert to ids, then try to pass the result to the model, your code will fail. This is because of what is stated above, you are trying to pass a single value to a function that is expecting to perform some kind of batch processing
* You can also "pad" the input using a predetermined value (for whatever specific model you are using) - basically if you have an array with 1 3-element value and another 2-element value --  this cannot be cast to a tensor because it is not rectangular in shape. You can add the padding value to the second set of elements and end up with something that has the shape of (2, 3), which can be cast to a tensor for the model to process.
    * padding values can be found by calling tokenizer.pad_token_id
### Attention Masks







