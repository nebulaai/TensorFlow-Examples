# Orion sample AI task - single GPU - pytorch #


# Model Description #

This is a testing code base on pytorch framework. It is a seq2seq translation model and attention mechanism was applied.
The model trains a neural network to translate from French to English.

- Load data from _data/eng-fra.txt_, which includes English-French language translation pairs.
- Pre-process language pairs, and train a Seq2seq model based on Recurrent Neural Network.
- Evaluate translation for random sentences.

# Results #

Results of this model include:

- Training and evaluation result log.

E.g. 
```
[KEY: > input, = target, < output]

> nous sommes armees .
= we re armed .
< we re ruined . <EOS>

> je suis gras .
= i m fat .
< i m fat . <EOS>

> il est tres courageux .
= he is very brave .
< he is very brave . <EOS>

> nous regrettons de ne pas pouvoir vous aider .
= we re sorry we can t help you .
< we re sorry we can t help you . <EOS>

> je suis un tel idiot .
= i m such a fool .
< i m a little fool . <EOS>

> il est tres deprime .
= he is very depressed .
< he is very depressed . <EOS>

> je garde ce siege pour tom .
= i m saving this seat for tom .
< i m doing this for tom . <EOS>

```

# Remarks # 

- The entire task-processing time on Orion platform takes around 20 minutes on single 1080Ti GPU.
- Nebula AI working node will save all results in the directory _Result_, users will be able to retrieve contents from result directory, together with system execution log.
- For more details on how-to-guides for task submission, please refer to instructions on Nebula AI developer portal. 
