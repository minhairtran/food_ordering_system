# Contents
- [Introduction](#introduction)
- [Constructing KWS models](#constructing-kws-models)
- [Contructing system detecting keyword directly](#contructing-system-detecting-keyword-directly)
- [Constructing application of KWS in food ording in restaurants](#constructing-application-of-kws-in-food-ording-in-restaurants)
- [Instruction guides](#instruction-guides)
- [Summary and future development](#summary-and-future-development)

# Introduction
Application of keyword spotting system in food ordering in restaurants is an open source, aiming at  reducing the interaction between serveurs and customers. Restaurants also have number of serveurs in restaurants reduce when using the application, which reduces the operation budgets. On creating this application, I hope to help restaurants in Vietnam overcome the difficult time in COVID period.

In this application, I solved the following problems:
- Building Vietnamese dataset based on text to speech of [FPT.AI](https://fpt.ai/) and [SpecAugment method](https://arxiv.org/abs/1904.08779). The dataset in other languages can also be generated with the same approach by using text to speech from [Google Cloud](https://cloud.google.com/text-to-speech) or other providers.
- Building models detecting keywords based on [attention-based method](https://arxiv.org/pdf/1803.10916.pdf). In general, the precision in test dataset reaches more than 99%
- Building a cheap method to differentiate human voice from restaurant environment noise and therefore, building a system for detecting keyword from human voice directly with precision of 81.25% and 91.67% for the 2 models built.
- Bulding a simple user interface for serveurs in restaurants to manage the application.

# Constructing KWS models
In this module, I explain in detail about how the model was built, how the dataset was generated and related results.

The image below is the diagram of the model based on attention-based. With the view to understanding the diagram, you should have basic knowledge about:
- Audio signal preprocessing. I recommend [this video list](https://www.youtube.com/watch?v=iCwMQJnKk2c&list=PL-wATfeyAMNqIee7cH3q1bh4QJFAaeNv0) 
- Deep leanrning. I recommend [this book](https://www.deeplearningbook.org/).

![image](https://user-images.githubusercontent.com/49912069/123512273-c1445080-d6b0-11eb-9ec9-5187fe7bc20c.png).

The model details can be found in [training folder](https://github.com/minhairtran/food_ordering_system/tree/main/train). NLL loss function was used for adjusting weights of the models. The model was optimized with ADAM optimizer. Precision was used to measure the quality of the models. In the project, I set fix learning rate as 0.0001 because it took time for understanding and implementing One Cycle Learning Rate Scheduler introduced in the paper [Super-Convergence: Very Fast Training of Neural Networks Using Large Learning Rates](https://arxiv.org/abs/1708.07120). The model was trained in 50 epochs or when the precision reached more than 99.9%.

The dataset was built in 2 steps:
- Getting the original voice from [FPT.AI](https://fpt.ai/). FPT.AI provides 9 different voices, varying from region in Vietnam and gender. Also, they allow users to change the speed of the generated audio with 13 different levels so that it can be more likely to adapt with voices in real situation. Codes to discover more about generating dataset can be found in [gen data folder](https://github.com/minhairtran/food_ordering_system/tree/main/gen_data). 
- In order to have more data, I used SpecAugment for changing a part of the original data by masking a part of signal energy in frequency and time dimension randomly. This only allows to make data "noisier", which help adapt with the real situation. For changing the original voice in terms of tones, which can be achieved by adjusting the energy in frequency domain, [data augmentation from Pytorch](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#data-augmentation) can be a sound solution.

There're about 2700 audio files for each keyword, which is acceptable because the data is various and is larger than [Google Command Dataset](https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html) which has about 2300 audio files for each keyword



# Contructing system detecting keyword directly

# Constructing application of KWS in food ording in restaurants

# Instruction guides
The models are small foot-print so it can be trained in a your PC using CPU, with 8gb RAM. But it's trained faster with GPU of 4gb.   
# Summary and future development
Unfortunately, this application can't be used in restaurants because of following problems:
- Precision of the system detecting directly drops when added more dishes
- The proposed method for detecting human voice from noise environment is not stable
- The system can't handle multiple directly detecting requests at the same time




