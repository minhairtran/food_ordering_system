# Contents
- [Introduction](#introduction)
- [Constructing application of KWS in food ording in restaurants and using instructions](#constructing-application-of-kws-in-food-ording-in-restaurants-and-using-instructions)
- [Constructing KWS models](#constructing-kws-models)
- [Contructing system detecting keyword directly](#contructing-system-detecting-keyword-directly)
- [Training instruction guides](#training-instruction-guides)
- [Summary and future development](#summary-and-future-development)

# Introduction
Application of keyword spotting system in food ordering in restaurants is an open source, aiming at  reducing the interaction between staffs and customers. Restaurants also have number of staffs in restaurants reduce when using the application, which reduces the operation budgets. On creating this application, I hope to help restaurants in Vietnam overcome the difficult time in COVID period.

In this application, I solved the following problems:
- Building Vietnamese dataset based on text to speech of [FPT.AI](https://fpt.ai/) and [SpecAugment method](https://arxiv.org/abs/1904.08779). The dataset in other languages can also be generated with the same approach by using text to speech from [Google Cloud](https://cloud.google.com/text-to-speech) or other providers.
- Building models detecting keywords based on [attention-based method](https://arxiv.org/pdf/1803.10916.pdf). In general, the precision in test dataset reaches more than 99%
- Building a cheap method to differentiate human voice from restaurant environment noise and therefore, building a system for detecting keyword from human voice directly with precision of 81.25% and 91.67% for the 2 models built.
- Bulding a simple user interface for staffs to manage the application.

# Constructing application of KWS in food ording in restaurants and using instructions
In this module, I describe business specifications of the application and the UI designs. Instructions for using the application can be found in business specifications.

There are 3 actors: Customer, robot and staff:
- Customer is the actor coming to restaurant and order food with robot.
- Robot is the actor chatting with customer and send order requests to staff.
- Staff is the actor managing the operation of the application and managing the order requests.

The image below is the general use case model. The details of business specifications can be found [here](https://github.com/minhairtran/food_ordering_system/blob/main/doc/Business%20specifications.md).

![Phân tích hệ thống-tổng quan](https://user-images.githubusercontent.com/49912069/123767691-dd830000-d8f1-11eb-9b30-a7cacdf4ff94.png)

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

There's 2 models in the system:
- One for detecting "co" and "khong", which are "yes" and "no" in English. Detail about the precision of the confirming model can be found in [Comet](https://www.comet.ml/hai321/confirming/view/new). Model having training dataset masked with 14 parameters in frequency domain and 10 parameter in time domain reached  
- One for detecting the dishes name. Detail about the precision of the food model can be found in [Comet](https://www.comet.ml/hai321/food/view/new)

# Contructing system detecting keyword directly
In this module, I explain how the system detecting keyword directly was built and related results.

The diagram below describes the system in general:
![image](https://user-images.githubusercontent.com/49912069/123517901-c6fc5f00-d6cd-11eb-8b05-308b78524b23.png)

The streaming audio from customers is cut into segments for easily detecting. The segments in which there's no voice found will be skipped. If the following segment has voice in it, it'll be merged with the previous one. This process keeps running until there's no voice in the next segment. The merged segments or segment will be then put into the keyword spotting for detecting which keyword trained was said.

So the problem was to built a voice detecting activity or VAD. In the project, I suggested using max local amplitude in a segment for the VAD. In theory, the background noise has the amplitude lower than the human voice and closer the noise to the microphone, the larger the amplitude. Based on this, I suggested having the max amplitude in a segment compare with a thread *a*. If the max amplitude is larger than *a*, it should contain voice. After experiments measuring in 4 restaurant, I found that the thread *a* could be 0.049 for the [Samson pro microphone](http://www.samsontech.com/samson/products/microphones/usb-microphones/c01upro/). Details about the experiments can be found at [this link](https://drive.google.com/drive/folders/1GteIwc3bIkq8h88DhhsCSCLqArAHHO0u?usp=sharing)

The test with the system detecting keyword directly was conducted with 6 people from diffent region in Vietnam. From each region, there're 2 people, different in gender. The model having training dataset masked with 14 parameters in frequency domain and 10 parameter in time domain reached:
- 91.67% in precision for confirming model
- 81.25% in precision for the food model

The detail of the test can be found [here](https://drive.google.com/drive/folders/1hL3m-5ZzbRo8DsBMiFZY9JESI0Buz3uI?usp=sharing)

# Training instruction guides
- The models are small foot-print so it can be trained in a your PC using CPU, with 8gb RAM. But it's trained faster with GPU of 4gb. 
- All libraries in [requirements.txt](https://github.com/minhairtran/food_ordering_system/blob/main/requirements) and [python](https://www.python.org/downloads/) should be installed.
- You should first go to the crawl.py for getting the desired data. Then go to prepare data (ex: prepare_data_confirming.py) for preprocesing your data. Then go train for training the model (ex: training_confirming.py) and test in real time in predict folder (ex: predict_confirming.py).

# Summary and future development
This application can't be used in restaurants because of existing problems:
- Precision of the system detecting directly drops when added more dishes. This can be handled by adding more data variant in tone with the help of [data augmentation from Pytorch](https://pytorch.org/tutorials/beginner/audio_preprocessing_tutorial.html#data-augmentation).
- The proposed method for detecting human voice from noise environment is not stable. This can be handled by building VAD with KWS. Dataset includes the noise and the person close to the microphone sound. 
- The system can't handle multiple directly detecting requests at the same time. This can be handled by letting the client solve the audio streaming. This means you can install the application in a microcontroller such as Arduino or in a smart, small device such as a tablet. If tablet is chosen, you can build a web application so that the customer can also order by torching the screen. This is even better for the user experience in a long term. 
- Staff starts the conversation for customer and robot is not a good user experience. It's better if a KWS for triggering the conversation based on the same method is built. Dataset can be achieved by looking at the [attention-based article](https://arxiv.org/pdf/1803.10916.pdf) or other articles in the feild such as [KWS by CRNN](https://arxiv.org/abs/1703.05390)
- Robot detect falsely when customer says a word/term that's not trained. The solution for this is to add a keyword "Unknown" in the dataset. This dataset should include a large amount of word that's not food in the menu or confirming word so that it is "abtract" enough to fulfil the requirement of "unknown keyword".  

If you're interested in the project and develop it, you can send me request to review your code. 
