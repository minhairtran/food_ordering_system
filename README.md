# Contents
- [Introduction](#introduction)
- [Constructing KWS models](#constructing-kws-models)
- [Contructing system detecting keyword directly](#contructing-system-detecting-keyword-directly)
- [Constructing application of KWS in food ording in restaurants](#constructing-application-of-kws-in-food-ording-in-restaurants)
- [Instruction guides](instruction-guides)
- [Summary and future development](summary-and-future-development)

# Introduction
Application of keyword spotting system in food ordering in restaurants is an open source, aiming at  reducing the interaction between serveurs and customers. Restaurants also have number of serveurs in restaurants reduce when using the application, which reduces the operation budgets. On creating this application, I hope to help restaurants in Vietnam overcome the difficult time in COVID period.

In this application, I solved the following problems:
- Building Vietnamese dataset based on text to speech of [FPT.AI](https://fpt.ai/) and [SpecAugment method](https://arxiv.org/abs/1904.08779). The dataset in other languages can also be generated with the same approach by using text to speech from [Google Cloud](https://cloud.google.com/text-to-speech) or other providers.
- Building models detecting keywords based on [Attention-based method](https://arxiv.org/pdf/1803.10916.pdf). In general, the precision in test dataset reaches more than 99%
- Building a cheap method to differentiate human voice from restaurant environment noise and therefore, building a system for detecting keyword from human voice directly with precision of 81.25% and 91.67% for the 2 models built
- Bulding a simple user interface for serveurs in restaurants to manage the application

# Constructing KWS models

# Contructing system detecting keyword directly

# Constructing application of KWS in food ording in restaurants

# Instruction guides

# Summary and future development
Unfortunately, this application can't be used in restaurants because of following problems:
- Precision of the system detecting directly drops when added more dishes
- The proposed method for detecting human voice from noise environment is not stable
- The system can't handle multiple directly detecting requests at the same time




