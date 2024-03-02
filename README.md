# That’s what the data said: <br> An NLP Analysis of Script Lines from the US TV-Show "The Office"

![project_status](https://img.shields.io/badge/ProjectStatus-done-green)
![delivery_status](https://img.shields.io/badge/DeliveryStatus-published-green)

## Project Goal
Our objective is to apply various traditional and methorn methods of NLP in order to gain interesting insights into the show and its characters by only looking at "what the data says". More specific, we analyze characters, relationships, sentiments and topics to identify speaking styles and developments. We want to provide additional insights both for fans and for people who did not watch
the show.

Find our used data [here](https://data.world/abhinavr8/the-office-scripts-dataset).

This repository also contains scripts to train models to generate scenes (such as the scene above) and to classify the speaker of a line. 

### Use our models
We uploaded the fine-tuned models to HuggingFace to make them easy accessible for everyone.
There you can find the [Speaker Classification](https://huggingface.co/mo374z/theoffice_speaker_classification) and [Scene Generation](https://huggingface.co/mo374z/theoffice_scene_generation) models and directly test them via Inference API.

### Read more in our blog articles
- [That’s what the data said (Part I): Analyzing Script Lines from the US TV-Show “The Office“](https://medium.com/@luisa.ibele/thats-what-the-data-said-part-i-analyzing-script-lines-from-the-us-tv-show-the-office-39e67bb90b18)
- [That’s what who said (Part II): “The Office” Speaker Classification (DistilBERT) and Scene Generation (GPT)](https://medium.com/@luisa.ibele/thats-what-who-said-part-ii-the-office-speaker-classification-distilbert-and-scene-c5c4299da502)

![the-office-whoa](https://user-images.githubusercontent.com/87521684/226451449-217a1c25-535c-4b3a-9377-8305765eb320.gif)

This project was done in the course of the lecture "Intelligent Text Analysis" at Ravensburg Cooperative State University (DHBW).
The paper we wrote on our results can also be found in this repository.
