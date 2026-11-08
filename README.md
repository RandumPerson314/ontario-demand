Note: the website that this repository holds the code for can be accessed here: [https://randumperson314.github.io/ontario-demand/](https://randumperson314.github.io/ontario-demand/)

# Ontario Power Demand Prediction Models
I can't seem to remember how this idea came to me, but I decided to make a website with a host of models that predict power demand in Ontario in the coming hours and days. More accurately, I decided to enlist Claude to make a website with a host of models that predict power demand in Ontario in the coming hours and days. This was my first experience using an LLM to code for me and I am fairly happy with the result. What follows is a comparison of the different models and additional points about the project.

## Polynomial
While I knew the simplicity of this approach would likely cause the predictions to be less accurate (which is contradictory the primary goal of the project), there are a few benefits to the polynomial that I am quite fond of. First, I think it's neat that the entire model can be displayed on the page in a form that is somewhat recognizable. While the polynomial is longer than any I've ever worked with, it's still easier to grasp than what something like a neural network is. Additionally, I can talk about the polynomial with anyone who has gone through secondary school math. This is cool.

## Gradient Boost
This model type was suggested to me by Claude as an alternative to the polynomial. It is generally more accurate than the polynomial and it did not require harnessing my GPU to train. However, it is still less accurate than the neural network and it lacks the benefits that come with the simplicity of the polynomial. 

## Neural Network
I was very pleased that this was the clear winner among the three models, as after creating a [neural network from scratch with numpy](https://github.com/liamstaig/numpy-neural-network), I gained a fondness for neural networks. This was also an opportunity to learn how to use CUDA to enable the model to train using my GPU.

## IESO Forecast
It was not long after I began this project that I found out the IESO publishes an official forecast for Ontario Power Demand, as the forecast is on the same site as the live power demand. This model, unsurprisingly, outperforms all three models. I had the opportunity to speak with an employee from the IESO at a career event at my school and I learned that they use, if I remember correctly, multiple neural networks and then a weighting model applied to the outputs of each neural network. This is a potential next step for what I could try with this project.

## About the Website
One of the coolest parts about having models that can predict power demand in Ontario is that it allows one to see clearly the relationship between different factors and the demand. I hoped to make this information easily accessible by including graphical representations of the relationship and allowing the user to drag sliders changing the current conditions and seeing how the prediction changes.
