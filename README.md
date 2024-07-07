![Image](wineteller_logo_v1.png)

Born at Le Wagon Paris (#885) as part of the final project and motivated by this [poster](https://shop.winefolly.com/products/how-to-choose-wine?_gl=1%2A1ikmdky%2A_ga%2AMTQ5MjQ1MjEwNi4xNzIwMzgxMjY3%2A_ga_J39YF5X2EX%2AMTcyMDM4MTI2Ny4xLjEuMTcyMDM4MTI3Mi41NS4wLjA.) from winefolly, wineteller is a wine recommendation app that uses data science to pair the characteristics of a wine with the tone of an occasion. Check our [app](https://wineteller.streamlit.app/)

# 🔍 Context 

Picking a wine bottle is not always an easy exercise : mostly because some of us have limited knowledge about wine and the other reason may be the way too large number of options that we are given at the wine corner of the market. Usually we have either the option to ask for a wine expert or to simply use one of the widely spread wine apps (e.g Vivino, to name one).

While these conventional methods are handy, we think that they miss out on one key variable, that is the social context in which a wine bottle is opened. In our daily life, picking a wine bottle does not happen in an abstract setting where we aim for the best wine possible, that is the one with the best rating, the best taste or best winery. Rather, we aim for a decent bottle, which characteristics are likely to correspond to the specificity of an audience, such as the intimacy, casuality and wine knowledge one shares with the encoutered people. In other words, in the real word, it's not about picking a good bottle, but the right one.

Matching the characteristics of wine (such as acidity, body, length) with a specific context is challenging : on one hand, there is no clear evidence that matching wine with an occasion provides a better experience than simply choosing a wine good in every way. On the other hand, wine characteristics and context characteristics are not fundamentally related, meaning that we need a "bridge" between them to make them comparable and matchable. This will likely lead to simplifications both on oenology and data science sides. However, we think that matching wine with context has the potential to improve the global wine experience by alleviating the necessity of technical wine knowledge and recommending the wine that is likely to be enjoyable not only by the buyer but by the whole group.

# 🔨 Framework
Our model takes an occasion (i.e a description) as an input to generate a wine recommendation as an output.

An old version of the model (wineteller v0) is available [here]([https://github.com/chyunoo/wineteller](https://github.com/chyunoo/wineteller/tree/master/wineteller)). It leveraged a survey results where participants were asked to assess for a given occasion (e.g drinking with colleagues, with friends, at home, etc) which intensity of wine characteristics were the most suitable (e.g how much body, how much sweetness, etc). Each occasion and each wine were converted to a set of wine descriptors represented as vectors, using a Word2vec model trained on [wine reviews](https://www.kaggle.com/datasets/zynicide/wine-reviews). A KNN model then performed the pairing, by putting together occasions and wines that contained similar descriptors. This approach was quickly abandonned due to two reasons : it was unable to match occasions that were out of the scope of the survey and leveraged a subset of wine characteristics (complexity, body, length, sweetness, alcohol) that yielded a low diversity of occasions and wines.

Our second model (wineteller v1), which is currently deployed [🚀,](https://wineteller.streamlit.app/) is inspired by [Roald Schuring's model](https://towardsdatascience.com/robosomm-chapter-5-food-and-wine-pairing-7a4a4bb08e9e) that pairs wine with food by leveraging core taste scores(fat, acidity, bitter, etc) calculated by grouping food items that denote the most each core taste. This approach allows to compare and match wine against food by identifying words that have a similar meaning in both lexical fields. Similarly, our model leverages a set of core occasion attributes (romantic, casual, fancy, moody) to score wines across these attributes. Each occasion is converted into a set of wine descriptors that evokes a specific lexical field : for instance romantic is compounded with wine words denoting flowers, fancy is compounded with wine words denoting refinement (elegant, classy), etc. Such occasion attributes can be represented as an average vector and used to calculate a score between 0 and 1 from the similarity with each wine of the dataset. The pairing becomes much more straightforward : the recommendation is the list of wines that match the occasion scores that are requested by the user. For the full approach see here.

The current model is not exempt of limitations. One can argue that the choice of words defining each occasion is biased as not all wines with flower notes are suitable for a romantic occasion, that _elegant, classy_ do not have the same meaning in everyday and wine language. This is especially true as the validity of the model relies heavily on how we define our initial occasion attributes. We do not claim that our model yields the best performance in matching wine with occasion (as such performance criterion is yet to be defined) but we believe it demonstrates a novel approach to wine pairing that can be refined gradually.

Both of wineteller v0 and v1 are static models, that can not be fine-tuned through training. This mostly stems from the lack of a way to evaluate the performance of our model. However, we plan to switch gradually to an iterable version of our model, checkout our roadmap below.

# 💎 Current features
* **Occasion-wine pairing 🥂** : describe your occasion (romantic, moody, casual, fancy) and get wine recommendations
* **Wine recommendation visualization 📊** : view your wine recommendation's profile
* **Sommelier justification widget 🤖** : learn more about how your wine recommendation was made 

# 📍 Roadmap
* Allow language switch, 🇫🇷 in particular (user request)
* Allow to re-shuffle wine recommendations
* Allow to select region, wine style
* Allow to explore full wine dataset (by chunks)
* Create a performance metric
